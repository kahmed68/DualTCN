"""
benchmark_warmstart.py
======================
Stronger conventional inversion baselines addressing reviewer concerns about
the fairness of the original multi-start benchmark.

Three improvements over the original benchmark_multistart.py:

  1. DualTCN warm-start (WS-LM, WS-LBFGSB)
       Start 0 = DualTCN single-pass prediction placed in normalised [0,1]
       space (the same space the optimisers work in).  Starts 1–7 = same
       random draws as the original benchmark (same seed).  This leverages
       the network's global approximation to avoid distant local minima.

  2. Tikhonov-regularised L-BFGS-B (TIK-LBFGSB)
       Misfit = normalised L2 data misfit + lambda * ||theta - 0.5||^2
       with lambda = 0.01.  The prior theta_prior = 0.5 (parameter space
       centre) weakly penalises excursions to the boundaries where the
       objective is flattest, improving conditioning without strongly
       biasing the solution.  Same 8 starts as the original benchmark.

  3. Tightened tolerances for LM (TT-LM)
       ftol=1e-8, xtol=1e-8 (vs 1e-6 in the original benchmark), max_nfev
       raised to 5000 per start.  No other changes.

All methods use the same 500 test samples and random starting models as the
original benchmark for a direct comparison.

Outputs
-------
paper/benchmark_warmstart_results.csv   — R², RMSE, time, nfev per method
paper/benchmark_warmstart_comparison.png — bar chart comparing mean R² and
                                           per-parameter R² for all methods
logs/benchmark_warmstart.out / .err     — PBS log files
"""

import os
import sys
import time
import numpy as np
import torch
import scipy.optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, set_start_method

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_p9d import (
    RECEIVER_OFFSETS,
    DATA_PATH_V4, TRAIN_SPLIT, VAL_SPLIT,
    IN_CHANNELS, N_TIME, N_DEPTH, LATENT_DIM, DROPOUT,
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
)
from forward_model_v4 import compute_multi_receiver_timeseries
from model_p9d import LateTimePCRN
from dataset_v4 import MCSEMDatasetV4
from torch.utils.data import DataLoader

# ── Configuration ──────────────────────────────────────────────────────────────
N_CONV      = 500           # same 500 samples as original benchmark
N_STARTS    = 8             # total starts: 1 warm/mid + 7 random
N_WORKERS   = min(32, cpu_count())
SEED        = 42            # same seed for reproducibility

TIK_LAMBDA  = 0.01          # Tikhonov regularisation weight
TIK_PRIOR   = 0.5           # prior = parameter space centre

MODEL_PATH  = os.path.join(DIR, "best_model_p9d.pt")
OUT_CSV     = os.path.join(DIR, "paper", "benchmark_warmstart_results.csv")
OUT_PNG     = os.path.join(DIR, "paper", "benchmark_warmstart_comparison.png")

# ── Parameter space ────────────────────────────────────────────────────────────
_BOUNDS_LOG = np.array([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
    [LOG_D1_MIN,     LOG_D1_MAX    ],
    [LOG_D2_MIN,     LOG_D2_MAX    ],
], dtype=np.float64)

_THETA0_MID = np.full(4, 0.5)
_BOUNDS_01  = [(0.0, 1.0)] * 4


def _theta_to_phys(theta):
    log_p = _BOUNDS_LOG[:, 0] + theta * (_BOUNDS_LOG[:, 1] - _BOUNDS_LOG[:, 0])
    return 10.0 ** log_p


def _phys_to_theta(sigma1, sigma2, d1, d2):
    log_p = np.log10(np.array([sigma1, sigma2, d1, d2], dtype=np.float64))
    lo, hi = _BOUNDS_LOG[:, 0], _BOUNDS_LOG[:, 1]
    return np.clip((log_p - lo) / (hi - lo), 0.0, 1.0)


def _misfit_scalar(theta, E_obs_abs):
    """Normalised L2 misfit J(θ) ∈ [0, 4]."""
    sigma1, sigma2, d1, d2 = _theta_to_phys(np.clip(theta, 0.0, 1.0))
    try:
        fwd = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0=0.0)
    except Exception:
        return 4.0
    J = 0.0
    for j, r in enumerate(RECEIVER_OFFSETS):
        E_n, la = fwd[r]
        E_pred  = E_n * (10.0 ** la)
        E_obs   = E_obs_abs[j]
        denom   = float(np.dot(E_obs, E_obs)) + 1e-60
        J      += float(np.dot(E_obs - E_pred, E_obs - E_pred)) / denom
    return J


def _misfit_tikhonov(theta, E_obs_abs):
    """Data misfit + Tikhonov regularisation toward parameter-space centre."""
    data_J = _misfit_scalar(theta, E_obs_abs)
    reg    = TIK_LAMBDA * float(np.sum((theta - TIK_PRIOR) ** 2))
    return data_J + reg


def _misfit_residuals(theta, E_obs_abs):
    """Concatenated normalised residuals for LM."""
    sigma1, sigma2, d1, d2 = _theta_to_phys(np.clip(theta, 0.0, 1.0))
    try:
        fwd = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0=0.0)
    except Exception:
        N_T = E_obs_abs.shape[1]
        return np.ones(4 * N_T) * 1e3
    residuals = []
    for j, r in enumerate(RECEIVER_OFFSETS):
        E_n, la = fwd[r]
        E_pred  = E_n * (10.0 ** la)
        E_obs   = E_obs_abs[j]
        scale   = np.sqrt(np.dot(E_obs, E_obs) + 1e-60)
        residuals.append((E_obs - E_pred) / scale)
    return np.concatenate(residuals)


# ── Per-sample inversion ────────────────────────────────────────────────────────

def _invert_one(args):
    """
    Invert one sample with four methods.

    Starts layout:
        starts[0] = DualTCN warm-start prediction (or midpoint for methods
                    that don't use the warm start)
        starts[1..N_STARTS-1] = random uniform in [0,1]^4

    Returns dict with results for all four methods.
    """
    idx, E_multi_i, log_amps_i, params_true_i, starts_random, theta_dualtcn = args

    E_obs_abs  = (E_multi_i * (10.0 ** log_amps_i[:, np.newaxis])).astype(np.float64)
    theta_true = _phys_to_theta(*params_true_i)

    out = {"idx": idx, "theta_true": theta_true}

    # Helper: run one LM start
    def _run_lm(theta0, ftol=1e-8, xtol=1e-8, max_nfev=5000):
        return scipy.optimize.least_squares(
            _misfit_residuals, theta0, args=(E_obs_abs,),
            method="lm", max_nfev=max_nfev, ftol=ftol, xtol=xtol,
        )

    # Helper: run one L-BFGS-B start (data misfit only)
    def _run_bfgs(theta0):
        return scipy.optimize.minimize(
            _misfit_scalar, theta0, args=(E_obs_abs,),
            method="L-BFGS-B", bounds=_BOUNDS_01,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )

    # Helper: run one Tikhonov L-BFGS-B start
    def _run_bfgs_tik(theta0):
        return scipy.optimize.minimize(
            _misfit_tikhonov, theta0, args=(E_obs_abs,),
            method="L-BFGS-B", bounds=_BOUNDS_01,
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
        )

    # ── Method 1: WS-LM — warm-start + tight tolerances ───────────────────────
    ws_starts = np.vstack([theta_dualtcn[np.newaxis, :], starts_random])
    best_misfit, best_theta, total_nfev = np.inf, _THETA0_MID.copy(), 0
    t0 = time.perf_counter()
    for theta0 in ws_starts:
        res = _run_lm(theta0)
        total_nfev += res.nfev
        m = _misfit_scalar(res.x, E_obs_abs)
        if m < best_misfit:
            best_misfit, best_theta = m, np.clip(res.x, 0.0, 1.0)
    out["t_ws_lm"]      = time.perf_counter() - t0
    out["theta_ws_lm"]  = best_theta
    out["nfev_ws_lm"]   = total_nfev
    out["misfit_ws_lm"] = best_misfit

    # ── Method 2: WS-LBFGSB — warm-start + tight tolerances ──────────────────
    best_misfit, best_theta, total_nfev = np.inf, _THETA0_MID.copy(), 0
    t0 = time.perf_counter()
    for theta0 in ws_starts:
        res = _run_bfgs(theta0)
        total_nfev += res.nfev
        m = float(res.fun) if np.isfinite(res.fun) else 4.0
        if m < best_misfit:
            best_misfit, best_theta = m, np.clip(res.x, 0.0, 1.0)
    out["t_ws_bfgs"]      = time.perf_counter() - t0
    out["theta_ws_bfgs"]  = best_theta
    out["nfev_ws_bfgs"]   = total_nfev
    out["misfit_ws_bfgs"] = best_misfit

    # ── Method 3: TIK-LBFGSB — Tikhonov + 8 random starts ────────────────────
    mid_starts = np.vstack([_THETA0_MID[np.newaxis, :], starts_random])
    best_misfit, best_theta, total_nfev = np.inf, _THETA0_MID.copy(), 0
    t0 = time.perf_counter()
    for theta0 in mid_starts:
        res = _run_bfgs_tik(theta0)
        total_nfev += res.nfev
        # Evaluate unregularised misfit for fair comparison
        m = _misfit_scalar(res.x, E_obs_abs)
        if m < best_misfit:
            best_misfit, best_theta = m, np.clip(res.x, 0.0, 1.0)
    out["t_tik"]      = time.perf_counter() - t0
    out["theta_tik"]  = best_theta
    out["nfev_tik"]   = total_nfev
    out["misfit_tik"] = best_misfit

    # ── Method 4: TT-LM — original midpoint+random starts, tight tolerances ───
    best_misfit, best_theta, total_nfev = np.inf, _THETA0_MID.copy(), 0
    t0 = time.perf_counter()
    for theta0 in mid_starts:
        res = _run_lm(theta0)
        total_nfev += res.nfev
        m = _misfit_scalar(res.x, E_obs_abs)
        if m < best_misfit:
            best_misfit, best_theta = m, np.clip(res.x, 0.0, 1.0)
    out["t_tt_lm"]      = time.perf_counter() - t0
    out["theta_tt_lm"]  = best_theta
    out["nfev_tt_lm"]   = total_nfev
    out["misfit_tt_lm"] = best_misfit

    return out


# ── Helpers ────────────────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(DIR, "paper"), exist_ok=True)
    os.makedirs(os.path.join(DIR, "logs"),  exist_ok=True)

    # ── Load test data ─────────────────────────────────────────────────────────
    print("Loading dataset …")
    d       = np.load(os.path.join(DIR, DATA_PATH_V4))
    n       = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    sl      = slice(n_train + n_val, n)

    E_multi  = d["E_multi"][sl]
    log_amps = d["log_amps"][sl]
    params   = d["params"][sl]
    profiles = d["sigma_profiles"][sl] if "sigma_profiles" in d \
               else np.zeros((len(E_multi), 64), dtype=np.float32)

    # Same 500 sample indices as original benchmark (same seed = same indices)
    idx_sel = np.sort(rng.choice(len(E_multi), size=N_CONV, replace=False))
    E_sel   = E_multi[idx_sel]
    A_sel   = log_amps[idx_sel]
    P_sel   = params[idx_sel]
    Pr_sel  = profiles[idx_sel]

    # Same random starts as original benchmark (generated after idx_sel with same rng)
    rand_starts = rng.uniform(0.0, 1.0, size=(N_STARTS - 1, 4))
    print(f"Test samples : {N_CONV}")
    print(f"Random starts: {N_STARTS - 1}  (shared across all samples)")

    # ── DualTCN warm-start predictions ────────────────────────────────────────
    print("\nRunning DualTCN for warm-start predictions …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = LateTimePCRN(
        in_ch=IN_CHANNELS, in_len=N_TIME,
        out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT,
    )
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = MCSEMDatasetV4(E_sel, A_sel, Pr_sel, P_sel, augment=False)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False,
                         num_workers=0, pin_memory=(device.type == "cuda"))

    ws_preds = []
    with torch.no_grad():
        for batch in loader:
            x_in = batch[0].to(device)
            _, p_norm, _ = model(x_in)
            ws_preds.append(p_norm.cpu().numpy())
    theta_dualtcn = np.concatenate(ws_preds, axis=0)   # (N_CONV, 4) in [0,1]
    print(f"  DualTCN predictions computed: shape {theta_dualtcn.shape}")

    # Compute DualTCN-only R² for reference column in results
    theta_true_all = np.stack([
        _phys_to_theta(*P_sel[i]) for i in range(N_CONV)
    ])
    r2_dualtcn = _r2(theta_true_all, theta_dualtcn)
    print(f"  DualTCN R²: σ1={r2_dualtcn[0]:.3f}  σ2={r2_dualtcn[1]:.3f}"
          f"  d1={r2_dualtcn[2]:.3f}  d2={r2_dualtcn[3]:.3f}"
          f"  mean={r2_dualtcn.mean():.3f}")

    # ── Build per-sample argument list ────────────────────────────────────────
    args = [
        (i, E_sel[i], A_sel[i], P_sel[i], rand_starts, theta_dualtcn[i])
        for i in range(N_CONV)
    ]

    # ── Parallel inversion ────────────────────────────────────────────────────
    print(f"\nRunning 4 methods × {N_CONV} samples × {N_STARTS} starts "
          f"({N_WORKERS} workers) …")
    t_wall = time.perf_counter()
    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_invert_one, args)
    t_wall = time.perf_counter() - t_wall
    print(f"Total wall time: {t_wall/60:.1f} min")

    # ── Collect results ───────────────────────────────────────────────────────
    theta_true      = np.stack([r["theta_true"]    for r in raw])
    theta_ws_lm     = np.stack([r["theta_ws_lm"]   for r in raw])
    theta_ws_bfgs   = np.stack([r["theta_ws_bfgs"] for r in raw])
    theta_tik       = np.stack([r["theta_tik"]     for r in raw])
    theta_tt_lm     = np.stack([r["theta_tt_lm"]   for r in raw])

    methods = [
        ("DualTCN (single pass)",
         theta_dualtcn,
         np.zeros(N_CONV), np.zeros(N_CONV), np.ones(N_CONV)),
        ("WS-LM (warm-start + tight tol)",
         theta_ws_lm,
         [r["t_ws_lm"] for r in raw],
         [r["nfev_ws_lm"] for r in raw],
         None),
        ("WS-LBFGSB (warm-start + tight tol)",
         theta_ws_bfgs,
         [r["t_ws_bfgs"] for r in raw],
         [r["nfev_ws_bfgs"] for r in raw],
         None),
        ("TIK-LBFGSB (Tikhonov λ=%.2f + 8 starts)" % TIK_LAMBDA,
         theta_tik,
         [r["t_tik"] for r in raw],
         [r["nfev_tik"] for r in raw],
         None),
        ("TT-LM (tight tol + 8 starts, no warm-start)",
         theta_tt_lm,
         [r["t_tt_lm"] for r in raw],
         [r["nfev_tt_lm"] for r in raw],
         None),
    ]

    pnames      = ["σ1", "σ2", "d1", "d2"]
    param_latex = [r"$\sigma_1$", r"$\sigma_2$", r"$d_1$", r"$d_2$"]

    print(f"\n{'='*72}")
    print(f"  Benchmark results  (N={N_CONV} test samples)")
    print(f"{'='*72}")

    rows = []
    for tag, theta_pred, tv, nfv, _ in methods:
        r2v   = _r2(theta_true, theta_pred)
        rmse  = np.sqrt(np.mean((theta_true - theta_pred) ** 2, axis=0))
        tv    = np.asarray(tv, dtype=float)
        nfv   = np.asarray(nfv, dtype=float)
        t_mean = float(tv.mean())
        nf_mean = float(nfv.mean())
        print(f"\n{tag}")
        for i, p in enumerate(pnames):
            print(f"  {p}: R²={r2v[i]:.3f}  RMSE={rmse[i]:.4f}")
        print(f"  Mean R²={r2v.mean():.3f}  |  "
              f"Time/sample={t_mean:.2f}s  |  Evals/sample={nf_mean:.0f}")
        rows.append((tag, r2v, rmse, t_mean, nf_mean))

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w") as f:
        f.write("Method,R2_s1,R2_s2,R2_d1,R2_d2,meanR2,"
                "RMSE_s1,RMSE_s2,RMSE_d1,RMSE_d2,time_s,nfev\n")
        for tag, r2v, rmse, tms, nfm in rows:
            f.write(
                f"{tag},"
                f"{r2v[0]:.4f},{r2v[1]:.4f},{r2v[2]:.4f},{r2v[3]:.4f},"
                f"{r2v.mean():.4f},"
                f"{rmse[0]:.4f},{rmse[1]:.4f},{rmse[2]:.4f},{rmse[3]:.4f},"
                f"{tms:.3f},{nfm:.0f}\n"
            )
    print(f"\nSaved → {OUT_CSV}")

    # ── Comparison figure ─────────────────────────────────────────────────────
    labels   = [r[0].split("(")[0].strip() for r in rows]
    mean_r2s = [r[1].mean() for r in rows]
    per_r2s  = np.array([r[1] for r in rows])   # (n_methods, 4)
    times    = [r[3] for r in rows]

    n_methods = len(rows)
    x = np.arange(4)
    width = 0.15
    colours = ["#2c7bb6", "#1a9641", "#d7191c", "#fdae61", "#984ea3"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-parameter R²
    ax = axes[0]
    for i, (tag, r2v, *_) in enumerate(rows):
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, r2v, width,
               label=labels[i], color=colours[i], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(param_latex, fontsize=12)
    ax.set_ylabel("$R^2$", fontsize=12)
    ax.set_title("Per-parameter $R^2$", fontsize=12)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylim(-0.5, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # Right: mean R² vs time/sample
    ax2 = axes[1]
    for i, (mr2, t, tag) in enumerate(zip(mean_r2s, times, labels)):
        ax2.scatter(t + 1e-4, mr2, s=120, color=colours[i],
                    zorder=5, label=tag)
        ax2.annotate(tag, (t + 1e-4, mr2),
                     textcoords="offset points", xytext=(6, 3),
                     fontsize=7, color=colours[i])
    ax2.set_xscale("log")
    ax2.set_xlabel("Time per sample (s, log scale)", fontsize=12)
    ax2.set_ylabel("Mean $\\bar{R}^2$", fontsize=12)
    ax2.set_title("Accuracy vs. Cost", fontsize=12)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"Benchmark: DualTCN vs Improved Conventional Baselines  (N={N_CONV})\n"
        f"WS = DualTCN warm-start | TIK = Tikhonov λ={TIK_LAMBDA} | TT = tight tol",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")
    print("Done.")


if __name__ == "__main__":
    set_start_method("fork", force=True)
    main()
