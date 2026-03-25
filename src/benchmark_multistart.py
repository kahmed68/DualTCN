"""
benchmark_multistart.py

Multi-start conventional inversion benchmark against DualTCN.

For each of N_CONV test samples, both NLS-LM and NLS-LBFGSB are run
from N_STARTS independent starting models:
    - Start 0   : midpoint of normalised space  (θ₀ = [0.5, 0.5, 0.5, 0.5])
    - Starts 1–7: uniformly random in [0, 1]^4  (reproducible via SEED)

The best solution (lowest misfit after convergence) is kept per method
per sample.  This is a fairer benchmark because the MCSEM misfit
surface has multiple local minima; the best of several starts is much
more likely to be near the global minimum than a single midpoint start.

Runtime note
------------
Total wall-time scales as N_CONV × N_STARTS × (time per start).
With N_CONV=500, N_STARTS=8, ~2.75 s/start for LM and ~8 s/start for
L-BFGS-B we expect roughly 90 min for LM and 270 min for L-BFGS-B.
Both run in parallel across N_WORKERS processes (one sample per worker).

Outputs
-------
paper/benchmark_multistart_results.csv
paper/scatter_benchmark_multistart.png
"""

import os
import sys
import time
import numpy as np
import scipy.optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count, set_start_method

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_v4 import (
    RECEIVER_OFFSETS,
    DATA_PATH_V4, TRAIN_SPLIT, VAL_SPLIT,
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
)
from forward_model_v4 import compute_multi_receiver_timeseries

# ── Configuration ──────────────────────────────────────────────────────────────
N_CONV    = 500           # number of test samples to invert
N_STARTS  = 8             # total starts per sample (1 midpoint + 7 random)
N_WORKERS = min(32, cpu_count())
SEED      = 42

OUT_CSV = os.path.join(DIR, "paper", "benchmark_multistart_results.csv")
OUT_PNG = os.path.join(DIR, "paper", "scatter_benchmark_multistart.png")

# ── Parameter space ────────────────────────────────────────────────────────────
_BOUNDS_LOG = np.array([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
    [LOG_D1_MIN,     LOG_D1_MAX    ],
    [LOG_D2_MIN,     LOG_D2_MAX    ],
], dtype=np.float64)

_THETA0_MID = np.full(4, 0.5)          # midpoint start
_BOUNDS_01  = [(0.0, 1.0)] * 4         # L-BFGS-B box constraints


def _theta_to_phys(theta):
    log_p = _BOUNDS_LOG[:, 0] + theta * (_BOUNDS_LOG[:, 1] - _BOUNDS_LOG[:, 0])
    return 10.0 ** log_p


def _phys_to_theta(sigma1, sigma2, d1, d2):
    log_p = np.log10(np.array([sigma1, sigma2, d1, d2], dtype=np.float64))
    lo = _BOUNDS_LOG[:, 0]
    hi = _BOUNDS_LOG[:, 1]
    return np.clip((log_p - lo) / (hi - lo), 0.0, 1.0)


def _misfit_scalar(theta, E_obs_abs):
    """Scalar normalised L2 misfit J(θ) ∈ [0, 4]."""
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
        diff    = E_obs - E_pred
        J      += float(np.dot(diff, diff)) / denom
    return J


def _misfit_residuals(theta, E_obs_abs):
    """Concatenated residuals for LM (4 × N_TIME,)."""
    sigma1, sigma2, d1, d2 = _theta_to_phys(np.clip(theta, 0.0, 1.0))
    try:
        fwd = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0=0.0)
    except Exception:
        N_TIME = E_obs_abs.shape[1]
        return np.ones(4 * N_TIME) * 1e3
    residuals = []
    for j, r in enumerate(RECEIVER_OFFSETS):
        E_n, la = fwd[r]
        E_pred  = E_n * (10.0 ** la)
        E_obs   = E_obs_abs[j]
        scale   = np.sqrt(np.dot(E_obs, E_obs) + 1e-60)
        residuals.append((E_obs - E_pred) / scale)
    return np.concatenate(residuals)


# ── Per-sample multi-start inversion ──────────────────────────────────────────

def _invert_one(args):
    """
    Invert one sample with N_STARTS starts for each method.

    Starting models:
        start 0 : midpoint θ = [0.5, 0.5, 0.5, 0.5]
        starts 1–(N_STARTS-1): random uniform in [0, 1]^4

    The best solution (lowest scalar misfit) is selected per method.

    Returns dict with theta_true, best thetas, timings, nfev.
    """
    idx, E_multi_i, log_amps_i, params_true_i, starts = args

    E_obs_abs  = (E_multi_i * (10.0 ** log_amps_i[:, np.newaxis])).astype(np.float64)
    theta_true = _phys_to_theta(*params_true_i)

    out = {"idx": idx, "theta_true": theta_true}

    # ── NLS-LM ────────────────────────────────────────────────────────────────
    best_lm_misfit = np.inf
    best_lm_theta  = _THETA0_MID.copy()
    total_nfev_lm  = 0
    t0 = time.perf_counter()

    for theta0 in starts:
        res = scipy.optimize.least_squares(
            _misfit_residuals, theta0,
            args=(E_obs_abs,),
            method="lm",
            max_nfev=2000,
            ftol=1e-6, xtol=1e-6,
        )
        total_nfev_lm += res.nfev
        misfit = _misfit_scalar(res.x, E_obs_abs)
        if misfit < best_lm_misfit:
            best_lm_misfit = misfit
            best_lm_theta  = np.clip(res.x, 0.0, 1.0)

    out["t_lm"]       = time.perf_counter() - t0
    out["theta_lm"]   = best_lm_theta
    out["nfev_lm"]    = total_nfev_lm
    out["misfit_lm"]  = best_lm_misfit

    # ── NLS-LBFGSB ────────────────────────────────────────────────────────────
    best_lb_misfit = np.inf
    best_lb_theta  = _THETA0_MID.copy()
    total_nfev_lb  = 0
    t0 = time.perf_counter()

    for theta0 in starts:
        res = scipy.optimize.minimize(
            _misfit_scalar, theta0,
            args=(E_obs_abs,),
            method="L-BFGS-B",
            bounds=_BOUNDS_01,
            options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7},
        )
        total_nfev_lb += res.nfev
        misfit = res.fun if np.isfinite(res.fun) else 4.0
        if misfit < best_lb_misfit:
            best_lb_misfit = misfit
            best_lb_theta  = np.clip(res.x, 0.0, 1.0)

    out["t_lbfgsb"]      = time.perf_counter() - t0
    out["theta_lbfgsb"]  = best_lb_theta
    out["nfev_lbfgsb"]   = total_nfev_lb
    out["misfit_lbfgsb"] = best_lb_misfit

    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    # ── Load test data ─────────────────────────────────────────────────────
    print("Loading dataset …")
    d       = np.load(os.path.join(DIR, DATA_PATH_V4))
    n       = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    sl      = slice(n_train + n_val, n)

    E_multi  = d["E_multi"][sl]
    log_amps = d["log_amps"][sl]
    params   = d["params"][sl]

    idx_sel = np.sort(rng.choice(len(E_multi), size=N_CONV, replace=False))
    E_sel   = E_multi[idx_sel]
    A_sel   = log_amps[idx_sel]
    P_sel   = params[idx_sel]

    # ── Build per-sample starting sets ─────────────────────────────────────
    # Start 0 is always the midpoint.  Starts 1..(N_STARTS-1) are random.
    # Each sample uses the SAME set of starts so results are reproducible.
    rand_starts = rng.uniform(0.0, 1.0, size=(N_STARTS - 1, 4))
    starts_all  = np.vstack([_THETA0_MID[np.newaxis, :], rand_starts])  # (N_STARTS, 4)

    print(f"Multi-start benchmark: {N_CONV} samples × {N_STARTS} starts "
          f"({N_WORKERS} workers)")
    print(f"  Start 0: midpoint [0.5, 0.5, 0.5, 0.5]")
    for k in range(1, N_STARTS):
        print(f"  Start {k}: {np.round(starts_all[k], 3).tolist()}")

    args = [(i, E_sel[i], A_sel[i], P_sel[i], starts_all) for i in range(N_CONV)]

    t_wall = time.perf_counter()
    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_invert_one, args)
    t_wall = time.perf_counter() - t_wall
    print(f"\nTotal wall time: {t_wall/60:.1f} min")

    # ── Collect ─────────────────────────────────────────────────────────────
    theta_true   = np.stack([r["theta_true"]   for r in raw])
    theta_lm     = np.stack([r["theta_lm"]     for r in raw])
    theta_lbfgsb = np.stack([r["theta_lbfgsb"] for r in raw])
    t_lm         = np.array([r["t_lm"]         for r in raw])
    t_lbfgsb     = np.array([r["t_lbfgsb"]     for r in raw])
    nfev_lm      = np.array([r["nfev_lm"]      for r in raw])
    nfev_lbfgsb  = np.array([r["nfev_lbfgsb"]  for r in raw])

    pnames = ["σ1", "σ2", "d1", "d2"]

    print(f"\n{'='*70}")
    print(f" Multi-Start Benchmark  (N={N_CONV} samples, {N_STARTS} starts each)")
    print(f"{'='*70}")

    methods = [
        ("NLS-LM (best of %d starts)" % N_STARTS,     theta_lm,     t_lm,     nfev_lm),
        ("NLS-LBFGSB (best of %d starts)" % N_STARTS, theta_lbfgsb, t_lbfgsb, nfev_lbfgsb),
    ]

    rows = []
    for tag, theta_pred, tv, nfv in methods:
        r2v    = _r2(theta_true, theta_pred)
        rmsev  = np.sqrt(np.mean((theta_true - theta_pred) ** 2, axis=0))
        t_mean = tv.mean()
        t_std  = tv.std()
        nf_mean = nfv.mean()

        print(f"\n{tag}:")
        for i, p in enumerate(pnames):
            print(f"  {p}: R²={r2v[i]:.3f}  RMSE={rmsev[i]:.3f}")
        print(f"  Mean R²:    {r2v.mean():.3f}")
        print(f"  Time/sample (total {N_STARTS} starts): "
              f"{t_mean:.2f} ± {t_std:.2f} s  |  Total func evals: {nf_mean:.0f}")
        rows.append((tag, r2v, rmsev, t_mean, nf_mean))

    # ── CSV ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("Method,N_starts,R2_s1,R2_s2,R2_d1,R2_d2,meanR2,"
                "RMSE_s1,RMSE_s2,RMSE_d1,RMSE_d2,time_s,nfev\n")
        for tag, r2v, rmsev, tms, nfm in rows:
            f.write(f"{tag},{N_STARTS},"
                    f"{r2v[0]:.4f},{r2v[1]:.4f},{r2v[2]:.4f},{r2v[3]:.4f},"
                    f"{r2v.mean():.4f},"
                    f"{rmsev[0]:.4f},{rmsev[1]:.4f},{rmsev[2]:.4f},{rmsev[3]:.4f},"
                    f"{tms:.2f},{nfm:.0f}\n")
    print(f"\nSaved → {OUT_CSV}")

    # ── Scatter plot ─────────────────────────────────────────────────────────
    method_data = [
        ("NLS-LM\n(best of %d)" % N_STARTS,
         theta_lm,     _r2(theta_true, theta_lm),
         np.sqrt(np.mean((theta_true - theta_lm)     ** 2, axis=0))),
        ("NLS-LBFGSB\n(best of %d)" % N_STARTS,
         theta_lbfgsb, _r2(theta_true, theta_lbfgsb),
         np.sqrt(np.mean((theta_true - theta_lbfgsb) ** 2, axis=0))),
    ]
    param_labels = [r"$\sigma_1$", r"$\sigma_2$", r"$d_1$", r"$d_2$"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for row, (tag, theta_pred, r2v, rmsev) in enumerate(method_data):
        for col in range(4):
            ax = axes[row, col]
            ax.scatter(theta_true[:, col], theta_pred[:, col],
                       s=4, alpha=0.4, color="steelblue", rasterized=True)
            lo = min(theta_true[:, col].min(), theta_pred[:, col].min())
            hi = max(theta_true[:, col].max(), theta_pred[:, col].max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.0)
            ax.set_xlabel(f"True {param_labels[col]} [norm]")
            ax.set_ylabel(f"Pred {param_labels[col]} [norm]")
            ax.set_title(f"{tag.replace(chr(10),' ')} — {param_labels[col]}\n"
                         f"R²={r2v[col]:.3f}  RMSE={rmsev[col]:.3f}")
            ax.grid(alpha=0.3)
    plt.suptitle(
        f"Multi-Start Conventional Inversion Benchmark  "
        f"(N={N_CONV} samples, {N_STARTS} starts each)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")
    print("Done.")


if __name__ == "__main__":
    set_start_method("fork", force=True)
    main()
