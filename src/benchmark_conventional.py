"""
benchmark_conventional.py

Benchmarks two standard parametric inversion methods against P9 on
N_CONV randomly selected test-set samples.

Methods
-------
NLS-LM     : scipy.optimize.least_squares, method='lm'
             (Levenberg–Marquardt, unconstrained nonlinear least squares)
NLS-LBFGSB : scipy.optimize.minimize, method='L-BFGS-B'
             (gradient-based bounded optimisation)

Both methods minimise the normalised L2 misfit between the observed
(E_multi, log_amps) and the empymod forward prediction.

Parameter space for optimisation
---------------------------------
θ ∈ [0, 1]^4 (normalised):  θ_i linearly maps to log(param_i) ∈ [lo_i, hi_i]
Starting model : midpoint of parameter space (θ₀ = [0.5, 0.5, 0.5, 0.5])
Source velocity : fixed at v0 = 0 m/s (not available in test metadata)

Misfit (normalised L2 in absolute E-field domain)
---------------------------------------------------
E_obs_abs[r] = E_multi[r] × 10^{log_amps[r]}
E_pred_abs[r] = computed by empymod for candidate θ
J(θ) = Σ_r  ||E_obs_abs[r] − E_pred_abs[r]||² / ||E_obs_abs[r]||²

Parallelism
-----------
Each sample is inverted independently.  N_WORKERS parallel processes
are used (multiprocessing.Pool with 'fork' start method on Linux).

Outputs
-------
paper/benchmark_results.csv  — per-method R², RMSE, timing statistics
paper/scatter_benchmark.png  — scatter plots for both methods
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
N_CONV    = 500
N_WORKERS = min(32, cpu_count())
SEED      = 42

OUT_CSV = os.path.join(DIR, "paper", "benchmark_results.csv")
OUT_PNG = os.path.join(DIR, "paper", "scatter_benchmark.png")

# ── Parameter space ────────────────────────────────────────────────────────────
_BOUNDS_LOG = np.array([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
    [LOG_D1_MIN,     LOG_D1_MAX    ],
    [LOG_D2_MIN,     LOG_D2_MAX    ],
], dtype=np.float64)

_THETA0 = np.full(4, 0.5)                   # midpoint of normalised space
_BOUNDS_01 = [(0.0, 1.0)] * 4               # scipy L-BFGS-B bounds


def _theta_to_phys(theta):
    """Normalised θ → (σ1, σ2, d1, d2) physical."""
    log_p = _BOUNDS_LOG[:, 0] + theta * (_BOUNDS_LOG[:, 1] - _BOUNDS_LOG[:, 0])
    return 10.0 ** log_p


def _phys_to_theta(sigma1, sigma2, d1, d2):
    """Physical → normalised θ ∈ [0,1]^4."""
    log_p = np.log10(np.array([sigma1, sigma2, d1, d2], dtype=np.float64))
    lo = _BOUNDS_LOG[:, 0]
    hi = _BOUNDS_LOG[:, 1]
    return np.clip((log_p - lo) / (hi - lo), 0.0, 1.0)


def _normalise_params_batch(params_phys: np.ndarray) -> np.ndarray:
    """(N, 4) physical → normalised [0,1]."""
    log_p = np.log10(np.maximum(params_phys, 1e-12))
    lo = _BOUNDS_LOG[:, 0][np.newaxis, :]
    hi = _BOUNDS_LOG[:, 1][np.newaxis, :]
    return np.clip((log_p - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# ── Misfit function ────────────────────────────────────────────────────────────

def _misfit_scalar(theta, E_obs_abs):
    """
    Scalar normalised L2 misfit  J(θ) ∈ [0, 4].

    Parameters
    ----------
    theta     : (4,) float64  normalised parameter vector
    E_obs_abs : (4, N_TIME) float64  absolute E-field observations
    """
    sigma1, sigma2, d1, d2 = _theta_to_phys(np.clip(theta, 0.0, 1.0))
    try:
        fwd = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0=0.0)
    except Exception:
        return 4.0

    J = 0.0
    for j, r in enumerate(RECEIVER_OFFSETS):
        E_n, la = fwd[r]
        E_pred = E_n * (10.0 ** la)
        E_obs  = E_obs_abs[j]
        denom  = float(np.dot(E_obs, E_obs)) + 1e-60
        diff   = E_obs - E_pred
        J     += float(np.dot(diff, diff)) / denom
    return J


def _misfit_residuals(theta, E_obs_abs):
    """
    Concatenated residuals for least_squares (Levenberg–Marquardt).

    Returns a flat (4 × N_TIME,) residual vector.
    """
    sigma1, sigma2, d1, d2 = _theta_to_phys(np.clip(theta, 0.0, 1.0))
    try:
        fwd = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0=0.0)
    except Exception:
        N_TIME = E_obs_abs.shape[1]
        return np.ones(4 * N_TIME) * 1e3

    residuals = []
    for j, r in enumerate(RECEIVER_OFFSETS):
        E_n, la = fwd[r]
        E_pred = E_n * (10.0 ** la)
        E_obs  = E_obs_abs[j]
        scale  = np.sqrt(np.dot(E_obs, E_obs) + 1e-60)
        residuals.append((E_obs - E_pred) / scale)
    return np.concatenate(residuals)


# ── Single-sample inversion ────────────────────────────────────────────────────

def _invert_one(args):
    """
    Invert one test sample with NLS-LM and NLS-LBFGSB.

    Returns dict with theta_true, recovered thetas, timings, nfev.
    """
    idx, E_multi_i, log_amps_i, params_true_i = args

    # Absolute E-field observations
    E_obs_abs = (E_multi_i * (10.0 ** log_amps_i[:, np.newaxis])).astype(np.float64)
    theta_true = _phys_to_theta(*params_true_i)

    out = {"idx": idx, "theta_true": theta_true}

    # ── NLS-LM (Levenberg–Marquardt) ──────────────────────────────────────
    t0  = time.perf_counter()
    res = scipy.optimize.least_squares(
        _misfit_residuals, _THETA0,
        args=(E_obs_abs,),
        method="lm",
        max_nfev=2000,
        ftol=1e-6, xtol=1e-6,
    )
    out["t_lm"]       = time.perf_counter() - t0
    out["theta_lm"]   = np.clip(res.x, 0.0, 1.0)
    out["nfev_lm"]    = res.nfev
    out["success_lm"] = res.success

    # ── NLS-LBFGSB (gradient-based, bounded) ──────────────────────────────
    t0  = time.perf_counter()
    res = scipy.optimize.minimize(
        _misfit_scalar, _THETA0,
        args=(E_obs_abs,),
        method="L-BFGS-B",
        bounds=_BOUNDS_01,
        options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-7},
    )
    out["t_lbfgsb"]      = time.perf_counter() - t0
    out["theta_lbfgsb"]  = np.clip(res.x, 0.0, 1.0)
    out["nfev_lbfgsb"]   = res.nfev
    out["success_lbfgsb"] = res.success

    return out


# ── Aggregation helpers ────────────────────────────────────────────────────────

def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)

    # ── Load test data ─────────────────────────────────────────────────────
    print("Loading dataset …")
    d       = np.load(os.path.join(DIR, DATA_PATH_V4))
    n       = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    sl      = slice(n_train + n_val, n)

    E_multi  = d["E_multi"][sl]   # (N_test, 4, N_TIME)
    log_amps = d["log_amps"][sl]  # (N_test, 4)
    params   = d["params"][sl]    # (N_test, 4) physical

    idx_sel  = np.sort(rng.choice(len(E_multi), size=N_CONV, replace=False))
    E_sel    = E_multi[idx_sel]
    A_sel    = log_amps[idx_sel]
    P_sel    = params[idx_sel]

    print(f"Inverting {N_CONV} samples with {N_WORKERS} workers …")
    args = [(i, E_sel[i], A_sel[i], P_sel[i]) for i in range(N_CONV)]

    t_wall = time.perf_counter()
    with Pool(processes=N_WORKERS) as pool:
        raw = pool.map(_invert_one, args)
    t_wall = time.perf_counter() - t_wall
    print(f"Total wall time: {t_wall/60:.1f} min")

    # ── Collect arrays ─────────────────────────────────────────────────────
    theta_true   = np.stack([r["theta_true"]   for r in raw])
    theta_lm     = np.stack([r["theta_lm"]     for r in raw])
    theta_lbfgsb = np.stack([r["theta_lbfgsb"] for r in raw])
    t_lm         = np.array([r["t_lm"]         for r in raw])
    t_lbfgsb     = np.array([r["t_lbfgsb"]     for r in raw])
    nfev_lm      = np.array([r["nfev_lm"]      for r in raw])
    nfev_lbfgsb  = np.array([r["nfev_lbfgsb"]  for r in raw])

    methods = [
        ("NLS-LM (Levenberg-Marquardt)", theta_lm,     t_lm,     nfev_lm),
        ("NLS-LBFGSB (L-BFGS-B)",       theta_lbfgsb, t_lbfgsb, nfev_lbfgsb),
    ]

    pnames = ["σ1", "σ2", "d1", "d2"]
    print(f"\n{'='*65}")
    print(f" Benchmark Results  (N={N_CONV}, starting model = midpoint)")
    print(f"{'='*65}")

    rows = []
    for tag, theta_pred, tv, nfv in methods:
        r2v   = _r2(theta_true, theta_pred)
        rmsev = np.sqrt(np.mean((theta_true - theta_pred) ** 2, axis=0))
        t_mean = tv.mean()
        t_std  = tv.std()
        nf_mean = nfv.mean()

        print(f"\n{tag}:")
        for i, p in enumerate(pnames):
            print(f"  {p}: R²={r2v[i]:.3f}  RMSE={rmsev[i]:.3f}")
        print(f"  Mean R²: {r2v.mean():.3f}")
        print(f"  Time/sample: {t_mean:.2f} ± {t_std:.2f} s  |  "
              f"Func evals: {nf_mean:.0f}")
        rows.append((tag, r2v, rmsev, t_mean, nf_mean))

    # ── CSV ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("Method,R2_s1,R2_s2,R2_d1,R2_d2,meanR2,"
                "RMSE_s1,RMSE_s2,RMSE_d1,RMSE_d2,time_s,nfev\n")
        for tag, r2v, rmsev, tms, nfm in rows:
            f.write(f"{tag},"
                    f"{r2v[0]:.4f},{r2v[1]:.4f},{r2v[2]:.4f},{r2v[3]:.4f},"
                    f"{r2v.mean():.4f},"
                    f"{rmsev[0]:.4f},{rmsev[1]:.4f},{rmsev[2]:.4f},{rmsev[3]:.4f},"
                    f"{tms:.2f},{nfm:.0f}\n")
    print(f"\nSaved {OUT_CSV}")

    # ── Scatter plot ───────────────────────────────────────────────────────
    method_data = [
        ("NLS-LM",     theta_lm,     _r2(theta_true, theta_lm),
         np.sqrt(np.mean((theta_true - theta_lm) ** 2, axis=0))),
        ("NLS-LBFGSB", theta_lbfgsb, _r2(theta_true, theta_lbfgsb),
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
            ax.set_title(f"{tag} — {param_labels[col]}\n"
                         f"R²={r2v[col]:.3f}  RMSE={rmsev[col]:.3f}")
            ax.grid(alpha=0.3)
    plt.suptitle(f"Conventional Inversion Benchmark (N={N_CONV} test samples)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print("Done.")


if __name__ == "__main__":
    # Use 'fork' on Linux for safe multiprocessing with empymod
    set_start_method("fork", force=True)
    main()
