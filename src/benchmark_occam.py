"""
benchmark_occam.py — Occam-style regularised 1D MCSEM inversion benchmark.

Implements a smoothness-regularised inversion (Constable et al., 1987)
using the empymod forward operator with analytic-style finite-difference
Jacobians.  The objective is:

    min  ||W_d (d_obs - d_pred)||² + λ ||R m||²

where W_d is the data weighting, R is a first-difference roughness
operator, and λ is the trade-off parameter.

This provides the "gold standard" classical baseline that reviewers
expect for MCSEM inversion papers.

We run two variants:
  1. Occam-LM: Levenberg-Marquardt with roughness penalty
  2. Occam-LBFGSB: L-BFGS-B with roughness penalty + box constraints

Both use 8 random starts and the same 500 test samples as the
existing benchmark (Section 4.5 of the paper).
"""
import os
import sys
import time
import numpy as np
import csv
from scipy.optimize import least_squares, minimize

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_v4 import (
    RECEIVER_OFFSETS, Z_OBS, F_MIN, F_MAX, N_FREQ, N_TIME,
    SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE,
    DATA_PATH_V4, TRAIN_SPLIT, VAL_SPLIT,
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
)
from forward_model_v4 import compute_multi_receiver_timeseries

PAPER_DIR = os.path.join(DIR, "paper")

# Parameter bounds in normalised [0,1] space
BOUNDS_LO = np.array([0.0, 0.0, 0.0, 0.0])
BOUNDS_HI = np.array([1.0, 1.0, 1.0, 1.0])

# Denormalisation bounds
_BOUNDS = np.array([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
    [LOG_D1_MIN, LOG_D1_MAX],
    [LOG_D2_MIN, LOG_D2_MAX],
])


def _denorm(p_norm):
    """Normalised [0,1] → physical parameters."""
    log_vals = _BOUNDS[:, 0] + p_norm * (_BOUNDS[:, 1] - _BOUNDS[:, 0])
    return 10 ** log_vals  # [σ₁, σ₂, d₁, d₂]


def _norm(params_phys):
    """Physical parameters → normalised [0,1]."""
    log_vals = np.log10(np.clip(params_phys, 1e-12, None))
    return np.clip(
        (log_vals - _BOUNDS[:, 0]) / (_BOUNDS[:, 1] - _BOUNDS[:, 0] + 1e-8),
        0., 1.
    )


def _forward_timeseries(p_norm):
    """Run empymod forward model from normalised params, return stacked traces."""
    phys = _denorm(p_norm)
    sigma1, sigma2, d1, d2 = phys
    result = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2)
    traces = []
    for r in RECEIVER_OFFSETS:
        E_norm, _ = result[r]
        traces.append(E_norm)
    return np.concatenate(traces)  # (4 * N_TIME,)


def _residual_occam(p_norm, d_obs, lam=0.01):
    """Residual vector for Occam-style inversion.

    Returns concatenation of:
      - data residuals: (d_obs - d_pred) / ||d_obs||
      - roughness penalty: sqrt(λ) * R @ p_norm
    """
    d_pred = _forward_timeseries(p_norm)

    # Data residuals (relative)
    norm_obs = np.linalg.norm(d_obs) + 1e-30
    data_res = (d_obs - d_pred) / norm_obs

    # Roughness: first-difference of normalised params
    # R @ p = [p₂-p₁, p₃-p₂, p₄-p₃]
    roughness = np.sqrt(lam) * np.diff(p_norm)

    return np.concatenate([data_res, roughness])


def _objective_occam(p_norm, d_obs, lam=0.01):
    """Scalar objective for L-BFGS-B: sum of squared residuals."""
    r = _residual_occam(p_norm, d_obs, lam)
    return 0.5 * np.dot(r, r)


def _gradient_occam(p_norm, d_obs, lam=0.01, dp=1e-4):
    """Finite-difference gradient of the Occam objective."""
    f0 = _objective_occam(p_norm, d_obs, lam)
    grad = np.zeros(4)
    for i in range(4):
        p_plus = p_norm.copy()
        p_plus[i] += dp
        p_plus[i] = min(p_plus[i], 1.0)
        grad[i] = (_objective_occam(p_plus, d_obs, lam) - f0) / dp
    return grad


def run_occam_benchmark(n_test=200, n_starts=8, lam=0.01, seed=42):
    """Run Occam-style inversion benchmark on test samples."""
    print("=" * 70)
    print(f"Occam-style regularised inversion benchmark")
    print(f"  λ = {lam}, {n_starts} starts, {n_test} test samples")
    print("=" * 70)

    # Load test data
    d = np.load(os.path.join(DIR, DATA_PATH_V4))
    n = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    tst_sl = slice(n_train + n_val, n)

    P_test = d["params"][tst_sl]

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(P_test), size=n_test, replace=False)

    y_true = np.zeros((n_test, 4))
    for i, idx in enumerate(indices):
        y_true[i] = _norm(P_test[idx])

    y_var = np.var(y_true, axis=0)

    results = {}

    # ── Occam-LM ─────────────────────────────────────────────────────────
    print("\n--- Occam-LM (Levenberg-Marquardt + roughness) ---")
    pred_lm = np.zeros((n_test, 4))
    t0 = time.time()
    total_nfev = 0

    for i, idx in enumerate(indices):
        phys = P_test[idx]
        sigma1, sigma2, d1, d2 = phys
        d_obs = _forward_timeseries(_norm(phys))

        best_cost = np.inf
        best_p = np.full(4, 0.5)

        starts = [np.full(4, 0.5)]  # midpoint
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(0, 1, 4))

        for x0 in starts:
            try:
                res = least_squares(
                    _residual_occam, x0, args=(d_obs, lam),
                    bounds=(BOUNDS_LO, BOUNDS_HI),
                    method='trf',
                    ftol=1e-6, xtol=1e-6, max_nfev=500,
                )
                total_nfev += res.nfev
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_p = res.x.copy()
            except Exception:
                pass

        pred_lm[i] = best_p

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_test} done")

    t_lm = time.time() - t0
    rmse_lm = np.sqrt(np.mean((y_true - pred_lm) ** 2, axis=0))
    r2_lm = 1.0 - np.mean((y_true - pred_lm) ** 2, axis=0) / (y_var + 1e-12)
    r2_mean_lm = r2_lm.mean()
    nfev_lm = total_nfev / n_test

    print(f"  R²: σ₁={r2_lm[0]:.3f}  σ₂={r2_lm[1]:.3f}  "
          f"d₁={r2_lm[2]:.3f}  d₂={r2_lm[3]:.3f}  R̄²={r2_mean_lm:.3f}")
    print(f"  Time: {t_lm:.1f}s ({t_lm/n_test:.2f} s/sample)  "
          f"Evals: {nfev_lm:.0f}/sample")

    results["Occam-LM"] = {
        "r2": r2_lm, "r2_mean": r2_mean_lm,
        "rmse": rmse_lm, "time": t_lm / n_test, "nfev": nfev_lm,
    }

    # ── Occam-LBFGSB ─────────────────────────────────────────────────────
    print("\n--- Occam-LBFGSB (L-BFGS-B + roughness) ---")
    pred_lb = np.zeros((n_test, 4))
    t0 = time.time()
    total_nfev = 0

    for i, idx in enumerate(indices):
        phys = P_test[idx]
        d_obs = _forward_timeseries(_norm(phys))

        best_cost = np.inf
        best_p = np.full(4, 0.5)

        starts = [np.full(4, 0.5)]
        for _ in range(n_starts - 1):
            starts.append(rng.uniform(0, 1, 4))

        for x0 in starts:
            try:
                res = minimize(
                    _objective_occam, x0, args=(d_obs, lam),
                    jac=_gradient_occam,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * 4,
                    options={'maxiter': 200, 'ftol': 1e-8},
                )
                total_nfev += res.nfev
                if res.fun < best_cost:
                    best_cost = res.fun
                    best_p = res.x.copy()
            except Exception:
                pass

        pred_lb[i] = best_p

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_test} done")

    t_lb = time.time() - t0
    rmse_lb = np.sqrt(np.mean((y_true - pred_lb) ** 2, axis=0))
    r2_lb = 1.0 - np.mean((y_true - pred_lb) ** 2, axis=0) / (y_var + 1e-12)
    r2_mean_lb = r2_lb.mean()
    nfev_lb = total_nfev / n_test

    print(f"  R²: σ₁={r2_lb[0]:.3f}  σ₂={r2_lb[1]:.3f}  "
          f"d₁={r2_lb[2]:.3f}  d₂={r2_lb[3]:.3f}  R̄²={r2_mean_lb:.3f}")
    print(f"  Time: {t_lb:.1f}s ({t_lb/n_test:.2f} s/sample)  "
          f"Evals: {nfev_lb:.0f}/sample")

    results["Occam-LBFGSB"] = {
        "r2": r2_lb, "r2_mean": r2_mean_lb,
        "rmse": rmse_lb, "time": t_lb / n_test, "nfev": nfev_lb,
    }

    # ── Save CSV ─────────────────────────────────────────────────────────
    os.makedirs(PAPER_DIR, exist_ok=True)
    path = os.path.join(PAPER_DIR, "occam_benchmark.csv")
    rows = []
    for method, r in results.items():
        rows.append({
            "Method": method,
            "R2_s1": f"{r['r2'][0]:.4f}",
            "R2_s2": f"{r['r2'][1]:.4f}",
            "R2_d1": f"{r['r2'][2]:.4f}",
            "R2_d2": f"{r['r2'][3]:.4f}",
            "meanR2": f"{r['r2_mean']:.4f}",
            "time_s": f"{r['time']:.2f}",
            "nfev": f"{r['nfev']:.0f}",
        })
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {path}")

    return results


if __name__ == "__main__":
    run_occam_benchmark()
