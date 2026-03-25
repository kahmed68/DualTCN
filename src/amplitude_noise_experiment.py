"""
amplitude_noise_experiment.py
==============================
Quantify DualTCN sensitivity to amplitude-channel noise and bias.

The standard noise robustness experiment (Section 4.4) adds Gaussian
noise only to the peak-normalised waveform channels while holding the
log-peak-amplitude channels clean.  This script closes that gap by
perturbing the log_amp inputs directly and measuring the resulting
degradation in R² for all four parameters, with particular focus on
σ₂ and d₂.

Two perturbation modes
----------------------
(A) Random noise  — model finite stacking uncertainty.
    For each sample/receiver, add Gaussian noise N(0, σ_amp) to the
    log₁₀ peak amplitude.  In physical units a perturbation of σ_amp
    in log₁₀ corresponds to a multiplicative amplitude factor of
    10^σ_amp (e.g. σ_amp = 0.05 → ±12 % amplitude).
    Levels tested: σ_amp = 0.01, 0.02, 0.05, 0.10, 0.20 (log₁₀ units).

(B) Systematic bias — model calibration drift.
    Add a fixed offset β to all four log_amp channels of every sample.
    Levels tested: β = −0.20, −0.10, −0.05, 0.00, +0.05, +0.10, +0.20.

Outputs
-------
paper/amplitude_noise_results_random.csv
paper/amplitude_noise_results_bias.csv
paper/amplitude_noise_figure.png   — two-panel figure for the paper
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_p9d import (
    DATA_PATH_V4, TRAIN_SPLIT, VAL_SPLIT,
    IN_CHANNELS, N_TIME, N_DEPTH, LATENT_DIM, DROPOUT,
)
from model_p9d import LateTimePCRN

SEED        = 42
BATCH_SIZE  = 1024
MODEL_PATH  = os.path.join(DIR, "best_model_p9d.pt")
PAPER_DIR   = os.path.join(DIR, "paper")

NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]   # σ_amp in log₁₀ units
BIAS_LEVELS  = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20]  # β in log₁₀


# ── helpers ───────────────────────────────────────────────────────────────────

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def build_x_tensor(E_multi: np.ndarray,
                   log_amps: np.ndarray,
                   amp_delta: np.ndarray) -> torch.Tensor:
    """
    Construct the (N, 8, N_TIME) input tensor with perturbed log_amp.

    E_multi  : (N, 4, N_TIME)   normalised waveform channels
    log_amps : (N, 4)           original log₁₀ peak amplitudes
    amp_delta: (N, 4)           additive perturbation in log₁₀ units
    """
    N, n_recv, n_time = E_multi.shape
    perturbed = log_amps + amp_delta           # (N, 4)
    # Interleave: channel order [E_r0, amp_r0, E_r1, amp_r1, ...]
    x = np.zeros((N, 8, n_time), dtype=np.float32)
    for j in range(n_recv):
        x[:, 2 * j,     :] = E_multi[:, j, :]                  # waveform
        x[:, 2 * j + 1, :] = perturbed[:, j, np.newaxis]       # broadcast scalar
    return torch.tensor(x)


@torch.no_grad()
def run_inference(model, x_tensor: torch.Tensor,
                  device: torch.device) -> np.ndarray:
    """Return (N, 4) normalised parameter predictions."""
    preds = []
    for i in range(0, len(x_tensor), BATCH_SIZE):
        xb = x_tensor[i: i + BATCH_SIZE].to(device)
        _, p_pred, _ = model(xb)
        preds.append(p_pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PAPER_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── load model ─────────────────────────────────────────────────────────
    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print("Model loaded — deterministic eval mode.")

    # ── load test split ────────────────────────────────────────────────────
    print("Loading dataset …")
    d = np.load(os.path.join(DIR, DATA_PATH_V4))
    n = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    sl = slice(n_train + n_val, n)

    E_test   = d["E_multi"][sl]      # (N, 4, 128)
    A_test   = d["log_amps"][sl]     # (N, 4)
    P_test   = d["params"][sl]       # (N, 4) physical
    N_test   = len(E_test)
    print(f"Test samples: {N_test}")

    # Normalised ground-truth for R² computation
    from config_v4 import (
        LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
        LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
        LOG_D1_MIN, LOG_D1_MAX,
        LOG_D2_MIN, LOG_D2_MAX,
    )
    bounds = np.array([
        [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
        [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
        [LOG_D1_MIN,     LOG_D1_MAX    ],
        [LOG_D2_MIN,     LOG_D2_MAX    ],
    ])
    log_params = np.log10(P_test + 1e-12)
    p_norm_true = np.clip(
        (log_params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-8),
        0.0, 1.0
    ).astype(np.float32)              # (N, 4)

    param_names = ["σ₁", "σ₂", "d₁", "d₂"]

    # ── Experiment A: random noise ─────────────────────────────────────────
    print("\n=== Experiment A: random noise on log-amplitude channels ===")
    results_random = []
    for sigma_amp in NOISE_LEVELS:
        if sigma_amp == 0.0:
            delta = np.zeros_like(A_test)
        else:
            delta = rng.normal(0.0, sigma_amp, size=A_test.shape).astype(np.float32)
        x_t = build_x_tensor(E_test, A_test, delta)
        p_pred = run_inference(model, x_t, device)
        r2s = [r2(p_norm_true[:, k], p_pred[:, k]) for k in range(4)]
        mean_r2 = float(np.mean(r2s))
        row = {"sigma_amp": sigma_amp, "R2_s1": r2s[0], "R2_s2": r2s[1],
               "R2_d1": r2s[2], "R2_d2": r2s[3], "mean_R2": mean_r2}
        results_random.append(row)
        print(f"  σ_amp={sigma_amp:.2f}  "
              f"R²(σ₁)={r2s[0]:.4f}  R²(σ₂)={r2s[1]:.4f}  "
              f"R²(d₁)={r2s[2]:.4f}  R²(d₂)={r2s[3]:.4f}  "
              f"mean={mean_r2:.4f}")

    # ── Experiment B: systematic bias ──────────────────────────────────────
    print("\n=== Experiment B: systematic bias on log-amplitude channels ===")
    results_bias = []
    for beta in BIAS_LEVELS:
        delta = np.full_like(A_test, beta)
        x_t = build_x_tensor(E_test, A_test, delta)
        p_pred = run_inference(model, x_t, device)
        r2s = [r2(p_norm_true[:, k], p_pred[:, k]) for k in range(4)]
        mean_r2 = float(np.mean(r2s))
        row = {"beta": beta, "R2_s1": r2s[0], "R2_s2": r2s[1],
               "R2_d1": r2s[2], "R2_d2": r2s[3], "mean_R2": mean_r2}
        results_bias.append(row)
        print(f"  β={beta:+.2f}  "
              f"R²(σ₁)={r2s[0]:.4f}  R²(σ₂)={r2s[1]:.4f}  "
              f"R²(d₁)={r2s[2]:.4f}  R²(d₂)={r2s[3]:.4f}  "
              f"mean={mean_r2:.4f}")

    # ── Save CSVs ──────────────────────────────────────────────────────────
    rnd_csv = os.path.join(PAPER_DIR, "amplitude_noise_results_random.csv")
    with open(rnd_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results_random[0].keys()))
        w.writeheader(); w.writerows(results_random)
    print(f"\nSaved → {rnd_csv}")

    bias_csv = os.path.join(PAPER_DIR, "amplitude_noise_results_bias.csv")
    with open(bias_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results_bias[0].keys()))
        w.writeheader(); w.writerows(results_bias)
    print(f"Saved → {bias_csv}")

    # ── Figure ─────────────────────────────────────────────────────────────
    colors = {"σ₁": "#1f77b4", "σ₂": "#ff7f0e", "d₁": "#2ca02c", "d₂": "#d62728"}
    keys   = ["R2_s1", "R2_s2", "R2_d1", "R2_d2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A — random noise
    ax = axes[0]
    sigma_vals = [r["sigma_amp"] for r in results_random]
    # Convert σ_amp to approximate amplitude % for twin-x label
    for k, pname in zip(keys, param_names):
        ax.plot(sigma_vals, [r[k] for r in results_random],
                "o-", color=colors[pname], label=pname, lw=2, ms=6)
    ax.set_xlabel(r"Amplitude noise $\sigma_{\mathrm{amp}}$ (log$_{10}$ units)", fontsize=11)
    ax.set_ylabel(r"$R^2$", fontsize=11)
    ax.set_title("(A)  Random amplitude noise\n(models finite stacking error)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    # Annotate percent amplitude equivalent on top axis
    ax2 = ax.twiny()
    amp_pct = [100.0 * (10.0 ** s - 1.0) for s in sigma_vals]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(sigma_vals)
    ax2.set_xticklabels([f"{p:.0f}%" for p in amp_pct], fontsize=8)
    ax2.set_xlabel("Approx. amplitude uncertainty (%)", fontsize=9)

    # Panel B — systematic bias
    ax = axes[1]
    beta_vals = [r["beta"] for r in results_bias]
    for k, pname in zip(keys, param_names):
        ax.plot(beta_vals, [r[k] for r in results_bias],
                "o-", color=colors[pname], label=pname, lw=2, ms=6)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(r"Amplitude bias $\beta$ (log$_{10}$ units)", fontsize=11)
    ax.set_ylabel(r"$R^2$", fontsize=11)
    ax.set_title("(B)  Systematic amplitude bias\n(models calibration drift)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    # Annotate multiplicative factor on top axis
    ax2b = ax.twiny()
    factors = [10.0 ** b for b in beta_vals]
    ax2b.set_xlim(ax.get_xlim())
    ax2b.set_xticks(beta_vals)
    ax2b.set_xticklabels([f"×{f:.2f}" for f in factors], fontsize=8)
    ax2b.set_xlabel("Multiplicative amplitude factor", fontsize=9)

    fig.suptitle(
        "DualTCN sensitivity to amplitude-channel perturbations\n"
        "(150 000 test samples; waveform channels unperturbed)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = os.path.join(PAPER_DIR, "amplitude_noise_figure.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
