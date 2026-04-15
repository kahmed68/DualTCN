"""
structured_amplitude_experiment.py
===================================
Extended amplitude robustness analysis addressing reviewer concern:

  "Please quantify performance under more structured amplitude perturbations
   (e.g., linear drift, per-receiver bias)."

The existing amplitude_noise_experiment.py covers:
  (A) i.i.d. random noise on all receivers equally
  (B) uniform systematic bias on all receivers equally

This script adds three *structured* perturbation scenarios that better
model real field acquisition artefacts:

  (C) Linear drift — amplitude calibration drifts linearly across the
      test set (simulating slow drift along a towline).  Parameterised by
      the total drift magnitude Δ in log10 units; the per-sample offset
      ramps from −Δ/2 to +Δ/2.

  (D) Per-receiver independent bias — each of the four receivers has its
      own random calibration offset drawn once per "line" (block of
      consecutive samples).  Parameterised by σ_recv (std of the
      per-receiver log10 bias).

  (E) Combined: waveform noise (SNR-matched) + per-receiver bias +
      slow linear drift applied simultaneously.

All experiments are run on BOTH the base model (best_model_p9d.pt) and the
amplitude-augmented model (best_model_p9d_ampaug.pt) for direct comparison.

Outputs
-------
data/csv/structured_amp_linear_drift.csv
data/csv/structured_amp_recv_bias.csv
data/csv/structured_amp_combined.csv
paper/structured_amplitude_figure.png   — three-panel figure
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
from config_v4 import (
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
    N_RECEIVERS,
)
from model_p9d import LateTimePCRN

SEED       = 42
BATCH_SIZE = 1024

BASE_MODEL_PATH   = os.path.join(DIR, "best_model_p9d.pt")
AMPAUG_MODEL_PATH = os.path.join(DIR, "best_model_p9d_ampaug.pt")
CSV_DIR   = os.path.join(DIR, "data", "csv")
PAPER_DIR = os.path.join(DIR, "paper")

# ── Perturbation levels ──────────────────────────────────────────────────────

# (C) Linear drift: total drift magnitude Δ (log10 units)
#     A drift of 0.10 means amplitude changes by ×0.89 → ×1.12 over the line
DRIFT_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]

# (D) Per-receiver bias: σ_recv (log10 units)
RECV_BIAS_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

# (E) Combined scenarios: (waveform_snr_db, recv_bias_sigma, drift_delta)
COMBINED_SCENARIOS = [
    ("Clean",              np.inf, 0.00, 0.00),
    ("Mild field",         30.0,   0.02, 0.02),
    ("Moderate field",     20.0,   0.05, 0.05),
    ("Challenging field",  10.0,   0.10, 0.10),
    ("Worst-case field",   5.0,    0.20, 0.20),
]

PARAM_NAMES = ["sigma1", "sigma2", "d1", "d2"]
PARAM_LABELS = [r"$\sigma_1$", r"$\sigma_2$", r"$d_1$", r"$d_2$"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def load_model(path, device):
    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT)
    sd = torch.load(path, map_location=device, weights_only=True)
    # Handle torch.compile prefix
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    return model


def build_x_tensor(E_multi, log_amps_perturbed):
    """Build (N, 8, N_TIME) from waveforms and perturbed log-amplitudes."""
    N, n_recv, n_time = E_multi.shape
    x = np.zeros((N, 2 * n_recv, n_time), dtype=np.float32)
    for j in range(n_recv):
        x[:, 2 * j,     :] = E_multi[:, j, :]
        x[:, 2 * j + 1, :] = log_amps_perturbed[:, j, np.newaxis]
    return torch.tensor(x)


def build_x_tensor_with_waveform_noise(E_multi, log_amps_perturbed,
                                        snr_db, rng):
    """Build input tensor with additive waveform noise at given SNR."""
    N, n_recv, n_time = E_multi.shape
    x = np.zeros((N, 2 * n_recv, n_time), dtype=np.float32)
    for j in range(n_recv):
        wave = E_multi[:, j, :].copy()
        if np.isfinite(snr_db):
            # SNR relative to RMS of each waveform
            rms = np.sqrt(np.mean(wave ** 2, axis=1, keepdims=True)) + 1e-12
            noise_std = rms * 10.0 ** (-snr_db / 20.0)
            wave = wave + rng.normal(0.0, 1.0, size=wave.shape).astype(np.float32) * noise_std
        x[:, 2 * j,     :] = wave
        x[:, 2 * j + 1, :] = log_amps_perturbed[:, j, np.newaxis]
    return torch.tensor(x)


@torch.no_grad()
def run_inference(model, x_tensor, device):
    preds = []
    for i in range(0, len(x_tensor), BATCH_SIZE):
        xb = x_tensor[i:i + BATCH_SIZE].to(device)
        _, p_pred, _ = model(xb)
        preds.append(p_pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_r2s(p_norm_true, p_pred):
    return [r2(p_norm_true[:, k], p_pred[:, k]) for k in range(4)]


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading base model ...")
    model_base = load_model(BASE_MODEL_PATH, device)
    print("Loading ampaug model ...")
    model_aug  = load_model(AMPAUG_MODEL_PATH, device)
    models = {"Base": model_base, "AmpAug": model_aug}

    # Load test data
    print("Loading dataset ...")
    d = np.load(os.path.join(DIR, DATA_PATH_V4))
    n = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    sl = slice(n_train + n_val, n)

    E_test = d["E_multi"][sl]     # (N, 4, 128)
    A_test = d["log_amps"][sl]    # (N, 4)
    P_test = d["params"][sl]      # (N, 4)
    N_test = len(E_test)
    print(f"Test samples: {N_test}")

    # Normalised ground truth
    bounds = np.array([
        [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
        [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
        [LOG_D1_MIN,     LOG_D1_MAX],
        [LOG_D2_MIN,     LOG_D2_MAX],
    ])
    log_params = np.log10(P_test + 1e-12)
    p_norm_true = np.clip(
        (log_params - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-8),
        0.0, 1.0,
    ).astype(np.float32)

    # ==================================================================
    # (C) Linear drift along towline
    # ==================================================================
    print("\n" + "=" * 60)
    print("Experiment C: Linear amplitude drift (towline simulation)")
    print("=" * 60)

    rows_drift = []
    for delta in DRIFT_LEVELS:
        # Ramp from -delta/2 to +delta/2 across the test set
        ramp = np.linspace(-delta / 2, delta / 2, N_test).astype(np.float32)
        A_perturbed = A_test + ramp[:, np.newaxis]  # same drift on all receivers

        x_t = build_x_tensor(E_test, A_perturbed)
        row = {"drift_delta": delta,
               "drift_factor_range": f"x{10**(-delta/2):.3f} to x{10**(delta/2):.3f}"}

        for mname, model in models.items():
            p_pred = run_inference(model, x_t, device)
            r2s = compute_r2s(p_norm_true, p_pred)
            for k, pn in enumerate(PARAM_NAMES):
                row[f"R2_{pn}_{mname}"] = r2s[k]
            row[f"mean_R2_{mname}"] = float(np.mean(r2s))

        rows_drift.append(row)
        print(f"  Delta={delta:.2f}  "
              f"Base mean={row['mean_R2_Base']:.4f}  "
              f"AmpAug mean={row['mean_R2_AmpAug']:.4f}")

    drift_fields = ["drift_delta", "drift_factor_range"] + \
        [f"R2_{p}_{m}" for m in models for p in PARAM_NAMES] + \
        [f"mean_R2_{m}" for m in models]
    save_csv(os.path.join(CSV_DIR, "structured_amp_linear_drift.csv"),
             rows_drift, drift_fields)

    # ==================================================================
    # (D) Per-receiver independent bias
    # ==================================================================
    print("\n" + "=" * 60)
    print("Experiment D: Per-receiver independent calibration bias")
    print("=" * 60)

    # Simulate "lines" of 500 consecutive samples sharing the same
    # per-receiver bias (models a calibration state that persists over
    # a short survey segment then changes).
    LINE_LENGTH = 500
    n_lines = (N_test + LINE_LENGTH - 1) // LINE_LENGTH

    rows_recv = []
    for sigma_recv in RECV_BIAS_LEVELS:
        if sigma_recv == 0.0:
            A_perturbed = A_test.copy()
        else:
            # Draw one bias per receiver per line
            biases = rng.normal(0.0, sigma_recv,
                                size=(n_lines, N_RECEIVERS)).astype(np.float32)
            # Expand to per-sample
            A_perturbed = A_test.copy()
            for li in range(n_lines):
                s = li * LINE_LENGTH
                e = min(s + LINE_LENGTH, N_test)
                A_perturbed[s:e] += biases[li]

        x_t = build_x_tensor(E_test, A_perturbed)
        row = {"sigma_recv": sigma_recv}

        for mname, model in models.items():
            p_pred = run_inference(model, x_t, device)
            r2s = compute_r2s(p_norm_true, p_pred)
            for k, pn in enumerate(PARAM_NAMES):
                row[f"R2_{pn}_{mname}"] = r2s[k]
            row[f"mean_R2_{mname}"] = float(np.mean(r2s))

        rows_recv.append(row)
        print(f"  sigma_recv={sigma_recv:.2f}  "
              f"Base mean={row['mean_R2_Base']:.4f}  "
              f"AmpAug mean={row['mean_R2_AmpAug']:.4f}")

    recv_fields = ["sigma_recv"] + \
        [f"R2_{p}_{m}" for m in models for p in PARAM_NAMES] + \
        [f"mean_R2_{m}" for m in models]
    save_csv(os.path.join(CSV_DIR, "structured_amp_recv_bias.csv"),
             rows_recv, recv_fields)

    # ==================================================================
    # (E) Combined realistic scenarios
    # ==================================================================
    print("\n" + "=" * 60)
    print("Experiment E: Combined realistic field scenarios")
    print("=" * 60)

    rows_combined = []
    for scenario_name, snr_db, sigma_recv, drift_delta in COMBINED_SCENARIOS:
        # Per-receiver bias
        if sigma_recv > 0:
            biases = rng.normal(0.0, sigma_recv,
                                size=(n_lines, N_RECEIVERS)).astype(np.float32)
            A_perturbed = A_test.copy()
            for li in range(n_lines):
                s = li * LINE_LENGTH
                e = min(s + LINE_LENGTH, N_test)
                A_perturbed[s:e] += biases[li]
        else:
            A_perturbed = A_test.copy()

        # Linear drift
        if drift_delta > 0:
            ramp = np.linspace(-drift_delta / 2, drift_delta / 2,
                               N_test).astype(np.float32)
            A_perturbed += ramp[:, np.newaxis]

        # Build input with waveform noise
        x_t = build_x_tensor_with_waveform_noise(
            E_test, A_perturbed, snr_db, rng)

        row = {"scenario": scenario_name,
               "snr_db": snr_db if np.isfinite(snr_db) else "inf",
               "sigma_recv": sigma_recv,
               "drift_delta": drift_delta}

        for mname, model in models.items():
            p_pred = run_inference(model, x_t, device)
            r2s = compute_r2s(p_norm_true, p_pred)
            for k, pn in enumerate(PARAM_NAMES):
                row[f"R2_{pn}_{mname}"] = r2s[k]
            row[f"mean_R2_{mname}"] = float(np.mean(r2s))

        rows_combined.append(row)
        print(f"  {scenario_name:20s}  "
              f"Base mean={row['mean_R2_Base']:.4f}  "
              f"AmpAug mean={row['mean_R2_AmpAug']:.4f}")

    combined_fields = ["scenario", "snr_db", "sigma_recv", "drift_delta"] + \
        [f"R2_{p}_{m}" for m in models for p in PARAM_NAMES] + \
        [f"mean_R2_{m}" for m in models]
    save_csv(os.path.join(CSV_DIR, "structured_amp_combined.csv"),
             rows_combined, combined_fields)

    # ==================================================================
    # Figure: three panels
    # ==================================================================
    print("\nGenerating figure ...")
    colors = {r"$\sigma_1$": "#1f77b4", r"$\sigma_2$": "#ff7f0e",
              r"$d_1$": "#2ca02c", r"$d_2$": "#d62728"}
    model_styles = {"Base": ("--", "o", 0.6), "AmpAug": ("-", "s", 1.0)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel C: Linear drift
    ax = axes[0]
    for mname, (ls, mk, alpha) in model_styles.items():
        for pn, pl in zip(PARAM_NAMES, PARAM_LABELS):
            vals = [r[f"R2_{pn}_{mname}"] for r in rows_drift]
            label = f"{pl} ({mname})" if pn == "sigma2" else (
                f"{pl}" if mname == "Base" else None)
            ax.plot(DRIFT_LEVELS, vals, marker=mk, ls=ls, color=colors[pl],
                    alpha=alpha, lw=2, ms=5,
                    label=f"{pl} {mname}" if pn in ("sigma2", "d2") else None)
    ax.set_xlabel(r"Total drift $\Delta$ (log$_{10}$ units)", fontsize=11)
    ax.set_ylabel(r"$R^2$", fontsize=11)
    ax.set_title("(C) Linear amplitude drift\n(towline calibration drift)",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    # Top axis: multiplicative factor range
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(DRIFT_LEVELS)
    ax2.set_xticklabels([f"x{10**(d/2):.2f}" for d in DRIFT_LEVELS], fontsize=7)
    ax2.set_xlabel("Max multiplicative factor", fontsize=9)

    # Panel D: Per-receiver bias
    ax = axes[1]
    for mname, (ls, mk, alpha) in model_styles.items():
        for pn, pl in zip(PARAM_NAMES, PARAM_LABELS):
            vals = [r[f"R2_{pn}_{mname}"] for r in rows_recv]
            ax.plot(RECV_BIAS_LEVELS, vals, marker=mk, ls=ls, color=colors[pl],
                    alpha=alpha, lw=2, ms=5,
                    label=f"{pl} {mname}" if pn in ("sigma2", "d2") else None)
    ax.set_xlabel(r"Per-receiver bias $\sigma_{\mathrm{recv}}$ (log$_{10}$)",
                  fontsize=11)
    ax.set_ylabel(r"$R^2$", fontsize=11)
    ax.set_title("(D) Per-receiver independent bias\n"
                 "(inter-receiver calibration mismatch)", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Panel E: Combined scenarios (grouped bar chart)
    ax = axes[2]
    scenario_names = [r["scenario"] for r in rows_combined]
    x_pos = np.arange(len(scenario_names))
    width = 0.35
    for i, (mname, (ls, mk, alpha)) in enumerate(model_styles.items()):
        means = [r[f"mean_R2_{mname}"] for r in rows_combined]
        offset = -width / 2 + i * width
        bars = ax.bar(x_pos + offset, means, width, label=mname, alpha=0.8)
        # Annotate σ2 R² on each bar
        s2_vals = [r[f"R2_sigma2_{mname}"] for r in rows_combined]
        for j, (bar, s2v) in enumerate(zip(bars, s2_vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{s2v:.2f}", ha="center", va="bottom", fontsize=7,
                    color=colors[r"$\sigma_2$"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel(r"Mean $R^2$", fontsize=11)
    ax.set_title("(E) Combined realistic scenarios\n"
                 r"(numbers: $R^2_{\sigma_2}$)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(bottom=min(0, min(r[f"mean_R2_Base"] for r in rows_combined) - 0.1))

    fig.suptitle(
        "DualTCN sensitivity to structured amplitude perturbations\n"
        f"({N_test:,} test samples; Base vs Amplitude-Augmented model)",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()
    out_png = os.path.join(PAPER_DIR, "structured_amplitude_figure.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
