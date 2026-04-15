"""
tau_sensitivity_experiment.py
==============================
Sensitivity analysis of the soft-step decoder transition width tau.

Addresses reviewer concern:

  "How sensitive are results to the soft-step decoder's transition width tau?
   Did you perform a sweep over tau (e.g., 1-8 m) to assess any induced bias
   in sigma2 and d2 estimates or profile misfit?"

Two experiments are performed:

  (F) Inference-time tau sweep — The model trained with tau=2 m is evaluated
      with different tau values at inference time.  Since the parameter
      predictions (sigma1, sigma2, d1, d2) are produced BEFORE the decoder,
      they are tau-independent.  This experiment isolates the effect of tau
      on the *profile* reconstruction quality (MSE vs ground truth) and shows
      that parameter estimates are invariant to this hyperparameter.

  (G) Training-time tau sweep — The model is retrained from scratch with
      tau in {0.5, 1, 2, 4, 8} m.  This measures whether different tau
      values during training induce biases in learned parameter estimates,
      since the profile loss gradient flows back through the decoder.

      This experiment requires GPU time (~5 hours per tau value).  If a
      trained model for a given tau is found in weights/, it is loaded
      instead of retraining.

Outputs
-------
data/csv/tau_sweep_inference.csv    — Experiment F
data/csv/tau_sweep_training.csv     — Experiment G (if models available)
paper/tau_sensitivity_figure.png    — multi-panel figure
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_p9d import (
    DATA_PATH_V4, TRAIN_SPLIT, VAL_SPLIT,
    IN_CHANNELS, N_TIME, N_DEPTH, LATENT_DIM, DROPOUT,
    LR, N_EPOCHS, WARMUP_EPOCHS,
    PROFILE_WEIGHT, PARAM_WEIGHT, PARAM_WEIGHTS,
    DSF_AUX_WEIGHT,
)
from config_v4 import (
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
    Z_MAX, BATCH_SIZE,
)
from model_p9d import LateTimePCRN
from model_v3 import reconstruct_profile, _Z_ARRAY
from model_v4 import denormalise_params_v4
from train_utils import WeightedPCRNLoss, evaluate, save_training_curve

SEED = 42
EVAL_BATCH = 1024

TAU_VALUES = [0.5, 1.0, 2.0, 4.0, 8.0]
DEFAULT_TAU = 2.0

BASE_MODEL_PATH = os.path.join(DIR, "best_model_p9d.pt")
CSV_DIR   = os.path.join(DIR, "data", "csv")
PAPER_DIR = os.path.join(DIR, "paper")

PARAM_NAMES  = ["sigma1", "sigma2", "d1", "d2"]
PARAM_LABELS = [r"$\sigma_1$", r"$\sigma_2$", r"$d_1$", r"$d_2$"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_model(path, device):
    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT)
    sd = torch.load(path, map_location=device, weights_only=True)
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    return model


def build_x_tensor(E_multi, log_amps):
    N, n_recv, n_time = E_multi.shape
    x = np.zeros((N, 2 * n_recv, n_time), dtype=np.float32)
    for j in range(n_recv):
        x[:, 2 * j,     :] = E_multi[:, j, :]
        x[:, 2 * j + 1, :] = log_amps[:, j, np.newaxis]
    return torch.tensor(x)


def build_gt_profile_with_tau(params, tau):
    """Reconstruct ground-truth profiles using a given tau for comparison."""
    N = len(params)
    log_s1 = torch.tensor(np.log10(params[:, 0] + 1e-12), dtype=torch.float32)
    log_s2 = torch.tensor(np.log10(params[:, 1] + 1e-12), dtype=torch.float32)
    log_d1 = torch.tensor(np.log10(params[:, 2] + 1e-12), dtype=torch.float32)
    log_d2 = torch.tensor(np.log10(params[:, 3] + 1e-12), dtype=torch.float32)
    with torch.no_grad():
        profiles = reconstruct_profile(log_s1, log_s2, log_d1, log_d2, tau=tau)
    return profiles.numpy()


def build_hard_step_profile(params):
    """Ground-truth hard-step profile (tau->0)."""
    z_arr = np.linspace(0.0, Z_MAX, N_DEPTH)
    N = len(params)
    profiles = np.zeros((N, N_DEPTH), dtype=np.float32)
    for i in range(N):
        s1, s2, d1, d2 = params[i]
        seafloor = d1 + d2
        sigma = np.where(z_arr < seafloor, s1, s2)
        profiles[i] = np.log10(sigma + 1e-12)
    return profiles


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved -> {path}")


# ── Experiment F: inference-time tau sweep ────────────────────────────────────

def experiment_F(model, E_test, A_test, P_test, gt_profiles_hard,
                 p_norm_true, device):
    """
    Sweep tau at inference time using the model trained with tau=2.

    Since the model's parameter heads produce predictions BEFORE the
    decoder, the parameter R² values should be IDENTICAL across all tau.
    Only the profile MSE changes (larger tau = smoother profile = higher MSE
    vs hard-step ground truth, but lower MSE near the interface).
    """
    print("\n" + "=" * 60)
    print("Experiment F: Inference-time tau sweep")
    print("  (model trained with tau=2.0, evaluated with varying tau)")
    print("=" * 60)

    rows = []

    for tau in TAU_VALUES:
        # Get parameter predictions (tau-independent)
        all_p_norm = []
        all_p_phys = []
        all_profiles = []

        with torch.no_grad():
            for i in range(0, len(E_test), EVAL_BATCH):
                sl = slice(i, min(i + EVAL_BATCH, len(E_test)))
                x = build_x_tensor(E_test[sl], A_test[sl]).to(device)

                # Run encoder + parameter heads (same regardless of tau)
                z_full = model.encoder_full(x)
                z_late = model.encoder_late(x[:, :, -N_TIME // 2:])
                z_comb = torch.cat([z_full, z_late], dim=-1)

                s1d1 = model.head_s1d1(z_full)
                s2d2 = model.head_s2d2(
                    torch.cat([z_comb, s1d1.detach()], dim=-1))

                p_norm = torch.stack(
                    [s1d1[:, 0], s2d2[:, 0], s1d1[:, 1], s2d2[:, 1]], dim=1)
                p_phys = denormalise_params_v4(p_norm)

                # Reconstruct profile with THIS tau
                profile = reconstruct_profile(
                    p_phys[:, 0], p_phys[:, 1],
                    p_phys[:, 2], p_phys[:, 3], tau=tau)

                all_p_norm.append(p_norm.cpu().numpy())
                all_p_phys.append(p_phys.cpu().numpy())
                all_profiles.append(profile.cpu().numpy())

        p_norm_pred = np.concatenate(all_p_norm, axis=0)
        p_phys_pred = np.concatenate(all_p_phys, axis=0)
        profiles_pred = np.concatenate(all_profiles, axis=0)

        # Parameter R² (should be constant across tau)
        r2_params = [r2(p_norm_true[:, k], p_norm_pred[:, k]) for k in range(4)]

        # Profile MSE vs hard-step ground truth
        profile_mse_hard = float(np.mean((profiles_pred - gt_profiles_hard) ** 2))

        # Profile MSE vs soft-step ground truth at same tau
        gt_soft = build_gt_profile_with_tau(P_test, tau)
        profile_mse_matched = float(np.mean((profiles_pred - gt_soft) ** 2))

        # Physical-unit biases for sigma2 and d2
        # p_phys: [log10(s1), log10(s2), log10(d1), log10(d2)]
        gt_log = np.log10(P_test + 1e-12)
        bias_s2 = float(np.mean(p_phys_pred[:, 1] - gt_log[:, 1]))
        bias_d2 = float(np.mean(p_phys_pred[:, 3] - gt_log[:, 3]))
        rmse_s2 = rmse(gt_log[:, 1], p_phys_pred[:, 1])
        rmse_d2 = rmse(gt_log[:, 3], p_phys_pred[:, 3])

        row = {
            "tau": tau,
            "R2_sigma1": r2_params[0], "R2_sigma2": r2_params[1],
            "R2_d1": r2_params[2], "R2_d2": r2_params[3],
            "mean_R2": float(np.mean(r2_params)),
            "profile_mse_vs_hard": profile_mse_hard,
            "profile_mse_vs_matched": profile_mse_matched,
            "bias_log_sigma2": bias_s2, "bias_log_d2": bias_d2,
            "rmse_log_sigma2": rmse_s2, "rmse_log_d2": rmse_d2,
        }
        rows.append(row)
        print(f"  tau={tau:4.1f}m  "
              f"R2(s2)={r2_params[1]:.4f}  R2(d2)={r2_params[3]:.4f}  "
              f"prof_MSE(hard)={profile_mse_hard:.4f}  "
              f"prof_MSE(match)={profile_mse_matched:.4f}  "
              f"bias(s2)={bias_s2:+.4f}  bias(d2)={bias_d2:+.4f}")

    return rows


# ── Experiment G: training-time tau sweep ─────────────────────────────────────

def experiment_G(E_test, A_test, P_test, gt_profiles_hard,
                 p_norm_true, device):
    """
    Evaluate models retrained with different tau values.
    Looks for weight files: best_model_p9d_tau{tau}.pt
    If a weight file doesn't exist, that tau is skipped with a message.
    """
    print("\n" + "=" * 60)
    print("Experiment G: Training-time tau sweep")
    print("  (separate model retrained for each tau)")
    print("=" * 60)

    rows = []
    for tau in TAU_VALUES:
        if tau == DEFAULT_TAU:
            model_path = BASE_MODEL_PATH
        else:
            model_path = os.path.join(DIR, f"best_model_p9d_tau{tau:.1f}.pt")

        if not os.path.exists(model_path):
            print(f"  tau={tau:.1f}m  SKIPPED (no model at {model_path})")
            continue

        print(f"  tau={tau:.1f}m  Loading {os.path.basename(model_path)} ...")
        model = load_model(model_path, device)

        all_p_norm = []
        all_p_phys = []
        all_profiles = []

        with torch.no_grad():
            for i in range(0, len(E_test), EVAL_BATCH):
                sl = slice(i, min(i + EVAL_BATCH, len(E_test)))
                x = build_x_tensor(E_test[sl], A_test[sl]).to(device)

                # Forward pass uses model's own tau (baked in during training)
                # but we override for evaluation consistency
                z_full = model.encoder_full(x)
                z_late = model.encoder_late(x[:, :, -N_TIME // 2:])
                z_comb = torch.cat([z_full, z_late], dim=-1)

                s1d1 = model.head_s1d1(z_full)
                s2d2 = model.head_s2d2(
                    torch.cat([z_comb, s1d1.detach()], dim=-1))

                p_norm = torch.stack(
                    [s1d1[:, 0], s2d2[:, 0], s1d1[:, 1], s2d2[:, 1]], dim=1)
                p_phys = denormalise_params_v4(p_norm)

                # Use this model's training tau for profile
                profile = reconstruct_profile(
                    p_phys[:, 0], p_phys[:, 1],
                    p_phys[:, 2], p_phys[:, 3], tau=tau)

                all_p_norm.append(p_norm.cpu().numpy())
                all_p_phys.append(p_phys.cpu().numpy())
                all_profiles.append(profile.cpu().numpy())

        p_norm_pred = np.concatenate(all_p_norm, axis=0)
        p_phys_pred = np.concatenate(all_p_phys, axis=0)
        profiles_pred = np.concatenate(all_profiles, axis=0)

        r2_params = [r2(p_norm_true[:, k], p_norm_pred[:, k]) for k in range(4)]

        profile_mse_hard = float(np.mean((profiles_pred - gt_profiles_hard) ** 2))

        gt_soft = build_gt_profile_with_tau(P_test, tau)
        profile_mse_matched = float(np.mean((profiles_pred - gt_soft) ** 2))

        gt_log = np.log10(P_test + 1e-12)
        bias_s2 = float(np.mean(p_phys_pred[:, 1] - gt_log[:, 1]))
        bias_d2 = float(np.mean(p_phys_pred[:, 3] - gt_log[:, 3]))
        rmse_s2 = rmse(gt_log[:, 1], p_phys_pred[:, 1])
        rmse_d2 = rmse(gt_log[:, 3], p_phys_pred[:, 3])

        row = {
            "tau": tau,
            "R2_sigma1": r2_params[0], "R2_sigma2": r2_params[1],
            "R2_d1": r2_params[2], "R2_d2": r2_params[3],
            "mean_R2": float(np.mean(r2_params)),
            "profile_mse_vs_hard": profile_mse_hard,
            "profile_mse_vs_matched": profile_mse_matched,
            "bias_log_sigma2": bias_s2, "bias_log_d2": bias_d2,
            "rmse_log_sigma2": rmse_s2, "rmse_log_d2": rmse_d2,
        }
        rows.append(row)
        print(f"           R2(s2)={r2_params[1]:.4f}  R2(d2)={r2_params[3]:.4f}  "
              f"prof_MSE(hard)={profile_mse_hard:.4f}  "
              f"bias(s2)={bias_s2:+.4f}  bias(d2)={bias_d2:+.4f}")

    return rows


# ── Training script for a single tau value ────────────────────────────────────

def train_single_tau(tau, device):
    """Retrain LateTimePCRN with a specific tau value."""
    from dataset_v4 import load_v4_data, get_dataloaders_v4
    from train_p9d import run_training_p9d

    model_path  = os.path.join(DIR, f"best_model_p9d_tau{tau:.1f}.pt")
    resume_path = os.path.join(DIR, f"resume_p9d_tau{tau:.1f}.pt")

    if os.path.exists(model_path):
        print(f"  Model already exists: {model_path}")
        return

    print(f"\n  Training tau={tau:.1f} m ...")

    # Monkey-patch the tau value used by reconstruct_profile
    import model_v3
    original_tau = model_v3.SOFT_STEP_TAU
    model_v3.SOFT_STEP_TAU = tau

    # Also patch in config_v4 in case it's read directly
    import config_v4
    config_v4.SOFT_STEP_TAU = tau

    E_multi, log_amps, sigma_profiles, params = load_v4_data()
    train_loader, val_loader, test_loader = get_dataloaders_v4(
        E_multi, log_amps, sigma_profiles, params)

    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_EPOCHS / N_EPOCHS,
        anneal_strategy="cos", final_div_factor=100,
    )
    criterion = WeightedPCRNLoss(
        profile_w=PROFILE_WEIGHT, param_w=PARAM_WEIGHT,
        param_weights=PARAM_WEIGHTS,
    )

    te, te_p, te_q, te_ppar = run_training_p9d(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler,
        model_path, resume_path, N_EPOCHS, device,
    )

    curve_path = os.path.join(DIR, f"training_curve_p9d_tau{tau:.1f}.png")
    save_training_curve(model, te, curve_path)
    print(f"  tau={tau:.1f}  Test loss={te:.4f}  "
          f"RMSE [s1={te_ppar[0]:.3f}, s2={te_ppar[1]:.3f}, "
          f"d1={te_ppar[2]:.3f}, d2={te_ppar[3]:.3f}]")

    # Restore original tau
    model_v3.SOFT_STEP_TAU = original_tau
    config_v4.SOFT_STEP_TAU = original_tau


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(rows_F, rows_G, P_test, N_test):
    """
    Create a figure with up to 4 panels:
      (a) Parameter R² vs tau (inference-time) — confirms tau-independence
      (b) Profile MSE vs tau (inference-time) — shows optimal tau
      (c) Parameter R² vs tau (training-time)  — if data available
      (d) sigma2/d2 bias vs tau (training-time) — if data available
    """
    has_G = len(rows_G) > 1  # need at least 2 points
    n_cols = 4 if has_G else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    colors = {r"$\sigma_1$": "#1f77b4", r"$\sigma_2$": "#ff7f0e",
              r"$d_1$": "#2ca02c", r"$d_2$": "#d62728"}

    taus_F = [r["tau"] for r in rows_F]

    # Panel (a): Parameter R² — inference-time
    ax = axes[0]
    for pn, pl in zip(PARAM_NAMES, PARAM_LABELS):
        vals = [r[f"R2_{pn}"] for r in rows_F]
        ax.plot(taus_F, vals, "o-", color=colors[pl], label=pl, lw=2, ms=6)
    ax.axvline(DEFAULT_TAU, color="gray", ls="--", lw=0.8, label=r"$\tau_{\mathrm{train}}$")
    ax.set_xlabel(r"Decoder $\tau$ (m)", fontsize=11)
    ax.set_ylabel(r"$R^2$", fontsize=11)
    ax.set_title("(a) Parameter $R^2$ vs inference $\\tau$\n"
                 "(confirms tau-independence)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(TAU_VALUES)
    ax.set_xticklabels([f"{t:.1f}" for t in TAU_VALUES])

    # Panel (b): Profile MSE — inference-time
    ax = axes[1]
    mse_hard = [r["profile_mse_vs_hard"] for r in rows_F]
    mse_match = [r["profile_mse_vs_matched"] for r in rows_F]
    ax.plot(taus_F, mse_hard, "s-", color="#e377c2", label="vs hard step", lw=2, ms=6)
    ax.plot(taus_F, mse_match, "^-", color="#8c564b", label=r"vs matched-$\tau$", lw=2, ms=6)
    ax.axvline(DEFAULT_TAU, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(r"Decoder $\tau$ (m)", fontsize=11)
    ax.set_ylabel("Profile MSE", fontsize=11)
    ax.set_title("(b) Profile MSE vs inference $\\tau$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(TAU_VALUES)
    ax.set_xticklabels([f"{t:.1f}" for t in TAU_VALUES])

    if has_G:
        taus_G = [r["tau"] for r in rows_G]

        # Panel (c): Parameter R² — training-time
        ax = axes[2]
        for pn, pl in zip(PARAM_NAMES, PARAM_LABELS):
            vals = [r[f"R2_{pn}"] for r in rows_G]
            ax.plot(taus_G, vals, "o-", color=colors[pl], label=pl, lw=2, ms=6)
        ax.axvline(DEFAULT_TAU, color="gray", ls="--", lw=0.8)
        ax.set_xlabel(r"Training $\tau$ (m)", fontsize=11)
        ax.set_ylabel(r"$R^2$", fontsize=11)
        ax.set_title("(c) Parameter $R^2$ vs training $\\tau$", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks(taus_G)
        ax.set_xticklabels([f"{t:.1f}" for t in taus_G])

        # Panel (d): Bias in sigma2 and d2 — training-time
        ax = axes[3]
        bias_s2 = [r["bias_log_sigma2"] for r in rows_G]
        bias_d2 = [r["bias_log_d2"] for r in rows_G]
        ax.plot(taus_G, bias_s2, "o-", color=colors[r"$\sigma_2$"],
                label=r"$\sigma_2$ bias", lw=2, ms=6)
        ax.plot(taus_G, bias_d2, "o-", color=colors[r"$d_2$"],
                label=r"$d_2$ bias", lw=2, ms=6)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.axvline(DEFAULT_TAU, color="gray", ls="--", lw=0.8)
        ax.set_xlabel(r"Training $\tau$ (m)", fontsize=11)
        ax.set_ylabel(r"Mean bias (log$_{10}$ units)", fontsize=11)
        ax.set_title(r"(d) $\sigma_2$ and $d_2$ bias vs training $\tau$",
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xscale("log")
        ax.set_xticks(taus_G)
        ax.set_xticklabels([f"{t:.1f}" for t in taus_G])

    fig.suptitle(
        r"Sensitivity to soft-step decoder transition width $\tau$"
        f"\n({N_test:,} test samples)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_png = os.path.join(PAPER_DIR, "tau_sensitivity_figure.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_png}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true",
                        help="Retrain models for each tau (requires GPU, ~5h each)")
    parser.add_argument("--tau", type=float, default=None,
                        help="Train a single tau value (use with --train)")
    args = parser.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Training mode ─────────────────────────────────────────────────────
    if args.train:
        taus_to_train = [args.tau] if args.tau else \
            [t for t in TAU_VALUES if t != DEFAULT_TAU]
        for tau in taus_to_train:
            train_single_tau(tau, device)
        if args.tau:
            print("Single-tau training done. Re-run without --train for evaluation.")
            return

    # ── Load test data ────────────────────────────────────────────────────
    print("Loading dataset ...")
    d = np.load(os.path.join(DIR, DATA_PATH_V4))
    n = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    sl = slice(n_train + n_val, n)

    E_test = d["E_multi"][sl]
    A_test = d["log_amps"][sl]
    P_test = d["params"][sl]
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

    gt_profiles_hard = build_hard_step_profile(P_test)

    # ── Experiment F ──────────────────────────────────────────────────────
    print("Loading base model ...")
    model = load_model(BASE_MODEL_PATH, device)

    rows_F = experiment_F(model, E_test, A_test, P_test,
                          gt_profiles_hard, p_norm_true, device)

    fields_F = ["tau", "R2_sigma1", "R2_sigma2", "R2_d1", "R2_d2", "mean_R2",
                "profile_mse_vs_hard", "profile_mse_vs_matched",
                "bias_log_sigma2", "bias_log_d2",
                "rmse_log_sigma2", "rmse_log_d2"]
    save_csv(os.path.join(CSV_DIR, "tau_sweep_inference.csv"), rows_F, fields_F)

    # ── Experiment G ──────────────────────────────────────────────────────
    rows_G = experiment_G(E_test, A_test, P_test,
                          gt_profiles_hard, p_norm_true, device)

    if rows_G:
        save_csv(os.path.join(CSV_DIR, "tau_sweep_training.csv"), rows_G, fields_F)
    else:
        print("\n  No retrained models found for Experiment G.")
        print("  Run with --train to retrain models for each tau.")

    # ── Figure ────────────────────────────────────────────────────────────
    make_figure(rows_F, rows_G, P_test, N_test)

    print("\nDone.")


if __name__ == "__main__":
    main()
