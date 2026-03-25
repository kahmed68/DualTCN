"""
train_p9d_weighted.py — Retrain DualTCN with inverse-σ₂ sample weighting.

Samples with low σ₂ (resistive targets) receive higher loss weight,
forcing the network to pay more attention to the hardest cases.

Weight per sample: w_i = 1 / (σ₂_norm_i + ε)
Normalised so that mean weight across the batch ≈ 1.

Uses curriculum-based amplitude augmentation (same as DualTCN-AmpAug)
combined with the sample weighting.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import csv

torch.backends.cudnn.benchmark = True

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_p9d import (
    N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS,
    LR, N_EPOCHS, WARMUP_EPOCHS,
    PROFILE_WEIGHT, PARAM_WEIGHT, PARAM_WEIGHTS, DROPOUT,
    DSF_AUX_WEIGHT, DATA_PATH_V4,
    TRAIN_SPLIT, VAL_SPLIT,
)
from dataset_v4 import load_v4_data, get_dataloaders_v4
from model_p9d import LateTimePCRN
from model_v4 import denormalise_params_v4
from train_utils import WeightedPCRNLoss, evaluate, save_training_curve

MODEL_PATH  = "best_model_p9d_weighted.pt"
RESUME_PATH = "resume_p9d_weighted.pt"
CURVE_PATH  = "training_curve_p9d_weighted.png"
PAPER_DIR   = os.path.join(DIR, "paper")


# ── Sample-weighted loss ─────────────────────────────────────────────────────

class SampleWeightedPCRNLoss(nn.Module):
    """WeightedPCRNLoss with per-sample inverse-σ₂ weighting."""

    def __init__(self, profile_w=PROFILE_WEIGHT, param_w=PARAM_WEIGHT,
                 param_weights=None, huber_delta=0.1):
        super().__init__()
        self.profile_w = profile_w
        self.param_w = param_w
        pw = param_weights if param_weights is not None else PARAM_WEIGHTS
        self.w = torch.tensor(pw, dtype=torch.float32)
        self.huber_delta = huber_delta

    def forward(self, profile_pred, profile_true, p_norm_pred, p_norm_true):
        w = self.w.to(p_norm_pred.device)

        # Profile loss per sample
        prof_per_sample = (profile_pred - profile_true).pow(2).mean(dim=1)  # (B,)

        # Param loss per sample
        diff = p_norm_pred - p_norm_true
        abs_diff = diff.abs()
        huber_per = torch.where(
            abs_diff <= self.huber_delta,
            0.5 * diff ** 2,
            self.huber_delta * (abs_diff - 0.5 * self.huber_delta),
        )  # (B, 4)
        param_per_sample = (huber_per * w).mean(dim=1)  # (B,)

        # Per-sample total loss (unweighted)
        loss_per_sample = (self.profile_w * prof_per_sample
                           + self.param_w * param_per_sample)  # (B,)

        # Inverse-σ₂ sample weight: σ₂ is param index 1 (normalised [0,1])
        sigma2_norm = p_norm_true[:, 1]  # (B,)
        sample_w = 1.0 / (sigma2_norm + 0.05)  # ε=0.05 prevents div-by-zero
        sample_w = sample_w / sample_w.mean()  # normalise so mean ≈ 1

        # Weighted mean
        total = (loss_per_sample * sample_w).mean()

        return total, prof_per_sample.mean().item(), param_per_sample.mean().item()


# ── Evaluation with per-σ₂-bin breakdown ─────────────────────────────────────

def evaluate_by_sigma2_bin(model_path, device):
    """Evaluate DualTCN accuracy binned by σ₂ value."""
    from config_v4 import (
        LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
        LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
        LOG_D1_MIN, LOG_D1_MAX,
        LOG_D2_MIN, LOG_D2_MAX,
    )
    from dataset_v4 import MCSEMDatasetV4
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("Per-σ₂-bin evaluation on sample-weighted model")
    print("=" * 60)

    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=True)
    if hasattr(model, '_orig_mod'):
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model._orig_mod.load_state_dict(clean_sd)
    else:
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
    model.eval()

    d = np.load(os.path.join(DIR, DATA_PATH_V4))
    n = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)
    tst_sl = slice(n_train + n_val, n)

    E_test = d["E_multi"][tst_sl]
    A_test = d["log_amps"][tst_sl]
    P_test = d["params"][tst_sl]
    Pr_test = d["sigma_profiles"][tst_sl]

    bounds = np.array([[LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
                       [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
                       [LOG_D1_MIN, LOG_D1_MAX],
                       [LOG_D2_MIN, LOG_D2_MAX]])

    def to_norm(P):
        lp = np.log10(P + 1e-12)
        return np.clip((lp - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-8),
                       0., 1.).astype(np.float32)

    y_test = to_norm(P_test)
    sigma2_phys = P_test[:, 1]  # physical σ₂ values

    # Get predictions
    ds = MCSEMDatasetV4(E_test, A_test, Pr_test, P_test, augment=False)
    loader = DataLoader(ds, batch_size=512, shuffle=False,
                       num_workers=0, pin_memory=True)
    all_pred = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            _, p, _ = model(x)
            all_pred.append(p.cpu().numpy())
    pred = np.concatenate(all_pred, 0)

    # Bin by σ₂
    bins = [(0.001, 0.01, "0.001–0.01"),
            (0.01, 0.05, "0.01–0.05"),
            (0.05, 0.1, "0.05–0.1"),
            (0.1, 0.3, "0.1–0.3"),
            (0.3, 1.0, "0.3–1.0")]

    print(f"\n{'σ₂ range':>15}  {'N':>6}  {'σ₁':>6}  {'σ₂':>6}  {'d₁':>6}  {'d₂':>6}  {'R̄²':>6}")
    print("-" * 60)

    rows = []
    for lo, hi, label in bins:
        mask = (sigma2_phys >= lo) & (sigma2_phys < hi)
        n_bin = mask.sum()
        if n_bin < 10:
            continue

        y_bin = y_test[mask]
        p_bin = pred[mask]
        y_var = np.var(y_bin, axis=0)
        rmse = np.sqrt(np.mean((y_bin - p_bin) ** 2, axis=0))
        r2 = 1.0 - np.mean((y_bin - p_bin) ** 2, axis=0) / (y_var + 1e-12)
        r2_mean = r2.mean()

        print(f"{label:>15}  {n_bin:>6}  {rmse[0]:>6.3f}  {rmse[1]:>6.3f}  "
              f"{rmse[2]:>6.3f}  {rmse[3]:>6.3f}  {r2_mean:>6.3f}")
        rows.append({
            "sigma2_range": label, "n_samples": n_bin,
            "rmse_sigma1": f"{rmse[0]:.4f}", "rmse_sigma2": f"{rmse[1]:.4f}",
            "rmse_d1": f"{rmse[2]:.4f}", "rmse_d2": f"{rmse[3]:.4f}",
            "r2_mean": f"{r2_mean:.4f}",
        })

    # Save CSV
    os.makedirs(PAPER_DIR, exist_ok=True)
    path = os.path.join(PAPER_DIR, "weighted_sigma2_bins.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved → {path}")

    # Also run the field benchmark
    from benchmark_field_models import run_benchmark
    print("\n")
    run_benchmark(model_path, device)


# ── AMP training with sample weighting ───────────────────────────────────────

def train_one_epoch_weighted(model, loader, optimizer, criterion, scheduler,
                             device, scaler, dsf_aux_w=DSF_AUX_WEIGHT):
    model.train()
    tot, prof_tot, par_tot = 0.0, 0.0, 0.0
    grad_norms = []
    huber = nn.HuberLoss(delta=0.1, reduction="mean")

    for x, sig, p_true in loader:
        x, sig, p_true = x.to(device, non_blocking=True), \
                          sig.to(device, non_blocking=True), \
                          p_true.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            profile_pred, p_norm_pred, p_phys_pred = model(x)
            loss_main, pl, ql = criterion(profile_pred, sig, p_norm_pred, p_true)

            with torch.no_grad():
                p_phys_true = denormalise_params_v4(p_true)
                d1_t = 10 ** p_phys_true[:, 2]
                d2_t = 10 ** p_phys_true[:, 3]
                dsf_true = model.norm_dsf(d1_t + d2_t).unsqueeze(1)

            loss_dsf = huber(model.dsf_pred, dsf_true)
            loss = loss_main + dsf_aux_w * loss_dsf

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        grad_norms.append(float(gn))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        b = x.size(0)
        tot += loss.item() * b
        prof_tot += pl * b
        par_tot += ql * b

    n = len(loader.dataset)
    return tot / n, prof_tot / n, par_tot / n, float(np.mean(grad_norms))


def run_training_weighted(model, train_loader, val_loader, test_loader,
                          criterion, optimizer, scheduler,
                          model_path, resume_path, n_epochs, device):
    start_epoch = 1
    best_val = float("inf")
    train_hist = {"total": [], "profile": [], "param": []}
    val_hist = {"total": [], "profile": [], "param": []}
    ppar_hist = []
    scaler = torch.amp.GradScaler("cuda")

    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if hasattr(model, '_orig_mod'):
            clean_sd = {k.replace('_orig_mod.', ''): v
                        for k, v in ckpt["model"].items()}
            model._orig_mod.load_state_dict(clean_sd)
        else:
            clean_sd = {k.replace('_orig_mod.', ''): v
                        for k, v in ckpt["model"].items()}
            model.load_state_dict(clean_sd)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        train_hist = ckpt.get("train_hist", train_hist)
        val_hist = ckpt.get("val_hist", val_hist)
        ppar_hist = ckpt.get("ppar_hist", ppar_hist)
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    print(f"\n{'Ep':>5}  {'Tr':>8}  {'Va':>8}  "
          f"{'s1':>6}  {'s2':>6}  {'d1':>6}  {'d2':>6}  "
          f"{'GradN':>7}")
    print("-" * 62)

    for epoch in range(start_epoch, n_epochs + 1):
        tr, tr_p, tr_q, gn = train_one_epoch_weighted(
            model, train_loader, optimizer, criterion, scheduler,
            device, scaler)
        va, va_p, va_q, ppar = evaluate(model, val_loader, criterion, device)

        train_hist["total"].append(tr)
        train_hist["profile"].append(tr_p)
        train_hist["param"].append(tr_q)
        val_hist["total"].append(va)
        val_hist["profile"].append(va_p)
        val_hist["param"].append(va_q)
        ppar_hist.append(ppar)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), model_path)

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "train_hist": train_hist,
                "val_hist": val_hist,
                "ppar_hist": ppar_hist,
            }, resume_path)

        if epoch % 20 == 0 or epoch == start_epoch:
            print(f"{epoch:>5}  {tr:>8.4f}  {va:>8.4f}  "
                  f"{ppar[0]:>6.3f}  {ppar[1]:>6.3f}  "
                  f"{ppar[2]:>6.3f}  {ppar[3]:>6.3f}  "
                  f"{gn:>7.3f}")

    # Final test
    sd = torch.load(model_path, map_location=device, weights_only=True)
    if hasattr(model, '_orig_mod'):
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model._orig_mod.load_state_dict(clean_sd)
    else:
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
    te, te_p, te_q, te_ppar = evaluate(model, test_loader, criterion, device)
    print(f"\nTest — total={te:.4f}  profile={te_p:.4f}  param={te_q:.4f}")
    print(f"Per-param RMSE [norm] — s1={te_ppar[0]:.3f}  s2={te_ppar[1]:.3f}"
          f"  d1={te_ppar[2]:.3f}  d2={te_ppar[3]:.3f}")

    model._train_hist = train_hist
    model._val_hist = val_hist
    model._ppar_hist = ppar_hist
    return te, te_p, te_q, te_ppar


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: DualTCN with inverse-σ₂ sample weighting")
    print(f"  Weight: w_i = 1 / (σ₂_norm + 0.05), normalised")
    print(f"  Model save path: {MODEL_PATH}")

    E_multi, log_amps, sigma_profiles, params = load_v4_data()
    train_loader, val_loader, test_loader = get_dataloaders_v4(
        E_multi, log_amps, sigma_profiles, params)

    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LateTimePCRN (DualTCN-Weighted) parameters: {n_par:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_EPOCHS / N_EPOCHS,
        anneal_strategy="cos", final_div_factor=100)
    criterion = SampleWeightedPCRNLoss(
        profile_w=PROFILE_WEIGHT, param_w=PARAM_WEIGHT,
        param_weights=PARAM_WEIGHTS)

    te, te_p, te_q, te_ppar = run_training_weighted(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler,
        MODEL_PATH, RESUME_PATH, N_EPOCHS, device)
    save_training_curve(model, te, CURVE_PATH)

    # Per-σ₂-bin evaluation + field benchmark
    evaluate_by_sigma2_bin(MODEL_PATH, device)


if __name__ == "__main__":
    main()
