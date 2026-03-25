"""
train_v5_3layer.py — Train DualTCN for three-layer MCSEM inversion.

Demonstrates that the DualTCN architecture and physics decoder
generalise to N>2 layers with minimal modification (6 output
parameters instead of 4, two sigmoid transitions instead of one).
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_v5 import (
    N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS,
    LR, N_EPOCHS, WARMUP_EPOCHS,
    PROFILE_WEIGHT, PARAM_WEIGHT, PARAM_WEIGHTS_V5, DROPOUT,
    N_PARAMS_V5,
)
from dataset_v5 import load_v5_data, get_dataloaders_v5
from model_v5 import DualTCN3Layer
from train_utils import save_training_curve

MODEL_PATH  = "best_model_v5_3layer.pt"
RESUME_PATH = "resume_v5_3layer.pt"
CURVE_PATH  = "training_curve_v5_3layer.png"
PAPER_DIR   = os.path.join(DIR, "paper")


class WeightedLoss3Layer(nn.Module):
    """Combined profile MSE + weighted per-parameter Huber loss (6 params)."""
    def __init__(self, profile_w=1.0, param_w=2.0, param_weights=None):
        super().__init__()
        self.profile_w = profile_w
        self.param_w = param_w
        self.huber = nn.HuberLoss(delta=0.1, reduction="none")
        if param_weights is not None:
            self.register_buffer("pw",
                                 torch.tensor(param_weights, dtype=torch.float32))
        else:
            self.register_buffer("pw", torch.ones(N_PARAMS_V5))

    def forward(self, profile_pred, profile_true, param_pred, param_true):
        prof_loss = nn.functional.mse_loss(profile_pred, profile_true)
        param_loss = (self.huber(param_pred, param_true) * self.pw).mean()
        total = self.profile_w * prof_loss + self.param_w * param_loss
        return total, prof_loss.item(), param_loss.item()


def train_one_epoch(model, loader, optimizer, criterion, scheduler,
                    device, scaler):
    model.train()
    tot, prof_tot, par_tot = 0.0, 0.0, 0.0

    for x, sig, p_true in loader:
        x = x.to(device, non_blocking=True)
        sig = sig.to(device, non_blocking=True)
        p_true = p_true.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            profile_pred, p_norm_pred, p_phys_pred = model(x)
            loss, pl, ql = criterion(profile_pred, sig, p_norm_pred, p_true)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        b = x.size(0)
        tot += loss.item() * b
        prof_tot += pl * b
        par_tot += ql * b

    n = len(loader.dataset)
    return tot / n, prof_tot / n, par_tot / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot, prof_tot, par_tot = 0.0, 0.0, 0.0
    all_pred, all_true = [], []

    for x, sig, p_true in loader:
        x = x.to(device, non_blocking=True)
        sig = sig.to(device, non_blocking=True)
        p_true = p_true.to(device, non_blocking=True)

        profile_pred, p_norm_pred, _ = model(x)
        loss, pl, ql = criterion(profile_pred, sig, p_norm_pred, p_true)

        b = x.size(0)
        tot += loss.item() * b
        prof_tot += pl * b
        par_tot += ql * b
        all_pred.append(p_norm_pred.cpu())
        all_true.append(p_true.cpu())

    n = len(loader.dataset)
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    return tot / n, prof_tot / n, par_tot / n, rmse


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Experiment: DualTCN 3-layer inversion (6 parameters)")
    print(f"  Parameters: σ₁, σ₂, σ₃, d₁, d₂, h")
    print(f"  Model save path: {MODEL_PATH}")

    E_multi, log_amps, sigma_profiles, params = load_v5_data()
    train_loader, val_loader, test_loader = get_dataloaders_v5(
        E_multi, log_amps, sigma_profiles, params)

    model = DualTCN3Layer(in_ch=IN_CHANNELS, in_len=N_TIME,
                           out_len=N_DEPTH, latent_dim=LATENT_DIM,
                           dropout=DROPOUT).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DualTCN3Layer parameters: {n_par:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_EPOCHS / N_EPOCHS,
        anneal_strategy="cos", final_div_factor=100)
    criterion = WeightedLoss3Layer(
        profile_w=PROFILE_WEIGHT, param_w=PARAM_WEIGHT,
        param_weights=PARAM_WEIGHTS_V5).to(device)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_val = float("inf")

    if os.path.exists(RESUME_PATH):
        print(f"Resuming from {RESUME_PATH} ...")
        ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
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
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    param_names = ["σ₁", "σ₂", "σ₃", "d₁", "d₂", "h"]
    print(f"\n{'Ep':>5}  {'Tr':>8}  {'Va':>8}  " +
          "  ".join(f"{n:>6}" for n in param_names))
    print("-" * 70)

    for epoch in range(start_epoch, N_EPOCHS + 1):
        tr, tr_p, tr_q = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, scaler)
        va, va_p, va_q, ppar = evaluate(
            model, val_loader, criterion, device)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), MODEL_PATH)

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
            }, RESUME_PATH)

        if epoch % 20 == 0 or epoch == start_epoch:
            rmse_str = "  ".join(f"{r:>6.3f}" for r in ppar)
            print(f"{epoch:>5}  {tr:>8.4f}  {va:>8.4f}  {rmse_str}")

    # Final test
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if hasattr(model, '_orig_mod'):
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model._orig_mod.load_state_dict(clean_sd)
    else:
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)

    te, te_p, te_q, te_rmse = evaluate(model, test_loader, criterion, device)
    print(f"\nTest — total={te:.4f}  profile={te_p:.4f}  param={te_q:.4f}")
    rmse_str = "  ".join(f"{n}={r:.3f}" for n, r in zip(param_names, te_rmse))
    print(f"Per-param RMSE [norm] — {rmse_str}")

    # Save results
    os.makedirs(PAPER_DIR, exist_ok=True)
    import csv
    path = os.path.join(PAPER_DIR, "3layer_results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric"] + param_names)
        w.writerow(["rmse"] + [f"{r:.4f}" for r in te_rmse])
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
