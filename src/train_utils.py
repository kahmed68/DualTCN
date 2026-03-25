"""
train_utils.py — Shared training utilities for MCSEM PCRN experiments.

Provides:
  WeightedPCRNLoss  — profile MSE + weighted per-parameter Huber loss
  train_one_epoch   — single epoch training step
  evaluate          — evaluation with per-parameter RMSE
  run_training      — full training loop with resume support
  save_training_curve — save loss/param-RMSE plot to PNG
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config_v4 import (
    PROFILE_WEIGHT, PARAM_WEIGHT, PARAM_WEIGHTS,
)


# ── Weighted parameter loss ────────────────────────────────────────────────────

class WeightedPCRNLoss(nn.Module):
    """
    L_total = PROFILE_WEIGHT × MSE(profile_pred, profile_true)
            + PARAM_WEIGHT   × Σ_i  w_i × Huber(pred_i, true_i)

    param_weights: per-parameter scale factors for [σ₁, σ₂, d₁, d₂].
    """

    def __init__(self, profile_w=PROFILE_WEIGHT, param_w=PARAM_WEIGHT,
                 param_weights=None, huber_delta=0.1):
        super().__init__()
        self.profile_w   = profile_w
        self.param_w     = param_w
        pw = param_weights if param_weights is not None else PARAM_WEIGHTS
        self.w           = torch.tensor(pw, dtype=torch.float32)
        self.huber_delta = huber_delta
        self.mse         = nn.MSELoss()

    def forward(self, profile_pred, profile_true, p_norm_pred, p_norm_true):
        profile_loss = self.mse(profile_pred, profile_true)

        w = self.w.to(p_norm_pred.device)
        diff     = p_norm_pred - p_norm_true          # (B, 4)
        abs_diff = diff.abs()
        huber_per = torch.where(
            abs_diff <= self.huber_delta,
            0.5 * diff ** 2,
            self.huber_delta * (abs_diff - 0.5 * self.huber_delta),
        )                                              # (B, 4)
        param_loss = (huber_per * w).mean()

        total = self.profile_w * profile_loss + self.param_w * param_loss
        return total, profile_loss.item(), param_loss.item()


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    tot, prof_tot, par_tot = 0.0, 0.0, 0.0
    grad_norms = []

    for x, sig, p_true in loader:
        x, sig, p_true = x.to(device), sig.to(device), p_true.to(device)

        optimizer.zero_grad()
        profile_pred, p_norm_pred, _ = model(x)
        loss, pl, ql = criterion(profile_pred, sig, p_norm_pred, p_true)
        loss.backward()

        gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(float(gn))
        optimizer.step()
        scheduler.step()   # OneCycleLR steps per batch

        b = x.size(0)
        tot      += loss.item() * b
        prof_tot += pl * b
        par_tot  += ql * b

    n = len(loader.dataset)
    return tot / n, prof_tot / n, par_tot / n, float(np.mean(grad_norms))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot, prof_tot, par_tot = 0.0, 0.0, 0.0
    param_sq_err = np.zeros(4)
    n_samples    = 0

    for x, sig, p_true in loader:
        x, sig, p_true = x.to(device), sig.to(device), p_true.to(device)
        profile_pred, p_norm_pred, _ = model(x)
        loss, pl, ql = criterion(profile_pred, sig, p_norm_pred, p_true)

        b = x.size(0)
        tot      += loss.item() * b
        prof_tot += pl * b
        par_tot  += ql * b

        err = (p_norm_pred - p_true).cpu().numpy() ** 2
        param_sq_err += err.sum(axis=0)
        n_samples    += b

    n = len(loader.dataset)
    per_param_rmse = np.sqrt(param_sq_err / n_samples)
    return tot / n, prof_tot / n, par_tot / n, per_param_rmse


# ── Full training loop with resume support ────────────────────────────────────

def run_training(model, train_loader, val_loader, test_loader,
                 criterion, optimizer, scheduler,
                 model_path, resume_path, n_epochs, device):
    """
    Full training loop with resume support.

    Saves best model to model_path.
    Saves full resume checkpoint to resume_path every 10 epochs.
    Prints epoch info every 20 epochs (and at start_epoch).

    Returns
    -------
    (te, te_p, te_q, te_ppar) — test set total/profile/param losses + per-param RMSE
    """
    start_epoch = 1
    best_val    = float("inf")
    train_hist  = {"total": [], "profile": [], "param": []}
    val_hist    = {"total": [], "profile": [], "param": []}
    ppar_hist   = []

    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        train_hist  = ckpt["train_hist"]
        val_hist    = ckpt["val_hist"]
        ppar_hist   = ckpt["ppar_hist"]
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    print(f"\n{'Ep':>5}  {'Tr':>8}  {'Va':>8}  "
          f"{'s1':>6}  {'s2':>6}  {'d1':>6}  {'d2':>6}  "
          f"{'GradN':>7}")
    print("-" * 62)

    for epoch in range(start_epoch, n_epochs + 1):
        tr, tr_p, tr_q, gn = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device
        )
        va, va_p, va_q, ppar = evaluate(
            model, val_loader, criterion, device
        )

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

        # Save full resume checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "best_val":   best_val,
                "train_hist": train_hist,
                "val_hist":   val_hist,
                "ppar_hist":  ppar_hist,
            }, resume_path)

        if epoch % 20 == 0 or epoch == start_epoch:
            print(f"{epoch:>5}  {tr:>8.4f}  {va:>8.4f}  "
                  f"{ppar[0]:>6.3f}  {ppar[1]:>6.3f}  "
                  f"{ppar[2]:>6.3f}  {ppar[3]:>6.3f}  "
                  f"{gn:>7.3f}")

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    te, te_p, te_q, te_ppar = evaluate(model, test_loader, criterion, device)
    print(f"\nTest — total={te:.4f}  profile={te_p:.4f}  param={te_q:.4f}")
    print(f"Per-param RMSE [norm] — s1={te_ppar[0]:.3f}  s2={te_ppar[1]:.3f}"
          f"  d1={te_ppar[2]:.3f}  d2={te_ppar[3]:.3f}")

    # Store histories for curve saving
    model._train_hist = train_hist
    model._val_hist   = val_hist
    model._ppar_hist  = ppar_hist

    return te, te_p, te_q, te_ppar


# ── Training curve ─────────────────────────────────────────────────────────────

def save_training_curve(model, te, curve_path):
    """
    Save a 3-panel training curve PNG.

    Reads histories from model._train_hist, model._val_hist, model._ppar_hist
    (set by run_training).
    """
    train_hist = getattr(model, "_train_hist", {"total": [], "profile": [], "param": []})
    val_hist   = getattr(model, "_val_hist",   {"total": [], "profile": [], "param": []})
    ppar_hist  = getattr(model, "_ppar_hist",  [])

    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_hist["total"], label="Train")
    ax.plot(val_hist["total"],   label="Validation")
    ax.axhline(te, color="red", linestyle="--", label=f"Test={te:.4f}")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(val_hist["profile"], label="Val Profile MSE")
    ax.plot(val_hist["param"],   label="Val Param (weighted Huber)")
    ax.set_title("Val Loss Components")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    if ppar_hist:
        ppar_arr = np.array(ppar_hist)   # (epochs, 4)
        for i, lbl in enumerate(["s1", "s2", "d1", "d2"]):
            ax.plot(ppar_arr[:, i], label=lbl)
    ax.set_title("Per-Parameter Val RMSE  [normalised 0-1 space]")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("PCRN — Training History", fontsize=13)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"Training curve saved to '{curve_path}'")
