"""
train_p9d.py — P9d: LateTimePCRN + standard input + d_sf auxiliary loss.

Extends the standard WeightedPCRNLoss with an auxiliary Huber loss on
the total seafloor depth d_sf = d₁ + d₂ predicted by model.head_dsf.
"""
import os
import numpy as np
import torch
import torch.nn as nn

from config_p9d import (
    N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS,
    LR, N_EPOCHS, WARMUP_EPOCHS,
    PROFILE_WEIGHT, PARAM_WEIGHT, PARAM_WEIGHTS, DROPOUT,
    DSF_AUX_WEIGHT,
)
from dataset_v4 import load_v4_data, get_dataloaders_v4
from model_p9d import LateTimePCRN
from model_v4 import denormalise_params_v4
from train_utils import (
    WeightedPCRNLoss, evaluate, run_training, save_training_curve,
)

MODEL_PATH  = "best_model_p9d.pt"
RESUME_PATH = "resume_p9d.pt"
CURVE_PATH  = "training_curve_p9d.png"


# ── Custom training epoch with d_sf auxiliary loss ────────────────────────────

def train_one_epoch_p9d(model, loader, optimizer, criterion, scheduler, device,
                        dsf_aux_w=DSF_AUX_WEIGHT):
    model.train()
    tot, prof_tot, par_tot = 0.0, 0.0, 0.0
    grad_norms = []
    huber = nn.HuberLoss(delta=0.1, reduction="mean")

    for x, sig, p_true in loader:
        x, sig, p_true = x.to(device), sig.to(device), p_true.to(device)

        optimizer.zero_grad()
        profile_pred, p_norm_pred, p_phys_pred = model(x)

        # Standard PCRN loss
        loss_main, pl, ql = criterion(profile_pred, sig, p_norm_pred, p_true)

        # Auxiliary d_sf loss: Huber( dsf_pred, norm_dsf(d1_true + d2_true) )
        with torch.no_grad():
            p_phys_true = denormalise_params_v4(p_true)
            d1_t = 10 ** p_phys_true[:, 2]    # physical d₁ [m]
            d2_t = 10 ** p_phys_true[:, 3]    # physical d₂ [m]
            dsf_true = model.norm_dsf(d1_t + d2_t).unsqueeze(1)  # (B,1)

        loss_dsf = huber(model.dsf_pred, dsf_true)
        loss = loss_main + dsf_aux_w * loss_dsf
        loss.backward()

        gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(float(gn))
        optimizer.step()
        scheduler.step()

        b = x.size(0)
        tot      += loss.item() * b
        prof_tot += pl * b
        par_tot  += ql * b

    n = len(loader.dataset)
    return tot / n, prof_tot / n, par_tot / n, float(np.mean(grad_norms))


# ── Customised run_training that swaps in our epoch function ──────────────────

def run_training_p9d(model, train_loader, val_loader, test_loader,
                     criterion, optimizer, scheduler,
                     model_path, resume_path, n_epochs, device):
    """Like run_training but uses train_one_epoch_p9d."""
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
        tr, tr_p, tr_q, gn = train_one_epoch_p9d(
            model, train_loader, optimizer, criterion, scheduler, device,
        )
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

    # Final test evaluation
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    te, te_p, te_q, te_ppar = evaluate(model, test_loader, criterion, device)
    print(f"\nTest — total={te:.4f}  profile={te_p:.4f}  param={te_q:.4f}")
    print(f"Per-param RMSE [norm] — s1={te_ppar[0]:.3f}  s2={te_ppar[1]:.3f}"
          f"  d1={te_ppar[2]:.3f}  d2={te_ppar[3]:.3f}")

    model._train_hist = train_hist
    model._val_hist   = val_hist
    model._ppar_hist  = ppar_hist
    return te, te_p, te_q, te_ppar


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    E_multi, log_amps, sigma_profiles, params = load_v4_data()
    train_loader, val_loader, test_loader = get_dataloaders_v4(
        E_multi, log_amps, sigma_profiles, params
    )

    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LateTimePCRN (P9d) parameters: {n_par:,}")
    print(f"  DSF auxiliary weight: {DSF_AUX_WEIGHT}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=WARMUP_EPOCHS / N_EPOCHS,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    criterion = WeightedPCRNLoss(
        profile_w=PROFILE_WEIGHT,
        param_w=PARAM_WEIGHT,
        param_weights=PARAM_WEIGHTS,
    )

    te, te_p, te_q, te_ppar = run_training_p9d(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler,
        MODEL_PATH, RESUME_PATH, N_EPOCHS, device,
    )
    save_training_curve(model, te, CURVE_PATH)


if __name__ == "__main__":
    main()
