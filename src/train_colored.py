"""
train_p9d_colored.py — Retrain DualTCN (P9d) with colored (1/f) waveform noise.

Replaces white Gaussian waveform noise with temporally correlated
pink noise (1/f spectrum).  Real MCSEM noise is dominated by
ocean-current motion noise and magnetotelluric signals, which have
a red/pink power spectrum.  After training, evaluates noise
robustness at multiple SNR levels for both white and colored noise.
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
from dataset_v4 import load_v4_data
from dataset_v4_colored import get_dataloaders_v4_colored, _generate_pink_noise
from model_p9d import LateTimePCRN
from model_v4 import denormalise_params_v4
from train_utils import WeightedPCRNLoss, evaluate, save_training_curve

MODEL_PATH  = "best_model_p9d_colored.pt"
RESUME_PATH = "resume_p9d_colored.pt"
CURVE_PATH  = "training_curve_p9d_colored.png"
PAPER_DIR   = os.path.join(DIR, "paper")


# ── Colored noise evaluation ─────────────────────────────────────────────────

def evaluate_colored_robustness(model_path, device):
    """Evaluate robustness to both white and colored noise."""
    from config_v4 import (
        LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
        LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
        LOG_D1_MIN, LOG_D1_MAX,
        LOG_D2_MIN, LOG_D2_MAX,
    )
    from dataset_v4 import MCSEMDatasetV4
    from torch.utils.data import DataLoader

    print("\n" + "=" * 60)
    print("Noise robustness evaluation on colored-noise-trained model")
    print("=" * 60)

    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=True)
    # Handle torch.compile: strip _orig_mod. prefix if model is not compiled,
    # or add it if model is compiled but keys don't have it
    if hasattr(model, '_orig_mod'):
        # Model is compiled — load into the underlying module
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model._orig_mod.load_state_dict(clean_sd)
    else:
        # Model is not compiled — strip prefix if present
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
    model.eval()

    d = np.load(os.path.join(DIR, DATA_PATH_V4))
    n = len(d["E_multi"])
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    tst_sl  = slice(n_train + n_val, n)

    E_test  = d["E_multi"][tst_sl]
    A_test  = d["log_amps"][tst_sl]
    P_test  = d["params"][tst_sl]
    Pr_test = d["sigma_profiles"][tst_sl]

    bounds = np.array([[LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
                       [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
                       [LOG_D1_MIN,     LOG_D1_MAX    ],
                       [LOG_D2_MIN,     LOG_D2_MAX    ]])

    def to_norm(P):
        lp = np.log10(P + 1e-12)
        return np.clip((lp - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0] + 1e-8),
                       0., 1.).astype(np.float32)

    y_test = to_norm(P_test)
    y_var  = np.var(y_test, axis=0)

    def r2_per_param(y_true, y_pred):
        ss_res = np.mean((y_true - y_pred) ** 2, axis=0)
        return 1.0 - ss_res / (y_var + 1e-12)

    @torch.no_grad()
    def predict_with_noise(E, A, Pr, P, noise_type="none", snr_db=None):
        """Predict with injected white or colored noise at given SNR."""
        ds = MCSEMDatasetV4(E, A, Pr, P, augment=False)
        loader = DataLoader(ds, batch_size=512, shuffle=False,
                           num_workers=0, pin_memory=True)
        preds = []
        for x, _, _ in loader:
            x = x.to(device)
            if noise_type != "none" and snr_db is not None:
                bs, nc, nt = x.shape
                for ch in [0, 2, 4, 6]:  # waveform channels
                    signal = x[:, ch, :]
                    rms = signal.pow(2).mean(dim=1, keepdim=True).sqrt()
                    sigma_n = rms / (10 ** (snr_db / 20.0))
                    if noise_type == "white":
                        noise = torch.randn_like(signal) * sigma_n
                    elif noise_type == "colored":
                        noise = torch.stack(
                            [_generate_pink_noise(nt).to(device) for _ in range(bs)]
                        ) * sigma_n
                    else:
                        noise = 0
                    x[:, ch, :] = signal + noise
            _, p, _ = model(x)
            preds.append(p.cpu().numpy())
        return np.concatenate(preds, 0)

    # Clean baseline
    pred_clean = predict_with_noise(E_test, A_test, Pr_test, P_test)
    r2_clean = r2_per_param(y_test, pred_clean)
    print(f"\nClean baseline R²: σ1={r2_clean[0]:.3f}  σ2={r2_clean[1]:.3f}"
          f"  d1={r2_clean[2]:.3f}  d2={r2_clean[3]:.3f}")

    # White noise at various SNRs
    rows_white = []
    print(f"\n{'SNR':>6} {'Type':>8}  {'R²_σ1':>7}  {'R²_σ2':>7}  {'R²_d1':>7}  {'R²_d2':>7}  {'R̄²':>7}")
    print("-" * 60)
    for snr in [40, 30, 20, 10, 5]:
        pred = predict_with_noise(
            E_test, A_test, Pr_test, P_test,
            noise_type="white", snr_db=snr)
        r2 = r2_per_param(y_test, pred)
        r2_mean = r2.mean()
        print(f"{snr:>6}  {'white':>8}  {r2[0]:>7.3f}  {r2[1]:>7.3f}"
              f"  {r2[2]:>7.3f}  {r2[3]:>7.3f}  {r2_mean:>7.3f}")
        rows_white.append({
            "snr_db": snr, "noise_type": "white",
            "r2_sigma1": f"{r2[0]:.4f}", "r2_sigma2": f"{r2[1]:.4f}",
            "r2_d1": f"{r2[2]:.4f}", "r2_d2": f"{r2[3]:.4f}",
            "r2_mean": f"{r2_mean:.4f}",
        })

    # Colored noise at various SNRs
    rows_colored = []
    for snr in [40, 30, 20, 10, 5]:
        pred = predict_with_noise(
            E_test, A_test, Pr_test, P_test,
            noise_type="colored", snr_db=snr)
        r2 = r2_per_param(y_test, pred)
        r2_mean = r2.mean()
        print(f"{snr:>6}  {'colored':>8}  {r2[0]:>7.3f}  {r2[1]:>7.3f}"
              f"  {r2[2]:>7.3f}  {r2[3]:>7.3f}  {r2_mean:>7.3f}")
        rows_colored.append({
            "snr_db": snr, "noise_type": "colored",
            "r2_sigma1": f"{r2[0]:.4f}", "r2_sigma2": f"{r2[1]:.4f}",
            "r2_d1": f"{r2[2]:.4f}", "r2_d2": f"{r2[3]:.4f}",
            "r2_mean": f"{r2_mean:.4f}",
        })

    # Save CSV
    os.makedirs(PAPER_DIR, exist_ok=True)
    all_rows = rows_white + rows_colored
    fields = ["snr_db", "noise_type", "r2_sigma1", "r2_sigma2",
              "r2_d1", "r2_d2", "r2_mean"]
    path = os.path.join(PAPER_DIR, "colored_noise_results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Saved → {path}")


# ── AMP-enabled training epoch ────────────────────────────────────────────────

def train_one_epoch_p9d_amp(model, loader, optimizer, criterion, scheduler,
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
        tot      += loss.item() * b
        prof_tot += pl * b
        par_tot  += ql * b

    n = len(loader.dataset)
    return tot / n, prof_tot / n, par_tot / n, float(np.mean(grad_norms))


def run_training_p9d_amp(model, train_loader, val_loader, test_loader,
                         criterion, optimizer, scheduler,
                         model_path, resume_path, n_epochs, device):
    start_epoch = 1
    best_val    = float("inf")
    train_hist  = {"total": [], "profile": [], "param": []}
    val_hist    = {"total": [], "profile": [], "param": []}
    ppar_hist   = []
    scaler      = torch.amp.GradScaler("cuda")

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
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    print(f"\n{'Ep':>5}  {'Tr':>8}  {'Va':>8}  "
          f"{'s1':>6}  {'s2':>6}  {'d1':>6}  {'d2':>6}  "
          f"{'GradN':>7}")
    print("-" * 62)

    for epoch in range(start_epoch, n_epochs + 1):
        tr, tr_p, tr_q, gn = train_one_epoch_p9d_amp(
            model, train_loader, optimizer, criterion, scheduler, device,
            scaler,
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
                "scaler":     scaler.state_dict(),
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

    sd = torch.load(model_path, map_location=device, weights_only=True)
    # Handle torch.compile: strip _orig_mod. prefix if model is not compiled,
    # or add it if model is compiled but keys don't have it
    if hasattr(model, '_orig_mod'):
        # Model is compiled — load into the underlying module
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model._orig_mod.load_state_dict(clean_sd)
    else:
        # Model is not compiled — strip prefix if present
        clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        model.load_state_dict(clean_sd)
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
    print(f"Experiment: DualTCN with colored (1/f) waveform noise augmentation")
    print(f"  α = 1.0 (pink noise spectrum)")
    print(f"  Model save path: {MODEL_PATH}")

    E_multi, log_amps, sigma_profiles, params = load_v4_data()
    train_loader, val_loader, test_loader = get_dataloaders_v4_colored(
        E_multi, log_amps, sigma_profiles, params, alpha=1.0
    )

    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LateTimePCRN (P9d-Colored) parameters: {n_par:,}")

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

    te, te_p, te_q, te_ppar = run_training_p9d_amp(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler,
        MODEL_PATH, RESUME_PATH, N_EPOCHS, device,
    )
    save_training_curve(model, te, CURVE_PATH)

    # Run colored noise robustness evaluation
    evaluate_colored_robustness(MODEL_PATH, device)


if __name__ == "__main__":
    main()
