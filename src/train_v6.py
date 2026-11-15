"""
train_v6.py -- DualTCN (LateTimePCRN) on the v6 dataset, with an optional
EM forward-consistency (data-misfit) loss through forward_torch.

Arms
  baseline : --misfit-w 0    (same losses as train_p9d.py)
  misfit   : --misfit-w L    adds L * ramp(epoch) * misfit, where
             misfit = MSE(E_norm(p_hat), E_obs) + amp_w * MSE(log_amp(p_hat), log_amp_obs)
             p_hat = predicted physical parameters, forward = forward_torch (float64).

Same as train_p9d.py: model, profile MSE + weighted Huber param loss, d_sf
auxiliary Huber (w 0.5), AdamW(lr 5e-4, wd 1e-3), OneCycle (5 warm-up epochs),
grad clip 1.0, batch 256, 70/15/15 split in file order, training noise
std = 10**U(-3,-1) per sample and receiver on the waveform channels.
Different: batches are assembled on the GPU; checkpoint selection uses the
SUPERVISED validation loss in both arms (so the misfit term cannot select itself).

Outputs in --out: best.pt, last.pt (resume), log.csv, test_metrics.json
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config_v6 import (N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS, DROPOUT, LR,
                       WARMUP_EPOCHS, PROFILE_WEIGHT, PARAM_WEIGHT, PARAM_WEIGHTS,
                       BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, RECEIVER_OFFSETS, Z_OBS,
                       FREQS_V6, DATA_PATH_V6,
                       LOG_SIGMA1_MIN, LOG_SIGMA1_MAX, LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
                       LOG_D1_MIN, LOG_D1_MAX, LOG_D2_MIN, LOG_D2_MAX)
from config_p9d import DSF_AUX_WEIGHT
from model_p9d import LateTimePCRN
from model_baseline import PCRN_V4
from model_p5a import TCNOnlyPCRN
from model_ablation import NoLatePCRN

ARCHS = {"dualtcn": LateTimePCRN, "dualtcn_noaux": LateTimePCRN, "dualtcn_nolate": NoLatePCRN,
         "pcrn": PCRN_V4, "tcnonly": TCNOnlyPCRN}
from train_utils import WeightedPCRNLoss
import forward_torch as ft

BOUNDS = torch.tensor([[LOG_SIGMA1_MIN, LOG_SIGMA1_MAX], [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
                       [LOG_D1_MIN, LOG_D1_MAX], [LOG_D2_MIN, LOG_D2_MAX]], dtype=torch.float32)
NAMES = ["s1", "s2", "d1", "d2"]


def load(path, device):
    d = np.load(path)
    E = torch.from_numpy(d["E_multi"]).to(device)
    L = torch.from_numpy(d["log_amps"]).to(device)
    S = torch.from_numpy(d["sigma_profiles"]).to(device)
    P = torch.from_numpy(d["params"]).double()
    raw = torch.log10(P + 1e-12).float()
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    Pn = ((raw - lo) / (hi - lo + 1e-8)).clamp(0, 1).to(device)
    return E, L, S, Pn


def make_x(E, L, noise):
    """(B,4,T),(B,4) -> (B,8,T) interleaved [E_r, logamp_r]; optional noise."""
    if noise:
        std = 10 ** torch.empty(E.shape[0], E.shape[1], 1, device=E.device).uniform_(-3, -1)
        E = E + std * torch.randn_like(E)
    A = L[..., None].expand_as(E)
    return torch.stack([E, A], 2).reshape(E.shape[0], -1, E.shape[-1]), E


RAW_NOISE = False   # set by --raw-noise (N1, 2026-11-13)


def raw_perturb(E, L, g=None):
    """Noise on the RAW transient before peak extraction (R2.4/R4.3): raw = E_norm*10^L; per-receiver gain error
    1+N(0,s_g), s_g = 10^U(-3,-1.3) (0.1-5%); white noise std = peak*10^U(-3,-1) (SNR 60-20 dB); then the peak,
    log10 amplitude and normalised trace are recomputed. Returns (E_norm', L')."""
    B, R, T = E.shape
    kw = dict(device=E.device, generator=g)
    raw = E * (10 ** L)[..., None]
    peak = raw.abs().amax(-1, keepdim=True)
    sg = 10 ** (torch.rand(B, R, 1, **kw) * 1.7 - 3)
    raw = raw * (1 + sg * torch.randn(B, R, 1, **kw))
    sn = 10 ** (torch.rand(B, R, 1, **kw) * 2 - 3)
    raw = raw + peak * sn * torch.randn(B, R, T, **kw)
    pk = raw.abs().amax(-1)
    return raw / (pk[..., None] + 1e-60), torch.log10(pk + 1e-40)


def misfit_terms(p_phys, E_obs, L_obs):
    """p_phys: (B,4) log10 physical params (float32). Returns per-sample trace MSE, logamp MSE."""
    q = (10 ** p_phys.double())
    En, la = ft.time_features(
        ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6),
        N_TIME)
    tr = ((En - E_obs.double()) ** 2).mean(dim=(1, 2))
    am = ((la - L_obs.double()) ** 2).mean(dim=1)
    return tr.float(), am.float()


@torch.no_grad()
def recalibrate_bn(model, E, L, tr_idx, n_batches=100, bs=256, seed=12345):
    """Recompute BatchNorm running statistics as an exact average over n_batches training batches
    (with the training noise), dropout off. The EMA running statistics (momentum 0.1) were found to be
    far off the batch statistics (11-13: val R2 d2 -1.57 with running stats vs 0.67 with batch stats on the
    same weights). Uses its own generator, so the training random stream is not touched; weights unchanged."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    g = torch.Generator(device=E.device)
    g.manual_seed(seed)
    mom = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                      # cumulative average
    was_training = model.training
    model.eval()
    for m in bns:
        m.train()
    perm = tr_idx[torch.randperm(len(tr_idx), device=E.device, generator=g)[:n_batches * bs]]
    for i in range(0, len(perm), bs):
        b = perm[i:i + bs]
        e, l = E[b], L[b]
        if RAW_NOISE:
            e, l = raw_perturb(e, l, g)
        else:
            std = 10 ** (torch.rand(e.shape[0], e.shape[1], 1, device=e.device, generator=g) * 2 - 3)
            e = e + std * torch.randn(e.shape, device=e.device, generator=g)
        x, _ = make_x(e, l, noise=False)
        model(x)
    for m, mo in zip(bns, mom):
        m.momentum = mo
    model.train(was_training)


@torch.no_grad()
def evaluate(model, crit, E, L, S, Pn, idx, bs, with_misfit, dsf_w):
    model.eval()
    tot = n = 0
    preds, trues, trs, ams = [], [], [], []
    huber = nn.HuberLoss(delta=0.1)
    for i0 in range(0, len(idx), bs):
        b = idx[i0:i0 + bs]
        x, Eo = make_x(E[b], L[b], noise=False)
        prof, pn, pp = model(x)
        loss, _, _ = crit(prof, S[b], pn, Pn[b])
        if dsf_w > 0:
            d = 10 ** (BOUNDS[2:, 0].to(x.device) + Pn[b][:, 2:] * (BOUNDS[2:, 1] - BOUNDS[2:, 0]).to(x.device))
            loss = loss + dsf_w * huber(model.dsf_pred, model.norm_dsf(d.sum(1)).unsqueeze(1))
        tot += loss.item() * len(b)
        n += len(b)
        preds.append(pn.float().cpu())
        trues.append(Pn[b].cpu())
        if with_misfit:
            tr, am = misfit_terms(pp, Eo, L[b])
            trs.append(tr.cpu())
            ams.append(am.cpu())
    p, t = torch.cat(preds), torch.cat(trues)
    r2 = 1 - ((p - t) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
    out = dict(sup_loss=tot / n, **{f"r2_{k}": float(v) for k, v in zip(NAMES, r2)},
               r2_mean=float(r2.mean()))
    if with_misfit:
        out["trace_rmse"] = float(torch.cat(trs).mean().sqrt())
        out["logamp_rmse"] = float(torch.cat(ams).mean().sqrt())
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_PATH_V6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--misfit-w", type=float, default=0.0)
    ap.add_argument("--amp-w", type=float, default=1.0)
    ap.add_argument("--misfit-start", type=int, default=5, help="epoch where ramp starts")
    ap.add_argument("--misfit-ramp", type=int, default=10, help="epochs to reach full weight")
    ap.add_argument("--val-misfit-n", type=int, default=20000)
    ap.add_argument("--misfit-sub", type=int, default=0,
                    help="misfit on the first K samples of each (randomly permuted) batch; 0 = all")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: steps per epoch cap")
    ap.add_argument("--subset", type=int, default=0, help="smoke: use first N samples")
    ap.add_argument("--arch", default="dualtcn", choices=sorted(ARCHS),
                    help="E1c ranking check; dualtcn = E1 model")
    ap.add_argument("--profile-w", type=float, default=PROFILE_WEIGHT,
                    help="E1b decoder ablation: 0 drops the profile (decoder) loss")
    ap.add_argument("--no-bn-recal", action="store_true",
                    help="evaluate with the EMA BatchNorm statistics (pre-11-13 behaviour)")
    ap.add_argument("--raw-noise", action="store_true",
                    help="N1: training noise on the raw transient before normalisation (gain + white noise)")
    ap.add_argument("--time-limit", type=float, default=0, help="seconds; save+exit before")
    a = ap.parse_args()
    global RAW_NOISE
    RAW_NOISE = a.raw_noise
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t_start = time.time()

    E, L, S, Pn = load(a.data, dev)
    if a.subset:
        E, L, S, Pn = E[:a.subset], L[:a.subset], S[:a.subset], Pn[:a.subset]
    n = len(E)
    n_tr, n_va = int(n * TRAIN_SPLIT), int(n * VAL_SPLIT)
    tr_idx = torch.arange(0, n_tr, device=dev)
    va_idx = torch.arange(n_tr, n_tr + n_va, device=dev)
    te_idx = torch.arange(n_tr + n_va, n, device=dev)
    steps = math.ceil(n_tr / BATCH_SIZE)
    if a.max_steps:
        steps = min(steps, a.max_steps)

    dsf_w = DSF_AUX_WEIGHT if a.arch in ("dualtcn", "dualtcn_nolate") else 0.0
    model = ARCHS[a.arch](in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH,
                         latent_dim=LATENT_DIM, dropout=DROPOUT).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=a.epochs,
                                              steps_per_epoch=steps,
                                              pct_start=min(WARMUP_EPOCHS / a.epochs, 0.3),
                                              anneal_strategy="cos", final_div_factor=100)
    crit = WeightedPCRNLoss(profile_w=a.profile_w, param_w=PARAM_WEIGHT,
                            param_weights=PARAM_WEIGHTS)
    huber = nn.HuberLoss(delta=0.1)
    g = torch.Generator(device=dev)
    g.manual_seed(a.seed)

    start, best = 1, float("inf")
    last_path = os.path.join(a.out, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)  # RNG states must stay CPU ByteTensors; model/opt load_state_dict move tensors to dev
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"]); g.set_state(ck["gen"])
        torch.set_rng_state(ck["cpu_rng"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start, best = ck["epoch"] + 1, ck["best"]
        if not a.no_bn_recal and not ck.get("bn_recal", False):
            best = float("inf")   # earlier val losses used EMA BN statistics; not comparable
        print(f"resumed at epoch {start}, best {best:.5f}", flush=True)
    with open(os.path.join(a.out, "args.json"), "w") as fh:
        json.dump(vars(a), fh, indent=2)

    log_path = os.path.join(a.out, "log.csv")
    for ep in range(start, a.epochs + 1):
        t0 = time.time()
        ramp = 0.0
        if a.misfit_w > 0 and ep >= a.misfit_start:
            ramp = min(1.0, (ep - a.misfit_start + 1) / max(a.misfit_ramp, 1))
        perm = tr_idx[torch.randperm(n_tr, device=dev, generator=g)]
        sums = dict(loss=0.0, sup=0.0, tr=0.0, am=0.0)
        for s in range(steps):
            b = perm[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
            if RAW_NOISE:
                e_b, Lo = raw_perturb(E[b], L[b])
                x, Eo = make_x(e_b, Lo, noise=False)
            else:
                x, Eo = make_x(E[b], L[b], noise=True)
                Lo = L[b]
            opt.zero_grad(set_to_none=True)
            prof, pn, pp = model(x)
            loss, _, _ = crit(prof, S[b], pn, Pn[b])
            if dsf_w > 0:
                d = 10 ** (BOUNDS[2:, 0].to(dev) + Pn[b][:, 2:] * (BOUNDS[2:, 1] - BOUNDS[2:, 0]).to(dev))
                loss = loss + dsf_w * huber(model.dsf_pred, model.norm_dsf(d.sum(1)).unsqueeze(1))
            sums["sup"] += loss.item()
            if ramp > 0:
                k = a.misfit_sub if a.misfit_sub > 0 else len(b)
                tr, am = misfit_terms(pp[:k], Eo[:k], Lo[:k])
                mis = tr.mean() + a.amp_w * am.mean()
                loss = loss + a.misfit_w * ramp * mis
                sums["tr"] += tr.mean().item(); sums["am"] += am.mean().item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            sums["loss"] += loss.item()
        tr_time = time.time() - t0
        if not a.no_bn_recal:
            recalibrate_bn(model, E, L, tr_idx)
        vm = va_idx[:a.val_misfit_n] if a.val_misfit_n else va_idx
        val = evaluate(model, crit, E, L, S, Pn, va_idx, 2048, False, dsf_w)
        valm = evaluate(model, crit, E, L, S, Pn, vm, 512, True, dsf_w)
        row = dict(epoch=ep, ramp=ramp, train_loss=sums["loss"] / steps,
                   train_sup=sums["sup"] / steps, train_trace_mse=sums["tr"] / steps,
                   train_amp_mse=sums["am"] / steps, **{f"val_{k}": v for k, v in val.items()},
                   val_trace_rmse=valm["trace_rmse"], val_logamp_rmse=valm["logamp_rmse"],
                   sec_train=round(tr_time, 1), sec_total=round(time.time() - t0, 1))
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if val["sup_loss"] < best:
            best = val["sup_loss"]
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(),
                        gen=g.get_state(), cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
                        epoch=ep, best=best, bn_recal=not a.no_bn_recal), last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)  # atomic: a preemption mid-save cannot corrupt last.pt
        ep_time = time.time() - t0
        if a.time_limit and (time.time() - t_start) + 1.3 * ep_time > a.time_limit and ep < a.epochs:
            print(f"time limit: stopping after epoch {ep}", flush=True)
            return

    model.load_state_dict(torch.load(os.path.join(a.out, "best.pt"), map_location=dev, weights_only=True))
    test = evaluate(model, crit, E, L, S, Pn, te_idx, 512, True, dsf_w)
    with open(os.path.join(a.out, "test_metrics.json"), "w") as fh:
        json.dump(dict(test=test, n_test=len(te_idx), args=vars(a)), fh, indent=2)
    with open(os.path.join(a.out, "DONE"), "w") as fh:
        fh.write("done\n")
    print("TEST", json.dumps(test), flush=True)


if __name__ == "__main__":
    main()
