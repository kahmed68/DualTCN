#!/usr/bin/env python3
"""N1 + U1 (2026-11-13, EMS revision; R2.4, R3.3, R4.3, R3.7).

N1: noise is added to the RAW transient E(t) = E_norm * 10^logamp of every receiver, BEFORE the peak is extracted
and the trace is normalised, so both input channels (normalised trace and log10 peak amplitude) are affected
consistently. Conditions (test split, 15% in file order, as train_v6):
  clean; white Gaussian noise at SNR 40/30/20/10 dB per trace (std = peak * 10^(-SNR/20));
  receiver gain error 1/2/5% (one multiplicative factor 1+N(0,g) per receiver and sample, i.e. calibration error);
  combined SNR 30 dB + 2% gain; and the original protocol (noise 10^U(-3,-1) on the normalised trace only).
U1: prediction intervals at 68% and 95% for each parameter and condition:
  (a) deep ensemble of the given runs, Gaussian mu +- z*sd;
  (b) MC dropout (first run, T passes, dropout on, BatchNorm in eval), Gaussian;
  (c) split conformal on the ensemble mean, calibrated on CLEAN validation data (what a user would do);
  (d) normalised conformal (score |r|/sd_ens), calibrated on clean validation data;
  (e) split conformal calibrated on validation data with the SAME condition (oracle on the noise level);
  (f) [11-15, D5] split conformal calibrated on validation data perturbed with the TRAINING noise distribution
      (train_v6.raw_perturb: SNR 60-20 dB log-uniform + gain 0.1-5%), i.e. no knowledge of the survey noise;
  (g) [11-15, D5] the same, normalised by the seed spread.
All in normalised log10 parameter space [0,1]. Reports R2 per run (mean +- sd), PICP and mean width (MPIW).
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='base_s1,base_s2,base_s3')
ap.add_argument('--root', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/e1')
ap.add_argument('--arch', default='dualtcn')
ap.add_argument('--out', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/n1_u1')
ap.add_argument('--tag', default='base')
ap.add_argument('--n-cal', type=int, default=50000)
ap.add_argument('--n-test', type=int, default=0, help='0 = full test split')
ap.add_argument('--mc', type=int, default=20)
ap.add_argument('--subset', type=int, default=0, help='smoke: use the first N samples of the file')
ap.add_argument('--seed', type=int, default=777)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import torch
from train_v6 import load, ARCHS, NAMES, raw_perturb
from config_v6 import IN_CHANNELS, N_TIME, N_DEPTH, LATENT_DIM, DROPOUT, TRAIN_SPLIT, VAL_SPLIT, DATA_PATH_V6

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # 11-16: TF32 convs on A40 shifted d2 R2 by up to 0.01
os.makedirs(a.out, exist_ok=True); t0 = time.time()
E, L, S, Pn = load(DATA_PATH_V6, dev)
if a.subset: E, L, Pn = E[:a.subset], L[:a.subset], Pn[:a.subset]
n = len(E); n_tr = int(n * TRAIN_SPLIT); n_va = int(n * VAL_SPLIT)
cal = torch.arange(n_tr, n_tr + min(a.n_cal, n_va), device=dev)
te = torch.arange(n_tr + n_va, n, device=dev)
if a.n_test: te = te[:a.n_test]
print(f'data n={n} cal={len(cal)} test={len(te)} {dev} {time.time()-t0:.0f}s', flush=True)

COND = [('clean', {}), ('snr40', dict(snr=40)), ('snr30', dict(snr=30)), ('snr20', dict(snr=20)), ('snr10', dict(snr=10)),
        ('gain1', dict(gain=0.01)), ('gain2', dict(gain=0.02)), ('gain5', dict(gain=0.05)),
        ('snr30_gain2', dict(snr=30, gain=0.02)), ('orig_protocol', dict(orig=True))]


def perturb(idx, c, g):
    """Return (B,8,T) network input for samples idx under condition c (fixed generator g)."""
    En, la = E[idx], L[idx]
    if c.get('trainmix'):
        En, la = raw_perturb(En, la, g)
        En = En.float(); la = la.float()
    elif c.get('orig'):
        std = 10 ** (torch.rand(En.shape[0], En.shape[1], 1, device=dev, generator=g) * 2 - 3)
        En = En + std * torch.randn(En.shape, device=dev, generator=g)
    elif c:
        raw = En.double() * (10 ** la.double())[..., None]                  # raw transient per receiver
        peak = raw.abs().amax(-1, keepdim=True)
        if 'gain' in c:
            raw = raw * (1 + c['gain'] * torch.randn(raw.shape[0], raw.shape[1], 1, device=dev, generator=g, dtype=raw.dtype))
        if 'snr' in c:
            raw = raw + peak * 10 ** (-c['snr'] / 20) * torch.randn(raw.shape, device=dev, generator=g, dtype=raw.dtype)
        pk = raw.abs().amax(-1)
        la = torch.log10(pk + 1e-40).float()
        En = (raw / (pk[..., None] + 1e-60)).float()
    A = la[..., None].expand_as(En)
    return torch.stack([En, A], 2).reshape(En.shape[0], -1, En.shape[-1])


def build(run):
    m = ARCHS[a.arch](in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT).to(dev)
    m.load_state_dict(torch.load(os.path.join(a.root, run, 'best.pt'), map_location=dev, weights_only=True))
    return m.eval()


runs = a.runs.split(',')
models = [build(r) for r in runs]


@torch.no_grad()
def predict(idx, c, cseed, mc=0):
    """Same perturbation for every model (generator reseeded per condition and split). Returns (M,N,4) [, (T,N,4)]."""
    g = torch.Generator(device=dev); g.manual_seed(cseed)
    outs = [[] for _ in models]; mcs = []
    for i in range(0, len(idx), 4096):
        x = perturb(idx[i:i + 4096], c, g)
        for k, m in enumerate(models):
            outs[k].append(m(x)[1].float())
        if mc:
            m0 = models[0]
            for mod in m0.modules():
                if isinstance(mod, torch.nn.Dropout): mod.train()
            mcs.append(torch.stack([m0(x)[1].float() for _ in range(mc)]))
            m0.eval()
    P = torch.stack([torch.cat(o) for o in outs]).cpu().numpy()
    return (P, torch.cat(mcs, 1).cpu().numpy()) if mc else P


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def cov(y, lo, hi):
    return dict(picp=((y >= lo) & (y <= hi)).mean(0).round(4).tolist(), mpiw=(hi - lo).mean(0).round(4).tolist())


def cq(scores, level):
    nc = len(scores); return np.quantile(scores, min(1.0, np.ceil((nc + 1) * level) / nc), axis=0, method='higher')


Z = {0.68: 0.9945, 0.95: 1.9600}
y_cal = Pn[cal].cpu().numpy(); y_te = Pn[te].cpu().numpy()
Pc = predict(cal, {}, a.seed)                                   # clean calibration predictions
mu_c, sd_c = Pc.mean(0), Pc.std(0, ddof=1) + 1e-6
Pm = predict(cal, {'trainmix': True}, a.seed + 3000)            # calibration with the training noise mixture (D5)
mu_m, sd_m = Pm.mean(0), Pm.std(0, ddof=1) + 1e-6
res = dict(runs=runs, arch=a.arch, n_cal=len(cal), n_test=len(te), mc=a.mc, order=NAMES, space='normalised log10 params',
           conditions={})
for ci, (name, c) in enumerate(COND):
    P, MC = predict(te, c, a.seed + 1000 + ci, mc=a.mc)
    mu, sd = P.mean(0), P.std(0, ddof=1) + 1e-6
    mmc, smc = MC.mean(0), MC.std(0) + 1e-6
    r2s = np.stack([r2(y_te, p) for p in P])
    out = dict(r2_runs=r2s.round(4).tolist(), r2_mean=r2s.mean(0).round(4).tolist(), r2_sd=r2s.std(0, ddof=1).round(4).tolist(),
               r2_ensemble=r2(y_te, mu).round(4).tolist(), r2_mcdropout=r2(y_te, mmc).round(4).tolist(), uq={})
    Pv = predict(cal, c, a.seed + 2000 + ci) if c else Pc         # oracle calibration at the same condition
    for lev in (0.68, 0.95):
        z = Z[lev]; u = {}
        u['ensemble_gauss'] = cov(y_te, mu - z * sd, mu + z * sd)
        u['mcdropout_gauss'] = cov(y_te, mmc - z * smc, mmc + z * smc)
        q = cq(np.abs(y_cal - mu_c), lev); u['conformal_clean'] = cov(y_te, mu - q, mu + q)
        qn = cq(np.abs(y_cal - mu_c) / sd_c, lev); u['conformal_norm_clean'] = cov(y_te, mu - qn * sd, mu + qn * sd)
        qo = cq(np.abs(y_cal - Pv.mean(0)), lev); u['conformal_same_condition'] = cov(y_te, mu - qo, mu + qo)
        qm = cq(np.abs(y_cal - mu_m), lev); u['conformal_trainmix'] = cov(y_te, mu - qm, mu + qm)
        qmn = cq(np.abs(y_cal - mu_m) / sd_m, lev); u['conformal_norm_trainmix'] = cov(y_te, mu - qmn * sd, mu + qmn * sd)
        out['uq'][str(lev)] = u
    res['conditions'][name] = out
    print(f"{name:14s} R2 mean {np.round(out['r2_mean'],3)} | 95% PICP conf_clean {out['uq']['0.95']['conformal_clean']['picp']}"
          f" ens {out['uq']['0.95']['ensemble_gauss']['picp']}  {time.time()-t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(a.out, f'n1_u1_{a.tag}.json' if not a.subset else f'smoke_{a.tag}.json'), 'w'), indent=1)
print('DONE', f'{time.time()-t0:.0f}s')
