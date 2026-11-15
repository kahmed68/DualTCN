#!/usr/bin/env python3
"""J1: Jacobian sensitivity and linearised identifiability of (sigma1, sigma2, d1, d2) (R2.5, R2.2; 2026-11-13).

Data vector per model = the DualTCN input features of the v6 operator: normalised traces E_norm (4 receivers x 128)
and log10 peak amplitudes (4). Parameters theta = the network's targets, normalised log10 params in [0,1]^4.
J = d data / d theta by central differences (h=1e-4) through forward_torch (complex128); checked against autograd
on a subset. For noise std s_E on E_norm and s_A on log_amp (dex), W = diag(1/s^2):
  - singular values of W^1/2 J, condition number;
  - linearised (Gauss-Newton) posterior covariance C = (J^T W J + 12 I)^-1 with the uniform prior's variance 1/12
    as a Gaussian prior; ceiling R2_k = 1 - mean(C_kk) / (1/12)  (a rough, linear, Gaussian approximation);
  - posterior correlations (trade-offs), and Fisher information split by receiver offset (R2.2).
Noise grid: s_E in {1e-3, 1e-2, 1e-1} (training range 10^U(-3,-1)), s_A in {0.002, 0.01, 0.04} dex (training adds
none to log_amp; the clean test set has none either).
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--batch', type=int, default=100)
ap.add_argument('--n-check', type=int, default=10)
ap.add_argument('--out', default='/work/hdd/bgte/kahmed2/DualTCN_runs/analysis/j1')
ap.add_argument('--threads', type=int, default=32)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import torch
import forward_torch as ft
from config_v6 import RECEIVER_OFFSETS, Z_OBS, FREQS_V6, N_TIME, SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE
torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True); t0 = time.time()
LO = np.array([SIGMA1_LOG_RANGE[0], SIGMA2_LOG_RANGE[0], np.log10(D1_RANGE[0]), np.log10(D2_RANGE[0])])
HI = np.array([SIGMA1_LOG_RANGE[1], SIGMA2_LOG_RANGE[1], np.log10(D1_RANGE[1]), np.log10(D2_RANGE[1])])
LO_t, RG_t = torch.tensor(LO), torch.tensor(HI - LO)
R = len(RECEIVER_OFFSETS)


def feats(th):                                   # th (B,4) normalised -> (B, R*128 + R)
    p = 10 ** (LO_t + th * RG_t)
    En, la = ft.time_features(ft.freq_response(p[:, 0], p[:, 1], p[:, 2], p[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6), N_TIME)
    return torch.cat([En.reshape(len(th), -1), la], 1)


rng = np.random.default_rng(a.seed)              # prior: same as the data generator (log-uniform sigma, uniform d)
s1 = rng.uniform(*SIGMA1_LOG_RANGE, a.n); s2 = rng.uniform(*SIGMA2_LOG_RANGE, a.n)
d1 = rng.uniform(*D1_RANGE, a.n); d2 = rng.uniform(*D2_RANGE, a.n)
th = (np.stack([s1, s2, np.log10(d1), np.log10(d2)], 1) - LO) / (HI - LO)
h = 1e-4
J = np.zeros((a.n, R * N_TIME + R, 4))
with torch.no_grad():
    for i in range(0, a.n, a.batch):
        T = torch.tensor(th[i:i + a.batch], dtype=torch.float64)
        for k in range(4):
            e = torch.zeros_like(T); e[:, k] = h
            J[i:i + a.batch, :, k] = ((feats(T + e) - feats(T - e)) / (2 * h)).numpy()
print(f'Jacobians {a.n} models {time.time()-t0:.0f}s', flush=True)

chk = []                                         # autograd cross-check
for i in range(min(a.n_check, a.n)):
    Ja = torch.autograd.functional.jacobian(lambda t: feats(t[None])[0], torch.tensor(th[i], dtype=torch.float64),
                                            vectorize=True).numpy()
    chk.append(float(np.abs(Ja - J[i]).max() / (np.abs(Ja).max() + 1e-30)))
print('autograd check max rel err', max(chk) if chk else None, flush=True)

nE = R * N_TIME
names = ['sigma1', 'sigma2', 'd1', 'd2']
out = dict(n=a.n, seed=a.seed, h=h, autograd_rel_err_max=max(chk) if chk else None, offsets=list(map(float, RECEIVER_OFFSETS)),
           theta='normalised log10 params in [0,1]', grid=[])
for sE in (1e-3, 1e-2, 1e-1):
    for sA in (0.002, 0.01, 0.04):
        w = np.r_[np.full(nE, 1 / sE ** 2), np.full(R, 1 / sA ** 2)]
        F = np.einsum('nik,i,nil->nkl', J, w, J)                       # Fisher (n,4,4)
        C = np.linalg.inv(F + 12 * np.eye(4))
        sv = np.sqrt(np.clip(np.linalg.eigvalsh(F), 0, None))[:, ::-1]  # singular values of W^1/2 J
        sd = np.sqrt(np.einsum('nkk->nk', C))
        corr = C / (sd[:, :, None] * sd[:, None, :])
        # Fisher diagonal per receiver offset (traces + its log_amp), share of the total
        share = {}
        for r in range(R):
            m = np.zeros(nE + R, bool); m[r * N_TIME:(r + 1) * N_TIME] = True; m[nE + r] = True
            Fr = np.einsum('nik,i,nik->nk', J[:, m], w[m], J[:, m])
            share[str(int(RECEIVER_OFFSETS[r]))] = Fr
        tot = sum(share.values())
        out['grid'].append(dict(
            s_E=sE, s_A=sA,
            r2_ceiling=dict(zip(names, (1 - C.diagonal(0, 1, 2).mean(0) * 12).round(4).tolist())),
            post_sd_median=dict(zip(names, np.median(sd, 0).round(4).tolist())),
            singular_values_median=np.median(sv, 0).round(3).tolist(),
            condition_median=float(np.median(sv[:, 0] / np.maximum(sv[:, 3], 1e-30))),
            corr_median={f'{names[i]}-{names[j]}': float(np.median(corr[:, i, j]).round(3)) for i in range(4) for j in range(i + 1, 4)},
            fisher_share_by_offset={o: dict(zip(names, np.median(v / tot, 0).round(3).tolist())) for o, v in share.items()},
            trace_vs_amp_share=dict(zip(names, np.median(np.einsum('nik,i,nik->nk', J[:, :nE], w[:nE], J[:, :nE]) / np.einsum('nkk->nk', F), 0).round(3).tolist()))))
        g = out['grid'][-1]; print(f"sE={sE:g} sA={sA:g} R2ceil {g['r2_ceiling']} corr s2-d2 {g['corr_median']['sigma2-d2']}", flush=True)
np.savez_compressed(os.path.join(a.out, 'jacobians.npz'), theta=th, J=J.astype(np.float32))
json.dump(out, open(os.path.join(a.out, 'j1_summary.json'), 'w'), indent=2)
print('DONE', f'{time.time()-t0:.0f}s')
