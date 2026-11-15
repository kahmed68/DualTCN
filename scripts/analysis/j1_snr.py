#!/usr/bin/env python3
"""J1 at the N1 SNR levels (R2.4, R2.5; Khaled 2026-11-14). Reuses the J1 Jacobians (analysis/j1/jacobians.npz,
2000 prior models, seed 2026) and maps each SNR to (s_E, s_A) EMPIRICALLY with the exact N1 perturbation
(white noise std = peak*10^(-SNR/20) on the raw transient, then peak re-extracted and the trace re-normalised):
s_E = rms change of E_norm, s_A = rms change of log10 peak (dex), over the same 2000 models x 50 noise draws.
Linearised Gaussian ceiling as in J1: C = (J^T W J + 12 I)^-1, R2_k = 1 - 12 mean(C_kk). Also reports the
bias of log_amp (max of noisy trace is biased upward), which the linear ceiling ignores."""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import torch
import forward_torch as ft
from config_v6 import RECEIVER_OFFSETS, Z_OBS, FREQS_V6, N_TIME, SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE
torch.set_num_threads(8)
O = '/work/hdd/bgte/kahmed2/DualTCN_runs/analysis/j1'
z = np.load(os.path.join(O, 'jacobians.npz')); th = z['theta']; J = z['J'].astype(np.float64)
LO = np.array([SIGMA1_LOG_RANGE[0], SIGMA2_LOG_RANGE[0], np.log10(D1_RANGE[0]), np.log10(D2_RANGE[0])])
HI = np.array([SIGMA1_LOG_RANGE[1], SIGMA2_LOG_RANGE[1], np.log10(D1_RANGE[1]), np.log10(D2_RANGE[1])])
p = torch.tensor(10 ** (LO + th * (HI - LO)))
with torch.no_grad():
    Ef = ft.freq_response(p[:, 0], p[:, 1], p[:, 2], p[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6)
    En, la = ft.time_features(Ef, N_TIME)
raw = En * (10 ** la)[..., None]; peak = raw.abs().amax(-1, keepdim=True)
g = torch.Generator().manual_seed(777); R = len(RECEIVER_OFFSETS); nE = R * N_TIME
names = ['sigma1', 'sigma2', 'd1', 'd2']; out = dict(source='j1/jacobians.npz', n=len(th), draws=50, levels=[])
for snr in (40, 30, 20, 10):
    dE, dA = [], []
    for _ in range(50):
        r2 = raw + peak * 10 ** (-snr / 20) * torch.randn(raw.shape, generator=g, dtype=raw.dtype)
        pk = r2.abs().amax(-1); dE.append((r2 / pk[..., None] - En).numpy()); dA.append((torch.log10(pk) - la).numpy())
    dE = np.stack(dE); dA = np.stack(dA)
    sE = float(np.sqrt((dE ** 2).mean())); sA = float(np.sqrt(((dA - dA.mean()) ** 2).mean())); bA = float(dA.mean())
    w = np.r_[np.full(nE, 1 / sE ** 2), np.full(R, 1 / sA ** 2)]
    F = np.einsum('nik,i,nil->nkl', J, w, J); C = np.linalg.inv(F + 12 * np.eye(4))
    rec = dict(snr_db=snr, s_E=round(sE, 5), s_A=round(sA, 5), logamp_bias_dex=round(bA, 5),
               r2_ceiling=dict(zip(names, (1 - C.diagonal(0, 1, 2).mean(0) * 12).round(4).tolist())))
    out['levels'].append(rec); print(rec, flush=True)
json.dump(out, open(os.path.join(O, 'j1_snr.json'), 'w'), indent=1); print('DONE')
