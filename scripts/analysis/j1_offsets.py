#!/usr/bin/env python3
"""J1-offsets: does a longer receiver spread add information on sigma2 and d2? (R2.2; 2026-11-13).

Same method, prior, seed, data features and noise grid as j1_sensitivity.py (linearised Fisher analysis with the
uniform prior as a Gaussian of variance 1/12), but with extra receivers at longer offsets (default 300, 500, 1000 m,
same depth Z_OBS). Outputs per noise level:
  - r2_estimate for NESTED spreads: the original 20-200 m set, then adding receivers up to 300, 500, 1000 m;
  - Fisher-diagonal share by offset over all receivers (median over models);
  - peak log10 amplitude per offset relative to the 20 m receiver (median and 5th/95th percentiles).
Noise model caveat: noise is RELATIVE (s_E on peak-normalised traces, s_A in decades on log10 peak amplitude), as in
training. It ignores an absolute instrument noise floor, so it is optimistic for weak long-offset signals; the
amplitude decay is reported so this can be judged.
Gate: the 20-200 m subset must reproduce analysis/j1/j1_summary.json r2_ceiling to 1e-3 (same seed and models).
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=2000)
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--batch', type=int, default=100)
ap.add_argument('--extra', type=float, nargs='+', default=[300.0, 500.0, 1000.0])
ap.add_argument('--out', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/j1_offsets')
ap.add_argument('--ref', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/j1/j1_summary.json')
ap.add_argument('--threads', type=int, default=32)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import torch
import forward_torch as ft
from config_v6 import RECEIVER_OFFSETS, Z_OBS, FREQS_V6, N_TIME, SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE
torch.set_num_threads(a.threads); os.makedirs(a.out, exist_ok=True); t0 = time.time()
OFFS = list(map(float, RECEIVER_OFFSETS)) + sorted(a.extra)
R, N0 = len(OFFS), len(RECEIVER_OFFSETS)
LO = np.array([SIGMA1_LOG_RANGE[0], SIGMA2_LOG_RANGE[0], np.log10(D1_RANGE[0]), np.log10(D2_RANGE[0])])
HI = np.array([SIGMA1_LOG_RANGE[1], SIGMA2_LOG_RANGE[1], np.log10(D1_RANGE[1]), np.log10(D2_RANGE[1])])
LO_t, RG_t = torch.tensor(LO), torch.tensor(HI - LO)


def feats(th):                                   # (B,4) normalised -> (B, R*128 + R)
    p = 10 ** (LO_t + th * RG_t)
    En, la = ft.time_features(ft.freq_response(p[:, 0], p[:, 1], p[:, 2], p[:, 3], OFFS, Z_OBS, FREQS_V6), N_TIME)
    return torch.cat([En.reshape(len(th), -1), la], 1)


rng = np.random.default_rng(a.seed)              # identical draw order to j1_sensitivity.py
s1 = rng.uniform(*SIGMA1_LOG_RANGE, a.n); s2 = rng.uniform(*SIGMA2_LOG_RANGE, a.n)
d1 = rng.uniform(*D1_RANGE, a.n); d2 = rng.uniform(*D2_RANGE, a.n)
th = (np.stack([s1, s2, np.log10(d1), np.log10(d2)], 1) - LO) / (HI - LO)
h, nE = 1e-4, R * N_TIME
J = np.zeros((a.n, nE + R, 4)); LA = np.zeros((a.n, R))
with torch.no_grad():
    for i in range(0, a.n, a.batch):
        T = torch.tensor(th[i:i + a.batch], dtype=torch.float64)
        LA[i:i + a.batch] = feats(T)[:, nE:].numpy()
        for k in range(4):
            e = torch.zeros_like(T); e[:, k] = h
            J[i:i + a.batch, :, k] = ((feats(T + e) - feats(T - e)) / (2 * h)).numpy()
print(f'Jacobians {a.n} models x {R} offsets {time.time()-t0:.0f}s', flush=True)


def rows(recs):                                  # feature mask for a set of receiver indices
    m = np.zeros(nE + R, bool)
    for r in recs:
        m[r * N_TIME:(r + 1) * N_TIME] = True; m[nE + r] = True
    return m


names = ['sigma1', 'sigma2', 'd1', 'd2']
nested = {f'20-{int(OFFS[j])}': list(range(j + 1)) for j in range(N0 - 1, R)}
rel = LA - LA[:, :1]
out = dict(n=a.n, seed=a.seed, offsets=OFFS, z_obs=float(Z_OBS), nested_sets=nested, noise='relative (see docstring)',
           peak_log10amp_rel_20m={str(int(o)): dict(median=float(np.median(rel[:, r])), p05=float(np.percentile(rel[:, r], 5)),
                                                    p95=float(np.percentile(rel[:, r], 95))) for r, o in enumerate(OFFS)},
           grid=[])
ref = json.load(open(a.ref)) if os.path.exists(a.ref) else None
gate = []
for sE in (1e-3, 1e-2, 1e-1):
    for sA in (0.002, 0.01, 0.04):
        w = np.r_[np.full(nE, 1 / sE ** 2), np.full(R, 1 / sA ** 2)]
        g = dict(s_E=sE, s_A=sA, r2_estimate={})
        for key, recs in nested.items():
            m = rows(recs)
            F = np.einsum('nik,i,nil->nkl', J[:, m], w[m], J[:, m])
            C = np.linalg.inv(F + 12 * np.eye(4))
            g['r2_estimate'][key] = dict(zip(names, (1 - C.diagonal(0, 1, 2).mean(0) * 12).round(4).tolist()))
        sh = {}
        for r, o in enumerate(OFFS):
            m = rows([r]); sh[str(int(o))] = np.einsum('nik,i,nik->nk', J[:, m], w[m], J[:, m])
        tot = sum(sh.values())
        g['fisher_share_by_offset_all'] = {o: dict(zip(names, np.median(v / tot, 0).round(3).tolist())) for o, v in sh.items()}
        if ref:
            rg = [x for x in ref['grid'] if x['s_E'] == sE and x['s_A'] == sA][0]['r2_ceiling']
            gate.append(max(abs(rg[k] - g['r2_estimate']['20-200'][k]) for k in names))
        out['grid'].append(g)
        print(f"sE={sE:g} sA={sA:g} " + ' | '.join(f"{k}: s2 {v['sigma2']:.3f} d2 {v['d2']:.3f}" for k, v in g['r2_estimate'].items()), flush=True)
out['gate_vs_j1_max_abs_diff'] = max(gate) if gate else None
out['gate_pass'] = bool(gate) and max(gate) < 1e-3
json.dump(out, open(os.path.join(a.out, 'j1_offsets_summary.json'), 'w'), indent=2)
print('peak log10 amp rel 20 m (median):', {k: round(v['median'], 2) for k, v in out['peak_log10amp_rel_20m'].items()})
print('GATE', out['gate_pass'], out['gate_vs_j1_max_abs_diff'], 'DONE', f'{time.time()-t0:.0f}s')
