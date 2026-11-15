#!/usr/bin/env python3
"""v1 BatchNorm check (2026-11-13, EMS revision).

Does the submitted (v1) DualTCN suffer from the same inaccurate BatchNorm running statistics that were found in
train_v6? Evaluates v1 weights (best_model_p9d.pt, sha256 121e79ee...) on v1's own test split (last 15% of
mcsem_dataset_v4.npz, clean traces, R2 on normalised log-parameters, exactly as v1 test_utils.run_test) with
  (a) the stored EMA statistics (what v1 reported),
  (b) statistics recomputed as an exact average over K training batches with v1's training noise
      (10^U(-3,-1) per sample and receiver on the normalised traces), dropout off, K = 100 and 400,
  (c) same as (b) without training noise.
Uses only the v1 code from the recovered archive (Applied.tar.gz).
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--v1', default='/work/hdd/bgte/kahmed2/DualTCN_v1_archive/AppliedMath/mcsem_inversion')
ap.add_argument('--out', default='/work/hdd/bgte/kahmed2/DualTCN_runs/analysis/v1_bn_check')
ap.add_argument('--data', default='')
ap.add_argument('--n-test', type=int, default=0, help='0 = full test split (smoke: small)')
ap.add_argument('--threads', type=int, default=32)
a = ap.parse_args()
sys.path.insert(0, a.v1)                       # v1 modules only (config_v4, model_p9d, dataset_v4)
import torch
from config_p9d import N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS, DROPOUT, TRAIN_SPLIT, VAL_SPLIT
from model_p9d import LateTimePCRN
from config_v4 import LOG_SIGMA1_MIN, LOG_SIGMA1_MAX, LOG_SIGMA2_MIN, LOG_SIGMA2_MAX, LOG_D1_MIN, LOG_D1_MAX, LOG_D2_MIN, LOG_D2_MAX
_BOUNDS_NP = np.array([[LOG_SIGMA1_MIN, LOG_SIGMA1_MAX], [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX], [LOG_D1_MIN, LOG_D1_MAX], [LOG_D2_MIN, LOG_D2_MAX]], dtype=np.float32)  # = dataset_v4._BOUNDS_NP (that module imports empymod)
torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
t0 = time.time()

d = np.load(a.data or os.path.join(a.v1, 'mcsem_dataset_v4.npz'))
E, L, P = d['E_multi'], d['log_amps'], d['params']
n = len(E); n_tr = int(n * TRAIN_SPLIT); n_va = int(n * VAL_SPLIT)
te = np.arange(n_tr + n_va, n)
if a.n_test: te = te[:a.n_test]
lo, hi = _BOUNDS_NP[:, 0], _BOUNDS_NP[:, 1]
y_true = np.clip((np.log10(P[te] + 1e-12) - lo) / (hi - lo + 1e-8), 0, 1)
print(f'loaded n={n} test={len(te)} {time.time()-t0:.0f}s', flush=True)


def make_x(idx, noise, gen=None):
    e = torch.from_numpy(E[idx]).float()
    if noise:
        std = 10 ** (torch.rand(e.shape[0], e.shape[1], 1, generator=gen) * 2 - 3)
        e = e + std * torch.randn(e.shape, generator=gen)
    la = torch.from_numpy(L[idx]).float()[:, :, None].expand_as(e)
    return torch.stack([e, la], 2).reshape(e.shape[0], -1, e.shape[2])   # [E_r0, la_r0, E_r1, ...]


def r2(yt, yp):
    return 1 - ((yt - yp) ** 2).sum(0) / (((yt - yt.mean(0)) ** 2).sum(0) + 1e-12)


@torch.no_grad()
def evaluate(model):
    model.eval(); out = []
    for i in range(0, len(te), 2048):
        out.append(model(make_x(te[i:i + 2048], False))[1].numpy())
    return r2(y_true, np.concatenate(out))


def bn_layers(m):
    return [x for x in m.modules() if isinstance(x, torch.nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def recal(model, k, noise, seed=12345, bs=256):
    g = torch.Generator().manual_seed(seed)
    for b in bn_layers(model):
        b.reset_running_stats(); b.momentum = None      # cumulative (exact) average
    model.eval()
    for b in bn_layers(model): b.train()
    for _ in range(k):
        idx = torch.randint(0, n_tr, (bs,), generator=g).numpy()
        model(make_x(np.sort(idx), noise, g))
    model.eval()


def fresh():
    m = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT)
    m.load_state_dict(torch.load(os.path.join(a.v1, 'best_model_p9d.pt'), map_location='cpu', weights_only=True))
    return m


res = {}
m = fresh(); ema = [(b.running_mean.clone(), b.running_var.clone()) for b in bn_layers(m)]
res['ema_v1_as_reported'] = evaluate(m).tolist()
print('EMA', np.round(res['ema_v1_as_reported'], 4), f'{time.time()-t0:.0f}s', flush=True)
for k, noise in ((100, True), (400, True), (100, False)):
    m = fresh(); recal(m, k, noise)
    key = f'recal_k{k}_{"trainnoise" if noise else "clean"}'
    res[key] = evaluate(m).tolist()
    if k == 100 and noise:   # how far the stored EMA statistics are from the exact ones
        res['ema_vs_recal'] = [dict(layer=i, rel_mean=float((em - b.running_mean).abs().mean() / (b.running_var.sqrt().mean())),
                                    var_ratio_median=float((ev / b.running_var).median()))
                               for i, ((em, ev), b) in enumerate(zip(ema, bn_layers(m)))]
    print(key, np.round(res[key], 4), f'{time.time()-t0:.0f}s', flush=True)
res.update(order=['sigma1', 'sigma2', 'd1', 'd2'], n_test=int(len(te)),
           note='R2 on normalised log10 params, v1 test split (last 15%), clean traces; v1 paper: sigma2 0.898, d2 0.627')
for k in list(res):
    if k.startswith(('ema_v1', 'recal')): res[k + '_mean'] = float(np.mean(res[k]))
json.dump(res, open(os.path.join(a.out, 'v1_bn_check.json' if not a.n_test else 'smoke.json'), 'w'), indent=2)
print('DONE', f'{time.time()-t0:.0f}s')
