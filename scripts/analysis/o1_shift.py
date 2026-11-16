#!/usr/bin/env python3
"""O1: shifted parameter distributions (EMS revision, R2.6; design approved by Khaled 2026-11-13).

Set 'shelf' (inside the training ranges, realistic continental shelf):
  sigma1 log-uniform 3-5 S/m, sigma2 log-uniform 0.3-1 S/m, (d1, d2) from their training ranges with a Gaussian
  copula, rho = 0.5, uniform marginals (training assumes independent parameters).
Set 'extrap' (outside): every normalised log10 parameter uniform on [-w, 1+w] (w = 0.2), keep only samples with at
  least one parameter outside [0, 1]. Min d1 = 39.6 m > Z_OBS (forward_torch requirement).
Reference 'ref': the first --n-ref samples of the standard v6 test split (clean), as in train_v6.
Features: identical pipeline to generate_torch_dataset.py (forward_torch.freq_response + time_features, float32).
Metrics per run: R2 per parameter (normalised log10 space; extrap targets lie outside [0,1]), data misfit
(train_v6.misfit_terms: trace MSE, log-amp MSE) of the predictions.
Misfit-warning check (NOT REQUESTED BY REVIEWERS; kept by Khaled 11-13 because the R2.8 Discussion claims the misfit
is a quality measure): AUC of the per-sample misfit (trace MSE + log-amp MSE) for extrap vs ref and shelf vs ref
(Mann-Whitney), and Spearman correlation between misfit and parameter error (Euclidean, normalised space).
"""
import argparse, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs', default='mis_s1,mis_s2,mis_s3')
ap.add_argument('--root', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/e1')
ap.add_argument('--arch', default='dualtcn')
ap.add_argument('--n', type=int, default=10000)
ap.add_argument('--n-ref', type=int, default=10000)
ap.add_argument('--widen', type=float, default=0.2)
ap.add_argument('--rho', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=611)
ap.add_argument('--batch', type=int, default=500)
ap.add_argument('--out', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/o1')
ap.add_argument('--tag', default='mis')
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import torch
from scipy.stats import norm, spearmanr
import forward_torch as ft
from train_v6 import load, ARCHS, NAMES, BOUNDS, misfit_terms, make_x
from config_v6 import (RECEIVER_OFFSETS, Z_OBS, FREQS_V6, N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS, DROPOUT,
                       TRAIN_SPLIT, VAL_SPLIT, DATA_PATH_V6, D1_RANGE, D2_RANGE)

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False   # 11-16: TF32 convs on A40 shifted d2 R2 by up to 0.01
os.makedirs(a.out, exist_ok=True); t0 = time.time()
LO, HI = BOUNDS[:, 0].double().numpy(), BOUNDS[:, 1].double().numpy()
rng = np.random.default_rng(a.seed)


def feats(logp):                                    # (n,4) log10 physical -> E (n,R,T) float32, L (n,R) float32
    E, L = [], []
    with torch.no_grad():
        for i in range(0, len(logp), a.batch):
            p = torch.tensor(10 ** logp[i:i + a.batch], dtype=torch.float64, device=dev)
            En, la = ft.time_features(ft.freq_response(p[:, 0], p[:, 1], p[:, 2], p[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6), N_TIME)
            E.append(En.float()); L.append(la.float())
    return torch.cat(E), torch.cat(L)


def shelf(n):
    s1 = rng.uniform(np.log10(3.0), np.log10(5.0), n); s2 = rng.uniform(np.log10(0.3), 0.0, n)
    z = rng.multivariate_normal([0, 0], [[1, a.rho], [a.rho, 1]], n); u = norm.cdf(z)
    d1 = D1_RANGE[0] + u[:, 0] * (D1_RANGE[1] - D1_RANGE[0]); d2 = D2_RANGE[0] + u[:, 1] * (D2_RANGE[1] - D2_RANGE[0])
    return np.stack([s1, s2, np.log10(d1), np.log10(d2)], 1)


def extrap(n):
    out = []
    while sum(len(x) for x in out) < n:
        th = rng.uniform(-a.widen, 1 + a.widen, (4 * n, 4))
        out.append(th[np.any((th < 0) | (th > 1), 1)])
    th = np.concatenate(out)[:n]
    return LO + th * (HI - LO)


def build(run):
    m = ARCHS[a.arch](in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT).to(dev)
    m.load_state_dict(torch.load(os.path.join(a.root, run, 'best.pt'), map_location=dev, weights_only=True))
    return m.eval()


@torch.no_grad()
def evaluate(m, E, L):
    P, tr, am = [], [], []
    for i in range(0, len(E), a.batch):
        x, _ = make_x(E[i:i + a.batch], L[i:i + a.batch], False)
        pn = m(x)[1].float()
        pp = (BOUNDS[:, 0].to(dev) + pn * (BOUNDS[:, 1] - BOUNDS[:, 0]).to(dev)).float()
        t, s = misfit_terms(pp, E[i:i + a.batch], L[i:i + a.batch])
        P.append(pn.cpu()); tr.append(t.cpu()); am.append(s.cpu())
    return torch.cat(P).double().numpy(), torch.cat(tr).double().numpy(), torch.cat(am).double().numpy()


def r2(y, p):
    return (1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).round(4).tolist()


def auc(pos, neg):                                  # P(score_pos > score_neg), Mann-Whitney with ties
    s = np.concatenate([pos, neg]); r = np.argsort(np.argsort(s, kind='mergesort'), kind='mergesort') + 1.0
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); avg = csum - (cnt - 1) / 2.0; r = avg[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# data
Eall, Lall, _, Pn = load(DATA_PATH_V6, dev)
n = len(Eall); te0 = int(n * TRAIN_SPLIT) + int(n * VAL_SPLIT)
ref_idx = torch.arange(te0, min(te0 + a.n_ref, n), device=dev)
sets = {'ref': (Eall[ref_idx], Lall[ref_idx], Pn[ref_idx].double().cpu().numpy())}
del Eall, Lall
for name, gen in (('shelf', shelf), ('extrap', extrap)):
    lp = gen(a.n); E, L = feats(lp)
    sets[name] = (E, L, (lp - LO) / (HI - LO))
    print(f'{name}: n={len(lp)} theta range {np.round(sets[name][2].min(0), 3)} .. {np.round(sets[name][2].max(0), 3)} '
          f'{time.time()-t0:.0f}s', flush=True)

res = dict(runs=a.runs.split(','), arch=a.arch, n=a.n, n_ref=len(ref_idx), widen=a.widen, rho=a.rho, seed=a.seed,
           order=NAMES, space='normalised log10 params', per_run={})
for run in res['runs']:
    m = build(run); out = {}
    for name, (E, L, y) in sets.items():
        P, tr, am = evaluate(m, E, L)
        err = np.sqrt(((P - y) ** 2).sum(1)); mis = tr + am
        rho, _ = spearmanr(mis, err)
        out[name] = dict(r2=r2(y, P), rmse=np.sqrt(((P - y) ** 2).mean(0)).round(4).tolist(), trace_rmse=float(np.sqrt(tr.mean())), logamp_rmse=float(np.sqrt(am.mean())),
                         misfit_median=float(np.median(mis)), spearman_misfit_vs_error=float(rho), _mis=mis, _err=err)
    for name in ('shelf', 'extrap'):
        out[name]['auc_vs_ref'] = auc(out[name]['_mis'], out['ref']['_mis'])
    allm = np.concatenate([out[k]['_mis'] for k in sets]); alle = np.concatenate([out[k]['_err'] for k in sets])
    out['spearman_misfit_vs_error_pooled'] = float(spearmanr(allm, alle)[0])
    for k in sets:
        del out[k]['_mis'], out[k]['_err']
    res['per_run'][run] = out
    print(run, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()}) for k, v in out.items()}), flush=True)
json.dump(res, open(os.path.join(a.out, f'o1_{a.tag}.json'), 'w'), indent=1)
print('DONE', f'{time.time()-t0:.0f}s', flush=True)
