#!/usr/bin/env python3
"""G3: validate forward_al611 (torch) against empymod.bipole on the AL611 Profile 8 geometry.
Reports max relative error per model for (a) exact DLF vs empymod standard DLF (both transforms pts_per_dec=0),
(b) interp mode vs (a), (c) empymod default (lagged Fourier DLF) vs (a); plus a finite-difference gradient check
and timings. Writes JSON to --out."""
import argparse, json, os, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=8); ap.add_argument('--out', required=True)
ap.add_argument('--emdata', default='/work/hdd/bgte/kahmed2/field_data/AL611/raw/AL611_CSEM_Profile8.emdata')
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); src = os.path.join(here, '..', '..', 'src')
sys.path[:0] = [src, os.path.join(src, 'field_al611'), os.path.join(here, '..', '..', '.pydeps')]
import torch, empymod
from read_emdata import read_emdata
from forward_al611 import stepoff
torch.set_num_threads(8)
e = read_emdata(a.emdata); t = e['times']; tx = e['tx']; rxd = e['rx']; d = e['data']
itx = 0; L = float(tx['length'][itx]); zs = float(tx['z'][itx])
rx = []
for irx in sorted(set(d['irx'][d['itx'] == itx])):
    rx.append((float(abs(rxd['y'][irx] - tx['y'][itx])), float(rxd['z'][irx])))
print('geometry L', L, 'zs', zs, 'rx', rx, 'gates', len(t), t.min(), t.max(), flush=True)

rng = np.random.default_rng(11)
models = [dict(sw=3.33, hw=14.0, res=[0.74, 1.39, 3.68], thk=[39.2, 150.0])]       # near the G2 fit
for i in range(a.n - 1):
    K = int(rng.integers(1, 5))
    models.append(dict(sw=float(rng.uniform(2.5, 5)), hw=float(rng.uniform(8, 25)),
                       res=list(10 ** rng.uniform(-0.5, 2, K)), thk=list(rng.uniform(5, 150, K - 1))))

XG, WG = np.polynomial.legendre.leggauss(11)
def emp_sum(m):
    """Reference: Gauss-Legendre sum of empymod point dipoles (standard DLF for both transforms)."""
    dep = [0, m['hw']] + list(m['hw'] + np.cumsum(m['thk'])); res = [2e14, 1 / m['sw']] + m['res']
    kw = dict(depth=dep, res=res, freqtime=t, signal=-1, verb=1, htarg={'dlf': 'key_201_2009', 'pts_per_dec': 0},
              ftarg={'dlf': 'key_201_2012', 'pts_per_dec': 0})
    return np.stack([sum(w / 2 * np.asarray(empymod.dipole(src=[x * L / 2, 0, zs], rec=[off, 0, zr], **kw)) for x, w in zip(XG, WG))
                     for off, zr in rx])

def emp(m, ftarg):
    dep = [0, m['hw']] + list(m['hw'] + np.cumsum(m['thk']))
    res = [2e14, 1 / m['sw']] + m['res']
    return np.stack([np.asarray(empymod.bipole(src=[-L/2, L/2, 0, 0, zs, zs], rec=[off, 0, zr, 0, 0], depth=dep, res=res,
            freqtime=t, signal=-1, srcpts=11, strength=0, verb=1, htarg={'dlf': 'key_201_2009', 'pts_per_dec': 0},
            ftarg=ftarg)) for off, zr in rx])

def tor(m, mode, ppd=20, grad=False):
    T = lambda v: torch.tensor(v, dtype=torch.float64)[None]
    p = [T(m['sw']), T(m['hw']), T(m['res']), T(m['thk']) if m['thk'] else torch.zeros(1, 0, dtype=torch.float64)]
    return stepoff(*p, t, rx, L=L, zs=zs, mode=mode, ppd=ppd)[0].numpy()

rel = lambda x, y: float(np.max(np.abs(x - y) / np.abs(y)))
res = []
for k, m in enumerate(models):
    t0 = time.time(); bip = emp(m, {'dlf': 'key_201_2012', 'pts_per_dec': 0}); te = time.time() - t0
    lag = emp(m, {'dlf': 'key_201_2012'}); ref = emp_sum(m)
    t0 = time.time(); ex = tor(m, 'exact'); tx_ = time.time() - t0
    r = dict(model=m, exact_vs_empymod=rel(ex, ref), empymod_bipole_vs_sum=rel(bip, ref), empymod_bipole_vs_sum_gates2plus=rel(bip[:, 1:], ref[:, 1:]), empymod_lagged_vs_std=rel(lag, ref), t_empymod=te, t_exact=tx_)
    for ppd in (10, 20, 40):
        t0 = time.time(); ip = tor(m, 'interp', ppd); r[f'interp{ppd}_vs_empymod'] = rel(ip, ref); r[f't_interp{ppd}'] = time.time() - t0
    res.append(r)
    print(k, {kk: (round(v, 9) if isinstance(v, float) else v) for kk, v in r.items() if kk != 'model'}, 'K', len(m['res']), flush=True)

# gradient check: autograd vs central FD on log-parameters, interp mode, model 0
m = models[0]
def f(theta):
    sw, hw, r1, r2, r3, h1, h2 = theta
    return stepoff(sw[None], hw[None], torch.stack([r1, r2, r3])[None], torch.stack([h1, h2])[None], t, rx, L=L, zs=zs)[0]
th = torch.tensor([m['sw'], m['hw']] + m['res'] + m['thk'], dtype=torch.float64, requires_grad=True)
y = f(th); w = torch.randn(y.shape, dtype=y.dtype, generator=torch.Generator().manual_seed(0)) / y.detach().abs()
g = torch.autograd.grad((w * y).sum(), th)[0].numpy()
fd = []
for i in range(len(th)):
    h = 1e-5 * float(th[i]); tp = th.detach().clone(); tm = th.detach().clone(); tp[i] += h; tm[i] -= h
    fd.append(float(((w * (f(tp) - f(tm))).sum() / (2 * h))))
fd = np.array(fd); grel = float(np.max(np.abs(g - fd) / (np.abs(fd) + 1e-30)))
print('grad autograd vs FD max rel', grel, flush=True)
out = dict(geometry=dict(L=L, zs=zs, rx=rx, n_gates=len(t)), models=res, grad_rel=grel,
           summary=dict(exact_max=max(r['exact_vs_empymod'] for r in res),
                        interp20_max=max(r['interp20_vs_empymod'] for r in res),
                        interp40_max=max(r['interp40_vs_empymod'] for r in res),
                        lagged_max=max(r['empymod_lagged_vs_std'] for r in res),
                        bipole_vs_sum_max=max(r['empymod_bipole_vs_sum'] for r in res)),
           note='reference = Gauss-Legendre (11) sum of empymod.dipole, standard DLF; empymod.bipole differs from that sum '
                'by up to ~4e-5 at the first gate only (empymod internal), <3e-7 elsewhere')
json.dump(out, open(a.out, 'w'), indent=1); print('SUMMARY', out['summary'])
