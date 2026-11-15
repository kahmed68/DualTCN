#!/usr/bin/env python3
"""G2 with the torch operator: 1D least-squares fit per sounding (log-data residuals / relative error),
seawater fixed (default 3.33 S/m = 0.3 ohm-m, as in Lohrberg et al. 2026), water depth free, 3 sediment layers.
Jacobian by autograd. Multi-start (3). Writes JSON."""
import argparse, json, os, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--tx', default='1,27,53'); ap.add_argument('--sw', type=float, default=3.33)
ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu'); ap.add_argument('--max-nfev', type=int, default=200); ap.add_argument('--ppd', type=int, default=20)
ap.add_argument('--emdata', default='/work/hdd/bgte/kahmed2/field_data/AL611/raw/AL611_CSEM_Profile8.emdata')
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__)); src = os.path.join(here, '..', '..', 'src')
sys.path[:0] = [src, os.path.join(src, 'field_al611')]
import torch
from scipy.optimize import least_squares
from read_emdata import read_emdata
from forward_al611 import stepoff
torch.set_num_threads(8); DEV = torch.device(a.device)
e = read_emdata(a.emdata); t = e['times']
tx = e['tx']; rxd = e['rx']; d = e['data']
if a.tx == 'all': a.tx = ','.join(str(i + 1) for i in range(len(tx['x'])))
# p = [hw, log10 r1, log10 h1, log10 r2, log10 h2, log10 r3]
lo = np.array([5, -1, 0.3, -1, 0.3, -1]); hi = np.array([30, 3, 2.5, 3, 2.7, 3])
out = []
for itx in [int(s) - 1 for s in a.tx.split(',')]:
    rx, obs, err = [], [], []
    for irx in sorted(set(d['irx'][d['itx'] == itx])):
        m = (d['itx'] == itx) & (d['irx'] == irx); o = np.zeros(len(t)); s = np.full(len(t), np.inf)
        o[d['it'][m]] = d['value'][m]; s[d['it'][m]] = d['err'][m]
        rx.append((float(abs(rxd['y'][irx] - tx['y'][itx])), float(rxd['z'][irx]))); obs.append(o); err.append(s)
    obs = np.stack(obs); err = np.stack(err); ok = np.isfinite(err) & (obs > 0)
    lobs = np.log(np.where(ok, obs, 1.0)); sl = np.where(ok, err / np.where(ok, obs, 1.0), np.inf)
    L, zs = float(tx['length'][itx]), float(tx['z'][itx])
    @torch.no_grad()
    def model(P):                                     # P (B,6) -> log data (B,R,T)
        P = torch.as_tensor(np.atleast_2d(P), dtype=torch.float64, device=DEV)
        y = stepoff(torch.full((len(P),), a.sw, dtype=torch.float64, device=DEV), P[:, 0], 10 ** P[:, [1, 3, 5]], 10 ** P[:, [2, 4]],
                    t, rx, L=L, zs=zs, ppd=a.ppd)
        return np.log(np.clip(y.cpu().numpy(), 1e-30, None))
    def r(p):
        return ((model(p)[0] - lobs) / sl)[ok]
    def J(p):                                         # central differences, all 12 perturbed models in one batch
        h = 1e-4 * (hi - lo); P = np.concatenate([p + np.diag(h), p - np.diag(h)])
        Y = model(P); D = (Y[:6] - Y[6:]) / (2 * h[:, None, None])
        return (D / sl[None]).transpose(1, 2, 0)[ok]
    t0 = time.time(); best = None
    for p0 in ([14, 0.0, 1.5, 0.3, 1.8, 0.5], [12, -0.3, 1.0, 0.0, 2.0, 1.0], [18, 0.3, 1.8, 0.5, 1.5, 0.0]):
        sol = least_squares(r, p0, jac=J, bounds=(lo, hi), x_scale=hi - lo, max_nfev=a.max_nfev)
        if best is None or sol.cost < best.cost: best = sol
    p = best.x; chi = float(np.sqrt(2 * best.cost / ok.sum()))
    rec = dict(tx=itx + 1, rms=chi, n=int(ok.sum()), hw=float(p[0]), rho=[float(10 ** p[i]) for i in (1, 3, 5)],
               h=[float(10 ** p[i]) for i in (2, 4)], sw_fixed=a.sw, rx=rx, sec=time.time() - t0,
               at_bound=[bool(np.isclose(p[i], lo[i], atol=1e-3) or np.isclose(p[i], hi[i], atol=1e-3)) for i in range(6)])
    out.append(rec)
    print(f"Tx{itx+1}: RMS {chi:.2f} (n={ok.sum()})  hw {p[0]:.1f} m  rho {rec['rho'][0]:.2f}/{rec['rho'][1]:.2f}/{rec['rho'][2]:.2f} ohm-m  "
          f"h {rec['h'][0]:.0f}/{rec['h'][1]:.0f} m  bounds {rec['at_bound']}  {rec['sec']:.0f}s", flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)                 # incremental
json.dump(out, open(a.out, 'w'), indent=1)
