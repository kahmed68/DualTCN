#!/usr/bin/env python3
"""td_conversion.py -- Level 2 (Khaled 2026-11-15): can time-gated step-off records be converted to the
band-limited input of DualTCN?  NOT requested by the reviewers (mock round 2 only); evaluation only, no training.

For test models of the corrected dataset:
  1. Simulate step-off records E_step(t) = int_t^inf g dtau at log-spaced gates (default 40 gates, 0.1 ms to 1 s)
     with the same layered-earth kernel (forward_torch.freq_response) and the 201-point cosine Fourier filter
     (empymod signal=-1 convention, as forward_al611.stepoff), plus the on-time (DC) level E_step(0) = E(f -> 0).
  2. Convert to the frequency response at the generator bins f_k = k/32 Hz (k=0 -> 1e-3 Hz proxy) by
        E(f) = E_step(0) - i 2 pi f * int_0^inf E_step(t) exp(-i 2 pi f t) dt
     (integration by parts of E(f) = int g exp(-i w t) dt, time dependence exp(+i w t)); the integral is
     evaluated on a dense log grid interpolated from the gates; E_step = 0 after the last gate.
  3. Build the network input exactly as the generator (forward_torch.time_features) and evaluate the trained
     networks. Conditions: direct input (reference), conversion of dense exact gates (check of the formula),
     conversion of the field-like gates without noise, and with relative Gaussian noise on every gate and on the
     on-time level (1, 2, 5 %).
Output: analysis/td_conversion/td_conversion.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import forward_torch as ft
import train_v6 as tv
from config_v4 import RECEIVER_OFFSETS, Z_OBS, N_TIME
from config_v6 import FREQS_V6, DATA_PATH_V6, IN_CHANNELS, N_DEPTH, LATENT_DIM, DROPOUT, TRAIN_SPLIT, VAL_SPLIT

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=5000)
ap.add_argument('--gates', type=int, default=40)
ap.add_argument('--tmin', type=float, default=1e-4)
ap.add_argument('--tmax', type=float, default=1.0)
ap.add_argument('--noise', default='0.01,0.02,0.05')
ap.add_argument('--runs', default='rn_base_s1,rn_base_s2,rn_base_s3')
ap.add_argument('--root', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/e1')
ap.add_argument('--out', default=os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/td_conversion')
ap.add_argument('--ppd', type=int, default=20)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--subset', type=int, default=0, help='smoke: use only the first N samples of the file')
a = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
t0 = time.time()
FOUR = np.load(os.path.join(ROOT, 'src', 'field_al611', 'key_201_2012_cos.npz'))
FB = torch.as_tensor(FOUR['base'], dtype=torch.float64, device=dev)
FC = torch.as_tensor(FOUR['cos'], dtype=torch.float64, device=dev)


def interp_logf(lq, lg, y):
    """cubic Hermite of y(B,R,F) on uniform log10 grid lg at lq (Q,)."""
    dx = lg[1] - lg[0]
    m = torch.empty_like(y)
    m[..., 1:-1] = (y[..., 2:] - y[..., :-2]) / (2 * dx); m[..., 0] = (y[..., 1] - y[..., 0]) / dx
    m[..., -1] = (y[..., -1] - y[..., -2]) / dx
    i = torch.clamp(((lq - lg[0]) / dx).floor().long(), 0, len(lg) - 2)
    u = (lq - lg[i]) / dx
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    return h00 * y[..., i] + h10 * dx * m[..., i] + h01 * y[..., i + 1] + h11 * dx * m[..., i + 1]


def stepoff(q, times):
    """q: (B,4) physical params. Returns E_step (B,R,T) float64 at times (T,), cosine DLF (signal=-1)."""
    t = torch.as_tensor(times, dtype=torch.float64, device=dev)
    w = FB[None, :] / t[:, None]                                         # (T,201)
    lw = torch.log10(w / (2 * math.pi))
    lo = math.floor(lw.min().item() * a.ppd) / a.ppd - 2.0 / a.ppd
    hi = math.ceil(lw.max().item() * a.ppd) / a.ppd + 2.0 / a.ppd
    lg = torch.arange(lo, hi + 0.5 / a.ppd, 1.0 / a.ppd, dtype=torch.float64, device=dev)
    f = 10 ** lg
    Ef = ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3], RECEIVER_OFFSETS, Z_OBS, f)   # (B,R,F)
    y = (-Ef / (1j * 2 * math.pi * f)).real * f                         # smooth in log f
    g = interp_logf(lw.reshape(-1), lg, y) / 10 ** lw.reshape(-1)       # (B,R,T*201)
    g = g.reshape(*g.shape[:2], *w.shape)
    return (g * FC).sum(-1) / t * (2.0 / math.pi)


def e_dc(q):
    return ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3], RECEIVER_OFFSETS, Z_OBS, [1e-5]).real[..., 0]


def convert(Est, times, Edc, n_dense=2000):
    """Est (B,R,T) at gates `times`; Edc (B,R). Returns E(f_k) (B,R,65) complex."""
    tg = torch.as_tensor(times, dtype=torch.float64, device=dev)
    td = torch.logspace(math.log10(times[0]), math.log10(times[-1]), n_dense, dtype=torch.float64, device=dev)
    # linear interpolation in log t
    x, xq = torch.log10(tg), torch.log10(td)
    i = torch.clamp(torch.searchsorted(x, xq) - 1, 0, len(x) - 2)
    u = (xq - x[i]) / (x[i + 1] - x[i])
    Ed = Est[..., i] * (1 - u) + Est[..., i + 1] * u                    # (B,R,D)
    tt = torch.cat([torch.zeros(1, dtype=torch.float64, device=dev), td])
    EE = torch.cat([Edc[..., None], Ed], -1)                            # straight line from t=0 to the first gate
    fk = torch.as_tensor(FREQS_V6, dtype=torch.float64, device=dev)
    ph = torch.exp(-1j * 2 * math.pi * fk[:, None] * tt[None, :])       # (F,D+1)
    vals = EE[..., None, :] * ph                                        # (B,R,F,D+1)
    I = (0.5 * (vals[..., 1:] + vals[..., :-1]) * (tt[1:] - tt[:-1])).sum(-1)
    return Edc[..., None].to(torch.complex128) - 1j * 2 * math.pi * fk * I


def features(Ef):
    En, la = ft.time_features(Ef, N_TIME)
    A = la[..., None].expand_as(En)
    return torch.stack([En.float(), A.float()], 2).reshape(En.shape[0], -1, En.shape[-1])


E, L, S, Pn = tv.load(DATA_PATH_V6, dev)
if a.subset:
    E, L, Pn = E[:a.subset], L[:a.subset], Pn[:a.subset]
n = len(E); n_tr = int(n * TRAIN_SPLIT); n_va = int(n * VAL_SPLIT)
te = torch.arange(n_tr + n_va, min(n, n_tr + n_va + a.n), device=dev)
B0 = tv.BOUNDS.to(dev).double()
logp = B0[:, 0] + Pn[te].double() * (B0[:, 1] - B0[:, 0])
Q = 10 ** logp                                                          # physical params (B,4)
print(f'n_test={len(te)} {dev} {time.time()-t0:.0f}s', flush=True)

models = []
for r in a.runs.split(','):
    m = tv.ARCHS['dualtcn'](in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT).to(dev)
    m.load_state_dict(torch.load(os.path.join(a.root, r, 'best.pt'), map_location=dev, weights_only=True)); m.eval()
    models.append(m)

gates = np.logspace(math.log10(a.tmin), math.log10(a.tmax), a.gates)
dense = np.logspace(-7, 1.5, 600)
noise = [0.0] + [float(x) for x in a.noise.split(',') if x]
conds = ['direct', 'regen', 'dense_exact'] + [f'gates_{e:g}' for e in noise]
X = {c: [] for c in conds}
chk = dict(regen_vs_file=[], dense_vs_direct=[], gates_vs_direct=[])
g = torch.Generator(device=dev); g.manual_seed(a.seed)
with torch.no_grad():
    for b0 in range(0, len(te), a.bs):
        q = Q[b0:b0 + a.bs]; idx = te[b0:b0 + a.bs]
        En, la = E[idx], L[idx]
        X['direct'].append(torch.stack([En, la[..., None].expand_as(En)], 2).reshape(En.shape[0], -1, En.shape[-1]).float())
        Ef_true = ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3], RECEIVER_OFFSETS, Z_OBS, FREQS_V6)
        X['regen'].append(features(Ef_true))
        chk['regen_vs_file'].append((ft.time_features(Ef_true, N_TIME)[0] - En).abs().amax().item())
        Edc = e_dc(q)
        Ed = convert(stepoff(q, dense), dense, Edc)
        chk['dense_vs_direct'].append(((Ed - Ef_true).abs() / Ef_true.abs().amax(-1, keepdim=True)).amax().item())
        X['dense_exact'].append(features(Ed))
        Eg = stepoff(q, gates)
        for e in noise:
            Egn = Eg * (1 + e * torch.randn(Eg.shape, device=dev, generator=g, dtype=Eg.dtype))
            Edn = Edc * (1 + e * torch.randn(Edc.shape, device=dev, generator=g, dtype=Edc.dtype))
            Ec = convert(Egn, gates, Edn)
            if e == 0:
                chk['gates_vs_direct'].append(((Ec - Ef_true).abs() / Ef_true.abs().amax(-1, keepdim=True)).amax().item())
            X[f'gates_{e:g}'].append(features(Ec))
        if b0 == 0:
            print('first batch done', f'{time.time()-t0:.0f}s', flush=True)

y = Pn[te].float()
res = dict(args=vars(a), gates_s=gates.tolist(), checks={k: float(np.max(v)) for k, v in chk.items()}, conditions={})


def r2(p):
    return (1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).cpu().numpy()


with torch.no_grad():
    for c in conds:
        x = torch.cat(X[c])
        rr = np.stack([r2(torch.cat([m(x[i:i + 2048])[1].float() for i in range(0, len(x), 2048)])) for m in models])
        res['conditions'][c] = dict(r2_runs=rr.round(4).tolist(), r2_mean=rr.mean(0).round(4).tolist(),
                                    r2_sd=rr.std(0, ddof=1).round(4).tolist() if len(models) > 1 else None)
        print(f'{c:14s} R2 mean {rr.mean(0).round(3)}', flush=True)
print('checks', res['checks'])
os.makedirs(a.out, exist_ok=True)
fn = os.path.join(a.out, 'td_conversion.json' if not a.subset else 'smoke_td_conversion.json')
json.dump(res, open(fn, 'w'), indent=1)
print('wrote', fn, f'{time.time()-t0:.0f}s')
