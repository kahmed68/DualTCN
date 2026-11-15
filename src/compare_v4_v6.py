"""
compare_v4_v6.py -- how much the v6 data model changes the network input.
Compares, on the same random models (no noise):
  v4      : linspace(0.05,2,64) grid + velocity factor (v0 ~ U[0,100]) [v1 paper data]
  v4_nov0 : linspace grid, no velocity factor
  v6      : exact FFT grid (config_v6.FREQS_V6), no velocity factor   [revision data]
Reports mean/median per-trace Pearson correlation and |delta log_amp| (decades).
Usage: PYTHONPATH=src python src/compare_v4_v6.py --n 2000
"""
import argparse, json, os
import numpy as np, torch
from config_v4 import RECEIVER_OFFSETS, Z_OBS, F_MIN, F_MAX, N_FREQ, N_TIME, V0_RANGE
from config_v6 import SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE, FREQS_V6
import forward_torch as ft

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--out", default="results/step1_forward_torch/compare_v4_v6.json"); a = ap.parse_args()
rng = np.random.default_rng(2026)
P = [10 ** rng.uniform(*SIGMA1_LOG_RANGE, a.n), 10 ** rng.uniform(*SIGMA2_LOG_RANGE, a.n),
     rng.uniform(*D1_RANGE, a.n), rng.uniform(*D2_RANGE, a.n)]
v0 = torch.tensor(rng.uniform(*V0_RANGE, a.n), dtype=torch.float64)
T = [torch.tensor(p, dtype=torch.float64) for p in P]
f4 = np.linspace(F_MIN, F_MAX, N_FREQ)
with torch.no_grad():
    E4 = ft.freq_response(*T, RECEIVER_OFFSETS, Z_OBS, f4)
    E6 = ft.freq_response(*T, RECEIVER_OFFSETS, Z_OBS, FREQS_V6)
    feats = dict(v4=ft.time_features(E4 * ft.velocity_factor(f4, v0, T[0]), N_TIME),
                 v4_nov0=ft.time_features(E4, N_TIME), v6=ft.time_features(E6, N_TIME))

def cmp(x, y):
    (ex, lx), (ey, ly) = feats[x], feats[y]
    xc = ex - ex.mean(-1, keepdim=True); yc = ey - ey.mean(-1, keepdim=True)
    r = (xc * yc).sum(-1) / (xc.norm(dim=-1) * yc.norm(dim=-1))
    d = (lx - ly).abs()
    return dict(corr_mean=float(r.mean()), corr_median=float(r.median()), corr_p05=float(r.quantile(0.05)),
                dlogamp_mean=float(d.mean()), dlogamp_median=float(d.median()), dlogamp_max=float(d.max()))

s = dict(n=a.n, seed=2026, grid_only=cmp("v4_nov0", "v6"), v0_only=cmp("v4", "v4_nov0"), v1_to_v6=cmp("v4", "v6"))
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(s, open(a.out, "w"), indent=2); print(json.dumps(s, indent=2))
