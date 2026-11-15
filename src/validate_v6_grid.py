"""
validate_v6_grid.py -- torch forward vs empymod on the v6 frequency grid
(incl. the 1e-3 Hz near-DC bin). Gate: spectra rel err <= 1e-6 for every model.
Usage: PYTHONPATH=.pydeps:src python src/validate_v6_grid.py --n 300
"""
import argparse
import json
import os

import numpy as np
import torch
import empymod

from config_v6 import (RECEIVER_OFFSETS, Z_OBS, SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE,
                       D1_RANGE, D2_RANGE, FREQS_V6)
import forward_torch as ft

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default="results/step1_forward_torch/v6_grid_summary.json")
a = ap.parse_args()
rng = np.random.default_rng(777)
P = [10 ** rng.uniform(*SIGMA1_LOG_RANGE, a.n), 10 ** rng.uniform(*SIGMA2_LOG_RANGE, a.n),
     rng.uniform(*D1_RANGE, a.n), rng.uniform(*D2_RANGE, a.n)]
Ef = ft.freq_response(*[torch.tensor(p, dtype=torch.float64) for p in P],
                      RECEIVER_OFFSETS, Z_OBS, FREQS_V6).numpy()
err = np.zeros((a.n, len(RECEIVER_OFFSETS)))
for i in range(a.n):
    s1, s2, d1, d2 = (p[i] for p in P)
    for j, r in enumerate(RECEIVER_OFFSETS):
        ref = empymod.dipole(src=[0, 0, d1], rec=[r, 0, Z_OBS], depth=[0, d1 + d2],
                             res=[1e23, 1 / s1, 1 / s2], freqtime=FREQS_V6, ab=11, verb=0)
        err[i, j] = np.abs(Ef[i, j] - ref).max() / np.abs(ref).max()
s = dict(n=a.n, spec_rel_max=float(err.max()), spec_rel_median=float(np.median(err)),
         PASS=bool(err.max() <= 1e-6))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(s, open(a.out, "w"), indent=2)
print(json.dumps(s))
