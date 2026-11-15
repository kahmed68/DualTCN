"""
generate_torch_dataset.py -- v6 dataset with the differentiable torch forward.

Same sampling ranges and output keys as generate_large_dataset.py
(E_multi, log_amps, sigma_profiles, params), plus 'freqs' and 'meta'.
No v0 factor; frequency grid FREQS_V6 (see config_v6).

Usage:
  PYTHONPATH=src python src/generate_torch_dataset.py --n 1000000 --out <path> [--device cuda]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from config_v6 import (RECEIVER_OFFSETS, Z_OBS, N_TIME, N_DEPTH, Z_MAX,
                       SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE,
                       FREQS_V6, DATA_PATH_V6)
from forward_model_v2 import build_sigma_profile
import forward_torch as ft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--out", default=DATA_PATH_V6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    rng = np.random.default_rng(a.seed)
    n = a.n
    s1 = 10 ** rng.uniform(*SIGMA1_LOG_RANGE, n)
    s2 = 10 ** rng.uniform(*SIGMA2_LOG_RANGE, n)
    d1 = rng.uniform(*D1_RANGE, n)
    d2 = rng.uniform(*D2_RANGE, n)

    R = len(RECEIVER_OFFSETS)
    E_multi = np.zeros((n, R, N_TIME), np.float32)
    log_amps = np.zeros((n, R), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n, a.batch):
            sl = slice(i0, min(i0 + a.batch, n))
            T = [torch.tensor(v[sl], dtype=torch.float64, device=a.device)
                 for v in (s1, s2, d1, d2)]
            Ef = ft.freq_response(*T, RECEIVER_OFFSETS, Z_OBS, FREQS_V6)
            En, la = ft.time_features(Ef, N_TIME)
            E_multi[sl] = En.float().cpu().numpy()
            log_amps[sl] = la.float().cpu().numpy()
            if (i0 // a.batch) % 500 == 0:
                print(f"{sl.stop}/{n}  {time.time() - t0:.0f}s", flush=True)

    z = np.linspace(0.0, Z_MAX, N_DEPTH)
    seafloor = (d1 + d2)[:, None]
    sig_prof = np.where(z[None, :] < seafloor, s1[:, None], s2[:, None])
    sig_prof = np.log10(sig_prof + 1e-12).astype(np.float32)
    # identical to forward_model_v2.build_sigma_profile; spot-check
    assert np.allclose(sig_prof[0], build_sigma_profile(s1[0], s2[0], d1[0], d2[0], z))
    params = np.stack([s1, s2, d1, d2], 1).astype(np.float32)

    if not (np.isfinite(E_multi).all() and np.isfinite(log_amps).all()):
        raise RuntimeError("non-finite values in generated data")

    meta = dict(seed=a.seed, n=n, freqs="FREQS_V6", v0="none",
                forward="forward_torch.freq_response (Key-201 DLF, empymod-equivalent)",
                device=a.device, seconds=round(time.time() - t0, 1))
    np.savez(a.out, E_multi=E_multi, log_amps=log_amps, sigma_profiles=sig_prof,
             params=params, freqs=FREQS_V6, meta=json.dumps(meta))
    print(json.dumps(meta), f"-> {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
