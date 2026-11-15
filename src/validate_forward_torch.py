"""
validate_forward_torch.py -- Step 1 gate: torch forward model vs empymod.

Draws N random earth models from the config_v4 training ranges and compares
  (a) frequency-domain Ex (per spectrum: max|diff| / max|ref|),
  (b) normalised time traces (max abs diff) and log_amp (abs diff),
with the dataset path forward_model_v4.compute_multi_receiver_timeseries
(v0 included). Also runs torch.autograd.gradcheck on a few models.

Gate (fixed before running): spectra rel err <= 1e-6 for every model,
time traces <= 1e-5, log_amp <= 1e-6, gradcheck passes.

Usage: PYTHONPATH=.pydeps:src python src/validate_forward_torch.py --n 2000
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import empymod

from config_v4 import (RECEIVER_OFFSETS, Z_OBS, F_MIN, F_MAX, N_FREQ, N_TIME,
                       SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE,
                       V0_RANGE)
from forward_model_v4 import compute_multi_receiver_timeseries
import forward_torch as ft

GATE = dict(spec_rel=1e-6, trace_abs=1e-5, logamp_abs=1e-6)


def empymod_spectra(s1, s2, d1, d2, freqs):
    out = np.zeros((len(RECEIVER_OFFSETS), len(freqs)), complex)
    for j, r in enumerate(RECEIVER_OFFSETS):
        out[j] = empymod.dipole(src=[0, 0, d1], rec=[r, 0, Z_OBS],
                                depth=[0, d1 + d2], res=[1e23, 1 / s1, 1 / s2],
                                freqtime=freqs, ab=11, verb=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results/step1_forward_torch")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    n = a.n
    P = dict(s1=10 ** rng.uniform(*SIGMA1_LOG_RANGE, n),
             s2=10 ** rng.uniform(*SIGMA2_LOG_RANGE, n),
             d1=rng.uniform(*D1_RANGE, n), d2=rng.uniform(*D2_RANGE, n),
             v0=rng.uniform(*V0_RANGE, n))
    freqs = np.linspace(F_MIN, F_MAX, N_FREQ)
    T = {k: torch.tensor(v, dtype=torch.float64) for k, v in P.items()}

    t0 = time.time()
    Ef_t = []
    for i0 in range(0, n, 100):
        sl = slice(i0, i0 + 100)
        Ef_t.append(ft.freq_response(T["s1"][sl], T["s2"][sl], T["d1"][sl],
                                     T["d2"][sl], RECEIVER_OFFSETS, Z_OBS, freqs))
    Ef_t = torch.cat(Ef_t)
    tn, la_t = ft.time_features(Ef_t * ft.velocity_factor(freqs, T["v0"], T["s1"]),
                                N_TIME)
    t_torch = time.time() - t0

    spec_rel = np.zeros((n, len(RECEIVER_OFFSETS)))
    trace = np.zeros((n, len(RECEIVER_OFFSETS)))
    logamp = np.zeros((n, len(RECEIVER_OFFSETS)))
    t0 = time.time()
    for i in range(n):
        ref = empymod_spectra(P["s1"][i], P["s2"][i], P["d1"][i], P["d2"][i], freqs)
        got = Ef_t[i].numpy()
        spec_rel[i] = np.abs(got - ref).max(-1) / np.abs(ref).max(-1)
        res = compute_multi_receiver_timeseries(P["s1"][i], P["s2"][i], P["d1"][i],
                                                P["d2"][i], P["v0"][i])
        for j, r in enumerate(RECEIVER_OFFSETS):
            En, la = res[r]
            trace[i, j] = np.abs(tn[i, j].numpy() - En.astype(np.float64)).max()
            logamp[i, j] = abs(la_t[i, j].item() - la)
    t_emp = time.time() - t0

    # gradcheck on 3 models (complex128 internally, real outputs)
    def f(x):
        En, la = ft.forward_features(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
        return torch.cat([En.flatten(), la.flatten()])
    x = torch.stack([T["s1"][:3], T["s2"][:3], T["d1"][:3], T["d2"][:3]], 1)
    x = x.clone().requires_grad_(True)
    gc = bool(torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-5, rtol=1e-4))

    summary = dict(
        n=n, seed=a.seed, gate=GATE,
        spec_rel_max=float(spec_rel.max()), spec_rel_median=float(np.median(spec_rel)),
        trace_abs_max=float(trace.max()), logamp_abs_max=float(logamp.max()),
        gradcheck=gc, worst_model=int(spec_rel.max(1).argmax()),
        sec_torch_cpu=round(t_torch, 2), sec_empymod=round(t_emp, 2),
    )
    summary["PASS"] = bool(summary["spec_rel_max"] <= GATE["spec_rel"]
                           and summary["trace_abs_max"] <= GATE["trace_abs"]
                           and summary["logamp_abs_max"] <= GATE["logamp_abs"] and gc)
    w = summary["worst_model"]
    summary["worst_params"] = {k: float(v[w]) for k, v in P.items()}
    np.savez(os.path.join(a.out, "errors.npz"), spec_rel=spec_rel, trace=trace,
             logamp=logamp, **P)
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
