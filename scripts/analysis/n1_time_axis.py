"""
n1_time_axis.py -- N1 check (mock R3, 2026-11-15): does the 128-sample, 0-2 Hz
trace (dt 0.25 s, 0-32 s) represent the physical transient?

For a set of models in the training ranges we compute
  (a) the broadband impulse response h(t) on a fine grid (fs = 20 kHz, T = 4 s;
      and fs = 200 Hz, T = 256 s for the tail), with the same operator
      forward_torch.freq_response;
  (b) the network input trace x[n] = irfft(E(FREQS_V6), 128) exactly as in the
      generator, before peak normalisation.
Reports: peak time and t(|h| < 1% peak) of h; energy fractions of x in sample 0,
samples 1-63 (early branch) and 64-127 (late branch = t in [16,32) s, which is
t in [-16,0) s of the periodic irfft); and a check that x equals h low-passed at
2 Hz and sampled at 0.25 s.  CPU only.  Output: analysis/n1_time_axis.json
"""
import json, math, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import forward_torch as ft
from config_v4 import RECEIVER_OFFSETS, Z_OBS, N_TIME
from config_v6 import FREQS_V6

torch.set_num_threads(8)

MODELS = {  # name: (sigma1 S/m, sigma2 S/m, d1 m, d2 m)
    "typical":      (3.3, 1.0,   100.0, 30.0),
    "resistive_sf": (3.3, 0.001, 100.0, 30.0),
    "low_s1":       (0.1, 0.03,  100.0, 30.0),
    "shallow":      (3.3, 0.1,   50.0,  10.0),
    "deep":         (3.3, 0.1,   150.0, 50.0),
    "mid":          (1.0, 0.03,  100.0, 30.0),
    "corner_slow":  (5.01, 1.0,  150.0, 50.0),   # slowest diffusion in the prior
    "corner_slow_res": (5.01, 0.001, 150.0, 50.0),
}


def fr(p, freqs, chunk=4000):
    s1, s2, d1, d2 = [torch.tensor([v], dtype=torch.float64) for v in p]
    out = []
    for i in range(0, len(freqs), chunk):
        out.append(ft.freq_response(s1, s2, d1, d2, RECEIVER_OFFSETS, Z_OBS,
                                    freqs[i:i + chunk])[0])
    return torch.cat(out, -1).numpy()  # (R, F)


def impulse(p, fs, T):
    n = int(round(fs * T))
    f = np.arange(n // 2 + 1) / T
    f[0] = 1e-4
    E = fr(p, f)
    h = np.fft.irfft(E, n=n, axis=-1) * fs  # continuous-time scaling
    t = np.arange(n) / fs
    return t, h, f, E


def main():
    res = {}
    for name, p in MODELS.items():
        r = {}
        # (b) network input trace
        Ex = fr(p, FREQS_V6)
        x = np.fft.irfft(Ex, n=N_TIME, axis=-1)
        # (a) fine impulse response, early part
        t, h, f, E = impulse(p, 20000.0, 4.0)
        # (a') tail, coarse
        t2, h2, _, _ = impulse(p, 200.0, 256.0)
        for k, off in enumerate(RECEIVER_OFFSETS):
            hk = h[k, : len(t) // 2]
            tk = t[: len(t) // 2]
            ip = int(np.argmax(np.abs(hk)))
            pk = abs(hk[ip])
            above = np.nonzero(np.abs(hk) >= 0.01 * pk)[0]
            t01 = float(tk[above[-1]]) if len(above) else float("nan")
            above3 = np.nonzero(np.abs(hk) >= 1e-3 * pk)[0]
            t001 = float(tk[above3[-1]]) if len(above3) else float("nan")
            e_tot = np.sum(hk ** 2)
            e_after_025 = float(np.sum(hk[tk >= 0.25] ** 2) / e_tot)
            # tail beyond 2 s relative to peak, from coarse run
            h2k = h2[k, : len(t2) // 2]; t2k = t2[: len(t2) // 2]
            tail16 = float(np.max(np.abs(h2k[(t2k >= 16) & (t2k < 32)])) / pk)
            xk = x[k]
            ex = np.sum(xk ** 2)
            # x vs h low-passed at 2 Hz, sampled at 0.25 s (from coarse run, T=256 s,
            # ideal brick-wall filter at 2 Hz, then periodise at 32 s)
            n2 = len(t2)
            H2 = np.fft.rfft(h2[k]) / 200.0
            f2 = np.fft.rfftfreq(n2, 1 / 200.0)
            H2[f2 > 2.0 + 1e-9] = 0.0
            hl = np.fft.irfft(H2, n=n2) * 200.0
            samp = hl[::50]  # 0.25 s grid over 256 s
            per = samp.reshape(-1, N_TIME).sum(0) * 0.25  # periodise at 32 s, dt scaling
            rel = float(np.linalg.norm(per - xk) / np.linalg.norm(xk))
            r[f"r{int(off)}"] = dict(
                t_peak_ms=1e3 * float(tk[ip]),
                t_1pct_s=t01, t_0p1pct_s=t001,
                energy_frac_after_0p25s=e_after_025,
                max_abs_16_32s_over_peak=tail16,
                x_energy_sample0=float(xk[0] ** 2 / ex),
                x_energy_1_63=float(np.sum(xk[1:64] ** 2) / ex),
                x_energy_64_127=float(np.sum(xk[64:] ** 2) / ex),
                x_absmax_64_127_over_absmax=float(np.max(np.abs(xk[64:])) / np.max(np.abs(xk))),
                x_argmax=int(np.argmax(np.abs(xk))),
                x_vs_lowpassed_h_relerr=rel,
                diffusion_time_mu_s_r2_s=4e-7 * math.pi * p[0] * off ** 2,
            )
        res[name] = dict(params=dict(zip(["sigma1", "sigma2", "d1", "d2"], p)), rec=r)
        print(name, json.dumps(r, indent=None)[:2000], flush=True)
    out = os.path.join(os.environ.get("DUALTCN_RUNS", ROOT), "analysis", "n1_time_axis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
