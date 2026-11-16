"""Timing of the differentiable forward operator (Introduction, R1.14), 2026-11-15.
Same call as train_v6.misfit_terms: freq_response + time_features in float64 on the GPU.
Reports median wall time over 50 repeats (5 warm-up) for forward only and forward+backward, batch 256 and 64."""
import os
import json, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
import forward_torch as ft
import train_v6 as T
dev = 'cuda'
print('torch', torch.__version__, torch.cuda.get_device_name(0), flush=True)
lo = torch.tensor([-1.0, -3.0, np.log10(50), 1.0], dtype=torch.float64)
hi = torch.tensor([np.log10(5.01), 0.0, np.log10(150), np.log10(50)], dtype=torch.float64)
res = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
g = torch.Generator().manual_seed(0)
for B in (256, 64):
    for mode in ('fwd', 'fwd+bwd'):
        ts = []
        for rep in range(55):
            p = (lo + (hi - lo) * torch.rand(B, 4, generator=g, dtype=torch.float64)).to(dev)
            p.requires_grad_(mode == 'fwd+bwd')
            torch.cuda.synchronize(); t0 = time.perf_counter()
            q = 10 ** p
            En, la = ft.time_features(ft.freq_response(q[:, 0], q[:, 1], q[:, 2], q[:, 3],
                                                       T.RECEIVER_OFFSETS, T.Z_OBS, T.FREQS_V6), T.N_TIME)
            if mode == 'fwd+bwd':
                (En.square().mean() + la.square().mean()).backward()
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            if rep >= 5: ts.append(dt * 1e3)
        res[f'B{B}_{mode}_ms_median'] = float(np.median(ts)); res[f'B{B}_{mode}_ms_p90'] = float(np.percentile(ts, 90))
        print(B, mode, 'median %.1f ms  p90 %.1f ms' % (np.median(ts), np.percentile(ts, 90)), flush=True)
json.dump(res, open(os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/time_forward_torch.json', 'w'), indent=1)
