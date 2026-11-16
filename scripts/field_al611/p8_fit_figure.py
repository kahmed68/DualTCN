#!/usr/bin/env python3
"""AL611 Profile 8: misfit per sounding and observed vs fitted transients (supplement figure, R3.6; 2026-11-14).
Re-simulates the G2 solutions (analysis/al611/g2_gpu_p{0,1,2}.json) with the same operator settings (interp, ppd 20)."""
import json, os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__)); src = os.path.join(here, '..', '..', 'src')
sys.path[:0] = [src, os.path.join(src, 'field_al611')]
import torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.style.use(os.path.join(here, '..', 'figures', 'dualtcn.mplstyle'))  # common style (R1.12)
from read_emdata import read_emdata
from forward_al611 import stepoff

A = '/work/hdd/bgte/kahmed2/DualTCN_runs/analysis/al611/'
OUT = sys.argv[1] if len(sys.argv) > 1 else 'p8_field_fit.pdf'
R = sorted(sum([json.load(open(A + f'g2_gpu_p{i}.json')) for i in range(3)], []), key=lambda x: x['tx'])
e = read_emdata('/work/hdd/bgte/kahmed2/field_data/AL611/raw/AL611_CSEM_Profile8.emdata')
t, tx, rxd, d = e['times'], e['tx'], e['rx'], e['data']
rms = np.array([x['rms'] for x in R]); txn = np.array([x['tx'] for x in R])
pick = [int(np.argmin(rms)), int(np.argsort(rms)[len(rms) // 2]), int(np.argmax(rms))]

fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.4))
ax[0].plot(txn, rms, 'o-', ms=2.5, lw=0.9, color='k'); ax[0].axhline(1, ls='--', color='gray')
for j in pick: ax[0].plot(txn[j], rms[j], 'o', ms=7, mfc='none', color='C1')
ax[0].set_xlabel('Transmitter position'); ax[0].set_ylabel('Error-normalised RMS misfit'); ax[0].set_title('(a) Profile 8, 53 soundings')
ax[0].set_ylabel('Normalized RMS misfit')
for k, j in enumerate(pick):
    r = R[j]; itx = r['tx'] - 1; a = ax[k + 1]
    rx = [tuple(v) for v in r['rx']]
    y = stepoff(torch.tensor([r['sw_fixed']], dtype=torch.float64), torch.tensor([r['hw']], dtype=torch.float64),
                torch.tensor([r['rho']], dtype=torch.float64), torch.tensor([r['h']], dtype=torch.float64),
                t, rx, L=float(tx['length'][itx]), zs=float(tx['z'][itx]), ppd=20)[0].numpy()
    for q, (irx, c) in enumerate(zip(sorted(set(d['irx'][d['itx'] == itx])), ['C0', 'C2'])):
        m = (d['itx'] == itx) & (d['irx'] == irx); ii = d['it'][m]
        a.errorbar(t[ii], d['value'][m], yerr=d['err'][m], fmt='o', ms=2.5, elinewidth=0.7, color=c, label=f'{rx[q][0]:.0f} m, observed')
        a.plot(t, y[q], '-', color=c, lw=1, label=f'{rx[q][0]:.0f} m, fitted')
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Time (s)')
    a.set_title(f"({'bcd'[k]}) Tx {r['tx']}, RMS {r['rms']:.2f}")
    if k == 0: a.set_ylabel('E (V/m per A m)')
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.6, -0.02))
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT)
print('picked', [(R[j]['tx'], round(R[j]['rms'], 2)) for j in pick], '->', OUT)
