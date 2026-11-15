"""Noise robustness figure (Section 3.3; replaces main-text Table 4, which moves to the Supplementary Material).
Inputs: analysis/n1_u1/n1_u1_base.json (original noise model), n1_u1_rn_base.json (noise on raw transients):
r2_mean and r2_sd over three seeds per condition; analysis/j1/j1_snr.json: linearized R2 estimate per SNR.
Also checks that every plotted mean rounds to the value printed in Table 4 (n1_compare_rn_base_3s.txt)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, 'dualtcn.mplstyle'))
A = '/work/hdd/bgte/kahmed2/DualTCN_runs/analysis/'
OUT = sys.argv[1]
orig = json.load(open(A + 'n1_u1/n1_u1_base.json'))
raw = json.load(open(A + 'n1_u1/n1_u1_rn_base.json'))
assert orig['order'] == raw['order'], (orig['order'], raw['order'])
order = orig['order']
lin = {L['snr_db']: L['r2_ceiling'] for L in json.load(open(A + 'j1/j1_snr.json'))['levels']}
KEYS = {'sigma1': r'$\sigma_1$', 'sigma2': r'$\sigma_2$', 'd1': r'$d_1$', 'd2': r'$d_2$'}
names = list(KEYS)
idx = [order.index(k) if isinstance(order, list) and k in order else i for i, k in enumerate(names)]

snr_c = ['clean', 'snr40', 'snr30', 'snr20', 'snr10']; snr_x = ['Clean', '40', '30', '20', '10']
gain_c = ['clean', 'gain1', 'gain2', 'gain5']; gain_x = ['0', '1', '2', '5']
YMIN = -2.0

fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0), sharey=True)
for col, (k, i) in enumerate(zip(names, idx)):
    for row, (conds, xl) in enumerate([(snr_c, snr_x), (gain_c, gain_x)]):
        ax = axs[row, col]; x = np.arange(len(conds))
        for d, lab, c, mk, ls in [(raw, 'Noise on raw transients', 'C0', 'o', '-'),
                                  (orig, 'Noise on normalized traces (original)', 'C1', 's', '--')]:
            m = np.array([d['conditions'][q]['r2_mean'][i] for q in conds])
            sd = np.array([d['conditions'][q]['r2_sd'][i] for q in conds])
            ax.errorbar(x, np.maximum(m, YMIN), yerr=sd, color=c, marker=mk, ls=ls, ms=3.5, capsize=2, label=lab)
            for xx, mm in zip(x, m):
                if mm < YMIN:
                    ax.annotate(f'{mm:.1f}', (xx, YMIN), textcoords='offset points', xytext=(0, 5), ha='center',
                                fontsize=6.5, color=c)
        if row == 0:
            ax.axvspan(3.5, 4.45, color='0.88', lw=0, zorder=0)
            if col == 1:
                ax.text(3.97, 0.68, 'outside\ntraining\nrange', ha='center', va='center', fontsize=6, color='0.35')
            ax.plot(x[1:], [lin[s][k] for s in (40, 30, 20, 10)], color='0.35', ls=':', marker='^', ms=3,
                    label='Linearized estimate')
            ax.set_title(KEYS[k]); ax.set_xlabel('SNR (dB)')
        else:
            ax.set_xlabel('Receiver gain error (%)')
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_ylim(YMIN - 0.1, 1.08); ax.axhline(0, color='0.6', lw=0.6)
axs[0, 0].set_ylabel('$R^2$'); axs[1, 0].set_ylabel('$R^2$')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT)

# consistency check against the table values (3 decimals as printed, 2 for |x| >= ~1 or rounded entries)
tab = {}
for line in open(A + 'n1_u1/n1_compare_rn_base_3s.txt').read().splitlines()[2:]:
    p = line.split()
    if p and p[0] in snr_c + gain_c + ['snr30_gain2']:
        import re
        vals = re.findall(r'(-?\d+\.\d+)->\s*(-?\d+\.\d+)', line)[:4]
        tab[p[0]] = [(float(a), float(b)) for a, b in vals]
bad = 0
for q, v in tab.items():
    for j, (a, b) in enumerate(v):
        i = idx[j]
        for d, ref in ((orig, a), (raw, b)):
            if abs(d['conditions'][q]['r2_mean'][i] - ref) > 6e-4:
                bad += 1; print('MISMATCH', q, names[j], d['conditions'][q]['r2_mean'][i], ref)
print('order', order, '| checked', len(tab), 'conditions, mismatches:', bad)
print('30 dB + gain 2% (not plotted, in Table S13):', {n: round(raw['conditions']['snr30_gain2']['r2_mean'][i], 3) for n, i in zip(names, idx)})
