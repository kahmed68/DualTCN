"""Noise robustness figure (Section 3.3; replaces main-text Table 4, which moves to the Supplementary Material).
11-16 (Khaled): main network = rn_noprof (no profile loss). Curves: rn_noprof (noise on raw transients),
noprof (original noise model, no profile loss) and rn_base (raw transients, with profile loss).
Inputs: analysis/n1_u1/n1_u1_{rn_noprof,noprof,rn_base_tm}.json (TF32 off), r2_mean and r2_sd over three seeds;
analysis/j1/j1_snr.json: linearized R2 estimate per SNR."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, 'dualtcn.mplstyle'))
A = os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/'
OUT = sys.argv[1]
raw = json.load(open(A + 'n1_u1/n1_u1_rn_noprof.json'))
orig = json.load(open(A + 'n1_u1/n1_u1_noprof.json'))
prof = json.load(open(A + 'n1_u1/n1_u1_rn_base_tm.json'))
assert orig['order'] == raw['order'] == prof['order'], (orig['order'], raw['order'], prof['order'])
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
                                  (orig, 'Noise on normalized traces (original)', 'C1', 's', '--'),
                                  (prof, 'Noise on raw transients, with profile loss', 'C2', 'D', '-.')]:
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
        ax.set_xticks(x); ax.set_xticklabels(xl); ax.set_ylim(YMIN - 0.25, 1.08); ax.axhline(0, color='0.6', lw=0.6)
axs[0, 0].set_ylabel('$R^2$'); axs[1, 0].set_ylabel('$R^2$')
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.11, 1, 1)); fig.savefig(OUT)

for nm, d in (('rn_noprof', raw), ('noprof', orig), ('rn_base', prof)):
    print(nm, {q: [round(d['conditions'][q]['r2_mean'][i], 3) for i in idx] for q in snr_c + gain_c + ['snr30_gain2']})
