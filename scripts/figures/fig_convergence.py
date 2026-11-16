"""Convergence history of the conventional inversions (R2.7, R4.4).
Reads analysis/c1/<run>/c1_per_sample.json written by scripts/analysis/c1_conventional.py (hist = objective per
function evaluation of the best start, first 20 samples). Plots median and interquartile range of the running
minimum of the objective over the 20 samples, for each method, plus the objective of the DualTCN prediction."""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); plt.style.use(os.path.join(HERE, 'dualtcn.mplstyle'))
C1 = os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/c1'
# 11-16 (Option 1): LM/L-BFGS-B from the full_* runs (independent of the network); WS rows and the DualTCN objective
# from the noprof_* runs (main network rn_noprof_s1). Same 200 samples (idx checked).
RUNS = [('full_n200_rn', 'noprof_n200_rn', r'(a) Clean data, $\beta=10^{-6}$'),
        ('full_n200_rn_snr40', 'noprof_n200_rn_snr40', r'(b) 40 dB, $\beta=10^{-6}$'),
        ('full_n200_rn_snr40_reg', 'noprof_n200_rn_snr40_reg', r'(c) 40 dB, $\beta=10^{-3}$')]
LAB = {'LM': 'LM', 'LBFGSB': 'L-BFGS-B', 'WS-LM': 'WS-LM', 'WS-LBFGSB': 'WS-L-BFGS-B'}
STY = {'LM': ('C0', '-'), 'LBFGSB': ('C1', '-'), 'WS-LM': ('C0', '--'), 'WS-LBFGSB': ('C1', '--')}
out = sys.argv[1]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=False)
stats = {}
for ax, (run_old, run, title) in zip(axs, RUNS):
    Jo = json.load(open(f'{C1}/{run_old}/c1_per_sample.json')); Jn = json.load(open(f'{C1}/{run}/c1_per_sample.json'))
    assert Jo['idx'] == Jn['idx']
    res = [{**{m: ro[m] for m in ('LM', 'LBFGSB') if m in ro}, **{m: rn[m] for m in ('WS-LM', 'WS-LBFGSB', 'DualTCN') if m in rn}}
           for ro, rn in zip(Jo['results'], Jn['results'])]
    methods = [m for m in LAB if m in res[0] and res[0][m].get('hist')]
    for m in methods:
        H = [np.minimum.accumulate(np.asarray(r[m]['hist'], float)) for r in res if r.get(m, {}).get('hist')]
        n = max(len(h) for h in H)
        A = np.array([np.pad(h, (0, n - len(h)), mode='edge') for h in H])
        med, q1, q3 = np.median(A, 0), np.percentile(A, 25, 0), np.percentile(A, 75, 0)
        x = np.arange(1, n + 1); c, ls = STY[m]
        ax.plot(x, med, color=c, ls=ls, label=LAB[m]); ax.fill_between(x, q1, q3, color=c, alpha=0.12, lw=0)
        e1 = [int(np.argmax(h <= h[-1] * 1.01)) + 1 for h in H]
        stats[(run, m)] = (len(H), n, float(med[-1])); print(run, m, 'median evals to within 1% of final:', np.median(e1))
    dt = [r['DualTCN']['obj'] for r in res[:len(H)] if 'DualTCN' in r and r['DualTCN'].get('obj') is not None]
    if dt:
        ax.axhline(np.median(dt), color='0.3', lw=1.0, ls=':', label='DualTCN')
        stats[(run, 'DualTCN')] = (len(dt), 0, float(np.median(dt)))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(title); ax.set_xlabel('Function evaluations')
axs[0].set_ylabel(r'Objective $\Phi$ (best so far)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out)
for k, v in stats.items(): print(k, 'n_samples=%d n_eval=%d final_median=%.3g' % v)
