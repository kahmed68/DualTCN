"""11-16 (Option 1): Supplementary Table S14, R2 of four networks under every noise condition (mean over 3 seeds,
150,000 test samples) + linearized estimate. Sources: analysis/n1_u1/n1_u1_{rn_noprof,rn_misnp,rn_base_tm,noprof}.json
(TF32 off), analysis/j1/j1_snr.json. Prints the LaTeX table to stdout."""
import json, os
A = os.environ.get('DUALTCN_RUNS', '/work/hdd/bgte/kahmed2/DualTCN_runs') + '/analysis/'
NETS = [('rn_noprof', 'Main'), ('rn_misnp', 'FC'), ('rn_base_tm', 'Prof.'), ('noprof', 'Orig.')]
D = {k: json.load(open(f'{A}n1_u1/n1_u1_{k}.json'))['conditions'] for k, _ in NETS}
LIN = {L['snr_db']: L['r2_ceiling'] for L in json.load(open(A + 'j1/j1_snr.json'))['levels']}
CONDS = [('clean', 'Clean', None), ('snr40', '40\\,dB', 40), ('snr30', '30\\,dB', 30), ('snr20', '20\\,dB', 20),
         ('snr10', '10\\,dB$^\\dagger$', 10), ('gain1', 'Gain 1\\%', None), ('gain2', 'Gain 2\\%', None),
         ('gain5', 'Gain 5\\%', None), ('snr30_gain2', '30\\,dB + gain 2\\%', None)]
R = '\\color{red}'
f = lambda x: f'${x:.3f}$' if x < 0 else f'{x:.3f}'
out = ['\\begin{table*}[p]', '\\centering',
       '\\caption{\\rev{Noise robustness of four networks, $R^2$ averaged over three seeds on 150,000 test samples. '
       'Main: noise on the raw transients, no profile loss (main network); FC: the same with the forward-consistency loss; '
       'Prof.: noise on the raw transients with the profile loss; Orig.: noise on the normalized traces only (original noise model), '
       'no profile loss. Lin.: linearized estimate of $R^2$ from the sensitivity analysis at the same noise level (main text, '
       'Section~3.2). $^\\dagger$Outside the SNR range of the training data. Main, Prof., and Orig.\\ are shown in Figure~3 of the main text.}}',
       '\\label{tab:supp_noise}', '\\scriptsize', '\\begin{tabular}{llcccc}', '\\toprule',
       'Condition & Network & $\\sigma_1$ & $\\sigma_2$ & $d_1$ & $d_2$ \\\\']
for ck, cl, snr in CONDS:
    out.append('\\midrule')
    for i, (k, lab) in enumerate(NETS):
        v = D[k][ck]['r2_mean']
        out.append((R + cl if i == 0 else '') + ' & ' + ' & '.join(R + x for x in [lab] + [f(t) for t in v]) + ' \\\\')
    if snr:
        L = LIN[snr]
        out.append(' & ' + ' & '.join(R + x for x in ['Lin.'] + [f'{L[p]:.2f}' for p in ('sigma1', 'sigma2', 'd1', 'd2')]) + ' \\\\')
out += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
print('\n'.join(out))
