#!/usr/bin/env python3
"""Compare N1 per-run R2 (and 95% conformal_clean PICP) of a raw-noise run against the base arm (11-14)."""
import json, sys
D = '/work/hdd/bgte/kahmed2/DualTCN_runs/analysis/n1_u1/'
b = json.load(open(D + 'n1_u1_base.json'))
r = json.load(open(D + f'n1_u1_{sys.argv[1] if len(sys.argv) > 1 else "rn_base_s1"}.json'))
names = b['order']
print('R2 per parameter: base mean of', b['runs'], 'vs', r['runs'], '| d2 column also shows base min-max')
print(f"{'condition':14s} " + ' '.join(f'{n:>15s}' for n in names) + f" {'d2 base range':>16s} {'PICP95 d2 b/r':>14s}")
for c in b['conditions']:
    cb, cr = b['conditions'][c], r['conditions'][c]
    cells = [f"{cb['r2_mean'][i]:6.3f}->{(cr['r2_mean'] if len(r['runs']) > 1 else cr['r2_runs'][0])[i]:6.3f}" for i in range(len(names))]
    d2 = [x[3] for x in cb['r2_runs']]
    pb, pr = cb['uq']['0.95']['conformal_clean']['picp'][3], cr['uq']['0.95']['conformal_clean']['picp'][3]
    print(f'{c:14s} ' + ' '.join(f'{s:>15s}' for s in cells) + f" {min(d2):7.3f}..{max(d2):6.3f} {pb:6.3f}/{pr:6.3f}")
