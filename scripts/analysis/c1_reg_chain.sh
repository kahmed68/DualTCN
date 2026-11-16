#!/bin/bash
# 11-14 (Khaled go): regularized LM at 40 dB (R2.7, R4.4). After the unregularized 40 dB run (pid $1) ends:
# (1) choose lambda on 40 VALIDATION samples (split val, seed 2027, noise seed 4041), LM only, grid 1e-5..1e-1, by mean R2;
# (2) run LM and WS-LM with that lambda on the SAME 200 test samples and the same noise (seed 2026, noise seed 4040).
cd "$(dirname "$0")/../.."   # repository root
P=${PYTHON:-python}; export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUNS=${DUALTCN_RUNS:-/work/hdd/bgte/kahmed2/DualTCN_runs}
C=$RUNS/analysis/c1; NET=${NET:-$RUNS/e1/rn_base_s1}   # release weights: NET=weights/rn_base_s1
[ -n "${1:-}" ] && while kill -0 $1 2>/dev/null; do sleep 30; done
echo "chain start $(date +%T)"
for L in 1e-5 1e-4 1e-3 1e-2 1e-1; do
  nice -n 5 $P -u -W ignore scripts/analysis/c1_conventional.py --out $C/val_snr40_lam$L --n 40 --workers 8 --n-hist 0 --snr 40 \
    --noise-seed 4041 --split val --seed 2027 --methods LM --lam $L --net-run $NET > $C/val_snr40_lam$L.log 2>&1
  echo "lam $L: $(grep '^LM ' $C/val_snr40_lam$L.log | cut -c1-120)"
done
BEST=$($P - <<PY
import json
best=None
for L in ['1e-5','1e-4','1e-3','1e-2','1e-1']:
    s=json.load(open(f'$C/val_snr40_lam{L}/c1_summary.json'))['methods']['LM']['r2_mean']
    if best is None or s>best[1]: best=(L,s)
print(best[0])
PY
)
echo "chosen lambda $BEST $(date +%T)"
nice -n 5 $P -u -W ignore scripts/analysis/c1_conventional.py --out $C/full_n200_rn_snr40_reg --n 200 --workers 8 --n-hist 20 --snr 40 \
  --noise-seed 4040 --methods LM,WS-LM --lam $BEST --net-run $NET > $C/c1_login_snr40_reg.log 2>&1
grep -E "^DualTCN|^LM|^WS-LM|DONE|Traceback" $C/c1_login_snr40_reg.log | cut -c1-200
echo "chain end $(date +%T)"
