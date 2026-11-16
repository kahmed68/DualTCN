"""
evaluate_release.py -- evaluate the released networks on the test split and print the rows of Table 3.

Usage (from the repository root, after generating the dataset with src/generate_torch_dataset.py):

    PYTHONPATH=src python scripts/evaluate_release.py --data data/mcsem_dataset_v6.npz --weights weights

For each folder weights/<condition>_s<seed>/ it loads best.pt (batch-normalization statistics already
recalibrated), evaluates it on the test split (the last 15% of the one million samples, 150,000 samples, in
file order) with the evaluation code used for the paper (src/train_v6.py:evaluate), compares the result with
the stored test_metrics.json, and prints the three-seed mean of each condition (Table 3 of the paper).

The parameter R2 needs only the network. The data-misfit columns (trace RMSE, log-amplitude RMSE) simulate
every prediction with the forward operator; this takes about 1 min per network on one GPU and much longer
on a CPU. Use --no-misfit to skip them, or --n-test N to evaluate only the first N test samples (a quick
check; the numbers then differ slightly from Table 3).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
torch.backends.cudnn.allow_tf32 = False  # exact FP32 (TF32 convolutions change R2 by up to 0.01 on some GPUs)
torch.backends.cuda.matmul.allow_tf32 = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import train_v6 as tv  # noqa: E402
from config_v6 import (IN_CHANNELS, N_TIME, N_DEPTH, LATENT_DIM, DROPOUT, PROFILE_WEIGHT,  # noqa: E402
                       PARAM_WEIGHT, PARAM_WEIGHTS, TRAIN_SPLIT, VAL_SPLIT)
from config_p9d import DSF_AUX_WEIGHT  # noqa: E402
from model_p9d import LateTimePCRN  # noqa: E402,F401  (default architecture)
from train_utils import WeightedPCRNLoss  # noqa: E402

CONDITIONS = [
    ("base", "Supervised, noise on normalized traces"),
    ("mis", "Forward-consistent, noise on normalized traces"),
    ("rn_base", "Supervised, noise on raw transients"),
    ("rn_mis", "Forward-consistent, noise on raw transients"),
    ("noprof", "Supervised, no profile loss (decoder ablation)"),
    ("rn_noprof", "Supervised, raw transients, no profile loss (main network)"),
    ("rn_misnp", "Forward-consistent, raw transients, no profile loss"),
    ("rn_nolate", "Supervised, raw transients, profile loss, no late branch"),
    ("rn_nolatenp", "Supervised, raw transients, no profile loss, no late branch"),
]
KEYS = ["r2_s1", "r2_s2", "r2_d1", "r2_d2", "r2_mean", "trace_rmse", "logamp_rmse"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="mcsem_dataset_v6.npz (one million samples)")
    ap.add_argument("--weights", default="weights", help="folder with <condition>_s<seed>/best.pt")
    ap.add_argument("--runs", nargs="*", default=None, help="evaluate only these run folders, e.g. rn_mis_s1")
    ap.add_argument("--n-test", type=int, default=0, help="first N test samples only (quick check)")
    ap.add_argument("--no-misfit", action="store_true", help="skip the data-misfit columns")
    ap.add_argument("--out", default=None, help="write all results to this JSON file")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E, L, S, Pn = tv.load(a.data, dev)
    n = len(E)
    if n != 1_000_000:
        print(f"WARNING: {n} samples; Table 3 uses the one-million-sample dataset (seed 42).")
    n_tr, n_va = int(n * TRAIN_SPLIT), int(n * VAL_SPLIT)
    te_idx = torch.arange(n_tr + n_va, n, device=dev)
    if a.n_test:
        te_idx = te_idx[:a.n_test]
    print(f"device {dev}; test samples {len(te_idx)}")

    runs = a.runs or sorted(os.path.basename(os.path.dirname(p))
                            for p in glob.glob(os.path.join(a.weights, "*_s*", "best.pt")))
    if not runs:
        sys.exit(f"no <condition>_s<seed>/best.pt folders found in {a.weights}")
    results = {}
    for run in runs:
        folder = os.path.join(a.weights, run)
        args = json.load(open(os.path.join(folder, "args.json")))
        arch = tv.ARCHS[args.get("arch", "dualtcn")]          # rn_nolate_s*: dualtcn_nolate
        model = arch(in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH,
                     latent_dim=LATENT_DIM, dropout=DROPOUT).to(dev)
        model.load_state_dict(torch.load(os.path.join(folder, "best.pt"), map_location=dev, weights_only=True))
        crit = WeightedPCRNLoss(profile_w=args.get("profile_w", PROFILE_WEIGHT), param_w=PARAM_WEIGHT,
                                param_weights=PARAM_WEIGHTS)
        r = tv.evaluate(model, crit, E, L, S, Pn, te_idx, 512, not a.no_misfit, DSF_AUX_WEIGHT)
        stored = json.load(open(os.path.join(folder, "test_metrics.json")))["test"]
        diff = max(abs(r[k] - stored[k]) for k in KEYS if k in r)
        results[run] = dict(result=r, stored=stored, max_abs_diff=diff)
        print(f"{run:12s} R2 s1 {r['r2_s1']:.3f} s2 {r['r2_s2']:.3f} d1 {r['r2_d1']:.3f} d2 {r['r2_d2']:.3f} "
              f"mean {r['r2_mean']:.3f}"
              + ("" if a.no_misfit else f"  trace {r['trace_rmse']:.4f} logamp {r['logamp_rmse']:.3f}")
              + f"  | max |diff| vs stored test_metrics.json {diff:.1e}", flush=True)

    print("\nTable 3 (mean over seeds; R2 mean also +- sd over seeds)")
    hdr = f"{'Condition':48s} {'n':>2s} {'s1':>6s} {'s2':>6s} {'d1':>6s} {'d2':>6s} {'mean':>13s}"
    if not a.no_misfit:
        hdr += f" {'trace':>7s} {'logamp':>7s}"
    print(hdr)
    for cond, label in CONDITIONS:
        rs = [v["result"] for k, v in results.items() if k.rsplit("_s", 1)[0] == cond]
        if not rs:
            continue
        m = {k: float(np.mean([x[k] for x in rs])) for k in KEYS if k in rs[0]}
        sd = float(np.std([x["r2_mean"] for x in rs], ddof=1)) if len(rs) > 1 else float("nan")
        line = (f"{label:48s} {len(rs):2d} {m['r2_s1']:6.3f} {m['r2_s2']:6.3f} {m['r2_d1']:6.3f} "
                f"{m['r2_d2']:6.3f} {m['r2_mean']:6.3f}+-{sd:.3f}")
        if not a.no_misfit:
            line += f" {m['trace_rmse']:7.4f} {m['logamp_rmse']:7.3f}"
        print(line)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(n_test=len(te_idx), results=results), fh, indent=2)


if __name__ == "__main__":
    main()
