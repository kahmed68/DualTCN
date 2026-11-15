# DualTCN: forward-consistent deep-learning inversion of time-domain marine CSEM data

**Khaled Ahmed and Ghada Omar**, Southern Illinois University Carbondale

This repository contains the code, the trained networks, and a sample dataset for the revised paper
(Environmental Modelling & Software, revision 1). The code of the original submission is kept in the
same repository; the files of the revision are listed below.

## What is in the revision

- `src/forward_torch.py`: the one-dimensional marine CSEM forward operator (horizontal electric dipole,
  air / seawater / seafloor half-space) in PyTorch, in double precision, with exact gradients. It
  reproduces `empymod` (Key 201-point digital filter) to a relative error of 2.8e-13 in the frequency
  domain on 2,000 random models (`src/validate_forward_torch.py`, `results/step1_forward_torch/`).
- `src/config_v6.py`, `src/generate_torch_dataset.py`: the corrected data generator (exact Fourier grid
  f_k = k/32 Hz, no source-motion factor). It generates the one million samples of the paper in about
  three minutes on one A100 GPU.
- `src/train_v6.py`: training of the supervised and the forward-consistent networks, with noise on the
  raw transients (`--raw-noise`) or on the normalized traces only, and batch-normalization
  recalibration before every evaluation.
- `scripts/analysis/`: the analyses of the paper (sensitivity, noise robustness and uncertainty,
  shifted distributions, three-layer test, comparison with conventional inversion, timing).
- `scripts/figures/`: the figures of the paper, with one common style file (`dualtcn.mplstyle`).
- `scripts/field_al611/`, `src/field_al611/`: the forward fit of the public AL611 field data
  (PANGAEA, doi:10.1594/PANGAEA.989421), which must be downloaded separately.

## Installation

```bash
git clone https://github.com/Kahmed68/DualTCN.git
cd DualTCN
pip install -r requirements.txt      # Python 3.10+, PyTorch 2.x, NumPy, SciPy, empymod, matplotlib
```

## Quick test with the sample dataset (under a minute on one GPU)

Download `sample_v6_5000.npz` from the release assets (the first 5,000 samples of the full dataset) and run

```bash
PYTHONPATH=src python src/train_v6.py --data sample_v6_5000.npz --out runs/quick --seed 1 --epochs 3 \
  --misfit-w 1 --amp-w 0.1 --misfit-start 1 --misfit-ramp 1 --misfit-sub 64 --val-misfit-n 750 --raw-noise
```

## Reproducing the paper

1. Generate the one million samples (seed 42):

   ```bash
   PYTHONPATH=src python src/generate_torch_dataset.py --n 1000000 --out data/mcsem_dataset_v6.npz --device cuda
   ```

2. Train the four conditions of Table 3 with seeds 1, 2, and 3 (100 epochs; 1.2 GPU-hours for the
   supervised and 4.35 GPU-hours for the forward-consistent network on one A100):

   ```bash
   # supervised (W=0) or forward-consistent (W=1); add --raw-noise for the noise on the raw transients
   PYTHONPATH=src python src/train_v6.py --data data/mcsem_dataset_v6.npz --out runs/rn_mis_s1 --seed 1 \
     --epochs 100 --misfit-w 1 --amp-w 0.1 --misfit-start 5 --misfit-ramp 10 --misfit-sub 64 \
     --val-misfit-n 20000 --raw-noise
   ```

   The decoder ablation of Section 3.1 adds `--profile-w 0` to the supervised command.
   Each run writes `best.pt`, `log.csv`, and `test_metrics.json` to its output folder.

3. Run the analyses in `scripts/analysis/` on the trained networks (each script lists its inputs in its
   header). The scripts and `src/config_v6.py` contain the data and output folders of our cluster;
   set them to your own folders, or pass `--data` and `--out` where the script accepts them.

## Trained networks

The release assets contain `weights/<condition>_s<seed>/best.pt` (PyTorch state dict of
`src/model_p9d.py:LateTimePCRN`, 379,370 parameters), with `args.json` and `test_metrics.json` of each run:

| Folder | Training noise | Loss |
|---|---|---|
| `base_s*` | normalized traces only | supervised |
| `mis_s*` | normalized traces only | forward-consistent |
| `rn_base_s*` | raw transients | supervised |
| `rn_mis_s*` | raw transients | forward-consistent |
| `noprof_s*` | normalized traces only | supervised, no profile loss (decoder ablation) |

The saved batch-normalization statistics are the recalibrated ones, so the networks can be evaluated directly.
`SHA256SUMS` lists the checksums of all release files.

## Citation

Please cite the paper and release v1.1-r1 of this repository (see `CITATION.cff`).

## License

MIT (see `LICENSE`).
