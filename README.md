# DualTCN: A Physics-Constrained Temporal Convolutional Network for Time-Domain Marine CSEM Inversion

**Khaled Ahmed and Ghada Omar**
Southern Illinois University Carbondale

Submitted to *IEEE Transactions on Geoscience and Remote Sensing*

---

## Overview

DualTCN is the first deep-learning framework for inverting time-domain marine controlled-source electromagnetic (MCSEM) transient data. It regresses four earth-model parameters from multi-receiver time-domain recordings and reconstructs the conductivity-depth profile through a differentiable physics decoder.

**Key results:**
- R-bar-squared = 0.877 on 150,000 test samples at 3.5 ms/sample inference (A100 GPU)
- Curriculum-based amplitude augmentation recovers robustness under +/-5% amplitude noise (R-bar-squared 0.363 -> 0.858)
- Three-layer extension (6 parameters) generalises with no architectural changes
- 7 published MCSEM survey site benchmarks with 12.1% MAPE using inverse-sigma2 weighting

## Repository Structure

```
DualTCN_repo/
├── src/                    # Source code
│   ├── model_dualtcn.py    # DualTCN architecture (379K params)
│   ├── model_baseline.py   # PCRN baseline
│   ├── model_3layer.py     # Three-layer extension (306K params)
│   ├── forward_model.py    # empymod-based forward model (2-layer)
│   ├── forward_model_3layer.py  # empymod forward model (3-layer)
│   ├── dataset.py          # Dataset loader (standard)
│   ├── dataset_ampaug.py   # Amplitude augmentation dataset
│   ├── dataset_colored.py  # Colored (1/f) noise dataset
│   ├── dataset_recvbias.py # Per-receiver bias dataset
│   ├── dataset_ampratio.py # Amplitude ratio dataset
│   ├── dataset_3layer.py   # Three-layer dataset
│   ├── train_dualtcn.py    # DualTCN training script
│   ├── train_ampaug.py     # Amplitude augmentation training
│   ├── train_colored.py    # Colored noise training
│   ├── train_recvbias.py   # Per-receiver bias training
│   ├── train_ampratio.py   # Amplitude ratio training
│   ├── train_weighted.py   # Inverse-sigma2 weighted training
│   ├── train_3layer.py     # Three-layer training
│   ├── train_utils.py      # Shared training utilities
│   ├── benchmark_*.py      # Conventional inversion benchmarks
│   ├── amplitude_noise_experiment.py
│   └── generate_*.py       # Dataset generation scripts
├── configs/                # Configuration files
│   ├── config_v4.py        # Main configuration (2-layer)
│   ├── config_v5.py        # Three-layer configuration
│   └── config_dualtcn.py   # DualTCN-specific config
├── weights/                # Trained model weights (Git LFS)
│   ├── best_model_p9d.pt           # DualTCN (unaugmented)
│   ├── best_model_p9d_ampaug.pt    # DualTCN-AmpAug
│   ├── best_model_p9d_colored.pt   # DualTCN-Colored
│   ├── best_model_p9d_ampratio.pt  # DualTCN-AmpRatio
│   ├── best_model_p9d_recvbias.pt  # DualTCN-RecvBias
│   ├── best_model_p9d_weighted.pt  # DualTCN-Weighted
│   └── best_model_v5_3layer.pt     # DualTCN-3Layer
├── paper/                  # Manuscript and figures
│   ├── DualTCN-TGRS.tex
│   ├── DualTCN-TGRS-supplement.tex
│   └── *.png / *.jpg
├── data/csv/               # Result CSV files
└── README.md
```

## Requirements

```
python >= 3.8
pytorch >= 2.0
numpy
scipy
empymod
matplotlib
```

## Quick Start

### Generate training data (1M samples)
```bash
cd src
python generate_large_dataset.py --n 1000000 --workers 8
```

### Train DualTCN
```bash
python train_dualtcn.py
```

### Train with curriculum amplitude augmentation
```bash
python train_ampaug.py
```

### Run field benchmark
```bash
python benchmark_field_models.py
```

## Model Variants

| Variant | File | Description |
|---------|------|-------------|
| DualTCN | `train_dualtcn.py` | Base model, best clean accuracy |
| DualTCN-AmpAug | `train_ampaug.py` | Curriculum amplitude augmentation |
| DualTCN-Colored | `train_colored.py` | 1/f waveform noise |
| DualTCN-AmpRatio | `train_ampratio.py` | Amplitude ratio input (7 channels) |
| DualTCN-Weighted | `train_weighted.py` | Inverse-sigma2 sample weighting |
| DualTCN-RecvBias | `train_recvbias.py` | Per-receiver bias augmentation |
| DualTCN-3Layer | `train_3layer.py` | Three-layer earth model (6 params) |

## Citation

```bibtex
@article{ahmed2026dualtcn,
  title={DualTCN: A Physics-Constrained Temporal Convolutional Network
         for Time-Domain Marine CSEM Inversion},
  author={Ahmed, Khaled and Omar, Ghada},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  note={Under review}
}
```

## Acknowledgements

This research used resources of the Argonne Leadership Computing Facility (ALCF, Polaris), supported by U.S. Department of Energy Contract DE-AC02-06CH11357.

## License

MIT License
