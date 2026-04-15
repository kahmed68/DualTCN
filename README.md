# DualTCN: A Physics-Constrained Temporal Convolutional Network for Time-Domain Marine CSEM Inversion

**Khaled Ahmed and Ghada Omar**
Southern Illinois University Carbondale

---

## Overview

DualTCN is the first deep-learning framework for inverting time-domain marine controlled-source electromagnetic (MCSEM) transient data. It regresses four earth-model parameters from multi-receiver time-domain recordings and reconstructs the conductivity-depth profile through a differentiable physics decoder.

**Key results:**
- Mean R-squared = 0.877 on 150,000 test samples at 3.5 ms/sample inference (A100 GPU)
- Curriculum-based amplitude augmentation recovers robustness under +/-5% amplitude noise (R-bar-squared 0.363 -> 0.858)
- Three-layer extension (6 parameters) generalises with no architectural changes
- 7 published MCSEM survey site benchmarks with 12.1% MAPE using inverse-sigma2 weighting
- 4-method UQ comparison (MC-Dropout, temperature scaling, split conformal prediction, deep ensemble)

## Repository Structure

```
DualTCN/
├── src/                        # All source code
│   ├── model_dualtcn.py        # DualTCN architecture (379K params)
│   ├── model_baseline.py       # PCRN baseline (638K params)
│   ├── model_3layer.py         # Three-layer extension (306K params)
│   ├── model_v3.py             # Shared building blocks (DilatedResBlock, reconstruct_profile)
│   ├── model_v4.py             # Parameter denormalisation utilities
│   ├── model_v5.py             # DualTCN-3Layer model
│   ├── model_p5a.py            # TCN-only encoder (used by DualTCN)
│   ├── model_p9d.py            # LateTimePCRN (DualTCN model class)
│   ├── forward_model.py        # empymod-based forward model (2-layer, wrapper)
│   ├── forward_model_v2.py     # Manual Hankel-transform forward model
│   ├── forward_model_v4.py     # empymod multi-receiver forward model (2-layer)
│   ├── forward_model_v5.py     # empymod forward model (3-layer)
│   ├── dataset.py              # Dataset generation wrapper
│   ├── dataset_v4.py           # Standard dataset loader
│   ├── dataset_ampaug.py       # Amplitude augmentation dataset
│   ├── dataset_v4_ampaug.py    # Amplitude augmentation dataloader
│   ├── dataset_colored.py      # Colored (1/f) noise dataset
│   ├── dataset_v4_colored.py   # Colored noise dataloader
│   ├── dataset_recvbias.py     # Per-receiver bias dataset
│   ├── dataset_v4_recvbias.py  # Per-receiver bias dataloader
│   ├── dataset_ampratio.py     # Amplitude ratio dataset
│   ├── dataset_v4_ampratio.py  # Amplitude ratio dataloader
│   ├── dataset_3layer.py       # Three-layer dataset
│   ├── dataset_v5.py           # Three-layer dataloader
│   ├── config.py               # Base configuration
│   ├── config_v3.py            # v3 configuration (shared constants)
│   ├── config_v4.py            # Main configuration (2-layer)
│   ├── config_v5.py            # Three-layer configuration
│   ├── config_p9d.py           # DualTCN-specific config
│   ├── train_dualtcn.py        # DualTCN training script
│   ├── train_p9d.py            # Core training loop for DualTCN variants
│   ├── train_ampaug.py         # Amplitude augmentation training
│   ├── train_colored.py        # Colored noise training
│   ├── train_recvbias.py       # Per-receiver bias training
│   ├── train_ampratio.py       # Amplitude ratio training
│   ├── train_weighted.py       # Inverse-sigma2 weighted training
│   ├── train_3layer.py         # Three-layer training
│   ├── train_utils.py          # Shared training utilities (loss, metrics, logging)
│   ├── generate_large_dataset.py   # Generate 1M training samples
│   ├── generate_3layer_dataset.py  # Generate 3-layer training samples
│   ├── benchmark_conventional.py   # Conventional inversion benchmark
│   ├── benchmark_multistart.py     # Multi-start LM/L-BFGS-B benchmark
│   ├── benchmark_warmstart.py      # DualTCN warm-start benchmark
│   ├── benchmark_occam.py          # Occam-style regularised inversion
│   ├── benchmark_field_models.py   # Published survey site benchmark
│   ├── amplitude_noise_experiment.py           # Amplitude noise robustness
│   ├── structured_amplitude_experiment.py      # Structured amplitude tests
│   └── tau_sensitivity_experiment.py           # Soft-step tau sweep
├── configs/                    # Configuration files (also in src/ for imports)
│   ├── config.py
│   ├── config_dualtcn.py
│   ├── config_v2.py
│   ├── config_v3.py
│   ├── config_v4.py
│   └── config_v5.py
├── weights/                    # Trained model weights
│   ├── best_model_p9d.pt           # DualTCN (unaugmented)
│   ├── best_model_p9d_ampaug.pt    # DualTCN-AmpAug
│   ├── best_model_p9d_colored.pt   # DualTCN-Colored
│   ├── best_model_p9d_ampratio.pt  # DualTCN-AmpRatio
│   ├── best_model_p9d_recvbias.pt  # DualTCN-RecvBias
│   ├── best_model_p9d_weighted.pt  # DualTCN-Weighted
│   └── best_model_v5_3layer.pt     # DualTCN-3Layer
├── data/csv/                   # Result CSV files for all experiments
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
└── README.md
```

## Requirements

```
python >= 3.8
pytorch >= 2.0
numpy >= 1.21
scipy >= 1.7
empymod >= 2.2
matplotlib >= 3.5
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

All scripts should be run from the `src/` directory:

```bash
cd src
```

### 1. Generate training data (1M samples)
```bash
python generate_large_dataset.py --n 1000000 --workers 8
```
This creates `mcsem_dataset_v4.npz` (~4 GB) containing 1M synthetic MCSEM samples generated with empymod.

### 2. Train DualTCN
```bash
python train_dualtcn.py
```
Training takes approximately 3-4 GPU-hours on an NVIDIA A100 40GB.

### 3. Train augmented variants
```bash
python train_ampaug.py      # Curriculum amplitude augmentation
python train_colored.py     # 1/f colored noise augmentation
python train_weighted.py    # Inverse-sigma2 sample weighting
python train_ampratio.py    # Amplitude ratio input representation
python train_recvbias.py    # Per-receiver bias augmentation
```

### 4. Run benchmarks
```bash
python benchmark_field_models.py      # Published survey site benchmark
python benchmark_conventional.py      # Conventional inversion comparison
python benchmark_warmstart.py         # DualTCN warm-start hybrid
```

### 5. Reproduce noise experiments
```bash
python amplitude_noise_experiment.py  # Random + systematic amplitude noise
```

## Pre-trained Weights

All seven model variants are provided in `weights/`. To load and run inference:

```python
import torch
from model_p9d import LateTimePCRN

model = LateTimePCRN()
model.load_state_dict(torch.load("weights/best_model_p9d.pt", map_location="cpu"))
model.eval()

# x: (batch, 8, 128) — 4 receivers x 2 channels x 128 time samples
with torch.no_grad():
    profile, p_norm, p_phys = model(x)
    # p_phys: [sigma1, sigma2, d1, d2] in physical units
```

## Model Variants

| Variant | Training Script | Weight File | Description |
|---------|----------------|-------------|-------------|
| DualTCN | `train_dualtcn.py` | `best_model_p9d.pt` | Base model, best clean accuracy |
| DualTCN-AmpAug | `train_ampaug.py` | `best_model_p9d_ampaug.pt` | Curriculum amplitude augmentation |
| DualTCN-Colored | `train_colored.py` | `best_model_p9d_colored.pt` | 1/f waveform noise |
| DualTCN-AmpRatio | `train_ampratio.py` | `best_model_p9d_ampratio.pt` | Amplitude ratio input (7 channels) |
| DualTCN-Weighted | `train_weighted.py` | `best_model_p9d_weighted.pt` | Inverse-sigma2 sample weighting |
| DualTCN-RecvBias | `train_recvbias.py` | `best_model_p9d_recvbias.pt` | Per-receiver bias augmentation |
| DualTCN-3Layer | `train_3layer.py` | `best_model_v5_3layer.pt` | Three-layer earth model (6 params) |

## Data

Training data is generated synthetically using [empymod](https://empymod.emsig.xyz). The `data/csv/` directory contains all experimental result data referenced in the paper (25 CSV files covering ablation, noise, benchmark, UQ, and field validation results).

## Citation

```bibtex
@article{ahmed2026dualtcn,
  title={DualTCN: A Physics-Constrained Temporal Convolutional Network
         for Time-Domain Marine CSEM Inversion},
  author={Ahmed, Khaled and Omar, Ghada},
  journal={Computers \& Geosciences},
  year={2026},
  note={Under review}
}
```

## Acknowledgements

This research used resources of the Argonne Leadership Computing Facility (ALCF, Polaris), supported by U.S. Department of Energy Contract DE-AC02-06CH11357.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
