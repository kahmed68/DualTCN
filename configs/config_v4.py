"""
config_v4.py — Multi-Receiver PCRN (v4).

Root-cause fix from v3 analysis:
  A single receiver at r=20m lies in the near-field of the dipole source.
  At this offset the E-field is dominated by the direct wavefield and carries
  almost NO information about σ₂ (seafloor) or d₁/d₂ (source geometry).
  In real MCSEM surveys, seafloor contrast is only detectable at offsets
  r ~ d_seafloor (60–200 m).  Adding receivers at r = 50, 100, 200 m provides
  the information the network needs to separate all 4 parameters.

Changes from v3:
  - RECEIVER_OFFSETS: 4 offsets instead of 1
  - IN_CHANNELS: 8 (2 channels × 4 receivers: waveform + log-amplitude)
  - Weighted param loss: σ₂, d₁, d₂ get 3× weight (harder to learn)
  - Slightly larger LATENT_DIM (256) to handle richer input
  - Dataset regenerated (multi-receiver forward model)
"""
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7
P0  = 8.7 * np.pi * 1e-29

# ── Multi-receiver geometry ───────────────────────────────────────────────────
RECEIVER_OFFSETS = [20.0, 50.0, 100.0, 200.0]   # horizontal offsets [m]
Z_OBS            = 20.0                           # all receivers at same depth

# ── Frequency / time ──────────────────────────────────────────────────────────
F_MIN  = 0.05
F_MAX  = 2.0
N_FREQ = 64
N_TIME = 128

# ── Earth-model parameter ranges ──────────────────────────────────────────────
SIGMA1_LOG_RANGE = (-1.0,  0.7)
SIGMA2_LOG_RANGE = (-3.0,  0.0)
D1_RANGE         = (50.0, 150.0)
D2_RANGE         = (10.0,  50.0)
V0_RANGE         = (0.0,  100.0)

# ── Output profile ────────────────────────────────────────────────────────────
N_DEPTH       = 64
Z_MAX         = 250.0
SOFT_STEP_TAU = 2.0

# ── Dataset ───────────────────────────────────────────────────────────────────
# generate_large_dataset.py writes this file with N samples (default 1M).
# N_SAMPLES_V4 is only used by dataset_v4.generate_dataset_v4() (the slow
# single-process fallback).  The parallel generator uses --n on the CLI.
N_SAMPLES_V4 = 1_000_000
RANDOM_SEED  = 42
DATA_PATH_V4 = "mcsem_dataset_v4.npz"

# ── Input channels ────────────────────────────────────────────────────────────
N_RECEIVERS = len(RECEIVER_OFFSETS)
IN_CHANNELS = N_RECEIVERS * 2   # waveform + log-amplitude per receiver  → 8

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 256     # larger batch affordable with 1M samples
LR            = 5e-4
N_EPOCHS      = 100     # 1M samples × 70% train = 700K → 1 epoch ≈ 2734 steps
WARMUP_EPOCHS = 5
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15

# ── Model ─────────────────────────────────────────────────────────────────────
LATENT_DIM = 256      # larger — richer multi-receiver input
DROPOUT    = 0.30

# ── Weighted parameter loss ───────────────────────────────────────────────────
# v3 showed σ₂ (R²=0.023) and d₁ (R²=0.025) are barely learned.
# Weight them 3× higher so the network prioritises their gradients.
PARAM_WEIGHTS  = [1.0, 3.0, 3.0, 2.0]  # [σ₁, σ₂, d₁, d₂]
PROFILE_WEIGHT = 1.0
PARAM_WEIGHT   = 2.0

# ── Parameter bounds (for normalisation) ─────────────────────────────────────
LOG_SIGMA1_MIN, LOG_SIGMA1_MAX = SIGMA1_LOG_RANGE
LOG_SIGMA2_MIN, LOG_SIGMA2_MAX = SIGMA2_LOG_RANGE
LOG_D1_MIN, LOG_D1_MAX         = np.log10(D1_RANGE[0]), np.log10(D1_RANGE[1])
LOG_D2_MIN, LOG_D2_MAX         = np.log10(D2_RANGE[0]), np.log10(D2_RANGE[1])

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH_V4 = "best_model_v4.pt"
