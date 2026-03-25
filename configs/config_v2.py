"""
config_v2.py — Updated hyperparameters addressing all v1 failure modes.

Changes from config.py:
  - Dataset enlarged to 10 000 samples (v1 had 4 000)
  - 2-channel input (waveform + log-amplitude)
  - Stronger regularisation: dropout, weight decay
  - Warmup + cosine LR scheduler
  - Smooth loss = MSE + TV (total variation) penalty
  - Added NOISE_STD for training augmentation
"""
import numpy as np

# ── Physical constants (unchanged) ───────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7
P0  = 8.7 * np.pi * 1e-29

# ── Geometry (unchanged) ─────────────────────────────────────────────────────
R_OFFSET = 20.0
Z_OBS    = 20.0

# ── Frequency / time (unchanged) ─────────────────────────────────────────────
F_MIN  = 0.05
F_MAX  = 2.0
N_FREQ = 64
N_TIME = 128

# ── Earth-model parameter ranges (unchanged) ─────────────────────────────────
SIGMA1_LOG_RANGE = (-1.0,  0.7)
SIGMA2_LOG_RANGE = (-3.0,  0.0)
D1_RANGE         = (50.0, 150.0)
D2_RANGE         = (10.0,  50.0)
V0_RANGE         = (0.0,  100.0)

# ── Conductivity profile output (unchanged) ──────────────────────────────────
N_DEPTH  = 64
Z_MAX    = 250.0

# ── Dataset — ENLARGED ───────────────────────────────────────────────────────
N_SAMPLES_V2 = 10_000          # was 4 000; larger → less overfitting
RANDOM_SEED  = 42
DATA_PATH_V2 = "mcsem_dataset_v2.npz"

# ── Training augmentation ─────────────────────────────────────────────────────
NOISE_STD = 0.05               # max std of additive Gaussian noise on E traces

# ── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE    = 64             # larger batch → more stable gradient estimates
LR            = 3e-4
N_EPOCHS      = 150
WARMUP_EPOCHS = 10             # linear LR warmup to avoid early instability
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15

# ── Model ─────────────────────────────────────────────────────────────────────
IN_CHANNELS  = 2               # channel 0: waveform, channel 1: log-amplitude
LATENT_DIM   = 256             # larger bottleneck (was 128)
DROPOUT      = 0.2             # dropout rate in encoder/decoder

# ── Loss weights ──────────────────────────────────────────────────────────────
TV_WEIGHT    = 0.05            # total-variation penalty weight (smoothness)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH_V2 = "best_model_v2.pt"
