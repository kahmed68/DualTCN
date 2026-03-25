"""
config_v3.py — Physics-Constrained Parameter Regression Network (PCRN) config.

Key change from v2:
  The network now predicts 4 physical parameters (log_σ1, log_σ2, log_d1, log_d2)
  and reconstructs σ(z) analytically via a differentiable soft step function.
  Output dimensionality: 64 depth points → 4 parameters.
"""
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7
P0  = 8.7 * np.pi * 1e-29

# ── Geometry ──────────────────────────────────────────────────────────────────
R_OFFSET = 20.0
Z_OBS    = 20.0

# ── Frequency / time ──────────────────────────────────────────────────────────
F_MIN  = 0.05
F_MAX  = 2.0
N_FREQ = 64
N_TIME = 128

# ── Earth-model parameter ranges ──────────────────────────────────────────────
SIGMA1_LOG_RANGE = (-1.0,  0.7)   # log10(σ₁) → [0.10, 5.0]  S/m
SIGMA2_LOG_RANGE = (-3.0,  0.0)   # log10(σ₂) → [0.001, 1.0] S/m
D1_RANGE         = (50.0, 150.0)  # depth of source from surface [m]
D2_RANGE         = (10.0,  50.0)  # height of source above seafloor [m]
V0_RANGE         = (0.0,  100.0)  # source speed [m/s]

# ── Output profile ────────────────────────────────────────────────────────────
N_DEPTH  = 64
Z_MAX    = 250.0

# ── Soft-step sharpness ───────────────────────────────────────────────────────
# Smaller tau → sharper transition at the seafloor interface.
# tau = 2.0 m gives a transition width of ~8 m (4τ), fine enough for 64 points.
SOFT_STEP_TAU = 2.0

# ── Dataset ───────────────────────────────────────────────────────────────────
# Reuse v2 dataset — same forward model, same parameters.
N_SAMPLES_V3 = 10_000
RANDOM_SEED  = 42
DATA_PATH_V3 = "mcsem_dataset_v2.npz"   # already generated

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
LR            = 5e-4
N_EPOCHS      = 200
WARMUP_EPOCHS = 15

TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15

# ── Model ─────────────────────────────────────────────────────────────────────
IN_CHANNELS  = 2
LATENT_DIM   = 128   # smaller bottleneck — we only need to predict 4 numbers
DROPOUT      = 0.30  # stronger regularisation (was 0.20)

# ── Loss weights ──────────────────────────────────────────────────────────────
# Total loss = PROFILE_W * profile_loss + PARAM_W * param_loss
PROFILE_WEIGHT = 1.0    # MSE on reconstructed log10(σ) profile
PARAM_WEIGHT   = 2.0    # MSE on the 4 normalised physical parameters (stronger)

# ── Output parameter normalisation bounds (for the regression head) ───────────
# Network outputs values in [0,1] which are decoded to physical ranges.
LOG_SIGMA1_MIN, LOG_SIGMA1_MAX = SIGMA1_LOG_RANGE
LOG_SIGMA2_MIN, LOG_SIGMA2_MAX = SIGMA2_LOG_RANGE
LOG_D1_MIN, LOG_D1_MAX         = np.log10(D1_RANGE[0]), np.log10(D1_RANGE[1])
LOG_D2_MIN, LOG_D2_MAX         = np.log10(D2_RANGE[0]), np.log10(D2_RANGE[1])

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH_V3 = "best_model_v3.pt"
