"""
config_v5.py — Three-layer MCSEM inversion configuration.

Earth model: Air | Seawater (σ₁) | Resistive layer (σ₂, thickness h) | Basement (σ₃)
Six parameters: σ₁, σ₂, σ₃, d₁ (source depth), d₂ (source-to-layer-top), h (layer thickness)

Represents: hydrocarbon reservoir or gas-hydrate layer sandwiched
between seawater and conductive basement.
"""
import numpy as np
from config_v4 import (
    MU0, P0, RECEIVER_OFFSETS, Z_OBS,
    F_MIN, F_MAX, N_FREQ, N_TIME,
    N_DEPTH, Z_MAX, SOFT_STEP_TAU,
    BATCH_SIZE, LR, WARMUP_EPOCHS, DROPOUT,
    RANDOM_SEED,
)

# ── Earth-model parameter ranges (three-layer) ──────────────────────────────
# σ₁: seawater conductivity
SIGMA1_LOG_RANGE_V5 = (-1.0, 0.7)      # 0.1–5.0 S/m (same as v4)

# σ₂: resistive layer conductivity (reservoir/hydrate)
SIGMA2_LOG_RANGE_V5 = (-3.0, -0.5)     # 0.001–0.316 S/m (resistive)

# σ₃: basement conductivity (conductive)
SIGMA3_LOG_RANGE_V5 = (-1.5, 0.5)      # 0.032–3.16 S/m

# d₁: source depth below sea surface
D1_RANGE_V5 = (50.0, 150.0)            # 50–150 m (same as v4)

# d₂: source to top of resistive layer
D2_RANGE_V5 = (10.0, 50.0)             # 10–50 m (same as v4)

# h: thickness of resistive layer
H_RANGE_V5 = (5.0, 50.0)              # 5–50 m

# ── Training ─────────────────────────────────────────────────────────────────
N_PARAMS_V5   = 6
IN_CHANNELS   = 8  # same 8-channel input as v4
N_EPOCHS      = 100
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
LATENT_DIM    = 256
N_SAMPLES_V5  = 1_000_000  # same scale as v4

# ── Loss weights ─────────────────────────────────────────────────────────────
# σ₂ and h are hardest to learn; σ₃ is moderately hard
PARAM_WEIGHTS_V5 = [1.0, 3.0, 2.0, 1.0, 3.0, 2.0]  # [σ₁, σ₂, σ₃, d₁, d₂, h]
PROFILE_WEIGHT   = 1.0
PARAM_WEIGHT     = 2.0

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH_V5  = "mcsem_dataset_v5_3layer.npz"
MODEL_PATH_V5 = "best_model_v5_3layer.pt"

# ── Parameter bounds (for normalisation) ─────────────────────────────────────
LOG_SIGMA1_MIN_V5, LOG_SIGMA1_MAX_V5 = SIGMA1_LOG_RANGE_V5
LOG_SIGMA2_MIN_V5, LOG_SIGMA2_MAX_V5 = SIGMA2_LOG_RANGE_V5
LOG_SIGMA3_MIN_V5, LOG_SIGMA3_MAX_V5 = SIGMA3_LOG_RANGE_V5
LOG_D1_MIN_V5, LOG_D1_MAX_V5 = np.log10(D1_RANGE_V5[0]), np.log10(D1_RANGE_V5[1])
LOG_D2_MIN_V5, LOG_D2_MAX_V5 = np.log10(D2_RANGE_V5[0]), np.log10(D2_RANGE_V5[1])
LOG_H_MIN_V5,  LOG_H_MAX_V5  = np.log10(H_RANGE_V5[0]),  np.log10(H_RANGE_V5[1])
