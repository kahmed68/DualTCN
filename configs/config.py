"""
config.py — Global parameters for the MCSEM Encoder-Decoder project.

Reference:
  Ghada M. Sami, "Motion of a Horizontal Electric Dipole in a Conducting Medium",
  IJECCE, 2015.
"""
import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7        # vacuum permeability [H/m]
P0  = 8.7 * np.pi * 1e-29     # dipole moment magnitude [C·m]

# ── Fixed survey geometry (Paper 3 baseline values) ──────────────────────────
R_OFFSET = 20.0    # horizontal source-receiver offset (x - x0) [m]
Z_OBS    = 20.0    # receiver depth below sea surface             [m]

# ── Frequency / time sampling ─────────────────────────────────────────────────
# N_FREQ must be a power-of-2 for efficient IRFFT; N_TIME = 2 * N_FREQ.
F_MIN   = 0.05     # minimum frequency [Hz]
F_MAX   = 2.0      # maximum frequency [Hz]
N_FREQ  = 64       # positive-frequency samples (= half FFT size)
N_TIME  = 128      # time-series length output by forward model (= 2*N_FREQ)
# Time axis: t in [0, N_TIME/F_MAX] seconds

# ── Earth-model parameter ranges ──────────────────────────────────────────────
# log10 values are sampled uniformly, giving log-uniform distribution in σ.
SIGMA1_LOG_RANGE = (-1.0,  0.7)   # log10(σ₁): [0.10, 5.0] S/m  → sea water
SIGMA2_LOG_RANGE = (-3.0,  0.0)   # log10(σ₂): [0.001,1.0] S/m  → seafloor
D1_RANGE         = (50.0, 150.0)  # depth of dipole from sea surface      [m]
D2_RANGE         = (10.0,  50.0)  # height of dipole above seafloor       [m]
V0_RANGE         = (0.0,  100.0)  # source speed                          [m/s]

# ── Conductivity profile (decoder output) ────────────────────────────────────
N_DEPTH  = 64       # number of depth-grid points
Z_MAX    = 250.0    # maximum depth of the output profile         [m]

# ── Dataset ───────────────────────────────────────────────────────────────────
N_SAMPLES   = 4000
RANDOM_SEED = 42
DATA_PATH   = "mcsem_dataset.npz"

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE   = 32
LR           = 1e-3
N_EPOCHS     = 100
TRAIN_SPLIT  = 0.70   # 70 % train
VAL_SPLIT    = 0.15   # 15 % validation  → 15 % test (remainder)
LATENT_DIM   = 128    # bottleneck dimension
MODEL_PATH   = "best_model.pt"
