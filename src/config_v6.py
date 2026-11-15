"""
config_v6.py -- Path A data configuration (2026-11-11).

Changes from config_v4 (all other constants are re-exported unchanged):
  1. No source-velocity ("Doppler") factor. v0 is removed from the data model;
     at real tow speeds (about 1-2 m/s) the effect is negligible.
  2. Proper FFT frequency grid: f_k = k * DF, k = 0..N_TIME/2, DF = 1/(N_TIME*DT).
     With N_TIME = 128 and DT = 0.25 s, DF = 1/32 Hz and f_64 = 2.0 Hz (Nyquist).
     v4 placed linspace(0.05, 2.0, 64) into the FFT bins, shifting the whole
     spectrum by 0.05 Hz. The DC bin uses a near-DC proxy F_DC (as in supp. S1).
"""

import numpy as np

from config_v4 import *  # noqa: F401,F403  (ranges, receivers, model sizes)
from config_v4 import N_TIME

DT = 0.25                                   # [s]
DF = 1.0 / (N_TIME * DT)                    # 0.03125 Hz
F_DC = 1e-3                                 # near-DC proxy for bin 0 [Hz]
FREQS_V6 = np.arange(N_TIME // 2 + 1) * DF  # 65 bins, 0 .. 2.0 Hz
FREQS_V6[0] = F_DC
N_FREQ_V6 = FREQS_V6.size

DATA_PATH_V6 = "/work/hdd/bgte/kahmed2/DualTCN_data/mcsem_dataset_v6.npz"
