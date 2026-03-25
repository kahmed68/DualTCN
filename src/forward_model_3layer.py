"""
forward_model_v5.py — Three-layer MCSEM forward model using empymod.

Earth model (z positive downward from sea surface):
  Layer 0  : air                  (z < 0)           — ρ_air = 1e23 Ω·m
  Layer 1  : seawater             (0 < z < d_sf)    — ρ₁ = 1/σ₁
  Layer 2  : resistive layer      (d_sf < z < d_sf+h) — ρ₂ = 1/σ₂
  Layer 3  : basement             (z > d_sf + h)    — ρ₃ = 1/σ₃

  d_sf = d₁ + d₂  (seafloor / top of resistive layer)
  h = thickness of resistive layer

Source: HED at depth d₁
Receivers: inline, depth z_obs, offsets RECEIVER_OFFSETS
"""

import numpy as np
import empymod

from config_v4 import (
    RECEIVER_OFFSETS, Z_OBS, F_MIN, F_MAX, N_FREQ, N_TIME, MU0,
)


def compute_3layer_timeseries(sigma1, sigma2, sigma3, d1, d2, h,
                               offsets=RECEIVER_OFFSETS, z_obs=Z_OBS):
    """
    Compute E-field time series for a three-layer earth model.

    Parameters
    ----------
    sigma1 : float  [S/m]   seawater conductivity
    sigma2 : float  [S/m]   resistive layer conductivity
    sigma3 : float  [S/m]   basement conductivity
    d1     : float  [m]     source depth below sea surface
    d2     : float  [m]     source to top of resistive layer
    h      : float  [m]     thickness of resistive layer
    """
    freqs = np.linspace(F_MIN, F_MAX, N_FREQ)

    d_sf = float(d1 + d2)          # top of resistive layer
    d_bot = float(d_sf + h)        # bottom of resistive layer

    rho1 = 1.0 / max(sigma1, 1e-9)
    rho2 = 1.0 / max(sigma2, 1e-9)
    rho3 = 1.0 / max(sigma3, 1e-9)

    # empymod: 4-layer model (air, seawater, resistive layer, basement)
    depth_model = np.array([0.0, d_sf, d_bot])
    res_model = np.array([1e23, rho1, rho2, rho3])

    src = [0.0, 0.0, float(d1)]

    results = {}
    for r in offsets:
        rec = [float(r), 0.0, float(z_obs)]

        try:
            E_f = empymod.dipole(
                src=src, rec=rec, depth=depth_model, res=res_model,
                freqtime=freqs, ab=11, verb=0,
            )
        except Exception:
            E_f = np.zeros(N_FREQ, dtype=complex)

        E_t = np.fft.irfft(E_f, n=N_TIME)
        peak = np.max(np.abs(E_t))
        log_amp = float(np.log10(peak + 1e-40))

        if peak < 1e-35:
            rng = np.random.default_rng()
            E_t = rng.normal(0, 1e-36, N_TIME)
            peak = np.max(np.abs(E_t))

        E_norm = (E_t / (peak + 1e-60)).astype(np.float32)
        results[r] = (E_norm, log_amp)

    return results


def build_3layer_profile(sigma1, sigma2, sigma3, d1, d2, h,
                          n_depth=64, z_max=250.0, tau=2.0):
    """Build log10-conductivity depth profile for a three-layer model."""
    z = np.linspace(0, z_max, n_depth)
    d_sf = d1 + d2
    d_bot = d_sf + h

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # σ(z) = σ₁ + (σ₂-σ₁)·sig((z-d_sf)/τ) + (σ₃-σ₂)·sig((z-d_bot)/τ)
    profile = (sigma1
               + (sigma2 - sigma1) * sigmoid((z - d_sf) / tau)
               + (sigma3 - sigma2) * sigmoid((z - d_bot) / tau))

    return np.log10(profile + 1e-12).astype(np.float32)
