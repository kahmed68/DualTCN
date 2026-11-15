"""
forward_torch.py -- Differentiable PyTorch port of forward_model_v4.py.

Reproduces empymod.dipole(ab=11, xdirect=False, ht='dlf', Key-201 filter,
pts_per_dec=0) for the fixed DualTCN geometry:

    layer 0 : air       z < 0            res = 1e23 Ohm.m
    layer 1 : seawater  0 < z < d1 + d2  sigma1
    layer 2 : seafloor  z > d1 + d2      sigma2

Source: x-directed HED at z = d1. Receivers: inline (angle 0) at z = Z_OBS.
Source and receivers are both in layer 1 (Z_OBS < d1), which is the only
case implemented here (asserted at run time).

The kernel follows empymod.kernel.greenfct / reflections / fields / wavenumber
specialised to lsrc = lrec = 1 in a three-layer, isotropic model
(Hunziker et al. 2015). The Hankel transform is the standard DLF of Key (2012):
    E(r) = [ (PJ1 . j1)/r + PJ0b . j0 + PJ0 . j0 ] / r,   lambda = base / r.

All computation is in complex128. Gradients flow to sigma1, sigma2, d1, d2.
"""

import math
import os

import numpy as np
import torch

MU0 = 4e-7 * math.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 * C0)
RES_AIR = 1e23

_FILT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "key_201_2009.npz")
_FILT_CACHE = {}


def _filter(device):
    key = str(device)
    if key not in _FILT_CACHE:
        f = np.load(_FILT_PATH)
        _FILT_CACHE[key] = tuple(
            torch.as_tensor(f[k], dtype=torch.float64, device=device)
            for k in ("base", "j0", "j1"))
    return _FILT_CACHE[key]


def freq_response(sigma1, sigma2, d1, d2, offsets, z_obs, freqs):
    """
    Frequency-domain inline Ex for a unit x-directed HED.

    sigma1, sigma2, d1, d2 : (B,) float64 tensors [S/m, S/m, m, m]
    offsets                : sequence of R receiver offsets [m]
    z_obs                  : receiver depth [m] (must satisfy z_obs < d1)
    freqs                  : sequence of F frequencies [Hz]

    Returns (B, R, F) complex128.
    """
    dev = sigma1.device
    base, j0, j1 = _filter(dev)
    off = torch.as_tensor(offsets, dtype=torch.float64, device=dev)       # (R,)
    f = torch.as_tensor(freqs, dtype=torch.float64, device=dev)           # (F,)
    if torch.any(d1 <= z_obs):
        raise ValueError("forward_torch supports only z_obs < d1 "
                         "(source and receivers in the water layer)")

    s = 2j * math.pi * f                                                  # (F,)
    zeta = (s * MU0)                                                      # (F,)
    eta0 = 1.0 / RES_AIR + s * EPS0                                       # (F,)
    eta1 = sigma1[:, None] + s[None, :] * EPS0                            # (B,F)
    eta2 = sigma2[:, None] + s[None, :] * EPS0                            # (B,F)

    lam = base[None, :] / off[:, None]                                    # (R,N)
    l2 = (lam * lam)[None, :, None, :]                                    # (1,R,1,N)

    # shapes below: (B, R, F, N)
    def gam(eta):
        return torch.sqrt(l2 + (zeta * eta)[..., None, :, None])

    G0 = gam(eta0[None, :])
    G1 = gam(eta1)
    G2 = gam(eta2)

    zs = d1[:, None, None, None]
    dsf = (d1 + d2)[:, None, None, None]
    zr = float(z_obs)

    E0 = eta0[None, None, :, None]
    E1 = eta1[:, None, :, None]
    E2 = eta2[:, None, :, None]
    Z = zeta[None, None, :, None]

    e_dsf = torch.exp(-2.0 * G1 * dsf)
    Wu = torch.exp(-G1 * (dsf - zr))
    Wd = torch.exp(-G1 * zr)
    direct = torch.exp(-G1 * torch.abs(zs - zr))

    greens = {}
    for TM in (True, False):
        if TM:
            e0, e1, e2 = E0, E1, E2
            pm = -1.0
        else:
            e0 = e1 = e2 = Z
            pm = 1.0
        Rp = (e2 * G1 - e1 * G2) / (e2 * G1 + e1 * G2)   # bottom interface
        Rm = (e0 * G1 - e1 * G0) / (e0 * G1 + e1 * G0)   # top interface
        den = 1.0 - Rm * Rp * e_dsf
        Pd = (torch.exp(-G1 * zs) + pm * Rp * torch.exp(-G1 * (2 * dsf - zs))) * Rm / den
        Pu = (torch.exp(-G1 * (dsf - zs)) + pm * Rm * torch.exp(-G1 * (dsf + zs))) * Rp / den
        g = Pu * Wu + Pd * Wd + (-direct if TM else direct)
        if TM:
            g = g * G1 / E1
        else:
            g = g * Z / G1
        greens[TM] = g

    GTM, GTE = greens[True], greens[False]
    lamb = lam[None, :, None, :]
    Ptot = (GTM + GTE) / (4.0 * math.pi)
    PJ0b = 0.5 * Ptot * lamb
    PJ1 = -Ptot
    PJ0 = (GTM - GTE) * lamb / (8.0 * math.pi)

    j0c = j0.to(torch.complex128)
    j1c = j1.to(torch.complex128)
    r = off[None, :, None]
    out = (PJ1 @ j1c) / r + PJ0b @ j0c          # angle factor cos(0) = 1
    out = out + PJ0 @ j0c
    return out / r


def velocity_factor(freqs, v0, sigma1):
    """Same ad hoc factor as forward_model_v4._velocity_factor, vectorised.
    v0, sigma1: (B,). Returns (B, 1, F) float64 (1 where v0 <= 1e-6)."""
    f = torch.as_tensor(freqs, dtype=torch.float64, device=sigma1.device)
    om = (2 * math.pi * f).abs().clamp_min(1e-6)[None, :]
    c_em = torch.sqrt(2.0 * om / (MU0 * sigma1.clamp_min(1e-9)[:, None]))
    fac = 1.0 + torch.minimum(v0[:, None] / c_em.clamp_min(1.0),
                              torch.full_like(c_em, 5.0))
    fac = torch.where(v0[:, None] > 1e-6, fac, torch.ones_like(fac))
    return fac[:, None, :]


def time_features(E_f, n_time):
    """irfft + peak normalisation, as in forward_model_v4.
    E_f: (B,R,F) complex. Returns E_norm (B,R,n_time), log_amp (B,R)."""
    E_t = torch.fft.irfft(E_f, n=n_time, dim=-1)
    peak = E_t.abs().amax(dim=-1)
    log_amp = torch.log10(peak + 1e-40)
    E_norm = E_t / (peak[..., None] + 1e-60)
    return E_norm, log_amp


def forward_features(sigma1, sigma2, d1, d2, v0=None, offsets=None,
                     z_obs=None, freqs=None, n_time=None):
    """Full DualTCN input features from physical parameters (differentiable)."""
    from config_v4 import RECEIVER_OFFSETS, Z_OBS, F_MIN, F_MAX, N_FREQ, N_TIME
    offsets = RECEIVER_OFFSETS if offsets is None else offsets
    z_obs = Z_OBS if z_obs is None else z_obs
    freqs = np.linspace(F_MIN, F_MAX, N_FREQ) if freqs is None else freqs
    n_time = N_TIME if n_time is None else n_time
    E_f = freq_response(sigma1, sigma2, d1, d2, offsets, z_obs, freqs)
    if v0 is not None:
        E_f = E_f * velocity_factor(freqs, v0, sigma1)
    return time_features(E_f, n_time)
