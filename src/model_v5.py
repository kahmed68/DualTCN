"""
model_v5.py — DualTCN for three-layer MCSEM inversion (6 parameters).

Extends the DualTCN architecture to predict 6 parameters:
  σ₁, σ₂, σ₃, d₁, d₂, h

The physics decoder uses two sigmoid transitions (Equation 14 in the paper):
  σ(z) = σ₁ + (σ₂-σ₁)·sig((z-d_sf)/τ) + (σ₃-σ₂)·sig((z-d_bot)/τ)

Stage 1: predicts [σ₁, d₁] from z_full (easy parameters)
Stage 2: predicts [σ₂, σ₃, d₂, h] from z_comb + detached Stage 1
"""
import torch
import torch.nn as nn
import numpy as np

from config_v5 import (
    N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS, DROPOUT,
    N_PARAMS_V5, Z_MAX, SOFT_STEP_TAU,
    LOG_SIGMA1_MIN_V5, LOG_SIGMA1_MAX_V5,
    LOG_SIGMA2_MIN_V5, LOG_SIGMA2_MAX_V5,
    LOG_SIGMA3_MIN_V5, LOG_SIGMA3_MAX_V5,
    LOG_D1_MIN_V5, LOG_D1_MAX_V5,
    LOG_D2_MIN_V5, LOG_D2_MAX_V5,
    LOG_H_MIN_V5, LOG_H_MAX_V5,
)

# Reuse encoder components from the original DualTCN
from model_p9d import TCNEncoder, _LateTCNEncoder


def _denorm_v5(p_norm):
    """Denormalise [0,1] → log10 physical."""
    bounds = torch.tensor([
        [LOG_SIGMA1_MIN_V5, LOG_SIGMA1_MAX_V5],
        [LOG_SIGMA2_MIN_V5, LOG_SIGMA2_MAX_V5],
        [LOG_SIGMA3_MIN_V5, LOG_SIGMA3_MAX_V5],
        [LOG_D1_MIN_V5, LOG_D1_MAX_V5],
        [LOG_D2_MIN_V5, LOG_D2_MAX_V5],
        [LOG_H_MIN_V5, LOG_H_MAX_V5],
    ], device=p_norm.device, dtype=p_norm.dtype)
    return bounds[:, 0] + p_norm * (bounds[:, 1] - bounds[:, 0])


class _PredHead(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DualTCN3Layer(nn.Module):
    """DualTCN extended to three-layer inversion (6 parameters)."""

    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME,
                 out_len=N_DEPTH, latent_dim=LATENT_DIM,
                 dropout=DROPOUT):
        super().__init__()
        late_dim = latent_dim // 2

        # Encoders (same as two-layer DualTCN)
        self.encoder_full = TCNEncoder(in_ch=in_ch, latent_dim=latent_dim,
                                       dropout=dropout)
        self.encoder_late = _LateTCNEncoder(in_ch=in_ch, out_dim=late_dim,
                                            dropout=dropout)

        comb_dim = latent_dim + late_dim  # 384

        # Stage 1: easy params [σ₁, d₁] from z_full
        self.head_s1d1 = _PredHead(latent_dim, 2, dropout)

        # Stage 2: hard params [σ₂, σ₃, d₂, h] from z_comb + detached Stage 1
        self.head_hard = _PredHead(comb_dim + 2, 4, dropout)

        # Depth grid for profile decoder
        self.register_buffer(
            "_z_grid",
            torch.linspace(0, Z_MAX, out_len).unsqueeze(0),  # (1, N_DEPTH)
        )
        self._tau = SOFT_STEP_TAU
        self._out_len = out_len

    def forward(self, x):
        z_full = self.encoder_full(x)
        z_late = self.encoder_late(x)
        z_comb = torch.cat([z_full, z_late], dim=1)

        # Stage 1: easy parameters
        p_s1d1 = self.head_s1d1(z_full)  # (B, 2) → [σ₁_norm, d₁_norm]

        # Stage 2: hard parameters, conditioned on Stage 1
        s1d1_detached = p_s1d1.detach()
        inp2 = torch.cat([z_comb, s1d1_detached], dim=1)
        p_hard = self.head_hard(inp2)  # (B, 4) → [σ₂, σ₃, d₂, h] norm

        # Assemble full parameter vector [σ₁, σ₂, σ₃, d₁, d₂, h]
        p_norm = torch.cat([
            p_s1d1[:, 0:1],   # σ₁
            p_hard[:, 0:1],   # σ₂
            p_hard[:, 1:2],   # σ₃
            p_s1d1[:, 1:2],   # d₁
            p_hard[:, 2:3],   # d₂
            p_hard[:, 3:4],   # h
        ], dim=1)  # (B, 6)

        # Denormalise to log10 physical
        p_log = _denorm_v5(p_norm)
        p_phys = 10 ** p_log

        # Physics decoder: three-layer profile
        sigma1 = p_phys[:, 0:1]  # (B, 1)
        sigma2 = p_phys[:, 1:2]
        sigma3 = p_phys[:, 2:3]
        d1 = p_phys[:, 3:4]
        d2 = p_phys[:, 4:5]
        h  = p_phys[:, 5:6]

        d_sf = d1 + d2           # top of resistive layer
        d_bot = d_sf + h         # bottom of resistive layer

        z = self._z_grid         # (1, N_DEPTH)

        # σ(z) = σ₁ + (σ₂-σ₁)·sig((z-d_sf)/τ) + (σ₃-σ₂)·sig((z-d_bot)/τ)
        profile = (sigma1
                   + (sigma2 - sigma1) * torch.sigmoid((z - d_sf) / self._tau)
                   + (sigma3 - sigma2) * torch.sigmoid((z - d_bot) / self._tau))

        profile_log = torch.log10(profile + 1e-12)  # (B, N_DEPTH)

        return profile_log, p_norm, p_phys
