"""
model_p5a.py — TCNOnlyPCRN: Pure TCN baseline (no Transformer in encoder).

Architecture
────────────
  Input projection: Conv1d(in_ch, 64) → BN → GELU
  6 × DilatedResBlock (dilation 1, 2, 4, 8, 16, 32) on 64 channels
  AdaptiveAvgPool1d(1) → squeeze
  FC(64, latent_dim) → GELU → Dropout
  ParamHead → p_norm → denormalise → reconstruct_profile

Input  : (B, 8, N_TIME)
Output : (profile: B×N_DEPTH, p_norm: B×4, p_phys: B×4)
"""

import torch
import torch.nn as nn

from config_v4 import (
    N_TIME, N_DEPTH, LATENT_DIM, DROPOUT, IN_CHANNELS, SOFT_STEP_TAU,
)
from model_v3 import DilatedResBlock, ParamHead, reconstruct_profile
from model_v4 import denormalise_params_v4

_TCN_CH = 64


# ── TCN-only encoder ──────────────────────────────────────────────────────────

class TCNEncoder(nn.Module):
    """
    Pure TCN encoder: 6 dilated residual blocks, no Transformer.

    Input : (B, in_ch, T)
    Output: (B, latent_dim)
    """

    def __init__(self, in_ch=IN_CHANNELS, tcn_ch=_TCN_CH,
                 n_dilated=6, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, tcn_ch, kernel_size=1),
            nn.BatchNorm1d(tcn_ch),
            nn.GELU(),
        )
        # Dilated blocks: dilation 1, 2, 4, 8, 16, 32
        self.tcn = nn.Sequential(*[
            DilatedResBlock(tcn_ch, kernel=3, dilation=2**i, dropout=dropout)
            for i in range(n_dilated)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(tcn_ch, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.input_proj(x)          # (B, tcn_ch, T)
        h = self.tcn(h)                 # (B, tcn_ch, T)
        h = self.pool(h).squeeze(-1)    # (B, tcn_ch)
        return self.fc(h)               # (B, latent_dim)


# ── Full model ────────────────────────────────────────────────────────────────

class TCNOnlyPCRN(nn.Module):
    """
    Physics-Constrained Parameter Regression Network — TCN-only encoder.

    No Transformer layers; pure dilated TCN for fast, low-memory training.
    Forward returns (profile, p_norm, p_phys) identical to PCRN_V4.
    """

    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME,
                 out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder    = TCNEncoder(in_ch=in_ch, latent_dim=latent_dim,
                                     dropout=dropout)
        self.param_head = ParamHead(latent_dim=latent_dim, dropout=dropout)

    def forward(self, x):
        z      = self.encoder(x)
        p_norm = self.param_head(z)
        p_phys = denormalise_params_v4(p_norm)

        log_s1, log_s2 = p_phys[:, 0], p_phys[:, 1]
        log_d1, log_d2 = p_phys[:, 2], p_phys[:, 3]

        profile = reconstruct_profile(log_s1, log_s2, log_d1, log_d2,
                                      tau=SOFT_STEP_TAU)
        return profile, p_norm, p_phys

    def encode(self, x):
        return self.encoder(x)

    def predict_params(self, x):
        z      = self.encoder(x)
        p_norm = self.param_head(z)
        return denormalise_params_v4(p_norm)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = TCNOnlyPCRN()
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy = torch.randn(4, IN_CHANNELS, N_TIME)
    prof, pn, pp = model(dummy)
    print(f"TCNOnlyPCRN — trainable parameters : {n:,}")
    print(f"  Input   : {dummy.shape}")
    print(f"  Profile : {prof.shape}")
    print(f"  Params  : {pp.shape}  -> {pp.detach().numpy().round(3)}")
