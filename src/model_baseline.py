"""
model_v4.py — PCRN V4 with multi-receiver input (8 channels).

Architecture is identical to V3 (HybridEncoder + ParamHead + physics decoder)
with the only change being IN_CHANNELS = 8 instead of 2.

The encoder now sees 8 time-series channels simultaneously:
  [E_r0, logamp_r0, E_r1, logamp_r1, E_r2, logamp_r2, E_r3, logamp_r3]

The far-offset channels (r=100, 200m) carry seafloor contrast information
that the near-field r=20m channel cannot provide.  The hybrid TCN+Transformer
encoder is free to learn cross-receiver attention patterns.

Input  : (B, 8, N_TIME)
Output : (B, N_DEPTH)  log10(σ) reconstructed from predicted parameters
"""

import torch
import torch.nn as nn

from config_v4 import (
    N_TIME, N_DEPTH, Z_MAX, LATENT_DIM, DROPOUT, IN_CHANNELS,
    SOFT_STEP_TAU,
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
)

# Reuse physics-decoder, normalisation, and building blocks from V3
from model_v3 import (
    reconstruct_profile,
    normalise_params,
    denormalise_params,
    DilatedResBlock,
    HybridEncoder,
    ParamHead,
)


# ── Override parameter bounds for V4 ─────────────────────────────────────────
# (V3 bounds are identical, but we re-register the tensor so it uses V4 config)

_BOUNDS_V4 = torch.tensor([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
    [LOG_D1_MIN,     LOG_D1_MAX    ],
    [LOG_D2_MIN,     LOG_D2_MAX    ],
], dtype=torch.float32)


def denormalise_params_v4(p_norm):
    b  = _BOUNDS_V4.to(p_norm.device)
    lo, hi = b[:, 0], b[:, 1]
    return lo + p_norm * (hi - lo)


# ── Full PCRN V4 ──────────────────────────────────────────────────────────────

class PCRN_V4(nn.Module):
    """
    Physics-Constrained Parameter Regression Network — V4.

    Identical to V3 PCRN except:
      - IN_CHANNELS = 8  (4 receivers × 2 channels each)
      - Uses V4 parameter bounds for denormalisation
    """

    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME,
                 out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder    = HybridEncoder(in_ch=in_ch, in_len=in_len,
                                        latent_dim=latent_dim, dropout=dropout)
        self.param_head = ParamHead(latent_dim=latent_dim, dropout=dropout)

    def forward(self, x):
        """
        Returns
        -------
        profile  : (B, N_DEPTH)  log10(σ) reconstructed from predicted params
        p_norm   : (B, 4)        normalised predicted parameters [0,1]
        p_phys   : (B, 4)        physical log10 parameters
        """
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
    model = PCRN_V4()
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy = torch.randn(4, IN_CHANNELS, N_TIME)
    prof, pn, pp = model(dummy)
    print(f"PCRN_V4 — trainable parameters : {n:,}")
    print(f"  Input   : {dummy.shape}")
    print(f"  Profile : {prof.shape}")
    print(f"  Params  : {pp.shape}  → {pp.detach().numpy().round(3)}")
