"""
model_v3.py — Physics-Constrained Parameter Regression Network (PCRN).

Architecture
────────────
  Encoder  : Hybrid TCN + Transformer
             TCN extracts multi-scale local features (dilated convolutions).
             Transformer refines with global self-attention over time.
             → latent vector z  (LATENT_DIM,)

  Regression head : z → 4 normalised parameters in [0,1]
                    [p_log_σ1, p_log_σ2, p_log_d1, p_log_d2]

  Profile decoder : Differentiable soft-step reconstruction of σ(z).
                    σ(z) = σ1 + (σ2−σ1) · sigmoid((z − d_seafloor) / τ)
                    This is analytic, differentiable, and enforces the
                    KNOWN structure of a 2-layer earth model.

Why this is better than predicting 64 depth points directly:
  1. Output dimensionality 64 → 4 (16× reduction).
  2. Physical constraint: profile is always a clean step function.
  3. Interface depth d_seafloor = d1+d2 is predicted explicitly.
  4. Latent space is forced to encode all 4 parameters, not just σ2.
  5. Huber loss on parameters is robust to outliers.

Input  : (B, 2, N_TIME)   — normalised E-trace + log-amplitude channel
Output : (B, N_DEPTH)     — log10(σ) reconstructed from predicted parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config_v3 import (
    N_TIME, N_DEPTH, Z_MAX, LATENT_DIM, DROPOUT, IN_CHANNELS,
    SOFT_STEP_TAU,
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
)


# ── Differentiable profile reconstruction ────────────────────────────────────

# Pre-compute the depth grid (constant, registered as a buffer)
_Z_ARRAY = torch.linspace(0.0, Z_MAX, N_DEPTH)   # shape (N_DEPTH,)


def reconstruct_profile(log_s1, log_s2, log_d1, log_d2, tau=SOFT_STEP_TAU):
    """
    Build a log10(σ) profile from the 4 predicted physical parameters.

    Uses a differentiable soft step function (sigmoid):
      σ(z) = σ1 + (σ2−σ1) · sigmoid((z − d_seafloor) / τ)

    The log10 version:
      log_σ(z) ≈ log_s1 + log10(1 + 10^(log_s2−log_s1) · f(z))
    but it is simpler and numerically equivalent to work in linear σ:
      σ(z) = 10^log_s1 + (10^log_s2 − 10^log_s1) · sigmoid(...)
    then take log10.

    Parameters
    ----------
    log_s1  : (B,)  log10(σ1)  [sea water]
    log_s2  : (B,)  log10(σ2)  [seafloor]
    log_d1  : (B,)  log10(d1)  [source depth from surface]
    log_d2  : (B,)  log10(d2)  [source height above seafloor]
    tau     : float  soft-step sharpness [m]

    Returns
    -------
    (B, N_DEPTH)  log10(σ(z)) profile
    """
    B = log_s1.shape[0]
    z = _Z_ARRAY.to(log_s1.device).unsqueeze(0).expand(B, -1)  # (B, N_DEPTH)

    sigma1     = 10.0 ** log_s1.unsqueeze(1)   # (B, 1)
    sigma2     = 10.0 ** log_s2.unsqueeze(1)
    d_seafloor = 10.0 ** log_d1.unsqueeze(1) + 10.0 ** log_d2.unsqueeze(1)

    # Soft transition at the seafloor
    weight = torch.sigmoid((z - d_seafloor) / tau)              # (B, N_DEPTH)
    sigma  = sigma1 + (sigma2 - sigma1) * weight                # (B, N_DEPTH)

    return torch.log10(sigma.clamp(min=1e-12))                  # (B, N_DEPTH)


# ── Parameter normalisation / de-normalisation ────────────────────────────────

# Bounds as tensors (registered on first use)
_BOUNDS = torch.tensor([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],   # row 0: log10(σ1)
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],   # row 1: log10(σ2)
    [LOG_D1_MIN,     LOG_D1_MAX    ],   # row 2: log10(d1)
    [LOG_D2_MIN,     LOG_D2_MAX    ],   # row 3: log10(d2)
], dtype=torch.float32)                 # shape (4, 2)


def normalise_params(params_true):
    """
    Convert true physical params (in log10 space) to [0,1] for loss computation.
    params_true : (B, 4)  [log_s1, log_s2, log_d1, log_d2]
    """
    b = _BOUNDS.to(params_true.device)
    lo, hi = b[:, 0], b[:, 1]        # (4,)
    return (params_true - lo) / (hi - lo + 1e-8)


def denormalise_params(p_norm):
    """
    Convert network output [0,1] → physical log10 values.
    p_norm : (B, 4)
    """
    b  = _BOUNDS.to(p_norm.device)
    lo, hi = b[:, 0], b[:, 1]
    return lo + p_norm * (hi - lo)


# ── TCN block ─────────────────────────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    """Dilated causal Conv1d with residual connection."""

    def __init__(self, channels, kernel=3, dilation=1, dropout=0.0):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel,
                               dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel,
                               dilation=dilation, padding=pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()

    def forward(self, x):
        T = x.size(-1)
        h = self.act(self.bn1(self.conv1(x)[..., :T]))
        h = self.drop(self.act(self.bn2(self.conv2(h)[..., :T])))
        return h + x


# ── Hybrid TCN-Transformer encoder ───────────────────────────────────────────

class HybridEncoder(nn.Module):
    """
    Stage 1 — TCN: extract local multi-scale temporal features.
    Stage 2 — Transformer: refine with global self-attention.
    Stage 3 — Pool + FC → latent vector.

    Motivation from v2 analysis:
      TCN had the best test score (0.5676) but was training-unstable.
      Transformer had the best generalisation gap.
      Combining both captures their complementary strengths.
    """

    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME,
                 tcn_ch=64, n_dilated=6,
                 d_model=128, nhead=4, n_tf_layers=2,
                 latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, tcn_ch, kernel_size=1),
            nn.BatchNorm1d(tcn_ch),
            nn.GELU(),
        )

        # ── TCN: dilation 1, 2, 4, 8, 16, 32 ─────────────────────────────────
        self.tcn = nn.Sequential(*[
            DilatedResBlock(tcn_ch, kernel=3, dilation=2**i, dropout=dropout)
            for i in range(n_dilated)
        ])

        # ── Project TCN → Transformer token dim ───────────────────────────────
        self.tcn_to_tf = nn.Conv1d(tcn_ch, d_model, kernel_size=1)
        self.pos_embed  = nn.Parameter(
            torch.randn(1, in_len, d_model) * 0.02
        )

        # ── Transformer ───────────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN for stable gradients
        )
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                  num_layers=n_tf_layers)

        # ── Pooling + bottleneck FC ────────────────────────────────────────────
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.input_proj(x)                           # (B, tcn_ch, T)
        h = self.tcn(h)                                  # (B, tcn_ch, T)
        h = self.tcn_to_tf(h).permute(0, 2, 1)          # (B, T, d_model)
        h = h + self.pos_embed
        h = self.transformer(h)                          # (B, T, d_model)
        h = self.pool(h.permute(0, 2, 1)).squeeze(-1)   # (B, d_model)
        return self.fc(h)                                # (B, latent_dim)


# ── Regression head ───────────────────────────────────────────────────────────

class ParamHead(nn.Module):
    """
    Maps latent vector → 4 normalised parameters in (0,1).

    Outputs are passed through sigmoid to enforce the [0,1] bound,
    which maps to the physical parameter ranges via denormalise_params().
    """

    def __init__(self, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 4),
            nn.Sigmoid(),              # output in (0,1)
        )

    def forward(self, z):
        return self.net(z)             # (B, 4) — normalised params


# ── Full PCRN ─────────────────────────────────────────────────────────────────

class PCRN(nn.Module):
    """
    Physics-Constrained Parameter Regression Network.

    Forward pass:
      1. Encode E-trace → latent z
      2. Regress z → 4 normalised parameters p_norm in (0,1)
      3. Decode p_norm → physical log10 parameters
      4. Reconstruct σ(z) profile analytically (soft step function)

    The profile reconstruction is fully differentiable, so gradients
    flow through the physics-based decoder into the encoder.
    """

    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME,
                 out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder   = HybridEncoder(in_ch=in_ch, in_len=in_len,
                                        latent_dim=latent_dim, dropout=dropout)
        self.param_head = ParamHead(latent_dim=latent_dim, dropout=dropout)

    def forward(self, x):
        """
        Returns
        -------
        profile  : (B, N_DEPTH) — log10(σ) reconstructed from predicted params
        p_norm   : (B, 4)       — normalised predicted parameters [0,1]
        p_phys   : (B, 4)       — physical log10 parameters
        """
        z      = self.encoder(x)               # (B, latent_dim)
        p_norm = self.param_head(z)             # (B, 4)  in (0,1)
        p_phys = denormalise_params(p_norm)     # (B, 4)  in log10 units

        log_s1, log_s2 = p_phys[:, 0], p_phys[:, 1]
        log_d1, log_d2 = p_phys[:, 2], p_phys[:, 3]

        profile = reconstruct_profile(log_s1, log_s2, log_d1, log_d2)
        return profile, p_norm, p_phys

    def encode(self, x):
        return self.encoder(x)

    def predict_params(self, x):
        """Convenience: return physical parameter predictions only."""
        z      = self.encoder(x)
        p_norm = self.param_head(z)
        return denormalise_params(p_norm)   # (B, 4) log10 units


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = PCRN()
    n      = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy  = torch.randn(4, IN_CHANNELS, N_TIME)
    prof, pn, pp = model(dummy)
    print(f"PCRN — trainable parameters : {n:,}")
    print(f"  Input   : {dummy.shape}")
    print(f"  Profile : {prof.shape}")
    print(f"  Params  : {pp.shape}  → {pp.detach().numpy().round(3)}")
