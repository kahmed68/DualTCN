"""
model_p9d.py — LateTimePCRN: dedicated late-time branch + d_sf auxiliary head.

Motivation
──────────
d₂ (source-layer thickness above seafloor) is encoded primarily in the
late-time diffusion tail of the EM response:

    E(t) ~ exp(-2 · u₁ · d₂)   where  u₁ ∝ sqrt(σ₁ · μ₀ / t)

The global TCNEncoder compresses all time scales with equal weighting.
A second, smaller encoder dedicated to the last N_TIME//2 time samples
(late-time window) gives the network a specialised view of this regime.

An auxiliary head predicts the total seafloor depth d_sf = d₁ + d₂, which
spans a wider log-decade range than d₂ alone and provides an additional
gradient signal toward correct d₂ estimation.

Architecture
────────────
  encoder_full : TCNEncoder(8 ch, all 128 steps) → z_full  (LATENT_DIM)
  encoder_late : 4-block dilated TCN, 32 channels
                 applied to x[:, :, 64:] (last 64 steps) → z_late (LATENT_DIM//2)
  z_comb       : cat(z_full, z_late)            (LATENT_DIM * 3//2)
  head_s1d1    : MLP(z_full)      → [σ₁_norm, d₁_norm]
  head_s2d2    : MLP(z_comb + 2)  → [σ₂_norm, d₂_norm]   (cond on Stage 1)
  head_dsf     : MLP(z_comb)      → d_sf_norm  (auxiliary; train only)

The auxiliary prediction is stored as `model.dsf_pred` after each forward
call so that train_p9d.py can add a Huber auxiliary loss without modifying
the standard (profile, p_norm, p_phys) return signature used by train_utils.

Parameter budget: ~380K
Input  : (B, 8, N_TIME)
Output : (profile: B×N_DEPTH, p_norm: B×4, p_phys: B×4)
         self.dsf_pred : (B, 1)  — set during forward; used by train_p9d.py
"""

import torch
import torch.nn as nn

from config_v4 import (
    N_TIME, N_DEPTH, LATENT_DIM, DROPOUT, IN_CHANNELS, SOFT_STEP_TAU,
)
from model_v3 import DilatedResBlock, reconstruct_profile
from model_p5a import TCNEncoder
from model_v4 import denormalise_params_v4

_LATE_CH  = 32
_LATE_DIM = LATENT_DIM // 2   # 128
_LATE_T   = N_TIME // 2       # 64 — last half of the time axis


# ── Small late-time encoder ───────────────────────────────────────────────────

class _LateTCNEncoder(nn.Module):
    """Small TCN for the late-time window (last _LATE_T samples)."""

    def __init__(self, in_ch=IN_CHANNELS, tcn_ch=_LATE_CH,
                 n_dilated=4, out_dim=_LATE_DIM, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, tcn_ch, kernel_size=1),
            nn.BatchNorm1d(tcn_ch),
            nn.GELU(),
        )
        # Dilations 1, 2, 4, 8 — receptive field covers all 64 late samples
        self.tcn = nn.Sequential(*[
            DilatedResBlock(tcn_ch, kernel=3, dilation=2**i, dropout=dropout)
            for i in range(n_dilated)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(tcn_ch, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.tcn(h)
        return self.fc(self.pool(h).squeeze(-1))   # (B, out_dim)


# ── Parameter heads ───────────────────────────────────────────────────────────

class _Head2(nn.Module):
    def __init__(self, in_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)   # (B, 2)


class _Head1(nn.Module):
    def __init__(self, in_dim, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)   # (B, 1)


# ── Full model ────────────────────────────────────────────────────────────────

class LateTimePCRN(nn.Module):
    """
    Physics-Constrained Regression Network with dedicated late-time branch
    and auxiliary seafloor-depth regression head.

    Auxiliary d_sf target: d_sf = d₁ + d₂ [metres].
    Normalised as: (log10(d_sf) - log10(60)) / (log10(200) - log10(60)).
    Range d_sf ∈ [60, 200] m.
    """

    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME,
                 out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT):
        super().__init__()
        late_dim = latent_dim // 2
        comb_dim = latent_dim + late_dim   # e.g. 384

        self.encoder_full = TCNEncoder(in_ch=in_ch, latent_dim=latent_dim,
                                       dropout=dropout)
        self.encoder_late = _LateTCNEncoder(in_ch=in_ch, out_dim=late_dim,
                                            dropout=dropout)

        # Stage 1: σ₁ and d₁ from full-time encoder
        self.head_s1d1 = _Head2(latent_dim, dropout=dropout)

        # Stage 2: σ₂ and d₂, conditioned on Stage 1 + late-time context
        self.head_s2d2 = _Head2(comb_dim + 2, dropout=dropout)

        # Auxiliary: d_sf = d₁ + d₂ [normalised]
        self.head_dsf = _Head1(comb_dim, dropout=dropout)

        # d_sf normalisation constants (log10 space, d_sf ∈ [60, 200] m)
        self.register_buffer("_dsf_lo",  torch.tensor(1.77815, dtype=torch.float32))
        self.register_buffer("_dsf_rng", torch.tensor(0.52288, dtype=torch.float32))

        self.dsf_pred: torch.Tensor | None = None

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x):
        z_full = self.encoder_full(x)                       # (B, latent_dim)
        z_late = self.encoder_late(x[:, :, -_LATE_T:])     # (B, late_dim)
        z_comb = torch.cat([z_full, z_late], dim=-1)        # (B, comb_dim)

        # Stage 1
        s1d1 = self.head_s1d1(z_full)                      # (B, 2)

        # Stage 2 — conditioned on Stage 1 (detached)
        s2d2 = self.head_s2d2(
            torch.cat([z_comb, s1d1.detach()], dim=-1)
        )                                                   # (B, 2)

        # Auxiliary d_sf (stored; training script reads this)
        self.dsf_pred = self.head_dsf(z_comb)              # (B, 1)

        # Canonical p_norm: [σ₁, σ₂, d₁, d₂]
        p_norm = torch.stack(
            [s1d1[:, 0], s2d2[:, 0], s1d1[:, 1], s2d2[:, 1]],
            dim=1,
        )

        p_phys  = denormalise_params_v4(p_norm)
        profile = reconstruct_profile(
            p_phys[:, 0], p_phys[:, 1], p_phys[:, 2], p_phys[:, 3],
            tau=SOFT_STEP_TAU,
        )
        return profile, p_norm, p_phys

    # ── helpers ───────────────────────────────────────────────────────────────

    def norm_dsf(self, d_sf_metres):
        """Normalise physical d_sf tensor [metres] to (0, 1)."""
        log_dsf = torch.log10(d_sf_metres.clamp(min=1.0))
        return ((log_dsf - self._dsf_lo) / self._dsf_rng).clamp(0.0, 1.0)

    def encode(self, x):
        return self.encoder_full(x)

    def predict_params(self, x):
        _, _, p_phys = self.forward(x)
        return p_phys


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = LateTimePCRN()
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy = torch.randn(4, IN_CHANNELS, N_TIME)
    prof, pn, pp = model(dummy)
    print(f"LateTimePCRN — trainable parameters : {n:,}")
    print(f"  Input    : {dummy.shape}")
    print(f"  Profile  : {prof.shape}")
    print(f"  Params   : {pp.shape}  -> {pp.detach().numpy().round(3)}")
    print(f"  dsf_pred : {model.dsf_pred.shape}")
