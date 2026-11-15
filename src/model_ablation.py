"""E1c ablation variants for the corrected-data ranking check (2026-11-13).

dualtcn_nolate : DualTCN (LateTimePCRN) without the late-time branch; Stage 2 and the d_sf head see z_full only.
The other E1c arms reuse existing classes: PCRN_V4 (model_baseline, v1 baseline), TCNOnlyPCRN (model_p5a, P5a),
and LateTimePCRN trained with the auxiliary d_sf loss weight set to 0 (dualtcn_noaux).
The original P7 and P8 code is not in the repository (only one-line descriptions survive in supplement S7).
"""
import torch
import torch.nn as nn

from config_v4 import IN_CHANNELS, N_TIME, N_DEPTH, LATENT_DIM, DROPOUT, SOFT_STEP_TAU
from model_p5a import TCNEncoder
from model_p9d import LateTimePCRN, _Head1, _Head2
from model_v3 import reconstruct_profile
from model_v4 import denormalise_params_v4


class NoLatePCRN(LateTimePCRN):
    def __init__(self, in_ch=IN_CHANNELS, in_len=N_TIME, out_len=N_DEPTH, latent_dim=LATENT_DIM, dropout=DROPOUT):
        nn.Module.__init__(self)
        self.encoder_full = TCNEncoder(in_ch=in_ch, latent_dim=latent_dim, dropout=dropout)
        self.head_s1d1 = _Head2(latent_dim, dropout=dropout)
        self.head_s2d2 = _Head2(latent_dim + 2, dropout=dropout)
        self.head_dsf = _Head1(latent_dim, dropout=dropout)
        self.register_buffer("_dsf_lo", torch.tensor(1.77815, dtype=torch.float32))
        self.register_buffer("_dsf_rng", torch.tensor(0.52288, dtype=torch.float32))
        self.dsf_pred = None

    def forward(self, x):
        z = self.encoder_full(x)
        s1d1 = self.head_s1d1(z)
        s2d2 = self.head_s2d2(torch.cat([z, s1d1.detach()], dim=-1))
        self.dsf_pred = self.head_dsf(z)
        p_norm = torch.stack([s1d1[:, 0], s2d2[:, 0], s1d1[:, 1], s2d2[:, 1]], dim=1)
        p_phys = denormalise_params_v4(p_norm)
        profile = reconstruct_profile(p_phys[:, 0], p_phys[:, 1], p_phys[:, 2], p_phys[:, 3], tau=SOFT_STEP_TAU)
        return profile, p_norm, p_phys
