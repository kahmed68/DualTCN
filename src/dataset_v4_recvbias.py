"""
dataset_v4_recvbias.py — MCSEMDatasetV4 with per-receiver independent amplitude bias.

Revised with:
  1. Narrower bias range: β_max = 0.03 log10 (~±7% per receiver),
     matching realistic inter-receiver calibration drift.
  2. Curriculum support: bias strength ramps linearly from 0 to full
     over a configurable warmup window.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config_v4 import (
    N_RECEIVERS, N_TIME, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
)
from dataset_v4 import _params_to_norm


class MCSEMDatasetV4RecvBias(Dataset):
    """MCSEMDatasetV4 with per-receiver independent amplitude bias and curriculum."""

    def __init__(self, E_multi, log_amps, sigma_profiles, params,
                 augment=False, beta_max=0.03):
        self.E_multi        = torch.tensor(E_multi,        dtype=torch.float32)
        self.log_amps       = torch.tensor(log_amps,       dtype=torch.float32)
        self.sigma_profiles = torch.tensor(sigma_profiles, dtype=torch.float32)
        self.augment        = augment
        self.beta_max       = beta_max  # max bias in log10 units
        # Curriculum: fraction of full augmentation strength [0, 1]
        self.curriculum_scale = 1.0

        n = len(params)
        p_norm = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            s1, s2, d1, d2 = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            p_norm[i] = _params_to_norm(s1, s2, d1, d2)
        self.params_norm = torch.tensor(p_norm, dtype=torch.float32)

    def set_curriculum_scale(self, scale):
        """Set curriculum scale in [0, 1]. Called by training loop."""
        self.curriculum_scale = max(0.0, min(1.0, scale))

    def __len__(self):
        return len(self.E_multi)

    def __getitem__(self, idx):
        channels = []
        for j in range(N_RECEIVERS):
            e_wave  = self.E_multi[idx, j].clone()
            log_amp = self.log_amps[idx, j].clone()

            if self.augment:
                # Waveform noise (same as baseline)
                wave_noise_std = 10 ** torch.empty(1).uniform_(-3, -1)
                e_wave = e_wave + wave_noise_std * torch.randn_like(e_wave)

                # Per-receiver independent bias with curriculum scaling
                if self.curriculum_scale > 0:
                    effective_beta = self.beta_max * self.curriculum_scale
                    recv_bias = torch.empty(1).uniform_(
                        -effective_beta, effective_beta
                    )
                    log_amp = log_amp + recv_bias

            amp_ch = log_amp.expand(N_TIME)
            channels.append(e_wave)
            channels.append(amp_ch)

        x = torch.stack(channels, dim=0)   # (8, N_TIME)
        return x, self.sigma_profiles[idx], self.params_norm[idx]


def get_dataloaders_v4_recvbias(E_multi, log_amps, sigma_profiles, params,
                                beta_max=0.03):
    n       = len(E_multi)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    def _make(sl, aug):
        return MCSEMDatasetV4RecvBias(
            E_multi[sl], log_amps[sl], sigma_profiles[sl], params[sl],
            augment=aug, beta_max=beta_max,
        )

    train_sl = slice(0,             n_train)
    val_sl   = slice(n_train,       n_train + n_val)
    test_sl  = slice(n_train + n_val, n)

    train_ds = _make(train_sl, aug=True)
    val_ds   = _make(val_sl,   aug=False)
    test_ds  = _make(test_sl,  aug=False)

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
              persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"V4 RecvBias Dataset (β_max={beta_max}) — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
