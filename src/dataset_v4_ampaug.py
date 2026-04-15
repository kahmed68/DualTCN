"""
dataset_v4_ampaug.py — MCSEMDatasetV4 with amplitude-channel noise augmentation.

Revised with three improvements over the initial version:
  1. Narrower augmentation range: σ_amp in [0.001, 0.01] log10 (~0.2-2%)
     matching physically achievable stacking precision.
  2. Curriculum support: augmentation strength ramps linearly from 0 to
     full over a configurable number of warmup epochs.
  3. Per-receiver independent noise (each receiver gets its own draw).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from config_v4 import (
    N_RECEIVERS, N_TIME, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
)
from dataset_v4 import _params_to_norm
import numpy as np


class MCSEMDatasetV4AmpAug(Dataset):
    """MCSEMDatasetV4 with amplitude noise augmentation and curriculum."""

    def __init__(self, E_multi, log_amps, sigma_profiles, params,
                 augment=False, amp_lo=0.001, amp_hi=0.01):
        self.E_multi        = torch.tensor(E_multi,        dtype=torch.float32)
        self.log_amps       = torch.tensor(log_amps,       dtype=torch.float32)
        self.sigma_profiles = torch.tensor(sigma_profiles, dtype=torch.float32)
        self.augment        = augment
        self.amp_lo         = amp_lo   # min σ_amp in log10 units
        self.amp_hi         = amp_hi   # max σ_amp in log10 units
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

                # Amplitude noise with curriculum scaling
                # Narrowed range: [0.001, 0.01] log10 (~0.2-2% uncertainty)
                if self.curriculum_scale > 0:
                    amp_noise_std = 10 ** torch.empty(1).uniform_(
                        np.log10(self.amp_lo), np.log10(self.amp_hi)
                    )
                    # Scale by curriculum factor
                    amp_noise_std = amp_noise_std * self.curriculum_scale
                    log_amp = log_amp + amp_noise_std * torch.randn(1)

            amp_ch = log_amp.expand(N_TIME)
            channels.append(e_wave)
            channels.append(amp_ch)

        x = torch.stack(channels, dim=0)   # (8, N_TIME)
        return x, self.sigma_profiles[idx], self.params_norm[idx]


def get_dataloaders_v4_ampaug(E_multi, log_amps, sigma_profiles, params,
                               amp_lo=0.001, amp_hi=0.01):
    n       = len(E_multi)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    def _make(sl, aug):
        return MCSEMDatasetV4AmpAug(
            E_multi[sl], log_amps[sl], sigma_profiles[sl], params[sl],
            augment=aug, amp_lo=amp_lo, amp_hi=amp_hi,
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

    print(f"V4 AmpAug Dataset (σ_amp=[{amp_lo},{amp_hi}] log10) — "
          f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
