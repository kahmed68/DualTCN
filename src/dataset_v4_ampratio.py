"""
dataset_v4_ampratio.py — MCSEMDatasetV4 with amplitude ratios replacing absolutes.

Instead of 8 channels (4 waveforms + 4 absolute log-amplitudes), uses
7 channels (4 waveforms + 3 inter-receiver log-amplitude differences):

  Δ₁ = log A(50m) - log A(20m)
  Δ₂ = log A(100m) - log A(50m)
  Δ₃ = log A(200m) - log A(100m)

Ratios cancel common-mode amplitude errors (source-strength drift,
uniform calibration offset) while preserving the offset-dependent
amplitude decay that carries d₂ and σ₂ information.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config_v4 import (
    N_RECEIVERS, N_TIME, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
)
from dataset_v4 import _params_to_norm

# 7 channels: 4 waveforms + 3 amplitude ratios
IN_CHANNELS_RATIO = 7


class MCSEMDatasetV4AmpRatio(Dataset):
    """MCSEMDatasetV4 with inter-receiver amplitude ratios instead of absolutes."""

    def __init__(self, E_multi, log_amps, sigma_profiles, params,
                 augment=False):
        self.E_multi        = torch.tensor(E_multi,        dtype=torch.float32)
        self.log_amps       = torch.tensor(log_amps,       dtype=torch.float32)
        self.sigma_profiles = torch.tensor(sigma_profiles, dtype=torch.float32)
        self.augment        = augment

        n = len(params)
        p_norm = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            s1, s2, d1, d2 = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            p_norm[i] = _params_to_norm(s1, s2, d1, d2)
        self.params_norm = torch.tensor(p_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.E_multi)

    def __getitem__(self, idx):
        # Collect waveforms and log-amplitudes per receiver
        waveforms = []
        log_amps = []
        for j in range(N_RECEIVERS):
            e_wave  = self.E_multi[idx, j].clone()
            log_amp = self.log_amps[idx, j].clone()

            if self.augment:
                # Waveform noise (same as baseline)
                wave_noise_std = 10 ** torch.empty(1).uniform_(-3, -1)
                e_wave = e_wave + wave_noise_std * torch.randn_like(e_wave)

            waveforms.append(e_wave)
            log_amps.append(log_amp)

        # Build channels: 4 waveforms + 3 amplitude ratios
        channels = []
        for j in range(N_RECEIVERS):
            channels.append(waveforms[j])  # waveform channel

        # Inter-receiver log-amplitude differences (ratios in linear space)
        # Δ₁ = log A(50m) - log A(20m)
        # Δ₂ = log A(100m) - log A(50m)
        # Δ₃ = log A(200m) - log A(100m)
        for j in range(N_RECEIVERS - 1):
            delta = log_amps[j + 1] - log_amps[j]  # scalar
            channels.append(delta.expand(N_TIME))

        x = torch.stack(channels, dim=0)   # (7, N_TIME)
        return x, self.sigma_profiles[idx], self.params_norm[idx]


def get_dataloaders_v4_ampratio(E_multi, log_amps, sigma_profiles, params):
    n       = len(E_multi)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    def _make(sl, aug):
        return MCSEMDatasetV4AmpRatio(
            E_multi[sl], log_amps[sl], sigma_profiles[sl], params[sl],
            augment=aug,
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

    print(f"V4 AmpRatio Dataset (7ch: 4 waveforms + 3 ratios) — "
          f"train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
