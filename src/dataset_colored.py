"""
dataset_v4_colored.py — MCSEMDatasetV4 with colored (1/f) waveform noise.

Replaces the white Gaussian waveform noise with temporally correlated
1/f^alpha noise (pink noise, alpha=1).  Real MCSEM noise is dominated
by ocean-current motion noise and magnetotelluric signals, which have
a red/pink spectrum — low frequencies are noisier than high frequencies.
White noise understates early-time contamination relative to real data.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config_v4 import (
    N_RECEIVERS, N_TIME, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
)
from dataset_v4 import _params_to_norm


def _generate_pink_noise(n_samples, alpha=1.0):
    """Generate 1/f^alpha noise via spectral shaping.

    Parameters
    ----------
    n_samples : int
        Number of time samples.
    alpha : float
        Spectral exponent.  alpha=0 → white, alpha=1 → pink, alpha=2 → red.

    Returns
    -------
    torch.Tensor of shape (n_samples,), unit-variance pink noise.
    """
    white = torch.randn(n_samples)
    spectrum = torch.fft.rfft(white)
    freqs = torch.fft.rfftfreq(n_samples)
    # Avoid division by zero at DC; set DC component to 0
    freqs[0] = 1.0
    spectrum = spectrum / (freqs ** (alpha / 2.0))
    spectrum[0] = 0.0  # zero-mean
    colored = torch.fft.irfft(spectrum, n=n_samples)
    # Normalise to unit variance
    std = colored.std()
    if std > 1e-8:
        colored = colored / std
    return colored


class MCSEMDatasetV4Colored(Dataset):
    """MCSEMDatasetV4 with colored (1/f) waveform noise augmentation."""

    def __init__(self, E_multi, log_amps, sigma_profiles, params,
                 augment=False, alpha=1.0):
        self.E_multi        = torch.tensor(E_multi,        dtype=torch.float32)
        self.log_amps       = torch.tensor(log_amps,       dtype=torch.float32)
        self.sigma_profiles = torch.tensor(sigma_profiles, dtype=torch.float32)
        self.augment        = augment
        self.alpha          = alpha  # spectral exponent (1.0 = pink)

        n = len(params)
        p_norm = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            s1, s2, d1, d2 = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            p_norm[i] = _params_to_norm(s1, s2, d1, d2)
        self.params_norm = torch.tensor(p_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.E_multi)

    def __getitem__(self, idx):
        channels = []
        for j in range(N_RECEIVERS):
            e_wave  = self.E_multi[idx, j].clone()
            log_amp = self.log_amps[idx, j].clone()

            if self.augment:
                # Colored (1/f^alpha) waveform noise
                noise_std = 10 ** torch.empty(1).uniform_(-3, -1)
                colored = _generate_pink_noise(N_TIME, alpha=self.alpha)
                e_wave = e_wave + noise_std * colored

            amp_ch = log_amp.expand(N_TIME)
            channels.append(e_wave)
            channels.append(amp_ch)

        x = torch.stack(channels, dim=0)   # (8, N_TIME)
        return x, self.sigma_profiles[idx], self.params_norm[idx]


def get_dataloaders_v4_colored(E_multi, log_amps, sigma_profiles, params,
                               alpha=1.0):
    n       = len(E_multi)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    def _make(sl, aug):
        return MCSEMDatasetV4Colored(
            E_multi[sl], log_amps[sl], sigma_profiles[sl], params[sl],
            augment=aug, alpha=alpha,
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

    print(f"V4 Colored Noise Dataset (α={alpha}) — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
