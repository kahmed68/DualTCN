"""
dataset_v5.py — Dataset and dataloaders for three-layer MCSEM inversion.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config_v5 import (
    LOG_SIGMA1_MIN_V5, LOG_SIGMA1_MAX_V5,
    LOG_SIGMA2_MIN_V5, LOG_SIGMA2_MAX_V5,
    LOG_SIGMA3_MIN_V5, LOG_SIGMA3_MAX_V5,
    LOG_D1_MIN_V5, LOG_D1_MAX_V5,
    LOG_D2_MIN_V5, LOG_D2_MAX_V5,
    LOG_H_MIN_V5, LOG_H_MAX_V5,
    N_TIME, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
    DATA_PATH_V5, N_PARAMS_V5,
)
from config_v4 import N_RECEIVERS

DIR = os.path.dirname(os.path.abspath(__file__))

# Parameter bounds for normalisation
_BOUNDS_V5 = np.array([
    [LOG_SIGMA1_MIN_V5, LOG_SIGMA1_MAX_V5],
    [LOG_SIGMA2_MIN_V5, LOG_SIGMA2_MAX_V5],
    [LOG_SIGMA3_MIN_V5, LOG_SIGMA3_MAX_V5],
    [LOG_D1_MIN_V5,     LOG_D1_MAX_V5],
    [LOG_D2_MIN_V5,     LOG_D2_MAX_V5],
    [LOG_H_MIN_V5,      LOG_H_MAX_V5],
], dtype=np.float32)


def _params_to_norm_v5(params_phys):
    """Convert physical params [σ₁,σ₂,σ₃,d₁,d₂,h] to normalised [0,1]."""
    lp = np.log10(params_phys + 1e-12)
    return np.clip(
        (lp - _BOUNDS_V5[:, 0]) / (_BOUNDS_V5[:, 1] - _BOUNDS_V5[:, 0] + 1e-8),
        0., 1.
    ).astype(np.float32)


class MCSEMDatasetV5(Dataset):
    """Three-layer MCSEM dataset."""

    def __init__(self, E_multi, log_amps, sigma_profiles, params,
                 augment=False):
        self.E_multi = torch.tensor(E_multi, dtype=torch.float32)
        self.log_amps = torch.tensor(log_amps, dtype=torch.float32)
        self.sigma_profiles = torch.tensor(sigma_profiles, dtype=torch.float32)
        self.augment = augment

        n = len(params)
        p_norm = np.zeros((n, N_PARAMS_V5), dtype=np.float32)
        for i in range(n):
            p_norm[i] = _params_to_norm_v5(params[i])
        self.params_norm = torch.tensor(p_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.E_multi)

    def __getitem__(self, idx):
        channels = []
        for j in range(N_RECEIVERS):
            e_wave = self.E_multi[idx, j].clone()
            log_amp = self.log_amps[idx, j].clone()

            if self.augment:
                noise_std = 10 ** torch.empty(1).uniform_(-3, -1)
                e_wave = e_wave + noise_std * torch.randn_like(e_wave)

            amp_ch = log_amp.expand(N_TIME)
            channels.append(e_wave)
            channels.append(amp_ch)

        x = torch.stack(channels, dim=0)
        return x, self.sigma_profiles[idx], self.params_norm[idx]


def load_v5_data(path=None):
    if path is None:
        path = os.path.join(DIR, DATA_PATH_V5)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python generate_3layer_dataset.py")
    d = np.load(path)
    return d["E_multi"], d["log_amps"], d["sigma_profiles"], d["params"]


def get_dataloaders_v5(E_multi, log_amps, sigma_profiles, params):
    n = len(E_multi)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    def _make(sl, aug):
        return MCSEMDatasetV5(
            E_multi[sl], log_amps[sl], sigma_profiles[sl], params[sl],
            augment=aug)

    train_ds = _make(slice(0, n_train), aug=True)
    val_ds = _make(slice(n_train, n_train + n_val), aug=False)
    test_ds = _make(slice(n_train + n_val, n), aug=False)

    kw = dict(batch_size=BATCH_SIZE, num_workers=4, pin_memory=True,
              persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print(f"V5 3-Layer Dataset — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
