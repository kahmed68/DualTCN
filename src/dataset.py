"""
dataset_v4.py — Multi-receiver dataset generator and loader for V4.

Changes from v3:
  - Generates traces at 4 receiver offsets (20, 50, 100, 200 m).
  - Each sample has 8 input channels:
      [E_norm_r0, logamp_r0, E_norm_r1, logamp_r1,
       E_norm_r2, logamp_r2, E_norm_r3, logamp_r3]
    where r0..r3 = RECEIVER_OFFSETS.
  - Saved to mcsem_dataset_v4.npz with keys:
        E_multi   : (N, 4, N_TIME)  normalised traces per receiver
        log_amps  : (N, 4)          log10 amplitudes per receiver
        sigma_profiles : (N, N_DEPTH)  log10(σ(z))
        params    : (N, 4)          [σ1, σ2, d1, d2] physical values
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config_v4 import (
    RECEIVER_OFFSETS, N_RECEIVERS,
    SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE, D1_RANGE, D2_RANGE, V0_RANGE,
    N_SAMPLES_V4, RANDOM_SEED, DATA_PATH_V4,
    N_DEPTH, Z_MAX, N_TIME,
    BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT,
    LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
    LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
    LOG_D1_MIN, LOG_D1_MAX,
    LOG_D2_MIN, LOG_D2_MAX,
)
from forward_model_v4 import compute_multi_receiver_timeseries
from forward_model_v2 import build_sigma_profile


# ── Parameter normalisation ────────────────────────────────────────────────────

_BOUNDS_NP = np.array([
    [LOG_SIGMA1_MIN, LOG_SIGMA1_MAX],
    [LOG_SIGMA2_MIN, LOG_SIGMA2_MAX],
    [LOG_D1_MIN,     LOG_D1_MAX    ],
    [LOG_D2_MIN,     LOG_D2_MAX    ],
], dtype=np.float32)


def _params_to_norm(sigma1, sigma2, d1, d2):
    raw = np.array([
        np.log10(sigma1 + 1e-12),
        np.log10(sigma2 + 1e-12),
        np.log10(d1     + 1e-12),
        np.log10(d2     + 1e-12),
    ], dtype=np.float32)
    lo = _BOUNDS_NP[:, 0]
    hi = _BOUNDS_NP[:, 1]
    return np.clip((raw - lo) / (hi - lo + 1e-8), 0.0, 1.0)


# ── Dataset generation ─────────────────────────────────────────────────────────

def generate_dataset_v4(path=DATA_PATH_V4, n=N_SAMPLES_V4, seed=RANDOM_SEED):
    """
    Generate n multi-receiver MCSEM samples and save to path.

    Each sample:
      - Draw (σ1, σ2, d1, d2, v0) uniformly from the parameter ranges.
      - Compute E-field at 4 receiver offsets.
      - Compute the σ(z) profile.
    """
    if os.path.exists(path):
        print(f"Dataset already exists at '{path}'. Loading instead.")
        return load_v4_data(path)

    rng = np.random.default_rng(seed)

    # Pre-allocate arrays
    E_multi        = np.zeros((n, N_RECEIVERS, N_TIME), dtype=np.float32)
    log_amps_arr   = np.zeros((n, N_RECEIVERS), dtype=np.float32)
    z_arr          = np.linspace(0.0, Z_MAX, N_DEPTH)
    sigma_profiles = np.zeros((n, N_DEPTH), dtype=np.float32)
    params         = np.zeros((n, 4), dtype=np.float32)

    print(f"Generating {n} multi-receiver samples …")
    for i in range(n):
        if i % 1000 == 0:
            print(f"  {i}/{n}")

        # Sample parameters
        sigma1 = 10 ** rng.uniform(*SIGMA1_LOG_RANGE)
        sigma2 = 10 ** rng.uniform(*SIGMA2_LOG_RANGE)
        d1     = rng.uniform(*D1_RANGE)
        d2     = rng.uniform(*D2_RANGE)
        v0     = rng.uniform(*V0_RANGE)

        # Multi-receiver forward model
        results = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0)

        for j, r in enumerate(RECEIVER_OFFSETS):
            E_norm, log_amp = results[r]
            E_multi[i, j]      = E_norm
            log_amps_arr[i, j] = log_amp

        # Conductivity profile
        sigma_profiles[i] = build_sigma_profile(sigma1, sigma2, d1, d2, z_arr)
        params[i]         = [sigma1, sigma2, d1, d2]

    np.savez_compressed(
        path,
        E_multi        = E_multi,
        log_amps       = log_amps_arr,
        sigma_profiles = sigma_profiles,
        params         = params,
    )
    print(f"Dataset saved to '{path}'  shape E_multi: {E_multi.shape}")
    return E_multi, log_amps_arr, sigma_profiles, params


def load_v4_data(path=DATA_PATH_V4):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' not found.\n"
            f"Run: python -c \"from dataset_v4 import generate_dataset_v4; "
            f"generate_dataset_v4()\""
        )
    d = np.load(path)
    return d["E_multi"], d["log_amps"], d["sigma_profiles"], d["params"]


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class MCSEMDatasetV4(Dataset):
    """
    Returns:
      x          : (8, N_TIME) — 8-channel input
                    channels = [E_r0, logamp_r0, E_r1, logamp_r1,
                                E_r2, logamp_r2, E_r3, logamp_r3]
      sig_profile: (N_DEPTH,)  — log10(σ) ground-truth profile
      params_norm: (4,)        — normalised [log_σ1, log_σ2, log_d1, log_d2]
    """

    def __init__(self, E_multi, log_amps, sigma_profiles, params, augment=False):
        # E_multi  : (N, N_RECEIVERS, N_TIME)
        # log_amps : (N, N_RECEIVERS)
        self.E_multi        = torch.tensor(E_multi,        dtype=torch.float32)
        self.log_amps       = torch.tensor(log_amps,       dtype=torch.float32)
        self.sigma_profiles = torch.tensor(sigma_profiles, dtype=torch.float32)
        self.augment        = augment

        # Pre-compute normalised parameter targets
        n = len(params)
        p_norm = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            s1, s2, d1, d2 = params[i, 0], params[i, 1], params[i, 2], params[i, 3]
            p_norm[i] = _params_to_norm(s1, s2, d1, d2)
        self.params_norm = torch.tensor(p_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.E_multi)

    def __getitem__(self, idx):
        # Build 8-channel input: interleave waveform + logamp for each receiver
        channels = []
        for j in range(N_RECEIVERS):
            e_wave  = self.E_multi[idx, j].clone()
            log_amp = self.log_amps[idx, j]   # scalar

            if self.augment:
                noise_std = 10 ** torch.empty(1).uniform_(-3, -1)
                e_wave    = e_wave + noise_std * torch.randn_like(e_wave)

            amp_ch = log_amp.expand_as(e_wave)    # broadcast scalar → (N_TIME,)
            channels.append(e_wave)
            channels.append(amp_ch)

        x = torch.stack(channels, dim=0)   # (8, N_TIME)
        return x, self.sigma_profiles[idx], self.params_norm[idx]


# ── DataLoaders ────────────────────────────────────────────────────────────────

def get_dataloaders_v4(E_multi, log_amps, sigma_profiles, params):
    n       = len(E_multi)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    def _make(sl, aug):
        return MCSEMDatasetV4(
            E_multi[sl], log_amps[sl], sigma_profiles[sl], params[sl],
            augment=aug
        )

    train_sl = slice(0,             n_train)
    val_sl   = slice(n_train,       n_train + n_val)
    test_sl  = slice(n_train + n_val, n)

    train_ds = _make(train_sl, aug=True)
    val_ds   = _make(val_sl,   aug=False)
    test_ds  = _make(test_sl,  aug=False)

    kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    print(f"V4 Dataset — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
