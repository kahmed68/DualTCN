"""
generate_3layer_dataset.py — Generate three-layer MCSEM dataset.

Usage:
    python generate_3layer_dataset.py --n 200000 --workers 8
    python generate_3layer_dataset.py --n 50000   # small pilot
"""
import argparse
import os
import numpy as np
from multiprocessing import Pool
from functools import partial

DIR = os.path.dirname(os.path.abspath(__file__))

from config_v5 import (
    SIGMA1_LOG_RANGE_V5, SIGMA2_LOG_RANGE_V5, SIGMA3_LOG_RANGE_V5,
    D1_RANGE_V5, D2_RANGE_V5, H_RANGE_V5,
    RANDOM_SEED, DATA_PATH_V5, N_DEPTH, Z_MAX, SOFT_STEP_TAU,
    N_TIME,
)
from config_v4 import RECEIVER_OFFSETS, N_RECEIVERS
from forward_model_v5 import compute_3layer_timeseries, build_3layer_profile


def _generate_one(idx, seed_base=42):
    """Generate one sample (called by worker processes)."""
    rng = np.random.default_rng(seed_base + idx)

    # Sample parameters
    log_s1 = rng.uniform(*SIGMA1_LOG_RANGE_V5)
    log_s2 = rng.uniform(*SIGMA2_LOG_RANGE_V5)
    log_s3 = rng.uniform(*SIGMA3_LOG_RANGE_V5)
    d1 = rng.uniform(*D1_RANGE_V5)
    d2 = rng.uniform(*D2_RANGE_V5)
    h  = rng.uniform(*H_RANGE_V5)

    sigma1 = 10 ** log_s1
    sigma2 = 10 ** log_s2
    sigma3 = 10 ** log_s3

    # Forward model
    result = compute_3layer_timeseries(sigma1, sigma2, sigma3, d1, d2, h)

    # Collect multi-receiver data
    E_multi = np.zeros((N_RECEIVERS, N_TIME), dtype=np.float32)
    log_amps = np.zeros(N_RECEIVERS, dtype=np.float32)
    for j, r in enumerate(RECEIVER_OFFSETS):
        E_norm, log_amp = result[r]
        E_multi[j] = E_norm
        log_amps[j] = log_amp

    # Conductivity profile
    profile = build_3layer_profile(sigma1, sigma2, sigma3, d1, d2, h,
                                    n_depth=N_DEPTH, z_max=Z_MAX,
                                    tau=SOFT_STEP_TAU)

    # Parameters: [σ₁, σ₂, σ₃, d₁, d₂, h] in physical units
    params = np.array([sigma1, sigma2, sigma3, d1, d2, h], dtype=np.float32)

    return E_multi, log_amps, profile, params


def generate_dataset(n_samples=1_000_000, workers=8, seed=RANDOM_SEED,
                     output_path=None):
    if output_path is None:
        output_path = os.path.join(DIR, DATA_PATH_V5)

    print(f"Generating {n_samples} three-layer samples with {workers} workers...")

    gen_fn = partial(_generate_one, seed_base=seed)

    if workers <= 1:
        results = [gen_fn(i) for i in range(n_samples)]
    else:
        with Pool(workers) as pool:
            results = pool.map(gen_fn, range(n_samples))

    E_multi = np.stack([r[0] for r in results])
    log_amps = np.stack([r[1] for r in results])
    sigma_profiles = np.stack([r[2] for r in results])
    params = np.stack([r[3] for r in results])

    np.savez_compressed(output_path,
                        E_multi=E_multi,
                        log_amps=log_amps,
                        sigma_profiles=sigma_profiles,
                        params=params)
    print(f"Saved {n_samples} samples → {output_path}")
    print(f"  E_multi:        {E_multi.shape}")
    print(f"  log_amps:       {log_amps.shape}")
    print(f"  sigma_profiles: {sigma_profiles.shape}")
    print(f"  params:         {params.shape} [σ₁, σ₂, σ₃, d₁, d₂, h]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    generate_dataset(args.n, args.workers, args.seed, args.out)
