"""
generate_large_dataset.py — Parallel dataset generation for V4.

Uses multiprocessing.Pool to spread work across all CPU cores.
Each worker generates an independent chunk of samples, results are
concatenated and saved as a single .npz file.

Usage:
    # Generate 1M samples using all available cores (recommended)
    python generate_large_dataset.py --n 1000000 --out mcsem_dataset_v4.npz

    # Quick 100K test run
    python generate_large_dataset.py --n 100000 --out mcsem_dataset_v4.npz

    # Limit core count (e.g., leave 2 cores free for other work)
    python generate_large_dataset.py --n 1000000 --workers 6

Estimated wall-clock time (empymod, 4 receivers):
    10K  samples: ~3 min   (1 core)  /  ~30 s  (8 cores)
    100K samples: ~30 min  (1 core)  /  ~4 min (8 cores)
    1M   samples: ~5 h     (1 core)  /  ~40 min (8 cores)
"""

import argparse
import os
import time
import multiprocessing as mp

import numpy as np

from config_v4 import (
    RECEIVER_OFFSETS, N_RECEIVERS,
    SIGMA1_LOG_RANGE, SIGMA2_LOG_RANGE,
    D1_RANGE, D2_RANGE, V0_RANGE,
    N_DEPTH, Z_MAX, N_TIME,
    DATA_PATH_V4,
)
from forward_model_v4 import compute_multi_receiver_timeseries
from forward_model_v2 import build_sigma_profile


# ── Worker function (runs in a separate process) ───────────────────────────────

def _generate_chunk(args):
    """
    Generate `chunk_size` random samples.
    Each worker gets a unique seed derived from (base_seed, worker_id)
    so chunks never overlap in parameter space.
    """
    chunk_size, base_seed, worker_id = args
    rng = np.random.default_rng(base_seed + worker_id * 99991)

    z_arr          = np.linspace(0.0, Z_MAX, N_DEPTH)
    E_multi        = np.zeros((chunk_size, N_RECEIVERS, N_TIME), dtype=np.float32)
    log_amps_arr   = np.zeros((chunk_size, N_RECEIVERS),         dtype=np.float32)
    sigma_profiles = np.zeros((chunk_size, N_DEPTH),             dtype=np.float32)
    params         = np.zeros((chunk_size, 4),                   dtype=np.float32)

    for i in range(chunk_size):
        sigma1 = 10 ** rng.uniform(*SIGMA1_LOG_RANGE)
        sigma2 = 10 ** rng.uniform(*SIGMA2_LOG_RANGE)
        d1     = rng.uniform(*D1_RANGE)
        d2     = rng.uniform(*D2_RANGE)
        v0     = rng.uniform(*V0_RANGE)

        results = compute_multi_receiver_timeseries(sigma1, sigma2, d1, d2, v0)

        for j, r in enumerate(RECEIVER_OFFSETS):
            E_norm, log_amp = results[r]
            E_multi[i, j]      = E_norm
            log_amps_arr[i, j] = log_amp

        sigma_profiles[i] = build_sigma_profile(sigma1, sigma2, d1, d2, z_arr)
        params[i]         = [sigma1, sigma2, d1, d2]

    return E_multi, log_amps_arr, sigma_profiles, params


# ── Main ───────────────────────────────────────────────────────────────────────

def generate(n_total, out_path, n_workers=None, chunk_size=2000, seed=42):
    """
    Generate n_total samples in parallel and save to out_path.

    Parameters
    ----------
    n_total    : int   total number of samples
    out_path   : str   output .npz path
    n_workers  : int   number of CPU workers (default: all cores)
    chunk_size : int   samples per worker task (tune for memory)
    seed       : int   base random seed
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)   # leave 1 core free

    # Build task list: split n_total into chunks
    n_full   = n_total // chunk_size
    remainder = n_total % chunk_size
    tasks    = [(chunk_size, seed, i) for i in range(n_full)]
    if remainder > 0:
        tasks.append((remainder, seed, n_full))

    print(f"Generating {n_total:,} samples")
    print(f"  Workers   : {n_workers}")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunks    : {len(tasks)}")
    print(f"  Output    : {out_path}")
    print()

    t0 = time.time()

    # Allocate output arrays
    z_arr  = np.linspace(0.0, Z_MAX, N_DEPTH)
    E_all        = np.zeros((n_total, N_RECEIVERS, N_TIME), dtype=np.float32)
    log_amps_all = np.zeros((n_total, N_RECEIVERS),         dtype=np.float32)
    sig_all      = np.zeros((n_total, N_DEPTH),             dtype=np.float32)
    par_all      = np.zeros((n_total, 4),                   dtype=np.float32)

    with mp.Pool(processes=n_workers) as pool:
        done = 0
        for result in pool.imap_unordered(_generate_chunk, tasks):
            E_c, la_c, sig_c, par_c = result
            n_c = len(E_c)
            E_all[done:done+n_c]        = E_c
            log_amps_all[done:done+n_c] = la_c
            sig_all[done:done+n_c]      = sig_c
            par_all[done:done+n_c]      = par_c
            done += n_c

            elapsed = time.time() - t0
            rate    = done / elapsed
            eta     = (n_total - done) / max(rate, 1e-6)
            print(f"  {done:>8,} / {n_total:,}  "
                  f"  {rate:.0f} samp/s  "
                  f"  ETA {eta/60:.1f} min", flush=True)

    # Shuffle so sequential splits in DataLoader are random
    idx = np.random.default_rng(seed).permutation(n_total)
    E_all        = E_all[idx]
    log_amps_all = log_amps_all[idx]
    sig_all      = sig_all[idx]
    par_all      = par_all[idx]

    np.savez_compressed(
        out_path,
        E_multi        = E_all,
        log_amps       = log_amps_all,
        sigma_profiles = sig_all,
        params         = par_all,
    )

    total = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDone in {total/60:.1f} min  —  file size: {size_mb:.0f} MB")
    print(f"Saved to '{out_path}'")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=1_000_000,
                        help="Total number of samples (default: 1 000 000)")
    parser.add_argument("--out",     type=str, default=DATA_PATH_V4,
                        help="Output .npz path")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of CPU workers (default: cpu_count-1)")
    parser.add_argument("--chunk",   type=int, default=2000,
                        help="Samples per worker task (default: 2000)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Base random seed")
    args = parser.parse_args()

    generate(
        n_total    = args.n,
        out_path   = args.out,
        n_workers  = args.workers,
        chunk_size = args.chunk,
        seed       = args.seed,
    )
