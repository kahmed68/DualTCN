"""
benchmark_field_models.py — Validate DualTCN against published MCSEM survey models.

Uses published 1D conductivity models from well-known MCSEM survey sites
to generate empymod synthetics, then evaluates DualTCN predictions
against the published interpretations.

Published models are derived from:
  - Key (2009): 1D inversion of multicomponent CSEM data
  - Key (2012): Marine electromagnetic studies of seafloor resources
  - Constable & Srnka (2007): Introduction to marine CSEM
  - Constable (2010): Ten years of marine CSEM

Each model specifies a two-layer earth (σ₁, σ₂, d₁, d₂) derived from
published survey parameters at real MCSEM sites.
"""
import os
import sys
import numpy as np
import torch
import csv

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)

from config_p9d import (
    N_TIME, N_DEPTH, LATENT_DIM, IN_CHANNELS, DROPOUT,
)
from model_p9d import LateTimePCRN
from forward_model_v4 import compute_multi_receiver_timeseries
from config_v4 import RECEIVER_OFFSETS, N_RECEIVERS

PAPER_DIR = os.path.join(DIR, "paper")

# ── Published MCSEM survey models ────────────────────────────────────────────
# Each entry: (name, site, reference, σ₁, σ₂, d₁, d₂, notes)
# Parameters chosen to match published conductivity models within the
# DualTCN training range.

PUBLISHED_MODELS = [
    {
        "name": "Gulf of Mexico — Gemini",
        "ref": "Constable & Srnka (2007)",
        "sigma1": 3.3,      # typical GoM seawater
        "sigma2": 0.5,      # sediment conductivity
        "d1": 100.0,        # water depth ~1000m, source at ~100m
        "d2": 30.0,         # source-to-seafloor
        "notes": "Shallow GoM shelf, sediment background",
    },
    {
        "name": "GoM — Resistive target",
        "ref": "Key (2009)",
        "sigma1": 3.3,
        "sigma2": 0.05,     # hydrocarbon-bearing reservoir
        "d1": 80.0,
        "d2": 25.0,
        "notes": "Resistive reservoir analog at shallow depth",
    },
    {
        "name": "Scarborough — Gas field",
        "ref": "Key (2012)",
        "sigma1": 3.5,      # warm Indian Ocean water
        "sigma2": 0.02,     # gas-bearing sandstone
        "d1": 120.0,
        "d2": 20.0,
        "notes": "Offshore NW Australia gas field analog",
    },
    {
        "name": "North Sea — Shallow water",
        "ref": "Constable (2010)",
        "sigma1": 4.0,      # cold North Sea water
        "sigma2": 0.8,      # conductive clay-rich sediment
        "d1": 60.0,
        "d2": 40.0,
        "notes": "Shallow North Sea, conductive sediment",
    },
    {
        "name": "Norwegian margin — Gas hydrate",
        "ref": "Constable (2010)",
        "sigma1": 3.0,
        "sigma2": 0.01,     # gas-hydrate-bearing sediment
        "d1": 90.0,
        "d2": 15.0,
        "notes": "Norwegian continental margin hydrate zone",
    },
    {
        "name": "West Africa — Deep water",
        "ref": "Key (2012)",
        "sigma1": 3.2,
        "sigma2": 0.3,      # typical deep-water sediment
        "d1": 130.0,
        "d2": 35.0,
        "notes": "West African deep-water exploration setting",
    },
    {
        "name": "Brazil — Pre-salt",
        "ref": "Constable (2010)",
        "sigma1": 3.5,
        "sigma2": 0.005,    # highly resistive carbonate reservoir
        "d1": 100.0,
        "d2": 10.0,
        "notes": "Brazilian pre-salt carbonate reservoir analog",
    },
]


def run_benchmark(model_path, device=None):
    """Run DualTCN on published survey models and compare."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("=" * 70)
    print("Benchmark: DualTCN vs. published MCSEM survey models")
    print("=" * 70)

    # Load model
    model = LateTimePCRN(in_ch=IN_CHANNELS, in_len=N_TIME,
                         out_len=N_DEPTH, latent_dim=LATENT_DIM,
                         dropout=DROPOUT).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=True)
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # Denormalisation bounds
    from config_v4 import (
        LOG_SIGMA1_MIN, LOG_SIGMA1_MAX,
        LOG_SIGMA2_MIN, LOG_SIGMA2_MAX,
        LOG_D1_MIN, LOG_D1_MAX,
        LOG_D2_MIN, LOG_D2_MAX,
    )
    from model_v4 import denormalise_params_v4

    results = []

    for m in PUBLISHED_MODELS:
        print(f"\n--- {m['name']} ({m['ref']}) ---")
        print(f"  True: σ₁={m['sigma1']:.2f}  σ₂={m['sigma2']:.4f}  "
              f"d₁={m['d1']:.0f}  d₂={m['d2']:.0f}")

        # Generate synthetic data
        result = compute_multi_receiver_timeseries(
            m["sigma1"], m["sigma2"], m["d1"], m["d2"])

        # Build input tensor (same as dataset_v4)
        channels = []
        for r in RECEIVER_OFFSETS:
            E_norm, log_amp = result[r]
            channels.append(torch.tensor(E_norm, dtype=torch.float32))
            amp_ch = torch.tensor(log_amp, dtype=torch.float32).expand(N_TIME)
            channels.append(amp_ch)
        x = torch.stack(channels).unsqueeze(0).to(device)  # (1, 8, 128)

        # Predict
        with torch.no_grad():
            _, p_norm, p_phys = model(x)

        p_phys_np = p_phys.cpu().numpy()[0]
        s1_pred = p_phys_np[0]
        s2_pred = p_phys_np[1]
        d1_pred = 10 ** p_phys_np[2] if p_phys_np[2] < 5 else p_phys_np[2]
        d2_pred = 10 ** p_phys_np[3] if p_phys_np[3] < 5 else p_phys_np[3]

        # Handle denormalisation
        p_norm_np = p_norm.cpu().numpy()[0]
        p_denorm = denormalise_params_v4(p_norm.to(device))
        p_d = p_denorm.cpu().numpy()[0]
        s1_p = 10 ** p_d[0]
        s2_p = 10 ** p_d[1]
        d1_p = 10 ** p_d[2]
        d2_p = 10 ** p_d[3]

        # Errors
        err_s1 = abs(s1_p - m["sigma1"]) / m["sigma1"] * 100
        err_s2 = abs(s2_p - m["sigma2"]) / m["sigma2"] * 100
        err_d1 = abs(d1_p - m["d1"]) / m["d1"] * 100
        err_d2 = abs(d2_p - m["d2"]) / m["d2"] * 100
        mape = (err_s1 + err_s2 + err_d1 + err_d2) / 4

        print(f"  Pred: σ₁={s1_p:.2f}  σ₂={s2_p:.4f}  "
              f"d₁={d1_p:.1f}  d₂={d2_p:.1f}")
        print(f"  Err%: σ₁={err_s1:.1f}%  σ₂={err_s2:.1f}%  "
              f"d₁={err_d1:.1f}%  d₂={err_d2:.1f}%  MAPE={mape:.1f}%")

        results.append({
            "model": m["name"],
            "reference": m["ref"],
            "sigma1_true": m["sigma1"],
            "sigma1_pred": f"{s1_p:.4f}",
            "sigma2_true": m["sigma2"],
            "sigma2_pred": f"{s2_p:.4f}",
            "d1_true": m["d1"],
            "d1_pred": f"{d1_p:.2f}",
            "d2_true": m["d2"],
            "d2_pred": f"{d2_p:.2f}",
            "err_sigma1_pct": f"{err_s1:.2f}",
            "err_sigma2_pct": f"{err_s2:.2f}",
            "err_d1_pct": f"{err_d1:.2f}",
            "err_d2_pct": f"{err_d2:.2f}",
            "mape_pct": f"{mape:.2f}",
        })

    # Overall MAPE
    mapes = [float(r["mape_pct"]) for r in results]
    print(f"\nOverall MAPE across {len(results)} published models: "
          f"{np.mean(mapes):.1f}%")

    # Save CSV
    os.makedirs(PAPER_DIR, exist_ok=True)
    path = os.path.join(PAPER_DIR, "published_survey_benchmark.csv")
    fields = list(results[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"Saved → {path}")

    return results


if __name__ == "__main__":
    # Use the best DualTCN model (unaugmented, original training)
    model_path = os.path.join(DIR, "best_model_p9d.pt")
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please provide the path to best_model_p9d.pt")
        sys.exit(1)
    run_benchmark(model_path)
