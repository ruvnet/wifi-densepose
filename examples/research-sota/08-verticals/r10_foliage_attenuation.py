#!/usr/bin/env python3
"""R10 — through-foliage WiFi attenuation curves (ITU-R P.833 + per-species gait).

See docs/research/sota-2026-05-22/R10-through-foliage-wildlife.md.

Plots the ITU-R P.833 vegetation specific attenuation A_v over distance
for 2.4 GHz and 5 GHz CSI bands across three foliage densities. Compares
to a 1×1 SISO ESP32-S3's link budget to derive a maximum sensing range.
Pure NumPy, no plotting libs — emits a JSON file with the curves so a
downstream consumer can render them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def itu_p833_attenuation(freq_ghz: float, distance_m: float, foliage_density: str) -> float:
    """ITU-R P.833 specific-attenuation model for in-foliage propagation.

    Simplified parameterisation:
      A_max  = max attenuation through dense canopy (dB)
      gamma  = decay coefficient (1/m)

      A_v(d) = A_max * (1 - exp(-gamma * d))

    Realistic A_max / gamma per density (calibrated against in-leaf summer
    deciduous from ITU-R P.833-9 Table 1 + simulation studies):
      sparse  (orchard, savanna)     A_max=20 dB, gamma=0.10
      moderate (suburban tree cover) A_max=35 dB, gamma=0.20
      dense    (rainforest canopy)   A_max=50 dB, gamma=0.35
    The constant gets multiplied by sqrt(f_GHz / 1) for frequency scaling.
    """
    params = {
        "sparse":   (20.0, 0.10),
        "moderate": (35.0, 0.20),
        "dense":    (50.0, 0.35),
    }
    a_max, gamma = params[foliage_density]
    freq_scaling = np.sqrt(freq_ghz)  # higher freq → more attenuation
    return a_max * freq_scaling * (1.0 - np.exp(-gamma * distance_m))


def esp32_link_budget(freq_ghz: float) -> dict[str, float]:
    """ESP32-S3 1x1 SISO link budget at 2.4 / 5 GHz.

    Numbers from Espressif ESP32-S3 datasheet + standard WiFi specs:
      Tx power (max regulatory)       +20 dBm  (100 mW, FCC Part 15)
      Tx antenna gain (PCB)           +2 dBi
      Rx antenna gain (PCB)           +2 dBi
      Rx sensitivity (HT20, MCS0)    -97 dBm
    Total link budget (free-space)   = (20 + 2 + 2) - (-97) = 121 dB
    """
    return {
        "tx_power_dbm": 20.0,
        "tx_gain_dbi": 2.0,
        "rx_gain_dbi": 2.0,
        "rx_sensitivity_dbm": -97.0,
        "link_budget_db": 121.0,
    }


def fspl_db(freq_ghz: float, distance_m: float) -> float:
    """Free-space path loss in dB. FSPL = 20·log10(4π·d/λ)
    With f in GHz + d in m: FSPL = 32.45 + 20·log10(f) + 20·log10(d)"""
    if distance_m <= 0: return 0.0
    return 32.45 + 20 * np.log10(freq_ghz) + 20 * np.log10(distance_m)


def max_sensing_range(freq_ghz: float, foliage_density: str, snr_margin_db: float = 10.0) -> float:
    """Distance at which FSPL + foliage_attenuation = link_budget - snr_margin.
    Numerical solve by binary search. Returns metres."""
    lb = esp32_link_budget(freq_ghz)
    budget = lb["link_budget_db"] - snr_margin_db  # require SNR > snr_margin
    lo, hi = 0.1, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        total_loss = fspl_db(freq_ghz, mid) + itu_p833_attenuation(freq_ghz, mid, foliage_density)
        if total_loss < budget:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def gait_frequency_band(species: str) -> dict[str, float]:
    """Approximate gait stride-frequency bands per species class, from
    biomechanics literature (Schmitt 2003, Gambaryan 1974, Heglund 1988).
    These are the temporal frequencies a CSI motion-band filter would
    target — for context, human walking is ~1.7 Hz, jogging ~2.5 Hz."""
    bands = {
        "human-walking":   {"min_hz": 1.2, "max_hz": 2.5},
        "deer":            {"min_hz": 1.8, "max_hz": 4.0},
        "wolf":            {"min_hz": 1.5, "max_hz": 3.5},
        "bear":            {"min_hz": 0.5, "max_hz": 1.5},
        "fox":             {"min_hz": 2.0, "max_hz": 4.5},
        "squirrel":        {"min_hz": 4.0, "max_hz": 10.0},
        "mouse":           {"min_hz": 5.0, "max_hz": 15.0},
        "raccoon":         {"min_hz": 1.5, "max_hz": 3.5},
        "wild-boar":       {"min_hz": 1.0, "max_hz": 2.5},
        "elk":             {"min_hz": 1.5, "max_hz": 3.0},
    }
    return bands.get(species, {"min_hz": 0.5, "max_hz": 10.0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r10_foliage_results.json")
    args = parser.parse_args()

    distances = np.array([1, 2, 5, 10, 20, 50, 100, 200], dtype=np.float64)
    freqs = [2.4, 5.0]
    densities = ["sparse", "moderate", "dense"]

    curves = {}
    for freq in freqs:
        curves[str(freq)] = {}
        for density in densities:
            atts = [float(itu_p833_attenuation(freq, d, density)) for d in distances]
            fspls = [float(fspl_db(freq, d)) for d in distances]
            curves[str(freq)][density] = {
                "distance_m": distances.tolist(),
                "foliage_attenuation_db": atts,
                "fspl_db": fspls,
                "total_loss_db": [a + f for a, f in zip(atts, fspls)],
            }

    # Max sensing range per (freq, density)
    max_ranges = {}
    for freq in freqs:
        max_ranges[str(freq)] = {d: float(max_sensing_range(freq, d)) for d in densities}

    species_gaits = {s: gait_frequency_band(s) for s in
                     ["human-walking", "deer", "wolf", "bear", "fox",
                      "squirrel", "mouse", "raccoon", "wild-boar", "elk"]}

    out = {
        "model": "ITU-R P.833-9 specific-attenuation + free-space-path-loss",
        "link_budget": esp32_link_budget(2.4),
        "snr_margin_db": 10.0,
        "curves": curves,
        "max_sensing_range_m": max_ranges,
        "species_gait_bands_hz": species_gaits,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== ESP32-S3 through-foliage sensing range (link budget 121 dB, 10 dB SNR margin) ===")
    print(f"{'freq (GHz)':>10}  {'sparse':>9}  {'moderate':>11}  {'dense':>9}")
    for freq in freqs:
        row = f"{freq:>10.1f}  "
        for d in densities:
            row += f"{max_ranges[str(freq)][d]:>9.1f}m  " if d != "moderate" else f"{max_ranges[str(freq)][d]:>11.1f}m  "
        print(row)
    print()
    print("=== Per-species gait frequency bands (Hz) ===")
    for s, b in species_gaits.items():
        print(f"  {s:<16}  {b['min_hz']:.1f} - {b['max_hz']:.1f} Hz")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
