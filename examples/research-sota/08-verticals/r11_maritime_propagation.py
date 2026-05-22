#!/usr/bin/env python3
"""R11 — Maritime / through-bulkhead RF propagation.

See docs/research/sota-2026-05-22/R11-maritime-sensing.md.

Computes:
  - Steel bulkhead RF attenuation (skin depth) at WiFi bands
  - Seam-leakage diffraction loss
  - Saltwater attenuation (man-overboard surface sensing)
  - Composite link budget for three maritime scenarios

Pure NumPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8
MU_0 = 4 * np.pi * 1e-7      # H/m
EPS_0 = 8.854e-12            # F/m

# Material properties (typical values)
STEEL_SIGMA = 1.0e7          # S/m (mild steel conductivity)
SALTWATER_SIGMA = 4.8        # S/m (35 ppt at 20 deg C)
SALTWATER_EPSR = 81.0        # relative permittivity


def skin_depth_m(freq_ghz: float, sigma: float, mu_r: float = 1.0) -> float:
    """Classical skin depth: delta = 1 / sqrt(pi * f * mu * sigma)."""
    f = freq_ghz * 1e9
    return 1.0 / np.sqrt(np.pi * f * MU_0 * mu_r * sigma)


def bulk_attenuation_db_per_mm(freq_ghz: float, sigma: float, mu_r: float = 1.0) -> float:
    """Per-mm attenuation through bulk conductor."""
    delta = skin_depth_m(freq_ghz, sigma, mu_r)
    # Field decays as exp(-x/delta), power as exp(-2x/delta)
    # In dB per metre: 20/(delta*ln(10)) = 8.686/delta
    return 8.686 / delta / 1000  # divide by 1000 to get per-mm


def saltwater_attenuation_db_per_m(freq_ghz: float) -> float:
    """Saltwater attenuation per metre via lossy-dielectric model.
    alpha = (omega/c) * Im(sqrt(eps_r - j*sigma/(omega*eps_0)))
    Returns dB/m."""
    omega = 2 * np.pi * freq_ghz * 1e9
    eps_complex = SALTWATER_EPSR - 1j * SALTWATER_SIGMA / (omega * EPS_0)
    n_complex = np.sqrt(eps_complex)
    # Principal sqrt of (a - jb), b>0, has negative imag part. The wave
    # attenuation coefficient is alpha = omega/c * |Im(n)| -- take abs().
    alpha = omega * abs(n_complex.imag) / C  # Np/m
    return float(8.686 * alpha)              # dB/m


def seam_diffraction_loss_db(seam_width_mm: float, freq_ghz: float) -> float:
    """Approximate diffraction loss through a narrow slot in a conductor.
    For slot width w << lambda, the slot acts as a high-pass filter:
      L_slot = 20 * log10(lambda / (2 * w))    when w < lambda/2
              0                                 when w >= lambda/2
    Crude but captures the 1st-order physics. Real slot antennas are more
    complex; for forensic 'how much leaks through the door seal' work
    this is the right scale."""
    lam_mm = (C / (freq_ghz * 1e9)) * 1000
    if seam_width_mm >= lam_mm / 2:
        return 0.0
    return max(0.0, 20 * np.log10(lam_mm / (2 * seam_width_mm)))


def maritime_scenario(name: str, freq_ghz: float, bulkhead_mm: float,
                     seam_mm: float, free_air_m: float,
                     saltwater_m: float = 0.0) -> dict:
    """Composite path loss for a maritime sensing scenario."""
    # Free-space loss
    fspl = 32.45 + 20 * np.log10(freq_ghz) + 20 * np.log10(max(0.1, free_air_m + 0.1))
    # Bulkhead loss (if any propagation through metal)
    bulk_loss = bulkhead_mm * bulk_attenuation_db_per_mm(freq_ghz, STEEL_SIGMA)
    # Seam diffraction (alternative path)
    seam_loss = seam_diffraction_loss_db(seam_mm, freq_ghz) if seam_mm > 0 else 999.0
    # Saltwater loss
    water_loss = saltwater_m * saltwater_attenuation_db_per_m(freq_ghz)
    # The actual propagation path takes whichever is lower (bulk OR seam)
    best_metal_path = min(bulk_loss, seam_loss)
    total = fspl + best_metal_path + water_loss
    return {
        "scenario": name,
        "freq_ghz": freq_ghz,
        "fspl_db": fspl,
        "bulk_loss_db": bulk_loss,
        "seam_loss_db": seam_loss,
        "metal_path_used": "seam" if seam_loss < bulk_loss else "bulk",
        "metal_path_loss_db": best_metal_path,
        "saltwater_loss_db": water_loss,
        "total_loss_db": total,
        "esp32_link_budget_db": 121,
        "snr_margin_db": 121 - total - 10,  # 10 dB SNR margin for DSP
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r11_maritime_results.json")
    args = parser.parse_args()

    # 1. Skin depth + per-mm attenuation
    materials_grid = {}
    for f in [2.4, 5.0]:
        delta_steel_um = skin_depth_m(f, STEEL_SIGMA) * 1e6  # micrometres
        att_steel = bulk_attenuation_db_per_mm(f, STEEL_SIGMA)
        att_water = saltwater_attenuation_db_per_m(f)
        materials_grid[f"{f}_GHz"] = {
            "steel_skin_depth_um": delta_steel_um,
            "steel_atten_dB_per_mm": att_steel,
            "saltwater_atten_dB_per_m": att_water,
        }

    # 2. Three maritime scenarios
    scenarios = [
        maritime_scenario("man-overboard, surface-floating", 2.4,
                         bulkhead_mm=0, seam_mm=0, free_air_m=200, saltwater_m=0),
        maritime_scenario("man-overboard, head 30 cm underwater", 2.4,
                         bulkhead_mm=0, seam_mm=0, free_air_m=200, saltwater_m=0.3),
        maritime_scenario("crew vitals through 10 mm steel cabin door (closed)", 2.4,
                         bulkhead_mm=10, seam_mm=0, free_air_m=3),
        maritime_scenario("crew vitals through cabin door (2 mm seam gap)", 2.4,
                         bulkhead_mm=10, seam_mm=2, free_air_m=3),
        maritime_scenario("crew vitals through cabin door (5 mm seam gap)", 2.4,
                         bulkhead_mm=10, seam_mm=5, free_air_m=3),
        maritime_scenario("container intrusion (steel cargo container, 2 mm walls, 30 mm vent slot)", 2.4,
                         bulkhead_mm=2, seam_mm=30, free_air_m=10),
        maritime_scenario("through hull (submarine, 30 mm pressure hull)", 2.4,
                         bulkhead_mm=30, seam_mm=0, free_air_m=1),
    ]

    out = {
        "model": "skin-depth steel + lossy-dielectric saltwater + slot-diffraction seam",
        "materials": materials_grid,
        "scenarios": scenarios,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # Print headlines
    print("=== Skin depth + bulk attenuation ===")
    for fkey, m in materials_grid.items():
        print(f"  {fkey:>8}  steel: skin={m['steel_skin_depth_um']:>6.2f} um, "
              f"attenuation={m['steel_atten_dB_per_mm']:>9.1f} dB/mm    "
              f"saltwater={m['saltwater_atten_dB_per_m']:>6.1f} dB/m")
    print()
    print("=== Composite maritime scenarios @ 2.4 GHz ===")
    print(f"{'Scenario':<58}  {'FSPL':>6}  {'Metal':>6}  {'Water':>6}  {'Total':>6}  {'Margin':>7}")
    for s in scenarios:
        print(f"{s['scenario']:<58}  {s['fspl_db']:>6.1f}  "
              f"{s['metal_path_loss_db']:>6.1f}  {s['saltwater_loss_db']:>6.1f}  "
              f"{s['total_loss_db']:>6.1f}  {s['snr_margin_db']:>+7.1f}")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
