#!/usr/bin/env python3
"""R6 — Fresnel-zone forward model for CSI sensitivity.

See docs/research/sota-2026-05-22/R6-fresnel-forward-model.md.

For a Tx-Rx link, the first Fresnel zone is a prolate ellipsoid whose
radius at fractional position p (0..1) along the LOS path is:

    r_n(p) = sqrt(n * lambda * d * p * (1-p))     (for n=1)

A point scatterer that crosses the first Fresnel zone perpendicular to
the LOS introduces a path-length delta:

    delta_l(x) = sqrt(d1^2 + x^2) + sqrt(d2^2 + x^2) - d1 - d2

where x is the perpendicular offset. Phase shift on subcarrier k:

    phi_k = 2 * pi * f_k * delta_l / c

This is the bedrock forward model that the existing `wifi-densepose-signal`
DSP implicitly assumes. We make it explicit so:

1. R12's revision path (PABS basis grounded in Fresnel geometry) has
   somewhere to start.
2. R10's foliage-range estimates can be sanity-checked against Fresnel-
   ellipsoid clearance, not just FSPL + foliage attenuation.
3. Multi-subcarrier interference patterns from real scatterers become
   predictable rather than mysterious.

Pure NumPy — emits a JSON file with the predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8  # speed of light, m/s


def wavelength_m(freq_ghz: float) -> float:
    return C / (freq_ghz * 1e9)


def fresnel_radius_m(freq_ghz: float, link_length_m: float, p: float, n: int = 1) -> float:
    """Radius of the n-th Fresnel zone at fractional link position p.

    p=0 is at Tx, p=1 is at Rx. r is maximum at p=0.5 (midpoint).
    """
    lam = wavelength_m(freq_ghz)
    return float(np.sqrt(n * lam * link_length_m * p * (1.0 - p)))


def path_delta_m(d1: float, d2: float, perpendicular_offset_m: float) -> float:
    """Extra path length introduced by a point scatterer at perpendicular
    offset x from the LOS, with d1 / d2 the Tx- and Rx-side LOS distances."""
    x = perpendicular_offset_m
    return float(np.sqrt(d1**2 + x**2) + np.sqrt(d2**2 + x**2) - (d1 + d2))


def csi_phase_shift_rad(freq_ghz: float, path_delta: float) -> float:
    """Phase shift on a single subcarrier given the path-length delta."""
    return 2 * np.pi * freq_ghz * 1e9 * path_delta / C


def fresnel_zone_classification(freq_ghz: float, link_length_m: float,
                                scatterer_offset_m: float,
                                scatterer_position_m: float) -> str:
    """Is the scatterer inside the n-th Fresnel zone?

    Zone n is the volume where r_{n-1} < |offset| <= r_n.
    """
    p = scatterer_position_m / link_length_m
    if not (0 <= p <= 1):
        return "outside-link"
    abs_off = abs(scatterer_offset_m)
    for n in range(1, 10):
        r = fresnel_radius_m(freq_ghz, link_length_m, p, n)
        if abs_off <= r:
            return f"zone-{n}"
    return "far-field"


def subcarrier_phase_sweep(freq_ghz: float, link_length_m: float,
                          scatterer_offset_m: float,
                          scatterer_position_m: float,
                          n_subcarriers: int = 52,
                          subcarrier_spacing_khz: float = 312.5) -> dict:
    """Predict per-subcarrier phase shift from a single scatterer.

    Uses 802.11n/ac 20 MHz channels: 52 used subcarriers, spaced 312.5 kHz.
    Subcarrier indices -26..26 excluding DC/pilot tones (we don't bother
    excluding here — pure sweep).
    """
    d1 = scatterer_position_m
    d2 = link_length_m - scatterer_position_m
    if d1 <= 0 or d2 <= 0:
        raise ValueError("scatterer_position_m must be strictly inside [0, link_length_m]")
    delta = path_delta_m(d1, d2, scatterer_offset_m)
    # subcarrier frequencies
    sub_offsets_hz = (np.arange(n_subcarriers) - n_subcarriers // 2) * subcarrier_spacing_khz * 1e3
    f_per_sub = freq_ghz * 1e9 + sub_offsets_hz
    phases_rad = 2 * np.pi * f_per_sub * delta / C
    return {
        "path_delta_m": delta,
        "phase_rad_per_subcarrier": phases_rad.tolist(),
        "phase_rad_min": float(phases_rad.min()),
        "phase_rad_max": float(phases_rad.max()),
        "phase_rad_spread": float(phases_rad.max() - phases_rad.min()),
        "phase_wraps": int(np.floor((phases_rad.max() - phases_rad.min()) / (2 * np.pi))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_fresnel_results.json")
    args = parser.parse_args()

    # Scenario: 5-metre indoor link (typical bedroom/lab setup)
    link_lengths = [2.0, 5.0, 10.0]
    freqs = [2.4, 5.0]
    p_grid = [0.1, 0.25, 0.5, 0.75, 0.9]  # link position fractions

    out = {
        "model": "first-Fresnel-zone ellipsoid + per-subcarrier path-delta forward model",
        "constants": {"c_mps": C},
        "scenarios": [],
    }

    # 1. First Fresnel radii (the basic envelope)
    fresnel = {}
    for f in freqs:
        fresnel[str(f)] = {}
        lam = wavelength_m(f)
        fresnel[str(f)]["wavelength_mm"] = lam * 1000
        for L in link_lengths:
            radii = {f"p={p:.2f}": fresnel_radius_m(f, L, p, n=1) for p in p_grid}
            fresnel[str(f)][f"link_{L}m"] = radii
    out["first_fresnel_radii_m"] = fresnel

    # 2. Single-scatterer per-subcarrier sweep
    # Scatterer at midpoint, 10 cm off LOS (human standing near link)
    scenarios = [
        ("human-standing-at-midpoint", 5.0, 0.10, 2.5),
        ("human-walking-into-fresnel", 5.0, 0.25, 2.5),
        ("scatterer-outside-fresnel", 5.0, 1.50, 2.5),
        ("scatterer-near-Tx",          5.0, 0.05, 0.5),
    ]
    for name, L, x_off, x_pos in scenarios:
        case = {"name": name, "link_m": L, "scatterer_offset_m": x_off,
                "scatterer_position_m": x_pos}
        for f in freqs:
            r1 = fresnel_radius_m(f, L, x_pos / L, n=1)
            zone = fresnel_zone_classification(f, L, x_off, x_pos)
            sweep = subcarrier_phase_sweep(f, L, x_off, x_pos)
            case[f"freq_{f}_GHz"] = {
                "first_fresnel_radius_m": r1,
                "zone": zone,
                **sweep,
            }
        out["scenarios"].append(case)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== First Fresnel zone radii (m) ===")
    print(f"{'freq':>5} {'lambda':>8}  {'link':>5}  " + "  ".join(f"p={p:.2f}" for p in p_grid))
    for f in freqs:
        lam_mm = wavelength_m(f) * 1000
        for L in link_lengths:
            radii = [fresnel_radius_m(f, L, p, n=1) for p in p_grid]
            row = f"{f:>5.1f} {lam_mm:>5.1f}mm {L:>4.1f}m  " + "  ".join(f"{r:>6.3f}" for r in radii)
            print(row)
    print()

    print("=== Single-scatterer per-subcarrier predictions ===")
    for case in out["scenarios"]:
        print(f"{case['name']:>32}  ", end="")
        for f in freqs:
            k = f"freq_{f}_GHz"
            v = case[k]
            print(f"{f:.1f}GHz: r1={v['first_fresnel_radius_m']*100:.1f}cm "
                  f"zone={v['zone']:<8}  "
                  f"phase-spread={np.degrees(v['phase_rad_spread']):.3f} deg  "
                  f"wraps={v['phase_wraps']}", end="  ")
        print()
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
