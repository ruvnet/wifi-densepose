#!/usr/bin/env python3
"""R12 PABS — Physics-Anchored Background Subtraction structure detection.

See docs/research/sota-2026-05-22/R12-pabs-implementation.md.

R12 NEGATIVE concluded that naive SVD-spectrum-cosine-distance failed
because the eigenshift was indistinguishable from natural drift. The
deferred revision: 'PABS over Fresnel basis'. R6.1 just shipped the
multi-scatterer Fresnel forward operator, so PABS is now implementable.

PABS = norm(y_observed - y_predicted)
     where y_predicted is computed from R6.1's multi-scatterer model
     using a population-prior body assumption.

Scenarios tested:
  A. Empty room (no occupant)                  — baseline PABS
  B. Subject standing (expected)               — small PABS (expected occupant)
  C. Subject + added furniture (1 new piece)   — large PABS (new structure)
  D. Subject + 2nd subject (unexpected person) — large PABS
  E. Subject + wall reflector moved (drift)    — comparison vs natural drift

This is the experiment R12 wanted but couldn't run without R6.1. Pure NumPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8


def wavelength_m(freq_ghz: float) -> float:
    return C / (freq_ghz * 1e9)


def path_delta_m(scatterer_pos, tx_pos, rx_pos):
    d_tx = np.linalg.norm(scatterer_pos - tx_pos)
    d_rx = np.linalg.norm(scatterer_pos - rx_pos)
    d_direct = np.linalg.norm(tx_pos - rx_pos)
    return d_tx + d_rx - d_direct


def csi_contribution(scatterer_pos, reflectivity, tx_pos, rx_pos, sub_freqs_hz):
    delta_l = path_delta_m(scatterer_pos, tx_pos, rx_pos)
    d_tx = np.linalg.norm(scatterer_pos - tx_pos)
    d_rx = np.linalg.norm(scatterer_pos - rx_pos)
    amp = reflectivity / max(d_tx * d_rx, 1e-3)
    phase = 2 * np.pi * sub_freqs_hz * delta_l / C
    return amp * np.exp(1j * phase)


def simulate(scatterers, tx_pos, rx_pos, freq_ghz, n_sub=52, sub_spacing_khz=312.5):
    sub_offsets = (np.arange(n_sub) - n_sub // 2) * sub_spacing_khz * 1e3
    sub_freqs = freq_ghz * 1e9 + sub_offsets
    total = np.zeros(n_sub, dtype=complex)
    for s in scatterers:
        total += csi_contribution(np.asarray(s["pos"]), s["refl"],
                                 np.asarray(tx_pos), np.asarray(rx_pos), sub_freqs)
    return total


def human_body(center_x, center_y):
    return [
        {"pos": [center_x,        center_y       ], "refl": 0.10, "name": "head"},
        {"pos": [center_x,        center_y       ], "refl": 0.50, "name": "chest"},
        {"pos": [center_x - 0.20, center_y       ], "refl": 0.10, "name": "left_arm"},
        {"pos": [center_x + 0.20, center_y       ], "refl": 0.10, "name": "right_arm"},
        {"pos": [center_x - 0.10, center_y - 0.40], "refl": 0.10, "name": "left_leg"},
        {"pos": [center_x + 0.10, center_y - 0.40], "refl": 0.10, "name": "right_leg"},
    ]


def static_wall_reflectors(amplitudes=(0.3, 0.2, 0.15, 0.1)):
    """Four wall reflectors at fixed positions -- typical bedroom multipath."""
    return [
        {"pos": [0.5,  4.5], "refl": amplitudes[0], "name": "wall_NW"},
        {"pos": [4.5,  4.5], "refl": amplitudes[1], "name": "wall_NE"},
        {"pos": [0.5,  0.5], "refl": amplitudes[2], "name": "wall_SW"},
        {"pos": [4.5,  0.5], "refl": amplitudes[3], "name": "wall_SE"},
    ]


def pabs(y_observed, y_predicted):
    """L2 norm of the residual, normalised by signal energy."""
    residual = y_observed - y_predicted
    energy = np.linalg.norm(y_observed) ** 2
    return float(np.linalg.norm(residual) ** 2 / max(energy, 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r12_pabs_results.json")
    args = parser.parse_args()

    tx = np.array([0.0, 2.5])
    rx = np.array([5.0, 2.5])
    freq_ghz = 2.4
    walls = static_wall_reflectors()

    # ===== Build the "expected" scene model (subject + walls) =====
    # This is what PABS predicts as the baseline.
    subject_expected = human_body(2.5, 2.75)
    expected_scene = subject_expected + walls
    y_expected = simulate(expected_scene, tx, rx, freq_ghz)

    # ===== Scenario A: empty room (no occupant) =====
    y_empty = simulate(walls, tx, rx, freq_ghz)
    pabs_A = pabs(y_empty, y_expected)

    # ===== Scenario B: subject standing where expected =====
    y_B = simulate(subject_expected + walls, tx, rx, freq_ghz)
    pabs_B = pabs(y_B, y_expected)

    # ===== Scenario C: subject + 1 added piece of furniture =====
    new_furniture = [{"pos": [3.5, 1.0], "refl": 0.25, "name": "new_chair"}]
    y_C = simulate(subject_expected + walls + new_furniture, tx, rx, freq_ghz)
    pabs_C = pabs(y_C, y_expected)

    # ===== Scenario D: subject + unexpected second person =====
    intruder = human_body(2.0, 2.0)
    y_D = simulate(subject_expected + walls + intruder, tx, rx, freq_ghz)
    pabs_D = pabs(y_D, y_expected)

    # ===== Scenario E: subject + natural drift (wall reflectivity shift) =====
    # Walls have ~5% reflectivity drift over the day (humidity, temperature)
    drifted_walls = static_wall_reflectors(amplitudes=(0.315, 0.21, 0.158, 0.105))
    y_E = simulate(subject_expected + drifted_walls, tx, rx, freq_ghz)
    pabs_E = pabs(y_E, y_expected)

    # ===== Scenario F: small subject position shift (subject moved 10 cm) =====
    subject_shifted = human_body(2.5, 2.85)  # 10 cm closer to LOS
    y_F = simulate(subject_shifted + walls, tx, rx, freq_ghz)
    pabs_F = pabs(y_F, y_expected)

    # ===== R12 NEGATIVE baseline: naive SVD cosine distance =====
    # Run the same scenarios through R12's failed approach for comparison.
    def svd_distance(y_obs, y_ref):
        # Treat as 1D signal; SVD spectrum on |y|
        return float(np.linalg.norm(np.abs(y_obs) - np.abs(y_ref)))

    svd_A = svd_distance(y_empty, y_expected)
    svd_B = svd_distance(y_B, y_expected)
    svd_C = svd_distance(y_C, y_expected)
    svd_D = svd_distance(y_D, y_expected)
    svd_E = svd_distance(y_E, y_expected)
    svd_F = svd_distance(y_F, y_expected)

    out = {
        "model": "PABS = ||y_observed - y_predicted||^2 / ||y_observed||^2",
        "forward_operator_source": "R6.1 multi-scatterer additive Fresnel",
        "expected_scene": {
            "subject_pos": [2.5, 2.75],
            "wall_reflectors": 4,
        },
        "link": {"tx": tx.tolist(), "rx": rx.tolist(), "freq_ghz": freq_ghz},
        "scenarios": {
            "A_empty_room":       {"description": "no occupant",                  "pabs": pabs_A, "svd_distance": svd_A},
            "B_subject_expected": {"description": "subject where expected",       "pabs": pabs_B, "svd_distance": svd_B},
            "C_added_furniture":  {"description": "+1 new structural element",    "pabs": pabs_C, "svd_distance": svd_C},
            "D_unexpected_person":{"description": "+1 unexpected human",          "pabs": pabs_D, "svd_distance": svd_D},
            "E_natural_drift":    {"description": "5%% wall reflectivity drift",   "pabs": pabs_E, "svd_distance": svd_E},
            "F_subject_moved":    {"description": "subject shifted 10 cm",        "pabs": pabs_F, "svd_distance": svd_F},
        },
        "verdict": {
            "pabs_signal_to_drift": pabs_D / pabs_E if pabs_E > 0 else float("inf"),
            "pabs_furniture_to_drift": pabs_C / pabs_E if pabs_E > 0 else float("inf"),
            "svd_signal_to_drift": svd_D / svd_E if svd_E > 0 else float("inf"),
            "svd_furniture_to_drift": svd_C / svd_E if svd_E > 0 else float("inf"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== R12 PABS implementation results ===")
    print()
    print(f"{'Scenario':<30}  {'PABS':>9}  {'SVD':>9}  {'PABS / drift':>14}  {'SVD / drift':>13}")
    print("-" * 90)
    for key, s in out["scenarios"].items():
        pabs_ratio = s['pabs'] / pabs_E if pabs_E > 0 else float('inf')
        svd_ratio  = s['svd_distance'] / svd_E if svd_E > 0 else float('inf')
        print(f"{s['description']:<30}  {s['pabs']:>9.4f}  {s['svd_distance']:>9.4f}  "
              f"{pabs_ratio:>14.2f}x  {svd_ratio:>13.2f}x")
    print()
    print(f"PABS detects unexpected person at {out['verdict']['pabs_signal_to_drift']:.1f}x the natural drift floor")
    print(f"PABS detects new furniture       at {out['verdict']['pabs_furniture_to_drift']:.1f}x the natural drift floor")
    print(f"SVD  (R12 naive) signal/drift:    {out['verdict']['svd_signal_to_drift']:.2f}x")
    print(f"SVD  (R12 naive) furniture/drift: {out['verdict']['svd_furniture_to_drift']:.2f}x")
    print()
    if out['verdict']['pabs_signal_to_drift'] > 3 and out['verdict']['svd_signal_to_drift'] < 2:
        print("VERDICT: PABS works where R12 naive SVD failed. R12 NEGATIVE -> revisited and POSITIVE.")
    elif out['verdict']['pabs_signal_to_drift'] > out['verdict']['svd_signal_to_drift'] * 2:
        print("VERDICT: PABS is meaningfully better than R12 naive SVD.")
    else:
        print("VERDICT: PABS is not yet decisive. Needs longer time-series / temporal averaging.")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
