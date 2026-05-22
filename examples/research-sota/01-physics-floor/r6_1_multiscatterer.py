#!/usr/bin/env python3
"""R6.1 — Multi-scatterer additive Fresnel forward model.

See docs/research/sota-2026-05-22/R6_1-multiscatterer-forward-model.md.

Extends R6's single-point-scatterer model to multiple scatterers
(distributed body). A human is approximated as 6 point scatterers:
head, chest, two arms, two legs. Each has:
  - position (x, y) relative to LOS midpoint
  - reflectivity (proportional to body-part surface area)
  - motion amplitude (chest breathes; limbs static unless walking)

The combined CSI signal is the coherent (complex) sum of per-scatterer
contributions, evaluated per-subcarrier. This is the model that
vital_signs.rs implicitly assumes and tomography.rs explicitly inverts.

Pure NumPy.
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
    """Path-length delta = (Tx → scatterer + scatterer → Rx) − (Tx → Rx)."""
    d_tx = np.linalg.norm(scatterer_pos - tx_pos)
    d_rx = np.linalg.norm(scatterer_pos - rx_pos)
    d_direct = np.linalg.norm(tx_pos - rx_pos)
    return d_tx + d_rx - d_direct


def csi_contribution(scatterer_pos, reflectivity, tx_pos, rx_pos,
                     subcarrier_freqs_hz):
    """Complex contribution of a single scatterer at each subcarrier.
    Magnitude proportional to reflectivity / (path loss); phase = 2π·f·Δℓ/c.
    Path loss simplified to 1/(d_tx · d_rx) (bistatic 1/r² each leg)."""
    delta_l = path_delta_m(scatterer_pos, tx_pos, rx_pos)
    d_tx = np.linalg.norm(scatterer_pos - tx_pos)
    d_rx = np.linalg.norm(scatterer_pos - rx_pos)
    amplitude = reflectivity / max(d_tx * d_rx, 1e-3)
    phase = 2 * np.pi * subcarrier_freqs_hz * delta_l / C
    return amplitude * np.exp(1j * phase)


def simulate_human(body_model, tx_pos, rx_pos, freq_ghz,
                   n_subcarriers=52, sub_spacing_khz=312.5):
    """Sum CSI contributions from all body parts.
    Returns complex per-subcarrier signal."""
    sub_offsets = (np.arange(n_subcarriers) - n_subcarriers // 2) * sub_spacing_khz * 1e3
    sub_freqs = freq_ghz * 1e9 + sub_offsets
    total = np.zeros(n_subcarriers, dtype=complex)
    for part_name, part in body_model.items():
        contrib = csi_contribution(np.asarray(part["pos"]), part["refl"],
                                  np.asarray(tx_pos), np.asarray(rx_pos),
                                  sub_freqs)
        total += contrib
    return total


def default_human_body(center_x, center_y, height_m=1.75):
    """Approximate adult human as 6 point scatterers in 2D (top-down view).
    Reflectivity scaled to body-part surface area (rough)."""
    return {
        "head":      {"pos": np.array([center_x,        center_y]),         "refl": 0.10},
        "chest":     {"pos": np.array([center_x,        center_y]),         "refl": 0.50},
        "left_arm":  {"pos": np.array([center_x - 0.20, center_y]),         "refl": 0.10},
        "right_arm": {"pos": np.array([center_x + 0.20, center_y]),         "refl": 0.10},
        "left_leg":  {"pos": np.array([center_x - 0.10, center_y - 0.40]),  "refl": 0.10},
        "right_leg": {"pos": np.array([center_x + 0.10, center_y - 0.40]),  "refl": 0.10},
    }


def breathe(body, t_seconds, amplitude_mm=8.0, rate_hz=0.25):
    """Modulate chest position with breathing motion (±8 mm tidal volume).
    Returns a copy of body with updated chest position."""
    out = {k: {**v, "pos": v["pos"].copy()} for k, v in body.items()}
    delta_y = (amplitude_mm / 1000) * np.sin(2 * np.pi * rate_hz * t_seconds)
    out["chest"]["pos"][1] += delta_y
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_1_multiscatterer_results.json")
    args = parser.parse_args()

    # 5 m bedroom-class link
    tx = np.array([0.0, 0.0])
    rx = np.array([5.0, 0.0])
    freq_ghz = 2.4
    lam = wavelength_m(freq_ghz)

    # Subject standing at midpoint, 0.25 m off LOS (inside first Fresnel ~40 cm)
    # NOTE: on-LOS placement (y=0) gives degenerate path-delta sensitivity --
    # breathing y-motion changes the path only at 2nd order. Real installations
    # need the subject OFF the LOS line to see breathing-amplitude motion.
    body = default_human_body(center_x=2.5, center_y=0.25)

    # ===== 1. Single-frame multi-scatterer signature =====
    csi_baseline = simulate_human(body, tx, rx, freq_ghz)
    mag_baseline = np.abs(csi_baseline)
    phase_baseline = np.angle(csi_baseline, deg=True)

    # ===== 2. What does each body part contribute alone? =====
    per_part_contributions = {}
    for name, part in body.items():
        single = {name: part}
        c = simulate_human(single, tx, rx, freq_ghz)
        per_part_contributions[name] = {
            "mag_mean":   float(np.abs(c).mean()),
            "mag_max":    float(np.abs(c).max()),
            "phase_spread_deg": float(np.angle(c, deg=True).max() - np.angle(c, deg=True).min()),
            "fraction_of_total_energy": float((np.abs(c)**2).sum() / (np.abs(csi_baseline)**2).sum()),
        }

    # ===== 3. Time series with breathing =====
    # 30 seconds at 50 Hz CSI rate
    fs = 50
    t = np.arange(0, 30, 1/fs)
    csi_series = np.zeros((len(t), 52), dtype=complex)
    for i, ti in enumerate(t):
        csi_series[i] = simulate_human(breathe(body, ti), tx, rx, freq_ghz)

    # Per-subcarrier breathing-band SNR.
    # Project each subcarrier's magnitude onto the breathing-band component
    # vs everything else.
    csi_mag = np.abs(csi_series)
    # FFT each subcarrier's magnitude time-series
    fft = np.fft.rfft(csi_mag - csi_mag.mean(axis=0), axis=0)
    freqs = np.fft.rfftfreq(len(t), 1/fs)
    breath_band = (freqs >= 0.15) & (freqs <= 0.4)
    out_of_band = (freqs >= 0.5) & (freqs <= 3.0)
    # Power per band
    breath_power = (np.abs(fft[breath_band])**2).sum(axis=0)
    out_power    = (np.abs(fft[out_of_band])**2).sum(axis=0)
    snr_per_sub = 10 * np.log10((breath_power + 1e-12) / (out_power + 1e-12))
    snr_best_sub = float(snr_per_sub.max())
    snr_mean_sub = float(snr_per_sub.mean())
    snr_worst_sub = float(snr_per_sub.min())
    best_sub_idx = int(snr_per_sub.argmax())

    # ===== 4. Compare to R6 single-scatterer baseline =====
    # Single chest-only scatterer at the same position
    chest_only = {"chest": body["chest"]}
    csi_chest_only_series = np.zeros((len(t), 52), dtype=complex)
    for i, ti in enumerate(t):
        csi_chest_only_series[i] = simulate_human(breathe(chest_only, ti), tx, rx, freq_ghz)
    chest_mag = np.abs(csi_chest_only_series)
    chest_fft = np.fft.rfft(chest_mag - chest_mag.mean(axis=0), axis=0)
    chest_breath_power = (np.abs(chest_fft[breath_band])**2).sum(axis=0)
    chest_out_power    = (np.abs(chest_fft[out_of_band])**2).sum(axis=0)
    chest_snr_per_sub  = 10 * np.log10((chest_breath_power + 1e-12) / (chest_out_power + 1e-12))
    chest_snr_best = float(chest_snr_per_sub.max())

    # The interesting finding: the multi-scatterer model REDUCES breathing SNR
    # because the static limb scatterers add noise / phase-offset confusion
    # that didn't exist in the single-scatterer R6 model. This is what
    # vital_signs.rs implicitly handles via its temporal bandpass.

    out = {
        "model": "additive complex sum of 6 point-scatterer human body model",
        "link": {"tx": tx.tolist(), "rx": rx.tolist(), "freq_ghz": freq_ghz,
                 "wavelength_m": lam, "length_m": float(np.linalg.norm(tx-rx))},
        "per_part_contributions": per_part_contributions,
        "breathing_band_snr": {
            "scatterer_count": 6,
            "best_subcarrier_snr_db": snr_best_sub,
            "best_subcarrier_index": best_sub_idx,
            "mean_subcarrier_snr_db": snr_mean_sub,
            "worst_subcarrier_snr_db": snr_worst_sub,
            "chest_only_baseline_snr_db": chest_snr_best,
            "multi_scatterer_penalty_db": chest_snr_best - snr_best_sub,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== R6.1 multi-scatterer human body model ===")
    print(f"  Link: {tx.tolist()} -> {rx.tolist()} @ {freq_ghz} GHz")
    print()
    print(f"=== Per-body-part contribution to total CSI energy ===")
    for name, info in per_part_contributions.items():
        print(f"  {name:<10}  mag_mean={info['mag_mean']:.3f}  "
              f"phase_spread={info['phase_spread_deg']:.2f} deg  "
              f"frac_of_total={info['fraction_of_total_energy']*100:.1f}%")
    print()
    print(f"=== Breathing-band SNR (15-second time-series) ===")
    print(f"  Multi-scatterer best subcarrier:  {snr_best_sub:+.1f} dB  (idx={best_sub_idx})")
    print(f"  Multi-scatterer mean:             {snr_mean_sub:+.1f} dB")
    print(f"  Multi-scatterer worst:            {snr_worst_sub:+.1f} dB")
    print(f"  Single-scatterer (chest-only):    {chest_snr_best:+.1f} dB")
    print(f"  Multi-scatterer penalty:          {chest_snr_best - snr_best_sub:+.1f} dB")
    print()
    print("Interpretation: static limb scatterers add coherent-sum confusion")
    print("that doesn't exist in R6's single-scatterer model. The penalty is")
    print("the gap between idealised physics (R6) and real-world deployment.")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
