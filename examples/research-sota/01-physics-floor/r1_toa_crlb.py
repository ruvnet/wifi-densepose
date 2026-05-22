#!/usr/bin/env python3
"""R1 — Time-of-Arrival CRLB for WiFi multistatic localisation.

See docs/research/sota-2026-05-22/R1-toa-crlb.md.

Computes the Cramer-Rao Lower Bound on ToA precision as a function of
bandwidth and SNR, then compares it to the phase-based ranging precision
unlocked by R6's Fresnel forward model. The headline question:

  At WiFi-grade bandwidths (20 / 40 / 80 / 160 MHz), what is the best
  possible single-shot ranging precision via raw ToA, vs phase-derived
  ranging?

Standard ToA CRLB (Kay '93, Ch 3):

    sigma_ToA  >=  1 / ( 2 * pi * beta * sqrt(SNR) )           [s]
    sigma_d    =  c * sigma_ToA                                [m]

where beta is the effective (RMS) bandwidth. For a brick-wall pulse of
bandwidth B (matched-filter spectrum), beta = B / sqrt(3).

Phase-based ranging precision at carrier f_c (a single subcarrier):

    sigma_d_phi  =  (c / 2 * pi * f_c) * sigma_phi             [m]

where sigma_phi is the phase-noise standard deviation in radians.

Pure NumPy, no plotting libs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8

def toa_crlb_seconds(bandwidth_hz: float, snr_db: float) -> float:
    """ToA CRLB in seconds. Bandwidth is the matched-filter / signal
    bandwidth, NOT the carrier frequency. The factor of sqrt(3) comes
    from the brick-wall pulse RMS bandwidth: beta_rms = B / sqrt(3)."""
    snr_lin = 10 ** (snr_db / 10.0)
    beta_rms = bandwidth_hz / np.sqrt(3.0)
    return 1.0 / (2 * np.pi * beta_rms * np.sqrt(snr_lin))


def range_precision_toa_m(bandwidth_hz: float, snr_db: float) -> float:
    """Single-shot range precision (1 sigma) from ToA CRLB."""
    return C * toa_crlb_seconds(bandwidth_hz, snr_db)


def range_precision_phase_m(carrier_ghz: float, phase_noise_deg: float) -> float:
    """Single-subcarrier phase-based ranging precision. Assumes the
    integer-ambiguity (cycle slips) problem is solved by some other
    method (e.g. multi-subcarrier-frequency unwrap). This is the
    *unambiguous* precision, NOT the absolute distance."""
    sigma_phi = np.deg2rad(phase_noise_deg)
    lam = C / (carrier_ghz * 1e9)
    return lam * sigma_phi / (2 * np.pi)


def averaging_gain(n_samples: int) -> float:
    """Independent-sample averaging gain (1/sqrt(N))."""
    return 1.0 / np.sqrt(n_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r1_toa_crlb_results.json")
    args = parser.parse_args()

    # WiFi-relevant bandwidths
    bandwidths_mhz = [20, 40, 80, 160, 320]  # 802.11n/ac/ax/be
    snrs_db = [0, 10, 20, 30, 40]
    carriers_ghz = [2.4, 5.0, 6.0]

    # 1. ToA CRLB grid
    toa_grid = {}
    for bw_mhz in bandwidths_mhz:
        bw_hz = bw_mhz * 1e6
        col = {}
        for snr_db in snrs_db:
            sigma_t = toa_crlb_seconds(bw_hz, snr_db)
            sigma_d = range_precision_toa_m(bw_hz, snr_db)
            col[f"snr_{snr_db}dB"] = {
                "sigma_toa_ns": sigma_t * 1e9,
                "sigma_range_m": sigma_d,
            }
        toa_grid[f"bw_{bw_mhz}MHz"] = col

    # 2. Phase-based ranging precision (single subcarrier)
    phase_grid = {}
    for ghz in carriers_ghz:
        col = {}
        for phase_noise_deg in [0.5, 1.0, 2.0, 5.0, 10.0]:
            sigma_d = range_precision_phase_m(ghz, phase_noise_deg)
            col[f"sigma_phi_{phase_noise_deg}deg"] = {
                "sigma_range_mm": sigma_d * 1000,
                "sigma_range_m": sigma_d,
            }
        phase_grid[f"carrier_{ghz}GHz"] = col

    # 3. Practical comparison: 20 MHz HT20 channel, 20 dB SNR, 100 averaged samples
    bw_practical_hz = 20e6
    snr_practical = 20
    n_avg = 100

    toa_single = range_precision_toa_m(bw_practical_hz, snr_practical)
    toa_avg = toa_single * averaging_gain(n_avg)
    phase_single = range_precision_phase_m(2.4, 5.0)  # 5 deg phase noise
    phase_avg = phase_single * averaging_gain(n_avg)

    headline = {
        "scenario": "20 MHz HT20 channel, 20 dB SNR, 100 averaged frames",
        "toa_single_shot_m": toa_single,
        "toa_after_100_avg_m": toa_avg,
        "phase_single_shot_m": phase_single,
        "phase_after_100_avg_m": phase_avg,
        "phase_advantage_ratio": toa_single / phase_single,
    }

    # 4. Multistatic geometric dilution: 4 anchor nodes around a 5x5m room,
    # each contributes one range measurement. Position-error CRLB scales
    # with the inverse of the FIM trace, which is roughly:
    #   sigma_pos = sigma_range * sqrt(GDOP / N_anchors)
    # GDOP for a tight 4-anchor convex-hull is ~1.5 (vs ~3 for collinear).
    gdop_tight = 1.5
    n_anchors = 4
    toa_pos_precision = toa_single * np.sqrt(gdop_tight / n_anchors)
    phase_pos_precision = phase_single * np.sqrt(gdop_tight / n_anchors)
    multistatic = {
        "n_anchors": n_anchors,
        "gdop": gdop_tight,
        "toa_position_precision_m": toa_pos_precision,
        "phase_position_precision_m": phase_pos_precision,
    }

    out = {
        "model": "Cramer-Rao Lower Bound on ToA + phase ranging precision",
        "bandwidth_grid": toa_grid,
        "phase_grid": phase_grid,
        "headline_practical": headline,
        "multistatic_4anchor": multistatic,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== ToA single-shot range CRLB (m, 1 sigma) ===")
    hdr = f"{'BW':>8}" + "".join(f"{('SNR=' + str(s) + 'dB'):>12}" for s in snrs_db)
    print(hdr)
    for bw_mhz in bandwidths_mhz:
        row = f"{bw_mhz:>5} MHz"
        for snr_db in snrs_db:
            sigma_d = toa_grid[f"bw_{bw_mhz}MHz"][f"snr_{snr_db}dB"]["sigma_range_m"]
            row += f"{sigma_d:>12.2f}"
        print(row)
    print()
    print("=== Phase-based single-subcarrier range precision (mm, 1 sigma) ===")
    print(f"{'carrier':>9}" + "".join(f"{('phi=' + str(d) + 'deg'):>14}" for d in [0.5, 1, 2, 5, 10]))
    for ghz in carriers_ghz:
        row = f"{ghz:>6.1f} GHz"
        for phase_noise_deg in [0.5, 1.0, 2.0, 5.0, 10.0]:
            v = phase_grid[f"carrier_{ghz}GHz"][f"sigma_phi_{phase_noise_deg}deg"]
            row += f"{v['sigma_range_mm']:>14.2f}"
        print(row)
    print()
    print("=== Headline (20 MHz HT20, 20 dB SNR, 100 averaged frames) ===")
    print(f"  ToA single-shot range CRLB:   {toa_single:>8.3f} m")
    print(f"  ToA after 100x avg:           {toa_avg:>8.3f} m")
    print(f"  Phase single-subcarrier:      {phase_single*1000:>8.2f} mm")
    print(f"  Phase after 100x avg:         {phase_avg*1000:>8.2f} mm")
    print(f"  Phase advantage:              {headline['phase_advantage_ratio']:>8.0f}x")
    print()
    print(f"=== Multistatic 4-anchor convex hull (GDOP {gdop_tight}) ===")
    print(f"  ToA position precision:       {toa_pos_precision:>8.3f} m")
    print(f"  Phase position precision:     {phase_pos_precision*1000:>8.2f} mm")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
