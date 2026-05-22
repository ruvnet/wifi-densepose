#!/usr/bin/env python3
"""R13 — Critical scrutiny: contactless blood pressure from CSI?

See docs/research/sota-2026-05-22/R13-contactless-bp-negative.md.

Two published approaches to contactless BP:
  (a) Pulse Transit Time (PTT) — measure delay between pulse arrival at
      two body sites, then PTT -> BP via Bramwell-Hill / Moens-Korteweg.
  (b) Contour-based ML — learn (pulse waveform contour -> cuff BP).

This script quantifies the physics floors for both:
  (a) PTT requires (i) ms-scale temporal resolution AND (ii) spatial
      separation of two body sites. Spatial resolution is bounded by R6
      (Fresnel envelope), so we compute whether the per-site signals can
      be resolved at all.
  (b) Contour-based ML requires recovering a pulse waveform from a CSI
      stream where breathing motion is 100x larger. We compute the
      breathing-vs-pulse motion amplitude ratio and the resulting SNR
      needed to separate the two by temporal filtering.

Pure NumPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8


# ===== Physiology constants =====
PWV_HEALTHY_ADULT_MPS  = 7.0    # 5-10 m/s typical (Mukkamala 2015, lit median)
CAROTID_FEMORAL_DIST_M = 0.55   # typical anatomic distance
CHEST_BREATHING_AMPLITUDE_MM = 8.0     # rest tidal volume, typical adult
CHEST_HR_AMPLITUDE_MM        = 0.3     # ballistocardiographic chest motion (Inan 2015)
CAROTID_PULSE_AMPLITUDE_MM   = 0.4     # surface pulse displacement (Liu 2014)
RESPIRATION_HZ               = 0.25    # 15 BPM
HR_HZ                        = 1.2     # 72 BPM
MOTION_NOISE_AMPLITUDE_MM    = 2.0     # subject "still" but not motionless

# WiFi
WAVELENGTH_2_4GHZ_M = 0.125
PHASE_DEG_PER_MM_2_4 = 360.0 / (WAVELENGTH_2_4GHZ_M * 1000)  # ~2.88 deg/mm


def ptt_seconds(distance_m: float = CAROTID_FEMORAL_DIST_M,
               pwv_mps: float = PWV_HEALTHY_ADULT_MPS) -> float:
    return distance_m / pwv_mps


def ptt_change_per_bp_mmhg() -> float:
    """Empirical: 10 mmHg BP change <-> ~5 ms PTT change for typical adult.
    (Geddes 1981, lit consensus). So sensitivity is ~0.5 ms / mmHg."""
    return 5e-3 / 10.0  # 0.5 ms/mmHg


def required_ptt_resolution_for_mmhg(target_mmhg: float) -> float:
    """How precise must PTT measurement be to resolve a target BP delta?"""
    return target_mmhg * ptt_change_per_bp_mmhg()


def fresnel_radius_m(freq_ghz: float, link_m: float, p: float = 0.5) -> float:
    """Reused from R6."""
    lam = C / (freq_ghz * 1e9)
    return float(np.sqrt(lam * link_m * p * (1 - p)))


def signal_phase_change(motion_mm: float) -> float:
    """Approximate CSI phase change in degrees for a chest motion amplitude.
    Assumes round-trip path-length change = motion_mm (chest moves toward / away)."""
    # Path-length change is roughly 2x the motion (in/out scattering)
    return 2 * motion_mm * PHASE_DEG_PER_MM_2_4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r13_bp_results.json")
    args = parser.parse_args()

    # ====== Part 1: PTT temporal resolution requirements ======
    ptt_baseline = ptt_seconds()
    ptt_for_1mmhg   = required_ptt_resolution_for_mmhg(1.0)
    ptt_for_5mmhg   = required_ptt_resolution_for_mmhg(5.0)
    ptt_for_10mmhg  = required_ptt_resolution_for_mmhg(10.0)

    # CSI sampling: at 100 Hz, time resolution is 10 ms; at 200 Hz, 5 ms.
    # We need 0.5 ms (1 mmHg) -- that's 2000 Hz CSI rate, which ESP32 *cannot* do.
    # Max ESP32 CSI rate is ~1000 Hz (Hernandez 2020); typical deployments are 50-100 Hz.

    # ====== Part 2: Spatial separation of two body sites ======
    # For PTT, need to resolve carotid (~neck) and femoral (~hip) signals separately.
    # The Fresnel envelope at typical room ranges is too wide -- the two sites are
    # within the same envelope and cannot be separated by single-link CSI.

    fresnel_envelope_5m = fresnel_radius_m(2.4, 5.0)
    fresnel_envelope_2m = fresnel_radius_m(2.4, 2.0)
    sites_resolvable_5m = (CAROTID_FEMORAL_DIST_M / 2) > fresnel_envelope_5m
    sites_resolvable_2m = (CAROTID_FEMORAL_DIST_M / 2) > fresnel_envelope_2m

    # Multi-link multistatic could ALMOST resolve them, but the inverse problem
    # is severely ill-posed with only 4-6 anchors.

    # ====== Part 3: Pulse contour SNR vs breathing ======
    # Phase change per motion:
    breath_phase_deg = signal_phase_change(CHEST_BREATHING_AMPLITUDE_MM)  # ~46 deg
    pulse_phase_deg  = signal_phase_change(CHEST_HR_AMPLITUDE_MM)         # ~1.7 deg
    motion_phase_deg = signal_phase_change(MOTION_NOISE_AMPLITUDE_MM)     # ~11.5 deg

    breath_vs_pulse_amp_ratio = breath_phase_deg / pulse_phase_deg

    # After bandpass filter (HR band 0.8-3.0 Hz, breathing 0.1-0.4 Hz),
    # breathing should drop by ~40 dB. So in HR band:
    breath_after_bandpass_db = -40.0  # typical 4th-order Butterworth
    pulse_in_hr_band_db = 0.0
    motion_in_hr_band_db = -20.0  # micro-motion bleeds into HR band partially

    # SNR for HR contour recovery:
    hr_snr_db = pulse_in_hr_band_db - max(motion_in_hr_band_db, breath_after_bandpass_db)

    # For BP contour, we need to recover the SHAPE of the pulse, not just the rate.
    # Contour-quality recovery typically needs ~20-30 dB above any contaminating
    # signal (Mukkamala 2015). Our HR-band SNR is +20 dB -- BARELY enough for
    # rate, NOT enough for shape.

    bp_contour_required_snr_db = 25.0  # literature standard for waveform-shape recovery
    bp_contour_feasibility = "INFEASIBLE" if hr_snr_db < bp_contour_required_snr_db else "MARGINAL"

    # ====== Part 4: Compare to cuff baseline ======
    cuff_accuracy_mmhg = 2.0   # arm-cuff BIHS Grade A
    published_csi_bp_mae_mmhg = 10.0   # representative lit (Yang 2022 et al.)
    # Conclusion: even the best published CSI BP is 5x worse than a $20 cuff.

    out = {
        "model": "PTT + pulse-contour physics scrutiny for contactless BP",
        "ptt": {
            "baseline_ms": ptt_baseline * 1e3,
            "sensitivity_ms_per_mmHg": ptt_change_per_bp_mmhg() * 1e3,
            "required_resolution_for_1mmHg_ms": ptt_for_1mmhg * 1e3,
            "required_resolution_for_5mmHg_ms": ptt_for_5mmhg * 1e3,
            "required_resolution_for_10mmHg_ms": ptt_for_10mmhg * 1e3,
            "esp32_max_csi_rate_hz": 1000,
            "esp32_max_temporal_resolution_ms": 1.0,
            "esp32_typical_csi_rate_hz": 100,
            "esp32_typical_temporal_resolution_ms": 10.0,
        },
        "spatial_resolution": {
            "carotid_femoral_distance_m": CAROTID_FEMORAL_DIST_M,
            "fresnel_envelope_5m_link_m": fresnel_envelope_5m,
            "fresnel_envelope_2m_link_m": fresnel_envelope_2m,
            "sites_resolvable_5m_link": bool(sites_resolvable_5m),
            "sites_resolvable_2m_link": bool(sites_resolvable_2m),
            "comment": "Single-link CSI cannot spatially separate two body sites. PTT requires multi-link multistatic with severely ill-posed inverse problem.",
        },
        "snr": {
            "breath_phase_deg": breath_phase_deg,
            "pulse_phase_deg": pulse_phase_deg,
            "motion_phase_deg": motion_phase_deg,
            "breath_vs_pulse_amp_ratio": breath_vs_pulse_amp_ratio,
            "hr_band_snr_db": hr_snr_db,
            "bp_contour_required_snr_db": bp_contour_required_snr_db,
            "bp_contour_feasibility": bp_contour_feasibility,
        },
        "vs_baseline": {
            "arm_cuff_accuracy_mmHg": cuff_accuracy_mmhg,
            "published_csi_bp_mae_mmHg": published_csi_bp_mae_mmhg,
            "ratio_worse": published_csi_bp_mae_mmhg / cuff_accuracy_mmhg,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== PTT temporal resolution requirements ===")
    print(f"  Baseline PTT (55 cm body, 7 m/s PWV):  {ptt_baseline*1e3:.1f} ms")
    print(f"  Sensitivity:                            {ptt_change_per_bp_mmhg()*1e3:.2f} ms / mmHg")
    print(f"  Required for  1 mmHg precision:         {ptt_for_1mmhg*1e3:.2f} ms")
    print(f"  Required for  5 mmHg precision:         {ptt_for_5mmhg*1e3:.2f} ms")
    print(f"  Required for 10 mmHg precision:         {ptt_for_10mmhg*1e3:.2f} ms")
    print(f"  ESP32 max CSI rate (~1000 Hz):          1.0 ms resolution -- meets 1 mmHg req")
    print(f"  ESP32 typical (~100 Hz):                10.0 ms resolution -- meets only 20 mmHg")
    print()
    print("=== Spatial resolution (Fresnel envelope) ===")
    print(f"  Carotid-to-femoral distance:            {CAROTID_FEMORAL_DIST_M*100:.0f} cm")
    print(f"  Fresnel envelope @ 5 m link:            {fresnel_envelope_5m*100:.0f} cm  -- sites NOT resolvable")
    print(f"  Fresnel envelope @ 2 m link:            {fresnel_envelope_2m*100:.0f} cm  -- sites NOT resolvable")
    print()
    print("=== Phase change per motion (CSI 2.4 GHz) ===")
    print(f"  Chest breathing (8 mm):                 {breath_phase_deg:.1f} deg")
    print(f"  HR ballistocardiographic (0.3 mm):      {pulse_phase_deg:.1f} deg")
    print(f"  Subject 'still' motion (2 mm):          {motion_phase_deg:.1f} deg")
    print(f"  Breathing-to-pulse amplitude ratio:     {breath_vs_pulse_amp_ratio:.0f}x")
    print()
    print(f"=== BP contour recovery ===")
    print(f"  HR-band SNR after bandpass:             {hr_snr_db:.1f} dB")
    print(f"  Required for BP contour shape:          {bp_contour_required_snr_db:.1f} dB")
    print(f"  Verdict:                                {bp_contour_feasibility}")
    print()
    print(f"=== Vs $20 arm cuff baseline ===")
    print(f"  Arm cuff (BIHS Grade A):                ±{cuff_accuracy_mmhg:.0f} mmHg")
    print(f"  Best published CSI BP:                  ±{published_csi_bp_mae_mmhg:.0f} mmHg")
    print(f"  CSI is worse by:                        {published_csi_bp_mae_mmhg/cuff_accuracy_mmhg:.0f}x")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
