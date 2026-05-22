#!/usr/bin/env python3
"""R20.2 — Threshold-based hand-off fix for ADR-114 Bayesian fusion.

See docs/research/sota-2026-05-22/R20_2-threshold-handoff.md.

R20.1's naive precision-weighted Bayesian fusion gave 84 BPM for HR when
classical (105 BPM, 38% conf) and NV @ 1 m (72 BPM, 64% conf) disagreed.
Production needs threshold-based hand-off: when NV confidence > 60%
AND B-field amplitude > 3 pT, trust NV entirely (reject classical HR).

This implements the fix and verifies it recovers correct HR (72 BPM)
at bedside while gracefully degrading to classical when NV degrades.

Pure NumPy. Reuses R20.1 simulators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Reuse R20.1 simulator functions by importing them
sys.path.insert(0, str(Path(__file__).parent))
from r20_1_quantum_classical_fusion import (
    simulate_csi_breathing,
    simulate_nv_cardiac,
    estimate_rate_from_signal,
    extract_hrv_contour,
)


def fusion_threshold_handoff(classical_rate, classical_conf,
                            nv_rate, nv_conf, nv_amplitude_pT,
                            nv_conf_threshold=0.60,
                            nv_amplitude_threshold_pT=3.0):
    """Threshold-based hand-off:
    - If NV is "good enough" (conf > 0.6 AND amplitude > 3 pT), trust NV entirely.
    - Else fall back to precision-weighted average.
    - If NV has no signal, classical drives.
    """
    nv_trusted = (nv_conf > nv_conf_threshold) and (nv_amplitude_pT > nv_amplitude_threshold_pT)
    if nv_trusted:
        return nv_rate, nv_conf, "nv_drives"
    if classical_conf < 1e-3:
        return nv_rate, nv_conf, "fallback_nv"
    if nv_conf < 1e-3:
        return classical_rate, classical_conf, "fallback_classical"
    # Precision-weighted fallback (R20.1's naive default)
    w_c = classical_conf
    w_n = nv_conf
    fused = (w_c * classical_rate + w_n * nv_rate) / (w_c + w_n + 1e-9)
    conf = float(1 - (1 - classical_conf) * (1 - nv_conf))
    return fused, conf, "weighted_fallback"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/09-quantum-fusion/r20_2_threshold_results.json")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    true_breathing = 15.0
    true_hr = 72.0

    # Same setup as R20.1
    t_csi, csi = simulate_csi_breathing(duration_s=60, fs=50, true_rate_bpm=true_breathing, rng=rng)
    _, csi_hr_conf, _ = estimate_rate_from_signal(t_csi, csi, search_band=(0.8, 3.0))
    csi_hr_rate, csi_hr_conf, _ = estimate_rate_from_signal(t_csi, csi, search_band=(0.8, 3.0))

    # NV at five distances to show degradation
    results = []
    for d in [0.5, 1.0, 1.5, 2.0, 3.0]:
        t_nv, nv, amp = simulate_nv_cardiac(duration_s=60, fs=200, true_hr_bpm=true_hr,
                                            distance_m=d, rng=np.random.default_rng(int(42 + d * 10)))
        nv_rate, nv_conf, nv_snr = estimate_rate_from_signal(t_nv, nv, search_band=(0.8, 3.0))

        # R20.1 naive precision-weighted
        w_c, w_n = csi_hr_conf, nv_conf
        naive = (w_c * csi_hr_rate + w_n * nv_rate) / (w_c + w_n + 1e-9)

        # R20.2 threshold hand-off
        smart, smart_conf, regime = fusion_threshold_handoff(
            csi_hr_rate, csi_hr_conf, nv_rate, nv_conf, amp
        )

        err_naive = abs(naive - true_hr)
        err_smart = abs(smart - true_hr)

        results.append({
            "distance_m": d,
            "nv_amplitude_pT": amp,
            "nv_rate_bpm": nv_rate,
            "nv_conf": nv_conf,
            "naive_fused_bpm": naive,
            "smart_fused_bpm": smart,
            "regime": regime,
            "true_hr_bpm": true_hr,
            "naive_error_bpm": err_naive,
            "smart_error_bpm": err_smart,
        })

    out = {
        "true_hr_bpm": true_hr,
        "classical_hr_rate": csi_hr_rate,
        "classical_hr_conf": csi_hr_conf,
        "results_per_distance": results,
        "thresholds": {"nv_conf": 0.60, "nv_amplitude_pT": 3.0},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"=== R20.2 threshold-based hand-off ===")
    print(f"True HR: {true_hr} BPM")
    print(f"Classical HR: {csi_hr_rate:.2f} BPM (conf {csi_hr_conf*100:.1f}%)")
    print()
    print(f"{'distance':>9}  {'NV amp':>8}  {'NV rate':>8}  {'NV conf':>8}  {'naive':>7}  {'naive err':>9}  {'smart':>7}  {'smart err':>9}  {'regime':>20}")
    for r in results:
        print(f"{r['distance_m']:>7.1f} m  "
              f"{r['nv_amplitude_pT']:>6.2f} pT  "
              f"{r['nv_rate_bpm']:>6.2f} BPM  "
              f"{r['nv_conf']*100:>6.1f}%  "
              f"{r['naive_fused_bpm']:>5.1f} BPM  "
              f"{r['naive_error_bpm']:>+6.1f} BPM  "
              f"{r['smart_fused_bpm']:>5.1f} BPM  "
              f"{r['smart_error_bpm']:>+6.1f} BPM  "
              f"{r['regime']:>20}")
    print()
    # Total error
    total_naive = sum(r['naive_error_bpm'] for r in results)
    total_smart = sum(r['smart_error_bpm'] for r in results)
    print(f"Total naive error across 5 distances:  {total_naive:.1f} BPM")
    print(f"Total smart error across 5 distances:  {total_smart:.1f} BPM")
    print(f"Improvement factor:                    {total_naive / max(total_smart, 0.1):.2f}x")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
