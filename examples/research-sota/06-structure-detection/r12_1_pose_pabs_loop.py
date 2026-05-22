#!/usr/bin/env python3
"""R12.1 — Pose-PABS closed loop.

See docs/research/sota-2026-05-22/R12_1-pose-pabs-closed-loop.md.

R12 PABS (tick 19) had a false-alarm problem: subject moving 10 cm gave
PABS = 22,000x natural drift floor. R12 PABS noted: 'Real production
PABS needs a pose-aware forward model updating from pose_tracker.rs in
real-time. The actual structure-detection signal is PABS-after-pose-
update.'

This tick implements the closed loop in synthetic form:
  1. Subject moves on a continuous trajectory
  2. 'Pose tracker' estimates the subject position (with noise)
  3. Forward model uses the ESTIMATED position to predict expected CSI
  4. PABS = |observed - expected| using the pose-updated expected
  5. At tick T_intrude, insert an unexpected second subject
  6. Measure: does PABS-after-pose-update spike at T_intrude vs being
     noisy during subject motion?

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


def csi_contribution(pos, refl, tx, rx, sub_freqs_hz):
    d_tx = np.linalg.norm(pos - tx)
    d_rx = np.linalg.norm(pos - rx)
    d_direct = np.linalg.norm(tx - rx)
    delta_l = d_tx + d_rx - d_direct
    amp = refl / max(d_tx * d_rx, 1e-3)
    phase = 2 * np.pi * sub_freqs_hz * delta_l / C
    return amp * np.exp(1j * phase)


def simulate(scatterers, tx, rx, freq_ghz, n_sub=52, sub_spacing_khz=312.5):
    sub_offsets = (np.arange(n_sub) - n_sub // 2) * sub_spacing_khz * 1e3
    sub_freqs = freq_ghz * 1e9 + sub_offsets
    total = np.zeros(n_sub, dtype=complex)
    for s in scatterers:
        total += csi_contribution(np.asarray(s["pos"]), s["refl"],
                                 np.asarray(tx), np.asarray(rx), sub_freqs)
    return total


def human_body(cx, cy):
    return [
        {"pos": [cx,        cy       ], "refl": 0.10},  # head
        {"pos": [cx,        cy       ], "refl": 0.50},  # chest
        {"pos": [cx - 0.20, cy       ], "refl": 0.10},  # arms
        {"pos": [cx + 0.20, cy       ], "refl": 0.10},
        {"pos": [cx - 0.10, cy - 0.40], "refl": 0.10},  # legs
        {"pos": [cx + 0.10, cy - 0.40], "refl": 0.10},
    ]


def walls():
    return [
        {"pos": [0.5, 4.5], "refl": 0.30},
        {"pos": [4.5, 4.5], "refl": 0.25},
        {"pos": [0.5, 0.5], "refl": 0.20},
        {"pos": [4.5, 0.5], "refl": 0.15},
    ]


def pabs(observed, predicted):
    res = observed - predicted
    e_obs = np.linalg.norm(observed) ** 2
    return float(np.linalg.norm(res) ** 2 / max(e_obs, 1e-12))


def pose_tracker_estimate(true_pos, std_noise=0.05, rng=None):
    """Simulate a pose tracker with ~5 cm position noise.
    Real pose_tracker.rs achieves this at ~95% PCK@20."""
    rng = rng or np.random.default_rng(0)
    return true_pos + rng.standard_normal(2) * std_noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r12_1_pose_pabs_results.json")
    args = parser.parse_args()

    tx = np.array([0.0, 2.5])
    rx = np.array([5.0, 2.5])
    freq = 2.4
    rng = np.random.default_rng(7)

    # Subject walks from (2.0, 2.0) to (3.0, 3.5) over 50 frames
    n_frames = 50
    trajectory = np.linspace([2.0, 2.0], [3.0, 3.5], n_frames)
    walls_static = walls()

    # Intruder enters at frame T_intrude
    T_intrude = 25
    intruder_pos = (1.5, 1.5)

    # Two PABS pipelines:
    # (a) FIXED expected scene (R12 PABS naive — expects subject at start position)
    # (b) POSE-UPDATED expected scene (R12.1 — uses pose-tracker estimate)
    fixed_subject_pos = trajectory[0]  # never updated
    fixed_expected = human_body(*fixed_subject_pos) + walls_static
    y_fixed = simulate(fixed_expected, tx, rx, freq)

    pabs_fixed = []
    pabs_pose_updated = []
    pose_estimates = []

    for t in range(n_frames):
        true_pos = trajectory[t]
        # Build the observed scene
        scene_obs = human_body(*true_pos) + walls_static
        if t >= T_intrude:
            scene_obs = scene_obs + human_body(*intruder_pos)
        y_obs = simulate(scene_obs, tx, rx, freq)

        # (a) Fixed expected
        pabs_fixed.append(pabs(y_obs, y_fixed))

        # (b) Pose-updated expected
        est_pos = pose_tracker_estimate(true_pos, std_noise=0.05, rng=rng)
        pose_estimates.append(est_pos.tolist())
        expected_pose = human_body(*est_pos) + walls_static
        y_pose = simulate(expected_pose, tx, rx, freq)
        pabs_pose_updated.append(pabs(y_obs, y_pose))

    pabs_fixed = np.array(pabs_fixed)
    pabs_pose_updated = np.array(pabs_pose_updated)

    # Analysis:
    # During T<T_intrude: pose-updated should be LOW (pose tracker explains subject)
    # During T>=T_intrude: pose-updated should SPIKE (intruder unexplained)
    # Fixed should be HIGH throughout (subject motion always unexplained)

    pre_intrude_fixed_mean = pabs_fixed[:T_intrude].mean()
    post_intrude_fixed_mean = pabs_fixed[T_intrude:].mean()
    pre_intrude_pose_mean = pabs_pose_updated[:T_intrude].mean()
    post_intrude_pose_mean = pabs_pose_updated[T_intrude:].mean()

    pose_intruder_lift = post_intrude_pose_mean / max(pre_intrude_pose_mean, 1e-9)
    fixed_intruder_lift = post_intrude_fixed_mean / max(pre_intrude_fixed_mean, 1e-9)

    out = {
        "config": {
            "n_frames": n_frames,
            "trajectory_start": trajectory[0].tolist(),
            "trajectory_end": trajectory[-1].tolist(),
            "T_intrude": T_intrude,
            "intruder_pos": list(intruder_pos),
            "pose_tracker_std_m": 0.05,
        },
        "pabs_fixed":        pabs_fixed.tolist(),
        "pabs_pose_updated": pabs_pose_updated.tolist(),
        "pre_intrude_means": {
            "fixed": float(pre_intrude_fixed_mean),
            "pose":  float(pre_intrude_pose_mean),
        },
        "post_intrude_means": {
            "fixed": float(post_intrude_fixed_mean),
            "pose":  float(post_intrude_pose_mean),
        },
        "intruder_detection_lift": {
            "fixed":  fixed_intruder_lift,
            "pose":   pose_intruder_lift,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== R12.1 pose-PABS closed loop ===")
    print(f"  Subject walks {n_frames} frames from {trajectory[0]} to {trajectory[-1]}")
    print(f"  Intruder enters at frame {T_intrude} at position {intruder_pos}")
    print(f"  Pose tracker noise: 5 cm std (ADR-079 ~95% PCK@20 quality)")
    print()
    print(f"=== Mean PABS by phase ===")
    print(f"  Phase                  Fixed-expected   Pose-updated")
    print(f"  Pre-intruder  (T<25):  {pre_intrude_fixed_mean:>14.4f}   {pre_intrude_pose_mean:>13.4f}")
    print(f"  Post-intruder (T>=25): {post_intrude_fixed_mean:>14.4f}   {post_intrude_pose_mean:>13.4f}")
    print()
    print(f"=== Intruder detection lift ===")
    print(f"  FIXED-expected pipeline:  {fixed_intruder_lift:>7.2f}x   (R12 naive)")
    print(f"  POSE-UPDATED pipeline:    {pose_intruder_lift:>7.2f}x   (R12.1 closed loop)")
    print()
    if pose_intruder_lift > fixed_intruder_lift * 3:
        verdict = "CLOSED LOOP WORKS: pose-PABS lift > 3x the naive baseline. False-alarm problem from R12 PABS resolved."
    elif pose_intruder_lift > 2.0:
        verdict = "CLOSED LOOP WORKS: pose-PABS lift > 2x baseline. Intruder detection clean."
    else:
        verdict = "MARGINAL: pose-PABS lift not decisive vs baseline. May need temporal averaging."
    print(f"VERDICT: {verdict}")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
