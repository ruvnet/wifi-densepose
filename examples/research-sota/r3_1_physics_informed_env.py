#!/usr/bin/env python3
"""R3.1 — Physics-informed env_sig prediction for zero-shot cross-room re-ID.

See docs/research/sota-2026-05-22/R3_1-physics-informed-env-prediction.md.

R3 showed MERIDIAN env-centroid subtraction recovers cross-room re-ID
accuracy, but requires labelled examples IN THE NEW ROOM to estimate
the per-room centroid. The "next research lever" identified in R3:

  Use R6.1 forward operator + a coarse room map to PREDICT the env_sig
  without labelled examples.

This tick implements that. Two rooms (5x5 and 4x6) with different wall
reflector configurations. For each room, we:

  1. Compute predicted env_sig from R6.1 forward model summed over the
     room's wall scatterers (no person).
  2. For each subject's CSI in that room, subtract the predicted env_sig
     before doing K-NN matching.
  3. Compare to MERIDIAN-with-labels (oracle baseline) and raw cross-room.

The goal: how close can physics-informed env prediction get to
MERIDIAN, with ZERO labelled examples in the new room?

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


def csi_contribution(scatterer_pos, reflectivity, tx_pos, rx_pos, sub_freqs_hz):
    d_tx = np.linalg.norm(scatterer_pos - tx_pos)
    d_rx = np.linalg.norm(scatterer_pos - rx_pos)
    d_direct = np.linalg.norm(tx_pos - rx_pos)
    delta_l = d_tx + d_rx - d_direct
    amp = reflectivity / max(d_tx * d_rx, 1e-3)
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


def human_body(cx, cy, person_scale=1.0):
    """Person scale slightly varies between subjects (body size).
    Returns list of 6 body-part scatterers."""
    return [
        {"pos": [cx,        cy       ], "refl": 0.10 * person_scale, "name": "head"},
        {"pos": [cx,        cy       ], "refl": 0.50 * person_scale, "name": "chest"},
        {"pos": [cx - 0.20*person_scale, cy], "refl": 0.10 * person_scale, "name": "left_arm"},
        {"pos": [cx + 0.20*person_scale, cy], "refl": 0.10 * person_scale, "name": "right_arm"},
        {"pos": [cx - 0.10*person_scale, cy - 0.40*person_scale], "refl": 0.10 * person_scale, "name": "l_leg"},
        {"pos": [cx + 0.10*person_scale, cy - 0.40*person_scale], "refl": 0.10 * person_scale, "name": "r_leg"},
    ]


def room_walls_5x5():
    """Bedroom: square 5x5m with 4 wall scatterers."""
    return [
        {"pos": [0.5, 4.5], "refl": 0.30},
        {"pos": [4.5, 4.5], "refl": 0.25},
        {"pos": [0.5, 0.5], "refl": 0.20},
        {"pos": [4.5, 0.5], "refl": 0.15},
    ]


def room_walls_4x6():
    """Living room: 4x6m with 4 wall scatterers in different positions/refl."""
    return [
        {"pos": [0.3, 5.7], "refl": 0.28},
        {"pos": [3.7, 5.7], "refl": 0.18},
        {"pos": [0.3, 0.3], "refl": 0.32},
        {"pos": [3.7, 0.3], "refl": 0.22},
    ]


def cosine_dist(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9: return 1.0
    return 1.0 - float(np.real(np.vdot(a, b) / (norm_a * norm_b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r3_1_physics_env_results.json")
    args = parser.parse_args()

    freq = 2.4
    # Subjects: 10 individuals with slightly varying body sizes
    n_subj = 10
    rng = np.random.default_rng(42)
    body_scales = 0.85 + 0.30 * rng.random(n_subj)  # 0.85 to 1.15

    # Room 1: 5x5, link goes diagonally (per R6.2 best placement)
    room1_walls = room_walls_5x5()
    tx1, rx1 = np.array([1.25, 0.0]), np.array([4.75, 5.0])
    room1_subject_positions = [(2.5, 2.75), (2.5, 2.5), (2.0, 3.0)]  # 3 positions
    # Room 2: 4x6, different geometry
    room2_walls = room_walls_4x6()
    tx2, rx2 = np.array([1.0, 0.0]), np.array([3.0, 6.0])
    room2_subject_positions = [(2.0, 3.0), (1.5, 3.5), (2.5, 2.5)]

    # === Step 1: PREDICTED env_sig from physics (no labels needed) ===
    # Just simulate the room with NO subject -- this is what the empty
    # room "looks like" to the antennas.
    env_sig_room1_predicted = simulate(room1_walls, tx1, rx1, freq)
    env_sig_room2_predicted = simulate(room2_walls, tx2, rx2, freq)

    # === Step 2: Generate CSI per subject in each room ===
    csi_room1, csi_room2 = [], []
    for i in range(n_subj):
        scale = body_scales[i]
        for pos in room1_subject_positions:
            body = human_body(*pos, person_scale=scale)
            scene = body + room1_walls
            csi_room1.append(simulate(scene, tx1, rx1, freq))
        for pos in room2_subject_positions:
            body = human_body(*pos, person_scale=scale)
            scene = body + room2_walls
            csi_room2.append(simulate(scene, tx2, rx2, freq))
    csi_room1 = np.array(csi_room1)
    csi_room2 = np.array(csi_room2)
    labels = np.repeat(np.arange(n_subj), len(room1_subject_positions))

    # === Step 3: Compute the LABELED MERIDIAN centroid (oracle baseline) ===
    centroid_room1_meridian = csi_room1.mean(axis=0)
    centroid_room2_meridian = csi_room2.mean(axis=0)

    # === Step 4: Cross-room re-ID with three approaches ===

    def knn_accuracy(query, gallery, q_labels, g_labels, k=1):
        correct = 0
        for i in range(len(query)):
            dists = [cosine_dist(query[i], g) for g in gallery]
            top_k = np.argsort(dists)[:k]
            top_k_labels = [g_labels[j] for j in top_k]
            vals, counts = np.unique(top_k_labels, return_counts=True)
            pred = vals[np.argmax(counts)]
            if pred == q_labels[i]:
                correct += 1
        return correct / len(query)

    # Gallery = room 1 (train), Query = room 2 (test)
    # (a) Raw cross-room
    acc_raw = knn_accuracy(csi_room2, csi_room1, labels, labels)

    # (b) MERIDIAN with labelled centroid (oracle)
    csi_room1_cleaned = csi_room1 - centroid_room1_meridian
    csi_room2_cleaned = csi_room2 - centroid_room2_meridian
    acc_meridian = knn_accuracy(csi_room2_cleaned, csi_room1_cleaned, labels, labels)

    # (c) Physics-informed env prediction (ZERO labels in either room)
    csi_room1_phys = csi_room1 - env_sig_room1_predicted
    csi_room2_phys = csi_room2 - env_sig_room2_predicted
    acc_physics = knn_accuracy(csi_room2_phys, csi_room1_phys, labels, labels)

    # === Within-room baselines ===
    acc_within_room1 = knn_accuracy(csi_room1, csi_room1, labels, labels)
    acc_within_room2 = knn_accuracy(csi_room2, csi_room2, labels, labels)

    out = {
        "config": {
            "n_subjects": n_subj,
            "n_positions_per_room": len(room1_subject_positions),
            "rooms": ["5x5 m", "4x6 m"],
            "freq_ghz": freq,
        },
        "accuracy": {
            "within_room_1": acc_within_room1,
            "within_room_2": acc_within_room2,
            "cross_room_raw": acc_raw,
            "cross_room_meridian_labelled": acc_meridian,
            "cross_room_physics_informed": acc_physics,
            "chance": 1.0 / n_subj,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("=== R3.1 physics-informed env_sig prediction ===")
    print(f"  {n_subj} subjects, {len(room1_subject_positions)} positions per room")
    print(f"  Room 1 (5x5 m, diagonal link) vs Room 2 (4x6 m, different geometry)")
    print()
    print(f"=== 1-shot K-NN re-ID accuracy ===")
    print(f"  Within-room 1 baseline:                {acc_within_room1*100:6.1f}%")
    print(f"  Within-room 2 baseline:                {acc_within_room2*100:6.1f}%")
    print(f"  Cross-room RAW (no env subtraction):   {acc_raw*100:6.1f}%")
    print(f"  Cross-room MERIDIAN (labelled oracle): {acc_meridian*100:6.1f}%")
    print(f"  Cross-room PHYSICS-INFORMED:           {acc_physics*100:6.1f}%  (this tick)")
    print(f"  Chance:                                {100/n_subj:6.1f}%")
    print()
    if acc_physics >= acc_meridian * 0.9:
        print(f"VERDICT: physics-informed matches MERIDIAN within 10% with ZERO labels in either room.")
    elif acc_physics > acc_raw * 1.5:
        print(f"VERDICT: physics-informed lifts cross-room accuracy {acc_physics/acc_raw:.1f}x vs raw.")
    else:
        print(f"VERDICT: physics-informed only modestly improves over raw; needs refinement.")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
