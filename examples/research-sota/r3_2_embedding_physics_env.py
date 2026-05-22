#!/usr/bin/env python3
"""R3.2 — Embedding-level physics-informed env_sig prediction (R3.1 fix).

See docs/research/sota-2026-05-22/R3_2-embedding-level-physics-env.md.

R3.1 NEGATIVE found that physics-informed env subtraction at raw-CSI
level fails because within-room position variance dominates. The
corrected architecture:

  raw CSI -> AETHER embedding (position-invariant) -> physics env sub -> K-NN

This tick implements the corrected architecture and tests whether
cross-room K-NN now recovers.

AETHER simulation: per-subject-per-room mean across multiple positions
gives a position-invariant signature. (Real AETHER does this with
contrastive learning; for a synthetic test the averaging approximation
is sufficient.)

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
    return [
        {"pos": [cx, cy], "refl": 0.10 * person_scale},
        {"pos": [cx, cy], "refl": 0.50 * person_scale},
        {"pos": [cx - 0.20*person_scale, cy], "refl": 0.10 * person_scale},
        {"pos": [cx + 0.20*person_scale, cy], "refl": 0.10 * person_scale},
        {"pos": [cx - 0.10*person_scale, cy - 0.40*person_scale], "refl": 0.10 * person_scale},
        {"pos": [cx + 0.10*person_scale, cy - 0.40*person_scale], "refl": 0.10 * person_scale},
    ]


def room_walls_5x5():
    return [
        {"pos": [0.5, 4.5], "refl": 0.30},
        {"pos": [4.5, 4.5], "refl": 0.25},
        {"pos": [0.5, 0.5], "refl": 0.20},
        {"pos": [4.5, 0.5], "refl": 0.15},
    ]


def room_walls_4x6():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r3_2_embedding_results.json")
    args = parser.parse_args()

    freq = 2.4
    n_subj = 10
    rng = np.random.default_rng(42)
    body_scales = 0.85 + 0.30 * rng.random(n_subj)

    # Same setup as R3.1
    room1_walls = room_walls_5x5()
    tx1, rx1 = np.array([1.25, 0.0]), np.array([4.75, 5.0])
    room1_positions = [(2.5, 2.75), (2.5, 2.5), (2.0, 3.0)]
    room2_walls = room_walls_4x6()
    tx2, rx2 = np.array([1.0, 0.0]), np.array([3.0, 6.0])
    room2_positions = [(2.0, 3.0), (1.5, 3.5), (2.5, 2.5)]

    # Predicted env_sig (no labels)
    env_sig_room1 = simulate(room1_walls, tx1, rx1, freq)
    env_sig_room2 = simulate(room2_walls, tx2, rx2, freq)

    # Generate raw CSI per subject per position per room
    raw_r1 = np.zeros((n_subj, len(room1_positions), 52), dtype=complex)
    raw_r2 = np.zeros((n_subj, len(room2_positions), 52), dtype=complex)
    for i in range(n_subj):
        for p_idx, pos in enumerate(room1_positions):
            body = human_body(*pos, person_scale=body_scales[i])
            raw_r1[i, p_idx] = simulate(body + room1_walls, tx1, rx1, freq)
        for p_idx, pos in enumerate(room2_positions):
            body = human_body(*pos, person_scale=body_scales[i])
            raw_r2[i, p_idx] = simulate(body + room2_walls, tx2, rx2, freq)

    # === AETHER simulation: per-subject-per-room mean across positions ===
    # (Position-invariant signature; real AETHER would be a contrastive
    # learning head trained to achieve this invariance.)
    aether_r1 = raw_r1.mean(axis=1)  # (n_subj, 52)
    aether_r2 = raw_r2.mean(axis=1)

    # === Cross-room K-NN approaches ===
    labels = np.arange(n_subj)

    # (a) Raw AETHER (no env subtraction at all)
    acc_aether_raw = knn_accuracy(aether_r2, aether_r1, labels, labels)

    # (b) Labelled MERIDIAN at embedding level (oracle)
    centroid1 = aether_r1.mean(axis=0)
    centroid2 = aether_r2.mean(axis=0)
    aether_r1_meridian = aether_r1 - centroid1
    aether_r2_meridian = aether_r2 - centroid2
    acc_meridian = knn_accuracy(aether_r2_meridian, aether_r1_meridian, labels, labels)

    # (c) Physics-informed env at embedding level (no labels)
    # The env_sig is a single raw-CSI vector per room. When the embedding
    # space is the same as raw-CSI (which it is in our averaging-based
    # AETHER simulation), we just subtract the env vector directly.
    aether_r1_phys = aether_r1 - env_sig_room1
    aether_r2_phys = aether_r2 - env_sig_room2
    acc_physics = knn_accuracy(aether_r2_phys, aether_r1_phys, labels, labels)

    # (d) Physics-informed + within-room residual correction
    # If physics prediction is imperfect (it usually is), residual env error
    # can be estimated from the within-room mean of the physics-corrected
    # AETHER signatures.
    res_r1 = aether_r1_phys.mean(axis=0)
    res_r2 = aether_r2_phys.mean(axis=0)
    aether_r1_phys_plus = aether_r1_phys - res_r1
    aether_r2_phys_plus = aether_r2_phys - res_r2
    acc_physics_plus = knn_accuracy(aether_r2_phys_plus, aether_r1_phys_plus, labels, labels)

    # Within-room sanity check
    acc_within_r1 = knn_accuracy(aether_r1, aether_r1, labels, labels)
    acc_within_r2 = knn_accuracy(aether_r2, aether_r2, labels, labels)

    # Compare to R3.1 raw-CSI level
    print("=== R3.2 embedding-level cross-room re-ID ===")
    print(f"  10 subjects, 3 positions per room, 2 rooms (5x5 + 4x6 m)")
    print()
    print(f"=== 1-shot K-NN accuracy ===")
    print(f"  Within-room AETHER (sanity):                  {acc_within_r1*100:6.1f}% / {acc_within_r2*100:6.1f}%")
    print(f"  Cross-room AETHER raw (no env subtraction):   {acc_aether_raw*100:6.1f}%")
    print(f"  Cross-room AETHER + labelled MERIDIAN:        {acc_meridian*100:6.1f}%")
    print(f"  Cross-room AETHER + PHYSICS-INFORMED env:     {acc_physics*100:6.1f}%  (this tick)")
    print(f"  Cross-room AETHER + physics + residual:       {acc_physics_plus*100:6.1f}%  (refinement)")
    print(f"  Chance:                                       {100/n_subj:6.1f}%")
    print()

    # R3.1 baseline for comparison
    print(f"=== R3.1 RAW-CSI level (baseline) ===")
    print(f"  Cross-room RAW-CSI raw:                  10.0% (chance)")
    print(f"  Cross-room RAW-CSI labelled MERIDIAN:    10.0% (chance) -- R3.1 said this was the architecture error")
    print(f"  Cross-room RAW-CSI physics-informed:     10.0% (chance)")
    print()

    if acc_physics >= 0.8:
        verdict = f"VALIDATED: physics-informed at embedding level hits {acc_physics*100:.1f}% (R3.1 architecture error confirmed corrected)."
    elif acc_physics >= acc_aether_raw * 1.2:
        verdict = f"PARTIAL: physics-informed lifts {acc_physics/acc_aether_raw:.1f}x over raw AETHER cross-room. Not as good as labelled MERIDIAN but with ZERO labels."
    else:
        verdict = f"NOT VALIDATED: embedding-level physics-informed only marginal lift."
    print(f"VERDICT: {verdict}")

    out = {
        "config": {"n_subjects": n_subj, "rooms": ["5x5", "4x6"], "positions_per_room": 3},
        "accuracy": {
            "within_room_1": acc_within_r1,
            "within_room_2": acc_within_r2,
            "cross_aether_raw": acc_aether_raw,
            "cross_aether_meridian_labelled": acc_meridian,
            "cross_aether_physics_informed": acc_physics,
            "cross_aether_physics_plus_residual": acc_physics_plus,
            "chance": 1.0 / n_subj,
        },
        "r3_1_baseline_raw_csi": {
            "raw": 0.10, "meridian": 0.10, "physics": 0.10,
        },
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
