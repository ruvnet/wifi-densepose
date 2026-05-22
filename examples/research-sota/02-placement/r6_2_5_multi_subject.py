#!/usr/bin/env python3
"""R6.2.5 — Multi-subject occupancy union.

See docs/research/sota-2026-05-22/R6_2_5-multi-subject-union.md.

R6.2 / R6.2.3 picked one chest position per zone. Real households
have 2-4 occupants who can be in different positions simultaneously
(spouse in bed + child at desk + visitor on chair). R6.2.5 extends to
**union of chest envelopes** across all expected occupant positions.

Practical question: does the optimal placement degrade gracefully
when target zones multiply? Does N=5 still hit a useful coverage?

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


def in_first_fresnel(x, y, tx, rx, wavelength):
    r1 = np.sqrt((x - tx[0])**2 + (y - tx[1])**2)
    r2 = np.sqrt((x - rx[0])**2 + (y - rx[1])**2)
    direct = np.linalg.norm(tx - rx)
    return (r1 + r2) <= (direct + wavelength / 2)


def union_coverage(anchors, target_x, target_y, wavelength):
    if len(anchors) < 2: return 0.0
    covered = np.zeros(len(target_x), dtype=bool)
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            covered |= in_first_fresnel(target_x, target_y,
                                       anchors[i], anchors[j], wavelength)
    return float(covered.mean())


def rasterise_zones(zones, resolution=0.05):
    xs, ys = [], []
    for name, x0, y0, w, h in zones:
        zx = np.arange(x0, x0 + w, resolution)
        zy = np.arange(y0, y0 + h, resolution)
        gx, gy = np.meshgrid(zx, zy)
        xs.append(gx.ravel())
        ys.append(gy.ravel())
    return np.concatenate(xs), np.concatenate(ys)


def candidates(room_w, room_h, step):
    cands = []
    for x in np.arange(0, room_w + 0.001, step):
        cands.append(np.array([x, 0.0]))
        cands.append(np.array([x, room_h]))
    for y in np.arange(step, room_h, step):
        cands.append(np.array([0.0, y]))
        cands.append(np.array([room_w, y]))
    return cands


def greedy_search(cands, target_x, target_y, lam, n_anchors, restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    best = {"score": -1.0, "anchors": []}
    for r in range(restarts):
        idx0, idx1 = rng.choice(len(cands), size=2, replace=False)
        chosen = [cands[idx0], cands[idx1]]
        while len(chosen) < n_anchors:
            best_marg = -1
            best_idx = None
            for k, c in enumerate(cands):
                if any(np.allclose(c, a) for a in chosen): continue
                s = union_coverage(chosen + [c], target_x, target_y, lam)
                if s > best_marg:
                    best_marg = s
                    best_idx = k
            if best_idx is None: break
            chosen.append(cands[best_idx])
        score = union_coverage(chosen, target_x, target_y, lam)
        if score > best["score"]:
            best = {"score": score, "anchors": [a.tolist() for a in chosen]}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_2_5_multi_subject_results.json")
    args = parser.parse_args()

    room_w, room_h = 5.0, 5.0
    freq = 2.4
    lam = wavelength_m(freq)
    step = 0.25
    cands = candidates(room_w, room_h, step)

    # Scenarios with increasing occupant count
    # Each "chest zone" is a 40x40 cm patch
    scenarios = {
        "1 occupant (chair)": [
            ("chair_chest", 3.7, 3.7, 0.4, 0.4),
        ],
        "2 occupants (chair + bed)": [
            ("chair_chest", 3.7, 3.7, 0.4, 0.4),
            ("bed_chest",   2.2, 0.8, 0.6, 0.4),
        ],
        "3 occupants (chair + bed + desk)": [
            ("chair_chest", 3.7, 3.7, 0.4, 0.4),
            ("bed_chest",   2.2, 0.8, 0.6, 0.4),
            ("desk_chest",  0.5, 2.7, 0.4, 0.2),
        ],
        "4 occupants (+ 2nd chair)": [
            ("chair_chest",  3.7, 3.7, 0.4, 0.4),
            ("bed_chest",    2.2, 0.8, 0.6, 0.4),
            ("desk_chest",   0.5, 2.7, 0.4, 0.2),
            ("chair2_chest", 1.0, 4.2, 0.4, 0.4),
        ],
    }

    print(f"Room {room_w}x{room_h} m, freq {freq} GHz, chest-centric zones")
    print()

    # For each scenario, find optimum at N=5
    results = []
    for name, zones in scenarios.items():
        tx, ty = rasterise_zones(zones)
        result = greedy_search(cands, tx, ty, lam, n_anchors=5)
        # Total zone area
        zone_area = sum(w * h for _, _, _, w, h in zones)
        results.append({
            "scenario": name,
            "n_zones": len(zones),
            "total_zone_area_m2": zone_area,
            "coverage_n5": result["score"],
            "best_anchors": result["anchors"],
        })

    print(f"{'Scenario':<40} {'#zones':>6} {'Area':>7} {'Cov@N=5':>9}")
    print("-" * 75)
    for r in results:
        print(f"{r['scenario']:<40} {r['n_zones']:>6} {r['total_zone_area_m2']:>5.2f} m2 {r['coverage_n5']*100:>7.1f}%")
    print()

    # Stress test: scale N for the 4-occupant scenario
    print(f"=== 4-occupant scenario, scaling N from 2..7 ===")
    zones4 = scenarios["4 occupants (+ 2nd chair)"]
    tx, ty = rasterise_zones(zones4)
    print(f"{'N':>3}  {'Coverage':>9}  {'Marginal':>9}")
    prev = 0.0
    scale_curve = []
    for n in range(2, 8):
        result = greedy_search(cands, tx, ty, lam, n_anchors=n)
        marg = (result["score"] - prev) * 100
        print(f"{n:>3}  {result['score']*100:>7.1f}%  {marg:>+7.1f} pp")
        scale_curve.append({"n_anchors": n, "coverage": result["score"]})
        prev = result["score"]
    print()

    # Cross-eval: how does a single-subject-optimised placement perform on 4 subjects?
    single_zone = [("chair_chest", 3.7, 3.7, 0.4, 0.4)]
    tx1, ty1 = rasterise_zones(single_zone)
    single_opt = greedy_search(cands, tx1, ty1, lam, n_anchors=5)
    tx4, ty4 = rasterise_zones(zones4)
    cov_single_on_multi = union_coverage(
        [np.array(a) for a in single_opt["anchors"]], tx4, ty4, lam
    )
    print(f"=== Cross-eval ===")
    print(f"  Single-subject placement on 4-subject zones: {cov_single_on_multi*100:.1f}%")
    print(f"  4-subject-optimised placement on 4 zones:    {results[-1]['coverage_n5']*100:.1f}%")
    print(f"  Gain from multi-subject optimisation:        {(results[-1]['coverage_n5'] - cov_single_on_multi)*100:+.1f} pp")
    print()

    out = {
        "room": {"width_m": room_w, "height_m": room_h},
        "freq_ghz": freq,
        "scenarios_n5": results,
        "saturation_4subj": scale_curve,
        "cross_eval": {
            "single_opt_on_multi": cov_single_on_multi,
            "multi_opt_on_multi": results[-1]["coverage_n5"],
            "gain_pp": (results[-1]["coverage_n5"] - cov_single_on_multi) * 100,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
