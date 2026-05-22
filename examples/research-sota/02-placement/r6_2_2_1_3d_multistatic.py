#!/usr/bin/env python3
"""R6.2.2.1 — 3D N-anchor multistatic placement (compose R6.2.1 + R6.2.2).

See docs/research/sota-2026-05-22/R6_2_2_1-3d-multistatic.md.

R6.2.2 found a 2D knee at N=5 anchors for typical bedroom geometry.
R6.2.1 found ceiling-only mounting gives 0% coverage in 3D. R6.2.2.1
composes both: how does the saturation curve change in 3D with mixed-
height candidate anchors?

Practical question: with mixed-height multistatic deployment, does the
4-anchor practical default (ADR-029) hit acceptable coverage in 3D?

Pure NumPy. Greedy search with K=4 random restarts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8


def wavelength_m(freq_ghz: float) -> float:
    return C / (freq_ghz * 1e9)


def in_first_fresnel_3d(p: np.ndarray, tx: np.ndarray, rx: np.ndarray,
                       wavelength: float) -> np.ndarray:
    r1 = np.linalg.norm(p - tx, axis=1)
    r2 = np.linalg.norm(p - rx, axis=1)
    direct = np.linalg.norm(tx - rx)
    return (r1 + r2) <= (direct + wavelength / 2)


def union_coverage_3d(anchors, target_pts, wavelength):
    if len(anchors) < 2:
        return 0.0
    covered = np.zeros(len(target_pts), dtype=bool)
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            mask = in_first_fresnel_3d(target_pts, anchors[i], anchors[j], wavelength)
            covered |= mask
    return float(covered.mean())


def rasterise_targets_3d(zones, resolution=0.15):
    pts = []
    for name, x0, y0, z0, dx, dy, dz in zones:
        xs = np.arange(x0, x0 + dx, resolution)
        ys = np.arange(y0, y0 + dy, resolution)
        zs = np.arange(z0, z0 + dz, resolution)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        for x, y, z in zip(gx.ravel(), gy.ravel(), gz.ravel()):
            pts.append([x, y, z])
    return np.array(pts)


def candidate_positions_3d(room_w, room_h, room_z, step=0.75):
    cands = []
    # Wall mounts at three heights
    for z in [0.8, 1.5, 2.4]:
        for x in np.arange(0, room_w + 0.001, step):
            cands.append(np.array([x, 0.0, z]))
            cands.append(np.array([x, room_h, z]))
        for y in np.arange(step, room_h, step):
            cands.append(np.array([0.0, y, z]))
            cands.append(np.array([room_w, y, z]))
    # Ceiling mounts on a coarse grid
    for x in np.arange(1.0, room_w, 1.0):
        for y in np.arange(1.0, room_h, 1.0):
            cands.append(np.array([x, y, room_z]))
    return cands


def greedy_search(candidates, target_pts, wavelength, n_anchors, n_restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    best = {"anchors": [], "score": -1.0, "trace": []}
    for restart in range(n_restarts):
        idx0, idx1 = rng.choice(len(candidates), size=2, replace=False)
        chosen = [candidates[idx0], candidates[idx1]]
        trace = [union_coverage_3d(chosen, target_pts, wavelength)]
        while len(chosen) < n_anchors:
            best_marginal = -1.0
            best_idx = None
            for k, c in enumerate(candidates):
                if any(np.allclose(c, a) for a in chosen):
                    continue
                trial = chosen + [c]
                score = union_coverage_3d(trial, target_pts, wavelength)
                if score > best_marginal:
                    best_marginal = score
                    best_idx = k
            if best_idx is None: break
            chosen.append(candidates[best_idx])
            trace.append(best_marginal)
        if trace[-1] > best["score"]:
            best = {
                "anchors": [a.tolist() for a in chosen],
                "score": trace[-1],
                "trace": trace,
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_2_2_1_3d_multistatic_results.json")
    parser.add_argument("--n-max", type=int, default=7)
    parser.add_argument("--restarts", type=int, default=4)
    args = parser.parse_args()

    room_w, room_h, room_z = 5.0, 5.0, 2.5
    freq = 2.4
    lam = wavelength_m(freq)

    # Same 3D target zones as R6.2.1
    target_zones = [
        ("bed",      1.5, 0.5, 0.3, 2.0, 1.5, 0.3),
        ("chair",    3.5, 3.5, 0.5, 0.8, 0.8, 0.7),
        ("standing", 0.5, 3.5, 1.0, 1.0, 1.0, 0.7),
    ]
    target_pts = rasterise_targets_3d(target_zones, resolution=0.15)
    candidates = candidate_positions_3d(room_w, room_h, room_z, step=0.75)

    print(f"Room: {room_w}x{room_h}x{room_z} m at {freq} GHz")
    print(f"Targets: {len(target_pts)} 3D points across 3 zones")
    print(f"Candidates: {len(candidates)} positions (3 wall heights + ceiling grid)")
    print()

    saturation = []
    for n in range(2, args.n_max + 1):
        result = greedy_search(candidates, target_pts, lam,
                              n_anchors=n, n_restarts=args.restarts)
        # Anchor height histogram
        heights = [a[2] for a in result["anchors"]]
        n_low = sum(1 for h in heights if h < 1.0)
        n_mid = sum(1 for h in heights if 1.0 <= h < 2.0)
        n_high = sum(1 for h in heights if h >= 2.0)
        saturation.append({
            "n_anchors": n,
            "coverage": result["score"],
            "n_pairs": n * (n - 1) // 2,
            "heights": {"low_0.8m": n_low, "mid_1.5m": n_mid, "high_2.4m+": n_high},
        })

    print("=== 3D coverage saturation ===")
    print(f"{'N':>3}  {'Pairs':>6}  {'Coverage':>9}  {'Marginal':>9}  {'Heights (low/mid/high)':>25}")
    prev = 0.0
    for s in saturation:
        marg = (s["coverage"] - prev) * 100
        h = s["heights"]
        h_str = f"{h['low_0.8m']}/{h['mid_1.5m']}/{h['high_2.4m+']}"
        print(f"{s['n_anchors']:>3}  {s['n_pairs']:>6}  {s['coverage']*100:>7.1f}%  {marg:>+7.1f} pp  {h_str:>25}")
        prev = s["coverage"]

    # Knee detection
    marginal = []
    for i in range(1, len(saturation)):
        prev_cov = saturation[i-1]["coverage"]
        curr_cov = saturation[i]["coverage"]
        marginal.append({
            "from_n": saturation[i-1]["n_anchors"],
            "to_n": saturation[i]["n_anchors"],
            "marginal_pp": (curr_cov - prev_cov) * 100,
        })

    knee = None
    for m in marginal:
        if m["marginal_pp"] < 4.0:
            knee = m["from_n"]
            print(f"\nKnee at N={knee} (going to N={m['to_n']} adds only {m['marginal_pp']:.1f} pp)")
            break

    out = {
        "room": {"width_m": room_w, "depth_m": room_h, "ceiling_m": room_z},
        "freq_ghz": freq,
        "target_zones": [
            {"name": n, "x": x0, "y": y0, "z": z0, "dx": dx, "dy": dy, "dz": dz}
            for n, x0, y0, z0, dx, dy, dz in target_zones
        ],
        "saturation": saturation,
        "marginal": marginal,
        "knee_n_anchors": knee,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
