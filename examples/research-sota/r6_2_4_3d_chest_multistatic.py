#!/usr/bin/env python3
"""R6.2.4 — 3D chest-centric N-anchor multistatic (compose R6.2.2.1 + R6.2.3).

See docs/research/sota-2026-05-22/R6_2_4-3d-chest-multistatic.md.

R6.2.2.1 (3D N-anchor on body-footprint zones) showed N=5 gives only
49% coverage in 3D vs 97% in 2D -- the 2D-derived knee disappears.
R6.2.2.1 predicted: switching to chest-centric zones (R6.2.3) should
recover 80%+ in 3D at N=5.

This tick tests that prediction. Pure NumPy.
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


def rasterise_targets_3d(zones, resolution=0.10):
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
    for z in [0.8, 1.5, 2.4]:
        for x in np.arange(0, room_w + 0.001, step):
            cands.append(np.array([x, 0.0, z]))
            cands.append(np.array([x, room_h, z]))
        for y in np.arange(step, room_h, step):
            cands.append(np.array([0.0, y, z]))
            cands.append(np.array([room_w, y, z]))
    for x in np.arange(1.0, room_w, 1.0):
        for y in np.arange(1.0, room_h, 1.0):
            cands.append(np.array([x, y, room_z]))
    return cands


def greedy_search(candidates, target_pts, wavelength, n_anchors, n_restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    best = {"anchors": [], "score": -1.0}
    for restart in range(n_restarts):
        idx0, idx1 = rng.choice(len(candidates), size=2, replace=False)
        chosen = [candidates[idx0], candidates[idx1]]
        while len(chosen) < n_anchors:
            best_marg = -1.0
            best_idx = None
            for k, c in enumerate(candidates):
                if any(np.allclose(c, a) for a in chosen):
                    continue
                score = union_coverage_3d(chosen + [c], target_pts, wavelength)
                if score > best_marg:
                    best_marg = score
                    best_idx = k
            if best_idx is None: break
            chosen.append(candidates[best_idx])
        final = union_coverage_3d(chosen, target_pts, wavelength)
        if final > best["score"]:
            best = {"anchors": [a.tolist() for a in chosen], "score": final}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_2_4_3d_chest_results.json")
    parser.add_argument("--n-max", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=4)
    args = parser.parse_args()

    room_w, room_h, room_z = 5.0, 5.0, 2.5
    freq = 2.4
    lam = wavelength_m(freq)

    # 3D chest-centric zones (compose R6.2.3's 2D chest with R6.2.1's 3D heights)
    # Chest of: lying-down (z=0.3-0.5), sitting (z=0.7-1.0), standing (z=1.2-1.5)
    chest_zones_3d = [
        ("bed_chest",      2.2, 0.8, 0.3,  0.6, 0.4, 0.2),  # lying chest at z=0.3-0.5
        ("chair_chest",    3.7, 3.7, 0.7,  0.4, 0.4, 0.3),  # sitting chest z=0.7-1.0
        ("standing_chest", 0.5, 3.7, 1.2,  0.6, 0.4, 0.3),  # standing chest z=1.2-1.5
    ]
    target_pts = rasterise_targets_3d(chest_zones_3d, resolution=0.10)
    candidates = candidate_positions_3d(room_w, room_h, room_z, step=0.75)

    print(f"Room: {room_w}x{room_h}x{room_z} m at {freq} GHz")
    print(f"CHEST-CENTRIC 3D targets: {len(target_pts)} points across {len(chest_zones_3d)} zones")
    print(f"Candidates: {len(candidates)} positions (3 wall heights + ceiling)")
    print()

    saturation = []
    for n in range(2, args.n_max + 1):
        result = greedy_search(candidates, target_pts, lam,
                              n_anchors=n, n_restarts=args.restarts)
        heights = [a[2] for a in result["anchors"]]
        n_low = sum(1 for h in heights if h < 1.0)
        n_mid = sum(1 for h in heights if 1.0 <= h < 2.0)
        n_high = sum(1 for h in heights if h >= 2.0)
        saturation.append({
            "n_anchors": n,
            "coverage": result["score"],
            "heights": {"low": n_low, "mid": n_mid, "high": n_high},
            "anchors": result["anchors"],
        })

    print("=== 3D chest-centric saturation curve ===")
    print(f"{'N':>3}  {'Coverage':>9}  {'Marginal':>9}  {'Heights L/M/H':>15}")
    prev = 0.0
    for s in saturation:
        marg = (s["coverage"] - prev) * 100
        h = s["heights"]
        print(f"{s['n_anchors']:>3}  {s['coverage']*100:>7.1f}%  {marg:>+7.1f} pp  {h['low']}/{h['mid']}/{h['high']:>5}")
        prev = s["coverage"]

    # Compare to R6.2.2.1 (3D body-centric) at same N
    print()
    print("=== R6.2.2.1 prediction validation ===")
    print(f"R6.2.2.1 said: 'chest-centric should recover N=5 to 80%+ in 3D.'")
    n5 = next(s for s in saturation if s["n_anchors"] == 5)
    if n5["coverage"] >= 0.8:
        print(f"VALIDATED: 3D chest-centric N=5 = {n5['coverage']*100:.1f}% (>= 80% target)")
    elif n5["coverage"] >= 0.7:
        print(f"PARTIAL:   3D chest-centric N=5 = {n5['coverage']*100:.1f}% (close to 80% target)")
    else:
        print(f"NOT VALIDATED: 3D chest-centric N=5 = {n5['coverage']*100:.1f}% (well below 80%)")
    print()
    # Full 4-way comparison
    print("=== 4-way comparison at N=5 ===")
    print(f"  R6.2.2   (2D body):    96.8%")
    print(f"  R6.2.3   (2D chest):   82.4%")
    print(f"  R6.2.2.1 (3D body):    49.4%")
    print(f"  R6.2.4   (3D chest):   {n5['coverage']*100:.1f}%   (this tick)")

    out = {
        "room": {"width_m": room_w, "depth_m": room_h, "ceiling_m": room_z},
        "freq_ghz": freq,
        "target_zones": [
            {"name": n, "x": x0, "y": y0, "z": z0, "dx": dx, "dy": dy, "dz": dz}
            for n, x0, y0, z0, dx, dy, dz in chest_zones_3d
        ],
        "saturation": saturation,
        "comparison_at_n5": {
            "r6_2_2_2d_body": 0.968,
            "r6_2_3_2d_chest": 0.824,
            "r6_2_2_1_3d_body": 0.494,
            "r6_2_4_3d_chest": n5["coverage"],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
