#!/usr/bin/env python3
"""R6.2.1 — 3D Fresnel-aware antenna placement (ceiling + wall mounts).

See docs/research/sota-2026-05-22/R6_2_1-3d-placement.md.

R6.2 was 2D (top-down). Real human occupants stand at heights 0-1.8 m;
real WiFi APs typically sit at desk height (0.8 m), wall mounts at
1.5 m, or ceiling mounts at 2.5 m. The optimal placement depends on
whether antennas + target zones share an elevation.

This script extends R6.2 to 3D:
  - First Fresnel zone in 3D is a prolate ellipsoid (rotation of the
    2D ellipse around the Tx-Rx axis)
  - Target zones are 3D boxes representing where a person's torso
    occupies (e.g. chest height 1.0-1.5 m for standing, 0.5-1.0 m for
    sitting on a chair, 0.3-0.6 m for lying in bed)
  - Candidate antenna mounts: wall (z fixed by mount height) or
    ceiling (z = ceiling height)

A point (x, y, z) is inside the first Fresnel ellipsoid iff:
    |Tx - p| + |p - Rx| <= |Tx - Rx| + lambda/2

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


def in_first_fresnel_3d(p: np.ndarray, tx: np.ndarray, rx: np.ndarray,
                       wavelength: float) -> np.ndarray:
    """Boolean: is each point p (Nx3) inside the first Fresnel ellipsoid?"""
    r1 = np.linalg.norm(p - tx, axis=1)
    r2 = np.linalg.norm(p - rx, axis=1)
    direct = np.linalg.norm(tx - rx)
    return (r1 + r2) <= (direct + wavelength / 2)


def coverage_3d(tx: np.ndarray, rx: np.ndarray, target_zones: list,
               wavelength: float, resolution: float = 0.1) -> dict:
    """3D rectangular zones. Each zone: (name, x0, y0, z0, dx, dy, dz)."""
    per_zone = {}
    total_pts = 0
    total_covered = 0
    for name, x0, y0, z0, dx, dy, dz in target_zones:
        xs = np.arange(x0, x0 + dx, resolution)
        ys = np.arange(y0, y0 + dy, resolution)
        zs = np.arange(z0, z0 + dz, resolution)
        xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1)
        mask = in_first_fresnel_3d(pts, tx, rx, wavelength)
        per_zone[name] = {
            "n_points": len(pts),
            "n_covered": int(mask.sum()),
            "coverage_fraction": float(mask.mean()),
        }
        total_pts += len(pts)
        total_covered += mask.sum()
    return {
        "total_coverage": float(total_covered / total_pts) if total_pts > 0 else 0,
        "per_zone": per_zone,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_2_1_3d_results.json")
    args = parser.parse_args()

    room_w, room_h, room_z = 5.0, 5.0, 2.5
    freq = 2.4
    lam = wavelength_m(freq)

    # Three realistic 3D target zones:
    #   bed (lying down)          (1.5, 0.5, 0.3) - (3.5, 2.0, 0.6) at low altitude
    #   chair (sitting)           (3.5, 3.5, 0.5) - (4.3, 4.3, 1.2) at mid altitude
    #   standing zone (workspace) (0.5, 3.5, 1.0) - (1.5, 4.5, 1.7) at upper altitude
    target_zones = [
        ("bed",      1.5, 0.5, 0.3,  2.0, 1.5, 0.3),
        ("chair",    3.5, 3.5, 0.5,  0.8, 0.8, 0.7),
        ("standing", 0.5, 3.5, 1.0,  1.0, 1.0, 0.7),
    ]

    # Three candidate antenna placement strategies
    strategies = {
        "desk-height (0.8 m, wall)": {
            "z_options": [0.8],
            "where": "wall",
        },
        "wall-mount (1.5 m, wall)": {
            "z_options": [1.5],
            "where": "wall",
        },
        "ceiling (2.5 m, full ceiling grid)": {
            "z_options": [2.5],
            "where": "ceiling",
        },
        "wall + ceiling (mixed at any height)": {
            "z_options": [0.8, 1.5, 2.5],
            "where": "any",
        },
    }

    def gen_candidates(strategy_cfg, step=0.5):
        cands = []
        for z in strategy_cfg["z_options"]:
            if strategy_cfg["where"] in ("wall", "any"):
                # 4 walls
                for x in np.arange(0, room_w + 0.001, step):
                    cands.append(np.array([x, 0.0, z]))
                    cands.append(np.array([x, room_h, z]))
                for y in np.arange(step, room_h, step):
                    cands.append(np.array([0.0, y, z]))
                    cands.append(np.array([room_w, y, z]))
            if strategy_cfg["where"] in ("ceiling", "any") and z >= room_z - 0.01:
                # Ceiling grid
                for x in np.arange(0.5, room_w + 0.001, step):
                    for y in np.arange(0.5, room_h + 0.001, step):
                        cands.append(np.array([x, y, z]))
        # Deduplicate
        unique = []
        for c in cands:
            if not any(np.allclose(c, u) for u in unique):
                unique.append(c)
        return unique

    print(f"Room: {room_w}x{room_h}x{room_z} m at {freq} GHz")
    print(f"Target zones:")
    for name, x0, y0, z0, dx, dy, dz in target_zones:
        print(f"  {name}: ({x0},{y0},{z0}) - ({x0+dx},{y0+dy},{z0+dz})")
    print()

    results = {}
    for name, cfg in strategies.items():
        cands = gen_candidates(cfg)
        best_score = -1
        best_tx, best_rx = None, None
        n_evaluated = 0
        for i, tx in enumerate(cands):
            for j, rx in enumerate(cands):
                if j <= i: continue
                if np.linalg.norm(tx - rx) < 1.0:
                    continue
                cov = coverage_3d(tx, rx, target_zones, lam, resolution=0.1)
                n_evaluated += 1
                if cov["total_coverage"] > best_score:
                    best_score = cov["total_coverage"]
                    best_tx = tx.tolist()
                    best_rx = rx.tolist()
                    best_per_zone = cov["per_zone"]
        results[name] = {
            "best_score": float(best_score),
            "best_tx": best_tx,
            "best_rx": best_rx,
            "n_candidates": len(cands),
            "n_pairs_evaluated": n_evaluated,
            "best_per_zone": best_per_zone,
        }

    print("=== 3D placement strategy comparison ===")
    print(f"{'Strategy':<46}  {'Pairs':>6}  {'Coverage':>9}")
    for name, r in results.items():
        print(f"{name:<46}  {r['n_pairs_evaluated']:>6}  {r['best_score']*100:>7.1f}%")
    print()

    # Headline
    best_strategy = max(results, key=lambda k: results[k]["best_score"])
    desk_score = results["desk-height (0.8 m, wall)"]["best_score"]
    ceiling_score = results["ceiling (2.5 m, full ceiling grid)"]["best_score"]
    mixed_score = results["wall + ceiling (mixed at any height)"]["best_score"]
    lift = (mixed_score - desk_score) / desk_score * 100 if desk_score > 0 else 0

    print(f"Best strategy: {best_strategy}  ({results[best_strategy]['best_score']*100:.1f}%)")
    print(f"  Best Tx: {results[best_strategy]['best_tx']}")
    print(f"  Best Rx: {results[best_strategy]['best_rx']}")
    print()
    print(f"Desk-height baseline:   {desk_score*100:.1f}%")
    print(f"Ceiling-only:           {ceiling_score*100:.1f}%")
    print(f"Mixed wall+ceiling:     {mixed_score*100:.1f}%  (+{lift:.1f}% over desk-height)")
    print()

    out = {
        "room": {"width_m": room_w, "depth_m": room_h, "ceiling_m": room_z},
        "freq_ghz": freq,
        "target_zones": [
            {"name": n, "x": x0, "y": y0, "z": z0, "dx": dx, "dy": dy, "dz": dz}
            for n, x0, y0, z0, dx, dy, dz in target_zones
        ],
        "strategies": results,
        "headline": {
            "best_strategy": best_strategy,
            "desk_score": desk_score,
            "ceiling_score": ceiling_score,
            "mixed_score": mixed_score,
            "mixed_lift_over_desk_pct": lift,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
