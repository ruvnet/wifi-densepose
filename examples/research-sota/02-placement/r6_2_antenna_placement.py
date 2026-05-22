#!/usr/bin/env python3
"""R6.2 — Fresnel-aware antenna placement for room-scale CSI sensing.

See docs/research/sota-2026-05-22/R6_2-fresnel-antenna-placement.md.

Given a 2D room + a list of target occupancy zones (e.g. "the bed",
"the sofa"), search over candidate Tx/Rx positions and pick the pair
that maximises the fraction of target-zone area inside the first
Fresnel ellipse.

The first Fresnel zone in 2D is an ellipse with:
  - foci at Tx and Rx
  - semi-major axis a = (d + lambda/2) / 2
  - semi-minor axis b = sqrt(a^2 - (d/2)^2)
where d = |Tx - Rx| and lambda = c / f.

This is the natural progression from R6 (the 1-D Fresnel radius at
midpoint) -- now we evaluate coverage over arbitrary 2D zones.

Pure NumPy. CLI-shaped: takes room geometry and target zones as args,
emits the best Tx/Rx placement + a coverage fraction.

Example usage:
  python r6_2_antenna_placement.py \\
      --room 5.0 5.0 \\
      --target bed 1.0 0.5 2.0 1.5 \\
      --target sofa 0.5 3.0 1.5 1.0 \\
      --freq-ghz 2.4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

C = 2.998e8


def wavelength_m(freq_ghz: float) -> float:
    return C / (freq_ghz * 1e9)


def in_first_fresnel(x: np.ndarray, y: np.ndarray, tx: np.ndarray, rx: np.ndarray,
                     wavelength: float) -> np.ndarray:
    """Return boolean array: is each (x, y) inside the first Fresnel ellipse
    of the Tx-Rx link?"""
    r1 = np.sqrt((x - tx[0])**2 + (y - tx[1])**2)
    r2 = np.sqrt((x - rx[0])**2 + (y - rx[1])**2)
    direct = np.linalg.norm(tx - rx)
    return (r1 + r2) <= (direct + wavelength / 2)


def coverage_score(tx: np.ndarray, rx: np.ndarray, target_zones: list,
                  wavelength: float, grid_resolution: float = 0.05) -> dict:
    """Compute the fraction of total target-zone area inside the first
    Fresnel ellipse. Per-zone breakdowns also returned."""
    per_zone = {}
    total_area = 0.0
    total_covered = 0.0
    for name, x0, y0, w, h in target_zones:
        # Rasterise the zone
        xs = np.arange(x0, x0 + w, grid_resolution)
        ys = np.arange(y0, y0 + h, grid_resolution)
        xv, yv = np.meshgrid(xs, ys)
        xv = xv.ravel()
        yv = yv.ravel()
        mask = in_first_fresnel(xv, yv, tx, rx, wavelength)
        area_zone = len(xv) * grid_resolution ** 2
        covered_zone = mask.sum() * grid_resolution ** 2
        per_zone[name] = {
            "area_m2": float(area_zone),
            "covered_m2": float(covered_zone),
            "coverage_fraction": float(covered_zone / area_zone) if area_zone > 0 else 0,
        }
        total_area += area_zone
        total_covered += covered_zone
    return {
        "total_coverage_fraction": float(total_covered / total_area) if total_area > 0 else 0,
        "total_area_m2": float(total_area),
        "covered_area_m2": float(total_covered),
        "per_zone": per_zone,
    }


def search_optimal_placement(room_w: float, room_h: float, target_zones: list,
                            freq_ghz: float, candidate_step: float = 0.25,
                            grid_resolution: float = 0.05) -> dict:
    """Brute-force search over candidate (Tx, Rx) positions on the room
    perimeter. Returns the best pair + score."""
    lam = wavelength_m(freq_ghz)
    # Candidate positions: walls only (more realistic; antennas attached to walls)
    candidates = []
    for x in np.arange(0, room_w + 0.001, candidate_step):
        candidates.append(np.array([x, 0.0]))
        candidates.append(np.array([x, room_h]))
    for y in np.arange(candidate_step, room_h, candidate_step):
        candidates.append(np.array([0.0, y]))
        candidates.append(np.array([room_w, y]))

    best = {"score": -1, "tx": None, "rx": None}
    all_results = []
    for i, tx in enumerate(candidates):
        for j, rx in enumerate(candidates):
            if j <= i: continue
            # Skip degenerate (same wall, too close)
            if np.linalg.norm(tx - rx) < 1.0:
                continue
            result = coverage_score(tx, rx, target_zones, lam, grid_resolution)
            score = result["total_coverage_fraction"]
            if score > best["score"]:
                best = {
                    "score": score,
                    "tx": tx.tolist(),
                    "rx": rx.tolist(),
                    "link_length_m": float(np.linalg.norm(tx - rx)),
                    "result": result,
                }
            all_results.append({
                "tx": tx.tolist(), "rx": rx.tolist(),
                "link_m": float(np.linalg.norm(tx - rx)),
                "score": score,
            })
    return best, all_results


def main():
    parser = argparse.ArgumentParser(description="R6.2: Fresnel-aware antenna placement")
    parser.add_argument("--room", nargs=2, type=float, default=[5.0, 5.0],
                       help="Room dimensions: width height (m)")
    parser.add_argument("--target", nargs=5, action="append",
                       help="Target zone: name x0 y0 width height (m)")
    parser.add_argument("--freq-ghz", type=float, default=2.4)
    parser.add_argument("--step", type=float, default=0.25,
                       help="Candidate placement grid step (m)")
    parser.add_argument("--out", default="examples/research-sota/r6_2_placement_results.json")
    args = parser.parse_args()

    if not args.target:
        # Sensible defaults: a bedroom with a bed + a chair
        target_zones = [
            ("bed",   1.5, 0.5, 2.0, 1.5),
            ("chair", 3.5, 3.5, 0.8, 0.8),
        ]
    else:
        target_zones = []
        for t in args.target:
            name = t[0]
            x0, y0, w, h = float(t[1]), float(t[2]), float(t[3]), float(t[4])
            target_zones.append((name, x0, y0, w, h))

    print(f"Room: {args.room[0]:.1f} x {args.room[1]:.1f} m")
    print(f"Frequency: {args.freq_ghz:.2f} GHz (lambda = {wavelength_m(args.freq_ghz)*100:.2f} cm)")
    print(f"Target zones ({len(target_zones)}):")
    for name, x0, y0, w, h in target_zones:
        print(f"  {name}: ({x0:.1f}, {y0:.1f}) - ({x0+w:.1f}, {y0+h:.1f})  area={w*h:.2f} m^2")
    print()

    best, all_results = search_optimal_placement(
        args.room[0], args.room[1], target_zones, args.freq_ghz,
        candidate_step=args.step
    )

    # Worst placement, for contrast
    worst = min(all_results, key=lambda r: r["score"])
    median = sorted(all_results, key=lambda r: r["score"])[len(all_results) // 2]

    print(f"=== Search: evaluated {len(all_results)} antenna pairs ===")
    print()
    print(f"BEST placement:")
    print(f"  Tx:                {best['tx'][0]:.2f}, {best['tx'][1]:.2f}")
    print(f"  Rx:                {best['rx'][0]:.2f}, {best['rx'][1]:.2f}")
    print(f"  Link length:       {best['link_length_m']:.2f} m")
    print(f"  Coverage fraction: {best['score']*100:.1f}%")
    print(f"  Per-zone:")
    for name, info in best["result"]["per_zone"].items():
        print(f"    {name}: {info['coverage_fraction']*100:.1f}% covered ({info['covered_m2']:.2f} / {info['area_m2']:.2f} m^2)")
    print()
    print(f"MEDIAN placement: {median['score']*100:.1f}%")
    print(f"WORST  placement: {worst['score']*100:.1f}%  (link {worst['link_m']:.2f} m)")
    print()
    print(f"  Best/median improvement: {best['score']/median['score']:.2f}x")
    print(f"  Best/worst  improvement: {best['score']/(worst['score']+1e-6):.1f}x" if worst['score'] > 0 else "  Best/worst improvement: infinite (worst zero)")
    print()

    out = {
        "room": {"width_m": args.room[0], "height_m": args.room[1]},
        "frequency_ghz": args.freq_ghz,
        "wavelength_m": wavelength_m(args.freq_ghz),
        "target_zones": [
            {"name": n, "x0": x0, "y0": y0, "width": w, "height": h}
            for n, x0, y0, w, h in target_zones
        ],
        "best": best,
        "median_score": median["score"],
        "worst_score": worst["score"],
        "n_pairs_evaluated": len(all_results),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
