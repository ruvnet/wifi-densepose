#!/usr/bin/env python3
"""R6.2.2 — N-anchor multistatic Fresnel-coverage placement.

See docs/research/sota-2026-05-22/R6_2_2-multistatic-placement.md.

Extends R6.2 from single-pair to N anchors with all C(N,2) pairwise
Fresnel ellipses. A point is covered if it lies inside the union of
any pairwise Fresnel zone.

Practical question: how many seeds does a typical room need?
Answer: report saturation curve over N = 2..8 anchors.

Search is greedy + restart (full combinatorial O(M^N) is too expensive
for M ~100 candidates). Greedy adds the anchor that maximises marginal
coverage at each step; restart picks the best of K greedy runs from
different starting points to escape local minima.

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


def in_first_fresnel(x: np.ndarray, y: np.ndarray, tx: np.ndarray, rx: np.ndarray,
                     wavelength: float) -> np.ndarray:
    r1 = np.sqrt((x - tx[0])**2 + (y - tx[1])**2)
    r2 = np.sqrt((x - rx[0])**2 + (y - rx[1])**2)
    direct = np.linalg.norm(tx - rx)
    return (r1 + r2) <= (direct + wavelength / 2)


def union_coverage(anchors: list, target_grid_x: np.ndarray, target_grid_y: np.ndarray,
                   wavelength: float) -> float:
    """Fraction of target points covered by at least one pairwise Fresnel ellipse."""
    if len(anchors) < 2:
        return 0.0
    covered = np.zeros(len(target_grid_x), dtype=bool)
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            mask = in_first_fresnel(target_grid_x, target_grid_y,
                                   anchors[i], anchors[j], wavelength)
            covered |= mask
    return float(covered.sum() / len(target_grid_x))


def rasterise_targets(target_zones: list, resolution: float) -> tuple:
    """Flatten target zones into (x, y) arrays."""
    xs, ys = [], []
    for name, x0, y0, w, h in target_zones:
        zx = np.arange(x0, x0 + w, resolution)
        zy = np.arange(y0, y0 + h, resolution)
        gx, gy = np.meshgrid(zx, zy)
        xs.append(gx.ravel())
        ys.append(gy.ravel())
    return np.concatenate(xs), np.concatenate(ys)


def candidate_positions(room_w: float, room_h: float, step: float) -> list:
    """Wall-perimeter candidate antenna positions."""
    cands = []
    for x in np.arange(0, room_w + 0.001, step):
        cands.append(np.array([x, 0.0]))
        cands.append(np.array([x, room_h]))
    for y in np.arange(step, room_h, step):
        cands.append(np.array([0.0, y]))
        cands.append(np.array([room_w, y]))
    return cands


def greedy_search(candidates: list, target_x: np.ndarray, target_y: np.ndarray,
                  wavelength: float, n_anchors: int, n_restarts: int = 8,
                  seed: int = 0) -> dict:
    """Greedy: at each step, add the candidate that maximises marginal coverage.
    Restart K times from random initial pairs to escape local minima."""
    rng = np.random.default_rng(seed)
    best = {"anchors": [], "score": -1.0, "trace": []}
    for restart in range(n_restarts):
        # Random initial pair
        idx0, idx1 = rng.choice(len(candidates), size=2, replace=False)
        chosen = [candidates[idx0], candidates[idx1]]
        trace = [union_coverage(chosen, target_x, target_y, wavelength)]
        while len(chosen) < n_anchors:
            best_marginal = -1.0
            best_idx = None
            for k, c in enumerate(candidates):
                if any(np.allclose(c, a) for a in chosen):
                    continue
                trial = chosen + [c]
                score = union_coverage(trial, target_x, target_y, wavelength)
                if score > best_marginal:
                    best_marginal = score
                    best_idx = k
            if best_idx is None:
                break
            chosen.append(candidates[best_idx])
            trace.append(best_marginal)
        final = trace[-1]
        if final > best["score"]:
            best = {
                "anchors": [a.tolist() for a in chosen],
                "score": final,
                "trace": trace,
                "restart_used": restart,
            }
    return best


def main():
    parser = argparse.ArgumentParser(description="R6.2.2: N-anchor Fresnel multistatic placement")
    parser.add_argument("--room", nargs=2, type=float, default=[5.0, 5.0])
    parser.add_argument("--freq-ghz", type=float, default=2.4)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--n-max", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--out", default="examples/research-sota/r6_2_2_multistatic_results.json")
    args = parser.parse_args()

    target_zones = [
        ("bed",   1.5, 0.5, 2.0, 1.5),
        ("chair", 3.5, 3.5, 0.8, 0.8),
        ("desk",  0.2, 2.5, 1.0, 0.6),  # third zone for more interesting saturation
    ]
    lam = wavelength_m(args.freq_ghz)
    candidates = candidate_positions(args.room[0], args.room[1], args.step)
    target_x, target_y = rasterise_targets(target_zones, 0.1)

    print(f"Room: {args.room[0]:.1f} x {args.room[1]:.1f} m")
    print(f"Frequency: {args.freq_ghz} GHz (lambda = {lam*100:.2f} cm)")
    print(f"Targets: {len(target_zones)} zones, {len(target_x)} grid points")
    print(f"Candidates: {len(candidates)} positions (step={args.step}m)")
    print()

    saturation = []
    for n in range(2, args.n_max + 1):
        result = greedy_search(candidates, target_x, target_y, lam,
                              n_anchors=n, n_restarts=args.restarts)
        saturation.append({
            "n_anchors": n,
            "coverage": result["score"],
            "n_pairs_used": n * (n - 1) // 2,
            "anchors": result["anchors"],
        })

    # Marginal coverage per additional anchor
    marginal = []
    for i in range(1, len(saturation)):
        prev = saturation[i-1]["coverage"]
        curr = saturation[i]["coverage"]
        marginal.append({
            "from_n": saturation[i-1]["n_anchors"],
            "to_n": saturation[i]["n_anchors"],
            "marginal_coverage_pp": (curr - prev) * 100,
        })

    print("=== Coverage saturation ===")
    print(f"{'N anchors':>10}  {'Pairs':>6}  {'Coverage':>9}  {'Marginal':>9}")
    prev = 0.0
    for s in saturation:
        marg = (s["coverage"] - prev) * 100
        print(f"{s['n_anchors']:>10}  {s['n_pairs_used']:>6}  {s['coverage']*100:>7.1f}%  {marg:>+7.1f} pp")
        prev = s["coverage"]

    print()
    # Knee detection
    for i, m in enumerate(marginal):
        if m["marginal_coverage_pp"] < 5.0:
            print(f"Knee detected: going from N={m['from_n']} to N={m['to_n']} adds only {m['marginal_coverage_pp']:.1f} pp")
            print(f"  Practical N = {m['from_n']} anchors (diminishing returns past this)")
            break

    out = {
        "room": {"width_m": args.room[0], "height_m": args.room[1]},
        "frequency_ghz": args.freq_ghz,
        "target_zones": [
            {"name": n, "x0": x0, "y0": y0, "width": w, "height": h}
            for n, x0, y0, w, h in target_zones
        ],
        "saturation": saturation,
        "marginal_gains_pp": marginal,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
