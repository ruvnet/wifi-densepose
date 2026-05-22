#!/usr/bin/env python3
"""R6.2.3 — Chest-centric target zones for placement search.

See docs/research/sota-2026-05-22/R6_2_3-chest-centric-placement.md.

R6.1 quantified that the chest contributes 27.6% of the total CSI
energy from a standing human -- 5x any single limb. R15's gait /
breathing / RCS primitives are all dominated by chest dynamics.

This tick re-runs R6.2's placement search with chest-only target zones
instead of full-body zones, and asks:

  Does the optimal placement change when we target chest specifically?
  How much coverage is gained by aiming at the chest envelope alone?

If the answer is "no change", placement-time chest centring is
unnecessary. If the answer is "significant change", R6.2's CLI tool
should learn pose-aware zone definitions.

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


def coverage(tx, rx, target_zones, wavelength, resolution=0.05):
    per_zone = {}
    total_pts, total_covered = 0, 0
    for name, x0, y0, w, h in target_zones:
        xs = np.arange(x0, x0 + w, resolution)
        ys = np.arange(y0, y0 + h, resolution)
        gx, gy = np.meshgrid(xs, ys)
        mask = in_first_fresnel(gx.ravel(), gy.ravel(), tx, rx, wavelength)
        n_pts = len(gx.ravel())
        per_zone[name] = {
            "area_m2": float(n_pts * resolution ** 2),
            "covered_m2": float(mask.sum() * resolution ** 2),
            "coverage_fraction": float(mask.mean()),
        }
        total_pts += n_pts
        total_covered += mask.sum()
    return {
        "total_coverage_fraction": float(total_covered / total_pts) if total_pts > 0 else 0,
        "per_zone": per_zone,
    }


def candidate_positions(room_w, room_h, step):
    cands = []
    for x in np.arange(0, room_w + 0.001, step):
        cands.append(np.array([x, 0.0]))
        cands.append(np.array([x, room_h]))
    for y in np.arange(step, room_h, step):
        cands.append(np.array([0.0, y]))
        cands.append(np.array([room_w, y]))
    return cands


def search(target_zones, room_w, room_h, freq_ghz, step):
    lam = wavelength_m(freq_ghz)
    cands = candidate_positions(room_w, room_h, step)
    best = {"score": -1, "tx": None, "rx": None, "per_zone": None}
    for i, tx in enumerate(cands):
        for j, rx in enumerate(cands):
            if j <= i: continue
            if np.linalg.norm(tx - rx) < 1.0: continue
            cov = coverage(tx, rx, target_zones, lam)
            if cov["total_coverage_fraction"] > best["score"]:
                best = {
                    "score": cov["total_coverage_fraction"],
                    "tx": tx.tolist(), "rx": rx.tolist(),
                    "link_m": float(np.linalg.norm(tx - rx)),
                    "per_zone": cov["per_zone"],
                }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/research-sota/r6_2_3_chest_centric_results.json")
    args = parser.parse_args()

    room_w, room_h = 5.0, 5.0
    freq = 2.4
    step = 0.25

    # === BODY-CENTRIC zones (R6.2 default) ===
    # Bed (full lying area), chair (full sitting area), desk (full sitting area)
    body_zones = [
        ("bed",   1.5, 0.5, 2.0, 1.5),
        ("chair", 3.5, 3.5, 0.8, 0.8),
        ("desk",  0.2, 2.5, 1.0, 0.6),
    ]

    # === CHEST-CENTRIC zones (R6.2.3 new) ===
    # The chest is approximately the upper-torso 40x30 cm region of the body.
    # Bed lying: chest at (2.5, 1.0) ± 30 cm
    # Chair sitting: chest at (3.9, 3.9) ± 20 cm
    # Desk: chest at (0.7, 2.8) ± 20 cm
    chest_zones = [
        ("bed_chest",   2.2, 0.8, 0.6, 0.4),     # 60x40 cm chest patch
        ("chair_chest", 3.7, 3.7, 0.4, 0.4),     # 40x40 cm
        ("desk_chest",  0.5, 2.7, 0.4, 0.2),     # 40x20 cm
    ]

    print(f"Room: {room_w}x{room_h} m, freq {freq} GHz")
    print()

    print("=== Body-centric placement search ===")
    best_body = search(body_zones, room_w, room_h, freq, step)
    print(f"  Best Tx: {best_body['tx']}, Rx: {best_body['rx']}")
    print(f"  Link length: {best_body['link_m']:.2f} m")
    print(f"  Total body-area coverage: {best_body['score']*100:.1f}%")
    print()

    print("=== Chest-centric placement search ===")
    best_chest = search(chest_zones, room_w, room_h, freq, step)
    print(f"  Best Tx: {best_chest['tx']}, Rx: {best_chest['rx']}")
    print(f"  Link length: {best_chest['link_m']:.2f} m")
    print(f"  Total chest-area coverage: {best_chest['score']*100:.1f}%")
    print()

    # Cross-eval: how does the body-optimal placement perform on chest zones?
    lam = wavelength_m(freq)
    body_pl_on_chest = coverage(
        np.array(best_body["tx"]), np.array(best_body["rx"]), chest_zones, lam
    )
    chest_pl_on_body = coverage(
        np.array(best_chest["tx"]), np.array(best_chest["rx"]), body_zones, lam
    )

    print("=== Cross-evaluation ===")
    print(f"  Body-optimal placement on CHEST zones:  {body_pl_on_chest['total_coverage_fraction']*100:.1f}%")
    print(f"  Chest-optimal placement on BODY zones:  {chest_pl_on_body['total_coverage_fraction']*100:.1f}%")
    print()

    chest_gain_pp = (best_chest["score"] - body_pl_on_chest["total_coverage_fraction"]) * 100
    body_loss_pp  = (best_body["score"] - chest_pl_on_body["total_coverage_fraction"]) * 100
    print(f"  Chest-targeting gain on chest zones: {chest_gain_pp:+.1f} pp")
    print(f"  Body-loss when using chest-optimal:  {body_loss_pp:+.1f} pp")
    print()

    # Verdict
    if abs(np.array(best_chest["tx"]) - np.array(best_body["tx"])).sum() < 0.6 and \
       abs(np.array(best_chest["rx"]) - np.array(best_body["rx"])).sum() < 0.6:
        verdict = "PLACEMENT STABLE: chest-centric search produces nearly the same optimal placement as body-centric. R6.2.3 is unnecessary at the placement-time level; chest-centric matters in the DSP pipeline (vital_signs.rs limb-mask), not the geometry."
    elif chest_gain_pp > 10:
        verdict = "CHEST-CENTRIC WINS: significant placement-strategy change. R6.2.3 should be a CLI option."
    else:
        verdict = "MIXED: chest and body placements differ but coverage gain is moderate. Documentation says use chest-centric for vital-signs cogs, body-centric for pose / count cogs."
    print(f"VERDICT: {verdict}")
    print()

    out = {
        "room": {"width_m": room_w, "height_m": room_h},
        "freq_ghz": freq,
        "body_zones": [{"name": n, "x": x0, "y": y0, "w": w, "h": h}
                       for n, x0, y0, w, h in body_zones],
        "chest_zones": [{"name": n, "x": x0, "y": y0, "w": w, "h": h}
                        for n, x0, y0, w, h in chest_zones],
        "best_body_centric": best_body,
        "best_chest_centric": best_chest,
        "cross_eval": {
            "body_pl_on_chest": body_pl_on_chest["total_coverage_fraction"],
            "chest_pl_on_body": chest_pl_on_body["total_coverage_fraction"],
            "chest_gain_pp": chest_gain_pp,
            "body_loss_pp": body_loss_pp,
        },
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
