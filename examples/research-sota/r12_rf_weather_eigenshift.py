#!/usr/bin/env python3
"""R12 — RF weather: can SVD-eigenvalue drift detect structural changes?

See docs/research/sota-2026-05-22/R12-rf-weather-mapping.md.

The persistent-room field model in `wifi-densepose-signal/src/ruvsense/
field_model.rs` does an SVD on empty-room CSI to extract an eigenstructure
that describes "what this room's RF reflection looks like with nobody
in it". Today that's used to subtract the room's baseline so motion
detection isn't confused by static multipath.

This experiment asks a different question: **does the eigenvalue
*spectrum* itself drift in a detectable way when something structural
changes in the room?** "Structural change" = a new piece of furniture,
a window that opened, water in the wall, settled foundation, missing
ceiling tile. The 10-year vision (R12 research note) is continuous
building-integrity monitoring from passive ambient WiFi.

Test:
  1. Take the existing 1,077 CSI windows. Split first 50% = "before",
     last 50% = "after".
  2. Inject a synthetic "structural perturbation" into the "after"
     half — multiply 3 subcarriers by 0.85 (simulating a new reflective
     surface that attenuates those frequencies).
  3. For each half, stack the windows into a `[N, 56]` per-frame
     matrix (each row = one timestep), compute SVD, take the top-10
     singular values.
  4. Measure: do the singular-value spectra differ in a way that
     distinguishes "structural perturbation present" from "no
     perturbation"?
  5. Repeat with NO perturbation as control — the same first-half /
     second-half split should produce *similar* spectra (just temporal
     drift from operator movement, not structural).

If the perturbed-vs-control eigenvalue spectra are distinguishable by
a simple distance metric, RF-weather detection is feasible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

N_SUB, N_FRAMES = 56, 20


def load_windows(path: Path, max_samples: int | None = None) -> np.ndarray:
    csis = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            shape = d.get("csi_shape", [N_SUB, N_FRAMES])
            if shape != [N_SUB, N_FRAMES]:
                continue
            csi = np.asarray(d["csi"], dtype=np.float32).reshape(N_SUB, N_FRAMES)
            csis.append(csi)
            if max_samples and len(csis) >= max_samples:
                break
    return np.stack(csis)


def perturb_subcarriers(X: np.ndarray, indices: list[int], gain: float) -> np.ndarray:
    """Multiply the listed subcarriers by `gain` to simulate a structural
    change (e.g. a new reflector attenuates certain frequencies)."""
    out = X.copy()
    out[:, indices, :] *= gain
    return out


def per_frame_matrix(X: np.ndarray) -> np.ndarray:
    """Stack all windows' frames into a [N_total_frames, 56] matrix.
    Each row is one timestep, used as a multivariate observation of the
    56-subcarrier channel state."""
    return X.transpose(0, 2, 1).reshape(-1, N_SUB)


def top_k_singular_values(M: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute SVD on M, return top-k singular values."""
    M_centered = M - M.mean(axis=0, keepdims=True)
    # Use SVD on the centered matrix (== PCA without normalisation)
    s = np.linalg.svd(M_centered, compute_uv=False)
    return s[:k]


def spectrum_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """Cosine distance between two singular-value spectra. 0 = identical
    direction, 2 = opposite. Symmetric, scale-invariant."""
    s1n = s1 / (np.linalg.norm(s1) + 1e-9)
    s2n = s2 / (np.linalg.norm(s2) + 1e-9)
    return float(1.0 - np.dot(s1n, s2n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out", default="examples/research-sota/r12_rf_weather_results.json")
    parser.add_argument("--perturb-indices", default="30,41,52",
                        help="comma-separated subcarrier indices to perturb (chosen from R5's top-saliency list)")
    parser.add_argument("--perturb-gain", type=float, default=0.85)
    args = parser.parse_args()

    print(f"Loading windows from {args.paired}")
    X = load_windows(Path(args.paired))
    print(f"  total windows: {X.shape[0]} (shape {X.shape})")

    n = X.shape[0]
    half = n // 2
    X_before = X[:half]
    X_after_raw = X[half:]  # unmodified second half — the CONTROL
    perturb_idx = [int(x) for x in args.perturb_indices.split(",")]
    X_after_perturbed = perturb_subcarriers(X_after_raw, perturb_idx, args.perturb_gain)

    # Convert each half to a [N_frames, 56] matrix
    M_before = per_frame_matrix(X_before)
    M_after_raw = per_frame_matrix(X_after_raw)
    M_after_pert = per_frame_matrix(X_after_perturbed)
    print(f"  per-frame matrix: before={M_before.shape}, after={M_after_raw.shape}")

    # Top-10 singular values per half
    s_before = top_k_singular_values(M_before, k=10)
    s_after_raw = top_k_singular_values(M_after_raw, k=10)
    s_after_pert = top_k_singular_values(M_after_pert, k=10)

    print(f"\n  Singular value spectra (top-10):")
    print(f"    before        :  [{', '.join(f'{v:.1f}' for v in s_before)}]")
    print(f"    after  (raw)  :  [{', '.join(f'{v:.1f}' for v in s_after_raw)}]")
    print(f"    after  (pert) :  [{', '.join(f'{v:.1f}' for v in s_after_pert)}]")

    # Distances
    d_raw = spectrum_distance(s_before, s_after_raw)
    d_pert = spectrum_distance(s_before, s_after_pert)

    print(f"\n  Cosine distances from BEFORE:")
    print(f"    before -> after raw   (control, no perturbation): {d_raw:.5f}")
    print(f"    before -> after pert  (synthetic structural shift): {d_pert:.5f}")

    # Distance ratio = how much the perturbation amplifies the detection signal
    # over the natural temporal drift.
    if d_raw > 1e-9:
        ratio = d_pert / d_raw
        print(f"\n  Signal-to-natural-drift ratio: {ratio:.2f}x")

    if d_pert > d_raw * 3:
        verdict = "STRONG: perturbation easily distinguishable from natural temporal drift"
    elif d_pert > d_raw * 1.5:
        verdict = "MODERATE: perturbation detectable but with margin"
    else:
        verdict = "WEAK: structural perturbation gets lost in temporal drift"
    print(f"\n  Verdict: {verdict}")

    out = {
        "perturbation": {
            "subcarrier_indices": perturb_idx,
            "amplitude_gain": args.perturb_gain,
            "comment": "simulates a new reflective surface that attenuates these frequencies",
        },
        "n_before_windows": int(half),
        "n_after_windows": int(n - half),
        "spectra": {
            "before": s_before.tolist(),
            "after_raw_control": s_after_raw.tolist(),
            "after_perturbed": s_after_pert.tolist(),
        },
        "distances": {
            "before_to_after_raw": d_raw,
            "before_to_after_perturbed": d_pert,
            "signal_over_natural_drift": float(d_pert / max(d_raw, 1e-9)),
        },
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
