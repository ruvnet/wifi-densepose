#!/usr/bin/env python3
"""R9 — RSSI fingerprint topology: does temporal proximity = feature proximity?

See docs/research/sota-2026-05-22/R9-rssi-fingerprint-knn.md.

Hypothesis: if RSSI sequences from temporally-adjacent windows are
nearest-neighbours in feature space, RSSI-fingerprint localisation is
viable. If the K-NN of every query is random in time, RSSI sequences
don't carry stable enough fingerprints — fall back to multi-modal cues
(BSSID lists, signal-of-opportunity).

Test:
  1. Build the same 20-dim RSSI proxy from the 1,077 paired windows
     (band-mean across 56 subcarriers per frame).
  2. For each sample i, find K-NN in cosine-similarity space.
  3. Measure: what fraction of the K-NN come from windows within
     ±60 seconds of the query's timestamp?
  4. Compare to a random baseline (what would the fraction be if K-NN
     were chosen at random?).

If the temporal-K-NN fraction is ≫ random, RSSI fingerprints have stable
spatial structure → R9 viable.

Usage:
    python examples/research-sota/r9_rssi_fingerprint_knn.py \
        --paired data/paired/wiflow-p7-1779210883.paired.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

N_SUB, N_FRAMES = 56, 20


def load_rssi_proxy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_rssi, ts_seconds). X_rssi is [N, 20], ts is [N] float seconds."""
    csis, ts = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            shape = d.get("csi_shape", [N_SUB, N_FRAMES])
            if shape != [N_SUB, N_FRAMES]:
                continue
            csi = np.asarray(d["csi"], dtype=np.float32).reshape(N_SUB, N_FRAMES)
            csis.append(csi.mean(axis=0))  # band-mean → [20]
            t_iso = d.get("ts_start", "1970-01-01T00:00:00Z")
            ts.append(datetime.fromisoformat(t_iso.replace("Z", "+00:00")).timestamp())
    return np.stack(csis), np.asarray(ts, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out", default="examples/research-sota/r9_rssi_fingerprint_results.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temporal-window-s", type=float, default=60.0)
    args = parser.parse_args()

    print(f"Loading RSSI-proxy from {args.paired}")
    X, ts = load_rssi_proxy(Path(args.paired))
    print(f"  N samples: {X.shape[0]}, feature dim: {X.shape[1]}")
    print(f"  time range: {datetime.fromtimestamp(ts.min(), tz=timezone.utc):%H:%M:%S} - "
          f"{datetime.fromtimestamp(ts.max(), tz=timezone.utc):%H:%M:%S}  "
          f"({(ts.max() - ts.min()) / 60:.1f} min total)")

    # Z-score normalise across all samples — what a real device does via AGC
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    # All-pairs cosine similarity
    print(f"\nComputing all-pairs cosine similarity ({X.shape[0]}×{X.shape[0]} = "
          f"{X.shape[0]**2:,} pairs)...")
    norms = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-9
    Xnorm = Xn / norms
    sim = Xnorm @ Xnorm.T
    np.fill_diagonal(sim, -np.inf)  # exclude self-match

    N = X.shape[0]
    K = args.k
    W = args.temporal_window_s

    # For each query, find top-K nearest neighbours and measure how many are
    # within the temporal window
    print(f"\nMeasuring temporal-locality of top-{K} cosine-NN with window ±{W:.0f}s...")
    knn_idx = np.argsort(-sim, axis=1)[:, :K]   # [N, K]
    knn_ts = ts[knn_idx]                         # [N, K]
    delta_t = np.abs(knn_ts - ts[:, None])      # [N, K]
    within = (delta_t <= W).astype(np.float32)   # [N, K]
    per_query_within_frac = within.mean(axis=1) # [N] — fraction of K-NN within window
    overall_within_frac = within.mean()         # scalar

    # Random baseline: for each query, what fraction of all OTHER samples
    # fall within ±W of its timestamp?
    rand_within = np.zeros(N, dtype=np.float32)
    for i in range(N):
        delta = np.abs(ts - ts[i])
        delta[i] = np.inf
        rand_within[i] = (delta <= W).mean()
    rand_baseline = float(rand_within.mean())

    # Headline numbers
    lift = overall_within_frac / max(rand_baseline, 1e-9)

    print(f"\n=== R9 RSSI-fingerprint K-NN results ===")
    print(f"  K-NN within ±{W:.0f}s:   {overall_within_frac:.3f}")
    print(f"  Random baseline:        {rand_baseline:.3f}")
    print(f"  Lift over random:       {lift:.2f}×")
    print(f"  Per-query stdev:        {per_query_within_frac.std():.3f}")

    if lift >= 3.0:
        verdict = "STRONG: RSSI sequences carry stable spatial fingerprints"
    elif lift >= 1.5:
        verdict = "MODERATE: RSSI fingerprints work but with significant noise"
    else:
        verdict = "WEAK: RSSI-only fingerprint localisation is unreliable on this data"
    print(f"\n  Verdict: {verdict}")

    out = {
        "n_samples": int(N),
        "k": K,
        "temporal_window_s": W,
        "knn_within_window_fraction": float(overall_within_frac),
        "random_baseline": rand_baseline,
        "lift": float(lift),
        "per_query_within_fraction_stdev": float(per_query_within_frac.std()),
        "verdict": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
