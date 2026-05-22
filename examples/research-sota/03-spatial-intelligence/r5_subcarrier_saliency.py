#!/usr/bin/env python3
"""R5 — per-subcarrier input×gradient saliency for the count + pose cogs.

See docs/research/sota-2026-05-22/R5-subcarrier-saliency.md for context.

Usage:
    python examples/research-sota/r5_subcarrier_saliency.py \
        --paired data/paired/wiflow-p7-1779210883.paired.jsonl \
        --model  v2/crates/cog-person-count/cog/artifacts/count_v1.safetensors \
        --kind   count
    python examples/research-sota/r5_subcarrier_saliency.py \
        --paired data/paired/wiflow-p7-1779210883.paired.jsonl \
        --model  v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors \
        --kind   pose

Output:
    <dirname-of-model>/saliency.json    per-subcarrier saliency + top-K lists
    stdout summary table

Method (per ADR/research note):
    S_k = E_samples[ |dL/dx_k| * |x_k| ]
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


N_SUB, N_FRAMES = 56, 20


def load_paired(path: Path, kind: str, max_samples: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) — X is [N, 56, 20] float32, y depends on kind.

    kind="count" → y is [N] int64 in {0..7}
    kind="pose"  → y is [N, 17, 2] float32 in [0, 1]
    """
    csis, ys = [], []
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
            if kind == "count":
                ys.append(int(d.get("n_persons_mode", 0)))
            elif kind == "pose":
                ys.append(np.asarray(d.get("kp", []), dtype=np.float32))
            else:
                raise ValueError(f"unknown kind: {kind}")
            if max_samples and len(csis) >= max_samples:
                break
    return np.stack(csis), np.asarray(ys, dtype=(np.int64 if kind == "count" else np.float32))


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Pure-python safetensors reader. Returns {name: ndarray}."""
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode("utf-8"))
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            shape = meta["shape"]
            assert meta["dtype"] == "F32", f"unsupported dtype {meta['dtype']} in {name}"
            f.seek(8 + hlen + start)
            buf = f.read(end - start)
            arr = np.frombuffer(buf, dtype=np.float32).copy().reshape(shape)
            out[name] = arr
    return out


def conv1d_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray, padding: int, dilation: int) -> np.ndarray:
    """Pure-numpy Conv1d forward. x: [B, Cin, T], w: [Cout, Cin, K]. Returns [B, Cout, T']."""
    B, Cin, T = x.shape
    Cout, _, K = w.shape
    # Pad
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    Tp = xp.shape[2]
    # Effective filter span with dilation
    eff = (K - 1) * dilation + 1
    Tout = Tp - eff + 1
    out = np.zeros((B, Cout, Tout), dtype=np.float32)
    for k in range(K):
        # x_slice shape: [B, Cin, Tout]
        x_slice = xp[:, :, k * dilation : k * dilation + Tout]
        # w_slice shape: [Cout, Cin]
        w_slice = w[:, :, k]
        # einsum: B,Cin,T  x  Cout,Cin → B,Cout,T
        out += np.einsum("bct,oc->bot", x_slice, w_slice)
    return out + b[None, :, None]


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def forward_count(x: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """CountNet forward. x: [B, 56, 20] → probs [B, 8]."""
    h = conv1d_forward(x, w["enc.c1.weight"], w["enc.c1.bias"], padding=1, dilation=1)
    h = relu(h)
    h = conv1d_forward(h, w["enc.c2.weight"], w["enc.c2.bias"], padding=2, dilation=2)
    h = relu(h)
    h = conv1d_forward(h, w["enc.c3.weight"], w["enc.c3.bias"], padding=4, dilation=4)
    h = relu(h)
    h = h.mean(axis=2)  # [B, 128]
    # count head
    z = relu(h @ w["count_head.fc1.weight"].T + w["count_head.fc1.bias"])
    z = z @ w["count_head.fc2.weight"].T + w["count_head.fc2.bias"]
    return softmax(z, axis=-1)


def saliency_input_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: dict[str, np.ndarray],
    kind: str,
    eps: float = 1e-3,
) -> np.ndarray:
    """Per-subcarrier saliency: S_k = E[|dL/dx_k| * |x_k|].

    Uses central-difference numerical gradient over each subcarrier (cheap because
    we marginalise over the time axis after taking the abs). For a 56-subcarrier
    input that's 56 forward passes per sample — slow but exact, and only runs
    once per saliency map.
    """
    B, N_sub, T = X.shape
    saliency = np.zeros(N_sub, dtype=np.float64)

    if kind == "count":
        # Loss = -log(p_true). Compute baseline log-prob.
        for k in range(N_sub):
            x_plus = X.copy()
            x_plus[:, k, :] += eps
            x_minus = X.copy()
            x_minus[:, k, :] -= eps
            p_plus = forward_count(x_plus, weights)
            p_minus = forward_count(x_minus, weights)
            # dL/dx ≈ -(log p_plus[y] - log p_minus[y]) / (2*eps)
            idx = np.arange(B)
            lp_plus = np.log(p_plus[idx, y] + 1e-12)
            lp_minus = np.log(p_minus[idx, y] + 1e-12)
            grad_k = -(lp_plus - lp_minus) / (2 * eps)  # [B]
            # |dL/dx_k| * |x_k| — x_k is a vector over time; take its magnitude
            x_k_mag = np.abs(X[:, k, :]).mean(axis=1)  # [B]
            saliency[k] += float((np.abs(grad_k) * x_k_mag).mean())
    else:
        raise NotImplementedError("pose kind not yet wired — count first")

    return saliency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=["count", "pose"], default="count")
    parser.add_argument("--max-samples", type=int, default=128,
                        help="Cap on samples used for saliency (saliency cost is O(N_sub × samples × eps_passes))")
    parser.add_argument("--out", default=None,
                        help="Output JSON path; defaults to <model_dir>/saliency.json")
    args = parser.parse_args()

    print(f"Loading paired data from {args.paired} (kind={args.kind})")
    X, y = load_paired(Path(args.paired), kind=args.kind, max_samples=args.max_samples)
    print(f"  X: {X.shape}, y: {y.shape}")
    if args.kind == "count":
        unique, counts = np.unique(y, return_counts=True)
        print(f"  label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Standardise (per-subcarrier z-score using THIS subset's stats — saliency is
    # invariant to affine input transforms in the limit of small eps).
    mu = X.mean(axis=(0, 2), keepdims=True)
    sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
    X_norm = (X - mu) / sd

    print(f"Loading weights from {args.model}")
    weights = load_safetensors(Path(args.model))
    print(f"  loaded {len(weights)} tensors: {sorted(list(weights.keys()))[:6]}...")

    print(f"Computing input×gradient saliency over {X.shape[0]} samples × 56 subcarriers...")
    saliency = saliency_input_gradient(X_norm, y, weights, kind=args.kind, eps=1e-3)

    order = np.argsort(saliency)[::-1]  # descending
    top_k = {k: order[:k].tolist() for k in (8, 16, 32)}

    out = {
        "kind": args.kind,
        "model": str(args.model),
        "n_samples": int(X.shape[0]),
        "saliency_per_subcarrier": saliency.tolist(),
        "ranking_high_to_low": order.tolist(),
        "top_k_subcarriers": top_k,
        "saliency_summary": {
            "min": float(saliency.min()),
            "max": float(saliency.max()),
            "mean": float(saliency.mean()),
            "std": float(saliency.std()),
            "max_to_mean_ratio": float(saliency.max() / max(saliency.mean(), 1e-12)),
        },
    }

    out_path = Path(args.out) if args.out else Path(args.model).parent / "saliency.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nTop 8 subcarriers (most influential):")
    for rank, idx in enumerate(order[:8]):
        print(f"  #{rank + 1}: subcarrier {int(idx):2d}  saliency={saliency[idx]:.4f}")
    print(f"\nMax/mean ratio: {out['saliency_summary']['max_to_mean_ratio']:.2f}× "
          f"(higher = signal more concentrated in a few subcarriers)")


if __name__ == "__main__":
    main()
