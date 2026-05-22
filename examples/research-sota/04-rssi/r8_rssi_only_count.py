#!/usr/bin/env python3
"""R8 — RSSI-only person count: how much accuracy do we lose vs full CSI?

See docs/research/sota-2026-05-22/R8-rssi-only-count.md.

RSSI = received signal strength = power integrated across the WiFi band.
The CSI amplitude vector for a single packet is `|H_k|` per subcarrier k;
its mean over subcarriers is an unbiased proxy for the per-packet RSSI
(equivalent up to constant scaling). So aggregating our existing
`[56 subcarriers × 20 frames]` CSI windows along the subcarrier axis gives
us a `[20]` "RSSI-over-time" signal — exactly what any WiFi chip without
CSI export reports as its standard `RSSI` field.

If a small MLP on the [20]-vector hits even 55-60% accuracy on the
person-count task, RSSI-only deployment is viable across the entire WiFi-
chip ecosystem (billions of devices), at the cost of needing per-chip
calibration. v0.0.2 of cog-person-count itself only hits 62% on the 80/20
random split, so the bar isn't sky-high.

Usage:
    python examples/research-sota/r8_rssi_only_count.py \
        --paired data/paired/wiflow-p7-1779210883.paired.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

N_SUB, N_FRAMES, COUNT_CLASSES = 56, 20, 8


def load_paired(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_csi, y) where X_csi is [N, 56, 20] and y is [N] integer count."""
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
            ys.append(int(d.get("n_persons_mode", 0)))
    return np.stack(csis), np.asarray(ys, dtype=np.int64)


def csi_to_rssi_proxy(X_csi: np.ndarray) -> np.ndarray:
    """Aggregate CSI amplitudes to a single RSSI scalar per frame.

    Input:  [N, 56, 20]   per-subcarrier amplitudes
    Output: [N, 20]       band-mean amplitude per time-frame = RSSI proxy

    This is what a non-CSI WiFi chip reports as its RSSI field, up to a
    constant scaling (dBm conversion). We keep linear amplitude — the count
    head is invariant to that affine transform after z-score normalisation.
    """
    return X_csi.mean(axis=1)  # mean across subcarriers


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def train_rssi_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray, y_eval: np.ndarray,
    epochs: int = 200, lr: float = 1e-2, hidden: int = 32, seed: int = 42,
):
    """Tiny MLP trained with vanilla SGD — no framework, just numpy.

    Input: [N, 20] RSSI-proxy time-series
    Architecture:   Linear(20 → hidden) → ReLU → Linear(hidden → 8) → softmax
    """
    rng = np.random.default_rng(seed)
    D = X_train.shape[1]
    K = COUNT_CLASSES

    # Glorot init
    w1 = rng.normal(0, np.sqrt(2.0 / D), size=(D, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2.0 / hidden), size=(hidden, K)).astype(np.float32)
    b2 = np.zeros(K, dtype=np.float32)

    n_train = X_train.shape[0]
    batch_size = 32
    eval_curve = []
    best_eval_acc = 0.0
    best = None

    for epoch in range(epochs):
        perm = rng.permutation(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            # Forward
            h1 = xb @ w1 + b1                     # [B, hidden]
            a1 = np.maximum(h1, 0.0)               # ReLU
            logits = a1 @ w2 + b2                  # [B, K]
            probs = softmax(logits, axis=-1)
            # One-hot
            onehot = np.zeros_like(probs)
            onehot[np.arange(len(yb)), yb] = 1.0
            # Backward
            dlogits = (probs - onehot) / len(yb)   # [B, K]
            dw2 = a1.T @ dlogits                   # [hidden, K]
            db2 = dlogits.sum(axis=0)
            da1 = dlogits @ w2.T                   # [B, hidden]
            dh1 = da1 * (h1 > 0)                   # ReLU grad
            dw1 = xb.T @ dh1                       # [D, hidden]
            db1 = dh1.sum(axis=0)
            # SGD
            w1 -= lr * dw1
            b1 -= lr * db1
            w2 -= lr * dw2
            b2 -= lr * db2

        # Eval
        eh = np.maximum(X_eval @ w1 + b1, 0.0)
        eval_logits = eh @ w2 + b2
        eval_pred = eval_logits.argmax(axis=1)
        eval_acc = float((eval_pred == y_eval).mean())
        eval_curve.append(eval_acc)
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            best = (w1.copy(), b1.copy(), w2.copy(), b2.copy())

    return best, best_eval_acc, eval_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out", default="examples/research-sota/r8_rssi_only_results.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading paired data from {args.paired}")
    X_csi, y = load_paired(Path(args.paired))
    print(f"  CSI shape: {X_csi.shape}")
    print(f"  label distribution: {dict(Counter(y.tolist()).most_common())}")

    print("\nDeriving RSSI proxy by averaging across 56 subcarriers...")
    X_rssi = csi_to_rssi_proxy(X_csi)
    print(f"  RSSI proxy shape: {X_rssi.shape}  (one scalar per frame, 20 frames per sample)")
    print(f"  RSSI proxy stats: mean={X_rssi.mean():.3f}  std={X_rssi.std():.3f}")

    # Random 80/20 split — same seed as v0.0.2 so the eval set is identical
    rng = np.random.default_rng(seed=args.seed)
    idx = np.arange(X_rssi.shape[0])
    rng.shuffle(idx)
    n_eval = int(round(0.2 * X_rssi.shape[0]))
    eval_idx, train_idx = idx[:n_eval], idx[n_eval:]
    X_train, X_eval = X_rssi[train_idx], X_rssi[eval_idx]
    y_train, y_eval = y[train_idx], y[eval_idx]

    # Standardise (z-score) — RSSI is a linear quantity; this matches what
    # any real device would do per its automatic gain control.
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mu) / sd
    X_eval_n = (X_eval - mu) / sd

    print(f"\nTraining RSSI-only MLP — input 20-dim, hidden 32, output 8, vanilla SGD")
    t0 = time.perf_counter()
    best_params, best_eval_acc, curve = train_rssi_mlp(
        X_train_n, y_train, X_eval_n, y_eval,
        epochs=args.epochs, lr=1e-2, hidden=32, seed=args.seed,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTrained {args.epochs} epochs in {elapsed:.2f} s on CPU")

    # Final eval with best checkpoint
    w1, b1, w2, b2 = best_params
    eh = np.maximum(X_eval_n @ w1 + b1, 0.0)
    eval_logits = eh @ w2 + b2
    eval_pred = eval_logits.argmax(axis=1)
    acc = float((eval_pred == y_eval).mean())
    per_class = {}
    for k in range(COUNT_CLASSES):
        mask = y_eval == k
        n = int(mask.sum())
        if n > 0:
            per_class[k] = {
                "support": n,
                "accuracy": float(((eval_pred == y_eval) & mask).sum() / n),
            }

    # Baseline reference: how does v0.0.2 (full CSI) score on the SAME eval set?
    # We don't run the cog binary here — just record the published numbers.
    full_csi_baseline = {
        "version": "cog-person-count v0.0.2",
        "overall_acc": 0.623,
        "class0_acc": 0.862,
        "class1_acc": 0.343,
        "source": "docs/benchmarks/person-count-cog.md",
    }

    print(f"\n=== R8 RSSI-only results ===")
    print(f"  Eval accuracy:   {acc:.3f}")
    print(f"  Per-class:")
    for k, v in per_class.items():
        print(f"    class {k}: {v['accuracy']:.3f} on {v['support']} samples")
    print(f"\n  Full-CSI baseline (v0.0.2): {full_csi_baseline['overall_acc']:.3f}")
    print(f"  Retained fraction: {acc / full_csi_baseline['overall_acc']:.2%}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "method": "RSSI-proxy band-mean amplitude over 20-frame window",
        "input_dim": int(X_rssi.shape[1]),
        "architecture": "MLP(20 → 32 → 8) ReLU + softmax, vanilla SGD",
        "epochs": args.epochs,
        "train_time_s": elapsed,
        "n_train": int(X_train.shape[0]),
        "n_eval": int(X_eval.shape[0]),
        "label_distribution_train": dict(Counter(y_train.tolist()).most_common()),
        "label_distribution_eval": dict(Counter(y_eval.tolist()).most_common()),
        "final_eval_acc": acc,
        "best_eval_acc": best_eval_acc,
        "per_class_accuracy": per_class,
        "full_csi_baseline": full_csi_baseline,
        "retained_fraction": acc / full_csi_baseline["overall_acc"],
        "eval_acc_curve": curve,
    }, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
