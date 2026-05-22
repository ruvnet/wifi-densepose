#!/usr/bin/env python3
"""R7 — multi-link consistency detection via Stoer-Wagner-style mincut.

See docs/research/sota-2026-05-22/R7-multilink-consistency.md.

Premise: in a multi-node CSI mesh, all nodes observe the same physical
scene through slightly different channels. Their per-window CSI features
should cluster tightly under a similarity metric. If one node is
compromised (spoofed CSI, replay attack, jamming-induced corruption), its
features fall outside the cluster — and the mincut of the inter-node
similarity graph isolates it cleanly.

This demo:
  1. Synthesises 4 "honest" CSI windows from one underlying scene + per-node
     Gaussian noise (realistic multipath variability).
  2. Synthesises 1 "adversarial" CSI window via three attack modes:
       (a) replay  — paste in a stale window from earlier
       (b) shift   — add a constant offset to every subcarrier
       (c) noise   — pure white noise of the same magnitude as honest CSI
  3. Builds a 5×5 cross-node CSI cosine-similarity matrix.
  4. Solves Stoer-Wagner mincut on the resulting graph.
  5. Reports whether the mincut partition isolates the adversarial node.

No framework deps — pure NumPy.

Usage:
    python examples/research-sota/r7_multilink_consistency.py \
        --paired data/paired/wiflow-p7-1779210883.paired.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

N_SUB, N_FRAMES = 56, 20


def load_one_window(path: Path, idx: int = 0) -> np.ndarray:
    """Pull one [56, 20] CSI window from the paired data — the scene we'll synthesise around."""
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < idx:
                continue
            d = json.loads(line)
            shape = d.get("csi_shape", [N_SUB, N_FRAMES])
            if shape == [N_SUB, N_FRAMES]:
                return np.asarray(d["csi"], dtype=np.float32).reshape(N_SUB, N_FRAMES)
            return None
    return None


def synth_honest_nodes(base: np.ndarray, n_nodes: int = 4, noise_db: float = 6.0, seed: int = 42):
    """`n_nodes` honest observers — each sees the base scene through independent multipath
    (modelled as additive Gaussian on the per-subcarrier amplitudes at `noise_db` below signal)."""
    rng = np.random.default_rng(seed)
    sigma = base.std() * 10 ** (-noise_db / 20.0)
    return np.stack([base + rng.normal(0, sigma, size=base.shape).astype(np.float32) for _ in range(n_nodes)])


def synth_adversarial(base: np.ndarray, mode: str, replay_window: np.ndarray | None = None, seed: int = 7):
    """One adversarial observer. `mode` ∈ {replay, shift, noise}."""
    rng = np.random.default_rng(seed)
    if mode == "replay":
        if replay_window is None:
            raise ValueError("replay needs a stale window")
        # Stale window with a tiny perturbation to look "fresh"
        return replay_window + rng.normal(0, 0.01, size=base.shape).astype(np.float32)
    if mode == "shift":
        return base + 3.0 * base.std()  # constant offset — gives away the attack
    if mode == "noise":
        return rng.normal(base.mean(), base.std(), size=base.shape).astype(np.float32)
    raise ValueError(f"unknown adversarial mode: {mode}")


def cosine_sim_matrix(windows: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity on flattened windows. Returns [N, N] matrix."""
    flat = windows.reshape(windows.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9
    normalized = flat / norms
    return normalized @ normalized.T


def stoer_wagner_mincut(W: np.ndarray) -> tuple[float, list[int]]:
    """Classical Stoer-Wagner mincut. Input: symmetric [N, N] non-negative weights.

    Returns: (cut_value, partition_a_node_indices)

    The algorithm:
      while G has more than one node:
        do a minimum-cut-phase: find the order in which nodes are added
        the last node added is one side of a candidate cut; the rest is the other side
        merge the last two nodes into one super-node, accumulate their weights
      track the minimum candidate cut across all phases
    """
    n = W.shape[0]
    nodes = [{i} for i in range(n)]  # start with each node a singleton
    W = W.astype(np.float64).copy()
    best_cut = np.inf
    best_partition_b = None

    while len(nodes) > 1:
        # minimum-cut-phase
        n_left = len(nodes)
        A = [0]  # start anywhere
        in_A = np.zeros(n_left, dtype=bool); in_A[0] = True
        weights_to_A = W[:, 0].copy()
        weights_to_A[0] = -1
        last, second_last = 0, 0
        for _ in range(n_left - 1):
            # pick the not-yet-in-A node most tightly connected to A
            cand = int(np.argmax(np.where(in_A, -1, weights_to_A)))
            second_last = last
            last = cand
            in_A[cand] = True
            A.append(cand)
            # update weights — add cand's edges
            weights_to_A = np.where(in_A, -1, weights_to_A + W[:, cand])

        # cut-of-the-phase = sum of edges from `last` to all others
        cut_val = float((W[last, :].sum() - W[last, last]))
        if cut_val < best_cut:
            best_cut = cut_val
            best_partition_b = nodes[last].copy()

        # merge last + second_last
        merged = nodes[last] | nodes[second_last]
        # merge their rows/cols
        W[second_last, :] += W[last, :]
        W[:, second_last] += W[:, last]
        W[second_last, second_last] = 0
        # remove `last`
        keep = [i for i in range(n_left) if i != last]
        W = W[np.ix_(keep, keep)]
        nodes = [merged if i == second_last else nodes[i] for i in keep]

    partition_b = sorted(best_partition_b) if best_partition_b else []
    return best_cut, partition_b


def run_scenario(base: np.ndarray, replay_window: np.ndarray, mode: str, n_honest: int = 4):
    """Run one adversarial scenario, return diagnostic info."""
    honest = synth_honest_nodes(base, n_nodes=n_honest, noise_db=6.0)
    adv = synth_adversarial(base, mode=mode, replay_window=replay_window)
    windows = np.concatenate([honest, adv[None, ...]], axis=0)  # [n_honest + 1, 56, 20]
    adv_idx = n_honest  # last node is the adversarial one

    sim = cosine_sim_matrix(windows)
    # Convert similarity → edge weight. Mincut on similarity finds the
    # minimum-similarity partition, which is the *most-suspicious* split.
    # Use (1 - sim) as the weight if we want to minimise dissimilarity, but
    # the natural framing is: mincut over similarity-weighted graph isolates
    # the node least-similar to the rest.
    np.fill_diagonal(sim, 0.0)

    cut_val, partition_b = stoer_wagner_mincut(sim)
    detected = (set(partition_b) == {adv_idx}) or (set(range(len(windows))) - set(partition_b) == {adv_idx})

    return {
        "mode": mode,
        "n_honest": n_honest,
        "adv_idx": adv_idx,
        "sim_matrix": sim.round(4).tolist(),
        "mincut_value": float(cut_val),
        "partition_b": partition_b,
        "adv_isolated": bool(detected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", required=True)
    parser.add_argument("--out", default="examples/research-sota/r7_multilink_consistency_results.json")
    args = parser.parse_args()

    base = load_one_window(Path(args.paired), idx=10)
    stale = load_one_window(Path(args.paired), idx=900)
    if base is None or stale is None:
        raise SystemExit("need at least 901 samples in the paired file")

    results = {}
    for mode in ["replay", "shift", "noise"]:
        scenario = run_scenario(base, stale, mode=mode, n_honest=4)
        results[mode] = scenario
        print(f"\n=== adversarial mode: {mode} ===")
        print(f"  mincut value: {scenario['mincut_value']:.4f}")
        print(f"  partition B (less-similar side): {scenario['partition_b']}")
        print(f"  adversarial node isolated? {'YES' if scenario['adv_isolated'] else 'no'}")

    n_detected = sum(1 for r in results.values() if r["adv_isolated"])
    summary = {
        "n_scenarios": len(results),
        "n_detected": n_detected,
        "detection_rate": n_detected / len(results),
    }
    print(f"\n=== summary ===")
    print(f"  detection rate: {n_detected}/{len(results)} = {summary['detection_rate']:.0%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "scenarios": results}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
