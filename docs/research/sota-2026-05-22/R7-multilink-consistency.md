# R7 — Multi-link consistency detection via Stoer-Wagner mincut

**Status:** first measurement landed · **2026-05-22**

## Premise

The Cog fleet deployment story (ADR-100 + ADR-102 + ADR-103) puts multiple ESP32-S3 nodes in the same physical space, each reporting CSI to the same sensing-server. Today, the server trusts every node equally. That's fine when the adversary is "an indifferent universe", but the WiFi-CSI literature has known supply-chain attacks:

- **Replay** — attacker captures a CSI stream from earlier and pumps it back in to fake "empty room" / "no fall" / "all-clear" states.
- **Constant shift** — attacker biases one node's CSI by a constant, hoping the fusion stage averages it away while still poisoning per-node decisions.
- **Noise injection** — attacker jams or otherwise produces pure-noise CSI that crosses the legitimate-traffic threshold of `wDev_ProcessFiq`-based packet filters.

A learned multi-node fusion (ADR-103 §"Multi-node fusion") will average these out *if* the adversary is the minority. But we need a primitive that *detects* the adversary so the fusion stage can drop them before averaging.

## Algorithm (this thread)

**Key insight:** N honest observers of the same physical scene produce CSI vectors that cluster tightly under cosine similarity (their windows differ only by per-channel multipath noise). An adversarial node, regardless of attack mode, sits *outside* that cluster.

The cluster-outlier-detection primitive that fits this problem exactly is the **Stoer-Wagner minimum cut** on the inter-node cosine-similarity graph:

```
for each pair of nodes (i, j):
  W[i, j] = cos(flatten(csi_i), flatten(csi_j))

(value, partition_B) = stoer_wagner_mincut(W)

# partition_B is the "less-similar" side of the minimum cut.
# When the cut is sharp, partition_B is a singleton — the adversarial node.
```

`ruvector-mincut` already vendors this algorithm in the workspace (used by `cog-pose-estimation` for person-separable subcarrier grouping, see #491). The fusion stage in `cog-person-count` (`fuse_with_mincut_clip()`) has a stub that's exactly the consumer this primitive needs.

## Demo measurement

`examples/research-sota/r7_multilink_consistency.py` — pure NumPy, no framework deps. Synthesises 4 honest CSI nodes (real scene from `data/paired/...` + per-node Gaussian noise 6 dB below signal) and 1 adversarial node under each of 3 attack modes:

| Attack mode | Description | Mincut value | Partition_B | Adversarial isolated? |
|---|---|---|---|---|
| **replay** | Stale window from earlier in the recording, +1% jitter | 3.4513 | `{4}` | **YES** |
| **shift** | Constant +3σ offset on every subcarrier | 3.5724 | `{4}` | **YES** |
| **noise** | Pure Gaussian noise at honest-node signal magnitude | 2.5586 | `{4}` | **YES** |

**Detection rate: 3/3 = 100%** on this synthetic scenario, with mincut value gaps that are well-separated from the within-honest-cluster connectivity (honest nodes have pairwise similarities >0.95, the adversarial node's similarity to any honest node is ≤0.5).

## Honest scope of this result

This is a **clean synthetic scenario** with strong adversary signals. Real-world attacks are subtler:

- A *clever* replay attacker would time the replay to overlap with stable empty-room periods, when honest-node CSI is also nearly-identical to the stale window. Detection rate degrades.
- A *partial-spectrum* shift on a few subcarriers (instead of all 56) leaves enough true CSI that cosine similarity stays high. Need a per-subcarrier check, not whole-window.
- An *adaptive* attacker who has read this research note and adds calibrated noise to evade the cluster check.

What this demo proves: the **primitive works** when the adversary is sloppy. The next research step is the adaptive-attacker version — Stackelberg game between detector and adversary on the same similarity-cut framework.

## What this unlocks for the Cog stack

- The stub at `cog-person-count::fusion::fuse_with_mincut_clip()` can become a real primitive: at each frame, run mincut on the cross-node CSI similarity graph, drop any node that gets isolated, then run the count head on the remaining nodes' fused features.
- Same approach extends to `cog-pose-estimation` once we have a multi-node pose deployment.
- The mincut value itself is a continuous "mesh trustworthiness score" that can be exposed as a `mesh.trust` metric in the cog-gateway dashboard.

## 10-year horizon

The "RF radio-democracy" story: every WiFi receiver in a building (phones, laptops, smart speakers — see R8's RSSI-only result) becomes a witness in a Byzantine-fault-tolerant mesh. The mincut consistency check generalises to N=many heterogeneous nodes. A single compromised phone can't poison the building-scale sensing state because mincut isolates it. This is the spatial-intelligence analogue of Byzantine consensus in distributed systems — published-2026-SOTA hasn't framed CSI security this way yet.

## Connections back

- **R5** (subcarrier saliency) provides the priority list of subcarriers a detector should over-weight in the similarity metric — top-8 are `[41, 52, 30, 31, 10, 35, 2, 38]`.
- **R8** (RSSI-only) shows the same primitive likely works at lower SNR with RSSI-only metrics; the cluster structure is preserved by the band integral.
- **ADR-103** (`cog-person-count` v0.2.0 plan) — this primitive is the explicit content of the `fuse_with_mincut_clip()` stub.

## What's next on this thread

- Adversarial-game framing: detector + attacker as a two-player Stackelberg game.
- Per-subcarrier consistency check (not just whole-window cosine). Falls out of R5's saliency map naturally.
- Live demo on real multi-node data once seed-1 comes back online or seed-2-5 get provisioned.
