# ADR-103: Learned Multi-Person Counter (SOTA WiFi CSI counting)

- **Status:** Proposed
- **Date:** 2026-05-21
- **Deciders:** ruv
- **Motivating issue:** #499 (double skeletons with 3-node ESP32-S3 setup, closed by PR #491)
- **Related:** ADR-079 (camera-supervised training), ADR-100 (cog packaging), ADR-101 (pose cog), ADR-102 (edge module registry), PR #491 (RollingP95 + dedup_factor)

## Context

PR #491 stopped the bleeding on #499. The fix replaced hard-coded denominators (`variance/300`, `motion_band_power/250`, `spectral_power/500`) with a self-calibrating `RollingP95` streaming estimator and exposed the multi-node `dedup_factor` as a runtime knob. Day-0 deployments no longer collapse dynamic range, and operators can auto-tune the divisor from a known person count.

That gets us to a **stable heuristic that adapts to the room**. It does not get us to the published WiFi-CSI counting state of the art:

| System | Setup | Reported accuracy | Method |
|--------|-------|-------------------|--------|
| **WiCount** (CMU, 2017) | Intel 5300 3×3 MIMO | 89% within ±1 | LSTM over CSI amplitude |
| **DeepCount** (2018) | Atheros 3×3 | 92% within ±1, 5-room | CNN + cross-environment transfer |
| **CrossCount** (2019) | Atheros, 6 rooms | 84% cross-room within ±1 | Domain-adversarial CNN |
| **HeadCount** (2021) | Intel 5300 | <1 person MAE, 5 envs | Multi-stream CSI + attention |
| **RuView today** (PR #491) | ESP32-S3 1×1 SISO | Calibrated heuristic; not measured against ground truth | RollingP95 + dedup_factor |

The literature uses 3×3 MIMO research NICs. RuView uses 1×1 SISO ESP32-S3 nodes. The published number is therefore not directly attainable, but the **architectural gap** is large enough that a learned-counter approach on our hardware should comfortably beat today's slot heuristic — and the infrastructure to train one already exists in this repo (Candle + RTX 5080 trained `pose_v1.safetensors` in 2.1 s yesterday — see [`docs/benchmarks/pose-estimation-cog.md`](../benchmarks/pose-estimation-cog.md)).

Five primitives we already have but don't yet compose into a counter:

1. **Paired CSI + camera label dataset** — `scripts/collect-ground-truth.py` + `scripts/align-ground-truth.js` (PR #641 streaming-safe). 1,077 samples currently; #645 tracks the path to ~30K.
2. **Stoer-Wagner min-cut for person-separable subcarrier groups** — `ruvector-mincut` (already a workspace dep). The Candle trainer used it yesterday and reported `Min-cut value: 0.1538 — partition: [55, 1] subcarriers`.
3. **Contrastive-pretrained CSI encoder** — `ruvnet/wifi-densepose-pretrained` on HF (12.2M training steps, 60K frames, 128-dim embeddings, ~165k emb/s on M4 Pro).
4. **Candle training pipeline** — proven yesterday: 400 epochs in 2.1 s on RTX 5080, bit-perfect ONNX export, signed cog binary on GCS.
5. **Multi-node fusion stage** — `multistatic_bridge.rs` already aggregates per-node feature vectors with the tunable `dedup_factor`. The new model output can be a drop-in replacement for the existing dedup divisor.

## Decision

Train and ship a small **learned multi-person counter** as a new Cognitum Cog (`cog-person-count`), modelled on the same packaging path as `cog-pose-estimation` (ADR-101). Wire it into the sensing-server's existing person-count call site (`csi.rs::score_to_person_count`) as a drop-in replacement for the slot heuristic.

### Architecture (v0.1.0)

```
                              ┌──────────────────────────────┐
       per-node CSI window    │  Encoder (frozen first 50 ep) │
       [56 sub × 20 frames]  ─►  init from ruvnet/wifi-       │
                              │  densepose-pretrained         │
                              │  → 128-dim embedding          │
                              └──────────────┬───────────────┘
                                             │
                            ┌────────────────┴────────────────┐
                            ▼                                 ▼
                   ┌────────────────────┐       ┌────────────────────────┐
                   │  Count head        │       │  Confidence head       │
                   │  Linear(128→64)    │       │  Linear(128→32)        │
                   │  ReLU              │       │  ReLU                  │
                   │  Linear(64→8)      │       │  Linear(32→1) + sigmoid│
                   │  → softmax over    │       │  → calibrated p(correct)│
                   │     {0..7} persons │       └────────────────────────┘
                   └────────┬───────────┘
                            │                    (per-node prediction)
                            │
       N nodes' per-node    │
       counts + confidences ▼
                   ┌─────────────────────────────────────┐
                   │  Multi-node fusion (Stoer-Wagner)   │
                   │  • build graph: nodes × subcarrier  │
                   │    feature similarity               │
                   │  • min-cut → distinct-person bound  │
                   │  • combine with per-node count head │
                   │    via confidence-weighted vote     │
                   └──────────────────┬──────────────────┘
                                      ▼
                          { count: int,
                            confidence: float [0,1],
                            count_p95_low: int,
                            count_p95_high: int,
                            per_node_breakdown: [...] }
```

Five things to call out about this architecture:

1. **Frozen encoder for the first 50 epochs.** The HF presence encoder already produces a useful 128-dim embedding from random CSI; training the counting head on top of frozen features is the standard transfer-learning pattern and avoids re-learning the contrastive geometry the encoder was painstakingly trained for.
2. **Classification over `{0..7}` people**, not regression to a real number. Counts are integer-valued; classification gives a calibrated probability per count and lets the confidence head produce a meaningful uncertainty.
3. **Stoer-Wagner min-cut at fusion time, not training time.** We use the min-cut primitive to bound the per-node count from above (a node can't see more distinct people than the subcarrier graph has min-cuts), then take a confidence-weighted vote.
4. **Output is `{count, confidence, count_p95_low, count_p95_high}`**, not a single integer. Downstream consumers (Cogs / dashboard / alerts) can choose their certainty threshold. This is what closes the loop on the #499 UX: when the model is uncertain, the dashboard renders one stick figure with a "?" badge rather than two ghosts.
5. **No new hardware.** Same ESP32-S3 1×1 SISO that ships today. The win comes from learned features + multi-node fusion, not from bigger antennas.

### Training (Candle / RTX 5080 / proven path)

Same exact pipeline that produced `pose_v1.safetensors` yesterday. Differences:

| | Pose cog (today) | Count cog (this ADR) |
|---|---|---|
| Input | `[56, 20]` CSI window | `[56, 20]` CSI window (identical) |
| Encoder init | random (HF arch mismatch) | **from HF presence model** (architectures are compatible — same encoder Φ) |
| Output head | `Linear(128 → 256 → 34)` keypoints | `Linear(128 → 64 → 8)` count classes + `Linear(128 → 32 → 1)` confidence |
| Loss | Confidence-weighted SmoothL1 | Categorical cross-entropy + Brier-score uncertainty calibration |
| Labels | MediaPipe keypoints | Camera count (MediaPipe `pose_landmarks` length) |
| Data | 1,077 paired (P7) | **Same source, same script** — `collect-ground-truth.py` already records `n_persons` per frame |

Crucially we get the count labels **for free** from the existing pose data-collection pipeline — `collect-ground-truth.py` already records `"n_persons"` per camera frame and `align-ground-truth.js` already preserves it through windowing. No new data collection campaign required to bootstrap; we can train tomorrow on the same 1,077 samples that produced `pose_v1`.

### Multi-node fusion

The per-node count head + confidence head emit a categorical distribution over `{0..7}`. With N nodes, we have N such distributions plus N confidence scalars. Two fusion paths:

- **Confidence-weighted log-sum** (Bayesian product): `log p_fused(k) = Σ_n c_n · log p_n(k)`. Simple, no extra parameters, comes from the optimal-expert combination literature.
- **Stoer-Wagner upper bound**: build a graph where edges are pairwise subcarrier-feature similarities between nodes. Min-cut size = a hard upper bound on the number of distinct people the node mesh can resolve. Clip the per-node-fused distribution to support `{0..min-cut}` before re-normalising. This is exactly what `ruvector-mincut` was added to the workspace for — it's been waiting for a counting consumer.

Both fuse cleanly. v0.1.0 ships the log-sum; v0.2.0 adds the min-cut clipper after the first round of evaluation.

### Why this beats today's heuristic

| Failure mode of today's slot heuristic | How the learned counter avoids it |
|---|---|
| #499 — fixed denominators clamp → one person renders as 2+ groups | Encoder produces a fixed-dim embedding; the count head is invariant to feature magnitude, only to feature **shape** |
| `dedup_factor` per-room tuning is operator-visible toil | Count head's softmax is a learned per-room normaliser by construction |
| Adding nodes makes the count noisier under the slot heuristic | Multi-node fusion is **additive in confidence**, so each node either reduces uncertainty or stays neutral — never amplifies it |
| No per-frame uncertainty signal | `confidence` + `count_p95_low/high` exposed in every emit |
| Catastrophic failure on novel environments | LoRA per-room adapter (per ADR-079 P9 plan) hot-swappable without retraining |

### Acceptance gates

| Gate | v0.1.0 (initial release) | v0.2.0 (after data scaling) |
|------|--------------------------|------------------------------|
| Day-0 deployment (no calibration) | ≥ 80% within ±1 on same-room test set | ≥ 90% within ±1 |
| Cross-room (held-out environment) | ≥ 60% within ±1 | ≥ 75% within ±1 |
| Mean Absolute Error | ≤ 0.6 persons | ≤ 0.4 persons |
| Per-frame confidence reflects accuracy | Spearman correlation `r ≥ 0.5` between `confidence` and `(predicted == true)` | `r ≥ 0.7` |
| Inference latency on Pi 5 (Cog) | < 5 ms / frame cold-start | < 5 ms / frame |
| Binary size on GCS | ≤ 4 MB (matches `cog-pose-estimation`) | ≤ 4 MB |

`v0.1.0` is intentionally modest — it's bounded by data-collection scale (#645). The framework is the deliverable; the accuracy follows the data.

### Repo layout

```
v2/crates/cog-person-count/                   # NEW (this ADR)
├── Cargo.toml
├── src/
│   ├── main.rs                # cog runtime: version | manifest | health | run
│   ├── lib.rs
│   ├── inference.rs           # Candle forward pass on per-node CSI
│   ├── fusion.rs              # Stoer-Wagner upper-bound + confidence-weighted log-sum
│   └── publisher.rs           # emits {count, confidence, count_p95_low, count_p95_high}
├── cog/
│   ├── manifest.template.json
│   ├── config.schema.json
│   ├── README.md
│   └── artifacts/             # filled by the release pipeline
│       ├── count_v1.safetensors
│       ├── count_v1.onnx
│       └── train_results.json
└── tests/
    ├── smoke.rs               # 5+ tests
    └── fusion_test.rs         # multi-node-fusion math
```

Plus a small server-side wiring change:

- `v2/crates/wifi-densepose-sensing-server/src/csi.rs::score_to_person_count` — call the cog over the same `/api/v1/edge/registry`-discovered runtime as `cog-pose-estimation`. Falls back to today's PR #491 heuristic if the cog isn't installed (per the ADR-100 stub-fallback pattern).

## Consequences

### Positive

- Closes the conceptual loop opened by #499 — multi-person counting becomes a **learned task**, not a heuristic with a runtime knob.
- Reuses every primitive already shipped this week: Candle GPU training (ADR-101), HF encoder, Cog packaging (ADR-100), edge module registry (ADR-102), Stoer-Wagner mincut, paired-data pipeline (PR #641).
- Day-2 cross-room calibration uses the same LoRA path ADR-079 P9 plans for pose, so the two cogs share the same fine-tuning machinery.
- Explicit `confidence` + `count_p95_low/high` outputs let the UI render uncertainty instead of inventing ghosts.

### Negative

- Accuracy is bounded by the same paired-data scarcity that bounds `pose_v1` (#645). Without more multi-room data, v0.1.0 ships with modest absolute accuracy.
- Adds another Cog binary to maintain in the GCS catalog — 4 MB per arch.
- The fusion-stage min-cut adds ~0.3 ms per N-node frame on a Pi 5 in microbenchmarks of `ruvector-mincut`. Acceptable given the ≤ 5 ms budget but worth tracking.

### Risks

- **Label noise**: MediaPipe pose-detection rate was 47% in the P7 session — half the frames have `n_persons = 0` even when a person was clearly in the room. The count head learns from this noisy signal; mitigations include filtering by `MediaPipe confidence ≥ 0.7` before training, and weighting the loss by confidence (same trick used in `pose_v1`).
- **Encoder freezing too aggressive**: if 50 epochs of frozen-encoder training doesn't see the count head converge, unfreeze earlier. We have telemetry from `train_results.json` to make this call empirically.
- **Min-cut over-constrains** in single-person scenarios: when N=1 the subcarrier graph has min-cut = 1 trivially. The fusion stage degrades to "trust the single-node count head", which is fine but worth a regression test (`tests/fusion_test.rs::single_node_degrades_gracefully`).

## Migration

1. Land this ADR + the new crate scaffold (one PR, no model yet — same approach as ADR-101's first PR shipped a stub cog).
2. Train `count_v1.safetensors` on the existing 1,077 paired samples + `n_persons` labels. Same Candle pipeline that produced `pose_v1`.
3. Cross-compile + sign + GCS upload per ADR-100. Live install on `cognitum-v0` per ADR-101's pattern.
4. Wire `csi.rs::score_to_person_count` to call the cog when installed; keep PR #491's heuristic as fallback.
5. v0.2.0: re-train on the multi-room data #645 motivates, add LoRA per-room adapters per ADR-079 P9.

## See also

- ADR-079 — Camera-supervised training pipeline (same data path).
- ADR-100 — Cognitum Cog packaging spec (same shipping format).
- ADR-101 — Pose Estimation Cog (template for this Cog's first release).
- ADR-102 — Edge Module Registry (where this cog appears in the catalog).
- PR #491 — RollingP95 + `dedup_factor` (the heuristic this learned counter replaces).
- Issue #499 — Multi-node ghost skeletons (closed by #491, motivates this ADR).
- Issue #645 — PCK / data-collection plan (same data-bound limit; same fix path).
- `docs/benchmarks/pose-estimation-cog.md` — measured perf envelope for the cog runtime this ADR targets.
