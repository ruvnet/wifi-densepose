# ADR-084: RaBitQ Similarity Sensor for CSI / Pose / Memory Routing

| Field          | Value                                                                                   |
|----------------|-----------------------------------------------------------------------------------------|
| **Status**     | Accepted — Passes 1–5 + L1–L4 hardening implemented and merged via PR #435 (commit `d71ef9a`); acceptance numbers in §"Acceptance test" all measured and passing on synthetic AETHER-shape data; the `< 1 pp end-to-end accuracy regression` criterion is tracked as a post-merge soak test |
| **Date**       | 2026-04-26                                                                              |
| **Authors**    | ruv                                                                                     |
| **Refines**    | ADR-024 (AETHER re-ID embeddings), ADR-027 (cross-environment domain generalization), ADR-076 (CSI spectrogram embeddings), ADR-081 (5-layer firmware kernel) |
| **Companion**  | ADR-083 (per-cluster Pi compute hop)                                                    |
| **Implements** | `vendor/ruvector/crates/ruvector-core/src/quantization.rs::BinaryQuantized`             |

## Context

RuView's signal pipeline already produces several **dense float
embeddings** at different layers:

- AETHER 128-d re-ID embeddings on each `PoseTrack` (ADR-024)
- 64–256-d CSI spectrogram embeddings (ADR-076)
- per-room field-model eigenmode vectors (ADR-030)
- per-frame multistatic fused vectors (ADR-029)

Every one of these eventually answers the same shape of question:
**"have I seen something like this before?"** Today the answer is
computed by full float dot-product / Mahalanobis comparisons against a
candidate set. That cost grows linearly with stored vectors and
quadratically when used inside dynamic-mincut graph maintenance,
re-identification re-scoring, and cross-environment domain detection.

The vendored `ruvector-core` crate already ships a 1-bit quantization
(`BinaryQuantized`, 32× compression, SIMD popcnt + hamming distance)
that is functionally equivalent to the **RaBitQ** family of binary
sketches: a vector is reduced to one bit per dimension, compared via
hamming distance, and used as a coarse pre-filter before full
precision refinement. The same module also exposes `ScalarQuantized`
(int8, 4×) and `ProductQuantized` (PQ, 8–16×), so the tiered
quantization story is already implemented; the *deployment pattern* is
not.

The user observation that motivates this ADR: **RaBitQ-style sketches
are not just a vector compression trick — they are a cheap similarity
sensor.** Used as a sensor, they unlock:

- always-on novelty / anomaly gating that wakes heavy CNNs only on
  meaningful change
- cluster-Pi memory routing (which shard / room / model to query first)
- cross-node mesh exchange of compressed sketches instead of raw vectors
- privacy-preserving event logs (sketches, not reconstructable signals)

This ADR formalizes the deployment pattern across the RuView stack and
commits to `ruvector::quantization::BinaryQuantized` as the canonical
implementation.

## Decision

Adopt **RaBitQ-style binary sketches as a first-class, cheap
similarity sensor** at four points in the RuView pipeline:

1. **CSI / pose embedding hot-cache filter** at the cluster Pi.
2. **Drift / novelty sensor** between live observation and a
   per-room normal-state bank.
3. **Mesh-exchange compression** between sensor nodes when reporting
   cross-cluster events.
4. **Privacy-preserving event log** at the cluster Pi and gateway.

The canonical pattern at every point is:

```text
dense embedding  ──►  RaBitQ sketch  ──►  hamming/popcnt compare
                                       ├──►  candidate set (top-K)
                                       └──►  novelty score (0..1)
                                              │
                                              ▼
                          ┌── below threshold ──►  emit summary, no escalation
                          │
                          └── above threshold ──►  full-precision refinement
                                                     ├──►  ruvector mincut / HNSW
                                                     ├──►  AETHER re-ID rescoring
                                                     └──►  pose model / CNN wake
```

### Implementation home

- **Sketch type and SIMD primitives**:
  `vendor/ruvector/crates/ruvector-core/src/quantization.rs::BinaryQuantized`
  — already implemented, already SIMD-accelerated (NEON on aarch64,
  POPCNT on x86_64). Re-export through a new
  `crates/wifi-densepose-ruvector/src/sketch.rs` module so consumers in
  `signal`, `train`, `mat`, and `sensing-server` see a stable
  RuView-flavored API and don't bind directly to the vendor crate.

- **Per-room normal-state bank**: lives at the cluster Pi (ADR-083),
  not on the sensor MCU. Sensor MCUs continue to emit dense embeddings
  in the existing `rv_feature_state_t` packet shape; sketching happens
  on the Pi where the candidate bank is.

- **Sketch versioning**: each sketch carries a 16-bit `sketch_version`
  field so the Pi can tell incompatible sketches apart when an
  embedding model upgrades. Bumped on every embedding-model change.

### Where the sensor sits in the pipeline

| Pipeline stage | Today (full float) | With RaBitQ similarity sensor |
|---|---|---|
| AETHER re-ID match | full 128-d cosine on every active track × candidate | hamming pre-filter to top-K, then full cosine on K |
| Mincut subcarrier selection | full graph re-evaluation | sketch-flagged "likely-changed" boundary edges, full mincut on those |
| CSI room fingerprint | trained classifier on full embedding | sketch hamming to per-room sketch, classifier on miss |
| Field-model novelty (ADR-030) | residual-energy threshold | sketch novelty as second gate before SVD redo |
| Mesh / inter-cluster sync | dense embedding broadcast | sketch broadcast; full vector only on miss |
| Event log retention | full embedding stored | sketch + witness hash stored; raw embedding ephemeral |

In every row, the **decision boundary is unchanged** — full precision
still owns the final answer. The sketch is a sensor that only gates
which comparisons run, not what they decide.

### Acceptance criterion (per the source proposal)

The system-level acceptance test is:

> RaBitQ should reduce compare cost by **8× to 30×** while preserving
> top-k decisions well enough that full refinement changes **fewer
> than 10%** of final results.

Concretely, this means:

- Sketch compare must be measurably **8× cheaper** than the float
  comparison it replaces (criterion-bench in `signal/`).
- Top-K candidate set chosen by sketch must contain ≥ 90% of the
  candidates the full-float pass would have picked (offline replay
  against recorded CSI).
- End-to-end pose / re-ID accuracy must regress by **less than 1
  percentage point** vs the full-float baseline on the existing
  evaluation set.

If any of these three fail, the sensor is rolled back at that point in
the pipeline and the failing site reverts to full float; the rest of
the pipeline keeps using sketches. This is point-by-point, not
all-or-nothing.

## Consequences

### Positive

- **Cheaper hot path everywhere a "have I seen this" question lives.**
  AETHER re-ID, mincut maintenance, room fingerprinting, novelty
  detection, mesh sync, and event-log retention all run a 32×-smaller,
  popcnt-friendly comparison first.
- **Always-on anomaly gating becomes affordable.** The CNN / pose
  model only wakes when sketch novelty crosses a threshold. Energy
  budget per node drops materially in steady-state quiet rooms.
- **Privacy story improves.** Event logs and inter-cluster mesh
  traffic carry sketches and witness hashes, not reconstructable
  embeddings. The 1-bit quantization is *not* invertible to the
  original CSI.
- **Composes cleanly with ADR-083.** The cluster Pi is the natural
  home for the sketch bank; sensor MCUs remain unchanged.
- **No new dependency.** `BinaryQuantized` is already in the vendored
  `ruvector-core` and already SIMD-accelerated.

### Negative / risks

- **Sketch quality depends on embedding distribution.** Pure 1-bit
  sign quantization (which `BinaryQuantized` implements) works best
  when the embedding space is roughly zero-centered and isotropic.
  AETHER and CSI spectrogram embeddings need to be benchmarked for
  this assumption; if either fails, a randomized rotation
  (Johnson-Lindenstrauss / RaBitQ-paper-style) must be added before
  sketching. Out-of-scope for this ADR; tracked as a follow-up if
  the acceptance test fails.
- **Top-K coverage degrades for small candidate sets.** With < 16
  candidates, the sketch compare can pick the wrong K. Site-by-site
  fallback to full float is part of the rollout plan.
- **Sketch-version skew during model upgrades.** A model change
  invalidates all stored sketches; the cluster Pi must re-sketch the
  candidate bank when `sketch_version` bumps. Cost is bounded but
  non-zero.

### Neutral

- ADR-024, ADR-027, ADR-029, ADR-030, ADR-076 are unchanged in
  *what* they compute. They gain a sketch pre-filter at the comparison
  step.
- ADR-082's confirmed-track output filter is upstream of the sketch
  layer; it stays correct.

## Implementation

The implementation lands in five passes, each independently testable.
Every pass is gated by the acceptance criterion above; if any fail,
that site rolls back and the rest continue.

1. **`wifi-densepose-ruvector::sketch` module.** Re-export
   `BinaryQuantized` plus a thin RuView-flavored API
   (`Sketch::from_embedding`, `Sketch::distance`, `SketchBank::topk`).
   Add `sketch_version: u16` and `embedding_dim: u16` fields to the
   public type. Criterion benches: sketch ↔ float compare-cost ratio.

2. **AETHER re-ID pre-filter.** In
   `wifi-densepose-signal/src/ruvsense/pose_tracker.rs`, before
   computing the full 128-d cosine across active tracks × candidates,
   sketch both sides and reduce to top-K via hamming. Bench: re-ID
   pass time per frame, ID-stability under cross-room transitions.

3. **Cluster-Pi novelty sensor.** In
   `wifi-densepose-sensing-server`, maintain a per-room
   `SketchBank` of "normal-state" sketches; on each incoming
   `rv_feature_state_t`, compute embedding sketch, score novelty
   against the bank, and emit `novelty_score` as a new field on the
   WebSocket update envelope. Heavy CNN wake gate uses this score.

4. **Mesh-exchange compression.** Inter-cluster broadcasts (the
   ADR-066 swarm-bridge channel) carry sketch + witness instead of
   the full embedding when novelty is low. Full embedding only
   exchanged when novelty crosses threshold.

5. **Privacy-preserving event log.** Event log table on the cluster
   Pi stores `(sketch_bytes, sketch_version, novelty_score,
   witness_sha256)` instead of raw embeddings. Existing log readers
   are unchanged in API; only the storage layer rewrites.

Each pass adds tests: a property test (sketch ↔ float top-K agreement
≥ 90%), a criterion bench (≥ 8× compare cost reduction), and an
end-to-end accuracy regression test (< 1 pp drop).

## Validation

This ADR is **proposed**, not accepted. Acceptance requires the three
acceptance numbers above to hold on **at least three of the five
implementation passes** (the sites where the bulk of the load sits:
AETHER re-ID, cluster-Pi novelty, and event log). The mesh-exchange
and mincut prefilter passes are nice-to-haves; they can ship
afterward if their per-site numbers hold.

Validation runs against:

- the existing 1,539-test workspace suite (must stay green)
- a new `tests/integration/rabitq_sketch_pipeline.rs` integration test
  driving recorded CSI through the full pipeline with and without
  sketches, comparing top-K decisions and end-to-end pose accuracy
- ESP32-S3 on COM7 — sensor MCU unchanged; sketch happens at the
  cluster Pi, so this validation is a smoke test that the
  sensor → Pi UDP path still works after the cluster Pi gains the
  sketch bank

## Related

- **ADR-024** (Accepted) — AETHER re-ID embeddings. Primary consumer
  of the sketch pre-filter.
- **ADR-027** (Accepted) — Cross-environment domain generalization
  (MERIDIAN). Per-room sketch bank is the natural data structure for
  domain detection.
- **ADR-030** (Proposed) — RuvSense persistent field model. Sketch
  novelty is the cheap second gate before SVD recompute.
- **ADR-066** — Swarm bridge to coordinator. Inter-cluster sketch
  exchange.
- **ADR-076** (Accepted) — CSI spectrogram embeddings. Sketch
  consumer; embedding source.
- **ADR-081** (Accepted) — 5-layer adaptive CSI mesh firmware kernel.
  Sensor MCU unchanged by this ADR; sketches happen at the cluster Pi.
- **ADR-083** (Proposed) — Per-cluster Pi compute hop. Defines the
  device class that hosts the sketch bank.

## Pass 2 — randomized rotation + multi-bit (ADR-156 §8, landed 2026-06)

The "Open question" below ("does `BinaryQuantized` need a randomized
rotation pre-pass?") is now **answered with measured numbers** via
ADR-156 §10. Summary:

- **Pass 2 (randomized rotation) is implemented** —
  `crates/wifi-densepose-ruvector/src/rotation.rs`: a deterministic
  `R = H·D` (Fast Hadamard Transform + seeded ±1 sign flips), `O(d log d)`
  / `O(d)`, norm-preserving, reproducible from a stored `u64` seed. Opt-in
  via `Sketch::from_embedding_rotated` / `SketchBank::with_rotation`;
  Pass-1 API and wire format unchanged.
- **Measured top-K coverage** (anisotropic planted-cluster fixture,
  cosine ground truth, dim=128 N=2048 K=8): rotation lifts coverage
  **36.13% → 46.39%** at the strict `candidate_k = K` bar, and Pass-2
  reaches the **≥90% acceptance bar at candidate_k = 24 (~3× over-fetch)**.
  Multi-bit (≤4-bit) reaches 74% at the strict bar. **Honest verdict:
  neither rotation nor ≤4-bit multi-bit clears the strict-K 90% bar on
  this distribution; the bar is met via the over-fetch "candidate set"
  pattern this ADR specifies** (Decision §"the canonical pattern" — sketch
  picks the candidate set, full precision refines). Full numbers and
  reproduce commands in ADR-156 §10.
- **Pre-existing `SketchBank::topk` bug fixed** — the `n > k` heap path
  returned the k *farthest* sketches (min-heap mistaken for max-heap);
  only the `n ≤ k` fast path had test coverage. Fixed + regression-pinned
  (`topk_heap_path_returns_nearest`,
  `tight_clusters_give_high_coverage_with_overfetch`). This makes every
  prior top-K acceptance number in this ADR depend on the fixed path; the
  ≥90% coverage criterion is only meaningful post-fix.

## Pass 2b — RaBitQ unbiased distance estimator (ADR-156 §11, landed 2026-06)

The **real** RaBitQ contribution (Gao & Long, SIGMOD 2024) — an
**unbiased estimator of the inner product / distance** from the 1-bit
code + per-vector side info, not just sign bits — is now implemented and
**MEASURED against this ADR's ≥90% strict-K bar**:

- **Implemented** — `crates/wifi-densepose-ruvector/src/estimator.rs`:
  `EstimatorSketch` (Pass-2 sign code + 8 B/vec side info:
  `residual_norm` + `x_dot_o = ⟨x̄, o'⟩`), `DistanceEstimator`
  (`⟨o',q'⟩ ≈ ⟨x̄,q'⟩ / x_dot_o`, the paper's unbiased rescale), and
  `EstimatorBank` reranking candidates by the estimate instead of raw
  Hamming. **Zero-centroid simplification** (`c = 0`) documented;
  paper-faithful centroid path also built (`with_centroid`). Additive —
  Pass-1/Pass-2 and the wire format are unchanged.
- **MEASURED strict-K coverage** (same fixture as §"Pass 2", cosine
  ground truth): the estimator lifts the strict `candidate_k = K` bar
  **46.39% (Pass-2 sign) → 49.71% (estimator, cosine rerank)** — a real
  **+3.3 pp** lift, but **still ~40 pp short of the ≥90% strict bar.**
  At over-fetch the estimator does better than sign (95.12% vs 91.60% at
  candidate_k = 24). **Honest verdict: the unbiased estimator does NOT
  clear the strict-K 90% bar on this distribution** — the binding
  constraint is the 1-bit code's information ceiling, not estimator
  variance. The ≥90% acceptance bar is still met only via the over-fetch
  "candidate set" pattern this ADR's Decision specifies; the estimator
  **reduces the over-fetch factor** needed but does not remove it. This
  is a **published negative**, reported as such. Full numbers + reproduce
  commands in ADR-156 §11.

## Open questions

- **Does `BinaryQuantized` need a randomized rotation pre-pass for
  RuView's embedding distributions?** **ANSWERED (ADR-156 §10):** rotation
  is built and measured — it helps (+10pp at strict K) but is not
  sufficient alone for strict-K 90% on the tested anisotropic
  distribution; the over-fetch candidate-set pattern meets the bar.
  Pure sign quantization assumes zero-centered, isotropic embeddings; the
  rotation decorrelates anisotropic coords as the RaBitQ paper
  (Gao & Long, SIGMOD 2024) prescribes.
- **Sketch dimension target.** Default to the embedding's native
  dimension (128 for AETHER, 256 for spectrogram). Higher-dimensional
  sketches (Johnson-Lindenstrauss-projected to 512) trade compute for
  recall; benchmark before committing.
- **Per-room vs per-deployment sketch banks.** Defaulting to per-room
  for novelty detection. Cross-room re-ID may want a shared bank;
  decide once cross-room AETHER traces are available.
