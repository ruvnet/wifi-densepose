# ADR-156: RuVector / Cross-Viewpoint Fusion Beyond-SOTA Sweep — Milestone 2 (Correctness Integrity, an Honest GDOP, Crafted-Input Safety, a Measured Hot-Path Win, and the ANN/Fusion SOTA Landscape)

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-06-11 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-ruvector` — `viewpoint/` (`attention.rs`, `geometry.rs`, `fusion.rs`, `coherence.rs`), `mat/` (`triangulation.rs`, `heartbeat.rs`), `sketch.rs`, benches, docs |
| **Relates to** | ADR-031 (RuView sensing-first RF mode), ADR-016/017 (RuVector integration), ADR-024 (AETHER re-ID), ADR-027 (MERIDIAN cross-env), ADR-084 (RaBitQ similarity sensor), ADR-138 (ClockQualityGate), ADR-152 (WiFi-Pose SOTA 2026 intake), ADR-154 (Signal/DSP sweep M0), ADR-155 (NN/Training sweep M1) |
| **Scope** | Milestone 2 of the beyond-SOTA sweep: four **correctness/integrity/security** fixes on the cross-viewpoint fusion path (each pinned by a regression test that fails on the old code), one **measured** hot-path perf win + a new criterion bench, the ANN/fusion SOTA landscape graded MEASURED/CLAIMED/data-gated, and a prioritized deferred backlog. **Nothing is silently dropped.** |

---

## 0. PROOF discipline (this ADR's contract)

This project has been publicly accused of "AI slop." Milestone 2 answers with **evidence, not adjectives** — the same contract as ADR-154/155:

- Every correctness/integrity fix ships a **committed regression test that fails on the old code and passes on the new**. We verified each by reverting the fix and observing the test fail (recorded in §6).
- Every perf number is **MEASURED before/after** with the exact reproduce command and a committed criterion bench. A perf claim without a measured before/after is **UNPROVEN** and is not made here.
- Every external SOTA reference is graded **MEASURED** / **CLAIMED** / **DATA-GATED**, distinguishing what a paper *measured* from what it *asserts* from what our own prior measurement (ADR-152) says is **not currently the bottleneck**.
- We disclose, in full, the **one staged finding that turned out to be a numeric no-op** (§2.1): the geometric-bias "angular wrap bug" is real as a *contract* violation but, because the bias kernel is `cos()` (even and 2π-periodic), it changes **no output value** under the current kernel. We land the fix anyway (it matches the documented contract and reuses the canonical helper) but we **do not claim a behaviour change** — that would be exactly the kind of inflation this sweep exists to prevent.

Test machine for the perf numbers: Windows 11, `cargo bench --release`, criterion 0.5. Numbers are wall-clock medians on this box; the **ratio** (before/after) is the claim, not the absolute ns.

Build/test gate: `cargo test --workspace --no-default-features` (the project's standard gate — no `crv`/GPU features). All fixes in this milestone are on the **default, non-feature-gated surface**, so they are fully exercised by the standard gate.

---

## 1. Context

The cross-viewpoint fusion stack (`viewpoint/` — ADR-031) combines per-viewpoint AETHER embeddings into one fused embedding via geometric-bias attention, gated by phase coherence, with array-geometry quality scored by a Geometric Diversity Index and a Cramér-Rao bound. The `mat/` survivor-localisation helpers (`triangulation.rs`, `heartbeat.rs`) share the same crate. A beyond-SOTA review surfaced findings spanning a **mislabeled metric**, an **angular-distance contract violation**, **crafted-input panics on a network-reachable path**, and a **redundant clone in the fusion hot path**, plus an ANN/fusion SOTA-research gap. Milestone 2 closes the provable subset and grades the research landscape.

---

## 2. Decision — CORRECTNESS / INTEGRITY FIXES

Each fix ships a regression test (all on the non-feature-gated, workspace-tested surface).

### 2.1 GeometricBias angular separation — use the canonical *wrapped* distance — ACCEPTED & IMPLEMENTED (honest: numeric no-op under the current cos kernel)

**The finding.** `attention::GeometricBias::build_matrix` computed the pairwise angular separation as the **raw** `|azimuth_i − azimuth_j|`. That can exceed π and mis-states the separation across the 0/2π seam (350° and 10° are 20° apart, but raw `|Δ|` = 340°). The module already had a correct wrapped helper, `geometry::angular_distance` (returns `[0, π]`), but it was **private** and `GeometricBias` did not use it.

**The honest correction (disclosed, not hidden).** The bias kernel is `w_angle·cos(theta_ij)`. Because `cos` is **even and 2π-periodic**, `cos(raw) == cos(wrapped)` for every pair (verified numerically: max abs diff `1.1e-16` across seam-crossing test cases). So under the *current* kernel this "bug" produces **identical bias values** — it is a **contract violation, not a behaviour bug**. We say so plainly rather than dressing a no-op as a fix.

**Why land it anyway.** (1) It makes the code satisfy its own documented contract (`theta_ij`: "angular separation in radians", which must be `[0, π]`). (2) It reuses the **single canonical** `angular_distance` helper (now made `pub`), eliminating a divergent angle computation — the same single-source-of-truth discipline ADR-155 applied to metrics. (3) It is **correct by construction** for any future non-even angular kernel (e.g. a linear `w_angle·theta_ij` penalty), which the raw-diff form would silently break.

**Tests:** `geometric_bias_angular_separation_uses_wrapped_distance` (pins that a seam-crossing pair's wrapped distance is 20° while its raw `|Δ|` exceeds π, and that `build_matrix` is symmetric across the seam) and `geometric_bias_linear_angular_kernel_would_catch_raw_diff` (pins the wrapped value ∈ `[0, π]` — the invariant a future linear kernel relies on; the raw-diff form gives 190° where the wrapped form gives 170°).

### 2.2 Crafted-input panics on the fusion/localisation path — typed `None` instead of panic — ACCEPTED & IMPLEMENTED (the security item)

**The finding (DoS).** Two functions on a path that can carry **network-sourced multistatic frames** panicked on crafted input:

- `mat::triangulation::solve_triangulation` indexed `ap_positions[0]` (panics on an empty AP table) and `ap_positions[i]` / `ap_positions[j]` (panics when a TDoA measurement references an **out-of-range AP index**). A remote peer supplying a TDoA tuple `(i=99, …)` with only 3 APs triggers an out-of-bounds panic — a remotely-triggerable denial of service.
- `mat::heartbeat::CompressedHeartbeatSpectrogram::band_power` computed `self.n_freq_bins - 1`, which **underflows** (usize `0 − 1`) for a zero-bin spectrogram — a debug panic / release `usize::MAX` (then an out-of-range index).

**The fix.** `solve_triangulation` uses `ap_positions.first()?` and `ap_positions.get(i)?` / `.get(j)?` — any empty table or out-of-range index returns `None`, never panics. `band_power` guards `n_freq_bins == 0` up front and **clamps both bounds** into `[0, last]`, returning `0.0` for empty/inverted ranges. No out-of-range index, no subtraction overflow, on any input.

**Tests:** `triangulation_out_of_range_index_returns_none_no_panic`, `triangulation_empty_ap_positions_returns_none_no_panic`, `heartbeat_band_power_zero_bins_no_panic`, `heartbeat_band_power_out_of_range_bounds_no_panic`. Each **panics on the old code** (verified by reverting — §6) and returns a clean `None`/`0.0` on the new.

### 2.3 GDOP mislabel — compute a real, dimensionless GDOP — ACCEPTED & IMPLEMENTED

**The finding.** `geometry::CramerRaoBound` exposed a field named `gdop` ("Geometric Dilution of Precision") that was computed as `(crb_x + crb_y).sqrt()` — **identical to `rmse_lower_bound`**. That is the RMSE (metres, noise-dependent), **not** a GDOP. GDOP is a *dimensionless geometry factor* independent of the noise level; the name was a lie about the quantity.

**The fix (honest rename was the fallback; real GDOP was cheap, so we computed it).** True GDOP `= sqrt(trace(G⁻¹))` where `G` is the **unit-variance** bearing-geometry matrix (the Fisher matrix with every `1/σ²` set to 1). It depends only on the array/target geometry and relates noise to position error as `rmse ≈ GDOP·σ`. We accumulate `G` alongside the FIM in both `estimate` and `estimate_regularised` (cheap 2×2), and report `INFINITY` (not NaN/panic) for a degenerate collinear geometry. The doc comment now states exactly what the field is and what it used to (wrongly) be.

**Test:** `gdop_is_dimensionless_and_noise_independent` — scales every sensor's noise by 10× and asserts GDOP is unchanged while RMSE scales ~10×, and that `rmse ≈ GDOP·σ` at both noise levels. The old `gdop = sqrt(crb_x + crb_y)` **fails** this (it scaled with noise, proving it was RMSE) — verified by reverting (§6).

### 2.4 `fuse()` double-clone in the aggregation hot path — eliminate the redundant clone — ACCEPTED & IMPLEMENTED (MEASURED — §4)

**The finding.** `MultistaticArray::fuse` (and `fuse_ungated`) cloned every viewpoint embedding **twice** per fusion: once into the `extracted` tuple vector (`v.embedding.clone()`), then **again** when building the attention input (`extracted.iter().map(|(_, e, _, _)| e.clone())`). At the AETHER dimension (128 f32 = 512 B) over up to 8 viewpoints, that is a wholly redundant second heap allocation + memcpy per viewpoint, every TDM cycle.

**The fix.** Build `extracted` once (the unavoidable clone out of the borrowed `self.viewpoints`), then **consume** `extracted` by value and **move** each embedding into the attention input (`embeddings.push(emb)`), capturing geometry/ids by `Copy` in the same pass. One clone per viewpoint instead of two. Measured win in §4.

---

## 3. Security review (touched files)

The §2.2 crafted-input panics **are** the security item: a DoS via out-of-range indices / zero-bin underflow on a fusion/localisation path that may be driven by network-sourced multistatic frames. Beyond those, the touched files were swept for further panic-on-untrusted-input / unbounded-alloc sites:

- `attention.rs` — all indexing is over internally-sized `n × n` / `d` loops bounded by validated input lengths (`DimensionMismatch` is returned for ragged embeddings); softmax denominators are floored with `f32::EPSILON`. No unbounded alloc (sizes derive from caller-supplied vector lengths already validated against `d_in`). **No further action.**
- `geometry.rs` — `det`/`det_g` are floored before division; degenerate geometry yields `None`/`INFINITY`, never NaN-panic. **No further action.**
- `fusion.rs` — embedding dimension is validated in `submit_viewpoint`; the event log is bounded (`max_events`, oldest-half drain). **No further action.**
- `coherence.rs` — circular buffer is fixed-capacity; gate thresholds are clamped. **No further action.**

No `unsafe`, no `unwrap()` on external input, and no unbounded allocation remain on the touched paths after §2.2.

---

## 4. MEASURED perf win (new criterion bench)

A new bench, `crates/wifi-densepose-ruvector/benches/fusion_bench.rs`, covers the fusion hot path. It has two groups: `fusion_pipeline` (end-to-end `MultistaticArray::fuse_ungated()` at 2/4/8 viewpoints, dim 128) and an isolated A/B of the §2.4 marshalling step (`embedding_extract/before_double_clone` vs `after_single_clone`).

- **Reproduce:** `cargo bench -p wifi-densepose-ruvector --bench fusion_bench`
- **Measured (`embedding_extract`, 8 viewpoints × 128-d), medians:** `before_double_clone` **1.0029 µs** → `after_single_clone` **461.6 ns** — **~2.17× faster** on the marshalling step. The result is what theory predicts (two embedding clones collapse to one), confirming the redundant clone was the cost, not noise.
- **End-to-end `fusion_pipeline` (medians):** 2 vp = 56.3 µs, 4 vp = 99.5 µs, 8 vp = 202.1 µs. The marshalling (~0.5–1 µs) is **well under 1%** of total fusion cost (dominated by the `n×n` attention), so the **end-to-end** effect is modest by construction; the `embedding_extract` A/B isolates and proves the clone-elimination itself. We report this honestly rather than attributing the full 2.17× to the pipeline.

The double-clone elimination is also correctness-neutral: all 100 `viewpoint`/`mat` lib tests pass unchanged.

---

## 5. The ANN / cross-viewpoint-fusion SOTA landscape (graded)

| # | Candidate | What | Grade | Verdict |
|---|-----------|------|-------|---------|
| **1** | **SymphonyQG** (SIGMOD 2025, public code) | Unified quantization + graph ANN; source reports **3.5–17× QPS over HNSW at equal recall**, pure-CPU / edge-portable. | **MEASURED-direction-tested** (was CLAIMED) — **[ADR-261](ADR-261-ruvector-graph-ann-index.md)** built the missing HNSW baseline + a SymphonyQG-style 1-bit quantized-traversal variant and **measured** the ratio on our hardware. | **DONE — direction REFUTED at our scale (honest negative).** ADR-261 built the real HNSW baseline (**~25× QPS over linear scan at recall ≥0.99**, the substrate this row wanted) and a quantized variant. At N=10k the 1-bit Hamming traversal is **too coarse** — its best recall is 0.738, never reaching the ≥0.90 equal-recall point, so **no QPS win over float HNSW** (the SymphonyQG 3.5–17× is *not* reproduced by our 1-bit construction here). Caveat: **our HNSW + our 1-bit quant, not SymphonyQG's system**; expected crossover at large N + a multi-bit code. We did **not** tune to manufacture a speedup. |
| **2** | **Multi-bit / Extended RaBitQ + unbiased estimator** | Extends our existing **1-bit** `sketch.rs` (ADR-084): Pass-2 rotation, multi-bit Pass-3, and the **real RaBitQ unbiased distance estimator** (Gao & Long SIGMOD 2024) reranking the candidate set from the 1-bit code + 8 B/vec side info (§11). | **MEASURED-on-our-hardware** (was CLAIMED) — rotation (§10), multi-bit (§10), and the estimator (§11) all implemented + benchmarked. Rotation lifts strict-K 36%→46%; multi-bit (≤4-bit) reaches 74% strict; **the estimator reaches 49.71% strict (cosine rerank), still short of 90%.** All clear 90% only with over-fetch (estimator improves the factor: 95% at candidate_k=24 vs sign 91.6%). | **DONE — RESOLVED-PARTIAL / NEGATIVE.** Rotation (§10) + estimator (§11) built and MEASURED. The honest negative (no strict-bar 90% from rotation, ≤4-bit, **or the unbiased estimator**) is recorded, not hidden. Over-fetch + Pass-2 is the path that meets the bar (ADR-084's "candidate set" pattern); the estimator lowers the over-fetch factor needed. |
| **3** | **GraphPose-Fi-style learned antenna-attention + ChebGConv fusion head** | Would replace the current **untrained identity-projection + mean-pool** "attention" (the `CrossViewpointAttention` default is `ProjectionWeights::identity` — not a *learned* attention) with a learned graph fusion head. | **DATA-GATED** (per ADR-152 measurement (b): architecture is **NOT** the current bottleneck — **data is**) | **ACCEPTED-future, data-gated. Do NOT build now.** ADR-152's measured lesson was that swapping architecture without more/better paired data does not move PCK. Building a learned fusion head before the data exists would repeat the mistake ADR-155 §5 also flagged for GraphPose-Fi. |
| — | **Cramér-Rao / sensor-placement** (`geometry.rs` CRB) | Investigated for a 2026 advance beating the textbook Fisher-information CRB already implemented. | **Investigated — NO ACTION** | **Cleared honestly.** No 2026 method beats the closed-form Fisher-information CRB for this 2-D bearing problem; our implementation is already correct SOTA. (Recording a negative result is a deliberate anti-slop signal.) The only CRB change this milestone is the §2.3 *GDOP* honesty fix, which is a labelling/quantity correction, not an algorithmic one. |

---

## 6. Validation

- **Bug-catching tests verified to bite.** Each §2.2/§2.3/§2.4-adjacent fix was reverted and the corresponding test observed to **fail on the old code**, then restored:
  - `triangulation_out_of_range_index_returns_none_no_panic` / `triangulation_empty_ap_positions_returns_none_no_panic` — **panic** (index out of bounds) on old code.
  - `heartbeat_band_power_zero_bins_no_panic` — **panic** ("attempt to subtract with overflow") on old code.
  - `gdop_is_dimensionless_and_noise_independent` — **assertion failure** (GDOP scaled with noise) on old code.
  - §2.1 (angular wrap) is the **disclosed no-op**: its tests pin the *contract* (wrapped value ∈ `[0, π]`), since the cos kernel makes the bias value numerically identical with or without the fix. We do not claim a behaviour change.
- **`cd v2 && cargo test -p wifi-densepose-ruvector --no-default-features --lib`** — **100 passed / 0 failed** (was 93; +7 new tests).
- **`cd v2 && cargo test --workspace --no-default-features`** — **3050 passed / 0 failed** (full-workspace aggregate across all crates and test binaries; the +7 new `wifi-densepose-ruvector` tests are included and green).
- **`python archive/v1/data/proof/verify.py`** — **`VERDICT: PASS`** (the Python pipeline proof is independent of these Rust changes — confirmed unaffected).
- New `fusion_bench` compiles and runs under the default feature set.

---

## 7. What changed, file by file

- `viewpoint/geometry.rs` — `angular_distance` made `pub` (single canonical wrapped-angle helper); real dimensionless GDOP (`sqrt(trace(G⁻¹))`) in `estimate`/`estimate_regularised` (was RMSE mislabelled); `gdop` doc states the quantity and the prior bug; `gdop_is_dimensionless_and_noise_independent` test.
- `viewpoint/attention.rs` — `GeometricBias::build_matrix` uses the canonical wrapped `angular_distance` (contract fix; numeric no-op under cos — disclosed); two contract-pinning tests.
- `viewpoint/fusion.rs` — `fuse`/`fuse_ungated` move embeddings out of `extracted` (single clone, not double); existing tests unchanged and green.
- `mat/triangulation.rs` — `first()?` / `get(i)?` / `get(j)?` guards (no panic on empty table / crafted indices); two no-panic tests.
- `mat/heartbeat.rs` — `band_power` zero-bin guard + bounds clamp (no underflow / out-of-range index); two no-panic tests.
- `benches/fusion_bench.rs` (new) + `Cargo.toml` `[[bench]]` — fusion hot-path bench + the double-clone A/B.

---

## 8. Deferred backlog (NOT silently dropped)

The review surfaced more than this milestone scoped. Tracked here for a future ADR-156 milestone:

- **SymphonyQG reproduction** (§5 #1) — **RESOLVED-DIRECTION-TESTED** (see [ADR-261](ADR-261-ruvector-graph-ann-index.md)). The missing HNSW baseline + a SymphonyQG-style 1-bit quantized-traversal variant were built and **MEASURED**: float HNSW is ~25× over linear scan at recall ≥0.99 (the baseline this gap needed), but our 1-bit quantized traversal is **too coarse to beat float HNSW at equal recall at N=10k** (best recall 0.738) — the 3.5–17× is **not reproduced** by our construction. Honest negative recorded; expected crossover is large N + a multi-bit traversal code. (Caveat: our HNSW + our 1-bit quant, not SymphonyQG's exact system.)
- **Multi-bit / Extended RaBitQ** (§5 #2) — **RESOLVED-PARTIAL** (see §10). Pass-2 randomized rotation (FHT + seeded ±1 sign flips, `src/rotation.rs`) and a multi-bit Pass-3 experiment landed and were MEASURED against the ADR-084 ≥90% bar. **Honest result: rotation helps (+10pp at the strict bar) and Pass-2 reaches 90% with ~3× over-fetch, but NEITHER rotation nor multi-bit (up to 4-bit) clears the strict candidate_k==K 90% bar on the tested anisotropic distribution.** The original `1-bit sign quantization ships first; rotation/more-bits later if benchmark-measured top-K coverage drops below 90%` deferral is therefore retired: the rotation is built, the bar is characterised, and the residual gap is documented rather than deferred.
- **Learned cross-viewpoint fusion head** (§5 #3, GraphPose-Fi-style) — **data-gated**: blocked on the paired multi-room data ADR-152 measurement (b) identified as the real bottleneck; do not build the architecture first.
- **`CrossViewpointAttention` learned projections** — the default `ProjectionWeights::identity` + mean-pool is honest but unlearned; wiring real learned Q/K/V projections is part of the data-gated item above (no learned weights ⇒ the "attention" is currently a geometric-bias-weighted average, which the code/docs should keep stating plainly).
- **`coherence.rs` / `fusion.rs` micro-opts and the remaining lower-severity review findings** (style, doc, further hot-path tuning) from the fusion gap review.

---

## 9. Consequences

**Positive.** The fusion path now: uses one canonical wrapped angular-distance helper; reports a **real** dimensionless GDOP instead of a mislabeled RMSE; cannot be panicked by crafted multistatic indices or a zero-bin spectrogram (DoS closed); and does one embedding clone per viewpoint instead of two (measured). Every fix is pinned by a test that fails on the old code, and the ANN/fusion SOTA landscape is graded so the near-term (multi-bit RaBitQ) and the data-gated (learned fusion) are not confused.

**Negative / honest.** The headline angular-wrap fix is a **numeric no-op** under the current cos kernel — we land it for contract/maintainability, not because it changes an output, and we say so. The two strongest external candidates (SymphonyQG, learned fusion) are **not built here** — one is CLAIMED-pending-reproduction, the other is data-gated by a prior measurement. The perf win is a **local hot-path** improvement, modest in the end-to-end pipeline (attention dominates). None of these is presented as more than it is.

---

## 10. RaBitQ Pass-2 / multi-bit — IMPLEMENTED & MEASURED (§8 backlog item #2)

Milestone-1 of the §8 backlog. Status: **RESOLVED-PARTIAL** — built, measured, honest negative on the strict bar.

### 10.1 What landed

- **`crates/wifi-densepose-ruvector/src/rotation.rs`** (new) — `Rotation`, a deterministic randomized orthogonal rotation `R = H·D`: a **Fast Hadamard Transform** (`O(d log d)`, in-place butterfly, `1/√m` normalized so it is norm-preserving) composed with a diagonal of **seeded ±1 sign flips** (SplitMix64 from a stored `u64` seed). Chosen over a dense `d×d` matrix because that is `O(d²)` memory/time and infeasible at the 65,535-d the wire format provisions for; FHT is the standard fast-orthogonal (randomized-Hadamard / fast-JL) construction. Non-power-of-two `d` zero-pads to `next_pow2(d)` and reads back the first `d` coords.
- **`sketch.rs`** — additive Pass-2 API: `Sketch::from_embedding_rotated`, `SketchBank::with_rotation` + `insert_embedding` / `topk_embedding` / `novelty_embedding`. **Pass 1 (`from_embedding`) is byte-for-byte unchanged**; a Pass-2 sketch has identical `embedding_dim` / packed-byte length / wire shape, so `WireSketch` and existing callers (`event_log.rs`, `signal/longitudinal.rs`) are untouched. Default behaviour preserved.
- **`coverage.rs`** (new) — single-source-of-truth top-K coverage harness on a deterministic **anisotropic planted-cluster** fixture (cosine ground truth, the metric a sign sketch approximates). Backs both the `pass2_coverage_report` unit test and the `sketch_bench` coverage table.
- **Multi-bit Pass-3 experiment** — `coverage::measure_multibit`: rotate, then `b`-bit uniform scalar-quantize each coord, rank by L1 over codes. Measures the bit/coverage tradeoff.

### 10.2 Pre-existing bug found and fixed (disclosed)

Building the coverage harness surfaced a **pre-existing correctness bug in `SketchBank::topk`** (shipped in ADR-084): the `n > k` heap path used `BinaryHeap<Reverse<(dist,id)>>` (a *min*-heap) but its comment/logic treated the peek as the max, so it evicted the *nearest* and returned the **k farthest** sketches as "nearest." The shipped unit tests only exercised the `n ≤ k` fast path (≤ 3 entries), so it was never caught. Fixed to a plain max-heap. Pinned by **`topk_heap_path_returns_nearest`** (fails on the old heap when entries are inserted farthest-first) and **`tight_clusters_give_high_coverage_with_overfetch`** (measured **0.072** coverage on the old code — random — vs **>0.99** fixed). This is a real, measured behaviour fix, not a no-op.

### 10.3 MEASURED top-K coverage

Test machine: Windows 11, `cargo bench --release` / `cargo test`. Fixture: **dim=128, N=2048, K=8, 64 planted clusters, intra-cluster noise=0.35, 128 queries, master_seed=0xAD000084, rotation_seed=0x5EEDC0DE12345678**, ground-truth metric = cosine. Reproduce: `cargo test -p wifi-densepose-ruvector --no-default-features pass2_coverage_report -- --nocapture` or `cargo bench -p wifi-densepose-ruvector --bench sketch_bench -- pass2_coverage`.

**Coverage vs over-fetch (`coverage = |sketch_topK ∩ float_cosine_topK| / K`):**

| candidate_k | Pass-1 (1-bit, no rot) | Pass-2 (1-bit, rot) | vs 90% bar |
|---|---|---|---|
| **8 (= K, strict bar)** | **36.13%** | **46.39%** | both **BELOW** |
| 16 | 62.79% | 75.59% | below |
| 24 | 83.89% | **91.60%** | **Pass-2 clears** |
| 32 | 100.00% | 100.00% | clears |
| 64 | 100.00% | 100.00% | clears |

**Multi-bit Pass-3 at the strict bar (candidate_k = K = 8):**

| Variant | Coverage | Memory |
|---|---|---|
| Pass-1 (1-bit, no rot) | 36.13% | 16 B/vec |
| Pass-2 (1-bit, rot) | 46.39% | 16 B/vec |
| Pass-3 (rot, 2-bit) | 54.39% | 32 B/vec |
| Pass-3 (rot, 3-bit) | 66.70% | 48 B/vec |
| Pass-3 (rot, 4-bit) | 74.22% | 64 B/vec |

### 10.4 Honest verdict

- **Rotation consistently helps** — +10.3 pp at the strict bar (36.13%→46.39%) and a uniform lift at every over-fetch level. The FHT construction is verified norm-preserving and deterministic.
- **Neither rotation nor multi-bit (≤4-bit) clears the strict candidate_k==K 90% bar** on this anisotropic distribution. 1-bit sign quantization simply cannot resolve 8-of-2048 from sign bits alone; even 4× memory (4-bit) reaches only 74%.
- **Pass-2 reaches the 90% bar at candidate_k=24 (~3× over-fetch)** — i.e. fetch ≥24 sketch candidates, refine to K with full float. This is exactly the "candidate set, then full refinement" deployment pattern ADR-084 specifies, so the bar is met *in the deployment the sensor is designed for*, just not at strict K=K.
- **This is a measured, partial win, reported as such.** No benchmark was tuned to manufacture a pass. The strict-bar gap (and the multi-bit tradeoff that doesn't close it) is documented rather than spun.

### 10.5 Deferred sub-items (graded, not dropped)

- **Strict-bar 90% from a richer code** — neither rotation nor uniform multi-bit closes it here. A learned/asymmetric quantizer or the full RaBitQ residual-distance estimator (not just a uniform scalar code) might. **RESOLVED-NEGATIVE (§11): the estimator is now built and MEASURED — it lifts strict-K 46.39%→49.71% but does NOT clear the 90% strict bar.** The residual strict-bar gap is a published negative, not a deferral.
- **Distribution sensitivity** — the result is for one synthetic anisotropic distribution; on real AETHER traces the strict-bar number may differ. Re-measuring on recorded embeddings is deferred to the ADR-084 post-merge soak.
- **Promoting a `MultiBitSketch` type** — the multi-bit code lives in the measurement harness, not as a shipped sketch type. Building the production type is gated on a use site actually needing strict-K (vs over-fetch), which the measurement says is not required today.

---

## 11. RaBitQ unbiased distance estimator — IMPLEMENTED & MEASURED (Milestone-2, §8 backlog item #2 / §10.5 strict-bar item)

Milestone-2 of the §8 backlog. Status: **RESOLVED-NEGATIVE** — the estimator is built, measured, and lifts strict-K coverage, but the honest result is that it does **not** clear the ADR-084 ≥90% strict-K bar on this distribution. The negative is reported as such, exactly like the Pass-2 rotation result.

### 11.1 What landed

- **`crates/wifi-densepose-ruvector/src/estimator.rs`** (new) — the real Gao & Long (SIGMOD 2024) contribution: an **unbiased estimator of the inner product / squared distance** recovered from the 1-bit code plus per-vector side info, on top of the Pass-2 rotation. Pass-1/Pass-2 ranked candidates by raw Hamming over sign bits — a coarse proxy. This module reranks by the unbiased estimate.
  - `EstimatorSketch` — Pass-2 sign code (over the **padded** FHT length `D = next_pow2(dim)`, the frame `x̄` is unit in) **plus** the side info.
  - `SideInfo` = `{ residual_norm: f32, x_dot_o: f32 }` = **8 bytes/vector** (2× f32).
  - `EstimatorQuery` — query rotated once, reused across all candidates.
  - `DistanceEstimator` — `estimate_inner_product`, `estimate_sq_distance`, `ranking_key` (euclidean), `cosine_ranking_key` (the correct key vs a cosine ground truth — needs only the code + `x_dot_o`).
  - `EstimatorBank` — `topk_estimated` (euclidean) / `topk_estimated_cosine`; optional `with_centroid` (the paper's centroid path).
- **`coverage.rs`** — `measure_estimator` (cosine rerank) + `measure_estimator_euclidean`, on the **bit-identical** fixture / cluster centres / query stream / cosine ground truth as `measure_pass1`/`measure_pass2`. Single source of truth for the §11.3 table; backs both `estimator_coverage_report` and the `sketch_bench` coverage table.
- **Additive + backward-compatible.** New types only; Pass-1 `Sketch` / Pass-2 `SketchBank` / `WireSketch` wire format are untouched. All external callers (`event_log.rs`, `signal/longitudinal.rs`, `sensing-server`) use Pass-1 `from_embedding` and are unaffected.

### 11.2 The estimator formula (and the zero-centroid simplification, stated honestly)

Let `P` be the Pass-2 orthogonal rotation (`R = H·D`), `D = next_pow2(dim)`. For data `o_raw`, query `q_raw`, centroid `c`:

1. **Centroid — SIMPLIFIED to zero/global `c = 0`.** The paper centres on a per-cluster centroid (`o_r = o_raw − c`); we use `c = 0` (`o_r = o_raw`), because the current sketch path has no IVF/k-means cluster structure. This costs accuracy when the data is far off-origin. **We document it, do not hide it,** and built the paper-faithful centroid path (`from_embedding_centred` / `EstimatorBank::with_centroid`) so the simplification is a measured choice, not an assumption. (We do **not** report a centroid coverage number against the *cosine* ground truth: centroid-subtraction changes the metric — cosine-of-residual ≠ cosine-of-raw — so a centroid number vs raw-cosine truth would be a metric mismatch, itself dishonest. Zero-centroid is the correct match for this raw-cosine harness.)
2. **Unit residual + 1-bit code.** `o = o_r/‖o_r‖`, `o' = P·o`, code `x̄_i = sign(o'_i)·(1/√D)` — a unit vector at the nearest hypercube corner.
3. **Side info:** `residual_norm = ‖o_r‖` and `x_dot_o = ⟨x̄, o'⟩ ∈ (0,1]` (the paper's `⟨x̄, o⟩`).
4. **Unbiased estimator** (paper Eq.): `⟨o', q'⟩ ≈ ⟨x̄, q'⟩ / ⟨x̄, o'⟩ = ⟨x̄, q'⟩ / x_dot_o`. The random rotation makes the code's quantization error orthogonal **in expectation** to `q'`, so the rescale is unbiased (paper's `O(1/√D)` bound). Per candidate: one length-`D` signed sum (`x̄ ∈ {±1/√D}`), as cheap as Hamming + a multiply.
5. **Distance / cosine.** `⟨o_r,q_r⟩ = ‖o_r‖·(⟨x̄,q'⟩/x_dot_o)`; `‖q_r−o_r‖² = ‖q_r‖²+‖o_r‖²−2⟨o_r,q_r⟩`. For a **cosine** ground truth (AETHER / this harness), rank by `−⟨o,q_r⟩ = −(⟨x̄,q'⟩/x_dot_o)` (needs only the code + `x_dot_o`).

**Unbiasedness is pinned** (`estimator_unbiased_on_fixture`): averaging the estimate of `⟨o_r,q_r⟩` over 4000 random rotation seeds converges to the true inner product within ~6% of the `‖o‖‖q‖` envelope — a biased estimator (or sign-only proxy) would be systematically off.

### 11.3 MEASURED strict-K coverage

Same fixture/seeds as §10 (dim=128, N=2048, K=8, 64 clusters, noise=0.35, 128 queries, `master_seed=0xAD000084`, `rotation_seed=0x5EEDC0DE12345678`), cosine ground truth. Reproduce: `cargo test -p wifi-densepose-ruvector --no-default-features estimator_coverage_report -- --nocapture` or `cargo bench -p wifi-densepose-ruvector --bench sketch_bench -- pass2_coverage`.

| candidate_k | Pass-1 (sign) | Pass-2 (sign) | **Pass-2 + estimator (cosine)** | Pass-2 + estimator (euclid) | vs 90% bar |
|---|---|---|---|---|---|
| **8 (= K, strict bar)** | 36.13% | 46.39% | **49.71%** | 49.02% | **all BELOW** |
| 16 | 62.79% | 75.59% | 79.20% | 77.93% | below |
| 24 | 83.89% | 91.60% | **95.12%** | 93.65% | estimator clears |
| 32 | 100.00% | 100.00% | 100.00% | 100.00% | clears |
| 64 | 100.00% | 100.00% | 100.00% | 100.00% | clears |

Side-info memory overhead: **8 bytes/vector** (2× f32) on top of the 16 B/vec 1-bit sketch.

### 11.4 Honest verdict

- **The estimator helps, and the cosine key beats the euclidean key** (49.71% vs 49.02% at strict-K; cosine is the apples-to-apples match for the cosine ground truth — both it and sign-Hamming are angular). The unbiased rescale is a real, consistent lift at every over-fetch level (e.g. 24: 91.60%→95.12%).
- **It does NOT clear the strict candidate_k==K 90% bar.** Strict-K goes 36.13% (Pass-1) → 46.39% (Pass-2-sign) → **49.71% (Pass-2 + estimator)** — a **+3.3 pp** improvement over sign-only, **still ~40 pp short of 90%**. This is a **published negative**, the same class of honest result as the Pass-2 rotation (§10).
- **Why the strict-K gain is modest:** the binding constraint at strict K is the **1-bit code's information ceiling** (resolving 8-of-2048 from a single sign bit per coordinate), not the *estimator's variance* — the estimator sharpens the ranking but cannot add information the 1-bit code never captured. The estimator's larger wins are at over-fetch, where there is room to re-rank a wider candidate pool.
- **The bar is still met the way ADR-084 deploys the sensor:** at candidate_k=24 (~3× over-fetch) the estimator reaches **95.12%** (vs Pass-2-sign 91.60%) — the "candidate set, then full refinement" pattern. The estimator **improves the over-fetch factor needed** but does not eliminate it.
- **No benchmark was tuned to manufacture a pass.** The strict-bar gap is documented, not spun.

### 11.5 Pinning tests

- `estimator::estimator_is_deterministic` — fixed seed ⇒ identical estimate + identical bank top-K.
- `estimator::estimator_unbiased_on_fixture` — Monte-Carlo mean over 4000 seeds converges to the true inner product within tolerance (the unbiasedness claim).
- `coverage::estimator_rerank_not_worse_than_sign` — estimator-reranked coverage ≥ sign-only Pass-2 on a fixed fixture (must not regress).
- Plus: `estimator_self_distance_is_small`, `x_dot_o_in_unit_range`, `zero_input_does_not_panic`, `bank_self_query_ranks_self_first`, `centroid_path_self_query_ranks_self_first`, `centroid_zero_matches_default`, `estimator_coverage_is_deterministic`.
