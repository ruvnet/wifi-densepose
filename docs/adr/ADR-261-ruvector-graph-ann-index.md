# ADR-261: RuVector Graph-ANN Index — a real HNSW baseline + a SymphonyQG-style quantized variant, MEASURED

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-06-14 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-ruvector` — `hnsw.rs`, `hnsw_quantized.rs`, `ann_measure.rs`, `benches/ann_bench.rs`, docs |
| **Relates to** | ADR-084 (RaBitQ similarity sensor — 1-bit sketch), ADR-156 (RuVector beyond-SOTA sweep — §5 #1 SymphonyQG, §8/§10/§11 RaBitQ Pass-2/multi-bit/estimator), ADR-024 (AETHER re-ID), ADR-016/017 (RuVector integration) |
| **Scope** | Build the **missing HNSW graph-ANN baseline** in the ruvector retrieval path, build a **SymphonyQG-style quantized-traversal variant** on the same graph, and **MEASURE** the real recall/QPS ratio between them — closing the ADR-156 §5 #1 gap honestly. Resolves ADR-156 §8 backlog item **"SymphonyQG reproduction"** from **CLAIMED-only** to **MEASURED-direction-tested**. |

---

## 0. PROOF discipline (this ADR's contract)

This project has been publicly accused of "AI slop." This ADR answers with **evidence, not adjectives** — the same contract as ADR-154/156:

- The HNSW index ships a **committed recall@10 correctness gate** (≥ 0.95 vs brute force on a planted-cluster fixture). Low recall means a graph bug; the gate is wired to fail in that case. It **did** fail first — and caught a real index-out-of-bounds bug in the insert path (§4) — which is exactly what a real gate is for.
- Every QPS/recall number below is **MEASURED** on this box with a committed, deterministic, `--no-default-features`-runnable measurement (`src/ann_measure.rs`, `ann_bench_report`) and a committed criterion bench (`benches/ann_bench.rs`). Both call **one** shared fixture/measurement module, so the bench and the report can never measure different graphs.
- The **headline result is an honest negative**: at our test scale the SymphonyQG-style quantized variant **does not beat float HNSW at equal recall** — the 1-bit Hamming traversal is too coarse to keep recall up. We report the real numbers, explain *why*, and state the expected large-N crossover. **We did not tune the quantized path to manufacture the 3.5–17× the source claims.** A measured negative + a scale caveat is a valid, publishable result.
- We are explicit that this is **OUR HNSW + OUR 1-bit quantization, not SymphonyQG's exact system**. It tests the **direction** of the claim on our hardware/data, not a 1:1 reproduction.

Test machine: Windows 11, `cargo test --release`, `std::time::Instant` wall-clock. Numbers are warm medians on this box; the **ratio** is the claim, not the absolute QPS.

Reproduce:
```bash
cd v2 && cargo test -p wifi-densepose-ruvector --no-default-features --release \
  ann_bench_report -- --nocapture
# Larger N: ANN_BENCH_N=50000 cargo test ... --release ann_bench_report -- --nocapture
cargo bench -p wifi-densepose-ruvector --bench ann_bench
```

---

## 1. Context

The ruvector crate's retrieval path — AETHER re-ID hot-cache (ADR-024), the `sketch.rs` 1-bit prefilter (ADR-084), room fingerprinting — is, at its core, an **approximate nearest-neighbour (ANN)** problem: dense float embedding in, top-K similar ids out. But **the crate had no graph index**. Every `topk` was either a linear scan (`O(N·d)` per query) or a 1-bit Hamming prefilter over a linear scan. That is `O(N)` per query and does not scale.

[ADR-156 §5 #1](ADR-156-ruvector-fusion-beyond-sota.md) graded **SymphonyQG** (SIGMOD 2025) the **lead beyond-SOTA ANN candidate**, citing the source's claim of **3.5–17× QPS over HNSW at equal recall**, but marked it **CLAIMED**:

> *"author-measured; **not reproduced on our hardware** — reproduction is future work."*

And ADR-156 §8 was blunt about *why* it could not be reproduced: **there was no HNSW baseline to compare against.** You cannot measure a ratio against a baseline that does not exist. This ADR builds that missing baseline, builds the quantized variant that tests the direction of the SymphonyQG bet, and measures the real ratio.

---

## 2. Decision

1. Add a correct, dependency-free **float HNSW** graph index (`hnsw.rs`): the real Malkov & Yashunin (TPAMI 2018) algorithm — multi-layer navigable small-world graph, `ef_construction` / `ef_search`, the Algorithm-4 neighbour-selection heuristic, seeded-deterministic level assignment, L2 + cosine. This is the **baseline** ADR-156 said was missing.
2. Add a **SymphonyQG-style quantized-traversal variant** (`hnsw_quantized.rs`): the *same* graph (same seed, same structure), but the beam search scores candidates with a **cheap 1-bit Hamming distance** over the RaBitQ Pass-2 rotated sign code (reusing `rotation.rs` + the sign-quantization of `sketch.rs`), then **exact-float reranks** the final candidate set. This is the SymphonyQG bet — cheaper per-node scoring, recovered by a final exact rerank.
3. **Measure** linear vs float-HNSW vs quantized-HNSW (recall@10, QPS, equal-recall ratios) on one deterministic planted-cluster fixture, and record the honest verdict against the SymphonyQG 3.5–17× claim.

### Why 1-bit Hamming for the quantized traversal

The crate already had the exact pieces SymphonyQG fuses: a deterministic orthogonal rotation (`rotation.rs`, RaBitQ Pass-2) and sign-quantization (`sketch.rs`). A 1-bit code compares by POPCNT Hamming — a few machine words, no per-dimension float work — so it is the cheapest possible traversal score and the most direct test of "can a quantized score keep the beam on the right path." The cost (measured below): the 1-bit code is a *coarse* angle proxy (ADR-156 §10 measured ~46% strict-K coverage for sign-only), and that coarseness is what limits recall here.

---

## 3. Design

### 3.1 `hnsw.rs` — float HNSW (the baseline)

- **Graph.** `links[id][layer]` adjacency; layer 0 holds every node, higher layers exponentially sparser. `m_max` is `2·M` on layer 0, `M` above (the paper's asymmetric degree cap).
- **Insert.** Greedy-descend the upper layers to a good entry point, then for each layer from the node's level down to 0: `search_layer` for `ef_construction` candidates, `select_neighbours` (Algorithm 4 — keep a candidate only if it is closer to the new node than to any already-selected neighbour, giving diverse navigable edges), wire bidirectional edges, re-prune any neighbour that overflows `m_max`. The node is pushed into the arrays **before** wiring so every `links[*]` index is valid mid-insert (§4 — the bug the gate caught).
- **Search.** Greedy-descend layers `>0`, then best-first beam search of width `ef` on layer 0; return the closest `k`. Iterative (explicit heaps + visited set) — **no recursion**, bounded by the beam and the visited set.
- **Determinism.** Level assignment is the only randomness and is driven by a **seeded SplitMix64** (the exact pattern from `rotation.rs`) — never `Date::now`/OS RNG/unseeded `rand`. Same `(seed, params, insertion order)` ⇒ bit-identical graph and search (pinned by `hnsw_is_deterministic_for_seed`).
- **Robustness.** Empty index, `k==0`, `k>n`, single node, zero-dim, ragged query, `ef<k` all return cleanly — pinned by `*_no_panic` tests.

### 3.2 `hnsw_quantized.rs` — the SymphonyQG-style variant

Same graph as the float index (identical seed/structure — the **only** variable is the scoring), plus a per-node `ceil(D/8)`-byte 1-bit Pass-2 sign code (`D = next_pow2(dim)`). `search_quantized(query, k, ef, rerank)`:
1. Encode the query to its 1-bit code (one rotation + sign pack).
2. Greedy-descend + beam-search the graph scoring every visited node by **POPCNT Hamming** (query-code XOR node-code) — no per-dim float work.
3. **Exact-float rerank** the top `rerank` Hamming candidates with the true L2/cosine metric, return the best `k`.

### 3.3 Security / robustness

Both indices: bounded **iterative** traversal (no unbounded recursion), no panic on empty/degenerate/ragged/zero-dim input (the metric compares over the shorter prefix; zero-norm cosine returns max distance, not NaN). The 1-bit encode handles padded dims via the existing `Rotation::apply_padded`.

---

## 4. The bug the correctness gate caught (disclosed, not hidden)

The first run of the recall@10 gate **panicked**: `index out of bounds: the len is 33 but the index is 33` in `search_layer`. Root cause: `insert` wired bidirectional edges (`links[nbr][l].push(id)`) **before** pushing the new node's own `links[id]` row into the array. A later traversal step in the *same* insert could hop to a neighbour that now pointed at `id` and read `links[id]` — which did not exist yet. Fix: push the node (with empty per-layer link lists) into `vectors`/`links`/`levels` **up front**, then wire edges into its existing slot. The new node has no incoming edges and empty outgoing lists until wiring, so it is unreachable by the searches that run first — pushing early is safe and keeps every index valid. This is exactly why the recall gate exists: a silent low-recall graph and an out-of-bounds panic are both "slop" the gate forces into the open.

---

## 5. The SymphonyQG claim being tested

| Source | Claim | Grade (before this ADR) |
|--------|-------|-------------------------|
| SymphonyQG, SIGMOD 2025 | **3.5–17× QPS over HNSW at equal recall**, via quantization unified with graph traversal, pure-CPU/edge-portable | **CLAIMED** — author-measured, *not reproduced on our hardware (no HNSW baseline existed)* |

The bet: a quantized traversal score is cheap enough — and accurate enough to keep the beam on-path — that you pay far less per visited node and recover the small recall loss with a final exact rerank.

---

## 6. MEASURED results

Fixture: planted-cluster synthetic, **dim=128, N=10,000, 64 clusters, 200 queries, K=10, noise=0.35**, L2 metric, `M=16`, `ef_construction=200`. Graph seed `0x6261524741484E53`, rotation seed `0x5EEDC0DE12345678`. `--release`, warm wall-clock on the test machine. (The fixture and both indices are shared by the criterion bench.)

| Method | recall@10 | QPS | latency (µs) |
|--------|-----------|-----|--------------|
| **linear scan (brute force)** | 1.0000 | 1,022 | 978 |
| **float-HNSW** ef=16 | 0.9945 | **25,744** | 39 |
| float-HNSW ef=32 | 0.9990 | 21,470 | 47 |
| float-HNSW ef=64 | 1.0000 | 18,779 | 53 |
| float-HNSW ef=128 | 1.0000 | 12,722 | 79 |
| float-HNSW ef=256 | 1.0000 | 5,742 | 174 |
| quant-HNSW ef=32 rr=20 | 0.1620 | 30,005 | 33 |
| quant-HNSW ef=32 rr=100 | 0.2615 | 36,388 | 28 |
| quant-HNSW ef=64 rr=100 | 0.4865 | 20,603 | 49 |
| quant-HNSW ef=128 rr=100 | 0.6785 | 13,718 | 73 |
| quant-HNSW ef=256 rr=100 | **0.7380** | 6,578 | 152 |

### Equal-recall QPS ratios

| Target recall | Fastest float-HNSW | Fastest quant-HNSW meeting it | quant/float | float/linear |
|---------------|--------------------|-------------------------------|-------------|--------------|
| ≥ 0.90 | ef=16 → 25,744 QPS | **none** (best quant recall = 0.738) | — | **25.19×** |
| ≥ 0.95 | ef=16 → 25,744 QPS | **none** | — | **25.19×** |
| ≥ 0.99 | ef=16 → 25,744 QPS | **none** | — | **25.19×** |

---

## 7. Honest verdict

**The HNSW baseline is a decisive win over linear scan: ~25× QPS at recall ≥ 0.99** (ef=16: 0.9945 recall, 25,744 QPS vs linear 1,022 QPS). The correctness gate (recall@10 ≥ 0.95 vs brute force, both L2 and cosine) holds. This is the baseline ADR-156 §5 #1 said did not exist — it now does.

**The SymphonyQG-style quantized variant does NOT beat float HNSW at our scale — direction REFUTED at N=10k.** The 1-bit Hamming traversal is too coarse: its best achievable recall is **0.738** (ef=256, rr=100), and it never reaches even the 0.90 equal-recall point where a fair QPS comparison could be made. Where the quantized score *is* faster (ef=32: ~30–36k QPS, beating float's 25.7k), its recall collapses to 0.16–0.26 — a meaningless win. There is **no equal-recall operating point** at which quantized is faster, so the SymphonyQG 3.5–17× claim is **not reproduced** by our 1-bit construction here.

**Why** (so the negative is understood, not just stated):
1. The 1-bit sign code is a **coarse angle proxy** — ADR-156 §10 already measured it at ~46% strict-K coverage. Driving graph *traversal* by that coarse score steers the beam onto the wrong nodes, and the exact-float rerank can only recover what the beam actually visited. At N=10k, near-neighbours have nearly-identical sign codes, so Hamming cannot separate them.
2. At this scale **float distance is already cheap**: one 128-d L2 is a handful of µs; the per-node float compute the quantization saves is small relative to the recall it costs. SymphonyQG's win shows up at **much larger N** (millions), where (a) the float-distance fraction of query time dominates and (b) their *multi-bit RaBitQ-fused* code (not our 1-bit sign code) keeps recall high. **Expected crossover: large N + a higher-bit code.** ADR-156 §10 already measured that a ≤4-bit code reaches ~74% strict coverage vs 1-bit's ~46%, so a multi-bit traversal score is the obvious next lever — deferred, not claimed.

**Caveat (stated plainly):** this is **our** HNSW + **our** 1-bit quantization, not SymphonyQG's system. We tested the *direction* of the claim ("does quantized traversal + rerank beat float HNSW at equal recall?") on our hardware/data and got a **measured no at N=10k**. That neither confirms nor refutes SymphonyQG's own published numbers on their system/scale — it refutes the direction *for our construction at our scale*, and identifies the two levers (scale, code bit-depth) a real reproduction would need.

---

## 8. Validation

- **`cd v2 && cargo test -p wifi-densepose-ruvector --no-default-features --lib`** — **151 passed / 0 failed** (was 131; +20 new tests: 10 `hnsw`, 7 `hnsw_quantized`, 3 `ann_measure`).
- **`cargo test --workspace --no-default-features`** — GREEN (see §10 for the count).
- **Correctness gate verified to bite:** the recall@10 gate **panicked** on the first (buggy) insert path (§4); after the fix it passes at 0.99+ recall (L2 and cosine).
- **`cargo test -p wifi-densepose-ruvector --no-default-features --release ann_bench_report -- --nocapture`** — prints the §6 table; the numbers above are copied verbatim from that run.
- **`cargo bench -p wifi-densepose-ruvector --bench ann_bench`** — compiles and runs the same fixture through criterion.
- **`python archive/v1/data/proof/verify.py`** — **VERDICT: PASS** (the Rust ANN work is independent of the Python signal-proof pipeline; hash unchanged).

---

## 9. Consequences

**Positive.** ruvector now has a real, deterministic, pure-Rust HNSW graph index (25× over linear scan at high recall) usable by the AETHER re-ID / sketch-prefilter path — the ANN substrate ADR-156 §5 #1 wanted. The SymphonyQG claim is no longer CLAIMED-only: we built the missing baseline and **measured** the direction, with the bug-caught-by-the-gate disclosed.

**Negative / honest.** The 1-bit quantized variant is **not** an equal-recall QPS win at our scale; it is shipped as a measured experiment with a clearly-stated ceiling, not as a recommended default. Anyone reaching for it must read §7.

**Deferred (not silently dropped).**
- **Multi-bit / RaBitQ-estimator traversal score.** Replace 1-bit Hamming traversal with a ≤4-bit code or the `estimator.rs` unbiased rescale (ADR-156 §10/§11) — the lever most likely to lift quantized recall to the equal-recall regime.
- **Large-N crossover measurement.** Re-run §6 at N=100k–1M (`ANN_BENCH_N`) to find where quantization's per-node saving starts to dominate.
- **Wiring HNSW into the live re-ID path** (AETHER hot-cache / sketch prefilter) behind a flag.

---

## 10. What changed, file by file

- `hnsw.rs` (new) — float HNSW: graph, seeded-deterministic level assignment, Algorithm-2 beam search, Algorithm-4 neighbour selection, L2/cosine, brute-force ground truth, full degenerate-case guards; 10 tests incl. the recall@10 correctness gate (L2 + cosine) and determinism. The insert-order bug fix (§4).
- `hnsw_quantized.rs` (new) — SymphonyQG-style quantized-traversal index over the shared graph: 1-bit Pass-2 code per node, Hamming-scored greedy + beam, exact-float rerank; 7 tests incl. the rerank-recall gate and determinism.
- `ann_measure.rs` (new) — shared deterministic fixture + recall/QPS measurement for linear / float-HNSW / quant-HNSW, the `ann_bench_report` test (the §6 source of truth), `ANN_BENCH_N` override.
- `benches/ann_bench.rs` (new) + `Cargo.toml` `[[bench]]` — criterion bench over the same fixture/indices.
- `lib.rs` — `pub mod hnsw / hnsw_quantized / ann_measure`; re-export `HnswIndex`, `HnswParams`, `Metric`, `QuantizedHnswIndex`.
- `ADR-156-ruvector-fusion-beyond-sota.md` §5 #1 + §8 backlog — SymphonyQG regraded **CLAIMED → MEASURED-direction-tested (refuted at N=10k for our 1-bit construction)**, pointing here.
- `CHANGELOG.md` — `[Unreleased]` entry.
