# RuView Beyond-SOTA — 04: Performance Review & Optimization Roadmap

**Scope:** the streaming sensing pipeline (CSI ingest → multistatic fusion → CIR gate →
pose publish) in `v2/`, hot-path crates `wifi-densepose-signal` (ruvsense),
`wifi-densepose-engine`, `wifi-densepose-ruvector`, plus build-profile and edge-target
(Pi 5-class, WASM) considerations.

**Hard constraint (non-negotiable):** the witness chain (ADR-028, ADR-136 §2.5 replay
contract, ADR-137 §2.7 BLAKE3 witness in
`v2/crates/wifi-densepose-engine/src/lib.rs:437-448`) requires **bit-exact deterministic
float output**. Every recommendation below is tagged with its determinism risk. Anything
that reorders float additions, enables FMA contraction, fast-math, or parallel reduction
**changes the witness hash** and requires a coordinated proof-hash regeneration
(`verify.py --generate-hash`) plus witness-bundle re-issue.

---

## 1. What we actually have measured (and what we don't)

`/home/user/RuView/benchmark_baseline.json` is a **signal-quality soak baseline**, not a
latency benchmark: 1,566 samples (ticks 51131–52395) of
`variance / motion / presence / confidence / est_persons / kp_spread / rssi`, with a
summary block (`confidence_mean: 0.643`, `presence_ratio: 0.934`,
`kp_spread_mean: 86.7`, `person_count_changes: 10`). **It contains zero timing data.**
It is the accuracy guardrail for any optimization (post-change soak must reproduce these
distributions), not a latency baseline.

Latency benchmarks exist but no committed results were found in the repo:

| Bench | File | What it measures |
|---|---|---|
| `process_cycle_4nodes_56sc` | `v2/crates/wifi-densepose-engine/benches/engine_cycle.rs:34-48` | One full engine cycle, 4 nodes × 56 subcarriers, vs. the documented 50 ms budget (`engine_cycle.rs:3-6`) |
| `cir_bench` | `v2/crates/wifi-densepose-signal/benches/cir_bench.rs` | `CirEstimator::estimate()` per tier (HT20/HT40/HE20/HE40) + 12-link amortization |
| `sketch_bench` | `v2/crates/wifi-densepose-ruvector/benches/sketch_bench.rs:86-175` | Hamming sketch vs. float L2/cosine compare; top-K over 1,024-sketch bank |
| `signal_bench`, `calibration_bench`, `aether_prefilter_bench` | `v2/crates/wifi-densepose-signal/benches/` | Signal-path and ADR-135 calibration throughput |

**Action zero of the roadmap is to run these on a Pi 5 and commit the criterion
baselines.** All impact classes below are derived from operation counts read out of the
code (cited), not invented measurements.

---

## 2. Latency budget model — streaming pipeline

Two clock domains exist and must not be conflated:

- **TDMA sensing cycle: 20 Hz / 50 ms** — the architecture's own budget
  (`v2/crates/wifi-densepose-signal/src/ruvsense/mod.rs:5`, `RuvSenseConfig::target_hz =
  20.0` at `mod.rs:258`, and the bench doc `engine_cycle.rs:3`).
- **CSI ingest: 100 Hz per node** — raw frames arrive ~5× faster than the fused output
  rate; per-frame ingest work (parse, normalize, calibrate, window) must therefore fit a
  **10 ms** per-frame envelope while the fused path fits **< 50 ms end-to-end**.

Proposed per-stage budget for the 50 ms end-to-end target (4 nodes, HT20 / 56
subcarriers — the configuration the engine bench encodes):

| # | Stage | Code | Budget | Risk (from code reading) |
|---|---|---|---|---|
| 1 | Ingest + hardware normalize (per 100 Hz frame) | `hardware_norm`, `multiband.rs` | 2 ms | Low — vector ops on 56 floats |
| 2 | Calibration apply (ADR-135) | `ruvsense/calibration.rs` | 2 ms | Low — Welford lookups |
| 3 | Phase alignment | `phase_align.rs:117-152` | 1 ms | Low — ≤ 20 iterations over ≤ 17 static subcarriers (`config.max_iterations: 20`, `phase_align.rs:57`); allocation churn only (§3) |
| 4 | Multistatic fusion (attention + softmax) | `multistatic.rs:512-598` | 2 ms | Low — O(nodes × 56); but does duplicate work in `fuse_scored` (§3, F2) |
| 5 | **CIR gate (ISTA L1)** | `multistatic.rs:440-475` → `cir.rs:601-654` | 15 ms | **HIGH** — dominant cost, scales badly with PHY tier (below) |
| 6 | Coherence score + gate decision | `coherence.rs`, `coherence_gate.rs` | 2 ms | Low — z-scores over 56 subcarriers |
| 7 | Tomography (ADR-030 tier 2, when enabled) | `tomography.rs:236-323` | 8 ms | **Medium** — per-iteration allocation + loose step size (§3, F8/F9) |
| 8 | Pose tracker (17-kp Kalman + re-ID) | `pose_tracker.rs` | 8 ms | Medium — sketch prefilter (ADR-084) already mitigates the re-ID scan |
| 9 | Engine: quality score, privacy gate, WorldGraph node, BLAKE3 witness | `engine/src/lib.rs:304-368` | 5 ms | Low per cycle, but **unbounded memory growth** (§4) |
| 10 | Publish (WS/serde) | sensing-server | 5 ms | Low |
| | **Total** | | **50 ms** | |

### Why stage 5 is the at-risk stage — operation counts from the code

`ista_solve` (`cir.rs:601-654`) runs **two dense complex mat-vecs per iteration**
(`matvec_phi` at `cir.rs:717-726`, `matvec_phi_h` at `cir.rs:730-745`), each O(K·G)
complex MACs (≈ 8 FLOPs each), up to `max_iters: 100` (`cir.rs:176`). Per
`CirConfig` (`cir.rs:164-233`):

| Tier | K (active) | G (taps) | FLOPs/iter (2·K·G·8) | FLOPs @100 iters |
|---|---|---|---|---|
| HT20 | 52 | 156 | ≈ 0.13 M | ≈ 13 M |
| HT40 | 114 | 342 | ≈ 0.62 M | ≈ 62 M |
| HE20 | 242 | 726 | ≈ 2.8 M | ≈ 0.28 G |
| HE40 | 484 | 1,452 | ≈ 11.2 M | ≈ 1.1 G |

HT20 fits the 15 ms budget comfortably on a Pi 5; **HE40 at worst-case iteration count
is ~1.1 GFLOP of scalar, cache-unfriendly work per estimate and will not fit any 50 ms
budget without structural change** (F4 below). Today the gate runs once per cycle on the
first link only (`multistatic.rs:452-463`), which contains the damage; the 12-link
amortization pattern in `cir_bench.rs` shows the intended scale-up, which multiplies
this cost ×12.

---

## 3. Findings table — optimization opportunities

Impact: relative cycle-time/memory effect at the 4-node HT20 operating point unless
noted. Determinism: **EXACT** = bit-identical output guaranteed; **TIE** = only
tie-breaking/ordering may differ; **CHANGES-FLOATS** = output bits change, witness/proof
hash must be regenerated.

| ID | Finding (file:line) | Impact | Effort | Determinism |
|---|---|---|---|---|
| F1 | `FusedSensingFrame` deep-copies every input frame each cycle: `node_frames: node_frames.to_vec()` (`multistatic.rs:282`) — clones all per-node amplitude+phase vectors per 50 ms cycle even when downstream geometry consumers don't need them | Med | Low (Arc/Cow or borrow) | EXACT |
| F2 | `fuse_scored` re-derives the per-node amplitude views and recomputes `node_attention_weights` after `fuse` already computed them inside `attention_weighted_fusion` (`multistatic.rs:311-321` duplicating `multistatic.rs:520`) — full cosine-sim + softmax done twice per cycle | Low-Med | Low (return weights from `fuse`) | EXACT (same math, computed once) |
| F3 | CIR gate rebuilds a heap `CsiFrame` per cycle: `build_csi_frame_from_channel` allocates an `Array2<Complex64>` and converts amplitude/phase via `from_polar` per subcarrier (`multistatic.rs:488-506`, called from `multistatic.rs:462`), then `extract_csi_vector` converts back to `Complex32` (`cir.rs:505-530`) — f32→f64→f32 round-trip plus two allocations purely as glue | Med | Med (give `CirEstimator` a slice-based entry point) | EXACT if conversions reproduce exactly (f32→f64 is lossless; `from_polar` in f64 then truncate ≠ f32 polar — keep the f64 intermediate to stay exact, or accept CHANGES-FLOATS and regenerate hashes) |
| F4 | ISTA inner loop uses dense O(K·G) mat-vecs (`cir.rs:717-745`) although Φ is a sub-sampled DFT (`cir.rs:539-558`) — the products Φx and Φᴴr are computable via an FFT of length G in O(G log G), an ~8–40× FLOP cut at HE20/HE40 (table §2) | **High** (the only path to HE40 real-time) | High | **CHANGES-FLOATS** (different summation order than the sequential dot product) — must ship behind a feature flag, A/B against `cir_proof_runner`, regenerate `expected_features.sha256` + witness bundle |
| F5 | `neumann_warm_start` recomputes the diagonal of ΦᴴΦ with a full K×G pass **per frame** (`cir.rs:676-681`), rebuilds the COO→CSR diagonal matrix per frame (`cir.rs:683-685`), and collects `rhs_re`/`rhs_im` Vecs per frame (`cir.rs:689-690`) — yet `diag` depends only on Φ, which is fixed at `CirEstimator::new` | Med | Low (precompute diag+CSR in `new()`) | EXACT (same values, computed once) |
| F6 | `phase_variance` collects a `Vec<f32>` of phases per call (`cir.rs:792`) — replaceable by a two-pass loop with zero allocation | Low | Low | EXACT |
| F7 | Φ and Φᴴ are both stored densely (`cir.rs:546-547`): 2·K·G·8 bytes — Φᴴ entries are just conjugates of Φ (`cir.rs:555`), so a transposed-iteration kernel over Φ alone halves the footprint (HE40: 11.2 MB → 5.6 MB) | Low (latency) / Med (memory §4) | Med | EXACT (conjugation is exact; keep identical accumulation order in the transposed kernel) |
| F8 | Tomography allocates the gradient vector **inside** the solver iteration loop: `let mut gradient = vec![0.0_f64; self.n_voxels]` (`tomography.rs:266`) — one heap alloc + zeroing per iteration, up to `max_iterations: 100` (`tomography.rs:75`); hoist and `fill(0.0)` | Med (for tier-2 deployments) | Low | EXACT |
| F9 | Tomography step size uses the Frobenius-norm upper bound for the Lipschitz constant (`tomography.rs:253-259`, comment admits `‖WᵀW‖ ≤ ‖W‖_F²`) — a bound loose by up to the matrix rank, forcing proportionally more ISTA iterations than the power-method estimate used in `cir.rs:566-590` | Med | Low (reuse the cir.rs power-method pattern) | **CHANGES-FLOATS** (different step ⇒ different iterate path) |
| F10 | `apply_phase_correction` clones the amplitude vector and allocates a fresh corrected-phase Vec per channel per cycle (`phase_align.rs:258-268`, `frame.amplitude.clone()` at `phase_align.rs:264`); `align` additionally `frames.to_vec()`s on the single-channel path (`phase_align.rs:128`) — an in-place `align_mut` avoids all of it | Low-Med | Low | EXACT |
| F11 | Static-subcarrier selection fully sorts all subcarriers by variance (`phase_align.rs:180`) where `select_nth_unstable_by` suffices — trivial at 56 subcarriers, relevant at HE tiers (242–484) | Low | Low | **TIE** (equal-variance ties may select a different subcarrier set; pin a stable tie-break on index to stay EXACT) |
| F12 | Engine clones each node's amplitude vector for the array coordinator every cycle: `cf.amplitude.clone()` (`engine/src/lib.rs:385`); also allocates a `Vec<Option<CalibrationId>>` per cycle (`lib.rs:293`) and `format!("{e:?}")` strings for every evidence ref (`lib.rs:337`) | Low | Low | EXACT |
| F13 | `fuse_scored_calibrated` computes the modal calibration id in O(n²) (`multistatic.rs:404-410`) — harmless at n ≤ 15 nodes, noted for swarm-scale reuse (ADR-148) | Low | Low | EXACT |
| F14 | **No `rayon` and no SIMD feature exists anywhere in the hot crates** (grep over `crates/*/Cargo.toml`: zero hits for rayon/simd/target-feature outside wasm-opt flags). The 12-link CIR pattern (`cir_bench.rs:4-5`) and the per-node ingest path are embarrassingly parallel **across independent links/nodes** | High (multi-link tiers) | Med | **EXACT if and only if** parallelism stays at link/node granularity with results collected in deterministic (index) order and no shared float accumulator; intra-link parallel reductions are CHANGES-FLOATS and are banned |
| F15 | `Cir::top_k_taps` clones and fully sorts all G taps (`cir.rs:322-332`) — O(G log G) with a G-sized clone; a k-heap (the exact pattern already written in `sketch.rs:546-563`) is O(G log k) | Low | Low | TIE (equal-magnitude ordering; pin index tie-break) |
| F16 | Core `CsiFrame` carries `Complex64` while the entire ruvsense DSP path computes in f32 (conversion at `cir.rs:525`) — 2× memory and bandwidth on every ingest for precision the pipeline immediately discards | Med (memory/bandwidth) | High (core type change ripples everywhere) | **CHANGES-FLOATS** at the boundary; defer until a major version |
| F17 | Sketch path is already well-optimized: heap-based top-K with n ≤ k fast path (`sketch.rs:536-569`), 28-byte wire format (`sketch.rs:303`). Remaining win is build-level: `count_ones()` only lowers to POPCNT/NEON-vcnt when the target CPU enables it (see §5) | Low | Low | EXACT (integer ops) |

---

## 4. Memory-footprint analysis (Pi 5-class and WASM; ESP32 aggregation out of scope)

**Static, per-process (from struct definitions):**

| Component | Sizing source | Footprint |
|---|---|---|
| `CirEstimator` HT20 (Φ + Φᴴ, `Complex32`) | `cir.rs:546-547`, K=52 G=156 | 2 · 52 · 156 · 8 B ≈ **130 KB** |
| `CirEstimator` HE20 | K=242 G=726 | ≈ **2.8 MB** |
| `CirEstimator` HE40 | K=484 G=1452 | ≈ **11.2 MB** (halvable via F7) |
| Tomography weight matrix | `tomography.rs:214-217`, sparse per-link (voxel,weight) pairs; default grid 8×8×4 = 256 voxels (`tomography.rs:70-73`) | tens of KB at default grid |
| Sketch bank, 1,024 × 128-d | `sketch.rs` 1 bit/dim | 1,024 · 16 B ≈ **16 KB** (vs 512 KB float) |

A Pi 5 (4–8 GB) absorbs all of this trivially. The real memory risks are dynamic:

1. **Unbounded WorldGraph growth (the one genuine leak-class issue).** Every
   `process_cycle` appends a `SemanticState` node plus a `DerivedFrom` edge
   (`engine/src/lib.rs:346-352`), and change-points append `Event` nodes
   (`lib.rs:422-428`). At 20 Hz that is **1.73 M nodes/day** with no eviction anywhere
   in the engine. `snapshot_json` (`lib.rs:191-193`) then serializes the whole graph.
   **Required:** a retention/compaction policy (ring buffer or time-windowed rollup of
   SemanticStates). Determinism caveat: eviction changes snapshot *contents* (a product
   decision), not float math — the per-cycle witness (`lib.rs:437-448`) is unaffected.
2. **Per-cycle allocation churn** (F1, F3, F5, F8, F10, F12): at 20 Hz this is dozens of
   short-lived heap allocations per cycle. On a Pi 5 this is allocator pressure and
   cache pollution rather than RSS growth; on WASM (bump-ish dlmalloc, no MADV_FREE) it
   inflates the linear memory high-water mark, which is never returned to the host.
3. **WASM targets.** `wifi-densepose-wasm` is a browser binding crate (JS interop,
   serde, chrono — `crates/wifi-densepose-wasm/Cargo.toml`) and pulls `wifi-densepose-mat`
   optionally; it relies on `wasm-opt -O4` (`Cargo.toml` `[package.metadata.wasm-pack]`).
   `wifi-densepose-wasm-edge` is the disciplined one: `no_std` + `libm`, its own profile
   `opt-level = "s"`, lto, cgu=1 (`crates/wifi-densepose-wasm-edge/Cargo.toml`). Neither
   enables `+simd128` (§5). If the CIR estimator is ever compiled to wasm-edge, HE40's
   11.2 MB of sensing matrix alone is ~700 pages of linear memory — restrict edge WASM
   to HT20 (130 KB) or ship F4/F7 first.

---

## 5. Build-profile review & recommendations

Current release profile (`v2/Cargo.toml:213-218`) is already aggressive and correct:
`opt-level = 3`, `lto = true` (fat), `codegen-units = 1`, `panic = "abort"`,
`strip = true`; `bench` inherits release with debug symbols (`v2/Cargo.toml:225-227`).
There is nothing wrong to fix here — the gains left are target- and feedback-driven:

1. **Per-target CPU tuning (EXACT, do first).** No `target-cpu` is set anywhere. For
   Pi 5 fleet builds: `RUSTFLAGS="-C target-cpu=cortex-a76"` — enables NEON scheduling
   and `vcnt` for the sketch path (F17) without changing IEEE semantics. LLVM does not
   reassociate float reductions or contract to FMA without explicit fast-math/contract
   flags, so scalar float results stay bit-exact. **Verify with the existing proof
   runners** (`cir_proof_runner`, `calibration_proof_runner`,
   `signal/Cargo.toml`) as the acceptance gate — that is exactly what they exist for.
2. **WASM SIMD.** Add `-C target-feature=+simd128` for `wifi-densepose-wasm` builds and
   keep a non-SIMD artifact for older runtimes. Same determinism note as above; gate
   with the proof runners compiled to wasm where feasible.
3. **PGO: feasible and determinism-safe.** PGO changes inlining/layout, never FP
   semantics. The repo already has ideal deterministic training workloads: the proof
   runner binaries plus `engine_cycle` / `cir_bench`. Pipeline: `cargo pgo build` →
   run proof runners + benches → `cargo pgo optimize`. Expect mid-single-digit to ~15%
   on branchy paths (gate decisions, tracker lifecycle); the dense ISTA loop will see
   little. Cost: CI complexity. Verdict: do it after F1–F12, not before.
4. **Do not** enable `-ffast-math`-equivalents (`fadd_fast`, `core::intrinsics`,
   `-C llvm-args=-fp-contract=fast`) anywhere in the witness path. This must be a
   stated rule in CONTRIBUTING/ADR, not tribal knowledge.
5. **BOLT / `opt-level` experiments are not worth it** ahead of F4; the pipeline is
   FLOP-bound in one loop, not front-end bound.

---

## 6. Prioritized 90-day plan

### Phase 0 — Measure (days 1–10)
- Run and commit criterion baselines on a Pi 5 and an x86 dev box:
  `engine_cycle`, `cir_bench` (all four tiers), `sketch_bench`, `signal_bench`,
  `calibration_bench`. The 50 ms claim in `engine_cycle.rs:3` becomes a measured number.
- Add a lightweight per-stage timing histogram (feature-gated, off in witness builds) at
  the §2 stage boundaries; wire a CI perf-regression gate (±10%) on the committed
  baselines.
- Re-run the soak that produced `benchmark_baseline.json` and pin it as the accuracy
  guardrail for everything below.

### Phase 1 — Exact, zero-risk wins (days 10–35)
All EXACT findings; no witness impact; each lands with proof-runner verification:
- F5 (precompute warm-start diag/CSR in `CirEstimator::new`) — biggest exact CIR win.
- F8 (hoist tomography gradient buffer), F6, F10, F12, F1, F2 (allocation/duplication
  removal), F15 + F11 with pinned index tie-breaks.
- WorldGraph retention policy (the §4.1 unbounded-growth fix) — design ADR + ring-buffer
  implementation.
- Expected outcome: measurable cycle-time reduction and flat memory under 24 h soak;
  **identical witness hashes**.

### Phase 2 — Determinism-managed structural wins (days 35–70)
Each behind a feature flag, A/B'd against the legacy path (the `use_cir_gate` A/B switch
at `multistatic.rs:103` is the template), with proof-hash regeneration as an explicit,
witnessed release event:
- **F4: FFT-based Φ/Φᴴ application in ISTA** — the headline item; the only route to
  HE20/HE40 real-time and the 12-link pattern. Acceptance: cir_bench speedup ≥ 5× at
  HE20, soak metrics within guardrail, new `expected_features.sha256` published in a
  fresh witness bundle.
- F9 (power-method Lipschitz in tomography) riding the same hash-regen train.
- F3 (slice-based CIR entry point), choosing the exact-f64-intermediate variant if the
  hash train slips.
- F14: feature-gated `rayon` across **links/nodes only**, deterministic index-ordered
  collection; CI must run the determinism test (`engine/src/lib.rs:535-548`
  `cycle_is_deterministic`) with the feature on.

### Phase 3 — Platform & toolchain (days 70–90)
- Pi 5 `target-cpu=cortex-a76` fleet builds + proof-runner verification (§5.1).
- `+simd128` WASM artifact + size budget check for wasm-edge (§5.2, §4.3).
- PGO pilot in CI using proof runners as the training corpus (§5.3).
- Re-baseline: new criterion numbers, refreshed witness bundle, updated this document's
  §1 with real measured latencies.

**Out of 90-day scope, flagged for the architecture backlog:** F16 (Complex64→Complex32
in core), F7 (single-matrix Φ kernel — bundle with F4), and HE40-on-edge (blocked on
F4+F7).

---

## 7. Summary

The pipeline's only structural latency hazard is the dense ISTA CIR solver
(`cir.rs:601-654` + `cir.rs:717-745`): fine at HT20, ~1.1 GFLOP worst-case per estimate
at HE40, and slated to run per-link (×12). Everything else is allocation churn and
duplicated work that can be removed with **bit-exact** refactors (F1–F12), plus one
genuine memory bug-class issue: unbounded WorldGraph growth at 20 Hz
(`engine/src/lib.rs:346-352`). The build profile is already optimal; remaining toolchain
gains (target-cpu, wasm simd128, PGO) are determinism-safe and cheap. The determinism
constraint is workable because the repo already owns the right tools — deterministic
proof runners, an A/B gate pattern, and a per-cycle witness — so float-changing
optimizations become scheduled, witnessed hash-regeneration events rather than risks.
