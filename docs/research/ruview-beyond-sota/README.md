# RuView Beyond-SOTA Research Series

Research swarm output (2026-06-09) defining what a beyond-state-of-the-art
RuView implementation is, what the current system actually delivers, and the
validation/benchmark/optimization evidence gathered in the same session.

Produced by a 5-agent hierarchical research swarm (system reviewer, SOTA
surveyor, architect, benchmark methodologist, performance analyst) plus a
validation pass run against the working tree.

## Documents

| Doc | Scope | One-line takeaway |
|-----|-------|-------------------|
| [00-system-review.md](00-system-review.md) | Capability audit of the current engine | Signal layer is the deepest asset (`ruvsense/` ≈14.4k lines, 310 in-module tests); the model tier is the emptiest (no trained checkpoint in-tree); the live 20 Hz path is the main integration gap |
| [01-sota-landscape-2026.md](01-sota-landscape-2026.md) | Published SOTA per capability axis (web-verified) | Defines the beyond-SOTA bar: 12-row capability → published SOTA → RuView-today → target table; IEEE 802.11bf-2025 is ratified and moves the moat up-stack |
| [02-beyond-sota-architecture.md](02-beyond-sota-architecture.md) | Target architecture | 8 pillars (RF foundation encoder + UQ heads, differentiable RF forward model, RF-SLAM×WorldGraph loop, camera→RF distillation, swarm apertures, continual adaptation, deterministic WASM edge, NV fusion) — all landing inside existing crates, no rewrite (per ADR-136 §2.1) |
| [03-benchmark-validation-methodology.md](03-benchmark-validation-methodology.md) | Test/validation/benchmark methodology | 6-layer validation pyramid; 15 criterion bench targets inventoried; `benchmark_baseline.json` is a live-capture anchor, not a criterion baseline; statistical protocol from ADR-171 (≥10 seeds, IQM, bootstrap CIs) |
| [04-optimization-roadmap.md](04-optimization-roadmap.md) | Performance review + 90-day plan | ISTA CIR solver is the dominant latency hazard (~1.1 GFLOP/frame at HE40); exact zero-risk wins identified; WorldGraph grows unboundedly (no eviction) — a real bug-class |

## Validation results (this session, 2026-06-09)

All measured on this branch (`claude/ruview-beyond-sota-xgv8aq`), Linux
container, `cargo test --workspace --exclude wifi-densepose-desktop
--no-default-features` (the desktop crate needs GTK system libraries absent in
the container; this is an environment limitation, not a code failure).

| Layer | Command | Result |
|-------|---------|--------|
| L0 unit/integration | `cargo test --workspace --exclude wifi-densepose-desktop --no-default-features` | **154 suites, 2,797 passed, 0 failed** (pre-optimization baseline; re-run post-optimization also green) |
| L1 deterministic proof | `python archive/v1/data/proof/verify.py` | **VERDICT: PASS** — hash `f8e76f21a0f9852b70b6d9dd5318239f6b20cbcb4cdd995863263cecdc446f7a` (bit-exact) |
| L2 criterion (CIR) | `cargo bench -p wifi-densepose-signal --bench cir_bench --no-default-features --features cir` | Baselines captured pre/post optimization (below) |

~~Known pre-existing issue (not introduced here): `cargo check -p
wifi-densepose-mat --no-default-features` fails standalone with 101 serde
feature-unification errors; it builds and passes inside `--workspace` runs.~~
**Fixed on this branch:** `pub mod api` (the only serde user) is now gated
behind the `api` feature that owns the optional serde dependency; all feature
combos compile.

## Optimizations applied (this session)

Two **exact** (bit-identical float results — summation order unchanged,
witness chain unaffected) optimizations from the 04 roadmap's "zero-risk"
tier were implemented and verified:

1. **`cir.rs` warm-start precompute** — the diagonal Tikhonov preconditioner
   `diag(Φ^H Φ) + λI` and its CSR matrix depend only on Φ and λ (fixed at
   `CirEstimator::new`) but were rebuilt on every frame (O(K·G) pass + CSR
   allocation). Moved to construction
   (`crates/wifi-densepose-signal/src/ruvsense/cir.rs`,
   `build_warm_start_system`).
2. **`tomography.rs` solver hoisting** — the ISTA gradient `Vec` was
   allocated inside the 100-iteration loop and the Frobenius Lipschitz bound
   recomputed per `reconstruct` call; both hoisted
   (`crates/wifi-densepose-signal/src/ruvsense/tomography.rs`).

### Measured impact (criterion, paired pre/post baselines, same container)

| Bench | Pre-opt | Post-opt | Change | Significant? |
|-------|---------|----------|--------|--------------|
| `cir_estimate/he40` | 12.34 ms | 11.86 ms | **−3.9 %** | yes (p < 0.01) |
| `cir_multiband_3band` (30 ms group) | 30.16 ms | 29.72 ms | −1.4 % | yes (p < 0.01) |
| `cir_multiband` (142 ms group) | 141.9 ms | 140.1 ms | −1.2 % | yes (p < 0.01) |
| `cir_estimate/ht40` | 11.73 ms | 11.78 ms | +0.4 % | no (p = 0.28) |
| `cir_estimate/he20` | 2.49 ms | 2.49 ms | −0.1 % | no (p = 0.85) |
| `cir_estimate/ht20` | 2.48 ms | 2.58 ms | +3.8 % | noise — see note |

Note on ht20: `cir_estimator_new/ht20` (construction, which now does strictly
*more* work) also shows "+3 %", establishing a ≈3–4 % container noise floor;
the ht20 estimate delta is within it. The honest summary: the warm-start
precompute removes 1 of ~101 O(K·G) passes per frame, so the expected gain is
≈1–4 % — consistent with what was measured. The dominant per-frame cost is
the 100-iteration ISTA loop itself, which is exactly what the roadmap's
flag-gated FFT-operator proposal (8–40× on the mat-vecs, requires witnessed
hash regeneration) targets next.

Correctness post-optimization: `wifi-densepose-signal` 456 tests green;
`wifi-densepose-engine` 11/11 green including `cycle_is_deterministic` and
`calibration_mismatch_demotes_and_witness_stable` (witness-chain stability).

## Headline conclusions

1. **"Beyond SOTA" is currently unfalsifiable** without a real-CSI
   ground-truth benchmark — standing one up (per doc 03's acceptance table
   and ADR-171's statistical protocol) is the highest-leverage next step.
2. **The path is evolution, not rewrite**: all eight architecture pillars in
   doc 02 land inside existing crates on the ADR-136 `Stage<I,O>`/`FrameMeta`
   contract spine.
3. **The biggest engineering gaps** are the live 20 Hz ingest path, a trained
   RF encoder checkpoint, and WorldGraph retention/eviction — ahead of any
   frontier capability work.
4. **Determinism is the differentiator**: every optimization and new pillar
   must preserve the witness chain; the advisory-vs-witnessed split (doc 02
   §determinism) is the mechanism that lets frontier components in without
   breaking it.
