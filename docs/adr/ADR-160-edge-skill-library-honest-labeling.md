# ADR-160: Edge Skill Library (`wifi-densepose-wasm-edge`) — Honest Labeling & Soundness Cleanup

- **Status**: accepted
- **Date**: 2026-06-11
- **Deciders**: ruv
- **Tags**: wasm-edge, esp32, edge-skills, claim-surface, medical-overclaim, affect, prove-everything, soundness, static-mut
- **Amends**: ADR-159 (deferred-backlog line for wasm-edge now TRUE)

## Context

Beyond-SOTA sweep Milestone 6, over `v2/crates/wifi-densepose-wasm-edge` only,
executed under the project's **prove-everything / anti-"AI-slop"** directive.

### Headline — 0 stubs, 0 theater, all real DSP (REFUTES the slop accusation)

A read-only audit found this crate has **zero stubs and zero fake-output theater:
every one of the ~70 edge skills runs real DSP** (Welford statistics,
autocorrelation, DTW, sliced-Wasserstein, ISTA-style recovery, Kalman/HNSW, etc.).
The forward paths are genuine signal processing on real CSI-derived inputs. That
is the anti-slop win and it is cited here as a positive, not a fabrication.

What the audit correctly found was **not fake code but an over-confident claim
surface**: skill *names* and doc-comments asserting clinical/affective/security
capabilities that the **unvalidated** code cannot back, concentrated in the
medical (`med_*`) and affect (`exo_happiness`/`exo_emotion`) skills. The fix is
**honest labeling — making the labels TRUE — NOT making the claimed capability
real.** You cannot validate seizure detection, affect inference, or weapon
discrimination without clinical/labelled data and reference standards; this ADR
does not pretend to. It disclaims, renames, softens, and feature-gates so the
surface matches what the DSP actually delivers.

Grading vocabulary follows ADR-152 / ADR-158 / ADR-159:
- **MEASURED** — reproduced in this worktree, command + failing-on-old test recorded.
- **DATA-GATED** — real code path present; honestly flagged where data is absent.
- **NO-ACTION (already-honest)** — audited, found correct, cited as a positive.
- **ACCEPTED-FUTURE** — deliberately deferred, nothing dropped.

## Per-prefix classification

| Prefix | Class | Note |
|--------|-------|------|
| `sig_*` (signal intelligence) | **REAL-DSP, honest** | Algorithm-named (flash-attention, sparse-recovery, optimal-transport, temporal-compress, mincut). Names describe the math, not an overclaimed outcome. NO-ACTION on labels; A5 soundness applied. |
| `lrn_*` (adaptive learning) | **REAL-DSP, honest** | DTW/EWC/meta-adapt/attractor — algorithm-named. NO-ACTION on labels; A5 applied. |
| `spt_*` / `tmp_*` | **REAL-DSP, honest** | PageRank/HNSW/spiking-tracker; LTL-guard/GOAP/pattern-sequence. Algorithm-named. NO-ACTION on labels; A5 applied. |
| `qnt_*` | **REAL-DSP, honest (disclosed analogy)** | "quantum-**inspired**" / Grover-**inspired** are already disclosed analogies. NO-ACTION (DO-NOT-touch); A5 applied (mechanical, no label/behavior change). |
| `bld_*` / `ret_*` / `ind_*` / `occupancy`/`intrusion` | **REAL-DSP, honest** | Occupancy/queue/forklift/clean-room etc. describe physical observables. NO-ACTION on labels; A5 applied. |
| `sec_weapon_detect` | **REAL-DSP, overclaiming NAME** → fixed (A3) | Variance-ratio reflectivity renamed off "weapon". |
| `med_*` (5) | **REAL-DSP, overclaiming NAME/DOC** → fixed (A1) | Clinical detection asserted as fact; now disclaimed + softened + feature-gated. |
| `exo_happiness` / `exo_emotion` | **REAL-DSP, overclaiming NAME/DOC** → fixed (A2) | Affect outputs reframed as proxies; uncited stat removed. |
| `exo_dream_stage` / `exo_gesture_language` | **REAL-DSP, quasi-medical/over-named** → fixed (A4) | Disclaimers added; Research tag promoted to header. |
| `exo_time_crystal` / `exo_ghost_hunter` | **REAL-DSP, honest novelty** | Disclosed exploratory/novelty skills. NO-ACTION (DO-NOT-touch); A5 applied. |
| `nvsim` | out of scope | Disclaimer gold standard; copied its tone. |

## Decision — Fixes Landed

### §A1 Medical overclaim (HIGH) — MEASURED

The five `med_*` modules (`med_seizure_detect`, `med_cardiac_arrhythmia`,
`med_respiratory_distress`, `med_sleep_apnea`, `med_gait_analysis`) stated clinical
detection as fact with no disclaimer ("Detects tonic-clonic seizures…").

**Real fix (honest labeling — the DSP is kept, untouched):**
- **(a)** Every module's `//!` header now carries a mandatory disclaimer block,
  modelled on `sec_weapon_detect.rs` and `nvsim/src/lib.rs`: *"EXPERIMENTAL
  RESEARCH MODULE — NOT VALIDATED AGAINST CLINICAL DATA. NOT A MEDICAL DEVICE.
  Flags candidate <X>-like signatures only,"* citing ADR-160.
- **(b)** Doc verbs softened: *"Detects tonic-clonic seizures"* →
  *"Flags candidate tonic-clonic-seizure-like motion signatures (experimental)"*;
  similarly for cardiac/respiratory/apnea/gait.
- **(c)** All five gated behind a new **non-default** cargo feature
  `medical-experimental` (`#[cfg(feature = "medical-experimental")]` in `lib.rs`,
  `medical-experimental = []` in `Cargo.toml`, **not** in `default`) so they cannot
  be silently built into a shipping artifact.

**Failing-on-old tests** (`tests/honest_labeling.rs`):
`a1_med_modules_have_clinical_disclaimer`,
`a1_med_modules_gated_behind_medical_experimental`,
`a1_seizure_verbs_softened`. All fail on the old, undisclaimed, ungated source.
**Grade: MEASURED (label); per-skill clinical accuracy DATA-GATED.**

### §A2 Affect overclaim (HIGH) — MEASURED

`exo_happiness_score.rs` carried an **uncited** "Happy people walk ~12% faster"
statistic and emits `HAPPINESS_SCORE`; `exo_emotion_detect.rs` emits
`STRESS_INDEX`/`CALM_DETECTED`/`AGITATION_DETECTED`.

**Real fix (honest labeling — math kept):**
- Deleted the uncited "12% faster" / "~12% above" / "Happy people walk" statements.
- Added a prominent *"speculative, unvalidated affect heuristic; outputs are NOT
  measurements of emotion"* disclaimer to both `//!` headers, citing ADR-160.
- Reframed `HAPPINESS_SCORE` in the docs as a **"gait-energy proxy, not a validated
  affect measure."**

**Failing-on-old tests:** `a2_affect_modules_have_unvalidated_disclaimer`,
`a2_uncited_12_percent_stat_removed`, `a2_happiness_reframed_as_proxy`.
**Grade: MEASURED (label); affect validity DATA-GATED.**

### §A3 Security event-name overclaim (MEDIUM) — MEASURED

`sec_weapon_detect.rs`'s module doc was already honest (research-grade,
calibration-required), but the event/const names claimed weapon-grade
discrimination a variance ratio cannot deliver.

**Real fix (honest physical-quantity naming — behavior unchanged):**
- `EVENT_WEAPON_ALERT` → `EVENT_HIGH_METAL_REFLECTIVITY` (event id 221 unchanged).
- `WEAPON_RATIO_THRESH` → `HIGH_REFLECTIVITY_THRESH`.
- Internal fields/consts renamed (`weapon_run`→`high_refl_run`,
  `cd_weapon`→`cd_high_refl`, `WEAPON_DEBOUNCE`→`HIGH_REFLECTIVITY_DEBOUNCE`).
- `lib.rs` `event_types` registry: `WEAPON_ALERT` → `HIGH_METAL_REFLECTIVITY`.
- A reflectivity-vs-weapons honest-naming note added to the header.
The detector still flags a high amplitude-variance/phase-variance ratio (real RF
reflectivity); it just no longer *names* that "weapon".

**Failing-on-old tests:** `a3_weapon_names_renamed_to_reflectivity`,
`a3_registry_no_longer_exports_weapon_alert` (registry no longer exports a
`WEAPON_ALERT` name). **Grade: MEASURED.**

### §A4 Quasi-medical / sign-language exotic modules (MEDIUM) — MEASURED

`exo_dream_stage.rs` ("sleep stage classification", quasi-medical) and
`exo_gesture_language.rs` ("sign language letter recognition").

**Real fix (honest labeling — DSP kept):** added an experimental "NOT VALIDATED"
disclaimer to each `//!` header (citing ADR-160) and promoted the
**Exotic/Research** registry tag into the header where a reader sees it.
`exo_gesture_language` additionally states it is a coarse gesture-cluster
classifier that **does not recognize true sign language** (never evaluated on a
labelled ASL set).

**Failing-on-old test:** `a4_exotic_modules_have_experimental_disclaimer`.
**Grade: MEASURED (label); accuracy DATA-GATED.**

### §A5 `static mut` event-buffer soundness (MEDIUM) — the one real code fix — MEASURED

~61 per-call event scratch buffers across the crate used a module-level
`static mut EVENTS: [(i32,f32); N]` (a handful named `EV`/`TE`/`EMPTY`) and returned
`&EVENTS[..n]`. On a `cdylib`+`rlib` linkable into multithreaded/reentrant host
code this is latent aliasing UB, and `static_mut_refs` is deny-by-default on newer
Rust.

**Real fix (mechanical, behavior-preserving):** moved each scratch buffer off
`static mut` into an **owned per-instance field** (`events: [(i32,f32); N]` on the
detector struct, written via `&mut self` and returned as `&self.events[..n]`). The
public `-> &[(i32, f32)]` signature is **unchanged**, so no caller (in-module
tests, `ghost_hunter` bin, `budget_compliance`) needed editing. Two helper methods
that built events under `&self` (`spt_pagerank_influence::build_events`,
`spt_spiking_tracker::build_events`) and `sig_temporal_compress::on_timer` were
promoted to `&mut self`. Leftover now-redundant `unsafe { }` wrappers were removed.

**Count: 61 scratch buffers across 60 module files fixed** (the only `static mut`
left in `src/` are the two **legitimate WASM module singletons** — `lib.rs STATE`
and `bin/ghost_hunter.rs DETECTOR` — `#[cfg(target_arch="wasm32")]`,
`#[no_mangle]`, accessed via `core::ptr::addr_of_mut!`, single-threaded by the
wasm runtime contract; these are *not* the aliasing-UB scratch pattern and are
left as-is).

**Verification:** the full host build (`--features std` and
`std,medical-experimental`) compiles with **0 warnings** — there is no longer any
`static mut <name>` + `&<name>` source for `static_mut_refs` to fire on in the 60
fixed modules. (The pure-`wasm32-unknown-unknown` build, where the lint is
deny-by-default, could not be run in this worktree because the `wasm32` target is
not installed on the build toolchain; the source-level elimination is the
evidence, asserted per-module by `a5_claim_bearing_modules_have_no_static_mut_event_buffer`.)
**Grade: MEASURED (source-eliminated; residual = 2 legitimate singletons).**

## Negative Results (NO-ACTION positives — cited, not edited for labels)

Audited and found genuinely honest; cited as positives:
- **`qnt_quantum_coherence.rs`** — discloses "quantum-**inspired**" analogy.
- **`exo_time_crystal.rs`**, **`exo_ghost_hunter.rs`** — disclosed exploratory/novelty.
- **`qnt_interference_search.rs`** — disclosed "Grover-**inspired**".
- **`sig_*` / `lrn_*`** algorithm-named skills — names describe the DSP, not an outcome.
- **`nvsim`** — out of scope; the project's disclaimer gold standard (its tone was
  copied into the A1/A2/A4 disclaimers).

(These were A5-soundness-fixed mechanically where they used `static mut`, with no
label or behavior change, consistent with leaving their claim surface intact.)

## Deferred Backlog (Nothing Dropped)

- **Per-skill accuracy validation** — **PARTIALLY MEASURED-on-synthetic**
  (2026-06-13). For the subset of skills whose detection target is *constructible*
  with known ground truth, a synthetic-ground-truth harness
  (`tests/synthetic_validation.rs`, 12 tests) plants signals with known answers,
  runs the real detector, and **measures** detection accuracy / rate-error:
  `vital_trend`, `exo_time_crystal` (periodic-vs-aperiodic — its sub-harmonic-vs-
  clean-period claim is NOT separable, recorded honestly), `exo_ghost_hunter`
  (hidden breathing), `occupancy`, `intrusion`, `exo_rain_detect`,
  `sig_flash_attention` (8/8 peak localization), `spt_spiking_tracker` (4/4 zone
  localization, sparse plant), `sig_optimal_transport`, `sig_mincut_person_match`
  (0 id-swaps), `lrn_dtw_gesture_learn` (enrollment) — all 1.000 where claimed;
  `sig_sparse_recovery`'s recovery accuracy is reported **negative** (−2.2% vs
  unrecovered baseline) — only its trigger path is validated. Full numbers +
  reproduce commands in `benchmarks/edge-skills/RESULTS.md`.
  The **med_*/affect/sign-language/weapon** claims remain **DATA-GATED**:
  validating them requires labelled clinical/affective/ASL/metal-object data and
  reference standards that do not exist in this repo. Planting a "seizure-/weapon-/
  happy-like" synthetic signal validates nothing real and is explicitly refused;
  RESULTS.md lists each with the real data it needs. The disclaimers + feature gate
  are the honest stand-in. Nothing is claimed that is not measured.
- **Unified edge pipeline** — **MEASURED** (2026-06-13). `src/pipeline_all.rs`
  (`EdgePipeline`) + `src/skill_registry.rs` register **every** runtime skill
  behind one uniform `EdgeSkill` trait and run them all per CSI frame; `med_*` are
  registered only under `--features medical-experimental` (preserves the §A1 gate).
  `tests/pipeline_all.rs` (4 tests) proves all 59 default / 64 medical skills run
  without panic over 300 synthetic frames with a well-formed aggregated event
  stream. `examples/run_all_skills.rs` is a runnable demo. No skill DSP changed.
- **Criterion benches for `process_frame` budget claims** — **DONE (host)**
  (ADR-163, 2026-06-12). `benches/process_frame_bench.rs` benches the heaviest
  hot paths (`exo_time_crystal` 256×128 autocorrelation, `exo_ghost_hunter`
  periodicity, `sec_weapon_detect` per-subcarrier Welford, `med_seizure_detect`
  clonic rhythm) and reports committed **host** medians
  (`benchmarks/edge-latency/RESULTS.md`). `tests/budget_compliance.rs` continues
  to assert the L/S/H tier wall-clock budgets (25 tests, passing). **ESP32-on-
  hardware (Xtensa/WASM3) latency remains PENDING** — the host bench is an
  upper-bound algorithm-cost proxy, NOT the ESP32 figure (needs hardware).
- **`wasm32-unknown-unknown` `static_mut_refs` confirmation** — **ACCEPTED-FUTURE**
  (toolchain): the source pattern is eliminated; a CI job on the wasm target should
  assert zero `static_mut_refs` once the target is added to the build image.
- **The 2 residual `static mut` singletons** (`lib.rs STATE`, `ghost_hunter DETECTOR`)
  — **ACCEPTED-FUTURE**: these are the canonical wasm module-state pattern; migrating
  them to a safe cell is a separate, larger change with no current UB (single-threaded
  wasm runtime, `addr_of_mut!` access).

## Reproduction (MEASURED)

```bash
cd v2/crates/wifi-densepose-wasm-edge   # excluded from the v2 workspace; build here
cargo test --features std                          # default
cargo test --features std,medical-experimental     # med_* skills enabled
cargo test --no-default-features --features std     # no default-pipeline
cargo test --features std --test honest_labeling   # A1–A5 label invariants
```

(`std` is required for host tests — the crate is `no_std` for `wasm32`; pure
`--no-default-features` builds only on `wasm32-unknown-unknown`, where it
intentionally has no panic handler on the host.)

Result at time of writing (all 0 failed):
- **DEFAULT** (`--features std`) — **615 passed** (lib 504; budget 25; honest_labeling 10; bench 1; vendor 75)
- **MEDICAL** (`--features std,medical-experimental`) — **653 passed** (lib 542; +38 med_* tests; others unchanged)
- **NO-DEFAULT** (`--no-default-features --features std`) — **615 passed**
- Full host build emits **0 warnings**; **61** `static mut` scratch buffers eliminated, **2** legitimate wasm singletons remain.

## Consequences

- No edge skill's name or doc-comment claims a clinical, affective, security, or
  sign-language capability the unvalidated DSP cannot back.
- The five medical skills cannot be silently compiled into a shipping artifact
  (non-default `medical-experimental` gate).
- The security skill can never emit a "weapon alert" — it reports
  `HIGH_METAL_REFLECTIVITY`, the physical quantity it actually measures.
- The latent `static mut` aliasing-UB / `static_mut_refs` exposure is removed from
  60 modules; the public API and all runtime behavior are unchanged (615/653 tests
  prove behavior preservation).
- ADR-159's deferred-backlog statement *"wasm-edge … honestly labelled, not
  claimed"* is now actually TRUE.
