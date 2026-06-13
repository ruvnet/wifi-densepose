# RuView System Review — Capability Audit (Beyond-SOTA Series, Doc 00)

**Date:** 2026-06-09
**Scope:** The RuView product surface (ADR-031) and the 38-crate Rust workspace under `v2/crates/` that implements it, plus the ADR corpus (`docs/adr/`, 150 numbered ADRs) and the prior research corpus (`docs/research/sota-2026-05-22/`).
**Method:** Direct reads of `lib.rs`/`mod.rs` and key ADRs; static test counts via `grep -r '#[test]'` / `#[tokio::test]` per crate (counts are *static occurrences in source*, not CI pass counts). No metrics in this document are estimated — everything cited was read or measured in the working tree.

---

## 1. Executive Summary — What RuView IS Today

RuView is **not a crate**. Per ADR-136 §2.1 (`docs/adr/ADR-136-ruview-streaming-engine-frame-contracts.md`), RuView is the sensing-first *product surface and brand* (ADR-031, status: Proposed) layered on the existing `wifi-densepose-*` / `homecore*` / `cog-*` workspace. ADR-136 explicitly **rejects** a `ruview_*` crate rename and pins a normative ten-role mapping (ingest / signal / fusion / world / models / privacy / store / api / eval / observe) onto the existing crates.

What concretely exists:

1. **A deep, heavily-tested signal-processing layer.** `wifi-densepose-signal` contains 473 static `#[test]` occurrences, including a 22-file `ruvsense/` bounded context (`v2/crates/wifi-densepose-signal/src/ruvsense/`) implementing the ADR-029 six-stage multistatic pipeline plus ADR-030/032a/134/135/137/138/142/143 extensions (~14,000 lines, 330 in-module tests measured by per-file grep).
2. **A trust-traceable composition root.** `wifi-densepose-engine` (`src/lib.rs`, 752 lines, 11 tests) wires fusion quality (ADR-137), array coordination (ADR-138), evolution change-points (ADR-142), RF-SLAM anchors (ADR-143), the WorldGraph (ADR-139), and the BFLD privacy control plane (ADR-141) into one `StreamingEngine::process_cycle` (`lib.rs:285`) that emits a `TrustedOutput` (`lib.rs:80`) carrying evidence + model version + calibration version + privacy decision + a BLAKE3 witness (`lib.rs:437`).
3. **A privacy layer with structural invariants.** `wifi-densepose-bfld` (20 modules, 369 tests) implements ADR-118–123/141: raw BFI never exits the node (I1), identity embeddings are RAM-only (I2), cross-site identity correlation is cryptographically impossible (I3) — stated at `wifi-densepose-bfld/src/lib.rs:7-11`.
4. **A Home-Assistant-class world/state layer.** `homecore` + 9 sibling crates (state machine, event bus, plugins, automation, REST/WS API, recorder, HAP bridge, assist) — explicitly a "P1 scaffold" per `homecore/src/lib.rs:7` with deferred items listed at `lib.rs:24-31`.
5. **A drone-swarm extension.** `ruview-swarm` (17 modules, ~9,000 lines in subdirectories, 115 + 19 async tests), ADR-148 self-reports ~98% complete with the remaining 15% of M3 gated on real ESP32-S3 hardware (`ADR-148:940-953`).
6. **A large prior research corpus.** The 2026-05-22 autonomous SOTA loop: 41 ticks, 19 research threads (R1–R20), 22 numpy reference implementations, 7 ADRs, and a 6-tier production roadmap (`docs/research/sota-2026-05-22/00-summary.md`, `PRODUCTION-ROADMAP.md`).

The critical caveat, stated by the project itself: the ADR-136–146 series is *"a skeleton and nervous system, not a shipping product… Most of the series is not yet wired into the live 20 Hz pipeline"* (ADR-136 §8). The engine crate's own docs confirm what is absent: *"the live 20 Hz I/O loop (sensing-server), UWB hardware (ADR-144), and model training (ADR-146)"* (`wifi-densepose-engine/src/lib.rs:27-29`).

---

## 2. Capability Matrix — Pipeline Role → Crates → Maturity

Role mapping is normative per ADR-136 §2.1; maturity is this review's judgment from code + ADR status. Test counts: static `#[test]` + `#[tokio::test]` greps (2026-06-09).

| Role | Crate(s) | Key modules | Tests (sync+async) | Maturity | Evidence |
|---|---|---|---|---|---|
| **ingest** | `wifi-densepose-sensing-server`, `wifi-densepose-hardware`, `wifi-densepose-wifiscan` | `csi.rs`, `multistatic_bridge.rs`, `tracker_bridge.rs`, ESP32 TDM | 557+14, 137, 150 | **Production** (hardware-validated per ADR-028/039) | `sensing-server/src/` has 30+ modules incl. MQTT, Matter, RVF pipeline |
| **signal** | `wifi-densepose-signal` (incl. `ruvsense/`) | 6-stage pipeline (`ruvsense/mod.rs:9-23`), `cir.rs`, `calibration.rs`, `hampel.rs`, `fresnel.rs`, `phase_sanitizer.rs` | 473 | **Production** (unit level); live multistatic wiring **beta** | §3 below; ADR-014 Accepted, ADR-029 Proposed |
| **fusion** | `ruvsense/multistatic.rs`, `ruvsense/fusion_quality.rs`, `wifi-densepose-ruvector/src/viewpoint/` | `MultistaticFuser`, `QualityScore`, `CrossViewpointAttention`, GDI/Cramér-Rao (`viewpoint/geometry.rs`) | 20 (multistatic.rs), 3 (fusion_quality.rs), 136 (ruvector crate) | **Beta** — tested building blocks, composed only in `wifi-densepose-engine` tests | `viewpoint/mod.rs:1-30`; engine `lib.rs:317-319` |
| **world** | `homecore`, `wifi-densepose-worldgraph`, `wifi-densepose-geo`, `wifi-densepose-worldmodel` | `StateMachine`, `EventBus`, `WorldGraph` (rooms/sensors/person-tracks/semantic states), ENU geo registration | 9+11, 7, 16+1, 12+1 | **Beta** — homecore is explicit "P1 scaffold"; persistence/service dispatch deferred to P2 | `homecore/src/lib.rs:7, 24-31`; ADR-127 Proposed |
| **models** | `cog-pose-estimation`, `cog-person-count`, `wifi-densepose-nn`, `wifi-densepose-train`, `wifi-densepose-occworld-candle` | ONNX/Candle inference, training pipeline, OccWorld bridge | 7, 15, 30+1, 312, 12 | **Experimental** — no trained RF foundation encoder exists; ADR-147 benchmarked OccWorld with **random weights** | `ADR-168-benchmark-proof.md` ("random weights — pre-domain-fine-tuning baseline"); ADR-146/150 Proposed |
| **privacy** | `wifi-densepose-bfld` | `privacy_gate.rs`, `privacy_mode.rs` (mode registry + hash-chained attestation), `identity_risk.rs`, `signature_hasher.rs`, `embedding_ring.rs` | 369 | **Beta** — strongest-tested layer, but lib header still says "Status: P1 in progress" (`lib.rs:12`, stale vs 20 implemented modules) | ADR-118–123, 141 all Proposed |
| **store** | `homecore-recorder` | trajectory/event recording | 8+12 | **Experimental** | ADR-136 §2.1 |
| **api** | `homecore-api`, `homecore-server`, `cog-ha-matter`, `homecore-hap` | REST/WS, HA discovery, Matter, HomeKit | 7+11, 0, 63+1, 15+2 | **Experimental→Beta** (`homecore-server` has zero tests) | ADR-130/125/115 Proposed |
| **eval** | `wifi-densepose-train/src/ablation.rs`, `ruview-swarm/src/evals/` | ablation harness (ADR-145), swarm eval suite (ADR-171) | included in 312 / 115 | **Experimental** — ADR-145 self-labels "skeleton/scaffolding, mostly not yet on the live 20 Hz path" | `ablation.rs` exists; ADR-171 (swarm benchmarking, renumbered from ADR-149) Accepted |
| **observe** | `homecore-automation`, `homecore-assist` | automation engine, assistant/Ruflo bridge | 20+14, 3+20 | **Experimental** | ADR-129/133 Proposed |
| **(integration root)** | `wifi-densepose-engine` | `StreamingEngine`, `TrustedOutput`, privacy demotion, witness | 11 | **Beta** — the only crate that proves cross-role composition; not on a live I/O path | `engine/src/lib.rs:1-29, 457-751` |
| **(swarm)** | `ruview-swarm` | Raft/gossip topology, RRT-APF planning, Candle PPO MARL, CSI sensing payload, failsafe, Ruflo | 115+19 | **Experimental/simulation** — M3 needs real ESP32-S3 hardware | ADR-148:940-953 ("Overall ~98%", M3 85%) |
| **(adjacent)** | `nvsim`, `nvsim-server`, `ruv-neural`, `wifi-densepose-wasm-edge`, `wifi-densepose-mat`, `wifi-densepose-vitals` | NV-diamond sim, neural lib, WASM edge, MAT disaster tool, vitals | 50, 0, 364, 643, 165+9, 52 | Mixed — `mat`/`vitals`/`wasm-edge` mature unit-wise | crate listing |

**Workspace totals (measured):** 3,890 `#[test]` + 121 `#[tokio::test]` static occurrences across `v2/crates/`. (CLAUDE.md's "1,031+ tests" figure refers to an earlier `cargo test --workspace` run count; this review did not execute the suite.)

External vendored runtimes also present: `vendor/rvcsi` (ADR-095/096 edge RF runtime, own repo), `vendor/ruvector`, `vendor/midstream`, `vendor/sublinear-time-solver`.

---

## 3. Signal-Processing Capability Inventory — `ruvsense/`

Location: `v2/crates/wifi-densepose-signal/src/ruvsense/`. CLAUDE.md says "16 modules"; the directory now contains **22 `.rs` files** (21 modules + `mod.rs`) — the table below is the ground truth. Lines/tests measured per file (2026-06-09).

| Module | Lines | Tests | ADR | What it does |
|---|---:|---:|---|---|
| `mod.rs` | 510 | 14 | 029 | Pipeline shell, COCO-17 keypoint constants, `RuvSensePipeline` (concrete fields + `tick()`), re-exports |
| `multiband.rs` | 442 | 14 | 029 | Channel-hopping CSI → wideband virtual snapshot per node (`MultiBandCsiFrame`) |
| `phase_align.rs` | 460 | 13 | 029 | LO phase-offset estimation via circular mean + `ruvector-solver::NeumannSolver` |
| `multistatic.rs` | 957 | 20 | 029 | Attention-weighted N-node fusion → `FusedSensingFrame`; timestamp-spread guards |
| `coherence.rs` | 474 | 19 | 029 | Per-subcarrier z-score coherence vs rolling template; `DriftProfile` |
| `coherence_gate.rs` | 380 | 17 | 029 | Accept / PredictOnly / Reject / Recalibrate gate decisions |
| `pose_tracker.rs` | 1,577 | 38 | 029/026/082 | 17-keypoint Kalman tracker, lifecycle state machine, AETHER re-ID embeddings, skeleton constraints, temporal keypoint attention |
| `field_model.rs` | 1,417 | 22 | 030 | SVD room eigenstructure (persistent field model), perturbation extraction |
| `tomography.rs` | 751 | 12 | 030 | RF tomography, ISTA L1 voxel solver |
| `longitudinal.rs` | 1,020 | 20 | 030 | Welford long-horizon stats, biomechanics drift detection |
| `intention.rs` | 511 | 12 | 030 | Pre-movement lead signals (200–500 ms) |
| `cross_room.rs` | 626 | 13 | 030 | Environment fingerprinting + room-transition graph |
| `gesture.rs` | 579 | 14 | 030 | DTW template-matching gesture classifier |
| `adversarial.rs` | 586 | 13 | 030/032 | Physically-impossible-signal detection, multi-link consistency |
| `attractor_drift.rs` | 566 | 15 | 032a | Midstream-enhanced attractor drift detection |
| `temporal_gesture.rs` | 540 | 15 | 032a | Midstream temporal gesture stream |
| `cir.rs` | 1,025 | 10 | 134 | CSI→CIR via ISTA L1 sparse recovery, NeumannSolver warm-start, `Complex32` sub-DFT Φ |
| `calibration.rs` | 717 | 8 | 135 | Empty-room baseline (Welford amplitude + von Mises phase), drift-triggered recalibration |
| `fusion_quality.rs` | 188 | 3 | 137 | `QualityScore` with `EvidenceRef`s, `ContradictionFlag`s, `CalibrationId`, privacy-demotion predicate |
| `array_coordinator.rs` | 343 | 5 | 138 | Clock-quality gating + `DirectionalEvidence` (geometric admission) |
| `evolution.rs` | 406 | 7 | 142 | Cross-link change-point detection, Bayesian `TemporalVoxelMap` (privacy-gated) |
| `rf_slam.rs` | 301 | 6 | 143 | Persistent reflector discovery → static anchor learning (Wall/Furniture/Mobile classes) |

Subtotal: ~14,400 lines, 310 tests inside `ruvsense/` alone. The non-ruvsense signal layer adds Hampel filtering, CSI-ratio, phase sanitisation, Fresnel modeling, BVP, spectrograms, subcarrier selection, and hardware normalisation (`signal/src/*.rs`).

**Cross-viewpoint fusion** (`wifi-densepose-ruvector/src/viewpoint/`, 5 files): scaled dot-product attention with geometric bias (`attention.rs`), Geometric Diversity Index + Cramér-Rao bounds (`geometry.rs`), phase-phasor coherence with hysteresis + clock-quality gate (`coherence.rs`), and the `MultistaticArray` aggregate root (`fusion.rs`). 136 tests crate-wide.

---

## 4. The Trust Chain — What Actually Composes Today

`wifi-densepose-engine/src/lib.rs` is the proof-of-composition. One `process_cycle` (`lib.rs:285-368`):

1. ADR-138 array coordination (only if every node's geometry is registered, `lib.rs:372-389`)
2. ADR-137 `fuse_scored_calibrated` with **per-node calibration epochs** — mismatching `CalibrationId`s raise a contradiction (`lib.rs:304-319`)
3. ADR-142 change-point → WorldGraph `Event` node (`lib.rs:393-430`)
4. ADR-141 monotonic privacy demotion on any contradiction (`demote_one`, `lib.rs:452-455`)
5. ADR-139/140 `SemanticState` with mandatory provenance (evidence ‖ model ‖ calibration ‖ privacy decision) (`lib.rs:336-352`)
6. BLAKE3 witness over the trust decision (`witness_of`, `lib.rs:437-448`)

The 11 engine tests verify exactly the right invariants: full provenance flow (`cycle_carries_full_provenance`, `lib.rs:487`), contradiction→demotion (`lib.rs:517`), determinism (`lib.rs:535`), calibration-mismatch→Restricted+stable-witness (`lib.rs:648`), privacy-mode attestation chain (`lib.rs:741`), and persist→reload round-trip with **no raw RF in the snapshot** (`live_frame_to_reload_same_contents`, `lib.rs:696-736`).

This is genuinely strong design. But all inputs are synthetic `MultiBandCsiFrame`s constructed in the test module; no ingest crate calls `StreamingEngine` yet.

---

## 5. Strengths

1. **Deterministic witness chain, end to end in design.** ADR-028 proof (`archive/v1/data/proof/verify.py` + SHA-256), ADR-119 BLAKE3 frame witnesses (`bfld/src/signature_hasher.rs`), ADR-136 `CanonicalFrame`/`ComplexSample` LE contracts, and the engine's per-cycle trust witness form a coherent auditability story few sensing systems attempt.
2. **Privacy as a control plane, not a feature.** BFLD's three structural invariants (`bfld/src/lib.rs:7-11`), hash-rotation (ADR-120), identity-risk scoring (ADR-121), mode registry with hash-chained attestations, and *monotonic* demotion wired to fusion contradictions (engine `lib.rs:327-328`) — uncertainty automatically reduces information release.
3. **Multistatic fusion with physics-grounded quality.** Attention fusion + GDI + Cramér-Rao bounds + clock-quality gating means geometry and synchronisation deficits are first-class, measurable contradiction sources rather than silent failure modes.
4. **Test density at the unit level.** 3,890 static test functions; the signal core (473), BFLD (369), and sensing-server (571) are the deepest. ruvsense files average ~14 tests/module.
5. **Honest self-assessment culture.** ADR-136 §8's "skeleton, not a shipping product" framing, ADR-147's explicit "random weights" disclosure, and homecore's in-source TODO-P2 ledger (`homecore/src/lib.rs:24-31`) make the gap analysis below mostly a matter of reading what the project already admits.
6. **A real prior research base with negative results.** The sota-2026-05-22 loop catalogued negatives by resolution path (missing-tool / architecture-error / physics-floor) and produced a ship-recipe (N=5 chest-centric placement, 100% coverage for 1–4 occupants) consolidated into ADR-113.
7. **Hardware path exists and was audited.** ADR-028 (Accepted) and ADR-039 (Accepted, hardware-validated) anchor the ESP32-S3/C6 ingest tier; firmware release process includes real-CSI verification on COM ports.

---

## 6. Honest Gap Analysis — ADR vs Implemented vs Integrated

| Capability | ADR status | Code status | Integrated on live path? |
|---|---|---|---|
| Six-stage ruvsense pipeline | ADR-029 **Proposed** | Implemented + tested (310 tests) | Partially — sensing-server has `multistatic_bridge.rs`/`tracker_bridge.rs`, but `RuvSensePipeline` still holds concrete fields with `tick()` only (`mod.rs`); no uniform `Stage<I,O>` chain runs live |
| Frame contracts (`ComplexSample`, provenance fields, `Stage` traits) | ADR-136 Proposed | Built + 9 acceptance tests (per ADR-136 §8, commit `11f89727f`) | **No** — AC6 600-frame replay witness key and AC7 cross-arch CI matrix not done; provenance fields not populated by live calibration/model stages |
| Fusion quality / contradictions | ADR-137 Proposed | `fusion_quality.rs` (188 lines, 3 tests) + engine wiring | Engine-tests only |
| WorldGraph digital twin | ADR-139 Proposed | `wifi-densepose-worldgraph` (4 files, 7 tests) | Engine-tests only; no recorder-backed persistence loop |
| Privacy control plane | ADR-141 Proposed | `privacy_mode.rs` registry + attestation chain, tested | Engine-tests only; MQTT/HA exposure exists in BFLD but the *engine→BFLD sink* live path is unwired |
| UWB range fusion | ADR-144 Proposed | **No hardware, no crate** — acknowledged absent (`engine/src/lib.rs:28`) | No |
| Ablation/leakage eval harness | ADR-145 Proposed | `wifi-densepose-train/src/ablation.rs` exists | Self-labelled "skeleton/scaffolding" (ADR-145 §status) |
| RF encoder multi-task heads | ADR-146 Proposed | Not trained; `model_id`/`model_version` registry unowned | No — engine stamps `rfenc-v1` as a placeholder string (`lib.rs:338`) |
| RF foundation encoder | ADR-150 **Proposed** | ADR only | No |
| World-model forecasting (OccWorld) | ADR-147 (benchmark doc) | Runs on RTX 5080, 72.39M params — **random weights**, no domain checkpoint | No |
| HomeCore HA port | ADR-125–133 all Proposed | P1 scaffold + siblings; `homecore-server` has **0 tests**; persistence, service mpsc dispatch, device registry, witness integration all deferred (`homecore/src/lib.rs:24-31`) | Partially (API surfaces exist) |
| BFLD capture path (Nexmon/ESP32 BFI) | ADR-123 Proposed | rvCSI vendored runtime exists for nexmon `.pcap`; BFI-specific capture unverified in this review | Unclear |
| Drone swarm | ADR-148 In Progress | 17 modules, sim + Candle PPO complete per milestones | **Simulation only** — M3's 15% requires physical ESP32-S3 CSI capture (ADR-148:946) |
| Federation / DP-SGD / PQC | ADR-105–109 Proposed | `ruview-fed` crate **does not exist** (roadmap Tier 2 item) | No |
| Antenna-placement CLI (`plan-antennas`) | ADR-113 Proposed; Roadmap Tier 1.1 HIGH | numpy references only; not found as a Rust CLI subcommand | No |

**Pattern:** the unit layer is real and deep; the *integration* layer is one crate (`wifi-densepose-engine`) exercised solely by its own synthetic tests; the *model* layer (anything learned: RF encoder, pose model fine-tuned on CSI, OccWorld domain weights) is the emptiest tier. Nearly every ADR ≥118 carries status **Proposed** even where substantial tested code exists — ADR status hygiene lags implementation in both directions (BFLD code outruns its "P1 in progress" header; ADR-148's "98%" outruns its hardware evidence).

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Evidence / Notes |
|---|---|---|---|---|
| R1 | **Integration gap**: trust chain proven only against synthetic in-test frames; live 20 Hz ingest→engine→BFLD-sink path unwired, so the headline guarantee (auditable provenance on every emission) is unverified in production conditions | High | Critical | `engine/src/lib.rs:27-29`; ADR-136 §8 |
| R2 | **No trained model**: every learned component (RF encoder ADR-146/150, OccWorld ADR-147) is random-weight or absent; sensing claims beyond coherence/occupancy heuristics cannot ship | High | Critical | ADR-147 "random weights"; ADR-146/150 Proposed |
| R3 | **Synthetic-validation bias**: ruvsense/engine/swarm tests and the sota-loop results (e.g., R3 "100% (synthetic)", ADR-113 placement numbers) are simulation-derived; real-room domain gap unquantified | High | High | `00-summary.md:45`; PRODUCTION-ROADMAP 2.3 ("turns synthetic numbers into validated numbers") |
| R4 | **Witness chain incomplete at frame level**: `CsiFrame.data` is still `serde(skip)` (ADR-136 Gap 2); AC6 replay-witness key and AC7 cross-architecture matrix not landed — deterministic replay is a design, not a property | Medium | High | ADR-136 §1.1, §8 |
| R5 | **Float nondeterminism in fusion** across thread counts could silently break the witness/replay contract once wired | Medium | High | ADR-136 §3.3 risk table (project's own assessment) |
| R6 | **Privacy bypass via unwired paths**: BFLD invariants are enforced per-module, but until the engine is the *only* route from ingest to API, a sensing-server endpoint can emit ungated state (sensing-server already has 30+ modules incl. pose/vitals APIs predating the control plane) | Medium | Critical | `sensing-server/src/` module list vs engine isolation |
| R7 | **Hardware dependence + scale**: multistatic TDMA/channel-hopping timing validated on small ESP32 sets; ADR-148 M3 explicitly blocked on real hardware; clock-quality model in engine uses a hardcoded `ClockQualityScore` (`engine/src/lib.rs:384`) | Medium | High | ADR-148:946; hardcoded 50 µs stdev |
| R8 | **ADR/doc/status drift**: 150 ADRs with near-universal "Proposed" status, stale in-source status headers (`bfld/src/lib.rs:12`), CLAUDE.md "16 ruvsense modules" vs 22 on disk, duplicate ADR numbers (two ADR-050s, two ADR-147s, two ADR-149s, ADR-052 ×2 — **now RESOLVED: displaced files renumbered to ADR-166…171 per ADR-164 G1**) — institutional-memory value degrades | High | Medium | `ls docs/adr/`; this review §3 |
| R9 | **Workspace breadth vs maintenance capacity**: 38 workspace crates + 4 vendored subtrees + Python archive + firmware; several crates have 0 tests (`homecore-server`, `nvsim-server`, `wifi-densepose-wasm`, `homecore-plugin-example`); bus factor appears to be ~1 | High | Medium | crate test-count table §2 |
| R10 | **Eval debt**: no end-to-end accuracy benchmark on real CSI with ground truth exists in-repo (ADR-145 harness is scaffolding; ADR-079 camera ground truth not exercised here) — "beyond SOTA" claims are currently unfalsifiable | High | High | ADR-145 status note; absence of ground-truth datasets in tree |

---

## 8. Measurement Appendix

- Test counts: `grep -r '#[test]'` / `#[tokio::test]` per crate directory, 2026-06-09. Workspace totals: 3,890 / 121. Top crates: `wasm-edge` 643, `sensing-server` 557+14, `signal` 473, `bfld` 369, `ruv-neural` 364, `train` 312, `mat` 165+9, `wifiscan` 150, `hardware` 137, `ruvector` 136, `ruview-swarm` 115+19.
- ruvsense per-file lines/tests: `wc -l` + per-file `grep -c '#[test]'` (table in §3).
- Crate inventory: `ls v2/crates/` → 38 directories.
- ADR inventory: `ls docs/adr/` → 150 numbered files (with the duplicate numbers noted in R8); `docs/adr/README.md` self-reports "45 ADRs" (stale).
- Caveats: static `#[test]` counts include `#[cfg(feature = ...)]`-gated and ignored tests; they are an upper bound on what `cargo test --workspace --no-default-features` runs. No cargo build/test was executed for this review.

*Next in series: 01+ documents should target the R1/R2/R10 axis — wiring the live path, training the RF encoder, and standing up a falsifiable real-CSI benchmark — before any "beyond SOTA" claim is made.*
