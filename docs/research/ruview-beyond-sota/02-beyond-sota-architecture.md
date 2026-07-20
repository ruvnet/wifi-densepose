# RuView Beyond-SOTA Target Architecture

**Series:** ruview-beyond-sota (02)
**Date:** 2026-06-09
**Status:** Research design — components marked **PROPOSED** do not exist yet; everything else cites real code.
**Governing constraint:** ADR-136 §2.1 explicitly rejects renaming/rewriting the workspace. This document designs an **evolution** of the existing 38-crate `v2/` workspace (`v2/Cargo.toml`), not a new system. Every beyond-SOTA layer attaches to the ADR-136 `Stage<I,O>` / `FrameMeta` / `CanonicalFrame` contracts (`docs/adr/ADR-136-ruview-streaming-engine-frame-contracts.md` §2.2–2.5) and preserves the ADR-028 witness chain.

---

## 1. Where the system is today (grounding)

The ADR-136 ten-role pipeline (ingest → signal → fusion → world → models → privacy → store → api → eval → observe) is already mapped 1:1 onto existing crates (ADR-136 §2.1, normative table). The composition root exists: `v2/crates/wifi-densepose-engine/src/lib.rs` wires ADR-135..146 blocks into one `StreamingEngine::process_cycle` that emits a `TrustedOutput` carrying fusion `QualityScore`, privacy class, `SemanticProvenance`, RF-SLAM (`RfSlam` field), and a BLAKE3 `witness: [u8; 32]`.

Key existing substrate this design builds on:

| Substrate | Path | What it gives us |
|---|---|---|
| Frame contracts + witness | `v2/crates/wifi-densepose-core/src/types.rs` (`CsiFrame`, `CsiMetadata` + `calibration_id`/`model_id`/`model_version`), ADR-136 `ComplexSample`/`CanonicalFrame` | Deterministic LE bytes, BLAKE3 witness, provenance-append-only boundary rule |
| Six-stage signal pipeline | `v2/crates/wifi-densepose-signal/src/ruvsense/mod.rs` (+22 modules incl. `cir.rs`, `calibration.rs`, `tomography.rs`, `rf_slam.rs`, `fusion_quality.rs`, `array_coordinator.rs`) | CSI→CIR, baseline calibration, multistatic fusion, coherence gating |
| Fusion quality + evidence | ADR-137; `ruvsense/multistatic.rs`, `ruvsense/fusion_quality.rs`, `wifi-densepose-ruvector/src/viewpoint/fusion.rs` | `QualityScore` with `EvidenceRef`/`ContradictionFlag`, privacy demotion on contradiction |
| Digital twin | `v2/crates/wifi-densepose-worldgraph/src/lib.rs` (typed `StableDiGraph`, mandatory `SemanticProvenance`) | Persistent room/sensor/track/belief graph |
| World model bridge | `v2/crates/wifi-densepose-worldmodel/src/lib.rs` (`OccWorldBridge`, `TrajectoryPrior`, ADR-147) | Occupancy prediction priors into the Kalman tracker |
| NN + training | `v2/crates/wifi-densepose-train/src/{model.rs,rapid_adapt.rs,ablation.rs,proof.rs,eval.rs,ruview_metrics.rs}`, `wifi-densepose-nn` | Shared backbone + 2 heads, `AdaptationLoss::ContrastiveTTT`, ADR-145 ablation matrix, seeded proof harness |
| Swarm | `v2/crates/ruview-swarm/src/` (`sensing/{multiview.rs,payload.rs,occworld_bridge.rs}`, `marl/`, `topology.rs`) | Raft/hierarchical-mesh drone coordination with CSI payload (ADR-148) |
| Edge WASM | `v2/crates/wifi-densepose-wasm-edge/src/lib.rs` (WASM3 on ESP32-S3, `on_frame` host ABI), `wifi-densepose-wasm` | Hot-loadable on-device sensing modules |
| Quantum-adjacent sim | `v2/crates/nvsim/src/lib.rs` (deterministic NV-magnetometry forward pipeline, SHA-256 witness, WASM-ready) | Honest classical-quantum hybrid substrate (ADR-089) |
| Semantic record + agents | ADR-140 (`wifi-densepose-sensing-server/src/semantic/`), `homecore-assist` | Provenance-bearing semantic states, Ruflo agent bridge |

---

## 2. Target architecture diagram

The beyond-SOTA layers (★ = new/PROPOSED, ☆ = exists-but-not-wired) wrap the ADR-136 pipeline; nothing replaces it.

```
                            ╔═══════════════════ BEYOND-SOTA CONTROL PLANE ═══════════════════╗
                            ║  P6 Continual adaptation loop (TTT + EWC★)   P5 Swarm aperture   ║
                            ║  rapid_adapt.rs → encoder LoRA deltas        planner★ (Raft)     ║
                            ╚════════════▲══════════════════════▲══════════════▲══════════════╝
                                         │ adaptation deltas    │ quality      │ tasking
 [ingest]            [signal]            │      [fusion]        │  [world]     │        [models]
 ESP32/Pi mesh ─► RuvSensePipeline ──────┴──► fuse_scored ──────┴─► WorldGraph ┴──► RF Foundation
 + drone payload    multiband→phase_align     (ADR-137           (ADR-139      │    Encoder (P1)
 (ruview-swarm      →calibration(135)         QualityScore,      twin) ◄───────┘    7 heads + UQ
  sensing/payload)  →cir(134)→multistatic     EvidenceRef,         ▲  │             (ADR-146/150)
      │             →coherence→gate           Contradiction)       │  ▼                  │
      │                  │                        │            RF-SLAM(143)──OccWorld    │
      ▼                  ▼                        │            rf_slam.rs    worldmodel  ▼
 P7 WASM edge      P2 Differentiable RF           │            (P3 closed loop ☆)   P4 cross-modal
 inference★        forward model★                 │                                 distilled student★
 (wasm-edge,       (tomography.rs +               │                                 (camera-free deploy)
  deterministic    cir.rs ISTA as seed)           │
  replay)               │ residuals feed fusion as EvidenceRef★
      │                 ▼
      │            P8 NV-magnetometry fusion★ (nvsim forward model as a sensing node class)
      ▼
 ─────────────────────── ADR-136 CONTRACT SPINE (unchanged) ───────────────────────────────────
  CsiFrame{ComplexSample, FrameMeta{calibration_id, model_id, model_version}} → Stage<I,O>
  → CanonicalFrame::witness_hash() at EVERY stage boundary (BLAKE3, LE-deterministic)
 ───────────────────────────────────────────────────────────────────────────────────────────────
      │                    │                      │                  │
   [privacy]            [store]                [api]              [eval]            [observe]
   wifi-densepose-bfld  homecore-recorder     homecore-api       ADR-145 ablation   homecore-
   gate + demotion      + replay corpus★      /HA/Matter/HAP     (train/ablation.rs automation,
   (ADR-141)                                                      + P1-P8 variants)  Ruflo (ADR-140)
```

---

## 3. The eight pillars

Each pillar: what / why beyond-SOTA / builds-on / contract sketch / feasibility. All trait sketches are **PROPOSED** unless a path is cited.

### P1 — RF Foundation Encoder with multitask uncertainty heads (ADR-146 + ADR-150)

**What.** One shared, self-supervised RF encoder (`wifi-densepose-nn`) with seven typed heads (pose, presence, count, activity, vitals, gait, identity-embedding), each emitting calibrated uncertainty via the ADR-136 `QualityScored` trait, trained with the ADR-150 pose-contrastive objective (same-pose-across-subjects = positive) plus a coherence head that exposes channel instability.

**Why beyond SOTA.** Published WiFi-pose systems (MultiFormer, GraphPose-Fi lineage) report in-domain accuracy and hallucinate under domain shift. ADR-150 documents the real measured frontier: 81.63% torso-PCK@20 in-domain on MM-Fi vs ~11.6% leakage-free cross-subject, and that DANN and bigger capacity both failed (ADR-150 §1). A foundation encoder whose loss stack explicitly separates pose / identity / room / device factors *and* emits an RF-integrity signal per prediction is not in the published literature as a deployed, auditable artifact. Target (not a claim): close the cross-subject gap materially while every head output carries `confidence_bounds()`.

**Builds on.** `v2/crates/wifi-densepose-train/src/model.rs` (`WiFiDensePoseModel`, `kp_head`/`dp_head`); `v2/crates/wifi-densepose-sensing-server/src/embedding.rs` (`ProjectionHead` + LoRA + `info_nce_loss` — the existing seventh head, ADR-146 §1.1); `v2/crates/wifi-densepose-train/src/rapid_adapt.rs` (ContrastiveTTT precedent); ADR-146 §1.4 head fan-out; ADR-150 §2 loss stack.

**Contract sketch** (lands in `wifi-densepose-nn`, per ADR-146 §1.3):
```rust
pub trait RfEncoder: Send + Sync {
    fn encode(&self, window: &CsiWindowTensor) -> Embedding;        // z ∈ R^d_model
    fn model_id(&self) -> u16;                                       // FrameMeta binding (ADR-136 §2.2)
}
pub trait TaskHead<O: QualityScored>: Send + Sync {
    fn name(&self) -> &'static str;
    fn forward(&self, z: &Embedding) -> O;                           // value + uncertainty bounds
}
pub struct MultiTaskOutput { /* per-head QualityScored outputs + coherence: f32 */ }
```

**Feasibility: HIGH for the architecture, MEDIUM for the headline result.** The pure-Rust f32 ABI is proven (`embedding.rs`), the head taxonomy is specified (ADR-146), and the ablation harness to measure it exists (`wifi-densepose-train/src/ablation.rs`). The risk is scientific, not engineering: ADR-150's own data shows naive approaches fail; the pose-contrastive objective is plausible but unproven at scale. Mitigation: ADR-150 §3's frozen-decoder three-variant experiment gates promotion.

### P2 — Physics-informed differentiable RF forward model (PROPOSED)

**What.** A differentiable forward model `render(scene, link_geometry) -> predicted CSI/CIR` used three ways: (1) as a regularizer in encoder training (predictions must be consistent with a Born-approximation scattering model), (2) as an analysis-by-synthesis residual at inference (`|observed − rendered|` becomes an ADR-137 `EvidenceRef`), (3) as a synthetic-data generator complementing MM-Fi (ADR-015).

**Why beyond SOTA.** Published WiFi sensing is almost entirely discriminative; physics-informed neural fields exist for vision (NeRF) and acoustics but no deployed RF-human-sensing stack closes the loop *forward model → residual → fusion evidence → privacy decision*. Making physics disagreement a first-class, witnessed contradiction flag is novel system design, not just a model.

**Builds on.** The codebase already contains the seed of the forward model: `v2/crates/wifi-densepose-signal/src/ruvsense/tomography.rs` (`RfTomographer`, `LinkGeometry`, `OccupancyVolume` — a linear shadowing forward model inverted by ISTA), `ruvsense/cir.rs` (sub-DFT sensing matrix Φ, ISTA L1 — ADR-134), ADR-143 §1.3 (bistatic excess-delay geometry, the exact ray equations), and `nvsim` as the in-repo precedent for a *deterministic, witness-hashed forward physics pipeline* (`v2/crates/nvsim/src/{propagation.rs,pipeline.rs,proof.rs}`).

**Contract sketch** (new module `wifi-densepose-signal/src/ruvsense/forward_model.rs`, PROPOSED):
```rust
pub trait RfForwardModel: Versioned {
    /// Predict per-link CSI given a voxel scene + body primitive set.
    fn render(&self, scene: &OccupancyVolume, links: &[LinkGeometry]) -> Vec<PredictedCsi>;
    /// Physics residual in [0,1]; 0 = perfectly Maxwell/Born-consistent.
    fn residual(&self, observed: &CsiFrame, rendered: &PredictedCsi) -> PhysicsResidual; // → EvidenceRef
}
```

**Feasibility: MEDIUM, with one honest line drawn.** A full Maxwell FDTD-in-the-loop solver is **infeasible** at 20 Hz on this hardware and is a non-goal (§6). What is feasible: a first-order Born / ray-tracing bistatic model (the ADR-143 spheroid geometry generalized), differentiable through finite differences or a small Candle graph, validated against recorded calibration captures (ADR-135 baselines give per-link empty-room ground truth for free). "Maxwell-consistent" should be read as "consistent with a stated first-order approximation, with the approximation order recorded in the witness metadata."

### P3 — RF-SLAM × WorldGraph × OccWorld closed loop (exists in parts, wiring is the work)

**What.** Close the loop: RF-SLAM discovers reflectors/anchors → WorldGraph persists them as `object_anchor` nodes → OccWorld consumes graph occupancy → `TrajectoryPrior`s feed the Kalman tracker → improved tracks refine SLAM association. The environment model becomes self-acquiring and self-correcting (furniture moved ⇒ `BaselineTopologyChange` ⇒ recalibration trigger, ADR-143 §1.4).

**Why beyond SOTA.** Published RF-SLAM work maps *or* tracks; no published consumer system maintains a persistent, provenance-bearing, privacy-rolled-up environmental digital twin (`PrivacyRollup` in `wifi-densepose-worldgraph/src/graph.rs`) that is simultaneously the SLAM map, the automation substrate, and the audit record. The differentiator is the closed loop with evidence edges (`supports`/`contradicts`).

**Builds on.** All three vertices exist: `v2/crates/wifi-densepose-signal/src/ruvsense/rf_slam.rs` (`RfSlam::observe`, line 176, already a field of `StreamingEngine` — `wifi-densepose-engine/src/lib.rs:116`); `v2/crates/wifi-densepose-worldgraph/src/lib.rs`; `v2/crates/wifi-densepose-worldmodel/src/{bridge.rs,occupancy.rs}` (`worldgraph_to_occupancy`, `OccWorldBridge::predict`). The engine already upserts SLAM output and person tracks into the graph. Missing: prior-injection back into `ruvsense/pose_tracker.rs`, and the topology-change → ADR-135 recalibration edge.

**Contract sketch** (extends existing types):
```rust
impl StreamingEngine {
    /// PROPOSED: inject OccWorld priors into the next tracker cycle.
    pub fn apply_trajectory_priors(&mut self, priors: &[TrajectoryPrior]) -> Vec<WorldId>;
}
// WorldEdge gains (PROPOSED): PredictedBy { model_id: u16 }  — prior provenance edge
```

**Feasibility: HIGH.** This is mostly integration glue between tested crates. The two real risks are already named by ADR-143: no ground-truth oracle in a live home (mitigated by the v1-fixed / v2-flagged rollout, `#[cfg(feature = "rf-slam-v2")]`), and OccWorld's Python subprocess (ADR-147: 375 ms/inference) being off the deterministic path — priors must be treated as advisory, never witness-bearing (§5).

### P4 — Cross-modal distillation: camera-teacher → RF-student, privacy-preserving deployment (PROPOSED)

**What.** Train-time-only camera supervision: a vision pose teacher labels synchronized CSI (MM-Fi already provides paired modalities, ADR-015), distilling dense pose + uncertainty into the P1 encoder. Deployed systems ship **no camera and no camera-derived identity features**; the ADR-145 privacy-leakage metric (membership-inference score in `wifi-densepose-train/src/ablation.rs`) gates that the student does not retain identity.

**Why beyond SOTA.** Camera-supervised WiFi pose is the original DensePose-WiFi recipe; what is *not* published is distillation with a measured, CI-enforced privacy-leakage budget and a witnessed claim that the deployed artifact is camera-free. The beyond-SOTA move is making "privacy-preserving" a *measured property of the release pipeline*, not a marketing adjective.

**Builds on.** `v2/crates/wifi-densepose-train/src/{trainer.rs,losses.rs,dataset.rs}` (training substrate); ADR-015 paired datasets; ADR-145 `FeatureSet` matrix + privacy-leakage scalar; `v2/crates/wifi-densepose-bfld` (`privacy_gate.rs`, `signature_hasher.rs` — runtime identity controls, ADR-120 invariants I1–I3).

**Contract sketch** (in `wifi-densepose-train`, PROPOSED):
```rust
pub struct DistillationLoss { pub teacher: TeacherSource, pub temperature: f32, pub uq_transfer: bool }
pub enum TeacherSource { CachedPoseLabels(PathBuf), /* never a live camera in the serving graph */ }
/// Release gate: leakage(student) ≤ budget, asserted by the ADR-145 harness per variant.
pub struct PrivacyBudget { pub max_mia_score: f32 }
```

**Feasibility: HIGH.** All ingredients exist; the work is a loss term, a label cache format, and a CI gate. The honest caveat: MIA-based leakage scores are a lower bound on real leakage; the budget is a regression tripwire, not a formal guarantee.

### P5 — Swarm-distributed multistatic sensing with Raft-coordinated apertures (ADR-148, partially built)

**What.** Treat the drone swarm + fixed ESP32 mesh as one *reconfigurable multistatic aperture*: a Raft-elected cluster head plans node positions/channel assignments to maximize geometric diversity (GDI) for the current sensing task; per-node frames flow into the same `MultistaticFuser` path as fixed nodes.

**Why beyond SOTA.** Published multistatic WiFi sensing assumes fixed geometry. Closed-loop aperture optimization — moving the sensors to where the Fisher information is — driven by the GDI/Cramér–Rao machinery that already exists in `v2/crates/wifi-densepose-ruvector/src/viewpoint/geometry.rs` (per CLAUDE.md module table: `GeometricDiversityIndex`, Cramér-Rao bounds) is a genuinely new system class for SAR/MAT scenarios.

**Builds on.** `v2/crates/ruview-swarm/src/sensing/{multiview.rs,payload.rs,occworld_bridge.rs}`, `topology.rs`, `planning.rs`, `marl/` (MAPPO, `candle_ppo.rs`); `ruvsense/multistatic.rs` + `array_coordinator.rs` (ADR-138 clock-quality gating — moving nodes will stress exactly this); `wifi-densepose-mat` (the MAT use case).

**Contract sketch** (in `ruview-swarm`, PROPOSED):
```rust
pub trait AperturePlanner: Send + Sync {
    /// Given current twin + task, propose node placements maximizing expected GDI.
    fn plan(&self, twin: &WorldGraphSnapshot, task: &SwarmTask) -> Vec<(NodeId, Position3D)>;
}
// Output flows through Raft (topology.rs) as a normal SwarmTask; frames return as ArrayNodeInput.
```

**Feasibility: MEDIUM.** Coordination, MARL, and fusion code exist and are tested; the hard physical problems are honest unknowns: airborne CSI phase stability (rotor vibration), clock sync across mobile nodes (ADR-138 gate will reject a lot initially), and ADR-148 §1.3's own regulatory scoping. Simulation-first via `ruview-swarm/src/evals.rs` + `bench_support.rs`; hardware validation is Phase 3.

### P6 — Continual / test-time adaptation with EWC-style forgetting control (PROPOSED on existing TTT)

**What.** Promote `rapid_adapt.rs` from a per-deployment trick to a managed continual-learning loop: TTT/entropy adaptation produces LoRA deltas on the P1 encoder; an EWC (elastic weight consolidation) penalty — **which does not exist in the workspace today** (no EWC match in `wifi-densepose-train/src/rapid_adapt.rs`) — anchors weights important to previously-validated environments; every adaptation step is versioned as a new `model_version` (u16, ADR-136 §2.2) and must re-pass the ADR-145 acceptance matrix before activation.

**Why beyond SOTA.** TTT papers adapt and hope; nothing published couples adaptation to a *deterministic regression gate with witness hashes*, where an adapted model that regresses tier or leaks identity is automatically rejected and the `model_version` provenance lets any semantic state be traced to the exact adaptation step.

**Builds on.** `v2/crates/wifi-densepose-train/src/rapid_adapt.rs` (`AdaptationLoss::ContrastiveTTT`, entropy-minimization variant — lines 8–16); LoRA adapters in `sensing-server/src/embedding.rs` (rank-4 `lora_1`/`lora_2`); ADR-027 MERIDIAN evaluator (`train/src/eval.rs`); ADR-146 §2 calibration-robustness loss.

**Contract sketch** (in `wifi-densepose-train`, PROPOSED):
```rust
pub struct EwcPenalty { pub fisher_diag: Vec<f32>, pub anchor: Vec<f32>, pub lambda: f32 }
pub struct AdaptationStep {
    pub parent_model_version: u16, pub new_model_version: u16,
    pub loss: AdaptationLoss, pub ewc: Option<EwcPenalty>,
    pub acceptance: RuViewAcceptanceResult,           // must be ≥ parent tier
    pub witness: [u8; 32],                            // hash of delta + acceptance
}
```

**Feasibility: HIGH.** EWC over a small LoRA delta is cheap (Fisher diagonal over the replay corpus); the acceptance gate and proof seeds exist (`proof.rs`, `PROOF_SEED = 42`). Risk: online Fisher estimation from unlabeled home data is noisy — start with adaptation restricted to LoRA parameters only, backbone frozen.

### P7 — On-device WASM edge inference with deterministic replay (extends existing Tier-3)

**What.** Push P1 head subsets (presence, vitals, coarse activity) into hot-loadable WASM modules on ESP32-S3, and onto browsers/workers via `wifi-densepose-wasm`. Every edge module's output is replayable: the same `CanonicalFrame` input bytes through the same module hash produce the same output bytes, verified in CI on x86_64/aarch64/wasm32.

**Why beyond SOTA.** Edge WiFi-sensing exists; *bit-deterministic, witness-hashed edge inference with hot-swap and replay parity against the server pipeline* does not appear in published systems. It turns the edge from a trust hole into a witness-chain extension.

**Builds on.** `v2/crates/wifi-densepose-wasm-edge/src/lib.rs` (WASM3 host ABI: `csi_get_*`, `on_frame` at ~20 Hz, ADR-040 Tier 3); `nvsim` as the proof that a no-std-time, no-OS-entropy, seeded-PRNG crate runs identically on wasm32 (`nvsim/src/lib.rs` doc); ADR-136 AC7 cross-architecture byte-stability test.

**Contract sketch** (PROPOSED additions to the wasm-edge host ABI):
```rust
// exports added to module lifecycle:
//   on_replay_begin(seed: u64)              — pins any module-internal PRNG
//   witness_digest(buf_ptr: i32) -> i32     — module returns BLAKE3 of its output stream
pub trait EdgeStage: Stage<CsiFrameView, EdgeEvent> { fn module_hash(&self) -> [u8; 32]; }
```

**Feasibility: HIGH for presence/vitals heads, LOW for full pose on-ESP32.** WASM3 interpretation on Xtensa caps throughput; full 7-head inference stays on Pi/Hailo/browser. Float determinism across native vs WASM needs care (no fast-math, fixed reduction order — same obligation ADR-136 §3.2 already accepts).

### P8 — NV-magnetometry fusion: an honest classical-quantum hybrid (PROPOSED, simulation-first)

**What.** Add `nvsim`-modeled NV-magnetometer nodes as a *fourth sensing modality class* (after CSI, mmWave/ADR-021, BFLD) in the multistatic fusion: near-range (≤ tens of cm, per the physics review) cardiac/respiratory magnetic signatures fused with CSI/mmWave vitals under the ADR-137 evidence contract. Simulation-first: the modality lands end-to-end against `nvsim` before any hardware exists.

**Why beyond SOTA.** Not range — the Ghost Murmur review (`docs/research/quantum-sensing/16-ghost-murmur-ruview-spec.md`) documents why multi-mile cardiac magnetometry contradicts published physics, and this design adopts that conclusion. The beyond-SOTA element is architectural honesty: a fusion engine that can ingest a quantum-sensor modality with explicit, witnessed physics bounds (`nvsim`'s forward model states its approximations and hashes its output, `nvsim/src/proof.rs`), so that when real NV hardware matures, the integration path and the anti-hype guardrails already exist. No published consumer sensing stack has this.

**Builds on.** `v2/crates/nvsim/src/` (scene→source→attenuation→NV ensemble→digitiser, SHA-256 witness, ADR-089); `nvsim-server`; `wifi-densepose-vitals` (mmWave HR/BR — the modality NV would cross-validate); `ruvsense/multistatic.rs` fusion + ADR-137 `EvidenceRef`.

**Contract sketch** (PROPOSED): a `SensorModality::NvMagnetometer` variant on the existing `wifi-densepose-worldgraph` `SensorModality` enum, plus an `ArrayNodeInput` adapter from `nvsim` frames; vitals agreement/disagreement between NV and mmWave becomes an `EvidenceRef`/`ContradictionFlag` pair.

**Feasibility: HIGH in simulation, SPECULATIVE on hardware.** The sim path is days of glue; COTS NV magnetometers with the required sensitivity at consumer cost do not exist in 2026. This pillar's deliverable is the *contract and the simulated validation*, explicitly labeled as such.

---

## 4. Phased implementation plan

Phases are gated by the Pre-Merge Checklist (CLAUDE.md) and the witness chain (§5). Crate names per the ADR-136 §2.1 normative map — no new `ruview_*` crates except where a crate already exists (`ruview-swarm`).

**Phase 0 — Hardening (close the ADR-136 "integration glue" debt).**
- `wifi-densepose-signal`: wire the full 600-frame `Stage`-chain replay (ADR-136 AC6) and register `streaming_engine_replay_v1` in `archive/v1/data/proof/expected_features.sha256`.
- CI: cross-architecture witness matrix x86_64/aarch64 (AC7); add wasm32 lane for `nvsim` + `wifi-densepose-wasm`.
- `wifi-densepose-engine`: populate `FrameMeta.calibration_id`/`model_id` from the live calibration and model-binding stages (currently defaulted — ADR-136 §8).
- `homecore-recorder`: define the **replay corpus** format (canonical-bytes frame streams + witness manifest) that P4/P6 training and all ablations consume.

**Phase 1 — Encoder + measurement (P1, P4 groundwork, P6 skeleton).**
- `wifi-densepose-nn`: `RfEncoder`/`TaskHead` traits, seven-head fan-out, UQ layer (ADR-146); relocate `ProjectionHead` from `sensing-server/src/embedding.rs`.
- `wifi-densepose-train`: `ContrastiveBatcher`, ADR-150 loss stack, distillation loss + cached-teacher format (P4), `EwcPenalty` + `AdaptationStep` (P6); extend `ablation.rs` `FeatureSet` with per-head and per-pillar variants; pin `expected_ablation_*.sha256`.
- Run the ADR-150 three-variant frozen-decoder experiment; promotion gate on cross-subject delta.

**Phase 2 — Closed loop + edge (P3, P7).**
- `wifi-densepose-engine`: `apply_trajectory_priors` (OccWorld → `pose_tracker.rs`); `PredictedBy` provenance edge in `wifi-densepose-worldgraph`; topology-change → ADR-135 recalibration trigger.
- `wifi-densepose-wasm-edge`: replay ABI (`on_replay_begin`, `witness_digest`), presence/vitals head modules; parity test vs server pipeline on identical canonical bytes.
- Enable `rf-slam-v2` feature on the 7-day validation dataset (ADR-143 gate).

**Phase 3 — Frontier (P2, P5, P8).**
- `wifi-densepose-signal/src/ruvsense/forward_model.rs`: Born/ray forward model seeded from `tomography.rs`; `PhysicsResidual` → `EvidenceRef`; synthetic-data generator into `train/src/dataset.rs`.
- `ruview-swarm`: `AperturePlanner` over GDI (`ruvector/src/viewpoint/geometry.rs`); simulation evals in `evals.rs`; airborne CSI stability study before any hardware claim.
- `nvsim` ↔ `wifi-densepose-engine`: `SensorModality::NvMagnetometer` adapter, simulated NV+mmWave vitals cross-validation in the ablation matrix.

---

## 5. Determinism & witness-chain preservation

The non-negotiable invariant (ADR-136 §2.5–2.6, ADR-028): replaying recorded canonical bytes through the pipeline twice yields byte-identical outputs and equal BLAKE3 witness hashes. Strategy per component class:

1. **Everything on the trust path implements `CanonicalFrame`.** New frame types (`MultiTaskOutput`, `PhysicsResidual`, `AdaptationStep`, edge events, NV frames) get fixed-field-order LE encodings and `witness_hash()`; encoders are the only serializers (no ad-hoc serde on the witness path).
2. **Inference is witnessed by (input hash, model hash, output hash).** `model_id`/`model_version` on `FrameMeta` already bind frames to models; P1 adds a weights digest so the triple is closed. Pure-Rust f32 inference (ADR-146 ABI) with fixed reduction order; no GPU nondeterminism on the witness path — GPU/libtorch is training-only, and training determinism is pinned by the existing seeds (`proof.rs`: `PROOF_SEED = 42`, `MODEL_SEED = 0`).
3. **Advisory vs witnessed split.** Components that cannot be made deterministic — the OccWorld Python subprocess (ADR-147), live MARL exploration, any future LLM/agent output (ADR-140 Ruflo) — are **advisory**: their outputs may bias estimates but never enter `to_canonical_bytes()` directly; instead the *decision to use them* is recorded (prior id + content hash) so replay reproduces the decision even if the producer cannot be re-run. The Kalman tracker consumes priors as explicit inputs recorded in the replay corpus.
4. **Adaptation is a chain of witnessed steps.** P6's `AdaptationStep.witness` hashes (parent version ‖ delta ‖ acceptance result); the active model at any timestamp is reconstructible from the step chain — the model-weights analogue of the frame witness chain.
5. **Edge parity.** P7 modules must produce the same `witness_digest` as the server-side reference implementation on the AC6 fixture; the module hash joins the firmware `source-hashes.txt` in the ADR-028 witness bundle.
6. **Witness bundle growth is mechanical.** Each pillar adds expected-hash keys (`forward_model_residual_v1`, `edge_presence_replay_v1`, `nvsim` already ships `proof.rs`) to the existing `verify.py` chain rather than inventing new verification mechanisms.

---

## 6. Explicit non-goals

- **No workspace rename or rewrite.** Reaffirms ADR-136 §2.1/§4.1: no `ruview_*` crate prefix migration, no umbrella crate; pillars land inside the existing crates listed above.
- **No full-wave Maxwell solver in the runtime loop.** P2 is first-order Born/ray, with the approximation order declared. "Physics-informed" never means FDTD at 20 Hz.
- **No long-range cardiac magnetometry claims.** P8 is bounded by the physics review in `docs/research/quantum-sensing/16-ghost-murmur-ruview-spec.md`; ranges beyond published MCG physics are out of scope permanently, not just deferred.
- **No camera in any deployed serving graph** (P4 teachers are train-time, cached-label only) and **no identity recognition as a product feature** — identity embeddings remain in-RAM, hash-rotated (ADR-120 invariants).
- **No weaponization or LAWS capability in P5**, per ADR-148 §1.3; swarm work targets SAR/MAT and stays behind the ADR-148 regulatory gates.
- **No fabricated benchmarks.** All pillar performance statements in this document are targets; promotion of any pillar requires the ADR-145 ablation matrix delta plus pinned determinism hashes, in CI, before any external claim.
- **No new verification mechanisms.** The witness chain extends `verify.py` / BLAKE3 / `expected_*.sha256`; we do not introduce a second, parallel proof system.

---

## 7. Open questions for the next document in this series

1. Airborne CSI phase stability (P5): what does the ADR-138 clock-quality gate measure on a real quadrotor payload?
2. Forward-model fidelity floor (P2): what Born-residual magnitude on the ADR-135 empty-room captures is "good enough" to be a useful contradiction signal?
3. Replay-corpus governance (Phase 0): retention, privacy class of recorded canonical bytes, and consent — the recorder stores signal evidence, which is itself sensitive.
