# ADR-142: Evolution Tracker and Temporal VoxelMap Evidence Aggregation

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see Implementation Status, commit `1f8e180d6`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (`ruvsense/longitudinal.rs`, `ruvsense/attractor_drift.rs`, `ruvsense/calibration.rs`, `ruvsense/field_model.rs`, `ruvsense/tomography.rs`); `wifi-densepose-bfld` (`privacy_gate.rs`) |
| **Relates to** | ADR-030 (Persistent Field Model), ADR-134 (First-Class CIR Support), ADR-135 (Empty-Room Baseline Calibration), ADR-084, ADR-118, ADR-120 (BFLD Privacy Classes), ADR-136 (Streaming Engine), ADR-137 (Fusion Quality Scoring), ADR-139 (WorldGraph), ADR-141 (BFLD Privacy Control Plane) |

---

## 1. Context

### 1.1 The Gap

The RuvSense crate already contains every individual ingredient an "evolution tracker" would need, but they exist as five disconnected modules with no orchestrator that runs them together over time and across links. Searching `v2/crates/wifi-densepose-signal/src/ruvsense/` for `EvolutionTracker`, `change_point`, `VoxelMap`, and any cross-module driver finds nothing. What does exist:

- **`field_model.rs`** holds the per-link Welford baselines (`LinkBaselineStats`, `WelfordStats` at line 79), runs the SVD eigenstructure decomposition (`finalize_calibration()`, line 487), exposes `estimate_occupancy(&[Vec<f64>]) -> Result<usize, FieldModelError>` (line 741, with a `NotCalibrated` stub at line 821 when the `eigenvalue` feature is off), and tracks calibration freshness via `check_freshness(current_us) -> CalibrationStatus` (line 829) returning `Uncalibrated | Collecting | Fresh | Stale | Expired` (enum at line 300). Nothing aggregates freshness *across* links — each `FieldModel` instance is per-room and unaware of its siblings.
- **`calibration.rs`** (ADR-135) holds the empty-room amplitude/phase baseline: `BaselineCalibration` (line 228), `CalibrationRecorder` with a `W`-frame staleness window, `deviation(&CsiFrame) -> CalibrationDeviationScore` (line 238), and `CalibrationError` (line 128). Its `CalibrationDeviationScore` (line 372) carries the per-frame `drift_score`, but the drift signal is consumed only by that single link's recorder. There is no cross-link rule that says "3 links drifted simultaneously, therefore the room changed."
- **`longitudinal.rs`** holds the per-person `PersonalBaseline` (line 156) with five Welford metrics and an `EmbeddingHistory` FIFO (line 344, `push()` at line 389, `novelty()` at line 500). It produces a `DriftReport` (line 110) and a `MonitoringLevel` (line 99) per person — but per-person, never tied back to the per-link RF evidence that produced the embedding.
- **`attractor_drift.rs`** holds phase-space regime classification: `AttractorDriftAnalyzer` (line 203), `analyze()` (line 257) returning `AttractorDriftReport { regime_changed, ... }` (line 136), classifying `BiophysicalAttractor` (line 93). Again per-person-per-metric; nothing escalates a regime change into the field/calibration tier.
- **`tomography.rs`** holds the coarse RF tomographer: `RfTomographer` (line 178), `reconstruct(&[f64]) -> OccupancyVolume` (line 236) with an ISTA L1 solver, and an `OccupancyVolume` (line 121) of `densities: Vec<f64>`. Critically, **the `OccupancyVolume` is stateless** — every `reconstruct()` call produces a fresh volume from a single attenuation snapshot. There is no temporal memory: a voxel that has been occupied for 200 frames is indistinguishable from one that flickered for a single noisy frame. There is no per-voxel confidence, no `last_update_ns`, no evidence count, and no Doppler.

On the privacy side, `wifi-densepose-bfld/src/privacy_gate.rs` implements the monotonic `PrivacyGate::demote(BfldFrame, PrivacyClass)` (line 31) that zeroes payload sections going `Raw(0) → Derived(1) → Anonymous(2) → Restricted(3)` (classes defined in `bfld/src/lib.rs` line 84), refusing any promotion with `BfldError::InvalidDemote` (line 187). But the gate operates on `BfldFrame` payload sections (`compressed_angle_matrix`, `csi_delta`, `amplitude_proxy`, `phase_proxy`) — **it has no concept of a voxel grid**. A tomographic `OccupancyVolume`, if it were ever emitted, would leave the node ungated.

The gap is therefore twofold:

1. **No orchestrator.** Each link maintains its own baseline, drift score, attractor state, and occupancy estimate in isolation. A change in the physical environment (furniture moved, a wall opened) manifests as correlated drift across *several* links, but no module reads more than one link at a time. Cross-link change-point detection — the signal that distinguishes "the world changed" from "this one link is noisy" — does not exist.
2. **No temporal occupancy memory.** `RfTomographer::reconstruct()` is memoryless, so occupancy cannot accumulate evidence, cannot be assigned confidence, and cannot be Bayesian-updated across the 20 Hz reconstruction cadence. And whatever it produces is not gated for privacy.

ADR-030 (Persistent Field Model, Proposed) defines the per-room field model and Tier-2 tomography but says nothing about orchestrating multiple rooms/links or about temporal voxel state. This ADR extends ADR-030 with the missing orchestration layer and the missing temporal voxel layer, and routes both through the BFLD privacy gate (ADR-120/ADR-141).

### 1.2 What "Evolution" Means Here

"Evolution" is the second-order signal: not the instantaneous state of the field, but **how the field's statistical description is changing over time and whether that change is coherent across links**. Three concrete questions the EvolutionTracker answers that no current module can:

- *Are the per-link baselines still valid as a set?* (freshness across the mesh, not per-link)
- *Did the environment just change, or is one link misbehaving?* (cross-link change-point)
- *Does the model's occupancy estimate agree with the raw RF body-perturbation energy?* (occupancy-consistency, an internal contradiction check feeding ADR-137)

### 1.3 What This ADR Is Not

It is not a new tomography solver — it wraps the existing `RfTomographer`. It is not a new calibration algorithm — it reads ADR-135's `BaselineCalibration` and ADR-030's `FieldModel`. It is not a new privacy model — it reuses the `PrivacyGate::demote` pattern from `bfld/src/privacy_gate.rs`. It adds exactly two things: a coordinator (`EvolutionTracker`) and a stateful, gated occupancy memory (`VoxelMap` + `VoxelGate`).

### 1.4 Pipeline Position

```
Per-link CSI frame (baseline-subtracted, ADR-135)
  → CalibrationRecorder::record()      (ruvsense/calibration.rs)  → drift_score[link]
  → FieldModel::extract_perturbation() (ruvsense/field_model.rs)  → body_energy[link]
  → RfTomographer::reconstruct()       (ruvsense/tomography.rs)   → OccupancyVolume (snapshot)
        │                │                       │
        └────────────────┴───────────────────────┴──► EvolutionTracker::tick()   ← NEW
                                                          ├─ baseline freshness across mesh
                                                          ├─ cross-link change-point
                                                          ├─ occupancy-consistency check
                                                          └─ VoxelMap::ingest(volume)  ← NEW (temporal)
                                                                  │
                                                          VoxelGate::demote(map, mode) ← NEW (BFLD-gated)
                                                                  │
                            ┌─────────────────────────────────────┴───────────────────┐
                    ADR-137 contradiction flags                            ADR-139 WorldGraph nodes
```

`EvolutionTracker::tick()` runs once per reconstruction cycle (20 Hz). It reads the per-link drift scores and body-perturbation energies, the field model occupancy estimate, and the latest `OccupancyVolume`, then folds the volume into the persistent `VoxelMap`. Output leaves the node only through `VoxelGate`.

---

## 2. Decision

### 2.1 The `EvolutionTracker` Trait

`EvolutionTracker` is a trait (so the production aggregator and the test harness can supply different link-state providers) plus a default implementation `MeshEvolutionTracker`. It owns *references* to the per-link state already maintained by the existing modules; it does not duplicate their accumulators.

```rust
use wifi_densepose_signal::ruvsense::calibration::{BaselineCalibration, CalibrationDeviationScore};
use wifi_densepose_signal::ruvsense::field_model::CalibrationStatus;
use wifi_densepose_signal::ruvsense::tomography::OccupancyVolume;

/// Stable identifier for one TX→RX link in the mesh.
pub type LinkId = usize;

/// Per-link evidence handed to the tracker each tick.
#[derive(Debug, Clone)]
pub struct LinkObservation {
    pub link_id: LinkId,
    /// ADR-135 per-frame deviation (carries drift_score + rms_amplitude_z).
    pub deviation: CalibrationDeviationScore,
    /// ADR-030 field-model freshness for this link's room.
    pub freshness: CalibrationStatus,
    /// Body-perturbation energy from FieldModel::extract_perturbation(),
    /// the residual after environmental modes are projected out.
    pub body_energy: f32,
    /// Capture timestamp, nanoseconds since the 802.15.4 epoch (ADR-110).
    pub timestamp_ns: u64,
}

/// Aggregate result of one evolution tick.
#[derive(Debug, Clone)]
pub struct EvolutionReport {
    /// Worst freshness observed across all links this tick.
    pub mesh_freshness: CalibrationStatus,
    /// Links currently Stale or Expired (drives CoherenceAlert).
    pub stale_links: Vec<LinkId>,
    /// True if a cross-link change-point fired this tick (§2.2).
    pub change_point: bool,
    /// Links that participated in the change-point (≥2σ this window).
    pub change_point_links: Vec<LinkId>,
    /// Occupancy as the field model sees it.
    pub model_occupancy: usize,
    /// Occupancy implied by summed per-link body-perturbation energy.
    pub perturbation_occupancy: usize,
    /// True when |model − perturbation| > 1 (drives AnomalyWarn, §2.3).
    pub occupancy_disagreement: bool,
    /// Alerts emitted this tick (typed, for the streaming engine ADR-136).
    pub alerts: Vec<EvolutionAlert>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvolutionAlert {
    /// One or more baselines are no longer fresh across the mesh.
    CoherenceAlert { stale_links: Vec<LinkId> },
    /// Cross-link change-point: the environment likely changed.
    ChangePoint { links: Vec<LinkId> },
    /// Model occupancy and RF-energy occupancy disagree by >1 person.
    AnomalyWarn { model: usize, perturbation: usize },
}

pub trait EvolutionTracker {
    /// Fold one tick of per-link observations + the latest occupancy
    /// snapshot into the tracker's persistent state. Updates the VoxelMap.
    fn tick(
        &mut self,
        observations: &[LinkObservation],
        volume: &OccupancyVolume,
        now_ns: u64,
    ) -> EvolutionReport;

    /// Borrow the temporal voxel map for gated output (§2.5).
    fn voxel_map(&self) -> &VoxelMap;

    /// Configuration knobs.
    fn config(&self) -> &EvolutionConfig;
}
```

The default `MeshEvolutionTracker` holds the rolling windows the existing modules already require but does not re-implement them — it stores small ring buffers of the *scores* (not the raw CSI):

- per-link `VecDeque<f32>` of the last `W = 300` `drift_score` values (the same window ADR-135 `CalibrationConfig.drift_window_frames` uses);
- per-link `VecDeque<f32>` of `rms_amplitude_z` for the change-point test;
- the `EmbeddingHistory` FIFO (`longitudinal.rs`) and phase-space buffers (`attractor_drift.rs`) are *referenced by handle*, not copied — the tracker calls their existing `analyze()`/`novelty()` on demand.

```rust
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Change-point window length in frames. Default: 30 (1.5 s @ 20 Hz).
    pub change_point_window: usize,
    /// Per-link z threshold counting toward a change-point. Default: 2.0σ.
    pub change_point_sigma: f32,
    /// Minimum links exceeding threshold to declare a change-point. Default: 3.
    pub change_point_min_links: usize,
    /// Occupancy disagreement tolerance, in persons. Default: 1.
    pub occupancy_tolerance: usize,
    /// Per-voxel minimum evidence count before a voxel is "confident". Default: 5.
    pub min_evidence_frames: u32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            change_point_window: 30,
            change_point_sigma: 2.0,
            change_point_min_links: 3,
            occupancy_tolerance: 1,
            min_evidence_frames: 5,
        }
    }
}
```

### 2.2 Cross-Link Change-Point Detection

A single link drifting is noise; the whole environment changing shows up as *correlated* drift. The rule, evaluated every tick:

> Within the rolling `change_point_window` (default 30 frames / 1.5 s), if **3 or more links** each exceed `change_point_sigma` (default 2.0σ) on their `rms_amplitude_z`, emit a `ChangePoint` event naming those links.

```rust
fn detect_change_point(&self) -> Option<Vec<LinkId>> {
    let mut hot = Vec::new();
    for (link_id, window) in self.z_windows.iter() {
        // Count frames in the window above the sigma threshold.
        let n_hot = window.iter().filter(|&&z| z >= self.config.change_point_sigma).count();
        // A link "participates" if it was hot for a majority of the window.
        if n_hot * 2 > window.len() {
            hot.push(*link_id);
        }
    }
    (hot.len() >= self.config.change_point_min_links).then_some(hot)
}
```

The 3-link minimum is deliberately the same scale as ADR-135's `drift_confirm_frames` confirmation logic but operates spatially instead of temporally: ADR-135 confirms a single link's staleness over 45 s; this ADR confirms an environment change over 3 links in 1.5 s. The two are complementary — ADR-135 answers *"is this link's baseline old?"* and this rule answers *"did the world just move?"*. A `ChangePoint` is the upstream trigger that lets the operator (or, if `recalibrate_on_drift` from ADR-135 §2.6 is enabled) recalibrate the *whole mesh* rather than one link.

The 2.0σ threshold reuses ADR-135's interpretation: `rms_amplitude_z > 3.0` is "likely occupied" for a single frame, so a *sustained* 2.0σ across a 1.5 s window on multiple links is a structural shift, not a single body passing one link.

**Mesh freshness aggregation.** Independently of change-points, the tracker reduces per-link `CalibrationStatus` to one `mesh_freshness` using the worst-case ordering `Fresh < Stale < Expired` (with `Uncalibrated`/`Collecting` treated as worse than `Fresh`). Any link at `Stale` or `Expired` lands in `stale_links` and produces a `CoherenceAlert`. This is the cross-mesh freshness check that `field_model.rs::check_freshness` cannot do alone — it only knows one room.

### 2.3 Occupancy-Consistency Check

Two independent occupancy estimates exist and should agree:

- **Model occupancy**: `FieldModel::estimate_occupancy(recent_frames)` (field_model.rs line 741) — derived from eigenstructure energy in the off-environment subspace.
- **Perturbation occupancy**: a count derived from the summed per-link `body_energy` (the residual after `extract_perturbation()` projects out the environmental modes). The tracker bins total body energy into a person count using a fixed energy-per-person scale calibrated at install.

```rust
fn occupancy_consistency(&self, model_occ: usize, body_energy_total: f32) -> (usize, bool) {
    let perturbation_occ = (body_energy_total / self.energy_per_person).round() as usize;
    let disagree = model_occ.abs_diff(perturbation_occ) > self.config.occupancy_tolerance;
    (perturbation_occ, disagree)
}
```

When the two disagree by more than `occupancy_tolerance` (default 1 person), the tracker emits `AnomalyWarn { model, perturbation }`. This is exactly the kind of *internal contradiction* ADR-137's fusion quality scoring consumes: the semantic state record produced downstream carries this as a contradiction flag with references to both evidence sources (the field model version and the calibration version that produced each estimate). Per the project rule, every semantic state traces to **signal evidence** (the `LinkObservation` set), **model version** (the `FieldModel` SVD generation), **calibration version** (the `BaselineCalibration.captured_at_unix_s` from ADR-135), and **privacy decision** (the `VoxelGate` mode, §2.5).

### 2.4 Temporal `VoxelMap` with Bayesian Evidence Accumulation

The core new state. The existing `OccupancyVolume` (tomography.rs line 121) is a memoryless snapshot. The `VoxelMap` is the persistent companion that accumulates evidence across `reconstruct()` calls.

```rust
/// One voxel of persistent, evidence-accumulating occupancy state.
#[derive(Debug, Clone)]
pub struct Voxel {
    /// Center position (metres), copied from OccupancyVolume::voxel_center().
    pub center_xyz: [f32; 3],
    /// Bayesian occupancy probability ∈ [0, 1].
    pub occupancy: f32,
    /// Confidence ∈ [0, 1]; rises with evidence_count, falls with staleness.
    pub confidence: f32,
    /// Nanoseconds (802.15.4 epoch) of the last frame that updated this voxel.
    pub last_update_ns: u64,
    /// Number of frames that have contributed evidence to this voxel.
    pub evidence_count: u32,
    /// Welford mean/variance of the density observations (variance flags noise).
    pub density_mean: f32,
    pub density_m2: f32,
    /// Radial Doppler velocity estimate (m/s), when CIR phase rate is available.
    pub doppler_velocity: f32,
}

/// Persistent occupancy grid shared across all reconstruct() calls.
#[derive(Debug, Clone)]
pub struct VoxelMap {
    pub voxels: Vec<Voxel>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub bounds: [f64; 6],
    /// Half-life (frames) of the confidence decay for un-updated voxels.
    decay_half_life: f32,
}

impl VoxelMap {
    /// Allocate a VoxelMap matching an OccupancyVolume's geometry.
    pub fn from_geometry(volume: &OccupancyVolume) -> Self;

    /// Fold one fresh OccupancyVolume into the persistent map.
    ///
    /// For each voxel:
    /// 1. Bayesian log-odds update of `occupancy` from the new density
    ///    (density treated as a measurement likelihood via a logistic link).
    /// 2. Welford update of (density_mean, density_m2).
    /// 3. evidence_count += 1; last_update_ns = now_ns.
    /// 4. confidence ← logistic(evidence_count) × (1 − normalised_variance).
    /// Voxels NOT touched this frame decay confidence toward 0 with
    /// `decay_half_life`, but retain their last occupancy estimate.
    pub fn ingest(&mut self, volume: &OccupancyVolume, now_ns: u64, min_evidence: u32);

    /// Per-voxel Welford sample variance.
    pub fn density_variance(&self, idx: usize) -> f32;

    /// Voxels with evidence_count < min_evidence are LOW CONFIDENCE.
    pub fn low_confidence_indices(&self, min_evidence: u32) -> Vec<usize>;

    /// Occupancy histogram (counts per occupancy bucket) for Restricted mode.
    pub fn occupancy_histogram(&self, n_buckets: usize) -> Vec<u32>;
}
```

**Bayesian update.** Each voxel's `occupancy` is maintained in log-odds and updated with the new density observation through a logistic measurement model `p(occupied | density) = σ(k·(density − d₀))`. Log-odds accumulation is the standard occupancy-grid update (Moravec & Elfes, 1985; Thrun et al., 2005): it is commutative and numerically stable, and it lets a voxel that is repeatedly observed occupied converge toward 1.0 while a one-frame flicker barely moves the estimate. This directly solves the memoryless-snapshot problem: a 200-frame occupancy is now distinguishable from a 1-frame spike via `evidence_count` and the converged log-odds.

**Confidence and low-confidence flagging.** `confidence = logistic(evidence_count / min_evidence) × (1 − clamp(normalised_density_variance))`. Voxels with `evidence_count < min_evidence_frames` (default 5, §2.1) are returned by `low_confidence_indices()` and flagged downstream so the fusion engine (ADR-137) never treats a 4-frame voxel as a confident detection. This mirrors how `tomography.rs` already counts `occupied_count` at density > 0.01, but adds the *temporal* qualifier the snapshot lacks.

**Welford variance per voxel.** Reuses the exact `(mean, m2)` update form of `WelfordStats` from `field_model.rs` (line 79–162) so a voxel whose density is high but *noisy* (high variance) is correctly distrusted relative to a voxel that is steadily, quietly occupied.

### 2.5 CIR-Weighted Tomography (ADR-134 Integration)

When ADR-134 CIR is available, the `dominant_delay_sec()` / `dominant_tap_tof_s()` of a link's `Cir` (cir.rs lines 291–309) gives a time-of-flight, hence a distance, for the dominant reflector on that link. The `RfTomographer` weight matrix (tomography.rs line 182, `weight_matrix: Vec<Vec<(usize, f64)>>`) currently weights every voxel on the link path purely by Fresnel-radius proximity (`1.0 − dist/fresnel_radius`). With a CIR delay available, the tracker supplies a *distance prior*: voxels whose distance from TX matches the CIR-implied range get their weight boosted, focusing evidence near the reflector instead of smearing it along the whole ray.

```rust
/// Optional per-link CIR-derived distance prior, applied to the existing
/// Fresnel weights as a multiplicative Gaussian bump centred at the CIR range.
pub struct CirDistancePrior {
    pub link_id: LinkId,
    /// Reflector distance from TX (m), from Cir::dominant_distance_m().
    pub range_m: f64,
    /// Std-dev of the range bump (m), from tap_spacing → distance resolution.
    pub sigma_m: f64,
}
```

The prior is **optional**: when CIR is unavailable (single-antenna fallback, or the `eigenvalue`/CIR feature is off), the tomographer behaves exactly as today. This keeps the change additive and the existing `tomography.rs` tests untouched. The Doppler field of each `Voxel` (`doppler_velocity`) is similarly populated only when CIR phase-rate is available; otherwise it stays 0.0.

### 2.6 `VoxelGate`: BFLD-Gated Voxel Output

The raw `VoxelMap` is identity-leaky: a high-resolution occupancy grid plus per-voxel Doppler can reconstruct a person's trajectory and gait. It must never leave the node un-gated. `VoxelGate::demote` reuses the **monotonic-demotion** pattern of `bfld/src/privacy_gate.rs::PrivacyGate::demote` — it accepts a `PrivacyClass` (from `bfld/src/lib.rs`, classes `Raw(0) → Derived(1) → Anonymous(2) → Restricted(3)`), refuses any *promotion* with `BfldError::InvalidDemote`, and produces progressively coarser views. Like the BFLD gate, demotion is irreversible: once a field is zeroed, the bytes are gone.

```rust
use wifi_densepose_bfld::{BfldError, PrivacyClass};

/// Monotonic voxel-grid demotion, mirroring PrivacyGate::demote (ADR-120).
pub struct VoxelGate;

/// What actually leaves the node after gating.
#[derive(Debug, Clone)]
pub enum GatedVoxelOutput {
    /// Raw(0)/Derived(1): full VoxelMap (local-only by invariant; Raw never
    /// crosses a network sink — same structural rule as BFLD class 0).
    Full(VoxelMap),
    /// Anonymous(2): per-voxel doppler_velocity and confidence cleared to 0;
    /// occupancy retained but quantised. No trajectory reconstruction possible.
    Anonymous(VoxelMap),
    /// Restricted(3): NO voxel grid leaves the node — only an occupancy
    /// histogram (count of voxels per occupancy bucket).
    OccupancyHistogram(Vec<u32>),
}

impl VoxelGate {
    /// Demote the VoxelMap to the target class. Returns InvalidDemote if the
    /// target is a *lower* class number than `current` (i.e. would add info).
    pub fn demote(
        map: &VoxelMap,
        current: PrivacyClass,
        target: PrivacyClass,
    ) -> Result<GatedVoxelOutput, BfldError> {
        if target.as_u8() < current.as_u8() {
            return Err(BfldError::InvalidDemote {
                from: current.as_u8(),
                to: target.as_u8(),
            });
        }
        Ok(match target {
            PrivacyClass::Raw | PrivacyClass::Derived => GatedVoxelOutput::Full(map.clone()),
            PrivacyClass::Anonymous => {
                let mut m = map.clone();
                for v in m.voxels.iter_mut() {
                    v.doppler_velocity = 0.0;   // strip kinematic identity surface
                    v.confidence = 0.0;
                    v.occupancy = quantise(v.occupancy);
                }
                GatedVoxelOutput::Anonymous(m)
            }
            PrivacyClass::Restricted => {
                // The raw VoxelMap never leaves the node at Restricted.
                GatedVoxelOutput::OccupancyHistogram(map.occupancy_histogram(8))
            }
        })
    }
}
```

This mirrors `privacy_gate.rs` field-by-field: where BFLD zeroes `compressed_angle_matrix`/`csi_delta` at Anonymous and `amplitude_proxy`/`phase_proxy` at Restricted, the `VoxelGate` clears `doppler_velocity`/`confidence` at Anonymous and emits only a histogram at Restricted. The control-plane *which* class applies comes from ADR-141 (the named privacy mode and its runtime attestation), not from this ADR — `VoxelGate` is the mechanism, ADR-141 is the policy.

**Anomaly routing.** `EvolutionReport.alerts` (the `CoherenceAlert` / `ChangePoint` / `AnomalyWarn` variants) are not voxel data and are not subject to voxel demotion — they are *typed events*. They route to:

- **ADR-137** fusion contradiction flags: `AnomalyWarn` becomes a contradiction reference (model-occupancy vs perturbation-occupancy) attached to the semantic state record, with the model version and calibration version that produced each side.
- **ADR-139** WorldGraph nodes: a `ChangePoint` updates the environmental digital twin (e.g. a moved-furniture edge), and `CoherenceAlert` marks affected room nodes as needing recalibration.

### 2.7 Interface Boundaries

| Boundary | Direction | Type | Note |
|----------|-----------|------|------|
| `calibration.rs` → tracker | in | `CalibrationDeviationScore` (per link) | drift_score + rms_amplitude_z; no CSI crosses the boundary |
| `field_model.rs` → tracker | in | `CalibrationStatus`, `body_energy: f32`, `estimate_occupancy` | mesh freshness + model occupancy |
| `tomography.rs` → tracker | in | `&OccupancyVolume` (snapshot) | folded into `VoxelMap::ingest` |
| `cir.rs` → tracker | in (optional) | `CirDistancePrior` | distance-weighted evidence; absent ⇒ unchanged behaviour |
| tracker → ADR-137 | out | `EvolutionAlert` (typed) | contradiction flags, evidence references |
| tracker → ADR-139 | out | `EvolutionAlert` (typed) | WorldGraph mutations |
| tracker → network sink | out | `GatedVoxelOutput` only | never the raw `VoxelMap`; gated by `VoxelGate` |

The tracker holds **no raw CSI** and **no payload bytes** — only scores, occupancy estimates, and the voxel grid. The only path to the network is through `VoxelGate::demote`.

---

## 3. Consequences

### 3.1 Positive

- **Single orchestration point.** Five previously-isolated modules (`calibration`, `field_model`, `longitudinal`, `attractor_drift`, `tomography`) gain a coordinator that reads them together. Cross-link change-point detection becomes possible for the first time; no module was ever fed more than one link.
- **Temporal occupancy memory.** A 200-frame occupancy is now distinguishable from a single-frame noise spike via `evidence_count` and converged Bayesian log-odds. The fusion engine (ADR-137) gets per-voxel confidence instead of a binary snapshot threshold.
- **Mesh-wide freshness.** `field_model.rs::check_freshness` only knew one room; `EvolutionTracker` reduces per-link freshness to a mesh `CoherenceAlert`, closing the operational gap ADR-135's per-link drift score left open.
- **Internal contradiction detection.** The occupancy-consistency check turns two independent estimates (eigenstructure vs body-perturbation energy) into an `AnomalyWarn` that ADR-137 can score — a built-in sanity check the pipeline never had.
- **Privacy by construction.** No voxel grid reaches a network sink except through `VoxelGate::demote`, reusing the proven monotonic-demotion invariant from `bfld/src/privacy_gate.rs`. Doppler (the strongest gait-identity surface in a voxel grid) is cleared at Anonymous; the grid itself never leaves at Restricted.
- **Additive CIR integration.** The `CirDistancePrior` is optional; absent CIR, `tomography.rs` behaves identically and its existing tests are untouched.

### 3.2 Negative

- **New persistent state.** The `VoxelMap` is long-lived (one per monitored volume) and adds memory: an 8×8×4 grid is 256 voxels × ~40 bytes ≈ 10 KB — trivial — but a finer 16×16×8 grid is ~2,048 voxels and the decay loop runs every tick over all voxels. Bounded and cheap, but it is new always-on work at 20 Hz.
- **Energy-per-person scale is an install constant.** The occupancy-consistency check's `energy_per_person` is environment-specific and must be set at calibration time; a wrong value produces spurious `AnomalyWarn`s. It is derived from the same empty-room session as ADR-135's baseline.
- **Change-point window tuning.** The 30-frame / 3-link / 2σ defaults are reasoned from ADR-135's thresholds but not yet validated on real multi-room hardware; a noisy mesh could over-trigger `ChangePoint`. Mitigated by requiring majority-of-window hotness per link (§2.2), not a single hot frame.
- **Doppler is gated away early.** Useful kinematic information is cleared at Anonymous. This is intentional (it is the identity surface) but means trajectory analytics must run *before* the gate, inside the trusted node boundary, not on gated output.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `ChangePoint` over-triggers on a noisy mesh (HVAC, sunlight) | Medium | Spurious mesh-recalibration prompts | Majority-of-window per-link hotness + 3-link minimum; ADR-135 drift-confirm still gates auto-recalibration |
| Bayesian voxel converges to a stale occupancy after a person leaves | Medium | A vacated voxel reads occupied for several seconds | Confidence decay with `decay_half_life` for un-updated voxels; the log-odds is pulled toward "free" by subsequent low-density observations |
| `VoxelGate` Anonymous quantisation still leaks coarse trajectory | Low | Re-identification from coarse grid over time | Restricted mode (histogram only) for untrusted sinks; ADR-141 control plane chooses class per sink |
| CIR distance prior misplaces evidence when the dominant tap is the direct path, not the body | Medium | Evidence concentrated at the wall, not the person | Prior is multiplicative on existing Fresnel weights (cannot create evidence where the ray does not pass); body-perturbation energy still gates whether a voxel is occupied at all |
| Occupancy-consistency false `AnomalyWarn` from a wrong `energy_per_person` | Medium | Noise into ADR-137 contradiction stream | Tolerance default of 1 person; calibrate `energy_per_person` during the empty-room session and re-derive on `ChangePoint` |

---

## 4. Alternatives Considered

### 4.1 Make `OccupancyVolume` Stateful In-Place (Rejected)

The simplest path is to add `confidence`/`last_update_ns`/`evidence_count` fields directly to `tomography.rs::OccupancyVolume` and have `reconstruct()` mutate a retained instance. Rejected: `OccupancyVolume` is currently a pure output of `reconstruct()` and is cloned/inspected by tests that assume it is a snapshot (e.g. `test_zero_attenuation_empty_room` asserts `occupied_count == 0` for a fresh volume). Conflating snapshot and persistent state would break that contract and entangle the solver with temporal policy. The `VoxelMap` keeps the solver pure and the temporal state separate.

### 4.2 One Tracker Per Link (Rejected)

Keep the per-link isolation and run an independent tracker per link. Rejected: this is the *current* situation and is exactly what makes cross-link change-point and mesh freshness impossible. The whole value of an "evolution tracker" is the cross-link view.

### 4.3 Kalman / Particle Filter Per Voxel (Rejected for Now)

A per-voxel Kalman or particle filter would model occupancy *and* velocity jointly with a proper motion model. Rejected as overkill for a coarse 8×8×4 grid at the current sensing resolution: the log-odds occupancy grid is the standard, cheap, commutative choice (Thrun et al., 2005) and integrates trivially with the existing ISTA output. A motion-model filter belongs in the pose tracker (`pose_tracker.rs` already runs a 17-keypoint Kalman), not in the coarse occupancy grid. Revisit if voxel resolution increases materially.

### 4.4 Emit Raw VoxelMap and Gate Downstream (Rejected)

Let the raw `VoxelMap` leave the node and gate it at the consumer. Rejected on the same structural-invariant grounds as BFLD class 0 (`Raw` is local-only by invariant I1, `bfld/src/lib.rs`): once raw identity-leaky voxel data crosses a network boundary it cannot be un-leaked. Gating must happen *before* the sink, inside the node, which is exactly what `VoxelGate::demote` enforces.

### 4.5 New Privacy Mechanism for Voxels (Rejected)

Design a bespoke voxel-privacy scheme independent of BFLD. Rejected: the monotonic-demotion invariant in `privacy_gate.rs` is already proven and audited (ADR-120), and ADR-141 already defines the named-mode control plane. Reusing `PrivacyClass` and the `demote` pattern means one privacy model across the whole system, one set of attestation tests, and no second mechanism to audit.

---

## 5. Testing and Acceptance

### 5.1 Unit Tests

**T1 — Mesh freshness aggregation.** Feed `LinkObservation`s with mixed `CalibrationStatus` (`Fresh`, `Stale`, `Expired`). Assert `mesh_freshness` is the worst case and `stale_links` lists exactly the non-fresh links, and a `CoherenceAlert` is emitted iff any link is Stale/Expired.

**T2 — Cross-link change-point fires at 3 links.** Push 30-frame z-windows where exactly 2 links exceed 2.0σ for a majority of the window: assert no `ChangePoint`. Add a 3rd: assert `ChangePoint { links }` fires and names all three.

**T3 — Change-point does NOT fire on a single sustained link.** One link hot for the full window, all others quiet: assert no `ChangePoint` (this is ADR-135's single-link staleness domain, not an environment change).

**T4 — Occupancy-consistency.** Set `model_occupancy = 1`, supply body energy implying 1 person: assert no `AnomalyWarn`. Supply body energy implying 3 persons: assert `AnomalyWarn { model: 1, perturbation: 3 }` and `occupancy_disagreement == true`.

**T5 — VoxelMap evidence accumulation.** Ingest 200 identical occupied volumes for one voxel and 1 occupied volume for another. Assert the 200-frame voxel has `evidence_count == 200`, `occupancy > 0.95`, and is NOT in `low_confidence_indices(5)`; the 1-frame voxel IS in `low_confidence_indices(5)` and has `occupancy` far from 1.0.

**T6 — Low-confidence flagging at threshold.** Ingest exactly 4 frames for a voxel: assert it is low-confidence. Ingest a 5th: assert it leaves `low_confidence_indices(5)`.

**T7 — Confidence decay.** Ingest a voxel to high confidence, then ingest `decay_half_life` ticks where that voxel is not touched: assert its `confidence` halved while `occupancy` (last estimate) is retained.

**T8 — Per-voxel Welford variance.** Ingest densities `[0.9, 0.1, 0.9, 0.1, ...]` (noisy) vs `[0.5, 0.5, ...]` (steady) with equal mean: assert the noisy voxel has higher `density_variance()` and consequently lower `confidence`.

**T9 — VoxelGate monotonicity.** `demote(map, Anonymous, Derived)` returns `BfldError::InvalidDemote { from: 2, to: 1 }`. `demote(map, Derived, Anonymous)` succeeds and the returned `VoxelMap` has every `doppler_velocity == 0.0` and `confidence == 0.0`.

**T10 — VoxelGate Restricted emits no grid.** `demote(map, Anonymous, Restricted)` returns `GatedVoxelOutput::OccupancyHistogram` and never a `VoxelMap` — assert the variant is the histogram and its length equals the requested bucket count.

**T11 — CIR prior is additive.** Run `RfTomographer::reconstruct()` with and without a `CirDistancePrior`; assert the no-prior path is bit-identical to current `tomography.rs` output (existing tests unchanged), and the with-prior path concentrates density nearer the CIR range.

### 5.2 Integration Test (gated, `#[cfg(feature = "hardware-test")]`)

**T12 — Real multistatic mesh (COM9 + cognitum-seed-1).** With an empty room, run 30 s and assert no `ChangePoint`, `mesh_freshness == Fresh`, and the `VoxelMap` has all voxels at `occupancy < 0.2`. Walk through: assert occupied voxels rise above 0.8 along the path, `evidence_count` grows, and walking *out* lets confidence decay. Move a chair and leave: assert a `ChangePoint` fires within 1.5 s and the affected links are named.

### 5.3 Determinism / Witness (CI-compatible, extends ADR-028)

**T13 — Deterministic VoxelMap hash.** Build a fixed 600-tick synthetic occupancy stream (seed=42), ingest into a `VoxelMap`, and SHA-256 the serialised voxel state. Record under `archive/v1/data/proof/expected_features.sha256` as `voxelmap_evidence_v1`; `verify.py` regenerates and asserts the hash. Mirrors ADR-135's `calibration_nvs_baseline_v1` proof methodology.

### 5.4 Acceptance Criteria

1. `EvolutionTracker::tick()` runs in < 1 ms for an 8×8×4 grid and 12 links (20 Hz budget is 50 ms; ample headroom).
2. Change-point fires iff ≥ `change_point_min_links` exceed `change_point_sigma` for a window majority (T2, T3).
3. A voxel below `min_evidence_frames` is always reported low-confidence (T5, T6).
4. No code path emits a raw `VoxelMap` to a network sink without `VoxelGate::demote` (enforced by the interface boundary in §2.7; `VoxelGate` is the only public constructor of `GatedVoxelOutput`).
5. `VoxelGate::demote` is monotonic: a promotion attempt always returns `BfldError::InvalidDemote` (T9).
6. Every emitted semantic state (occupancy + alerts) carries references to signal evidence (the `LinkObservation` set), model version (FieldModel SVD generation), calibration version (`BaselineCalibration.captured_at_unix_s`), and privacy decision (`VoxelGate` target class).
7. The CIR distance prior is provably additive — the no-prior reconstruction is unchanged (T11).

---

## 6. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-030 (Persistent Field Model) | **Extended**: adds the cross-link orchestrator and temporal voxel layer ADR-030 left unspecified; consumes `FieldModel::estimate_occupancy` and `CalibrationStatus` |
| ADR-134 (First-Class CIR) | **Integrated (optional)**: `Cir::dominant_distance_m()` feeds the `CirDistancePrior` into the tomography weight matrix for distance-based evidence weighting |
| ADR-135 (Empty-Room Baseline) | **Prerequisite/consumer**: reads `CalibrationDeviationScore.drift_score`; the cross-link change-point is the spatial complement to ADR-135's single-link staleness; shares the `W=300` window and recalibration triggers |
| ADR-120 (BFLD Privacy Classes) | **Reused**: `VoxelGate::demote` is a direct application of the `PrivacyGate::demote` monotonic invariant and `PrivacyClass` enum |
| ADR-141 (BFLD Privacy Control Plane) | **Policy provider**: ADR-141 chooses *which* `PrivacyClass` applies per sink and attests it at runtime; this ADR supplies the voxel mechanism |
| ADR-137 (Fusion Quality Scoring) | **Consumer**: `AnomalyWarn` (occupancy disagreement) becomes a contradiction flag with evidence references in the semantic state record |
| ADR-139 (WorldGraph) | **Consumer**: `ChangePoint` and `CoherenceAlert` mutate the environmental digital twin (moved-furniture edges, room recalibration markers) |
| ADR-136 (Streaming Engine) | **Substrate**: `EvolutionReport`/`EvolutionAlert` are typed stage outputs flowing through the streaming engine's frame contracts |
| ADR-084 / ADR-118 | **Related**: longitudinal drift and persistence context for the per-person baselines referenced by the tracker |

---

## 7. References

### Production Code

- `v2/crates/wifi-densepose-signal/src/ruvsense/tomography.rs` — `RfTomographer`, `OccupancyVolume`, `weight_matrix` to gain the optional CIR prior; `VoxelMap` is its temporal companion
- `v2/crates/wifi-densepose-signal/src/ruvsense/field_model.rs` — `WelfordStats` (reused for per-voxel variance), `CalibrationStatus`, `estimate_occupancy`, `check_freshness`
- `v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs` — `CalibrationDeviationScore.drift_score` consumed per link (ADR-135)
- `v2/crates/wifi-densepose-signal/src/ruvsense/longitudinal.rs` — `PersonalBaseline`, `EmbeddingHistory` referenced by handle, not copied
- `v2/crates/wifi-densepose-signal/src/ruvsense/attractor_drift.rs` — `AttractorDriftAnalyzer::analyze` regime changes folded into evolution state
- `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs` — `Cir::dominant_distance_m()` / `dominant_tap_tof_s()` source of the distance prior
- `v2/crates/wifi-densepose-bfld/src/privacy_gate.rs` — `PrivacyGate::demote` monotonic-demotion pattern reused by `VoxelGate`
- `v2/crates/wifi-densepose-bfld/src/lib.rs` — `PrivacyClass` (Raw/Derived/Anonymous/Restricted), `BfldError::InvalidDemote`
- `archive/v1/data/proof/verify.py` — deterministic proof chain; `voxelmap_evidence_v1` hash extension
- `archive/v1/data/proof/expected_features.sha256` — hash entry to be added

### External References

- Moravec, H. & Elfes, A. (1985). "High Resolution Maps from Wide Angle Sonar." *Proc. IEEE ICRA*. — Origin of the occupancy-grid log-odds update used per voxel.
- Thrun, S., Burgard, W. & Fox, D. (2005). *Probabilistic Robotics*. MIT Press. Ch. 9 (Occupancy Grid Mapping). — Standard commutative log-odds occupancy update; basis for `VoxelMap::ingest`.
- Welford, B.P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics*, 4(3), 419–420. — Per-voxel mean/variance accumulation (same form as `field_model.rs::WelfordStats`).
- Wilson, J. & Patwari, N. (2010). "Radio Tomographic Imaging with Wireless Networks." *IEEE Trans. Mobile Computing*, 9(5). — Tomographic inversion basis for `tomography.rs`, extended here with temporal evidence accumulation.


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `1f8e180d6`, issue #846): `EvolutionTracker` (cross-link change-point), `TemporalVoxel` (Bayesian log-odds occupancy + confidence floor), and `VoxelGate` (privacy demotion to a histogram). 6 tests.

**Integration glue -- not yet on the live path:** driving `field_model.estimate_occupancy()` consistency checks and CIR-peak-delay distance weighting from live signals; routing detected anomalies to ADR-137 contradiction flags.

**Trust contribution:** *the room changed* is inferred from multi-link consensus (not one noisy link), and occupancy can be blurred to an aggregate histogram under privacy.
