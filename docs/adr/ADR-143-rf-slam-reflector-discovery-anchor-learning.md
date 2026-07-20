# ADR-143: RF SLAM v2: Persistent Reflector Discovery and Dynamic Anchor Learning

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block, v1 fixed-map default; v2 dataset-gated — see Implementation Status, commit `2d4f3dea5`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (`ruvsense/field_model.rs`, new `ruvsense/rf_slam.rs`); `wifi-densepose-mat` (`tracking/kalman.rs`, `localization/triangulation.rs`); `wifi-densepose-geo`; `wifi-densepose-ruvector` (`mat/triangulation.rs`) |
| **Relates to** | ADR-029 (RuvSense Multistatic), ADR-030 (Persistent Field Model), ADR-042 (Coherent Human Channel Imaging), ADR-134 (First-Class CIR Support), ADR-136 (RuView Streaming Engine), ADR-138 (LinkGroup / ArrayCoordinator), ADR-139 (WorldGraph), ADR-141 (BFLD Privacy Control Plane), ADR-142 (Evolution Tracker / Temporal VoxelMap) |

---

## 1. Context

### 1.1 The Gap

The codebase has the two ingredients RF SLAM needs — a delay-domain CIR per link and a per-link statistical baseline — but nothing that converts them into a *map of where the reflectors physically are*, and nothing that *learns* anchor positions from data instead of taking them as fixed configuration.

Grepping the workspace confirms the absence and the substrate:

- **CIR exists, geometry does not.** `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs` produces a `Cir` (lines 263–286) with `taps: Vec<Complex32>`, `tap_spacing_sec`, `dominant_tap_idx`, `dominant_tap_ratio`, `active_tap_count`, and `rms_delay_spread_s`. This is a per-link delay profile. There is no code that takes the *separation* between taps across two or more links and triangulates a reflector's `(x, y, z)` position, nor any code that tracks a tap cluster's position over hours. `Cir::dominant_distance_m()` (line 297) converts the dominant tap delay to a one-link range, but a single range is a sphere, not a point.

- **The field model centres on a mean, not a reflector list.** `ruvsense/field_model.rs` (`FieldModel`, `FieldNormalMode`) computes a per-link amplitude baseline (`baseline: Vec<Vec<f64>>`, line 265), an SVD over the per-subcarrier covariance, environmental eigenmodes, `variance_explained` (line 272), and a Marcenko-Pastur `baseline_eigenvalue_count` (line 278). It answers "how much energy is structured static environment" — it never answers "*which physical objects* produce that energy and *where are they*." There is no `Reflector`, no `anchor`, no spatial position in the entire module.

- **Localisation assumes fixed anchors.** `wifi-densepose-mat/src/localization/triangulation.rs` (`TriangulationConfig`, `Triangulator`, lines 7–88) takes `sensors: &[SensorPosition]` as given input and trilaterates a *person* from RSSI/ToA. `wifi-densepose-ruvector/src/mat/triangulation.rs::solve_triangulation()` (lines 28–53) takes `ap_positions: &[(f32, f32)]` as a fixed argument and solves a linearised TDoA system via `NeumannSolver`. Both treat anchor positions as configuration the operator must enter by hand. Neither has any path to *discover* an anchor (a static reflector or an AP) from the signal.

- **The tracker tracks people, not furniture.** `wifi-densepose-mat/src/tracking/kalman.rs` (`KalmanState`, lines 26–35) is a 6-state constant-velocity filter for a *survivor* position. There is no per-reflector tracker, no notion of a slow-moving (furniture) versus fast-moving (person) target, and no displacement-rate estimate.

- **`wifi-densepose-geo` has scene types but no RF objects.** `wifi-densepose-geo/src/types.rs` exposes `GeoPoint`, `GeoBBox`, `GeoRegistration`, `GeoScene`, `OsmFeature` — outdoor geospatial registration. There is no indoor reflector or anchor type.

So the gap is precise: **the system can measure multipath delay per link and can tell static from dynamic energy, but it cannot place reflectors in a room coordinate frame, cannot decide which reflectors are stable enough to use as localisation anchors, and cannot notice when the furniture has moved.** ADR-030 (§the persistent field model) and ADR-042 (CHCI) both assume a known room geometry; neither specifies how that geometry is acquired.

### 1.2 What "RF SLAM" Means Here (and What v1 Already Is)

SLAM — Simultaneous Localisation And Mapping — in the RF-sensing context means: *while* tracking moving targets (localisation), also *build and refine* the map of static scatterers (mapping). This ADR is explicitly **v2**. There is a **v1** that this ADR commits to shipping *first*:

- **RF SLAM v1 (ship now):** 3 fixed APs at operator-entered positions + a single static-reflector assumption. This is essentially what `triangulation.rs` and `solve_triangulation()` already do once the operator types in AP coordinates. v1 requires no new discovery code — it requires only wiring the fixed positions into the WorldGraph as immutable `object_anchor` nodes (ADR-139). v1 is honest about its limitation: it cannot adapt to a moved sofa.

- **RF SLAM v2 (this ADR, feature-flagged):** infer reflector positions from CIR tap separation, learn which reflectors are stable enough to serve as anchors, detect topology change, and estimate furniture movement — all gated behind a feature flag until a 7-day validation dataset is collected.

The reason for the two-tier rollout is the same reason ADR-135 makes recalibration operator-initiated: **there is no oracle for ground truth in a live home.** A reflector-discovery algorithm that places a wall 30 cm off does not announce its error; it silently degrades every downstream localisation. v2 must prove itself on 7 days of paired data before it is allowed to overwrite the v1 fixed map.

### 1.3 Why CIR Tap Separation Gives Geometry

For a link between TX at `p_tx` and RX at `p_rx`, a reflector at `p_r` produces a delayed copy of the direct path. The excess delay of that tap, relative to the direct (line-of-sight) tap, is:

```
Δτ = ( |p_tx − p_r| + |p_r − p_rx| − |p_tx − p_rx| ) / c
```

`Δτ` is exactly `(tap_idx − dominant_tap_idx) × tap_spacing_sec` from the `Cir` struct. A single link constrains the reflector to a **prolate spheroid** with foci at `p_tx` and `p_rx` (constant bistatic range = constant excess delay). Two links with shared geometry intersect their spheroids; three or more over-determine the reflector position and let least-squares resolve `(x, y, z)`. This is the dual of `solve_triangulation()` in `ruvector/mat/triangulation.rs`: that function solves for a person given fixed APs; reflector discovery solves for a static scatterer given the (now known, from v1) APs and the per-link excess-delay taps.

The bistatic-range geometry only resolves a point if the multipath cluster is **persistent and coherent** across the observation window. Hence discovery is gated on temporal coherence (the same von Mises phase-concentration machinery from ADR-135) and on the room genuinely being in a static regime (the ADR-030 Marcenko-Pastur threshold — if `estimate_occupancy() > 0`, the room is occupied and discovery is suspended).

### 1.4 Pipeline Position

```
Per-link CSI (ADR-135 baseline-subtracted, ADR-138 LinkGroup-grouped)
  → CirEstimator::estimate()                 (ADR-134)  → Cir { taps, ... }
  → FieldModel.feed_calibration / SVD        (ADR-030)  → variance_explained, MP count
  → ReflectorTracker::observe()              ← NEW (rf_slam.rs)
        · extract excess-delay taps per link
        · associate taps to reflector tracks (per-reflector Kalman)
        · bistatic multilateration → reflector (x,y,z) + covariance
        · coherence-gate: accept only persistent, von-Mises-concentrated taps
  → AnchorLearner::classify()                ← NEW
        · cluster persistent reflectors → walls / large objects
        · reject mobile reflectors (tap migration > 0.5 m/day)
        · emit StaticAnchor set
  → TopologyMonitor::tick()                  ← NEW
        · variance_explained drop > 15% / 4h  OR  covariance-rank change
        → BaselineTopologyChange event → recalibration trigger (ADR-135 §2.6)
  → FurnitureMovementEstimator::tick()       ← NEW
        · per-reflector tap-migration rate → hourly displacement ± 0.5 m
  → WorldGraph::upsert(object_anchor)        (ADR-139)  → persisted via RVF
```

v2 discovery code (everything marked NEW) is compiled behind `#[cfg(feature = "rf-slam-v2")]` and is a no-op at runtime unless `RfSlamConfig::enabled` is also set. v1's fixed-AP map flows straight to `WorldGraph::upsert(object_anchor)` with immutable positions.

---

## 2. Decision

### 2.1 v2 Reflector Discovery from CIR Tap Separation + Temporal Coherence

A reflector is discovered, not configured. The `ReflectorTracker` ingests one `Cir` per link per cycle (from ADR-138's `LinkGroup`, which guarantees the links it groups share a clock-quality tier so their delays are comparable) and maintains a set of reflector tracks.

**Discovery preconditions (all must hold for a cycle to contribute to discovery):**

1. **Room is static.** `FieldModel::estimate_occupancy()` (field_model.rs:741) returns 0 for the cycle's recent-frame window, *and* the ADR-030 Marcenko-Pastur significant-eigenvalue count equals the calibrated `baseline_eigenvalue_count`. If the room is occupied, the cycle is dropped for discovery (but still used for localisation). This reuses the existing eigenvalue gate rather than inventing a new occupancy detector.
2. **Tap is coherent over the window.** For a candidate tap index `g` on a link, the complex tap value `taps[g]` must have circular phase variance below `coherence_max` (default 0.15) over a rolling 24–72 h window, computed with the running `sin`/`cos` accumulator from ADR-135 §2.2 (von Mises projection). A tap whose phase wanders is a transient (a passing person's residual, an HVAC vane), not a static scatterer.
3. **Tap exceeds the noise floor.** `|taps[g]|` ≥ `1%` of the dominant tap — reusing the `active_tap_count` definition (cir.rs:278) so the discovery and CIR modules agree on what "a tap" is.

**Multilateration.** Each accepted tap gives one bistatic-range constraint per link. With ≥3 links observing a common scatterer (associated by excess-delay consistency, §2.4), the reflector position is solved by the **same Neumann-series least-squares machinery** as person localisation — `wifi-densepose-ruvector/src/mat/triangulation.rs::solve_triangulation()` is generalised so it can be fed reflector bistatic ranges instead of person TDoA. The reflector position carries a 3×3 covariance from the residual.

```rust
// v2/crates/wifi-densepose-signal/src/ruvsense/rf_slam.rs

use num_complex::Complex32;
use crate::ruvsense::cir::Cir;
use crate::ruvsense::field_model::WelfordStats;

/// A persistent static scatterer inferred from CIR tap separation.
#[derive(Debug, Clone)]
pub struct Reflector {
    /// Stable identifier assigned at first confident discovery.
    pub id: ReflectorId,
    /// Estimated room-frame position (metres). `None` until ≥3 links concur.
    pub position_m: Option<[f64; 3]>,
    /// 3×3 position covariance (metres²), row-major. `None` until localised.
    pub position_cov: Option<[[f64; 3]; 3]>,
    /// Per-observing-link excess delay (s) relative to that link's direct tap.
    pub excess_delay_s: Vec<(LinkId, f64)>,
    /// Welford amplitude statistics of the tap magnitude over the window.
    pub amp_stats: WelfordStats,
    /// Circular phase variance over the window ∈ [0, 1]; <0.15 ⇒ coherent.
    pub phase_circular_variance: f32,
    /// Number of discovery cycles this reflector has been continuously observed.
    pub persistence_cycles: u64,
    /// First-seen / last-seen UTC (Unix seconds).
    pub first_seen_unix_s: i64,
    pub last_seen_unix_s: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReflectorId(pub u64);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LinkId(pub u32);

#[derive(Debug, thiserror::Error)]
pub enum RfSlamError {
    #[error("RF SLAM v2 disabled (set RfSlamConfig.enabled and the rf-slam-v2 feature)")]
    Disabled,
    #[error("Room is occupied; discovery suspended for this cycle")]
    RoomOccupied,
    #[error("Insufficient observing links: need {needed}, have {got}")]
    InsufficientLinks { needed: usize, got: usize },
    #[error("Multilateration failed to converge")]
    NoConverge,
    #[error("Validation dataset not yet present: {0}")]
    ValidationGateClosed(String),
}

#[derive(Debug, Clone)]
pub struct RfSlamConfig {
    /// Master switch. False ⇒ all v2 entry points return `Disabled`.
    pub enabled: bool,
    /// Min links concurring before a reflector position is emitted. Default 3.
    pub min_links: usize,
    /// Max circular phase variance for a coherent tap. Default 0.15.
    pub coherence_max: f32,
    /// Coherence-window length in hours. Default 48 (range 24–72).
    pub coherence_window_h: f64,
    /// Mobile-reflector rejection threshold (metres/day). Default 0.5.
    pub mobile_reject_m_per_day: f64,
    /// variance_explained relative-drop fraction triggering topology change. Default 0.15.
    pub topology_var_drop: f64,
    /// Window over which the drop is measured (hours). Default 4.0.
    pub topology_window_h: f64,
}

impl Default for RfSlamConfig {
    fn default() -> Self {
        Self {
            enabled: false, // v2 is OFF until the 7-day dataset is validated.
            min_links: 3,
            coherence_max: 0.15,
            coherence_window_h: 48.0,
            mobile_reject_m_per_day: 0.5,
            topology_var_drop: 0.15,
            topology_window_h: 4.0,
        }
    }
}

/// Maintains reflector tracks across discovery cycles.
pub struct ReflectorTracker {
    config: RfSlamConfig,
    reflectors: Vec<Reflector>,
    next_id: u64,
}

impl ReflectorTracker {
    pub fn new(config: RfSlamConfig) -> Self;

    /// Ingest one CIR per observing link for the current cycle.
    ///
    /// `cirs`: `(LinkId, &Cir)` for every link in the ADR-138 LinkGroup.
    /// `occupied`: result of `FieldModel::estimate_occupancy() > 0`.
    ///
    /// Returns the set of reflectors updated or newly created this cycle.
    /// Returns `RoomOccupied` (no-op) if `occupied`, `Disabled` if not enabled.
    pub fn observe(
        &mut self,
        cirs: &[(LinkId, &Cir)],
        occupied: bool,
        now_unix_s: i64,
    ) -> Result<Vec<ReflectorId>, RfSlamError>;

    /// Current confident reflector set (position resolved, coherent).
    pub fn reflectors(&self) -> &[Reflector];
}
```

### 2.2 Static-Anchor Learning by Furniture Clustering

Not every reflector is a good localisation anchor. A wall is; a houseplant that sways is not; a chair that gets pushed in twice a day is not. The `AnchorLearner` partitions the reflector set into **static anchors** (usable for the v2 map) and **mobile reflectors** (tracked but excluded from the anchor set).

**Classification rules:**

| Class | Criterion | Rationale |
|-------|-----------|-----------|
| `StaticAnchor` | `phase_circular_variance < coherence_max` AND tap-migration rate `< mobile_reject_m_per_day` (0.5 m/day) AND `persistence_cycles` spans ≥ 24 h | Walls and large fixed objects (cabinet, fridge) produce a coherent tap whose position does not drift day to day. |
| `MobileReflector` | tap-migration rate ≥ 0.5 m/day | Furniture that is rearranged; tracked for movement inference (§2.4) but never used as a localisation anchor because its position is not trustworthy as a reference. |
| `TransientCandidate` | `phase_circular_variance ≥ coherence_max` OR `persistence_cycles` < 24 h | Not yet confident; held in a candidate buffer, promoted or aged out. |

**Spatial clustering into furniture categories.** Static anchors are clustered in room-frame `(x, y, z)` using density-based clustering (DBSCAN-style, `ε = 0.3 m`, `minPts = 2`). A cluster's bounding box and surface-normal (from the spread of contributing links' bistatic geometry) categorise it:

- A planar cluster spanning ≥ 1.5 m with a consistent normal → `Wall`.
- A compact cluster (< 1.0 m extent) at a fixed height → `LargeObject` (appliance, cabinet).

Categories are advisory metadata on the WorldGraph node (§2.5), not load-bearing for localisation — localisation uses the anchor *positions*, the category labels them for the operator and for ADR-140 semantic state records.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorClass { StaticAnchor, MobileReflector, TransientCandidate }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FurnitureCategory { Wall, LargeObject, Unknown }

#[derive(Debug, Clone)]
pub struct StaticAnchor {
    pub reflector_id: ReflectorId,
    pub position_m: [f64; 3],
    pub position_cov: [[f64; 3]; 3],
    pub category: FurnitureCategory,
    /// Tap-migration rate (metres/day) over the coherence window.
    pub migration_m_per_day: f64,
}

pub struct AnchorLearner { config: RfSlamConfig }

impl AnchorLearner {
    pub fn new(config: RfSlamConfig) -> Self;

    /// Classify the current reflector set and return the static anchors.
    pub fn classify(&self, reflectors: &[Reflector]) -> Vec<(ReflectorId, AnchorClass)>;

    /// Build the static-anchor set with spatial clustering + categorisation.
    pub fn learn_anchors(&self, reflectors: &[Reflector]) -> Vec<StaticAnchor>;
}
```

### 2.3 Topology-Change Detection via Variance and Covariance Rank

A reflector map is only valid while the room topology is unchanged. v2 detects topology change with two ADR-030 / ADR-134 signals, reusing values the field model already computes:

1. **`variance_explained` drop.** `FieldNormalMode.variance_explained` (field_model.rs:272) is the fraction of CSI variance captured by the calibrated environmental modes. When the furniture map shifts, the calibrated modes no longer fit and `variance_explained` falls. **Trigger: a relative drop > 15% sustained over a 4-hour window.** (Relative, not absolute — a room with `variance_explained = 0.8` dropping to `0.68` is the same proportional shift as `0.5 → 0.425`.)
2. **Covariance rank change.** The Marcenko-Pastur significant-eigenvalue count (`baseline_eigenvalue_count`, field_model.rs:278/589) is the structural rank of the static channel. A new fixed scatterer adds a mode; a removed one drops a mode. A *sustained* change in the MP count while the room is unoccupied (occupancy gate from §2.1) indicates a topology change, not a person.

Both conditions feed a `TopologyMonitor` that, on confirmed change, emits `BaselineTopologyChange` and routes it to the **existing** recalibration trigger described in ADR-135 §2.6 (`recalibrate_on_drift`). v2 does not invent a second recalibration path; it provides a more specific *cause* (topology change vs amplitude drift) than ADR-135's amplitude-only z-score drift.

```rust
#[derive(Debug, Clone)]
pub enum TopologyEvent {
    /// variance_explained dropped > config.topology_var_drop over the window.
    VarianceCollapse { from: f64, to: f64, window_h: f64 },
    /// Marcenko-Pastur significant-eigenvalue count changed while unoccupied.
    RankChange { from: usize, to: usize },
}

pub struct TopologyMonitor { config: RfSlamConfig, /* rolling history */ }

impl TopologyMonitor {
    pub fn new(config: RfSlamConfig) -> Self;

    /// Feed the current field-model summary for this cycle.
    /// Returns `Some(event)` when a topology change is confirmed.
    pub fn tick(
        &mut self,
        variance_explained: f64,
        mp_significant_count: usize,
        occupied: bool,
        now_unix_s: i64,
    ) -> Option<TopologyEvent>;
}
```

### 2.4 Furniture-Movement Inference

A `MobileReflector` is not noise — its *displacement over time* is information ("the chair moved 0.4 m at 14:00"). The `FurnitureMovementEstimator` tracks each reflector's tap-migration rate and emits hourly displacement estimates with a **0.5 m confidence band**, using ADR-042 CHCI cross-link consistency to reject spurious migrations.

**Per-reflector position tracking.** Each reflector gets a slow-dynamics Kalman filter. We **reuse the constant-velocity `KalmanState` from `wifi-densepose-mat/src/tracking/kalman.rs`** (the same 6-state `[px,py,pz,vx,vy,vz]` filter used for survivors, kalman.rs:26) but parameterised for furniture timescales: a tiny process-noise variance (`process_noise_var ≈ 1e-6 (m/s²)²`, vs the human-tracking value) so the filter only believes motion that persists across many hours. The velocity components, integrated over an hour, give the hourly displacement.

**CHCI cross-link consistency gate.** A genuine furniture move shifts the excess-delay tap *consistently* across every link that observes that reflector (the geometry changes for all of them coherently). A spurious migration (multipath self-interference, a transient) shows up on one link only. ADR-042's coherent cross-link phase machinery scores this consistency: a displacement is emitted only if ≥ `min_links` links agree on the direction of tap migration within the 0.5 m band. Reflectors that fail the consistency check have their displacement suppressed (reported as "unstable, no estimate").

```rust
#[derive(Debug, Clone)]
pub struct DisplacementEstimate {
    pub reflector_id: ReflectorId,
    /// Displacement vector this hour (metres, room frame).
    pub displacement_m: [f64; 3],
    /// 1-σ confidence radius (metres); ≤ 0.5 by construction or estimate suppressed.
    pub confidence_radius_m: f64,
    /// Number of links agreeing on the migration direction (CHCI consistency).
    pub consistent_links: usize,
    pub hour_unix_s: i64,
}

pub struct FurnitureMovementEstimator { config: RfSlamConfig /* per-reflector KalmanState */ }

impl FurnitureMovementEstimator {
    pub fn new(config: RfSlamConfig) -> Self;

    /// Advance one cycle; returns any hourly displacement estimates that
    /// completed this tick. CHCI-inconsistent reflectors are omitted.
    pub fn tick(
        &mut self,
        reflectors: &[Reflector],
        now_unix_s: i64,
    ) -> Vec<DisplacementEstimate>;
}
```

### 2.5 Persistence into the WorldGraph via RVF

Discovered reflectors, anchor assignments, and calibration timestamps are persisted as **`object_anchor` nodes in the ADR-139 WorldGraph** (the typed petgraph environmental digital twin), serialised through RVF. This is the single source of truth for room geometry that ADR-030, ADR-042, and the localisation triangulators all read.

Each `object_anchor` node carries the full evidence-and-provenance chain so the project rule "every semantic state traces to signal evidence + model version + calibration version + privacy decision" holds:

| Field | Source | Trace role |
|-------|--------|-----------|
| `position_m`, `position_cov` | bistatic multilateration (§2.1) | signal evidence (CIR taps) |
| `class`, `category` | `AnchorLearner` (§2.2) | derived label |
| `migration_m_per_day` | `FurnitureMovementEstimator` (§2.4) | temporal evidence |
| `discovery_model_version` | `rf_slam.rs` semantic version | **model version** |
| `calibration_version` | ADR-135 baseline `captured_at_unix_s` + device_id | **calibration version** |
| `first_seen / last_seen / last_topology_event` | tracker timestamps | provenance |
| `privacy_decision` | ADR-141 BFLD mode at time of write | **privacy decision** |
| `evidence_refs` | CIR cycle ids contributing to the position fit | **signal evidence references** |

ADR-142's Evolution Tracker / Temporal VoxelMap consumes the same `object_anchor` stream to aggregate reflector evidence into the room voxel map over time; ADR-136's streaming engine carries reflector updates as a stage output frame.

```rust
/// Snapshot written to the WorldGraph as an `object_anchor` node (ADR-139).
#[derive(Debug, Clone)]
pub struct ObjectAnchorRecord {
    pub reflector_id: ReflectorId,
    pub position_m: [f64; 3],
    pub position_cov: [[f64; 3]; 3],
    pub class: AnchorClass,
    pub category: FurnitureCategory,
    pub migration_m_per_day: f64,
    pub discovery_model_version: String,   // model version
    pub calibration_version: String,       // ADR-135 baseline id (device_id@captured_at)
    pub privacy_decision: String,          // ADR-141 BFLD mode label
    pub evidence_refs: Vec<u64>,           // contributing CIR cycle ids
    pub first_seen_unix_s: i64,
    pub last_seen_unix_s: i64,
}
```

**The v1/v2 feature gate, concretely.** All of §2.1–§2.5 is compiled under `#[cfg(feature = "rf-slam-v2")]` and is dormant unless `RfSlamConfig::enabled == true`. With the feature off (the default), `WorldGraph` is populated *only* by the v1 path: 3 fixed APs at operator-entered positions written as immutable `object_anchor` nodes (`class = StaticAnchor`, `category = Unknown`, `migration_m_per_day = 0.0`, `discovery_model_version = "v1-fixed"`), plus a single static-reflector assumption (one inferred wall reflector from the dominant non-direct tap, also immutable). v2 may be enabled only after the validation gate (§2.7) confirms a 7-day dataset exists and v2's discovered anchors agree with ground truth within 0.5 m.

### 2.6 Interface Boundaries

| Module | Reads | Writes | Boundary contract |
|--------|-------|--------|-------------------|
| `ruvsense/rf_slam.rs` (NEW) | `Cir` (cir.rs), `FieldModel` occupancy + `variance_explained` + MP count (field_model.rs), ADR-138 `LinkGroup` membership | `Reflector`, `StaticAnchor`, `TopologyEvent`, `DisplacementEstimate`, `ObjectAnchorRecord` | Pure compute; no I/O. `observe()` is `&mut self`, single-threaded per LinkGroup (same convention as ADR-135 `CalibrationRecorder`). |
| `ruvector/mat/triangulation.rs` | reflector bistatic ranges (generalised input) | reflector `(x,y)`/`(x,y,z)` | `solve_triangulation()` generalised to accept either person TDoA or reflector bistatic-range constraints; existing person-localisation signature preserved (additive, non-breaking). |
| `mat/tracking/kalman.rs` | per-reflector observations | per-reflector filtered position/velocity | `KalmanState` reused unchanged; only `process_noise_var` is retuned for furniture timescales by the caller. |
| `wifi-densepose-geo` | room-frame anchor positions | `GeoScene` indoor extension | New indoor `Anchor` type added alongside `OsmFeature`; geo registration places the room frame in a global frame when an outdoor `GeoRegistration` exists. Optional — indoor-only deployments skip geo. |
| ADR-139 `WorldGraph` | `ObjectAnchorRecord` | `object_anchor` petgraph nodes (RVF) | RF SLAM owns reflector geometry; WorldGraph owns persistence and cross-domain links (anchor ↔ room ↔ person). |
| ADR-135 calibration | — | consumes `TopologyEvent` | `BaselineTopologyChange` is a stronger-typed cause feeding the existing `recalibrate_on_drift` path; no new recalibration mechanism. |

### 2.7 Validation Gate: 7-Day Dataset Before v2 Ships

v2 discovery may not be enabled in production until a **7-day paired validation dataset** demonstrates it is correct. The gate is enforced in code: `ReflectorTracker::observe()` returns `RfSlamError::ValidationGateClosed` if `RfSlamConfig::enabled` is set but the validation manifest is absent.

**Dataset contents (collected on the fleet from CLAUDE.local.md):**
- 7 consecutive days of unoccupied-window CSI from a ≥ 3-link room (e.g. `cognitum-v0` appliance room with `cognitum-seed-1` + 2 provisioned seeds).
- Ground-truth anchor positions: tape-measured wall and large-object positions in the room frame.
- ≥ 2 deliberate furniture-move events with logged before/after positions (for §2.4 and §2.3 validation).

**Pass criteria (all required to flip `enabled`):**
1. Discovered `StaticAnchor` positions within **0.5 m** of tape-measured ground truth for ≥ 80% of anchors.
2. Each logged furniture move detected by `TopologyMonitor` within 4 hours; displacement estimate within the 0.5 m band.
3. Zero false `BaselineTopologyChange` events across the 7 days of genuinely static periods.
4. No mobile reflector (the moved object) ever admitted to the `StaticAnchor` set.

Until then, the system ships v1: fixed APs + single static reflector. This mirrors ADR-135's principle that calibration must not silently degrade sensing.

---

## 3. Consequences

### 3.1 Positive

- **Anchors stop being hand-entered.** Today an operator must measure and type AP positions into `TriangulationConfig`. v2 discovers the static scene from the signal, so a moved AP or a newly characterised wall is picked up automatically — the long-standing manual-survey step disappears once v2 is validated.
- **Topology change becomes observable.** Reusing `variance_explained` and the Marcenko-Pastur rank gives a principled "the furniture moved" signal that feeds ADR-135 recalibration with a *specific cause*, replacing the amplitude-only drift heuristic.
- **Reflector geometry sharpens CIR and CHCI.** Once reflector positions are known, ADR-042 CHCI can use them as fixed scatterers in the coherent-imaging forward model, and ADR-134 CIR ghost-tap suppression knows which low-delay taps are structural (walls) vs body-perturbed.
- **One source of geometric truth.** Persisting to the ADR-139 WorldGraph means localisation (`mat/triangulation.rs`), the field model (ADR-030), and the temporal voxel map (ADR-142) all read the same `object_anchor` set instead of each carrying its own anchor assumptions.
- **Reuse over reinvention.** No new Kalman filter (reuses `kalman.rs`), no new solver (reuses `solve_triangulation`/`NeumannSolver`), no new occupancy detector (reuses `estimate_occupancy`), no new phase-coherence math (reuses ADR-135 von Mises projection).

### 3.2 Negative

- **v2 is dormant for an unknown lead time.** The 7-day dataset gates everything; until it is collected and passes, all of §2.1–§2.5 is dead code behind a feature flag. The value is realised only after a validation campaign on the fleet.
- **Bistatic multilateration needs ≥ 3 well-separated links.** A 1- or 2-link room can never resolve reflector positions (the spheroids do not intersect to a point). Such rooms are permanently v1-only. ADR-138 LinkGroups with poor geometric diversity yield high-covariance, low-value reflectors.
- **DBSCAN parameters (`ε=0.3 m`, `minPts=2`) are room-scale assumptions.** A very large or very cluttered space may need retuning; the defaults are validated only against the 7-day dataset room.
- **Furniture-movement inference is slow by design.** The tiny process-noise variance means a real move takes up to an hour to be confidently reported. This is intentional (it suppresses false moves) but means v2 is not a fast "object moved" alarm.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| v2 discovers a phantom reflector from correlated multipath self-interference and pollutes the anchor set | Medium | Localisation degrades against a wrong anchor | Coherence gate (von Mises variance < 0.15) + CHCI cross-link consistency + ≥3-link concurrence; phantom taps fail at least one. Validation criterion 4 explicitly tests this. |
| Reflector discovery runs during a period the occupancy detector wrongly calls "empty" (a still person) | Medium | A person-shaped scatterer learned as furniture | `persistence_cycles ≥ 24 h` requirement: a person does not sit perfectly still in one spot for a day; tap migration > 0.5 m/day eventually reclassifies them `MobileReflector` and excludes them from anchors. |
| `variance_explained` drops for a benign reason (temperature, humidity) and triggers false topology change | Low–Medium | Spurious recalibration request | Relative-drop + 4 h sustained window + unoccupied gate; ADR-030 already attributes slow thermal drift to the *retained* environmental modes, so it does not reduce `variance_explained`. Validation criterion 3 caps false events at zero. |
| Generalising `solve_triangulation()` to reflectors introduces a regression in person localisation | Low | Survivor localisation breaks | The reflector path is additive; the existing person-TDoA signature and tests are preserved unchanged. A regression test asserts byte-identical person-localisation output pre/post change. |
| Operator enables `rf-slam-v2` without the dataset | Low | — (fails safe) | `ValidationGateClosed` error blocks `observe()`; system stays on v1. |

---

## 4. Alternatives Considered

### 4.1 Visual / Camera SLAM for the Room Map

The fleet has cameras (`ruvultra`, `cognitum-v0`). Camera SLAM would map furniture far more accurately. Rejected as the *primary* mechanism because: (a) the entire product premise is privacy-preserving RF sensing — adding a camera to map the room contradicts the ADR-141 BFLD privacy modes; (b) cameras do not see through walls, so they cannot characterise reflectors behind furniture that nonetheless affect the RF channel. Camera ground truth is, however, exactly what the §2.7 validation dataset uses — as an *offline validation oracle*, not a runtime dependency.

### 4.2 Full Graph-SLAM / Factor-Graph Back-End (g2o / GTSAM style)

A factor-graph back-end jointly optimising all reflector positions, anchor poses, and person trajectories is the "textbook" SLAM formulation. Rejected for v2 scope: it is a large new dependency and solver, and the per-reflector Kalman + per-cycle least-squares multilateration already in the codebase (`kalman.rs` + `NeumannSolver`) is sufficient for a static-scene map that changes only on rare furniture moves. A factor-graph back-end is reasonable for a v3 once v2 proves the discovery front-end works.

### 4.3 Neural Reflector Inference

Train a network to regress reflector positions from CIR. Rejected for the same reason ADR-135 §4.3 rejects neural baselines: no paired CIR→geometry dataset exists, the mapping is room-specific, and a network gives no covariance or failure mode. Bistatic multilateration is a closed-form geometric estimator with an explicit covariance and a clear "insufficient links" failure.

### 4.4 Skip v1, Ship v2 Directly

Tempting — v2 is strictly more capable. Rejected because v2 is unvalidated and silently degrades on error (§1.2). Shipping the fixed-AP v1 gives a working, debuggable baseline that the v2 discovery can be measured *against*, and gives users a functioning system during the multi-day v2 validation campaign.

### 4.5 EMA-Adapted Anchor Positions Instead of Discrete Topology Events

Continuously sliding anchor positions with an exponential moving average avoids the topology-change ceremony. Rejected for the same reason ADR-135 §4.4 rejects EMA for baselines: a person standing near a wall would slowly drag the wall's "anchor" toward them. Anchors must be stable between explicit topology events, not continuously adapted.

---

## 5. Testing and Acceptance

### 5.1 Unit Tests (CI, synthetic — no hardware, no feature gate needed for the math)

- **T1 — bistatic geometry round-trip.** Place a synthetic reflector at a known `(x,y,z)`; compute the exact excess delay for 4 synthetic links; feed taps to `ReflectorTracker::observe()`; assert recovered `position_m` is within `0.05 m` (numerical, noise-free) and `position_cov` is small.
- **T2 — sub-3-link insufficiency.** Same reflector, only 2 links → `observe()` leaves `position_m == None`, no `StaticAnchor` emitted.
- **T3 — coherence gate.** A tap whose synthetic phase is randomised (circular variance ≈ 1.0) is never promoted to `StaticAnchor` regardless of link count.
- **T4 — mobile rejection.** A reflector whose synthetic position drifts 1.0 m/day is classified `MobileReflector`, never `StaticAnchor` (validates the 0.5 m/day threshold).
- **T5 — occupancy gate.** With `occupied = true`, `observe()` returns `RoomOccupied` and mutates no track.
- **T6 — topology variance collapse.** Feed `variance_explained` dropping from 0.80 → 0.66 (17.5% relative) sustained 4 h, unoccupied → exactly one `VarianceCollapse` event; a 10% drop produces none.
- **T7 — topology rank change.** MP significant count 5 → 6 sustained while unoccupied → one `RankChange` event.
- **T8 — furniture displacement + CHCI consistency.** A reflector moved 0.4 m consistently across ≥3 links → one `DisplacementEstimate` with `confidence_radius_m ≤ 0.5`; the same migration on 1 link only → suppressed (no estimate).
- **T9 — WorldGraph record provenance.** `ObjectAnchorRecord` always carries non-empty `discovery_model_version`, `calibration_version`, `privacy_decision`, and `evidence_refs` (enforces the four-part trace rule).
- **T10 — validation gate.** `enabled = true` without the validation manifest → `ValidationGateClosed`; `enabled = false` → `Disabled`. v1 path still populates the WorldGraph with immutable fixed-AP anchors in both cases.
- **T11 — person-localisation regression.** Generalised `solve_triangulation()` produces byte-identical output to the pre-change version for the existing person-TDoA test vectors.

### 5.2 Integration Test (gated `#[cfg(feature = "hardware-test")]`, not in CI)

- **T12 — 7-day fleet validation campaign.** On `cognitum-v0` room with ≥3 provisioned seeds: collect the §2.7 dataset, run discovery, and assert the four pass criteria. This test *is* the validation gate; passing it is the precondition for setting `RfSlamConfig::enabled` in production config.

### 5.3 Acceptance Criteria (mirror §2.7)

1. ≥ 80% of discovered `StaticAnchor`s within **0.5 m** of tape-measured ground truth.
2. Every logged furniture move flagged by `TopologyMonitor` within **4 h**; displacement within the **0.5 m** band.
3. **Zero** false `BaselineTopologyChange` events over 7 static days.
4. The moved object is **never** admitted to the `StaticAnchor` set.
5. With the feature off, the v1 fixed-AP + single-reflector map is present in the WorldGraph and person localisation is unchanged (T11 green).

### 5.4 Witness / Proof

Per ADR-028, add witness rows to `docs/WITNESS-LOG-028.md`:

| Row | Capability | Evidence |
|-----|-----------|----------|
| W-39 | Bistatic reflector multilateration round-trip (synthetic 4-link) | `cargo test rf_slam::tests::bistatic_round_trip` |
| W-40 | Topology-change detection (variance collapse + rank change) | `cargo test rf_slam::tests::topology_events` |
| W-41 | Validation gate blocks v2 without dataset; v1 map intact | `cargo test rf_slam::tests::validation_gate` |

`source-hashes.txt` gains `SHA-256(ruvsense/rf_slam.rs)`.

---

## 6. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-029 (RuvSense Multistatic) | **Consumes**: reflector geometry refines the multistatic attention-weighting prior. |
| ADR-030 (Persistent Field Model) | **Reuses**: `variance_explained`, Marcenko-Pastur `baseline_eigenvalue_count`, and `estimate_occupancy()` are the topology-change and occupancy-gate signals; RF SLAM is the geometric layer ADR-030 assumed existed. |
| ADR-042 (CHCI) | **Reuses + enables**: cross-link consistency gates furniture-movement; in return, discovered reflector positions become fixed scatterers in the CHCI forward model. |
| ADR-134 (CIR) | **Prerequisite**: `Cir.taps` excess-delay separation is the raw input to reflector discovery. |
| ADR-135 (Empty-Room Baseline) | **Reuses**: von Mises phase-concentration math for tap coherence; emits `BaselineTopologyChange` into ADR-135's existing recalibration trigger. |
| ADR-136 (Streaming Engine) | **Consumer**: reflector/anchor updates are a stage output frame. |
| ADR-138 (LinkGroup / ArrayCoordinator) | **Substrate**: discovery operates per LinkGroup so grouped links share a clock-quality tier and comparable delays. |
| ADR-139 (WorldGraph) | **Persistence**: `ObjectAnchorRecord` becomes `object_anchor` petgraph nodes via RVF — the single geometric source of truth. |
| ADR-142 (Evolution Tracker / Temporal VoxelMap) | **Downstream**: aggregates the `object_anchor` stream into the temporal room voxel map. |

---

## 7. References

### Production Code

- `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs` — `Cir` struct (taps, `tap_spacing_sec`, `dominant_tap_idx`, `dominant_tap_ratio`, `active_tap_count`, `rms_delay_spread_s`); `Cir::dominant_distance_m()`. Excess-delay input to discovery.
- `v2/crates/wifi-densepose-signal/src/ruvsense/field_model.rs` — `FieldModel` (`variance_explained`, `baseline_eigenvalue_count`, `estimate_occupancy()`); `WelfordStats` reused for tap statistics.
- `v2/crates/wifi-densepose-mat/src/tracking/kalman.rs` — `KalmanState` 6-state constant-velocity filter, reused (retuned process noise) for per-reflector tracking.
- `v2/crates/wifi-densepose-mat/src/localization/triangulation.rs` — `Triangulator` / `TriangulationConfig` (person localisation against fixed anchors; v1 path).
- `v2/crates/wifi-densepose-ruvector/src/mat/triangulation.rs` — `solve_triangulation()` (Neumann-series TDoA least squares); generalised to accept reflector bistatic ranges.
- `v2/crates/wifi-densepose-geo/src/types.rs` — `GeoScene` / `GeoRegistration`; indoor `Anchor` extension point.
- `v2/crates/wifi-densepose-signal/src/ruvsense/rf_slam.rs` — **NEW** module: `Reflector`, `ReflectorTracker`, `AnchorLearner`, `TopologyMonitor`, `FurnitureMovementEstimator`, `ObjectAnchorRecord`.

### External

- Welford, B.P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products." *Technometrics*, 4(3). — Online statistics for per-reflector tap amplitude.
- Mardia, K.V. & Jupp, P.E. (2000). *Directional Statistics*. Wiley. — Circular variance `1 − R̄` used for tap coherence gating.
- Foy, W.H. (1976). "Position-Location Solutions by Taylor-Series Estimation." *IEEE Trans. AES*. — Linearised range/TDoA least-squares solved here via the Neumann series.
- Marčenko, V.A. & Pastur, L.A. (1967). "Distribution of eigenvalues for some sets of random matrices." *Math. USSR-Sbornik*. — Significant-eigenvalue threshold used for the occupancy and covariance-rank gates (already in `field_model.rs`).


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `2d4f3dea5`, issue #847): `RfSlam` reflector discovery with Welford position stability and Wall/Furniture/Mobile classification; ships v1 fixed-map mode by default. 6 tests.

**Integration glue -- not yet on the live path:** live CIR-tap -> reflector-position inference behind the ADR-030 Marcenko-Pastur eigenvalue gate; writing discovered anchors into the WorldGraph as `ObjectAnchor` nodes; the multi-day validation dataset before v2 discovery is enabled.

**Trust contribution:** landmarks are *learned and verified stable* (walls/furniture) while transient reflectors are rejected, so localization rests on trustworthy anchors.
