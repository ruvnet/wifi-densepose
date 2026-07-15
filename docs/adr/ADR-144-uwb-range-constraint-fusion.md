# ADR-144: UWB Range-Constraint Fusion with World-Graph Anchors

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; no UWB radio in fleet — see Implementation Status, commit `b10bc2e9a`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-hardware` (new UWB driver/parser/auto-detect in `src/`); `wifi-densepose-signal` (`ruvsense/pose_tracker.rs` constraint-aware Kalman update); `wifi-densepose-mat` (`localization/fusion.rs` constraint integration) |
| **Relates to** | ADR-016 (RuVector Integration), ADR-018 (ESP32 Dev Implementation / binary wire format), ADR-024 (Contrastive CSI Embedding / AETHER), ADR-029 (RuvSense Multistatic), ADR-031 (RuView Sensing-First RF Mode), ADR-063 (mmWave Sensor Fusion), ADR-136 (RuView Rust Streaming Engine), ADR-138 (WiFi-7 MLO LinkGroup / ArrayCoordinator), ADR-139 (WorldGraph Environmental Digital Twin), ADR-141 (BFLD Privacy Control Plane), ADR-145 (Ablation Evaluation Harness) |

---

## 1. Context

### 1.1 The Gap

WiFi CSI sensing in this codebase produces *relative* perturbation fields, not *metric* position. The pose tracker estimates 3D keypoint coordinates from those fields, but the only thing anchoring those coordinates to real-world metres is the geometry assumed at calibration time. There is no independent metric ranging source to correct scale drift, resolve the front/back ambiguity inherent in a single multistatic array, or disambiguate two tracks that cross. UWB (ultra-wideband, IEEE 802.15.4z) two-way ranging gives exactly that: a direct, hardware-grounded distance measurement with ±10 cm accuracy that is *orthogonal* to the CSI evidence.

Searching the workspace confirms there is no UWB support anywhere:

- `grep -ri "uwb\|802.15.4z\|two_way_ranging\|RangeConstraint" v2/crates/` returns nothing in production code. The only `802.15.4` reference is the *timesync* epoch on the ESP32-C6 (`c6_timesync_get_epoch_us()`, ADR-110), which is a clock primitive, not a ranging primitive.
- `v2/crates/wifi-densepose-hardware/src/` contains parsers for ESP32 CSI (`esp32_parser.rs`, ADR-018 magic `0xC5110001`), sibling RuView packets (`RUVIEW_VITALS_MAGIC` … `RUVIEW_TEMPORAL_MAGIC`), a UDP aggregator (`aggregator/`), a `bridge.rs` (`CsiFrame → CsiData`), and the radio-ops mirror (`radio_ops.rs`). Every magic constant in `esp32_parser.rs` is a *CSI-family* packet. There is no range/anchor frame type and no anchor-bearing device abstraction.
- `v2/crates/wifi-densepose-signal/src/ruvsense/pose_tracker.rs` (the 17-keypoint Kalman tracker, ADR-029 §2.7) has a position-only measurement model: `KeypointState::update()` takes `&[f32; 3]` and `KeypointState::mahalanobis_distance()` gates a *Cartesian* measurement. There is **no mechanism to apply a range constraint** — a measurement of the form "the centroid is `r ± σ` metres from a fixed anchor" — which is a nonlinear (spherical) observation, not a Cartesian one. `PoseTrack` has no field for accumulated range residuals.
- `v2/crates/wifi-densepose-mat/src/localization/fusion.rs` has a `PositionFuser` with an `EstimateSource` enum (`RssiTriangulation`, `TimeOfArrival`, `AngleOfArrival`, `CsiFingerprint`, `DepthEstimation`, `Fused`) and `Triangulator` that consumes RSSI. There is **no `TimeOfArrival` producer** — `EstimateSource::TimeOfArrival` is defined but nothing emits it, and `LocalizationService::simulate_rssi_measurements()` explicitly returns `vec![]` with a warning "No sensor hardware connected." The fusion machinery exists; the metric-ranging input does not.

The consequence is concrete. Three failure modes trace directly to the missing metric anchor:

- **Scale and front/back ambiguity in single-array sensing.** A monostatic or near-colinear multistatic CSI array cannot distinguish a person 2 m in front from a (geometrically mirrored) reflection 2 m behind without strong geometric diversity (ADR-029's `geometry.rs` Fisher-information bounds quantify exactly when this fails). A single UWB range to a known anchor collapses that ambiguity for the constrained dimension.
- **Track-crossing identity swaps.** When two `PoseTrack`s pass within the Mahalanobis gate of each other, assignment falls back to AETHER re-ID cosine similarity (`pose_tracker.rs` `embedding_weight = 0.4`). Re-ID alone is unreliable for similar body shapes. A UWB tag worn by one person (or a range that is consistent with only one of the two crossing tracks) breaks the tie deterministically.
- **No metric ground truth for the WorldGraph.** ADR-139's WorldGraph stores object anchors and person tracks as typed nodes; without a metric edge between them, anchor positions are never corrected and the digital twin slowly drifts from physical reality.

ADR-063 (mmWave Sensor Fusion, Accepted) already establishes the *pattern* for fusing an orthogonal ranging modality (60 GHz FMCW range/Doppler) with CSI, and `RUVIEW_FUSED_VITALS_MAGIC` (`0xC5110004`) is the on-wire fused packet. ADR-144 follows that established fusion pattern but for UWB metric range rather than mmWave radial velocity, and it routes the result through the WorldGraph (ADR-139) as a first-class graph edge rather than a flat fused packet.

### 1.2 What a "Range Constraint" Is Here

A UWB range constraint is a single scalar metric measurement plus its provenance:

- A measured line-of-sight distance `r` in metres between a fixed **anchor** of known position and a moving **tag/responder**, obtained by 802.15.4z single- or double-sided two-way ranging (SS/DS-TWR) or, where a synchronized anchor mesh exists, time-difference-of-arrival (TDoA).
- An uncertainty `σ_r` derived from the UWB module's reported first-path SNR / link quality. Clean LOS yields ~±10 cm; NLOS (through a wall) biases the range *long* and inflates `σ_r`.
- A timestamp in the same 802.15.4 epoch domain already used for multi-node CSI sync (ADR-110), so a range can be associated with the CSI frame closest in time.

What a range constraint is **not**: it is not a position. One range defines a *sphere* of possible tag positions centred on the anchor. Position emerges only when a range is *fused* with the CSI-derived track state (which already carries a 3D estimate and covariance). This is the core reason the fusion lives in `pose_tracker.rs`'s Kalman update rather than as a standalone trilateration solver: the CSI track *is* the prior, and the range *tightens* it.

### 1.3 Hardware Context

UWB is a separate radio from WiFi. Three deployment forms are evaluated (Decision §2.3); the working assumption is a **standalone ESP32-C6 + DW3000-class UWB transceiver bridge node** that speaks the existing ADR-018 UDP transport:

| Form factor | Radio | Role | Wire path | Cost |
|-------------|-------|------|-----------|------|
| Standalone UWB anchor (ESP32-C6 + Qorvo DW3000) | 802.15.4z UWB + 802.15.4 timesync | Fixed anchor, ranges to tags | New UDP magic frame over existing aggregator | ~$18 |
| Integrated radio (ESP32-C6 doing CSI *and* UWB on one node) | shared MCU | CSI sensing node that also ranges | Same node, interleaved magic | ~$15 (no extra node) |
| Bridge node (UWB-only MCU → serial → Pi 5) | DW3000 dev board | Anchor mesh, host does ranging math | `aggregator/` ingest | ~$25 |

All three converge on the **same host-side abstraction**: a stream of `UwbRangeFrame`s with `(anchor_id, tag_id, range_m, quality, epoch_us)`. The hardware abstraction layer (HAL) hides which form factor produced the frame, exactly as `esp32_parser.rs` hides whether CSI came from an S3 or a C6. The C6's existing `c6_timesync_get_epoch_us()` (±100 µs) is reused so UWB ranges and CSI frames share one clock.

### 1.4 Pipeline Position

```
UWB anchor/tag (802.15.4z TWR)
  → UwbFrameParser::parse()         ← NEW (wifi-densepose-hardware, ADR-018-style magic)
  → RangeConstraint { anchor_id, range_m, σ, epoch_us, quality }  ← NEW domain model
  → WorldGraph::upsert_range_edge() ← NEW edge (ADR-139), object_anchor → person_track
        │
        │  (association: which track does this range belong to?)
        │   Mahalanobis-to-sphere gate + AETHER re-ID disambiguation
        ▼
  → PoseTracker::apply_range_constraint()  ← NEW (constraint-aware Kalman update)
        │
        ▼
CSI-only track state ─────────┐
                              ├──→ LocalizationService (mat/fusion.rs)
                              │     EstimateSource::TimeOfArrival now PRODUCED
                              ▼
                        fused metric track (with constraint residual + confidence)
```

CSI flows down the existing pipeline unchanged. The UWB range enters as a *parallel* evidence stream, is associated to a track, and is applied as an extra Kalman update step *after* the normal CSI measurement update. If no range arrives in a given cycle, the tracker behaves exactly as today — UWB is strictly additive.

---

## 2. Decision

### 2.1 The `RangeConstraint` Domain Model

A `RangeConstraint` is the canonical, hardware-agnostic representation of one UWB range, defined in `wifi-densepose-hardware` (alongside `CsiFrame`) and re-exported for `signal` and `mat`. It carries enough provenance to satisfy the project rule that every semantic state traces to signal evidence + model version + calibration version + privacy decision.

```rust
use std::time::Duration;

/// Stable identifier for a fixed UWB anchor of known position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnchorId(pub u32);

/// Stable identifier for a mobile UWB tag / responder (may be a worn tag
/// or an unlabelled responder discovered during ranging).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TagId(pub u32);

/// Source of the metric range measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeMethod {
    /// Single-sided two-way ranging (one round trip; clock-offset sensitive).
    SsTwr,
    /// Double-sided two-way ranging (cancels clock offset; preferred).
    DsTwr,
    /// Time-difference-of-arrival against a synchronized anchor mesh.
    Tdoa,
}

/// One UWB metric range measurement with full provenance.
///
/// Defines a *sphere* of possible tag positions of radius `measured_range_m`
/// centred on the anchor at `AnchorId`. Fused with a CSI track to produce a
/// metric position (see §2.5).
#[derive(Debug, Clone)]
pub struct RangeConstraint {
    /// Fixed anchor this range was measured against.
    pub anchor_id: AnchorId,
    /// Tag/responder the range was measured to (if labelled).
    pub tag_id: Option<TagId>,
    /// Measured line-of-sight distance in metres.
    pub measured_range_m: f32,
    /// 1-sigma uncertainty in metres, derived from `signal_quality`.
    pub uncertainty_m: f32,
    /// 802.15.4 epoch microseconds (same domain as CSI timesync, ADR-110).
    pub timestamp_us: u64,
    /// First-path SNR / link-quality score in [0, 1]; 1 = clean LOS.
    pub signal_quality: f32,
    /// Ranging method used.
    pub method: RangeMethod,
}

impl RangeConstraint {
    /// True if quality is high enough to apply as a hard(er) constraint.
    /// NLOS ranges (low quality) are applied with inflated `uncertainty_m`
    /// rather than rejected outright.
    pub fn is_los(&self, los_threshold: f32) -> bool {
        self.signal_quality >= los_threshold
    }

    /// Effective measurement variance, NLOS-inflated.
    pub fn variance(&self) -> f32 {
        self.uncertainty_m * self.uncertainty_m
    }
}
```

**Why `uncertainty_m` derives from `signal_quality` rather than being fixed:** UWB NLOS does not fail loudly — it biases the range *long* (the first detectable path went around an obstacle). Rejecting low-quality ranges discards information; inflating their variance lets the Kalman filter down-weight them gracefully, which is the same philosophy ADR-135 used for multimodal-phase subcarriers (down-weight, do not drop).

### 2.2 WorldGraph Anchor Construction (ADR-139 Integration)

ADR-139's WorldGraph is a typed petgraph whose nodes include `object_anchor` and `person_track`. A `RangeConstraint` becomes a **typed, weighted, timestamped edge** between an `object_anchor` node (the UWB anchor's fixed position) and a `person_track` node (a `PoseTrack`).

```rust
/// Edge payload stored on a WorldGraph object_anchor → person_track edge.
#[derive(Debug, Clone)]
pub struct RangeEdge {
    pub anchor_id: AnchorId,
    pub track_id: TrackId,
    pub constraint: RangeConstraint,
    /// Mahalanobis distance of this range to the track's predicted sphere
    /// at association time (the association cost, see §2.4).
    pub assoc_cost: f32,
    /// Provenance triple required by the SSR rule (ADR-140):
    pub signal_evidence_id: u64,   // CSI frame seq that the track state came from
    pub model_version: u32,        // pose/embedding model version
    pub anchor_survey_version: u32, // anchor-registration ("calibration") version
}
```

The anchor node carries its surveyed 3D position and an `anchor_survey_version` that plays the same role for UWB that `schema_version`/`captured_at` plays for the ADR-135 baseline: a change to anchor geometry invalidates downstream range fusions tagged with the old survey version. The WorldGraph gains:

```rust
impl WorldGraph {
    /// Register or update a fixed anchor with a surveyed position.
    /// Bumps `anchor_survey_version` and marks all RangeEdges from this
    /// anchor stale.
    pub fn register_anchor(&mut self, id: AnchorId, pos: [f32; 3]) -> u32;

    /// Insert a range constraint as an object_anchor → person_track edge.
    /// Returns Err if `anchor_id` is not registered.
    pub fn upsert_range_edge(&mut self, edge: RangeEdge) -> Result<(), WorldGraphError>;

    /// All current range edges incident to a track (for the Kalman update).
    pub fn range_edges_for(&self, track: TrackId) -> Vec<&RangeEdge>;
}
```

Anchor positions are surveyed once and stored on the graph; this is the *anchor-registration policy* decision (§2.7). The WorldGraph is the single source of truth for anchor geometry so that `pose_tracker.rs` and `mat/fusion.rs` never disagree about where an anchor is.

### 2.3 UWB Hardware Abstraction Layer (ADR-018 Wire-Format Pattern)

A new module set in `wifi-densepose-hardware/src/` mirrors the `esp32_parser.rs` design: a magic-tagged binary frame over the existing UDP aggregator, a pure-bytes parser that never fabricates data, and an auto-detect that demultiplexes by magic.

```rust
/// UWB range frame magic (ADR-144), next in the 0xC511xxxx family after
/// RUVIEW_TEMPORAL_MAGIC (0xC5110007). Demultiplexed alongside CSI frames.
pub const UWB_RANGE_MAGIC: u32 = 0xC5110008;

/// ADR-018-style binary layout (little-endian):
///   0   4   Magic 0xC5110008
///   4   4   anchor_id (u32)
///   8   4   tag_id (u32; 0 = unlabelled responder → None)
///   12  4   range_mm (u32; millimetres, converted to f32 metres)
///   14  ...                              (see exact offsets in parser doc)
///   ..  2   uncertainty_mm (u16)
///   ..  1   method (0=SS-TWR,1=DS-TWR,2=TDoA)
///   ..  1   signal_quality (u8, 0..=255 → [0,1])
///   ..  8   epoch_us (u64, 802.15.4 timesync domain)
pub struct UwbFrameParser;

impl UwbFrameParser {
    /// Parse one UWB range frame from raw UDP bytes.
    /// Either parses real bytes or returns a specific `ParseError`
    /// (NEVER fabricates a range — matches the no-mock guarantee).
    pub fn parse(buf: &[u8]) -> Result<(RangeConstraint, usize), ParseError>;

    /// Returns true if `buf` begins with `UWB_RANGE_MAGIC`.
    pub fn is_uwb_frame(buf: &[u8]) -> bool;
}
```

**Form-factor decision (§1.3 candidates):** adopt the **standalone ESP32-C6 + DW3000 anchor** as the reference build, but the HAL admits all three because the parser only sees bytes. Rationale: (a) it reuses the C6's `c6_timesync_get_epoch_us()` so UWB ranges land in the *same clock* as CSI frames with no new timesync work; (b) it reuses the ADR-018 UDP aggregator, so no new transport, no new firmware OTA channel, no new port; (c) integrating UWB onto an existing CSI node (form 2) is a strict superset — the same parser handles its frames. The aggregator's existing demultiplex loop gains one arm: `if UwbFrameParser::is_uwb_frame(buf) { … } else if Esp32CsiParser` (the same `else if` ladder already used for the seven `RUVIEW_*_MAGIC` sibling packets).

**Interface boundary:** `wifi-densepose-hardware` owns parsing and the `RangeConstraint`/`AnchorId`/`TagId` types. It has **no dependency** on `signal` or `mat` — the dependency arrows point the other way, consistent with the crate publishing order (`hardware` has no internal deps; `signal` depends on `core`; `mat` depends on `signal`).

### 2.4 Constraint-to-Track Association (AETHER Re-ID Disambiguation)

A range from an unlabelled responder (`tag_id = None`) must be assigned to one of the live `PoseTrack`s before it can be applied. Labelled tags (`tag_id = Some(_)`) that have been bound to a track skip association. For unlabelled ranges, association uses a gated cost that mirrors the existing `pose_tracker.rs` assignment cost (`position_weight * maha + embedding_weight * embed_cost`) but with the *spherical* residual:

For each candidate track `T` with predicted centroid `c_T` and anchor at `a`:

```
sphere_residual(T) = | ‖c_T − a‖ − measured_range_m |          (metres off the sphere)
maha_sphere(T)     = sphere_residual(T) / sqrt(var_radial(T) + constraint.variance())
assoc_cost(T)      = range_pos_weight * maha_sphere(T)
                   + range_reid_weight * reid_ambiguity(T)
```

where `var_radial(T)` is the track's positional variance projected onto the anchor→centroid line (computed from the existing `KeypointState::covariance` diagonal), and `reid_ambiguity(T)` is invoked **only when two or more tracks are within the spherical Mahalanobis gate** — i.e. equidistant-from-anchor crossing tracks. In that case the range is associated to the track whose AETHER embedding best matches the tag's last-known embedding (for labelled tags) or whose recent CSI-only association confidence is highest (for unlabelled). This reuses `cosine_similarity()` and the 128-dim embedding already on `PoseTrack`.

```rust
/// Result of associating one RangeConstraint to the live track set.
pub enum RangeAssociation {
    /// Uniquely associated (single track inside the gate).
    Assigned { track: TrackId, cost: f32 },
    /// Multiple tracks inside the gate; resolved by AETHER re-ID.
    AmbiguousResolved { track: TrackId, runner_up: TrackId, margin: f32 },
    /// No track inside the spherical Mahalanobis gate — range buffered,
    /// not applied (may seed a new track if persistent).
    Unassigned,
}
```

`Unassigned` ranges are not discarded — a persistent unassigned range that is geometrically consistent over several cycles is evidence of a person the CSI array has not yet detected (e.g. behind a piece of furniture), and is surfaced to the WorldGraph as a low-confidence latent track candidate. This is the UWB analogue of ADR-135 logging drift rather than silently dropping it.

### 2.5 Constraint-Aware Kalman Update (`pose_tracker.rs`)

The current `KeypointState::update()` is a *linear* Cartesian update (`H = [I3 | 0]`). A range is a *nonlinear* spherical observation `h(x) = ‖x − a‖`. We apply it as an **Extended Kalman (EKF) measurement update on the track centroid**, then distribute the centroid correction back to the keypoints proportionally — rather than rebuilding the whole tracker as a factor graph.

**Algorithm decision: EKF spherical update with Mahalanobis gating and quality-weighted noise** (chosen over factor-graph batch optimization and over pure Mahalanobis gate-and-penalty; see §3 Alternatives). The centroid `c` already exists (`PoseTrack::centroid()`). For an anchor at `a`:

```
h(c)      = ‖c − a‖                                   (predicted range)
H         = (c − a)ᵀ / ‖c − a‖                        (1×3 Jacobian, unit LOS vector)
y         = measured_range_m − h(c)                   (scalar innovation)
S         = H P_c Hᵀ + R   where R = constraint.variance()   (NLOS-inflated)
K         = P_c Hᵀ S⁻¹                                 (3×1 gain)
c'        = c + K y
P_c'      = (I − K H) P_c
```

`P_c` is the 3×3 centroid covariance assembled from the per-keypoint covariance diagonals. After the centroid is corrected by `K y`, the same translational delta `(c' − c)` is added to every keypoint position and the radial variance reduction is applied to each keypoint's covariance, so the skeleton moves rigidly toward the constraint sphere without distorting its shape. This composes cleanly with the existing CSI update: CSI runs first (full skeleton update), then the range update nudges the whole skeleton onto the sphere.

The constraint update is **gated**: if `|y| / sqrt(S) > range_gate` (default 3.0, matching the existing chi-squared 3-sigma philosophy of `mahalanobis_gate = 9.0`), the range is rejected for this cycle and recorded as a residual outlier rather than applied — preventing a wild NLOS range from teleporting a track.

New state on `PoseTrack` (extending the struct, never replacing existing fields):

```rust
/// Range-constraint history appended to PoseTrack (bounded ring buffer).
#[derive(Debug, Clone, Default)]
pub struct ConstraintTrackState {
    /// Recent constraints applied to this track (bounded; e.g. last 32).
    pub buffer: VecDeque<RangeConstraint>,
    /// Last applied scalar range residual (metres, signed).
    pub last_constraint_residual: f32,
    /// Gate status of the most recent constraint.
    pub constraint_gate_status: ConstraintGateStatus,
    /// Distinct anchors that have contributed a range to this track.
    pub fused_range_sources: Vec<AnchorId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConstraintGateStatus {
    #[default]
    /// No range applied this cycle.
    None,
    /// Range passed the gate and was fused.
    Accepted,
    /// Range exceeded the gate; recorded as outlier, not applied.
    RejectedOutlier,
    /// Range applied with NLOS-inflated variance (low quality).
    AcceptedNlos,
}
```

```rust
impl PoseTracker {
    /// Apply one associated range constraint to a track via the spherical
    /// EKF update above. Updates ConstraintTrackState. No-op (returns
    /// RejectedOutlier) if the gate is exceeded.
    pub fn apply_range_constraint(
        &mut self,
        track: TrackId,
        anchor_pos: [f32; 3],
        constraint: &RangeConstraint,
        range_gate: f32,
    ) -> Result<ConstraintGateStatus, PoseTrackerError>;
}
```

`TrackerConfig` gains `range_gate: f32` (default 3.0), `range_pos_weight: f32` (default 0.7), `range_reid_weight: f32` (default 0.3), and `los_threshold: f32` (default 0.6). Defaults are off-path-safe: with no range frames, none of this code executes and the tracker is byte-for-byte its current behaviour.

### 2.6 `mat/fusion.rs` Integration

`EstimateSource::TimeOfArrival` (already defined, currently unproduced) becomes the producer slot for UWB-derived metric position. `LocalizationService` gains an optional UWB path so the MAT survivor-localization use case (rubble ranging) and the ambient-sensing use case share one fusion implementation:

```rust
impl LocalizationService {
    /// Produce a TimeOfArrival PositionEstimate from a set of range
    /// constraints to surveyed anchors (≥3 for full 3D, fewer constrains a
    /// subspace). Replaces the empty simulate_rssi_measurements() path when
    /// real UWB anchors are present.
    pub fn estimate_from_ranges(
        &self,
        ranges: &[(Coordinates3D /*anchor*/, RangeConstraint)],
    ) -> Option<PositionEstimate>;
}
```

The resulting `PositionEstimate { source: EstimateSource::TimeOfArrival, weight: f(signal_quality), .. }` flows into the existing `PositionFuser::fuse()`, whose `calculate_weight()` already ranks `TimeOfArrival` highest (`1.0`). UWB thus slots into a fusion ranking the codebase already encodes — no new fuser, only a new producer. This keeps the MAT crate's domain model intact.

### 2.7 Anchor-Registration Policy

**Decision: manual survey as the authoritative source, with optional auto-learn from track geometry as a *proposal* the operator confirms.**

- **Manual survey (default, authoritative).** The operator measures each anchor's 3D position once and calls `WorldGraph::register_anchor()`. This sets `anchor_survey_version`. This is the UWB analogue of ADR-135's operator-initiated calibration: there is no way to know an anchor's true position from the data alone with the accuracy fusion needs, so the system does not guess by default.
- **Auto-learn (opt-in, proposal only).** When ≥3 anchors range the *same* moving tag over a trajectory with sufficient geometric diversity (the Fisher-information criterion from ADR-029 `geometry.rs`), the anchor positions become observable up to a rigid transform. An offline solver can *propose* refined anchor positions, but they are applied only after the operator accepts — never silently — for the same reason ADR-135 refuses automatic recalibration: a self-modified anchor that is wrong corrupts every downstream fusion invisibly.

Either path bumps `anchor_survey_version`, which invalidates `RangeEdge`s tagged with the old version, mirroring ADR-135's stale-baseline invalidation.

### 2.8 Provenance and the SSR Rule

Every fused metric position is a semantic state and therefore carries the full provenance triple (ADR-140 SSR / ADR-141 privacy):

- **Signal evidence** — `RangeEdge.signal_evidence_id` (CSI frame sequence the track prior came from) + the `RangeConstraint.timestamp_us` of the UWB range.
- **Model version** — `RangeEdge.model_version` (pose + AETHER embedding model).
- **Calibration version** — `RangeEdge.anchor_survey_version` (anchor geometry survey).
- **Privacy decision** — UWB ranging reveals the *presence and distance of a tag-bearing person*, which is identity-adjacent. A range fusion is gated by the active BFLD privacy mode (ADR-141): in privacy modes that forbid identity binding, labelled `tag_id` association is suppressed and ranges are applied only as anonymous spherical constraints (no re-ID disambiguation, no tag→track binding stored).

### 2.9 Test Plan and Acceptance Criteria

**Tier 1 — Parser round-trip (unit test).** Encode a `RangeConstraint` to the §2.3 binary layout, parse with `UwbFrameParser::parse()`, assert field equality. Assert `is_uwb_frame()` returns `true` for `UWB_RANGE_MAGIC` and `false` for `ESP32_CSI_MAGIC` and all seven `RUVIEW_*_MAGIC`. Assert a truncated buffer yields `ParseError::InsufficientData` (no fabricated range).

**Tier 2 — Spherical EKF correctness (unit test).** Place a track centroid at `(2,0,0)` with a known `P_c`; supply a range of `1.8 m` to an anchor at the origin (true distance 2.0). Assert the corrected centroid moves *along the LOS toward the sphere* by approximately `K·y`, that `P_c` shrinks in the radial direction, and that the skeleton shape (inter-keypoint distances) is unchanged to f32 precision (rigid translation).

**Tier 3 — Gate rejection (unit test).** Same track; supply a range of `8.0 m` (4 m innovation, far beyond gate). Assert `apply_range_constraint()` returns `ConstraintGateStatus::RejectedOutlier`, the centroid is **unchanged**, and `last_constraint_residual` records the outlier.

**Tier 4 — Crossing disambiguation (unit test).** Two tracks at `(2,0,0)` and `(0,2,0)`, both ~2 m from an anchor at the origin (equidistant → both inside the spherical gate). Track A's embedding matches the tag's last embedding (cosine ≈ 0.95), Track B's does not (≈ 0.1). Assert association returns `AmbiguousResolved { track: A, .. }` with positive `margin`.

**Tier 5 — NLOS inflation (unit test).** A range with `signal_quality = 0.2` (NLOS). Assert `RangeConstraint::is_los(0.6) == false`, that `variance()` is inflated, and that the EKF gain `K` is correspondingly smaller than for a clean LOS range of the same innovation → status `AcceptedNlos`.

**Tier 6 — WorldGraph edge lifecycle (unit test).** Register an anchor → `upsert_range_edge()` → `range_edges_for(track)` returns it. Call `register_anchor()` again (re-survey) → assert `anchor_survey_version` bumps and stale edges are flagged.

**Tier 7 — `mat/fusion.rs` producer (unit test).** Feed three anchor+range pairs to `estimate_from_ranges()`; assert it yields a `PositionEstimate` with `source == EstimateSource::TimeOfArrival` and that `PositionFuser::fuse()` weights it at least as high as a co-located `RssiTriangulation` estimate.

**Tier 8 — Off-path no-op (regression test).** Run the existing `pose_tracker` test suite with `range_*` config at defaults and **zero** range frames; assert every existing assertion passes unchanged (UWB is strictly additive).

**Tier 9 — Determinism proof (CI-compatible, extends ADR-028).** A fixed synthetic trajectory + fixed range sequence (seeded) is fused; the SHA-256 of the resulting fused track positions is recorded in `archive/v1/data/proof/expected_features.sha256` under `uwb_range_fusion_v1`, and `verify.py` regenerates and asserts it. Adds witness rows to `docs/WITNESS-LOG-028.md` for parser round-trip, spherical EKF, and crossing disambiguation; `source-hashes.txt` gains the new parser and the `pose_tracker.rs` constraint additions.

**Tier 10 — Real hardware (integration, gated `#[cfg(feature = "hardware-test")]`).** With one DW3000 anchor and a tag walked along a measured 4 m path, assert fused track range tracks the tape-measured ground truth to < 15 cm RMS in LOS and that NLOS segments (tag behind a wall) inflate uncertainty rather than producing > 30 cm errors. Not run in CI.

---

## 3. Consequences

### 3.1 Positive

- **Metric grounding.** CSI tracks gain absolute scale and front/back disambiguation from an orthogonal modality. A single range collapses the ambiguity that ADR-029's geometry bounds show a near-colinear array cannot resolve from CSI alone.
- **Deterministic crossing resolution.** Track-swap identity errors at crossings are broken by range + AETHER re-ID, where re-ID alone was unreliable for similar body shapes.
- **Reuses, does not rebuild.** The HAL reuses the ADR-018 UDP transport, the `0xC511xxxx` magic family, the C6 802.15.4 timesync clock, the `pose_tracker.rs` cost-blend pattern, and the `mat/fusion.rs` `PositionFuser` ranking. The only genuinely new math is one EKF measurement update.
- **Activates dead code.** `EstimateSource::TimeOfArrival` finally has a producer; `simulate_rssi_measurements()`'s empty-handed path gains a real metric alternative for the MAT use case.
- **WorldGraph becomes metric.** ADR-139's anchor and track nodes get a real, version-tracked metric edge, so the digital twin can be corrected against physical ground truth rather than drifting.

### 3.2 Negative

- **New radio and hardware cost.** UWB is a second radio; even the cheapest form factor adds ~$15–25 per anchor and an anchor survey step. Sensing works without it (UWB is additive), but the metric benefit requires the hardware.
- **Anchor survey ceremony.** Like the ADR-135 baseline, anchors must be measured and registered before fusion is meaningful; a mis-surveyed anchor biases every range fused against it.
- **EKF linearization error.** The spherical update linearizes `h(x) = ‖x−a‖`; for a track very close to an anchor (small `‖c−a‖`), the Jacobian is ill-conditioned. Mitigated by a minimum-range guard and gating, but it is a real limit not present in the linear CSI update.
- **New struct surface.** `PoseTrack` grows a `ConstraintTrackState`, `TrackerConfig` grows four fields, and `WorldGraph` grows anchor/edge methods. All are additive and default-inert, but they widen the public API.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NLOS range biases a track long without being flagged | Medium (through-wall ranging) | Track pulled away from truth | Quality-derived `uncertainty_m` inflation + 3-sigma gate; persistent outliers logged, not applied |
| Wrong-track association at a crossing | Low–Medium | Identity swap with high confidence | Spherical Mahalanobis gate + AETHER re-ID; `AmbiguousResolved.margin` surfaced; privacy modes that forbid identity binding fall back to anonymous spherical constraint only |
| Mis-surveyed anchor | Medium (manual measurement) | Systematic bias on every fused range from that anchor | `anchor_survey_version` invalidation; optional auto-learn *proposal* for operator confirmation; never silent self-update |
| EKF divergence for a track adjacent to an anchor | Low | Gain blow-up, track teleport | Minimum-range guard on `‖c−a‖`; gate rejects the resulting large innovation |
| UWB frames starve the aggregator demux of CSI | Low | Dropped CSI frames | UWB ranges are ~10 Hz per tag vs CSI 20 Hz; demux is a cheap magic-match `else if` arm, same as the seven existing `RUVIEW_*` arms |

---

## 4. Alternatives Considered

### 4.1 Why Not a Full Factor Graph (GTSAM-style)

A factor graph would jointly optimize all keypoints, all ranges, and all anchors in one nonlinear least-squares batch — theoretically optimal. Rejected for this codebase because: (a) it would *replace* the existing real-time `pose_tracker.rs` EKF rather than extend it, discarding a tested, shipping tracker; (b) batch optimization is not naturally online and would complicate the 20 Hz real-time loop; (c) it pulls in a heavy nonlinear-solver dependency where the existing tracker uses only hand-rolled diagonal Kalman math. The incremental EKF range update captures ~all the benefit (range tightens the prior) at a fraction of the integration cost, and the *auto-learn anchor* path in §2.7 can use an offline batch solver where the batch formulation genuinely helps.

### 4.2 Why Not Pure Mahalanobis Gate-and-Penalty (No State Update)

The simplest option: use the range only to *score* association (penalize tracks inconsistent with the range) but never let it move the state. Rejected because it throws away the metric correction — the whole point. A range that says "this person is 1.8 m from the anchor" should *move* a CSI estimate that says 2.3 m, not merely down-rank an assignment. We keep the gating (it is good for outlier rejection) but pair it with the EKF state update.

### 4.3 Why Not Treat UWB as Just Another `PositionEstimate` Source in `mat/fusion.rs`

We could skip `pose_tracker.rs` entirely and only fuse UWB at the MAT `PositionFuser` level (where `TimeOfArrival` already exists). Rejected as the *sole* path because the `PositionFuser` does a weighted-average of *independent* position estimates; deriving a position from a single range first requires a prior, and the best available prior is the CSI track state inside the tracker. Fusing at the tracker (§2.5) uses that prior correctly; fusing only at `mat` would need ≥3 simultaneous ranges to trilaterate a standalone position, which is a much stronger hardware requirement. We do **both**: tracker-level for single-range tightening, MAT-level for the multi-anchor trilateration use case.

### 4.4 Why a New `0xC511xxxx` Magic Rather Than a New Transport

UWB could ride its own port/protocol. Rejected to avoid a second aggregator, a second timesync, and a second firmware OTA channel. Extending the ADR-018 magic family (next id `0xC5110008`) means the existing `aggregator/` demux, the C6 802.15.4 clock, and the existing provisioning path all apply unchanged — the same reasoning that made the seven `RUVIEW_*_MAGIC` sibling packets share one port.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-018 (ESP32 Dev Implementation) | **Pattern reused**: `UWB_RANGE_MAGIC = 0xC5110008` extends the `0xC511xxxx` binary frame family; `UwbFrameParser` follows the `esp32_parser.rs` no-mock, pure-bytes contract and rides the same UDP aggregator |
| ADR-016 (RuVector Integration) | **Reused**: AETHER embedding cosine similarity for crossing disambiguation runs through the same `ruvector-mincut::DynamicPersonMatcher` path the tracker already uses |
| ADR-024 (Contrastive CSI Embedding / AETHER) | **Disambiguator**: 128-dim AETHER embeddings on `PoseTrack` resolve constraint-to-track association when tracks are equidistant from an anchor (§2.4) |
| ADR-029 (RuvSense Multistatic) | **Extended**: range constraints supply the front/back and scale information the geometric-diversity (Fisher-information) bounds show a near-colinear array cannot recover from CSI alone |
| ADR-031 (RuView Sensing-First RF Mode) | **Consumer**: fused metric tracks and constraint residuals are sensing-mode outputs surfaced to the RuView stream |
| ADR-063 (mmWave Sensor Fusion) | **Pattern parallel**: establishes the orthogonal-ranging-modality fusion pattern (`RUVIEW_FUSED_VITALS_MAGIC`); ADR-144 applies the same fusion philosophy to UWB metric range instead of 60 GHz radial velocity |
| ADR-136 (RuView Rust Streaming Engine) | **Stage**: the UWB parse → associate → fuse path is a stream stage producing constraint-augmented track frames under the ADR-136 frame contract |
| ADR-138 (LinkGroup / ArrayCoordinator) | **Clock**: shares the 802.15.4 timesync epoch the ArrayCoordinator uses for clock-quality gating, so UWB ranges and CSI frames associate by time |
| ADR-139 (WorldGraph Environmental Digital Twin) | **Substrate**: `RangeConstraint` becomes an `object_anchor → person_track` `RangeEdge`; the WorldGraph is the single source of truth for anchor geometry and `anchor_survey_version` |
| ADR-135 (Empty-Room Baseline Calibration) | **Analogue**: anchor survey/`anchor_survey_version` mirrors the baseline calibration/staleness-invalidation model; both refuse silent automatic self-update |
| ADR-140 / ADR-141 (SSR Schema / BFLD Privacy) | **Governed**: every fused range carries the signal-evidence + model-version + survey-version + privacy-decision provenance triple; identity-binding is gated by the active privacy mode |

---

## 6. References

### Production Code (verified to exist)

- `v2/crates/wifi-densepose-hardware/src/esp32_parser.rs` — ADR-018 binary frame parser; `ESP32_CSI_MAGIC = 0xC5110001` and the `RUVIEW_*_MAGIC` family (`0xC5110002`–`0xC5110007`) that the new `UWB_RANGE_MAGIC = 0xC5110008` extends
- `v2/crates/wifi-densepose-hardware/src/lib.rs` — crate root; no-mock guarantee; re-exports `CsiFrame`, `CsiMetadata`, `Esp32CsiParser`, the magic constants
- `v2/crates/wifi-densepose-hardware/src/aggregator/` — UDP multi-node ingest; gains one `is_uwb_frame()` demux arm
- `v2/crates/wifi-densepose-hardware/src/csi_frame.rs` — `CsiFrame`, `CsiMetadata`, `PpduType`; new `RangeConstraint`/`AnchorId`/`TagId` types live alongside these
- `v2/crates/wifi-densepose-signal/src/ruvsense/pose_tracker.rs` — `KeypointState::update()` / `mahalanobis_distance()`, `PoseTrack`, `PoseTracker`, `TrackerConfig`, `cosine_similarity`; gains `apply_range_constraint()` and `ConstraintTrackState`
- `v2/crates/wifi-densepose-mat/src/localization/fusion.rs` — `PositionFuser`, `EstimateSource::TimeOfArrival` (defined, currently unproduced), `LocalizationService::simulate_rssi_measurements()` (returns empty); gains `estimate_from_ranges()`
- `v2/crates/wifi-densepose-mat/src/localization/triangulation.rs` — `Triangulator` for the multi-anchor trilateration use case
- `archive/v1/data/proof/verify.py` + `expected_features.sha256` — deterministic proof chain; `uwb_range_fusion_v1` hash to be added
- `docs/WITNESS-LOG-028.md` — witness rows for parser round-trip, spherical EKF, crossing disambiguation

### Related ADRs (verified to exist as files)

- `docs/adr/ADR-018-esp32-dev-implementation.md`
- `docs/adr/ADR-016-ruvector-integration.md`
- `docs/adr/ADR-024-contrastive-csi-embedding-model.md`
- `docs/adr/ADR-029-ruvsense-multistatic-sensing-mode.md`
- `docs/adr/ADR-063-mmwave-sensor-fusion.md`
- `docs/adr/ADR-135-empty-room-baseline-calibration.md`

### External

- Qorvo DW3000 / DWM3000 802.15.4z UWB transceiver datasheet — SS/DS-TWR primitives and first-path-SNR link-quality reporting that backs `signal_quality` → `uncertainty_m`.
- IEEE 802.15.4z-2020 — Enhanced Ultra-Wideband PHY; defines the TWR/TDoA ranging schemes referenced in `RangeMethod`.
- Welford, B.P. (1962). *Technometrics* 4(3) — referenced for consistency with ADR-135's online statistics; the spherical EKF here uses the same diagonal-covariance conventions as the existing `KeypointState` Kalman math.


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `b10bc2e9a`, issue #848): the `RangeConstraint` domain model and `RangeConstraintFusion::refine()` -- a Newton-normalized weighted least-squares that constrains a CSI/CIR prior, with Mahalanobis outlier gating. 4 tests.

**Integration glue -- not yet on the live path:** the UWB UART driver/parser in `wifi-densepose-hardware` (no UWB module in the device table yet); wiring `refine()` into `pose_tracker`'s Kalman update; anchors as WorldGraph `UwbBeacon` nodes.

**Trust contribution:** physical-distance anchoring that *rejects* bogus multipath/NLOS ranges before they corrupt the estimate.
