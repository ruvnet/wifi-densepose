# ADR-138: WiFi-7 MLO LinkGroup Abstraction and ArrayCoordinator Clock-Quality Gating

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see Implementation Status, commit `fc7674bde`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (`ruvsense/multiband.rs`, `ruvsense/multistatic.rs`); `wifi-densepose-ruvector` (`viewpoint/geometry.rs`, `viewpoint/coherence.rs`, `viewpoint/attention.rs`, `viewpoint/fusion.rs`) |
| **Relates to** | ADR-008 (CSI Frame Primitives), ADR-029 (RuvSense Multistatic), ADR-030 (Persistent Field Model), ADR-031 (RuView Sensing-First RF Mode), ADR-110 (ESP32-C6 Firmware Extension / 802.15.4 sync), ADR-136 (RuView Rust Streaming Engine — frame contracts), ADR-137 (Fusion Engine Quality Scoring — evidence references and contradiction flags) |

---

## 1. Context

### 1.1 The Gap

Searching across the two named crates for `LinkGroup`, `ArrayCoordinator`, `clock_quality`, `DirectionalEvidence`, and `FreqSet` finds no production module. The pieces that an MLO-aware coordinator would compose all exist, but each is wired to a *single* CSI stream, a *single* clock domain, and emits a *hard fused output* rather than weighted evidence. Concretely:

- **`ruvsense/multiband.rs`** has `MultiBandCsiFrame { node_id, timestamp_us, channel_frames: Vec<CanonicalCsiFrame>, frequencies_mhz: Vec<u32>, coherence }` and a `MultiBandBuilder` that fuses per-channel rows from a *channel-hopping* radio (one ESP32-S3 cycling 1/6/11). This is the closest thing to a per-band feature stream, but it models **sequential** channel hopping on one radio, not **simultaneous** WiFi-7 Multi-Link Operation (MLO) where bands stream concurrently. There is no aggregate that tracks which bands are currently *live* versus which have *dropped out*, and `coherence` is a single Pearson scalar (`compute_cross_channel_coherence`), not an inter-band consensus with promotion semantics.

- **`ruvsense/multistatic.rs`** has `MultistaticFuser::fuse(&[MultiBandCsiFrame]) -> FusedSensingFrame`. It already validates a `guard_interval_us` timestamp spread (`MultistaticConfig.guard_interval_us`, default 5000 µs) and computes `geometric_diversity(&[[f32;3]])` from node positions. But: (a) the timestamp spread is a hard accept/reject — there is no notion of *clock quality* (a node whose clock is merely *uncertain* is treated identically to one whose clock is *good*); (b) `geometric_diversity()` is a free function returning a bare `f32`, not gated into the fusion decision; (c) the output `FusedSensingFrame` is a committed `fused_amplitude`/`fused_phase` pose-bearing artifact, not directional evidence with credence intervals.

- **`viewpoint/geometry.rs`** has `GeometricDiversityIndex::compute(azimuths, node_ids) -> Option<Self>` with `value`, `n_effective`, `worst_pair`, `is_sufficient()` (threshold `value >= PI/N`), plus `CramerRaoBound::estimate(target, &[ViewpointPosition]) -> Option<Self>` returning `crb_x`, `crb_y`, `rmse_lower_bound`, `gdop`. This is exactly the GDI + Cramér-Rao machinery this ADR needs to convert into a gate and into credence intervals — but nothing currently calls it from the multistatic path. The two `geometric_diversity` implementations (the `multistatic.rs` free function and the `geometry.rs` `GeometricDiversityIndex`) are unaware of each other.

- **`viewpoint/coherence.rs`** has `CoherenceState` (rolling phasor window with `push`/`coherence()`) and `CoherenceGate { threshold, hysteresis, evaluate() }`. The gate already implements hysteresis and a duty cycle. But it gates **only on phase coherence** — there is no clock-quality term, and no "contradiction" notion: a coherence drop merely closes the gate, it does not demote a band/group to monitoring-only nor flag the contradiction for downstream.

- **`viewpoint/fusion.rs`** has `MultistaticArray` (the DDD aggregate root) with `submit_viewpoint`, `push_phase_diff`, `fuse() -> FusedEmbedding`, `compute_gdi()`, and a `ViewpointFusionEvent` enum (`ViewpointCaptured`, `TdmCycleCompleted`, `FusionCompleted`, `CoherenceGateTriggered`, `GeometryUpdated`). `fuse()` already filters by SNR and gates on coherence, returning `FusionError::CoherenceGateClosed` when the environment is unstable. But the aggregate is keyed on **embeddings** (AETHER 128-d vectors) and produces a **pose-feeding `FusedEmbedding`** — there is no per-band lifecycle, no clock-quality input, and the "gate closed" path silently drops the cycle rather than demoting to a monitoring-only state that still emits evidence.

- **`wifi-densepose-hardware/src/sync_packet.rs`** is fully implemented: `SyncPacket` decodes the ADR-110 §A0.12 wire format (magic `0xC511A110`, 32 bytes LE), exposes `local_minus_epoch_us()`, `apply_to_local()`, and `mesh_aligned_us_for_sequence(frame_seq, fps_hz)`. The sensing server (`wifi-densepose-sensing-server/src/main.rs`) already dispatches on `SYNC_PACKET_MAGIC` and applies a 9-second staleness gate (`mesh_aligned_us_for_csi_frame`). What is missing: a **clock-quality score** derived from the sync stream (offset dispersion / leader-vs-follower / staleness) that the *signal-domain* fusion can consult. The hardware crate recovers `mesh_aligned_us` but never propagates a *quality* of that alignment into `multistatic.rs` or `viewpoint/`.

The consequence: the array treats every node as if its clock were perfect and its geometry adequate, and it commits to a fused pose even when (a) only one MLO band survived, (b) the contributing nodes are clustered (low GDI), or (c) a node's clock has drifted past the point where its phase is comparable to its peers. ADR-137 (sibling, Proposed) requires every fused output to carry **evidence references and contradiction flags**; ADR-136 (sibling, Proposed) defines the `FrameMeta` frame contract that should carry `mesh_aligned_us` and clock metadata per frame. This ADR supplies the missing middle: a lifetime-managed `LinkGroup` that knows which bands are live, and an `ArrayCoordinator` service that gates on geometry *and* clock quality and emits `DirectionalEvidence` instead of a hard decision.

### 1.2 What "LinkGroup" and "ArrayCoordinator" Mean Here

- A **LinkGroup** is a lifetime-managed aggregate representing one *physical link* operating WiFi-7 MLO: a set of concurrent bands (2.4 / 5 / 6 GHz) that the radio streams simultaneously, each producing its own `CanonicalCsiFrame`. The LinkGroup wraps a `FreqSet` (the declared band membership) plus a rolling `Vec<MultiBandCsiFrame>` per band, and tracks **band lifecycle** — a band can `enter` (start streaming), `exit` (drop out, e.g. 6 GHz lost when the AP reboots), and be `promoted` to the consensus set once it agrees with its peers. This is distinct from today's `MultiBandCsiFrame`, which is a *snapshot* of one hop cycle with no membership lifecycle.

- An **ArrayCoordinator** is a **service** (not an aggregate). It consumes a set of `LinkGroup`s plus the per-node frames already modelled by `multistatic.rs`, applies two gates — a **geometry gate** (GDI / Cramér-Rao from `viewpoint/geometry.rs`) and a **clock-quality gate** (ADR-110 sync dispersion) — and returns `DirectionalEvidence`: attention weights per viewpoint plus credence intervals derived from the Cramér-Rao bound. It does **not** decide pose. The pose/semantic decision is downstream (ADR-137 fusion-engine quality scoring); the coordinator only says "here is what the array can and cannot see right now, and how much to trust each direction."

### 1.3 Why Not a Single Hard Gate

The existing `CoherenceGate::evaluate()` and `MultistaticConfig.guard_interval_us` are both **binary**: update / no-update, accept / reject. WiFi-7 MLO and multi-node arrays degrade *gracefully* — losing the 6 GHz band, or a node whose clock dispersion rose from 40 µs to 180 µs, does not invalidate the array; it narrows what it can resolve and widens the credence interval. A hard gate throws away usable evidence. The decision below replaces the binary gates with a **graded** coordinator output that downgrades rather than discards, and feeds the graded result into ADR-137's contradiction machinery.

### 1.4 Pipeline Position

```
Per-band CSI (MLO: 2.4 / 5 / 6 GHz concurrent)
  → multiband.rs MultiBandBuilder          (per-band CanonicalCsiFrame rows)
  → LinkGroup::ingest()           ← NEW     (band enter/exit + consensus promote)
  → ArrayCoordinator::coordinate()  ← NEW   (service: GDI gate + clock-quality gate)
        │  consumes: Vec<LinkGroup>, node_frames, Vec<SyncPacket> (ADR-110)
        │  uses:     GeometricDiversityIndex + CramerRaoBound (viewpoint/geometry.rs)
        │            ClockQualityGate  ← NEW (wraps viewpoint/coherence.rs CoherenceGate)
        ▼
  → DirectionalEvidence            ← NEW     (attention weights + credence intervals)
  → multistatic.rs MultistaticFuser.fuse()  (consumes weights, NOT a re-decision)
  → ADR-137 FusionEngine quality scoring + contradiction flags
```

The coordinator sits *between* per-band ingestion and the existing `MultistaticFuser`. It does not replace `fuse()`; it supplies the weights `fuse()` already wants (today `attention_weighted_fusion` derives them internally from amplitude similarity only) and the contradiction flags ADR-137 consumes.

---

## 2. Decision

### 2.1 `LinkGroup`: Lifetime-Managed MLO Aggregate

A `LinkGroup` is added to `ruvsense/multiband.rs` (it composes the existing `MultiBandCsiFrame` and `CanonicalCsiFrame`). It is an aggregate with explicit band lifecycle, not a snapshot.

```rust
use crate::hardware_norm::CanonicalCsiFrame;

/// The declared set of MLO bands a link operates on (WiFi-7: up to 3).
/// Membership is *declared* at construction; liveness is tracked separately.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreqSet {
    /// Center frequencies (MHz), sorted ascending. e.g. [2412, 5180, 5955].
    pub bands_mhz: Vec<u32>,
}

impl FreqSet {
    pub fn new(mut bands_mhz: Vec<u32>) -> Self {
        bands_mhz.sort_unstable();
        bands_mhz.dedup();
        Self { bands_mhz }
    }
    pub fn contains(&self, freq_mhz: u32) -> bool { self.bands_mhz.contains(&freq_mhz) }
    pub fn len(&self) -> usize { self.bands_mhz.len() }
    pub fn is_empty(&self) -> bool { self.bands_mhz.is_empty() }
}

/// Lifecycle state of one band within a LinkGroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandState {
    /// Declared in the FreqSet but no frame seen yet (warm-up).
    Pending,
    /// Streaming frames, but not yet agreeing with peers.
    Live,
    /// Live AND consensus-promoted: agrees with the group's other live bands.
    Promoted,
    /// Was Live, has missed `exit_after_missed` expected frames.
    Exited,
}

/// Domain events emitted by a LinkGroup (event-sourced state changes, per house rule).
#[derive(Debug, Clone, PartialEq)]
pub enum LinkGroupEvent {
    BandEntered { freq_mhz: u32, at_us: u64 },
    BandExited  { freq_mhz: u32, at_us: u64, missed: u32 },
    BandPromoted { freq_mhz: u32, at_us: u64, consensus: f32 },
    BandDemoted  { freq_mhz: u32, at_us: u64, reason: DemotionReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemotionReason {
    /// Inter-band consensus dropped below threshold.
    ConsensusLoss,
    /// Coherence fell >2σ from the rolling mean (contradiction; §2.5).
    CoherenceContradiction,
}

#[derive(Debug, thiserror::Error)]
pub enum LinkGroupError {
    #[error("Frequency {freq_mhz} MHz is not a member of this LinkGroup's FreqSet")]
    UnknownBand { freq_mhz: u32 },
    #[error("Subcarrier count mismatch on band {freq_mhz}: expected {expected}, got {got}")]
    SubcarrierMismatch { freq_mhz: u32, expected: usize, got: usize },
}

/// A WiFi-7 MLO physical link: a FreqSet plus per-band feature streams with
/// explicit enter/exit and consensus-promotion lifecycle.
///
/// # Concurrency
/// Requires `&mut self` for `ingest()`; not `Sync`. One ingest loop per link.
#[derive(Debug)]
pub struct LinkGroup {
    node_id: u8,
    freq_set: FreqSet,
    /// Most recent frame per band, indexed parallel to `freq_set.bands_mhz`.
    latest: Vec<Option<CanonicalCsiFrame>>,
    /// Lifecycle state per band (parallel to `freq_set.bands_mhz`).
    state: Vec<BandState>,
    /// Rolling per-band inter-band consensus score (Pearson vs. the group mean).
    consensus: Vec<f32>,
    /// Frame count per band since last seen, for exit detection.
    missed: Vec<u32>,
    /// Config: promote/exit thresholds.
    config: LinkGroupConfig,
    /// Pending domain events (drained by the ArrayCoordinator).
    events: Vec<LinkGroupEvent>,
}

#[derive(Debug, Clone)]
pub struct LinkGroupConfig {
    /// Pearson consensus required to promote a Live band to Promoted. Default 0.6.
    pub promote_consensus: f32,
    /// Consecutive missed expected frames before a Live band Exits. Default 5.
    pub exit_after_missed: u32,
}

impl Default for LinkGroupConfig {
    fn default() -> Self { Self { promote_consensus: 0.6, exit_after_missed: 5 } }
}

impl LinkGroup {
    pub fn new(node_id: u8, freq_set: FreqSet, config: LinkGroupConfig) -> Self;

    /// Ingest one band's frame. Marks the band Live (emitting BandEntered on the
    /// first frame), recomputes inter-band consensus against the current live
    /// mean, promotes/demotes per thresholds, and ages out unseen bands toward
    /// Exited. Bands not in `freq_set` are rejected with `UnknownBand`.
    pub fn ingest(&mut self, freq_mhz: u32, frame: CanonicalCsiFrame, at_us: u64)
        -> Result<(), LinkGroupError>;

    /// Bands currently in the consensus (Promoted) set.
    pub fn promoted_bands(&self) -> Vec<u32>;

    /// Build a MultiBandCsiFrame from the currently Promoted bands only.
    /// Returns None if fewer than 1 band is Promoted.
    pub fn consensus_frame(&self, at_us: u64) -> Option<MultiBandCsiFrame>;

    /// Drain pending domain events (the ArrayCoordinator forwards these to ADR-137).
    pub fn drain_events(&mut self) -> Vec<LinkGroupEvent>;
}
```

Inter-band consensus reuses the existing `pearson_correlation_f32` already in `multiband.rs` (private today; promoted to `pub(crate)`). The `consensus_frame()` output is intentionally a `MultiBandCsiFrame`, so the existing `MultistaticFuser` consumes it unchanged.

**Why an aggregate, not a snapshot.** MLO band membership is *stateful*: the 6 GHz band dropping for 250 ms and returning is a different physical situation from a node permanently losing 6 GHz. A snapshot (`MultiBandCsiFrame`) cannot represent "this band exited and we are now operating degraded." The lifecycle (`Pending → Live → Promoted`, with `→ Exited` and `→ Demoted` transitions) is the minimum state required to (a) feed graceful degradation into the coordinator and (b) emit the band-level contradiction events ADR-137 wants.

### 2.2 `ClockQualityScore` and the Clock-Quality Gate

A clock-quality term is derived from the ADR-110 `SyncPacket` stream and folded into a gate alongside the existing phase-coherence gate. The score lives in `viewpoint/coherence.rs` next to `CoherenceState`/`CoherenceGate`.

```rust
/// Per-node clock-quality summary derived from the ADR-110 sync stream.
///
/// All fields are computed by the host from the `SyncPacket` series for one
/// node (`wifi_densepose_hardware::sync_packet::SyncPacket`).
#[derive(Debug, Clone, Copy)]
pub struct ClockQualityScore {
    /// EMA stdev of (local_us - epoch_us) over the recent sync window (µs).
    /// This is the dispersion of the node's mesh-alignment offset.
    pub offset_stdev_us: f32,
    /// 802.15.4 stratum: 0 = leader, 1 = direct follower, etc.
    pub stratum: u8,
    /// Age of the most recent valid SyncPacket (µs); large = stale.
    pub age_us: u64,
    /// Whether the most recent packet had flags.is_valid set.
    pub valid: bool,
}

impl ClockQualityScore {
    /// Normalised quality in [0, 1]: 1.0 = leader-grade, 0.0 = unusable.
    /// Combines offset dispersion (vs. the ADR-110 ±100 µs target), stratum
    /// penalty, and staleness. 0.0 if `!valid`.
    pub fn quality(&self) -> f32;

    /// Convenience: the ADR-110 ±100 µs sync target as a hard usability floor.
    /// `offset_stdev_us < 200.0` (2× the target) is the gate's default accept.
    pub const SYNC_TARGET_US: f32 = 100.0;
}

/// Gate that admits a node's frames into directional fusion only when both
/// its phase coherence AND its clock quality are adequate. Wraps the existing
/// `CoherenceGate` (phase term) and adds the clock term.
#[derive(Debug, Clone)]
pub struct ClockQualityGate {
    /// Existing phase-coherence gate (unchanged semantics).
    pub coherence: CoherenceGate,
    /// Reject when offset_stdev_us >= this. Default 200.0 (2× ADR-110 target).
    pub max_offset_stdev_us: f32,
    /// Reject when sync age exceeds this. Default 9_000_000 (the sensing-server
    /// 9-second staleness gate already used in main.rs).
    pub max_age_us: u64,
}

impl ClockQualityGate {
    pub fn new(coherence: CoherenceGate, max_offset_stdev_us: f32, max_age_us: u64) -> Self;
    pub fn default_params() -> Self {
        Self::new(CoherenceGate::default_params(), 200.0, 9_000_000)
    }

    /// Evaluate both terms. Returns the gate decision for one node this cycle.
    /// `coherence_value` is the rolling phasor coherence (CoherenceState::coherence()).
    pub fn evaluate(&mut self, coherence_value: f32, clock: &ClockQualityScore)
        -> ClockGateDecision;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockGateDecision {
    /// Both terms pass: node admitted at full weight.
    Admit,
    /// Phase OK but clock degraded: admit at reduced weight (monitoring-only;
    /// frame contributes to evidence but NOT to model/environment update).
    MonitorOnly { clock_quality: f32 },
    /// Either term fails hard: node excluded this cycle.
    Reject { reason: ClockRejectReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockRejectReason { Incoherent, ClockStale, ClockDispersed, ClockInvalid }
```

**Why a 200 µs default floor.** ADR-110 §A0.10 measured the COM9↔COM12 follower offset stdev at ~104 µs after EMA smoothing, against the ±100 µs 802.15.4 target. A node whose dispersion has risen to 2× the measured baseline (200 µs) has lost roughly one phase wrap of cross-node comparability at 5 GHz (wavelength ≈ 5 cm; 200 µs of clock skew at sensing motion velocities corrupts the inter-node phase term that `attention_weighted_fusion` relies on). Below 200 µs the node is admitted; between 200 µs and the staleness ceiling it is `MonitorOnly` (evidence yes, environment update no); above the 9 s age ceiling — the same staleness gate the sensing server already enforces (`main.rs::mesh_aligned_us_honors_9s_staleness_gate`) — it is rejected.

**Why gate environment updates specifically.** The clock term must not block *evidence emission* — a clock-degraded node still sees real motion and should contribute weighted evidence. It must block *environment/model updates* (ADR-030 field model, ADR-031 model update path), because those updates assume cross-node phase comparability that a dispersed clock breaks. `MonitorOnly` encodes exactly this: contribute to `DirectionalEvidence`, do not promote to a model/environment change. This mirrors the existing `CoherenceGate` semantics ("only allow model updates when coherence exceeds threshold") and extends them with the clock dimension.

### 2.3 `ArrayCoordinator`: a Service, Not an Aggregate

`ArrayCoordinator` is added to `viewpoint/fusion.rs` alongside `MultistaticArray`. It holds no long-lived domain state of its own (the lifecycle state lives in the `LinkGroup`s and `MultistaticArray`); it is a stateless-per-call **domain service** that applies gates and projects evidence.

```rust
use crate::viewpoint::geometry::{GeometricDiversityIndex, CramerRaoBound, ViewpointPosition, NodeId};
use crate::viewpoint::coherence::{ClockQualityGate, ClockQualityScore, ClockGateDecision};

/// Directional evidence: what the array can resolve right now, and how much to
/// trust each direction. This is the coordinator's output — NOT a pose decision.
///
/// Per the house rule that every semantic state traces to evidence, this struct
/// carries the geometry + clock provenance that ADR-137 attaches to any state
/// it derives downstream.
#[derive(Debug, Clone)]
pub struct DirectionalEvidence {
    /// Per-viewpoint attention weight (softmax, sums to 1.0 over admitted nodes).
    pub weights: Vec<(NodeId, f32)>,
    /// Geometric Diversity Index at evaluation time.
    pub gdi: GeometricDiversityIndex,
    /// Cramér-Rao credence interval: RMSE lower bound (m) for a centroid target.
    /// `None` when fewer than 3 admitted viewpoints (under-determined).
    pub credence_rmse_m: Option<f32>,
    /// Per-node gate decisions (Admit / MonitorOnly / Reject) — the audit trail.
    pub gate_decisions: Vec<(NodeId, ClockGateDecision)>,
    /// Contradiction flags forwarded to ADR-137 (see §2.5).
    pub contradictions: Vec<ContradictionFlag>,
    /// Number of viewpoints admitted at full weight (Admit).
    pub n_admitted: usize,
    /// Number admitted MonitorOnly (evidence-only, no environment update).
    pub n_monitoring: usize,
}

// `ContradictionFlag` is NOT redefined here. It is the canonical enum owned by
// ADR-137 §2.3 (`wifi-densepose-signal::ruvsense::multistatic`). The coordinator
// imports it and emits only its array-origin variants:
//
//   use wifi_densepose_signal::ruvsense::multistatic::ContradictionFlag;
//
//   ContradictionFlag::CoherenceDrop { node_idx, sigma }   // coherence > Nσ off rolling mean
//   ContradictionFlag::GeometryInsufficient { gdi }        // array GDI below the floor
//
// A previously-Promoted band being demoted (inter-band disagreement) is surfaced
// through the per-node `gate_decisions` audit trail above, not as a contradiction
// flag — it suppresses the model update without contradicting the observation.
// `NodeId` → `node_idx` resolution happens at the ADR-137 hand-off (ADR-137 §2.3).

#[derive(Debug, Clone)]
pub struct ArrayCoordinatorConfig {
    /// Per-node clock+coherence gate.
    pub gate: ClockQualityGate,
    /// σ multiple defining a coherence contradiction. Default 2.0.
    pub contradiction_sigma: f32,
    /// Per-measurement noise std (m) for the Cramér-Rao credence estimate.
    pub crb_noise_std_m: f32,
    /// Attention temperature for the directional weight softmax. Default 1.0.
    pub attention_temperature: f32,
}

/// Domain service: gates LinkGroups + node frames on geometry and clock quality,
/// returns DirectionalEvidence. Holds NO aggregate state.
pub struct ArrayCoordinator {
    config: ArrayCoordinatorConfig,
}

impl ArrayCoordinator {
    pub fn new(config: ArrayCoordinatorConfig) -> Self;

    /// The single service operation. For each node:
    ///   1. Take its LinkGroup consensus frame (Promoted bands only).
    ///   2. Evaluate the clock-quality gate (coherence × clock).
    ///   3. Admit / MonitorOnly / Reject.
    /// Then over the admitted set:
    ///   4. Compute GDI (geometry.rs); raise GeometryInsufficient if !is_sufficient().
    ///   5. Compute Cramér-Rao credence RMSE for a centroid target.
    ///   6. Build attention weights (softmax over admitted nodes, biased by clock
    ///      quality and inverse-CRB so well-placed, well-clocked nodes weigh more).
    ///   7. Collect contradiction flags from LinkGroup demotions + coherence drops.
    ///
    /// `coherence_per_node` and `clock_per_node` are parallel to `viewpoints`.
    pub fn coordinate(
        &mut self,
        viewpoints: &[(NodeId, f32 /*azimuth*/, ViewpointPosition)],
        coherence_per_node: &[f32],
        clock_per_node: &[ClockQualityScore],
        link_events: &[LinkGroupEventRef],
    ) -> DirectionalEvidence;
}
```

The coordinator deliberately reuses, not reimplements:
- `GeometricDiversityIndex::compute` + `is_sufficient()` for the geometry gate.
- `CramerRaoBound::estimate` for the credence interval (its `rmse_lower_bound` *is* the credence radius).
- `ClockQualityGate::evaluate` for the per-node admit/monitor/reject decision.
- The softmax shape from `multistatic.rs::attention_weighted_fusion` (numerically stable, subtract-max), but biased by clock quality and inverse-CRB rather than amplitude-cosine alone.

**Why a service rather than folding this into `MultistaticArray`.** `MultistaticArray` is the *aggregate root* for ViewpointFusion — it owns embedding lifecycle and the coherence window. The coordinator's job spans *multiple* aggregates (every node's `LinkGroup` plus the array) and is *read-mostly*: it inspects state and projects evidence, but the authoritative state transitions (band promotion, viewpoint upsert) belong to the aggregates. Putting cross-aggregate gating logic in a stateless service keeps the aggregate boundaries clean (DDD) and makes the coordinator trivially testable with synthetic inputs.

### 2.4 Wiring the ADR-110 SyncPacket Decoder Into the Pipeline

Today `SyncPacket` is decoded in `wifi-densepose-sensing-server/src/main.rs` and used only to recover `mesh_aligned_us`. This ADR widens that path so the recovered alignment carries a *quality*:

1. The sensing server already keeps `NodeState::latest_sync: Option<SyncPacket>` and `latest_sync_at: Option<Instant>`. Add a rolling buffer `NodeState::sync_offsets: VecDeque<i64>` of the last N `local_minus_epoch_us()` values and an EMA. From these, build a `ClockQualityScore { offset_stdev_us, stratum, age_us, valid }` per node per cycle.
   - `stratum` is derived from `SyncPacketFlags::is_leader` (leader = 0, follower = 1; deeper strata are reserved).
   - `age_us` is `now - latest_sync_at` in the mesh domain.
   - `valid` is `latest_sync.flags.is_valid`.
2. Per ADR-136, the per-frame `FrameMeta` contract gains `mesh_aligned_us: Option<u64>` and `clock_quality: Option<ClockQualityScore>`, populated at frame ingestion by pairing `(node_id, sequence)` against the most recent `SyncPacket` (exactly the pairing `mesh_aligned_us_for_sequence` already implements). This keeps the *signal* crates free of any UDP/socket dependency — they receive `FrameMeta`, not raw packets.
3. The `ArrayCoordinator::coordinate()` call receives `clock_per_node: &[ClockQualityScore]` extracted from those `FrameMeta` records. No new socket code lands in `wifi-densepose-signal` or `wifi-densepose-ruvector`; the hardware crate remains the only owner of the wire format (`SYNC_PACKET_MAGIC = 0xC511A110`).

This preserves the existing crate dependency direction: hardware → (FrameMeta) → signal/ruvector. The coordinator never imports `wifi-densepose-hardware`; it sees only the `ClockQualityScore` value object.

### 2.5 Contradiction-to-Environment-Change Semantics

The coordinator converts two array-level conditions into ADR-137 contradiction flags, and uses them to demote rather than to commit:

- **Coherence drop > 2σ.** Each node's `CoherenceState` already maintains a rolling phasor coherence. The coordinator additionally tracks a rolling mean/std of that coherence per node (Welford, consistent with ADR-135's reuse of `WelfordStats`). When the current coherence falls more than `contradiction_sigma` (default 2.0) below the rolling mean, the coordinator (a) raises `ContradictionKind::CoherenceDrop { magnitude }`, and (b) the node's `ClockQualityGate` returns at most `MonitorOnly` for that cycle — its frame contributes evidence but cannot trigger an environment/model update. This is the signal-domain analogue of `LinkGroupEvent::BandDemoted { reason: CoherenceContradiction }`.

- **GDI below the sufficiency floor.** `GeometricDiversityIndex::is_sufficient()` already encodes the `value >= (2π/N) × 0.5` floor. When the admitted set's GDI is insufficient, the coordinator raises `ContradictionKind::GeometryInsufficient { magnitude: gdi.value }` and widens the credence interval (the Cramér-Rao `rmse_lower_bound` already grows automatically as geometry degrades, so this flag is advisory for ADR-137, not a separate widening).

A `LinkGroup` band demotion (`BandDemoted`) is forwarded verbatim as `ContradictionKind::BandDemoted`. In all three cases the rule is identical and is the core of this ADR: **a contradiction demotes to monitoring-only; it never forces an environment change.** Only a sustained *consensus* (admitted nodes agreeing across a window) promotes an environment update — and that promotion is owned downstream by ADR-137, which receives the coordinator's `DirectionalEvidence` complete with its contradiction list.

### 2.6 Provenance / Evidence Tracing

Per the project rule that every semantic state traces to signal evidence + model version + calibration version + privacy decision, the `DirectionalEvidence` struct is designed as the *evidence* half of that chain:

- **Signal evidence**: the per-node `weights` and `gate_decisions` are the audit trail of which viewpoints (and which MLO bands, via the `LinkGroup` consensus) contributed and how much.
- **Calibration version**: when an ADR-135 `BaselineCalibration` is loaded for a node, its `captured_at_unix_s`/device id flow through `FrameMeta`; the coordinator does not re-derive calibration but passes it through so ADR-137 can stamp it.
- **Model / privacy version**: these are not the coordinator's concern (it makes no model inference and no privacy decision); ADR-137 attaches `model_version` and the active privacy decision when it consumes `DirectionalEvidence`. The coordinator's contract is to make the evidence and contradiction set *complete enough* that ADR-137 can construct the full provenance tuple without re-reading raw frames.

### 2.7 Downstream Consumers and Interface Boundaries

| Consumer | What it receives | Change required |
|----------|-----------------|-----------------|
| `multistatic.rs::MultistaticFuser::fuse()` | `DirectionalEvidence.weights` instead of internally-derived amplitude-cosine weights | `MultistaticConfig` gains `external_weights: Option<Vec<(u8, f32)>>`; when present, `attention_weighted_fusion` uses them rather than recomputing. Backward compatible (`None` = today's behaviour). |
| `multiband.rs::MultiBandBuilder` | Unchanged; `LinkGroup::consensus_frame()` produces a `MultiBandCsiFrame` it already understands | No change to `MultiBandBuilder`; `pearson_correlation_f32` promoted to `pub(crate)` for `LinkGroup` reuse |
| `viewpoint/fusion.rs::MultistaticArray` | Coordinator runs *before* `fuse()`; the `CoherenceGateClosed` path is replaced by `MonitorOnly` evidence | New `ViewpointFusionEvent::DirectionalEvidenceEmitted { gdi, n_admitted, n_monitoring }`; `fuse()` no longer hard-drops on closed coherence — it returns evidence with zero admitted nodes |
| `viewpoint/geometry.rs` | Called by the coordinator (`GeometricDiversityIndex`, `CramerRaoBound`) | No API change; the existing `is_sufficient()` and `rmse_lower_bound` are exactly the gate/credence primitives |
| `viewpoint/coherence.rs` | Hosts the new `ClockQualityScore` / `ClockQualityGate` next to `CoherenceGate` | New types added; existing `CoherenceGate`/`CoherenceState` unchanged and reused as the phase term |
| ADR-137 FusionEngine | `DirectionalEvidence` (weights + credence + `contradictions`) | The coordinator is ADR-137's upstream; `ContradictionFlag` is the agreed hand-off type |
| ADR-136 streaming engine | Populates `FrameMeta.mesh_aligned_us` + `clock_quality` | The coordinator reads these from `FrameMeta`; ADR-136 owns the frame contract |

**Interface boundary statement.** The coordinator's only inputs are value objects (`ViewpointPosition`, `f32` coherence, `ClockQualityScore`, `LinkGroupEventRef`); its only output is the `DirectionalEvidence` value object. It imports from `viewpoint::geometry` and `viewpoint::coherence` within the same crate, and is invoked by the sensing server / streaming engine which assemble the inputs. It does **not** import `wifi-densepose-hardware`, does **not** touch sockets, and does **not** make pose or privacy decisions.

### 2.8 Test Plan / Acceptance Criteria

**T1 — LinkGroup band lifecycle (unit).** Construct a `LinkGroup` with `FreqSet::new(vec![2412, 5180, 5955])`. Ingest 2.4 + 5 GHz frames that correlate (consensus > 0.6) for 10 cycles; ingest 6 GHz frames that do not. Assert: 2.4 and 5 GHz reach `BandState::Promoted` (emitting `BandPromoted`); 6 GHz stays `Live`; `promoted_bands() == [2412, 5180]`; `consensus_frame()` yields a 2-band `MultiBandCsiFrame`.

**T2 — Band exit and re-entry (unit).** With the same group, stop feeding 6 GHz for `exit_after_missed` (5) cycles → assert `BandExited` emitted and state `Exited`. Resume 6 GHz → assert `BandEntered` emitted and state returns to `Live`.

**T3 — Clock-quality gate thresholds (unit).** Build `ClockQualityScore`s: (a) `offset_stdev_us = 50, valid = true, age_us = 1_000_000` → `quality() > 0.8` and gate `Admit`; (b) `offset_stdev_us = 250` (> 200 floor) but coherent → gate `MonitorOnly`; (c) `age_us = 10_000_000` (> 9 s) → gate `Reject { ClockStale }`; (d) `valid = false` → `Reject { ClockInvalid }` and `quality() == 0.0`.

**T4 — ArrayCoordinator geometry gate + credence (unit).** Four nodes at the corners of a 5×5 m room (reuse `geometry.rs::gdi_four_corners` layout), all `Admit`. Assert: `gdi.is_sufficient()`; `credence_rmse_m` is `Some` and decreases when a 5th well-placed node is added (mirrors `crb_decreases_with_more_viewpoints`); `weights` sum to 1.0; `n_admitted == 4`.

**T5 — Clustered nodes raise GeometryInsufficient (unit).** Four nodes clustered within 0.12 rad (reuse `gdi_clustered_viewpoints_have_low_value`). Assert `ContradictionKind::GeometryInsufficient` present and `credence_rmse_m` is much larger than T4.

**T6 — Coherence-drop contradiction demotes, not decides (unit).** Feed one node a stable coherence (~0.8) for 30 cycles to seed the rolling mean, then a single 0.2 coherence (> 2σ drop). Assert: `ContradictionKind::CoherenceDrop` raised for that node; its gate decision is at most `MonitorOnly`; the node still appears in `weights` (evidence preserved); `n_monitoring >= 1`.

**T7 — SyncPacket → ClockQualityScore (unit, hardware crate test reuse).** Using the canonical COM9 follower packet from `sync_packet.rs` (`local_minus_epoch_us() == 1_163_565`) and the COM12 leader packet, build offset series and assert: leader → `stratum == 0`, high `quality()`; follower with low dispersion → `Admit`. Assert no `wifi-densepose-hardware` symbol leaks into the coordinator's public API (compile-fence test).

**T8 — Determinism proof (CI-compatible, extends ADR-028 chain).** Drive a fixed synthetic 3-band, 4-node scenario through `LinkGroup::ingest` → `ArrayCoordinator::coordinate`, serialise `DirectionalEvidence.weights` (rounded to f32) and the sorted contradiction kinds, and SHA-256 the result. Record under `archive/v1/data/proof/expected_features.sha256` as `array_coordinator_evidence_v1`; `verify.py` regenerates and asserts the hash.

**Acceptance gate**: `cargo test -p wifi-densepose-signal -p wifi-densepose-ruvector --no-default-features` passes all of T1–T8; no new `unsafe`; the coordinator's public API contains no type from `wifi-densepose-hardware`.

---

## 3. Consequences

### 3.1 Positive

- **Graceful MLO degradation.** Losing the 6 GHz band narrows resolution and widens the credence interval rather than invalidating the link. The `LinkGroup` lifecycle makes "degraded but operating" a first-class state instead of an undetected silent failure.
- **Clock quality becomes observable and actionable.** Today a drifting node is treated identically to a good one until it crosses the 9 s staleness cliff. The `ClockQualityScore` exposes the *continuum*, and `MonitorOnly` lets a clock-degraded node still contribute evidence without corrupting environment updates.
- **Evidence, not premature decisions.** The coordinator emits `DirectionalEvidence` with attention weights and Cramér-Rao credence intervals, giving ADR-137 the provenance it needs and removing the hard `CoherenceGateClosed` drop that currently discards usable cycles.
- **Reuse over reinvention.** GDI, Cramér-Rao, coherence gate, sync-packet decode, and Pearson consensus already exist and are tested; this ADR composes them. The two duplicate `geometric_diversity` notions converge on `viewpoint/geometry.rs`.
- **Clean crate boundaries preserved.** No socket or wire-format code enters the signal/ruvector crates; the `FrameMeta` contract (ADR-136) is the only coupling point.

### 3.2 Negative

- **More state to manage.** `LinkGroup` adds per-band lifecycle state and an event buffer. For a 4-node, 3-band array that is 12 band state machines plus the coordinator — modest, but non-zero, and the events must be drained or they accumulate (bounded like `MultistaticArray::max_events`).
- **Two gates instead of one.** Operators and tests must reason about coherence *and* clock quality. The `MonitorOnly` middle state, while useful, is a third outcome that downstream code (ADR-137) must handle explicitly rather than a simple boolean.
- **Depends on sibling ADRs not yet landed.** `FrameMeta` (ADR-136) and the contradiction-consumer (ADR-137) are both Proposed. Until they land, the coordinator can be tested with synthetic `ClockQualityScore`s but cannot be wired end-to-end. The `mesh_aligned_us` plumbing exists today only in the sensing server, not in a shared `FrameMeta`.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `offset_stdev_us` is noisy on small sync windows, causing gate flapping between Admit/MonitorOnly | Medium | Weights jitter cycle-to-cycle | Use the `CoherenceGate` hysteresis pattern for the clock term too: open at 200 µs, close only above 240 µs; EMA the offset series (the firmware already EMA-smooths, per `smoothed_used` flag) |
| Inter-band consensus false-demotes a band that is genuinely seeing a different multipath (legitimately decorrelated across 2.4 vs 6 GHz) | Medium | A useful band drops out of consensus | `promote_consensus` default 0.6 is deliberately lenient; band frequency-dependent decorrelation is expected, so demotion requires sustained loss, and a demoted band still streams (it is not Exited) |
| Cramér-Rao credence assumes a centroid target; a real target off-centroid has a different bound | Low | Credence interval mildly optimistic/pessimistic off-centre | Documented as a centroid-referenced bound; ADR-137 may recompute per-hypothesis if it needs target-specific credence |
| ADR-136 `FrameMeta` shape changes during its own design, breaking the `clock_quality` field | Medium | Re-plumb the coordinator's input extraction | Coordinator consumes a `ClockQualityScore` value object, not `FrameMeta` directly; only the thin extraction adapter changes |

---

## 4. Alternatives Considered

### 4.1 Extend `MultiBandCsiFrame` In Place Instead of a New `LinkGroup`

Rejected. `MultiBandCsiFrame` is a value-type snapshot consumed throughout `multistatic.rs` and the sensing server; bolting mutable band-lifecycle state onto it would break its `Clone`-cheap, pass-by-value contract and entangle every consumer with lifecycle logic. A separate aggregate that *produces* `MultiBandCsiFrame` via `consensus_frame()` keeps the snapshot type immutable and the lifecycle isolated.

### 4.2 Make `ArrayCoordinator` Part of `MultistaticArray`

Rejected. `MultistaticArray` is an aggregate root with a single-aggregate invariant boundary (its viewpoints, its coherence window). Cross-aggregate gating that reads every node's `LinkGroup` belongs in a domain service, not inside an aggregate — folding it in would force the aggregate to hold references to other aggregates, violating DDD boundaries and making it untestable in isolation. The service is stateless-per-call and trivially unit-testable.

### 4.3 Keep the Binary Coherence Gate, Add Clock as a Second Binary Gate

Rejected. Two ANDed binary gates still throw away graded information: a node that is 90% coherent with a 210 µs clock would be hard-rejected, discarding real evidence. The `MonitorOnly` middle state is the whole point — it admits the evidence while withholding the environment update. A pure binary design cannot express "trust this for motion evidence but not for re-learning the room."

### 4.4 Derive Clock Quality on the ESP32 and Ship a Single Byte

Rejected for now. The ESP32 firmware already computes the EMA offset (the `smoothed_used` flag), and shipping a pre-computed quality byte would save host work. But the host has the *full* offset series across all nodes and can compute a *comparative* stratum and dispersion the single node cannot. Per-node self-assessment also cannot detect a node that is confidently wrong. Host-side derivation from the existing `SyncPacket` stream keeps the firmware unchanged (no reflash) and centralises the cross-node comparison. This may revisit once ADR-110 firmware exposes a richer sync telemetry field.

### 4.5 Use Raw `guard_interval_us` Rejection for Clock Handling

Rejected. The existing `MultistaticConfig.guard_interval_us` (5 ms spread) is a *timestamp-alignment* sanity check, not a clock-*quality* measure — it catches gross desync but says nothing about the sub-millisecond dispersion that corrupts cross-node phase. The two are complementary: `guard_interval_us` stays as the coarse alignment precondition; `ClockQualityScore.offset_stdev_us` is the fine-grained quality term feeding the gate.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-008 (CSI Frame Primitives) | **Substrate**: `CsiFrame`/`CanonicalCsiFrame` are the per-band frame types `LinkGroup` aggregates |
| ADR-029 (RuvSense Multistatic) | **Extended**: `LinkGroup::consensus_frame()` feeds the existing `MultistaticFuser`; the coordinator supplies the attention weights `fuse()` previously derived internally |
| ADR-030 (Persistent Field Model) | **Gated**: environment/model updates are exactly what `MonitorOnly` withholds when clock quality degrades |
| ADR-031 (RuView Sensing-First RF Mode) | **Extended**: this ADR builds directly on `viewpoint/geometry.rs`, `coherence.rs`, `attention.rs`, `fusion.rs` introduced by ADR-031 |
| ADR-110 (ESP32-C6 Firmware Extension) | **Substrate**: `SyncPacket` (magic `0xC511A110`) and its `local_minus_epoch_us`/`mesh_aligned_us_for_sequence` are the source of `ClockQualityScore`; the ±100 µs target defines the 200 µs gate floor |
| ADR-136 (RuView Rust Streaming Engine) | **Contract**: `FrameMeta` carries `mesh_aligned_us` + `clock_quality`; the coordinator reads these rather than raw packets |
| ADR-137 (Fusion Engine Quality Scoring) | **Downstream consumer**: `DirectionalEvidence.contradictions` (`ContradictionFlag`) is the agreed hand-off; ADR-137 attaches model/privacy version to complete the provenance tuple |

---

## 6. References

### Production Code

- `v2/crates/wifi-densepose-signal/src/ruvsense/multiband.rs` — `MultiBandCsiFrame`, `MultiBandBuilder`, `compute_cross_channel_coherence`, `pearson_correlation_f32` (consensus reuse); `LinkGroup` lands here
- `v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs` — `MultistaticFuser`, `FusedSensingFrame`, `attention_weighted_fusion`, `geometric_diversity`, `MultistaticConfig.guard_interval_us`
- `v2/crates/wifi-densepose-ruvector/src/viewpoint/geometry.rs` — `GeometricDiversityIndex::compute`/`is_sufficient`, `CramerRaoBound::estimate`, `ViewpointPosition`, `NodeId`
- `v2/crates/wifi-densepose-ruvector/src/viewpoint/coherence.rs` — `CoherenceState`, `CoherenceGate` (phase term); `ClockQualityScore`/`ClockQualityGate` land here
- `v2/crates/wifi-densepose-ruvector/src/viewpoint/attention.rs` — `CrossViewpointAttention`, `GeometricBias` (softmax shape reference)
- `v2/crates/wifi-densepose-ruvector/src/viewpoint/fusion.rs` — `MultistaticArray` aggregate, `ViewpointFusionEvent`, `FusionError::CoherenceGateClosed`; `ArrayCoordinator` lands here
- `v2/crates/wifi-densepose-hardware/src/sync_packet.rs` — `SyncPacket`, `SYNC_PACKET_MAGIC = 0xC511A110`, `local_minus_epoch_us`, `apply_to_local`, `mesh_aligned_us_for_sequence`
- `v2/crates/wifi-densepose-sensing-server/src/main.rs` — `NodeState::latest_sync`, `mesh_aligned_us_for_csi_frame`, 9 s staleness gate (source of `ClockQualityScore.age_us` ceiling)
- `docs/adr/ADR-110-esp32-c6-firmware-extension.md` — §A0.10 measured 104 µs offset stdev, §A0.12 sync-packet wire format
- `archive/v1/data/proof/expected_features.sha256` — hash entry `array_coordinator_evidence_v1` to be added; `verify.py` `array_coordinator_check()` extension

### External References

- Mardia, K.V. & Jupp, P.E. (2000). *Directional Statistics*. Wiley. — Circular phasor coherence underlying `CoherenceState` and the >2σ contradiction test.
- Van Trees, H.L. (2002). *Optimum Array Processing*. Wiley. Ch. 8. — Cramér-Rao bound and Fisher information matrix used by `CramerRaoBound` for the credence interval.
- IEEE 802.11be (WiFi-7) Multi-Link Operation. — Concurrent multi-band streaming model that the `LinkGroup` FreqSet abstraction targets.
- IEEE 802.15.4 time synchronization. — Stratum / mesh-epoch model underlying ADR-110's `SyncPacket` and the `ClockQualityScore.stratum` field.


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `fc7674bde`, issue #842): `ClockQualityGate` (in `wifi-densepose-ruvector`) and `ArrayCoordinator` + `DirectionalEvidence` (in `wifi-densepose-signal`, placed there to avoid a dependency cycle). 8 tests.

**Integration glue -- not yet on the live path:** the `LinkGroup` per-band consensus aggregate; the ADR-110 `SyncPacket` UDP decode -> `FrameMeta.mesh_aligned_us`; and live coherence/clock-quality feeds per node.

**Trust contribution:** only well-synced, well-placed nodes are allowed to change the world-model; a clock-degraded node still contributes evidence but is held in *watch-only* mode.
