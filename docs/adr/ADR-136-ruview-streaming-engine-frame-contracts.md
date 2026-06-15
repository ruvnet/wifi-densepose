# ADR-136: RuView Rust Streaming Engine: Architecture, Frame Contracts, and Stage Abstraction

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see §8 Implementation Status, commit `11f89727f`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-core` (`types.rs`: `CsiFrame`/`CsiMetadata`); `wifi-densepose-signal/src/ruvsense/mod.rs` (`RuvSensePipeline`, six-stage flow); `v2/Cargo.toml` (workspace topology) |
| **Relates to** | ADR-028 (ESP32 Capability Audit — witness/deterministic proof), ADR-031 (RuView Sensing-First RF Mode), ADR-119 (BFLD Frame Format and Wire Protocol — LE determinism + reserved-flag forward-compat), ADR-127 (HomeCore State Machine), ADR-134 (First-Class CIR Support), ADR-135 (Empty-Room Baseline Calibration), ADR-137 (Fusion Quality Scoring), ADR-138 (LinkGroup / ArrayCoordinator), ADR-140 (Semantic State Record), ADR-145 (Ablation Eval Harness) |

---

## 1. Context

This is the **foundational umbrella ADR** for the RuView streaming engine. It does not introduce a new algorithm or sensing capability. Instead it makes three load-bearing decisions that every downstream ADR in the 136–146 series depends on: (a) what the streaming engine *is* in terms of the existing crate workspace, (b) the unified typed frame contracts that flow between stages, and (c) the trait surface and determinism guarantee that lets stages compose and be replayed deterministically.

A future contributor reading the spec for "the RuView streaming engine" expects to find a crate named `ruview_engine` or a set of `ruview_*` crates. They will not find one. This ADR is the source-of-truth mapping that explains why, and what the spec's role names actually point at.

### 1.1 The Gap

Three concrete gaps exist in the codebase as of 2026-05-28.

**Gap 1 — No documented role→crate mapping.** The streaming-engine spec organises the system into ten roles: ingest, signal, fusion, world, models, privacy, store, api, eval, observe. The workspace under `v2/crates/` already contains 35 crates that fulfil these roles, but no document maps the spec vocabulary onto the real crates. `ls v2/crates/` returns `wifi-densepose-core`, `wifi-densepose-signal`, `wifi-densepose-bfld`, `homecore`, `homecore-api`, `homecore-automation`, `homecore-assist`, `homecore-recorder`, `cog-pose-estimation`, `cog-person-count`, `cog-ha-matter`, and others — names that predate the streaming-engine spec by months of commit history. A contributor cannot tell that `wifi-densepose-bfld` *is* the privacy/beamforming role or that `homecore` *is* the world/state role without reading source. This ADR fixes the mapping in writing.

**Gap 2 — No unified complex-sample or frame-metadata contract across stages.** The pipeline carries complex CSI in at least two distinct representations:

- `wifi-densepose-core/src/types.rs:370` — `CsiFrame.data: Array2<Complex64>` (f64 complex, `[spatial_streams, subcarriers]`), with `#[cfg_attr(feature = "serde", serde(skip))]` on `data`, `amplitude`, and `phase` (lines 369, 372, 375). **The complex payload is not serialised at all today** — only `CsiMetadata` survives a serde round-trip.
- `wifi-densepose-signal/src/ruvsense/cir.rs:27` — uses `num_complex::Complex32` (f32 complex) for CIR taps and the sub-DFT sensing matrix Φ.

There is no `ComplexSample` newtype unifying these, and no byte-order guarantee on the complex payload because it is `serde(skip)`-ped. ADR-119 already solved the same problem for `BfldFrame` (little-endian, `#[repr(C, packed)]`, BLAKE3 witness — see `wifi-densepose-bfld/src/frame.rs` and `signature_hasher.rs`), but that determinism contract is scoped to one frame type, not the whole pipeline.

`CsiMetadata` (`types.rs:311`) carries `timestamp`, `device_id`, `frequency_band`, `channel`, `bandwidth_mhz`, `antenna_config`, `rssi_dbm`, `noise_floor_dbm`, `sequence_number`. It carries **no `calibration_id`** (so a frame cannot be traced to the ADR-135 baseline that was subtracted from it) and **no `model_id` / `model_version`** (so a downstream `PoseEstimate` cannot be traced back to the inference context — `PoseEstimate.model_version: String` at `types.rs:964` is a free-form string set at the *end* of the pipeline, not propagated through frames).

**Gap 3 — No `Stage<I,O>` abstraction; pipeline stages are concrete and non-uniform.** `wifi-densepose-signal/src/ruvsense/mod.rs:9-23` documents six stages (multiband → phase_align → multistatic → coherence → coherence_gate → pose_tracker), but `RuvSensePipeline` (`mod.rs:184`) holds them as concrete fields (`phase_aligner: PhaseAligner`, `coherence_state: CoherenceState`, `gate_policy: GatePolicy`) and exposes only a `tick()` method (`mod.rs:232`) that increments a counter. There is no common `process(&self, I) -> Result<O>` trait, no `Versioned` trait, and no `QualityScored` trait. Each stage has a bespoke signature, so ADR-137 (quality scoring), ADR-138 (LinkGroup), and ADR-145 (ablation harness) cannot compose or swap stages without per-stage glue.

### 1.2 What This ADR Is and Is Not

It **is** a contract document: it pins down `ComplexSample`, `FrameMeta`, the three traits, the determinism guarantee, and the role→crate map. It establishes the vocabulary the 137–146 ADRs build on.

It is **not** a rewrite. It explicitly rejects renaming the workspace to `ruview_*` (§2.1). It adds fields to `CsiMetadata` and traits to the pipeline; it does not relayout `CsiFrame.data` or change the `ndarray` storage.

### 1.3 Pipeline Position

```
[ingest]        [signal]                              [fusion] [world]  [models] [privacy] [api]
ESP32/Pi  →  RuvSensePipeline six stages           →   fuse  →  state →  infer  →  gate  → publish
  │            │                                        │        │        │         │       │
  │      multiband → phase_align → calibration(135)      │   homecore   cog-*   bfld   homecore-api
  │            → cir(134) → multistatic → coherence      │
  └─ CsiFrame{ data, FrameMeta{calibration_id, model_id} } flows through every stage as Stage<I,O>
```

Every box above is an existing crate. The novelty of this ADR is the *contract on the arrow*: a single `CsiFrame` whose `FrameMeta` ties each sample to its calibration (ADR-135), its model context (ADR-146), and — downstream — its privacy decision (ADR-119/141), satisfying the project rule that every semantic state traces to signal evidence + model version + calibration version + privacy decision.

---

## 2. Decision

### 2.1 Adopt the Existing Workspace As the Streaming Engine — Reject `ruview_*` Rename

The streaming engine **is** the existing 35-crate `v2/` workspace. The spec's ten roles map 1:1 onto current crates:

| Spec role | Crate(s) | Evidence |
|-----------|----------|----------|
| **ingest** | `wifi-densepose-sensing-server`, `wifi-densepose-hardware`, `wifi-densepose-wifiscan` | Axum sensing server + ESP32 aggregator/TDM |
| **signal** | `wifi-densepose-signal` (incl. `ruvsense/`) | `RuvSensePipeline` six stages; `cir.rs`, `calibration.rs` |
| **fusion** | `wifi-densepose-signal/src/ruvsense/multistatic.rs`, `wifi-densepose-ruvector/src/viewpoint/` | `FusedSensingFrame`, cross-viewpoint attention (ADR-137) |
| **world** | `homecore` (`state.rs`, `entity.rs`, `registry.rs`, `bus.rs`), `wifi-densepose-geo` | HomeCore state machine (ADR-127); WorldGraph target (ADR-139) |
| **models** | `cog-pose-estimation`, `cog-person-count`, `wifi-densepose-nn`, `wifi-densepose-train` | inference + training |
| **privacy** | `wifi-densepose-bfld` (`privacy_gate.rs`, `sink.rs`, `signature_hasher.rs`) | byte-level privacy classes (ADR-119/141) |
| **store** | `homecore-recorder` | trajectory/event recording |
| **api** | `homecore-api`, `homecore-server`, `cog-ha-matter`, `homecore-hap` | REST/HA/Matter/HomeKit surfaces |
| **eval** | (new: ablation harness lands in `wifi-densepose-train` test crate per ADR-145) | ADR-145 |
| **observe** | `homecore-automation`, `homecore-assist` | automation + assistant bridge (ADR-140) |

**Decision: do not introduce a `ruview_*` prefix or new umbrella crate.** The rationale:

- **Commit history preservation.** `wifi-densepose-signal` carries the full provenance of ADR-014, -029, -030, -134, -135. A rename detaches blame/log lineage from 1,000+ tests and the ADR-028 witness chain that hashes `ruvsense/*.rs` source.
- **Migration cost with no functional gain.** A rename touches every `use wifi_densepose_*::` path across 35 crates, the `v2/Cargo.toml` `members` list, the publishing order in `CLAUDE.md`, and the witness `source-hashes.txt`. None of this changes runtime behaviour.
- **"RuView" is a product surface, not a crate.** RuView (ADR-031) is the sensing-first *mode* and UI/appliance brand (cognitum-v0 dashboard). The engine beneath it is the wifi-densepose/homecore workspace. Keeping the names distinct avoids implying a code reorganisation that is not happening.

This table is normative: ADR-137 through ADR-146 reference roles by this mapping, not by inventing crate names.

### 2.2 `FrameMeta`: Add `calibration_id` and `model_id` / `model_version`

`CsiMetadata` gains three fields so every frame links to its calibration and inference context. To avoid breaking the 1,000+ tests that call `CsiMetadata::new(...)`, the new fields default to "none" and are populated by the calibration and inference stages.

```rust
// wifi-densepose-core/src/types.rs — additions to CsiMetadata

use uuid::Uuid;

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CsiMetadata {
    // ... existing fields (timestamp, device_id, frequency_band, channel,
    //     bandwidth_mhz, antenna_config, rssi_dbm, noise_floor_dbm,
    //     sequence_number) unchanged ...

    /// UUID of the ADR-135 empty-room baseline subtracted from this frame.
    /// `None` ⇒ uncalibrated (no `BaselineCalibration::subtract()` applied).
    #[cfg_attr(feature = "serde", serde(default))]
    pub calibration_id: Option<Uuid>,

    /// Identifier of the RF encoder / model family that will consume this
    /// frame (ADR-146). Stable across a deployment; 0 ⇒ unassigned.
    #[cfg_attr(feature = "serde", serde(default))]
    pub model_id: u16,

    /// Monotonic model version (ADR-119 §2.1 reserved-flag pattern: the low
    /// byte is minor, high byte is major). 0 ⇒ unassigned.
    #[cfg_attr(feature = "serde", serde(default))]
    pub model_version: u16,
}
```

`FrameMeta` is the public alias the streaming-engine docs use; in code it *is* `CsiMetadata` (`pub use wifi_densepose_core::types::CsiMetadata as FrameMeta;` re-exported from `wifi-densepose-signal`). We keep one struct rather than two to avoid copy-on-cross-stage.

`calibration_id` is a `Uuid` (the workspace already depends on `uuid` — `types.rs:17`) and references the `BaselineCalibration` finalised by ADR-135. ADR-135's `BaselineCalibration` gains a `pub id: Uuid` field whose value is written here. This closes the trace from a fused semantic state back to the exact empty-room reference that conditioned it.

`model_id`/`model_version` are `u16` (not `String` like `PoseEstimate.model_version` at `types.rs:964`) because they ride on every frame and must be cheap to copy and to serialise in fixed width. The free-form `PoseEstimate.model_version: String` remains for human-readable reporting; the `u16` pair is the machine-traceable key.

### 2.3 `ComplexSample`: One Complex Wrapper with LE Serialisation

CSI uses `Complex64` (`types.rs:16`), CIR uses `Complex32` (`cir.rs:27`). Neither is serialised deterministically today (`CsiFrame.data` is `serde(skip)`). Introduce a single wrapper with a guaranteed little-endian byte order, following the ADR-119 pattern.

```rust
// wifi-densepose-core/src/types.rs (new) — re-exported by signal crate

use num_complex::Complex64;

/// Canonical complex sample for all RuView frame contracts (CSI, CIR, Doppler).
///
/// Wraps `num_complex::Complex64`. The `serde` impl writes `(re, im)` as two
/// little-endian f64, matching the ADR-119 endianness-stability guarantee so
/// x86_64 (ruvultra), aarch64 (cognitum-v0), and Xtensa (ESP32-S3) produce
/// bit-identical bytes. Downstream f32 paths (CIR taps) narrow on demand via
/// `as_complex32()`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct ComplexSample(pub Complex64);

impl ComplexSample {
    #[must_use] pub fn new(re: f64, im: f64) -> Self { Self(Complex64::new(re, im)) }
    #[must_use] pub fn norm(&self) -> f64 { self.0.norm() }
    #[must_use] pub fn arg(&self)  -> f64 { self.0.arg() }
    /// Narrow to f32 complex for CIR / NN paths (ADR-134, ADR-146).
    #[must_use] pub fn as_complex32(&self) -> num_complex::Complex32 {
        num_complex::Complex32::new(self.0.re as f32, self.0.im as f32)
    }
    /// Canonical 16-byte LE encoding: re||im, each f64 LE.
    #[must_use] pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0..8].copy_from_slice(&self.0.re.to_le_bytes());
        b[8..16].copy_from_slice(&self.0.im.to_le_bytes());
        b
    }
    #[must_use] pub fn from_le_bytes(b: [u8; 16]) -> Self {
        let re = f64::from_le_bytes(b[0..8].try_into().unwrap());
        let im = f64::from_le_bytes(b[8..16].try_into().unwrap());
        Self(Complex64::new(re, im))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for ComplexSample {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Two LE f64 — deterministic across architectures.
        use serde::ser::SerializeTuple;
        let mut t = s.serialize_tuple(2)?;
        t.serialize_element(&self.0.re)?;
        t.serialize_element(&self.0.im)?;
        t.end()
    }
}
```

`CsiFrame.data` stays `Array2<Complex64>` for ndarray-native math; `ComplexSample` is the *contract* representation used at stage boundaries and for the deterministic serialiser (§2.5). A new `CsiFrame::data_complex_samples()` view yields `ComplexSample` without copying the underlying buffer. CIR/Doppler frames (`CirFrame`, `DopplerFrame`) store `Vec<ComplexSample>` directly so all three frame types share one complex contract.

### 2.4 Stage, Versioned, QualityScored Traits

The six `RuvSensePipeline` stages (`mod.rs:9-23`) become uniform implementers of `Stage<I,O>`. Two marker/capability traits — `Versioned` and `QualityScored` — sit alongside it.

```rust
// wifi-densepose-signal/src/ruvsense/mod.rs (new traits)

/// A pipeline stage that transforms one typed frame into another.
///
/// Stages are `Send + Sync` and stateless w.r.t. determinism: given the same
/// input bytes and the same `&self` configuration, `process` MUST produce the
/// same output bytes (see §2.5). Mutable runtime state (rolling windows,
/// Welford accumulators) lives behind `&self` interior types whose effect on
/// output is captured in the deterministic-replay fixture.
pub trait Stage<I, O>: Send + Sync {
    /// Human/stage identifier, e.g. "phase_align", "calibration".
    fn name(&self) -> &'static str;
    /// Transform one input frame into one output frame.
    fn process(&self, input: I) -> StageResult<O>;
}

pub type StageResult<O> = std::result::Result<O, RuvSenseError>;

/// Forward-compatible version stamp. Mirrors ADR-119 §2.1: a `(major, minor)`
/// pair plus a reserved-flags word so future revisions extend without breaking
/// the deterministic byte layout.
pub trait Versioned {
    fn version(&self) -> (u8, u8);                 // (major, minor)
    fn reserved_flags(&self) -> u16 { 0 }          // ADR-119 reserved bits 2..15
    /// True if `other` can consume output produced at `self.version()`.
    fn is_compatible_with(&self, other: (u8, u8)) -> bool {
        self.version().0 == other.0 && self.version().1 >= other.1
    }
}

/// A stage output that carries a scalar quality score and a confidence
/// interval. Consumed by ADR-137 (fusion quality) and ADR-145 (ablation).
pub trait QualityScored {
    /// Scalar quality in [0.0, 1.0]; higher is better.
    fn quality_score(&self) -> f32;
    /// (lower, upper) confidence bounds in [0.0, 1.0], lower ≤ upper.
    fn confidence_bounds(&self) -> (f32, f32);
}
```

With `Stage<I,O>`, the six concrete stages compose as a heterogeneous chain (each adapter `Stage<FrameN, FrameN+1>`), and ADR-138's `ArrayCoordinator` can gate a `Stage` by clock quality, ADR-137's fusion can read `QualityScored`, and ADR-145's harness can substitute or ablate any stage by trait object. `RuvSensePipeline` keeps its concrete fields but each becomes a `Stage` impl; `tick()` is retained for the frame counter, and a new `run(frame) -> StageResult<FusedSensingFrame>` drives the chain.

**Boundary rule:** a `Stage<I,O>` never mutates its input's `FrameMeta.calibration_id` or `model_id` except the calibration stage (sets `calibration_id`) and the model-binding stage (sets `model_id`/`model_version`). This makes provenance append-only along the chain.

### 2.5 Deterministic Serialisation Contract for All Frame Types

Extend the ADR-119 `BfldFrame` determinism + BLAKE3 witness pattern to every frame type in the engine.

```rust
/// Every frame type that crosses a stage boundary or is recorded/replayed
/// implements `CanonicalFrame`. The bytes are stable across architectures
/// (LE per §2.3) and across runs (fixed field order), so a BLAKE3 of the
/// stream is a witness hash (ADR-028).
pub trait CanonicalFrame {
    /// Deterministic, architecture-independent encoding.
    fn to_canonical_bytes(&self) -> Vec<u8>;
    /// BLAKE3-32 of `to_canonical_bytes()` (ADR-119 signature_hasher pattern).
    fn witness_hash(&self) -> [u8; 32] {
        blake3::hash(&self.to_canonical_bytes()).into()
    }
}
```

`CsiFrame`, `CirFrame`, `DopplerFrame`, and `FusedSensingFrame` all implement `CanonicalFrame`. The canonical encoding rule:

1. `FrameMeta` fields in declared order, each fixed-width LE (timestamps as `i64`/`u32`, ids/versions as their integer widths, `calibration_id` as the 16 UUID bytes or 16 zero bytes for `None`).
2. Complex payload as `ComplexSample::to_le_bytes()` in stream-major (`[stream][subcarrier]`) order — the same layout ADR-135 §2.4 uses for the NVS baseline.
3. No `f32`/`f64` text formatting; raw IEEE-754 LE only.

`blake3` is already a workspace dependency (`wifi-densepose-bfld/src/signature_hasher.rs:20` `use blake3::Hasher;`). The **deterministic-replay contract** is: feeding a recorded `Vec<CsiFrame>` (from `homecore-recorder`) through the `Stage` chain twice yields byte-identical `FusedSensingFrame` streams, verified by equal `witness_hash()`. This is the property ADR-145's ablation harness and the ADR-028 witness bundle both rely on.

### 2.6 Provenance Invariant

Combining §2.2, §2.4, and §2.5 yields the engine-wide invariant that every downstream ADR may assume:

> Any `FusedSensingFrame` (and the semantic state derived from it in ADR-140) carries, transitively via its source `FrameMeta`: the **signal evidence** (`witness_hash()` of the source `CsiFrame`s), the **model version** (`model_id`/`model_version`), the **calibration version** (`calibration_id` → ADR-135 baseline), and — once it passes the `wifi-densepose-bfld` privacy gate — the **privacy decision** (`privacy_class`, ADR-119 §2.3). No stage may drop these fields; the boundary rule in §2.4 makes them append-only.

---

## 3. Consequences

### 3.1 Positive

- **One vocabulary for ten ADRs.** ADR-137–146 reference the role→crate table (§2.1) and the three traits instead of re-deriving them, eliminating cross-ADR drift.
- **No migration.** Rejecting `ruview_*` keeps every `use` path, the publishing order, and the ADR-028 witness `source-hashes.txt` intact.
- **End-to-end traceability.** `calibration_id` + `model_id`/`model_version` on `FrameMeta` close the provenance chain the project rule mandates; a fused state can be audited back to its baseline and model.
- **Composability.** `Stage<I,O>` lets ADR-138 gate stages, ADR-137 read `QualityScored`, and ADR-145 ablate any stage by trait object — no per-stage glue.
- **Witness extension is mechanical.** `CanonicalFrame::witness_hash()` plugs straight into the existing BLAKE3 path (`signature_hasher.rs`) and the `verify.py` expected-hash format (ADR-028, ADR-119 §3).

### 3.2 Negative

- **`CsiMetadata` grows by three fields.** Every `CsiMetadata::new()` call site (1,000+ tests) keeps compiling because the fields default, but serialised metadata changes shape — `serde(default)` handles forward reads, but any pinned metadata fixture hash in the witness bundle must be regenerated once.
- **Two complex types coexist during migration.** `ComplexSample` (Complex64) is the contract type; `cir.rs` keeps `Complex32` internally and narrows via `as_complex32()`. Until all call sites adopt the view method, both representations are live.
- **Determinism becomes a maintenance obligation.** Once `CanonicalFrame` is the witness substrate, any stage that introduces nondeterminism (HashMap iteration order, unseeded RNG, float reduction order) breaks the replay test — a stricter bar than the current `serde(skip)` payload imposes.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Contributors keep inventing `ruview_*` names because the spec uses them | Medium | Doc/code divergence; phantom crates in design talk | §2.1 table is normative and linked from `CLAUDE.md` crate table; PR review rejects new `ruview_*` crates |
| `Complex64` LE serialisation differs from the f32 CIR path, causing two witness lineages | Low | Replay hash mismatch between CSI and CIR stages | Single `ComplexSample::to_le_bytes()` is the only encoder; `as_complex32()` is a lossy *view*, never re-serialised as the witness form |
| Float reduction order in fusion (multistatic attention) is nondeterministic across thread counts | Medium | `to_canonical_bytes()` stable but `process()` output varies | Fusion stage fixes reduction order (stream-major, single-threaded reduction in the witness path); ADR-137 owns this |
| `model_id`/`model_version` u16 overflow as model families grow | Low | Wraparound collides ids | u16 gives 65k families/versions; ADR-146 owns the registry and reserves 0 = unassigned |

---

## 4. Alternatives Considered

### 4.1 Rename the Workspace to `ruview_*` (Rejected)

Create `ruview-engine`, `ruview-signal`, `ruview-fusion`, etc., matching the spec literally. **Rejected** for the reasons in §2.1: it detaches commit history, breaks the witness `source-hashes.txt` chain, churns 35 crates' `use` paths and the publishing order, and delivers zero runtime change. The spec roles are a *lens*, not a directory layout.

### 4.2 Separate `FrameMeta` Struct Distinct from `CsiMetadata` (Rejected)

Define a new `FrameMeta` and convert `CsiMetadata ↔ FrameMeta` at stage boundaries. **Rejected**: it doubles the metadata type surface and forces a copy on every cross-stage hop at 20 Hz × N links. Re-exporting `CsiMetadata as FrameMeta` gives the spec vocabulary with zero conversion cost.

### 4.3 Keep `Complex64`/`Complex32` Split, No `ComplexSample` (Rejected)

Leave the two complex types as-is and serialise ad hoc per frame type. **Rejected**: it reproduces Gap 2 — no single byte-order guarantee, so witness hashes for CSI vs CIR frames have independent, unverifiable encodings. One wrapper with one `to_le_bytes()` is the minimal fix.

### 4.4 Generic Pipeline via `async` Streams Instead of `Stage<I,O>` (Rejected)

Model the pipeline as a `futures::Stream` chain. **Rejected for the contract layer**: async stream combinators hide the per-stage `name()`/`version()`/`quality_score()` surface that ADR-137/138/145 need to introspect, and they complicate the deterministic-replay test (executor scheduling). A plain `Stage<I,O>` trait is synchronous, introspectable, and trivially replayable; async transport can wrap it at the ingest/api edges where it belongs.

### 4.5 Defer Provenance Fields to a Side-Channel (Rejected)

Carry `calibration_id`/`model_id` in a parallel map keyed by `FrameId` rather than on `FrameMeta`. **Rejected**: a side map can desync from the frame, and recording/replay (`homecore-recorder`) would have to persist two artifacts that must stay consistent. Inlining on `FrameMeta` makes provenance travel with the data and survive serialisation.

---

## 5. Testing and Acceptance

All tests live in `wifi-densepose-core` (contract types) and `wifi-densepose-signal/src/ruvsense/` (traits, replay). Hardware tests are gated behind `#[cfg(feature = "hardware-test")]` and excluded from CI.

**AC1 — `ComplexSample` LE round-trip (unit).** For 10,000 seeded random `(re, im)` f64 pairs, assert `ComplexSample::from_le_bytes(s.to_le_bytes()) == s` and that byte 0 equals the LSB of `re` (endianness pin). Run the same assertion under `cfg(target_endian = "big")` cross-check via manual byte construction.

**AC2 — `FrameMeta` provenance defaults (unit).** `CsiMetadata::new(...)` yields `calibration_id == None`, `model_id == 0`, `model_version == 0`. After a simulated ADR-135 `subtract()` and ADR-146 model bind, the fields are populated; assert the boundary rule (§2.4) — no other stage mutates them.

**AC3 — `serde(default)` forward-read (unit).** Deserialise a pre-ADR-136 `CsiMetadata` JSON fixture (without the three fields) and assert it loads with the documented defaults — proves the addition is backward-compatible.

**AC4 — `Stage` chain composition (unit).** Build a 6-stage mock chain (`Stage<FrameN, FrameN+1>`), feed one synthetic `CsiFrame`, assert the output `FusedSensingFrame` and that each stage's `name()` is visited in declared order.

**AC5 — `Versioned` compatibility (unit).** Assert `is_compatible_with` accepts equal-major/greater-or-equal-minor and rejects major mismatch, mirroring ADR-119 §2.1 reserved-flag forward-compat.

**AC6 — Deterministic replay / witness (CI-compatible).** Generate a fixed 600-frame synthetic `CsiFrame` stream (seed = 42, same generator as ADR-135 Tier 1). Run it through the `Stage` chain twice and assert byte-identical `FusedSensingFrame::to_canonical_bytes()` and equal `witness_hash()`. Record the final BLAKE3 in `archive/v1/data/proof/expected_features.sha256` under key `streaming_engine_replay_v1`; `verify.py` regenerates and re-asserts (extends the ADR-028 proof chain).

**AC7 — Cross-architecture byte stability (CI matrix).** Run AC6 on x86_64 and aarch64 CI runners (ruvultra, cognitum-v0 classes); assert identical `witness_hash()` across architectures — the ADR-119 §1 endianness guarantee at the whole-pipeline level.

**AC8 — `QualityScored` bounds invariant (unit).** For any stage output implementing `QualityScored`, assert `0.0 ≤ lower ≤ quality_score ≤ upper ≤ 1.0` is *not* required (score may sit outside bounds), but `0.0 ≤ lower ≤ upper ≤ 1.0` and `quality_score ∈ [0,1]` hold. Consumed by ADR-137.

**AC9 — Role→crate map is live (doc/CI lint).** A test asserts each crate named in the §2.1 table exists in `v2/Cargo.toml` `members`, preventing the mapping from rotting as crates are added/removed.

---

## 6. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-028 (ESP32 Capability Audit) | **Witness extended**: `CanonicalFrame::witness_hash()` adds `streaming_engine_replay_v1` to `expected_features.sha256`; `verify.py` regenerates it |
| ADR-031 (RuView Sensing-First Mode) | **Named**: clarifies RuView is the product mode/brand atop this engine, not a crate to rename to |
| ADR-119 (BFLD Frame Format) | **Generalised**: this ADR lifts ADR-119's LE determinism, reserved-flag forward-compat (§2.1), and BLAKE3 witness from one frame type to all frame types |
| ADR-127 (HomeCore State Machine) | **Consumer**: `homecore` is the `world` role; semantic state it holds traces to `FrameMeta` provenance |
| ADR-134 (First-Class CIR) | **Unified**: `CirFrame` adopts `ComplexSample`; `as_complex32()` feeds the ISTA path; CIR is a `Stage` in the chain |
| ADR-135 (Empty-Room Baseline) | **Linked**: `BaselineCalibration` gains `id: Uuid`, written into `FrameMeta.calibration_id` by the calibration stage |
| ADR-137 (Fusion Quality Scoring) | **Depends on**: `QualityScored` trait and `FusedSensingFrame` contract defined here |
| ADR-138 (LinkGroup / ArrayCoordinator) | **Depends on**: gates `Stage`s by clock quality using the trait surface here |
| ADR-140 (Semantic State Record) | **Depends on**: semantic states reference the §2.6 provenance invariant |
| ADR-145 (Ablation Eval Harness) | **Depends on**: ablates/substitutes `Stage` trait objects and relies on deterministic replay (AC6) |
| ADR-146 (RF Encoder Multi-Task Heads) | **Depends on**: owns the `model_id`/`model_version` registry written into `FrameMeta` |

---

## 7. References

### Production Code

- `v2/crates/wifi-densepose-core/src/types.rs` — `CsiFrame` (line 363), `CsiMetadata` (line 311), `Complex64` import (line 16), `uuid` import (line 17); `data`/`amplitude`/`phase` are `serde(skip)` (lines 369–376); `PoseEstimate.model_version: String` (line 964)
- `v2/crates/wifi-densepose-signal/src/ruvsense/mod.rs` — six-stage pipeline doc (lines 9–23), `RuvSensePipeline` (line 184), `tick()` (line 232), `RuvSenseError` (line 121)
- `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs` — `Complex32` use (line 27), sub-DFT Φ
- `v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs` — ADR-135 `BaselineCalibration` (gains `id: Uuid`)
- `v2/crates/wifi-densepose-bfld/src/signature_hasher.rs` — BLAKE3 keyed hash precedent (`use blake3::Hasher;`, line 20)
- `v2/crates/wifi-densepose-bfld/src/frame.rs`, `privacy_gate.rs`, `sink.rs` — ADR-119 frame/privacy precedent
- `v2/crates/homecore/src/{state.rs,entity.rs,registry.rs,bus.rs}` — `world` role (ADR-127)
- `v2/Cargo.toml` — workspace `members`; `num-complex = "0.4"` (line 102)
- `archive/v1/data/proof/verify.py`, `expected_features.sha256` — deterministic proof chain; `streaming_engine_replay_v1` key to be added

### Related ADR Documents

- `docs/adr/ADR-119-bfld-frame-format-and-wire-protocol.md` — §2.1 (reserved flags), §2.4 (deterministic serialisation), §1 (endianness stability)
- `docs/adr/ADR-127-homecore-state-machine-rust.md` — world/state role
- `docs/adr/ADR-134-*.md`, `docs/adr/ADR-135-empty-room-baseline-calibration.md` — signal-stage precedents reused here

### External

- IEEE 802.11bf-2024 WLAN Sensing — the multistatic sensing context the engine implements (referenced in `ruvsense/mod.rs`).
- BLAKE3 (Aumasson et al., 2020) — witness hash function, already vendored for ADR-119/120.


---

## 8. Implementation Status & Integration (2026-05-29)


> **Series context (ADR-136 series).** A *skeleton and nervous system, not a shipping product.* These ADRs deliver the **data contracts**, the **trust / privacy / audit machinery**, and the **algorithms** -- all real, tested, and compiling -- that give the *existing* sensing code a clean place to plug into. Most of the series is **not yet wired into the live 20 Hz pipeline**: each module is an independently tested building block; end-to-end wiring (plus model training in ADR-146) is the next phase, and every ADR's GitHub issue lists what is **Built** vs **Integration glue**. The throughline is **trust** -- *why believe the system when it says a person fell?* -- traceable evidence (137), sensor agreement (137/138), calibration provenance (135/136), and an auditable privacy posture (141).

**Built -- tested building block** (commit `11f89727f`, issue #840): `ComplexSample` (LE-canonical), `CsiMetadata` provenance fields (`calibration_id` / `model_id` / `model_version`), `CanonicalFrame` + BLAKE3 `witness_hash()`, and the `Stage`/`Versioned`/`QualityScored` traits. 9 acceptance tests; workspace builds clean.

**Integration glue -- not yet on the live path:** the full 600-frame `Stage`-chain replay (AC6) -> `streaming_engine_replay_v1` witness key; the cross-architecture CI matrix (AC7); and populating the provenance fields from the live calibration and model-binding stages.

**Trust contribution:** the root of traceability -- the frame contract that lets every fused state name its evidence, model, and calibration.
