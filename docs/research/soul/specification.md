# Soul Signature — Technical Specification

**Status:** Research Specification (Pre-Implementation)
**Date:** 2026-05-24
**Author:** ruv

---

## 1. Overview

A Soul Signature is a typed, content-addressed RVF graph encoding seven
electromagnetic observables extracted from a person in a WiFi-DensePose sensing
zone. The graph is stored as a single `.rvf` binary blob using the existing RVF
container format (`v2/crates/wifi-densepose-sensing-server/src/rvf_container.rs`)
extended with two new segment types defined below. A human-readable JSON sidecar
accompanies the blob for inspection and provenance.

The signature is probabilistic, not deterministic. Matching computes a weighted
cosine similarity across graph dimensions, producing a score in [0, 1] with a
calibrated false-accept rate (FAR). The FAR at a given threshold is an open
research question; the AETHER person re-identification baseline (ADR-024 §2.8:
>80% mAP at 5 subjects) is the lower bound for the primary embedding channel.

---

## 2. Design Principles

### 2.1 Per-Individual

The signature encodes features that are structurally unique to one person at the
sensing resolution of commodity WiFi hardware. Discriminative dimensions include:
cardiac timing (R-R interval structure), respiratory mechanics (tidal depth,
inspiration-to-expiration ratio), skeletal proportions (limb ratios from 17-keypoint
pose, ADR-079), gait cadence variability, and the RF backscatter profile shaped by
body mass distribution and geometry.

### 2.2 Passive at Enrollment Time

No explicit action from the subject is required at recognition time after
enrollment. Recognition fires whenever an enrolled person is detected in a sensing
zone. Enrollment itself requires a 60-second structured protocol (see
`scanning-process.md`). This is a deliberate asymmetry: passive recognition +
active enrollment — which is the same model used by FaceID (passive unlock after
initial face setup).

The passivity of post-enrollment recognition is a privacy concern addressed in full
in `security.md` §4.

### 2.3 Multi-Modal

Seven orthogonal channels contribute. Orthogonality matters: if one channel
degrades (e.g., cardiac is masked by motion), the remaining six carry the match.
No single channel is necessary for a positive identification above threshold;
the fused score is a weighted aggregate.

### 2.4 Persistent Across Time

The stored signature is valid over weeks to months for adults with stable anatomy
and health. Re-scan cadence is prescribed in `scanning-process.md`. The
`longitudinal.rs` module (ADR-030 Tier 4) provides the drift detection that
flags when a re-scan is necessary.

### 2.5 Defensible False-Accept Rate

The security model is not "unbreakable." It is "attacker cost exceeds value of
attack for the threat model in §security." See `security.md` §3.

---

## 3. Signature as a Typed RVF Graph

### 3.1 Container Format

The soul signature reuses the RVF binary container defined in
`v2/crates/wifi-densepose-sensing-server/src/rvf_container.rs` (lines 1–660).
Existing segment types used:

| Segment type | Const | Purpose in soul signature |
|---|---|---|
| `SEG_MANIFEST` | `0x05` | Graph metadata: schema version, enroll timestamp, device ID, person_id (opaque u64) |
| `SEG_VEC` | `0x01` | AETHER 128-dim embedding weights (backbone + projection head) |
| `SEG_META` | `0x07` | JSON overlay: all non-vector node attributes |
| `SEG_WITNESS` | `0x0A` | Ed25519 signature over `(content_hash_sha256 || timestamp_ns || enrolled_by_device_id)` |
| `SEG_EMBED` | `0x0C` | AETHER embedding config + projection head weights (ADR-024 Phase 7) |
| `SEG_LORA` | `0x0D` | Per-environment LoRA deltas for environment-adapted query |

Two new segment types are proposed for the soul signature extension:

| Segment type | Const | Purpose |
|---|---|---|
| `SEG_SOUL_GRAPH` | `0x10` | JSON-serialized graph: node list + edge list + attribute schemas |
| `SEG_SOUL_INDEX` | `0x11` | Per-node HNSW index serialization for fast graph-level query |

The `SegmentHeader` structure is unchanged. Each segment is 64-byte aligned
(field `alignment_pad` at offset `0x3C`). CRC32 content hash at offset `0x28`
covers the payload, providing tamper detection per the existing implementation
at `rvf_container.rs:line 70`.

### 3.2 Node Types

Each node is a typed struct. Serialized into SEG_META as a JSON object with a
`node_type` discriminator string. Vector fields (f32 arrays) are co-located in
a SEG_VEC segment indexed by the node's `vec_segment_id` field.

#### Node: AETHER_Embedding

Primary identity anchor. The contrastive CSI embedding from ADR-024.

```rust
pub struct AetherEmbeddingNode {
    pub node_type: &'static str,        // "AETHER_Embedding"
    pub vec_segment_id: u64,            // references SEG_VEC containing 128 f32s
    pub embedding_dim: usize,           // 128
    pub backbone: String,               // "csi-to-pose-transformer"
    pub pretrain_method: String,        // "simclr+vicreg"
    pub alignment_score: f32,           // Lowman alignment metric at enrollment time
    pub uniformity_score: f32,          // Hypersphere uniformity at enrollment time
    pub enrollment_frames: u32,         // Number of CSI windows averaged into this node
    pub environment_id: String,         // SHA-256 of field model eigenstate at enrollment
    pub confidence: f32,                // HNSW search confidence against person_track index
}
```

Stored size: 128 × 4 = 512 bytes in SEG_VEC; JSON metadata ~200 bytes in SEG_META.
Per ADR-024 §2.8, the person re-identification target is >80% mAP at 5 subjects.
At 10+ subjects the accuracy is open research; baseline TBD.

#### Node: Cardiac_HR_Profile

Extracted from the ADR-039 vitals pipeline (magic `0xC511_0002`, fields offset 6-11:
breathing_rate at `u16 LE` BPM×100, heart_rate at `u32 LE` BPM×10000).
For the soul signature, cardiac extraction uses the ADR-021 bandpass pipeline
(0.8–2.0 Hz) over a minimum 30-second rest window.

```rust
pub struct CardiacHRProfileNode {
    pub node_type: &'static str,        // "Cardiac_HR_Profile"
    pub baseline_bpm: f32,              // mean HR over enrollment window (40–180 BPM range)
    pub hrv_sdnn_ms: f32,               // SDNN: std dev of R-R intervals (ms)
    pub hrv_rmssd_ms: f32,              // RMSSD: root mean square successive differences
    pub hrv_lf_power: f32,              // LF band power (0.04–0.15 Hz), normalized
    pub hrv_hf_power: f32,              // HF band power (0.15–0.4 Hz), normalized
    pub hrv_lf_hf_ratio: f32,           // LF/HF ratio (autonomic balance marker)
    pub sinus_rhythm_class: u8,         // 0=regular, 1=irregular, 2=indeterminate
    pub confidence: f32,                // from ADR-021 VitalCoherenceGate PERMIT fraction
    pub window_seconds: u32,            // duration of the measurement window
}
```

WiFi CSI-based HRV extraction is an active research area. The SDNN and RMSSD values
are discriminative at group level (Zhao et al. 2017, Widar 3.0 2019) but per-person
uniqueness has not been independently validated at scale. Status: open research.

#### Node: Cardiac_Waveform_Morphology

Wavelet decomposition of the bandpass-filtered cardiac phase signal. Captures the
shape of the cardiac waveform, not just its rate. More discriminative than HR alone
but requires higher SNR and longer measurement window.

```rust
pub struct CardiacWaveformMorphologyNode {
    pub node_type: &'static str,        // "Cardiac_Waveform_Morphology"
    pub vec_segment_id: u64,            // references SEG_VEC: 64 f32 wavelet coefficients
    pub wavelet_family: String,         // "db4" (Daubechies 4, standard for cardiac)
    pub decomposition_levels: u8,       // 4 levels
    pub snr_db: f32,                    // measured SNR at enrollment; low-SNR nodes down-weighted
    pub confidence: f32,
}
```

Wavelet coefficient dimension: 64 floats = 256 bytes in SEG_VEC. Waveform
morphology from CSI is highly environment-dependent; the ADR-030 field model
subtraction must run before this measurement is taken to isolate body perturbation
from room standing-wave artifacts.

#### Node: Respiratory_Pattern

Extracted by the ADR-021 BreathingExtractor (0.1–0.5 Hz bandpass) plus the
ADR-030 persistence layer that accumulates statistics over the enrollment window.

```rust
pub struct RespiratoryPatternNode {
    pub node_type: &'static str,        // "Respiratory_Pattern"
    pub baseline_bpm: f32,              // mean RR (normal adult: 12–20 BPM)
    pub depth_amplitude_normalized: f32, // tidal depth proxy from CSI variance
    pub inspiration_expiration_ratio: f32, // I:E ratio (1:1.5 to 1:3 typical)
    pub hrv_rsa_power: f32,             // respiratory sinus arrhythmia spectral power
    pub apnea_index: f32,               // events per hour of significant pauses
    pub waveform_regularity: f32,       // coefficient of variation of breath intervals
    pub confidence: f32,
    pub window_seconds: u32,
}
```

Note: the `apnea_index` field is a biophysical proxy signal (pause events in
the signal), not a clinical AHI score. It is provided for signature
discriminability, not diagnostic use.

#### Node: Gait_Timing

Extracted from the 17-keypoint Kalman pose tracker (`pose_tracker.rs`, ADR-029
Sect 2.7) during the gait phase of the enrollment protocol. The tracker uses
ruvector-mincut for person separation and AETHER re-ID for identity continuity.

```rust
pub struct GaitTimingNode {
    pub node_type: &'static str,        // "Gait_Timing"
    pub cadence_steps_per_min: f32,     // steps per minute
    pub stride_period_variance: f32,    // coefficient of variation of stride period
    pub double_support_pct: f32,        // fraction of gait cycle in double support
    pub asymmetry_index: f32,           // |left_stride - right_stride| / mean_stride
    pub step_width_m: f32,              // lateral distance between foot strikes (proxy)
    pub velocity_variance: f32,         // gait speed variability
    pub confidence: f32,
    pub stride_count: u32,              // number of strides captured during enrollment
}
```

Gait biometrics from WiFi CSI are documented in WiGait (Adib et al., SIGCOMM
2015) and WiDraw (Wang et al., MobiCom 2014). Discrimination across 10+ subjects
in the same household is an open research question for the WiFi-only modality.

#### Node: Skeletal_Proportions

Derived from the ADR-079 camera + CSI paired keypoint pipeline when available,
or from CSI-only pose estimation (ADR-023 CsiToPoseTransformer) in camera-free
deployments. Encodes body geometry as ratios (not absolute values) for scale
invariance.

```rust
pub struct SkeletalProportionsNode {
    pub node_type: &'static str,        // "Skeletal_Proportions"
    pub torso_to_leg_ratio: f32,        // torso height / leg length
    pub shoulder_to_hip_ratio: f32,     // shoulder width / hip width
    pub upper_to_lower_arm_ratio: f32,  // upper arm / forearm
    pub upper_to_lower_leg_ratio: f32,  // thigh / shin
    pub head_to_torso_ratio: f32,       // head height / torso height
    pub arm_span_to_height_ratio: f32,  // Vitruvian ratio (close to 1.0 for most adults)
    pub confidence: f32,
    pub keypoint_source: String,        // "camera_paired" | "csi_only" | "fused"
}
```

CSI-only skeletal proportion estimation has ~15–25% error on individual ratio
values (open research; baseline from ADR-023 MPJPE ~91.7 mm at best, per
Person-in-WiFi 3D, CVPR 2024). Camera-paired values (ADR-079) are substantially
more accurate. The node degrades gracefully when only CSI is available.

#### Node: Subcarrier_Reflection_Profile

The per-subcarrier amplitude attenuation and phase shift profile measured when
the subject stands still at three orientations (0°, 90°, 180° rotation). This
encodes the body's RF backscatter cross-section shape, which is determined by
body mass distribution, limb geometry, and clothing/material factors.

```rust
pub struct SubcarrierReflectionProfileNode {
    pub node_type: &'static str,        // "Subcarrier_Reflection_Profile"
    pub vec_segment_id: u64,            // SEG_VEC: 56 × 3 × 2 = 336 f32s
                                        // (56 subcarriers × 3 orientations ×
                                        //  [amplitude_attenuation, phase_shift])
    pub n_subcarriers: u8,              // 56 (HT-LTF) or up to 242 (HE-LTF, ADR-110 C6)
    pub n_orientations: u8,             // 3
    pub frequency_mhz: u32,             // center frequency at measurement time
    pub environment_id: String,         // references field model used for subtraction
    pub confidence: f32,
}
```

This node directly exploits the ADR-030 field model: the empty-room baseline
eigenstate is subtracted before computing the reflection profile, isolating the
person's contribution. Without ADR-030 field subtraction, the profile is too
environment-coupled to be transferable across rooms. With MERIDIAN (ADR-027),
the hardware-normalizer layer maps ESP32-S3 (52 subcarriers HT-LTF) and
ESP32-C6 (242 subcarriers HE-LTF per ADR-110) into a canonical 56-subcarrier
representation before this measurement.

Stored: 336 × 4 = 1,344 bytes in SEG_VEC.

#### Node: Body_Field_Coupling

The AETHER attention map cells weighted by the ADR-030 room eigenmode structure.
Encodes how strongly the person's body couples to each dominant electromagnetic
mode of the room. This is the most physics-grounded node: it captures the
person's interaction with the actual electromagnetic geometry of the space.

```rust
pub struct BodyFieldCouplingNode {
    pub node_type: &'static str,        // "Body_Field_Coupling"
    pub vec_segment_id: u64,            // SEG_VEC: n_eigenmodes × n_keypoints f32s
    pub n_eigenmodes: u8,               // top-K SVD modes from field_model.rs (default K=8)
    pub n_keypoints: u8,                // 17 (COCO)
    pub eigenmode_energy_fractions: Vec<f32>, // fraction of total variance per mode
    pub environment_id: String,         // must match SubcarrierReflectionProfile env
    pub confidence: f32,
}
```

This node is only valid when the same room's field model is available. For
cross-room recognition, MERIDIAN's environment-disentangled embedding (ADR-027)
is used instead. The BodyFieldCoupling node provides additional discriminative
power in single-room deployments and degrades to optional in multi-room contexts.

---

### 3.3 Edge Types

Edges are stored in the SEG_SOUL_GRAPH JSON array. Each edge has a typed
relationship that constrains how the nodes may be used in matching.

| Edge type | Source node(s) | Target node(s) | Semantics |
|---|---|---|---|
| `derived_from` | FieldModel_Residual (implicit) | AetherEmbedding | The embedding was computed after field model subtraction |
| `correlates_with` | Cardiac_HR_Profile | Respiratory_Pattern | Cardiorespiratory coupling at measurement time; correlation coefficient stored as edge weight |
| `temporally_colocated` | Any pair | Any pair | Both nodes were measured in the same time window; ensures consistency |
| `temporally_after` | Post-gait node | Pre-gait node | Nodes acquired sequentially during enrollment protocol |
| `requires_field_model` | SubcarrierReflectionProfile | BodyFieldCoupling | Matching this node requires the same room's ADR-030 field model |
| `fuses` | AetherEmbedding | SubcarrierReflectionProfile | MERIDIAN-normalized fusion: both mapped to environment-invariant space |
| `attested_by` | Any leaf node | WitnessChain | Ed25519 witness covers this node's content hash |
| `derived_by_keypoint_tracker` | GaitTiming | SkeletalProportions | Both extracted from the same pose_tracker.rs output |
| `environment_normalized` | Any node with `environment_id` | MERIDIAN manifest | MERIDIAN (ADR-027) was applied; signature is cross-room capable |

---

### 3.4 The Aggregator vs. the Stored Profile

Two distinct graph instances exist in the runtime:

**Online Aggregator** — a mutable, in-memory graph that accumulates measurements
across multiple sensing windows. Nodes are incrementally updated with Welford
online statistics (`field_model.rs::WelfordStats`). Confidence fields grow toward
1.0 as more frames accumulate. The aggregator never writes to disk during
normal operation.

**Stored Profile** — an immutable, content-addressed `.rvf` file on disk. It is
generated from the aggregator at the end of the enrollment protocol, when all node
confidence fields exceed their minimum thresholds. The stored profile is the
canonical soul signature.

```
Online Aggregator (RAM)                Stored Profile (disk / secure enclave)
+----------------------+               +---------------------------+
| AETHER_Embedding     |  enrollment   | signature-<sha256>.rvf    |
| accumulated over     |  completion   | SEG_MANIFEST              |
| 60-second protocol   +-------------> | SEG_VEC (embedding + refl)|
| Confidence: 0.0→1.0  |  when all     | SEG_META (all node attrs) |
|                      |  gates pass   | SEG_EMBED (AETHER config) |
| Cardiac_HR_Profile   |               | SEG_WITNESS (Ed25519)     |
| accumulated 30s rest |               | SEG_SOUL_GRAPH (graph)    |
+----------------------+               +---------------------------+
```

The aggregator pattern ensures that a partial scan (e.g., subject leaves after
20 seconds) never produces a stored profile — the quality gates prevent premature
commitment (see `scanning-process.md §5`).

---

### 3.5 Serialization

**Binary container:** RVF blob, per `rvf_container.rs`. All numeric data is
little-endian, f32 IEEE 754. Segment alignment: 64 bytes. CRC32 (IEEE 802.3
polynomial) over each segment payload.

**Content addressing:** The file name is:
```
signature-<sha256-hex-of-rvf-bytes>.rvf
```
SHA-256 is computed over the complete concatenated RVF byte stream after
`RvfBuilder::build()`. This is a different hash from the per-segment CRC32;
the CRC32 provides corruption detection within segments, the SHA-256 provides
content-based addressing and enables deduplication.

**JSON-LD sidecar:** An optional `signature-<sha256>.json` file with the same
base name. Structure:

```json
{
  "@context": "https://ruv.net/soul-signature/v1",
  "schema_version": "0.1.0",
  "person_id": "<opaque_u64_hex>",
  "enrolled_at": "2026-05-24T00:00:00Z",
  "enrolled_by_device_id": "<mac_or_device_fingerprint>",
  "rvf_sha256": "<content_hash>",
  "nodes": [
    { "node_type": "AETHER_Embedding", "confidence": 0.92, ... },
    { "node_type": "Cardiac_HR_Profile", "confidence": 0.85, ... },
    ...
  ],
  "edges": [...],
  "witness": {
    "algorithm": "Ed25519",
    "public_key": "<hex>",
    "signature": "<hex>",
    "signed_fields": ["rvf_sha256", "enrolled_at", "enrolled_by_device_id"]
  }
}
```

The JSON-LD sidecar is human-readable and intended for audit and provenance.
It does not contain raw biometric vectors; those stay in the RVF blob.

**ISO/IEC 19794-4 alignment:** The soul signature's graph-based vector template
is conceptually analogous to the ISO/IEC 19794-4 finger image data format
and ISO/IEC 19794-2 minutiae data. The node/edge schema is not binary-compatible
with ISO 19794, but the design intent (typed attribute records, quality scores,
creator provenance) follows the same standard's principles. Future work may
include a conformance layer if regulatory certification is sought.

---

### 3.6 Matching Algorithm

> **Implementation status (§3.6 only):** The matching algorithm described below
> is **implemented and tested** in
> `v2/crates/wifi-densepose-bfld/src/soul_match.rs` (+ `soul_channels.rs`),
> with tests in `v2/crates/wifi-densepose-bfld/tests/soul_match.rs`. The
> implementation is the **first running** version of this formula in the repo:
> it computes calibrated per-channel scores and exposes a real
> `SoulMatchOracle` (`EnrolledMatcher`). **Caveats that remain true:** the
> weights below are unvalidated design intent; named-identity locking is
> **data-gated** — it requires the decisive high-weight channels (a real AETHER
> enrollment embedding + body-resonance) to be fed real measured data, which has
> NOT been done. Measured on synthetic data, the cardiac (0.15) + respiratory
> (0.10) channels **alone** produce a same-vs-cross-person score gap of ~0.0005
> (test `cardiac_alone_cannot_separate_identity_matches_audit`) — i.e. identity
> is NOT separable on those channels, exactly as expected. This status note
> applies to §3.6 ONLY; the broader Soul Signature system remains
> Pre-Implementation.

Given a stored profile `P` and a query embedding `Q` derived from a live sensing
window, the match score is computed as a weighted sum of per-channel cosine
similarities:

```
match_score = sum_i ( w_i * cosine_sim(P.channel_i, Q.channel_i) )
               / sum_i ( w_i * availability(P.channel_i, Q.channel_i) )
```

Where `availability` is 1.0 if both nodes are present and 0.0 if either is absent
(graceful degradation when a channel cannot be measured in the query window).

Default weights (open research; these are design intent, not validated):

| Channel | Weight | Rationale |
|---|---|---|
| AETHER_Embedding | 0.35 | Primary identity anchor; best-studied channel |
| Subcarrier_Reflection_Profile | 0.20 | Body geometry; angle-stable |
| Cardiac_HR_Profile | 0.15 | Physiologically stable in healthy adults |
| Gait_Timing | 0.15 | Well-studied biometric; discriminative |
| Respiratory_Pattern | 0.10 | More variable than cardiac |
| Skeletal_Proportions | 0.05 | Proxy for body shape; CSI-only is noisy |
| Body_Field_Coupling | 0.00 (single-room) / 0.10 (cross-room disabled) | Valid only when room field model available |
| Cardiac_Waveform_Morphology | 0.05 (supplementary) | High SNR requirement |

The threshold for a positive match is a deployment-specific parameter with a
documented FAR/FRR trade-off. The AETHER channel alone achieves >80% mAP at 5
subjects (ADR-024 §2.8 target). The fused multi-channel score is expected to
exceed this; the exact improvement is open research, baseline TBD.

---

### 3.7 Rust Type Sketch

The following sketch shows how the soul signature types would integrate with
the existing codebase. This is a design sketch, not implemented code.

```rust
// In a future: v2/crates/wifi-densepose-sensing-server/src/soul_signature.rs

pub const SEG_SOUL_GRAPH: u8 = 0x10;
pub const SEG_SOUL_INDEX: u8 = 0x11;

/// Complete soul signature as a graph container.
pub struct SoulSignature {
    /// Content-addressed identifier: SHA-256 of the RVF blob bytes.
    pub content_hash: [u8; 32],
    /// Opaque person identifier (never PII directly).
    pub person_id: u64,
    /// Unix timestamp of enrollment completion (nanoseconds).
    pub enrolled_at_ns: u64,
    /// Device that performed enrollment.
    pub enrolled_by_device_id: String,
    /// All graph nodes, typed.
    pub nodes: SoulNodes,
    /// All graph edges.
    pub edges: Vec<SoulEdge>,
    /// Ed25519 witness chain (per ADR-110).
    pub witness: WitnessChain,
}

pub struct SoulNodes {
    pub aether_embedding: Option<AetherEmbeddingNode>,
    pub cardiac_hr: Option<CardiacHRProfileNode>,
    pub cardiac_waveform: Option<CardiacWaveformMorphologyNode>,
    pub respiratory: Option<RespiratoryPatternNode>,
    pub gait_timing: Option<GaitTimingNode>,
    pub skeletal_proportions: Option<SkeletalProportionsNode>,
    pub subcarrier_reflection: Option<SubcarrierReflectionProfileNode>,
    pub body_field_coupling: Option<BodyFieldCouplingNode>,
}

pub struct SoulEdge {
    pub edge_type: SoulEdgeType,
    pub source_node_type: String,
    pub target_node_type: String,
    pub weight: f32, // edge attribute (e.g., correlation coefficient)
}

pub enum SoulEdgeType {
    DerivedFrom,
    CorrelatesWith,
    TemporallyColocated,
    TemporallyAfter,
    RequiresFieldModel,
    Fuses,
    AttestedBy,
    DerivedByKeypointTracker,
    EnvironmentNormalized,
}

impl SoulSignature {
    /// Serialize to an RVF binary blob.
    pub fn to_rvf(&self) -> Vec<u8>;
    /// Deserialize from an RVF binary blob.
    pub fn from_rvf(data: &[u8]) -> Result<Self, SoulError>;
    /// Compute the weighted match score against a query.
    pub fn match_score(&self, query: &SoulQuery, weights: &MatchWeights) -> f32;
    /// Check whether all required nodes meet minimum confidence thresholds.
    pub fn is_complete(&self, policy: &CompletenessPolicy) -> bool;
}
```

---

### 3.8 What the Signature Is NOT

- Not a fingerprint of the room (that is the ADR-030 field model, a separate object).
- Not a waveform recording (the enrolled vectors are statistics and embeddings, not raw CSI).
- Not invertible to the original CSI stream (the AETHER projection head's information bottleneck prevents reconstruction; see ADR-024 §4 Negative consequences).
- Not a single scalar. Reducing to one number for threshold comparison is a deployment decision; the underlying object is a 7-channel graph.
- Not equal to a stored pose. The AETHER embedding captures body dynamics over many windows, not a single body pose at one instant.
