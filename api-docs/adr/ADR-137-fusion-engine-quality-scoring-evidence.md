# ADR-137: Fusion Engine Quality Scoring with Evidence References and Contradiction Flags

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see Implementation Status, commit `4fa3847ac`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (`ruvsense/multistatic.rs` — `fuse`, `attention_weighted_fusion`); `wifi-densepose-ruvector` (`viewpoint/fusion.rs` — `MultistaticArray`); `wifi-densepose-bfld` (`event.rs`) |
| **Relates to** | ADR-029 (RuvSense Multistatic), ADR-031 (RuView Sensing-First RF Mode), ADR-118 (BFLD Beamforming Feedback Layer), ADR-134 (CSI→CIR Time-Domain Multipath), ADR-135 (Empty-Room Baseline Calibration), ADR-136 (RuView Rust Streaming Engine), ADR-138 (WiFi-7 MLO LinkGroup / ArrayCoordinator Clock-Quality Gating) |

---

## 1. Context

### 1.1 The Gap

The multistatic fusion stage decides how much to trust each sensing node and emits a single fused frame, but it discards every input it used to make that decision. Grepping the two fusion implementations confirms this:

- **`v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs`** (`MultistaticFuser::fuse`, lines 196–282) returns a `FusedSensingFrame` whose only quality field is `cross_node_coherence: f32` (line 80). That scalar is computed by `compute_weight_coherence()` (lines 441–460) as a normalized Shannon entropy over the softmax attention weights — a single number with no record of *which* weights produced it, which subcarriers drove the attention logits, or whether the CIR gate (`cir_gate_coherence`, lines 292–327) actually contributed or silently fell back on `CirError::UnsanitizedPhase`.
- **`v2/crates/wifi-densepose-ruvector/src/viewpoint/fusion.rs`** (`MultistaticArray::fuse`, lines 358–436) is richer — it emits `ViewpointFusionEvent` values (lines 183–219) and reports `gdi` / `n_effective` on `FusedEmbedding` — but its quality signal is still split across heterogeneous channels: a `coherence: f32` on the output struct, a `CoherenceGateTriggered { accepted }` event, and a `FusionError::CoherenceGateClosed` on the error path. There is no single auditable record that says *this fused output is trustworthy because X, Y, Z, but be aware of contradiction C*.

The validation that *does* happen is thrown away rather than recorded:

- `multistatic.rs::fuse` checks `timestamp_us` spread against `guard_interval_us` (lines 205–215) and returns `MultistaticError::TimestampMismatch` — but on the success path the fact that timestamps *passed* (and by how much margin) is never carried forward. A consumer cannot tell a frame fused from microsecond-aligned nodes from one fused at the 4999 µs edge of the 5000 µs guard.
- Neither implementation checks **calibration alignment**. ADR-135 finalises a per-node `BaselineCalibration` with a `captured_at_unix_s` and a `tier`, and `BaselineCalibration::subtract()` already returns `CalibrationError::TierMismatch`. But fusion does not know which baseline (if any) was applied to each node frame, so it cannot detect the dangerous case where node A's frame was baseline-subtracted against a fresh calibration and node B's against a stale one — producing amplitudes on incomparable scales that the attention softmax in `attention_weighted_fusion` (lines 364–435) will silently average together.
- **Amplitude scale comparability is assumed, not enforced.** `attention_weighted_fusion` computes a cosine similarity of each node's amplitude vector against the consensus mean (lines 384–397). Cosine similarity is scale-invariant *per node*, which masks the problem: two nodes with the same shape but a 2× gain difference look perfectly coherent, yet the weighted-sum fusion (lines 411–422) adds raw `w * amp[i]` and so the louder node dominates the fused amplitude regardless of its attention weight. The fix in §2.5 is to normalize before pooling, but today there is nothing in the codebase that does it explicitly.

Downstream, the BFLD privacy layer cannot react to fusion quality at all. `wifi-densepose-bfld/src/event.rs` constructs a `BfldEvent` with a `privacy_class` (line 60) and masks identity fields at `Restricted` via `apply_privacy_gating()` (lines 112–117), and `privacy_gate.rs::PrivacyGate::demote` (lines 31–75) is the monotonic-demote primitive. But the demotion decision is driven by policy, not by sensing evidence. There is no path by which "the fusion engine detected that two nodes disagree about the world" can lower the emitted privacy class. A contradictory fuse is published at the same class as a clean one.

### 1.2 What This ADR Adds

A single, serializable `QualityScore` that travels alongside every fused frame and answers four questions with evidence rather than a scalar:

1. **How good is this fusion?** — `base_coherence` plus the `per_node_weights` that produced it.
2. **Why is it good (or bad)?** — a list of `EvidenceRef` values naming the concrete checks that fired (coherence-gate threshold crossed, CIR dominant-tap ratio, weight entropy, calibration applied).
3. **What is wrong with it?** — a list of `ContradictionFlag` values for the validations that *failed* but were tolerated (timestamp at the guard edge, calibration-id disagreement, phase alignment failure, drift-profile conflict).
4. **Is it safe to publish at full fidelity?** — a non-empty contradiction set lowers the BFLD `privacy_class` and emits a witness record, honouring the project rule that every emitted semantic state traces to signal evidence + model/calibration version + a privacy decision.

This is the fusion-layer counterpart to ADR-135's `CalibrationDeviationScore`: where ADR-135 scores one frame against one baseline, ADR-137 scores one *fusion* against all of its contributing node frames and their baselines.

### 1.3 Pipeline Position

```
Per-node CSI (post phase_sanitizer, phase_align, ADR-135 subtract)
   → CalibratedFrame wrapper            ← NEW (carries calibration_id, capture_ns)
   → multistatic.rs::fuse()
        ├─ capture_ns epoch-alignment check  → ContradictionFlag::TimestampMismatch
        ├─ calibration_id agreement check     → ContradictionFlag::CalibrationIdMismatch
        ├─ normalize-then-concat (per §2.5)
        ├─ attention_weighted_fusion()        → EvidenceRef::WeightEntropy, per_node_weights
        └─ cir_gate_coherence()               → EvidenceRef::CirDominantTapRatio
   → (FusedSensingFrame, QualityScore)   ← NEW tuple return
   → ruvector MultistaticArray (embedding fusion, same QualityScore contract)
   → BFLD emitter
        └─ if !contradiction_flags.is_empty():
              privacy_class = privacy_class.max(Restricted)   (demote)
              emit witness record (ADR-134 proof chain)
   → BfldEvent
```

The `QualityScore` is computed *during* `fuse`, not bolted on afterward, because the evidence it records (attention weights, the CIR fallback decision, the timestamp margin) only exists inside that function's scope today.

---

## 2. Decision

### 2.1 `QualityScore`: the unified fusion-quality record

`QualityScore` is the canonical output of every fusion stage, returned next to the existing frame/embedding type. It is defined in `ruvsense/multistatic.rs` (re-exported from `ruvsense/mod.rs`) and consumed unchanged by `viewpoint/fusion.rs` and `wifi-densepose-bfld`.

```rust
use num_complex::Complex32;

/// Identifies which sensing family produced a fused frame.  Lets a single
/// QualityScore be correlated across the signal-domain fuser
/// (`multistatic.rs`) and the embedding-domain fuser (`viewpoint/fusion.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FamilyId {
    /// `ruvsense/multistatic.rs` CSI/CIR-domain fusion.
    MultistaticCsi,
    /// `ruvector/viewpoint/fusion.rs` AETHER-embedding fusion.
    ViewpointEmbedding,
}

/// Auditable quality record for one fused frame.
///
/// Every semantic state downstream of fusion traces back to exactly one
/// `QualityScore`, which in turn names the signal evidence
/// (`evidence_refs`), the calibration version (`calibration_id`), and the
/// privacy-relevant disagreements (`contradiction_flags`) that informed it.
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Which fuser produced this score.
    pub family_id: FamilyId,
    /// Capture-clock timestamp (ns) of the fused cycle, derived from the
    /// median of the contributing node `capture_ns` values.
    pub capture_ns: u64,
    /// The calibration epoch all contributing frames agreed on, or `None`
    /// when frames disagreed (see `ContradictionFlag::CalibrationIdMismatch`).
    pub calibration_id: Option<CalibrationId>,
    /// Coherence in [0, 1] before any contradiction penalty is applied.
    /// For the CSI fuser this is the entropy-of-weights value currently
    /// returned as `cross_node_coherence`; for the embedding fuser it is the
    /// `CoherenceState::coherence()` value.
    pub base_coherence: f32,
    /// Per-contributing-node attention weight, node-index aligned with the
    /// fused frame's `node_frames` / viewpoint list.  Sums to ~1.0.
    pub per_node_weights: Vec<f32>,
    /// Concrete checks that fired *in support* of this fusion.
    pub evidence_refs: Vec<EvidenceRef>,
    /// Tolerated-but-recorded disagreements.  A non-empty set forces a BFLD
    /// privacy demotion (see §2.7).
    pub contradiction_flags: Vec<ContradictionFlag>,
    /// Monotonic capture-clock time at which this score was computed (ns).
    pub timestamp_computed_ns: u64,
}

/// Calibration epoch identifier.  Derived from the ADR-135
/// `BaselineCalibration::captured_at_unix_s` plus device id; stable across
/// reboots, changes only on recalibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CalibrationId(pub u64);
```

`QualityScore` deliberately mirrors the shape of the `QualityScored` trait introduced in ADR-136 (the streaming-engine frame contract). It implements that trait so the streaming engine can pull a uniform quality view off any stage:

```rust
/// Defined in ADR-136 (`ruview-streaming-engine`); re-stated here for the
/// `impl`.  A stage that produces quality-scored output implements this so
/// the engine can route, gate, and log on quality uniformly.
pub trait QualityScored {
    fn quality(&self) -> &QualityScore;
}

impl QualityScored for (FusedSensingFrame, QualityScore) {
    fn quality(&self) -> &QualityScore {
        &self.1
    }
}
```

**Why a struct and not just more fields on `FusedSensingFrame`:** the two fusers (`multistatic.rs` and `viewpoint/fusion.rs`) produce different payloads (`FusedSensingFrame` vs `FusedEmbedding`) but should produce the *same* quality contract. A shared `QualityScore` is the only thing that lets the BFLD layer treat both uniformly. Inlining quality fields into each payload would force the privacy logic to branch on payload type.

### 2.2 `EvidenceRef`: why a fusion was trusted

`EvidenceRef` records the positive evidence. Each variant carries the *value that crossed a threshold*, not just a boolean, so the witness record (§2.7) is reproducible.

```rust
/// A single piece of positive evidence supporting a fusion decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvidenceRef {
    /// The coherence-gate threshold was met. `coherence` is the value,
    /// `threshold` the configured gate (mirrors ADR-031 coherence gate and
    /// `viewpoint/coherence.rs::CoherenceGate`).
    CoherenceGateThreshold { coherence: f32, threshold: f32 },
    /// The ADR-134 CIR dominant-tap ratio contributed to the gate. `ratio`
    /// is `Cir::dominant_tap_ratio`; `blended` is true when it was actually
    /// folded into `base_coherence` (false on `UnsanitizedPhase` fallback).
    CirDominantTapRatio { ratio: f32, blended: bool },
    /// Attention-weight entropy supported a balanced (multi-node) fusion.
    /// `normalized_entropy` is the `compute_weight_coherence` output.
    WeightEntropy { normalized_entropy: f32, n_nodes: usize },
    /// An ADR-135 baseline was applied to every contributing frame at a
    /// single agreed calibration epoch before pooling.
    CalibrationApplied { calibration_id: CalibrationId, n_frames: usize },
}
```

`CirDominantTapRatio { blended: false }` is itself useful evidence: it records that the CIR gate was *attempted* but fell back, which today is invisible (the `Err(CirError::UnsanitizedPhase)` arm at `multistatic.rs` line 321 silently returns `freq_coherence`).

### 2.3 `ContradictionFlag`: what was wrong but tolerated

`ContradictionFlag` records validations that failed without being fatal. These are the cases where today's code either hard-errors (losing the chance to degrade gracefully) or silently passes (losing the chance to warn).

```rust
/// A tolerated disagreement detected during fusion.  A non-empty set lowers
/// the emitted BFLD privacy_class (§2.7) and produces a witness record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContradictionFlag {
    /// Node capture_ns values spread within the guard interval but beyond a
    /// stricter "comparable" sub-threshold.  Carries the observed spread.
    TimestampMismatch { spread_ns: u64, soft_guard_ns: u64 },
    /// Contributing frames carried different calibration_id values.  `expected`
    /// is the modal (most common) id; `seen` counts the disagreeing frames.
    CalibrationIdMismatch { expected: CalibrationId, disagreeing: usize },
    /// Phase alignment (LO offset estimation, `phase_align.rs`) did not
    /// converge for at least one node, so its phase contribution is suspect.
    PhaseAlignmentFailed { node_idx: usize },
    /// A node's ADR-135 drift_score / DriftProfile conflicts with the array
    /// consensus (e.g., one node reports a static environment while the
    /// majority report motion), suggesting that node is mis-calibrated.
    DriftProfileConflict { node_idx: usize, drift_score: f32 },
    /// Raised upstream by the ADR-138 `ArrayCoordinator`: a node's coherence
    /// dropped beyond `sigma`σ of its rolling mean, so its observation
    /// contradicts the array's rolling expectation.
    CoherenceDrop { node_idx: usize, sigma: f32 },
    /// Raised upstream by the ADR-138 `ArrayCoordinator`: the array's Geometric
    /// Diversity Index fell below the geometry-sufficiency floor, so directional
    /// estimates are under-determined. Carries the observed GDI.
    GeometryInsufficient { gdi: f32 },
}
```

`ContradictionFlag` is the **single canonical type** for tolerated disagreements across the fusion path; it is defined here and re-used (not re-declared) by ADR-138. The first four variants originate inside `multistatic.rs::fuse` (§2.4); the last two (`CoherenceDrop`, `GeometryInsufficient`) originate one stage upstream in the ADR-138 `ArrayCoordinator` and arrive on `DirectionalEvidence.contradictions`, which `fuse` folds into the same `QualityScore.contradiction_flags` vector. `node_idx` is the index into the fused frame's node ordering; the coordinator's `NodeId` is resolved to that index at the hand-off.

The distinction between `MultistaticError::TimestampMismatch` (hard error, line 47) and `ContradictionFlag::TimestampMismatch` is intentional:

- The **hard error** fires when `spread > guard_interval_us` — frames are simply not from the same sensing cycle and must not be fused.
- The **soft flag** fires when `soft_guard_ns < spread <= guard_interval_us` — the frames *can* be fused (they are within the TDMA cycle) but the alignment is loose enough that the fused output should not be published at full identity fidelity. Default `soft_guard_ns = guard_interval_us / 5` (1000 ns when the guard is 5 µs).

### 2.4 `fuse()` rework: validate-record-fuse

`multistatic.rs::fuse` is changed to return `Result<(FusedSensingFrame, QualityScore), MultistaticError>`. The hard-error preconditions (`NoFrames`, `InsufficientNodes`, `DimensionMismatch`, and the *hard* `TimestampMismatch`) are unchanged. The new logic builds the evidence and contradiction lists during the existing passes.

```rust
pub fn fuse(
    &self,
    node_frames: &[CalibratedFrame],   // §2.5: wrapper, was &[MultiBandCsiFrame]
) -> Result<(FusedSensingFrame, QualityScore), MultistaticError> {
    if node_frames.is_empty() {
        return Err(MultistaticError::NoFrames);
    }

    let mut evidence = Vec::new();
    let mut contradictions = Vec::new();

    // ---- capture_ns epoch alignment (hard + soft) -----------------------
    if node_frames.len() > 1 {
        let min = node_frames.iter().map(|f| f.capture_ns).min().unwrap();
        let max = node_frames.iter().map(|f| f.capture_ns).max().unwrap();
        let spread = max - min;
        let guard_ns = self.config.guard_interval_us * 1000;
        if spread > guard_ns {
            return Err(MultistaticError::TimestampMismatch {
                spread_us: spread / 1000,
                guard_us: self.config.guard_interval_us,
            });
        }
        let soft = guard_ns / 5;
        if spread > soft {
            contradictions.push(ContradictionFlag::TimestampMismatch {
                spread_ns: spread,
                soft_guard_ns: soft,
            });
        }
    }

    // ---- calibration_id agreement ---------------------------------------
    let calibration_id = resolve_calibration_id(node_frames, &mut evidence, &mut contradictions);

    // ---- normalize then attention-pool (§2.5) ---------------------------
    let (amps, phases) = normalize_by_calibration(node_frames);
    let (fused_amp, fused_ph, base_coherence, weights) =
        attention_weighted_fusion(&amps, &phases, self.config.attention_temperature);
    evidence.push(EvidenceRef::WeightEntropy {
        normalized_entropy: base_coherence,
        n_nodes: weights.len(),
    });

    // ---- CIR gate (records blended/fallback as evidence) ----------------
    let coherence = self.cir_gate_coherence_recorded(base_coherence, node_frames, &mut evidence);

    // ---- phase-alignment + drift conflicts ------------------------------
    record_phase_and_drift_conflicts(node_frames, &mut contradictions);

    let now = monotonic_capture_ns();
    let quality = QualityScore {
        family_id: FamilyId::MultistaticCsi,
        capture_ns: median_capture_ns(node_frames),
        calibration_id,
        base_coherence,
        per_node_weights: weights,
        evidence_refs: evidence,
        contradiction_flags: contradictions,
        timestamp_computed_ns: now,
    };
    let frame = FusedSensingFrame { /* existing fields, coherence = coherence */ };
    Ok((frame, quality))
}
```

`attention_weighted_fusion` is changed only to *return* its `weights` vector (it already computes it at lines 401–408) instead of discarding it — `per_node_weights` is exactly that vector, costing nothing extra to surface.

**Interface boundary:** `FusedSensingFrame` keeps `cross_node_coherence` for backward compatibility, set to the post-gate `coherence`. New consumers read `QualityScore.base_coherence`; the scalar on the frame is now derived, not authoritative.

### 2.5 Normalize-then-concat: explicit `CalibratedFrame`

Today `fuse` consumes `&[MultiBandCsiFrame]` and relies on the implicit z-score normalization buried in `hardware_norm.rs::CanonicalCsiFrame`. ADR-137 makes calibration explicit by introducing a thin wrapper that carries the calibration provenance from ADR-135 to the fuser:

```rust
/// A node frame whose amplitude/phase have been baseline-subtracted and
/// normalized by a *named* ADR-135 calibration.  The wrapper makes the
/// calibration provenance an explicit fusion input rather than an implicit
/// property of CanonicalCsiFrame.
#[derive(Debug, Clone)]
pub struct CalibratedFrame {
    /// The underlying multi-band frame (per-channel amplitude/phase).
    pub inner: MultiBandCsiFrame,
    /// Capture-clock timestamp (ns).  Promoted from `timestamp_us * 1000`
    /// when the source only has microsecond resolution.
    pub capture_ns: u64,
    /// Which ADR-135 baseline normalized this frame, or `None` if the node
    /// is running uncalibrated (ADR-135 fallback mode).
    pub calibration_id: Option<CalibrationId>,
    /// Per-subcarrier gain applied during normalization (from the ADR-135
    /// `amp_mean` / `amp_variance`), retained so the fuser can renormalize
    /// onto a common scale before pooling.
    pub norm_gain: Vec<f32>,
    /// Per-subcarrier phase offset removed (from the ADR-135 circular mean).
    pub norm_phase_offset: Vec<f32>,
}
```

`normalize_by_calibration` divides each node's amplitude by its own `norm_gain` RMS so that, after normalization, every node's amplitude is unit-scaled regardless of per-node hardware gain. Only then does the attention pool run. This closes the scale-comparability hole described in §1.1: the cosine-similarity attention logits and the weighted sum now operate on the same scale, so attention weight (not loudness) determines a node's contribution.

**Why explicit over implicit:** `hardware_norm.rs` z-score normalization uses population statistics computed from the live signal including any occupant. The ADR-135 baseline statistics are computed from a *known-empty* room. Normalizing by the baseline (a) makes nodes comparable on a physically meaningful zero, and (b) gives the fuser the `calibration_id` it needs to detect cross-node calibration disagreement. The wrapper costs `O(K)` extra memory per node frame (two `Vec<f32>`), negligible against the `MultiBandCsiFrame` it wraps.

### 2.6 Embedding-domain fuser: same contract

`viewpoint/fusion.rs::MultistaticArray::fuse` is changed to return `Result<(FusedEmbedding, QualityScore), FusionError>` with `family_id: FamilyId::ViewpointEmbedding`. The mapping from its existing machinery to the unified record:

| `QualityScore` field | Source in `viewpoint/fusion.rs` |
|----------------------|----------------------------------|
| `base_coherence` | `self.coherence_state.coherence()` (line 382) |
| `per_node_weights` | attention weights from `self.attention.fuse(...)` (line 408) — surfaced, currently internal to `CrossViewpointAttention` |
| `evidence_refs` → `CoherenceGateThreshold` | `CoherenceGate::evaluate` (line 383) plus the configured `coherence_threshold` |
| `contradiction_flags` → `DriftProfileConflict` | a viewpoint whose `snr_db` passed the filter but whose phase-diff series diverges from the coherent majority |
| `calibration_id` | from each `ViewpointEmbedding`'s source `CalibratedFrame` |

The existing `ViewpointFusionEvent::CoherenceGateTriggered` and `FusionError::CoherenceGateClosed` are retained — they remain the *control-flow* signal — while `QualityScore` becomes the *data* signal that travels with the frame. The `CoherenceGateClosed` error still aborts fusion; `QualityScore` is only produced on the success path. A gate that is open but near the threshold records `EvidenceRef::CoherenceGateThreshold` with the margin, so a barely-open gate is auditable.

### 2.7 Wiring contradictions into the BFLD privacy boundary

This is where fusion quality becomes a privacy decision. The BFLD emitter (`wifi-densepose-bfld`) gains a single rule:

> A `QualityScore` with a non-empty `contradiction_flags` set forces the emitted `BfldEvent.privacy_class` to be **at least** `Restricted`.

Because `PrivacyClass` is ordered (`Raw=0 < Derived=1 < Anonymous=2 < Restricted=3`, `lib.rs` lines 84–94) and demotion is monotonic (`privacy_gate.rs::demote` rejects any decrease in class number), "at least Restricted" is `privacy_class.max(Restricted)` — i.e. a demote, never a promote:

```rust
// In the BFLD emitter, before BfldEvent::with_privacy_gating(...):
let effective_class = if quality.contradiction_flags.is_empty() {
    policy_class                       // normal policy decision
} else {
    policy_class.max(PrivacyClass::Restricted)   // demote on contradiction
};
```

At `Restricted`, `BfldEvent::apply_privacy_gating` (event.rs lines 112–117) already nulls `identity_risk_score` and `rf_signature_hash`. So a contradictory fuse — two nodes that disagree about calibration, timestamp, phase, or drift — automatically stops leaking the identity-surface fields. The rationale: contradiction means the system is not confident *whose* signal it fused; emitting an identity-risk score or RF signature hash on an un-trusted fusion is exactly the failure the privacy layer exists to prevent.

A non-empty contradiction set also emits a **witness record** through the ADR-134 proof chain (the `verify.py` / `expected_features.sha256` / `source-hashes.txt` witness schema, ADR-134 §2.10). The record captures: `capture_ns`, `family_id`, the `contradiction_flags` (with their carried values), the resulting `effective_class`, and a hash of `per_node_weights`. This makes every privacy demotion reproducible and auditable — satisfying the project invariant that each emitted semantic state traces to signal evidence + calibration version + a recorded privacy decision.

```
QualityScore.contradiction_flags non-empty
   ├─ effective_class = policy_class.max(Restricted)   (demote, monotonic)
   ├─ BfldEvent gated → identity_risk_score = None, rf_signature_hash = None
   └─ witness record { capture_ns, family_id, flags, effective_class,
                       blake3(per_node_weights) } → ADR-134 proof chain
```

**Interface boundary:** the BFLD crate depends only on `QualityScore` (a plain data struct re-exported from `wifi-densepose-signal`), not on the fusers themselves. No new control coupling is introduced; the emitter reads two fields (`contradiction_flags`, `calibration_id`) and a policy class.

### 2.8 Proposed Rust API surface (summary)

| Item | Location | Kind |
|------|----------|------|
| `QualityScore`, `FamilyId`, `CalibrationId` | `ruvsense/multistatic.rs`, re-exported `ruvsense/mod.rs` | struct / enum |
| `EvidenceRef`, `ContradictionFlag` | `ruvsense/multistatic.rs` | enum |
| `CalibratedFrame` | `ruvsense/multistatic.rs` | struct |
| `impl QualityScored for (FusedSensingFrame, QualityScore)` | `ruvsense/multistatic.rs` | trait impl (ADR-136 trait) |
| `MultistaticFuser::fuse → Result<(FusedSensingFrame, QualityScore), _>` | `ruvsense/multistatic.rs` | changed signature |
| `MultistaticArray::fuse → Result<(FusedEmbedding, QualityScore), _>` | `viewpoint/fusion.rs` | changed signature |
| BFLD emitter contradiction→demote rule | `wifi-densepose-bfld` emitter | new logic |

### 2.9 Testing / Acceptance

**T1 — Evidence is recorded on a clean fuse (unit, `multistatic.rs`).** Two `CalibratedFrame`s with identical `calibration_id`, `capture_ns` within `soft_guard_ns`, sanitized phase. Assert the returned `QualityScore` has `contradiction_flags.is_empty()`, contains `EvidenceRef::WeightEntropy` and `EvidenceRef::CalibrationApplied`, and `per_node_weights.len() == 2` summing to ~1.0.

**T2 — CIR fallback is recorded, not hidden (unit).** Feed a frame whose phase is unsanitized (phase variance > 10 rad², triggering `CirError::UnsanitizedPhase`). Assert `evidence_refs` contains `EvidenceRef::CirDominantTapRatio { blended: false, .. }` and `base_coherence` equals the pre-gate frequency coherence (graceful fallback preserved).

**T3 — Soft timestamp contradiction (unit).** Two frames with `capture_ns` spread `> soft_guard_ns` but `<= guard_interval`. Assert success (no `MultistaticError`) AND `contradiction_flags` contains `TimestampMismatch { spread_ns, .. }`.

**T4 — Calibration-id mismatch (unit).** Two frames with different `calibration_id`. Assert `QualityScore.calibration_id == None` and `contradiction_flags` contains `CalibrationIdMismatch { expected, disagreeing: 1 }`.

**T5 — Hard timestamp error still hard (unit, regression).** Spread `> guard_interval`. Assert `Err(MultistaticError::TimestampMismatch)` — no `QualityScore` produced. Confirms the existing test `timestamp_mismatch_error` (multistatic.rs line 585) still passes against the new signature.

**T6 — Normalize-then-concat scale invariance (unit).** Two nodes, identical amplitude shape, node B scaled 2×. Assert that after `normalize_by_calibration` the fused amplitude is within 1% of the single-node result (loudness no longer dominates) and `per_node_weights` are ~equal.

**T7 — Privacy demotion on contradiction (unit, `wifi-densepose-bfld`).** Build a `QualityScore` with one `ContradictionFlag` and a policy class of `Derived`. Assert the emitted `BfldEvent.privacy_class == Restricted`, and that `identity_risk_score` and `rf_signature_hash` serialize as absent (reuse the gating assertions in event.rs).

**T8 — Clean fuse keeps policy class (unit).** Same as T7 but with empty `contradiction_flags`. Assert `privacy_class == Derived` (no demotion) and identity fields present.

**T9 — Witness determinism (CI proof chain).** A fixed two-node contradictory fuse produces a `QualityScore` whose witness record hashes to a recorded value in `expected_features.sha256` under key `fusion_quality_contradiction_v1`. The `verify.py` extension `fusion_quality_check()` reproduces it. Mirrors ADR-135 §2.12 Tier 7 and ADR-134 §2.10.

**T10 — `QualityScored` trait round-trip (unit).** Assert `(frame, quality).quality()` returns the embedded `QualityScore` by reference, satisfying the ADR-136 contract.

**Acceptance criteria:** all existing `multistatic.rs` tests (lines 546–697) and `viewpoint/fusion.rs` tests (lines 564–743) pass after the signature change (adapted to destructure the tuple); T1–T10 pass; `cargo test --workspace --no-default-features` reports 0 failures; `verify.py` prints `VERDICT: PASS` with the new key.

---

## 3. Consequences

### 3.1 Positive

- **Fusion decisions become auditable.** Every fused frame now carries the evidence that produced its coherence and the disagreements that were tolerated. A field engineer can read why a frame was trusted without re-running the fuser.
- **Calibration disagreement is caught.** The `CalibrationIdMismatch` contradiction surfaces the previously-invisible failure where nodes are normalized against baselines of different vintage — the silent amplitude-scale corruption from §1.1.
- **CIR fallback stops being silent.** `EvidenceRef::CirDominantTapRatio { blended: false }` records the `UnsanitizedPhase` fallback that today disappears at `multistatic.rs` line 321.
- **Privacy degrades safely under uncertainty.** A contradictory fusion can no longer publish identity-surface fields; the demotion is monotonic and witnessed.
- **One contract, two fusers.** The signal-domain and embedding-domain fusers expose identical quality semantics, so the streaming engine (ADR-136) and BFLD layer treat them uniformly.
- **Traceability invariant satisfied.** Each `BfldEvent` traces to a `QualityScore` → `EvidenceRef`s (signal evidence) + `calibration_id` (calibration version) + the recorded `effective_class` (privacy decision).

### 3.2 Negative

- **Breaking signature change.** Both `fuse` functions change their return type to a tuple. Every call site and every existing test (multistatic.rs and viewpoint/fusion.rs) must destructure. This is mechanical but touches ~25 test functions.
- **`CalibratedFrame` wrapper churn.** `fuse` no longer takes `&[MultiBandCsiFrame]` directly; callers must wrap, threading the ADR-135 calibration through. Uncalibrated nodes pass `calibration_id: None` and lose the `CalibrationApplied` evidence (but still fuse).
- **Per-frame allocation.** `evidence_refs` and `contradiction_flags` are `Vec`s. In the common clean-fuse case they hold 2–3 small `Copy` enums; the allocation is bounded but non-zero on the hot path. Mitigation: a `SmallVec` could be substituted if profiling shows pressure (deferred — not premature).

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Over-eager demotion: a benign loose timestamp at the guard edge demotes every frame to Restricted, suppressing identity features the deployment legitimately needs | Medium | Identity-risk scoring effectively disabled in a node array with marginal clock sync | `soft_guard_ns` is configurable (default `guard/5`); ADR-138's `ArrayCoordinator` clock-quality gating can raise the bar so timestamp contradictions only fire on genuinely degraded clocks |
| `DriftProfileConflict` false-positives when one node legitimately sees motion the others cannot (occlusion geometry) | Medium | Spurious privacy demotions in multi-room arrays with partial line-of-sight | Conflict requires a *majority* disagreement, not any single dissenting node; threshold tunable per deployment |
| Witness record volume: a flapping contradiction produces a witness record per cycle (20 Hz) | Low | Witness log growth | Coalesce identical consecutive contradiction sets; emit a witness record only on contradiction-set *transitions*, not every frame |
| `calibration_id` derivation collides for two devices recalibrated in the same second | Low | Two nodes appear to agree on calibration when they don't | `CalibrationId` is `hash(device_id, captured_at_unix_s)`, not the timestamp alone |

---

## 4. Alternatives Considered

### 4.1 Keep the scalar `cross_node_coherence`, add a separate log channel

Rejected. A side-channel log decouples the quality record from the frame it describes; a consumer cannot atomically obtain "this frame and exactly the evidence that produced it." The BFLD privacy decision must be made from the same data that produced the frame, in the same call. A `QualityScore` returned in the tuple guarantees that coupling; a log does not.

### 4.2 Boolean flags instead of evidence-carrying enums

Rejected. `passed_coherence: bool` cannot be reproduced in a witness record — the threshold and value are lost. ADR-135 and ADR-134 both made determinism-by-recorded-value a requirement of the proof chain (`expected_features.sha256`). A boolean breaks that chain. The enums carry the crossing value precisely so the witness hash is reproducible.

### 4.3 Hard-error on every contradiction (no graceful degradation)

Rejected. Promoting `CalibrationIdMismatch` and soft `TimestampMismatch` to fatal `MultistaticError`s would make the array brittle: any transient clock skew or mid-session recalibration would drop the entire fused frame. The whole point of the contradiction flag is that the fusion is *usable but not fully trusted* — degrade fidelity (privacy demote), don't drop data. The genuinely unfusable cases (spread beyond the guard, dimension mismatch) remain hard errors.

### 4.4 Put the demotion logic in the fuser, not the BFLD emitter

Rejected. The fuser produces evidence; it should not know the privacy policy. Privacy class ordering and the `Restricted` semantics live in `wifi-densepose-bfld` (`PrivacyClass`, `PrivacyGate`). Keeping the `max(Restricted)` decision in the emitter preserves the bounded-context separation: signal-processing crates compute *what is true and how confident*, the BFLD crate decides *what may be emitted*. The fuser exports a data struct; the emitter owns the policy.

### 4.5 Reuse `ViewpointFusionEvent` for evidence

Rejected. `ViewpointFusionEvent` (viewpoint/fusion.rs lines 183–219) is an internal event-sourcing log for the `MultistaticArray` aggregate and exists only in the ruvector crate; it does not travel with the frame and is unknown to the signal-domain fuser or the BFLD crate. `QualityScore` is the shared, frame-attached contract both fusers and the privacy layer agree on.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-029 (RuvSense Multistatic) | **Extended**: `MultistaticFuser::fuse` gains the `(FusedSensingFrame, QualityScore)` return; the attention/coherence machinery is unchanged but its byproducts are now surfaced |
| ADR-031 (Sensing-First RF Mode) | **Extended**: `MultistaticArray::fuse` adopts the same `QualityScore` contract; coherence-gate events are retained as control flow |
| ADR-118 (BFLD Beamforming Feedback Layer) | **Consumer**: the BFLD emitter reads `contradiction_flags` to demote `privacy_class`; reuses `PrivacyClass`, `PrivacyGate::demote`, and `BfldEvent::apply_privacy_gating` |
| ADR-134 (CSI→CIR) | **Evidence source + witness chain**: `EvidenceRef::CirDominantTapRatio` records `Cir::dominant_tap_ratio`; the contradiction witness record uses the ADR-134 `verify.py` proof schema |
| ADR-135 (Empty-Room Baseline Calibration) | **Prerequisite**: `CalibratedFrame.calibration_id` / `norm_gain` / `norm_phase_offset` come from `BaselineCalibration`; `CalibrationIdMismatch` and `DriftProfileConflict` are defined against ADR-135 calibration and drift_score |
| ADR-136 (RuView Streaming Engine) | **Contract**: `QualityScore` implements ADR-136's `QualityScored` trait so the streaming engine routes/gates uniformly on fusion quality |
| ADR-138 (LinkGroup / ArrayCoordinator Clock-Quality Gating) | **Refines contradiction sensitivity**: ArrayCoordinator clock quality informs the `soft_guard_ns` threshold so `TimestampMismatch` flags fire on genuinely degraded clocks, not on healthy WiFi-7 MLO arrays |

---

## 6. References

### Production Code

- `v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs` — `MultistaticFuser::fuse` (196–282), `attention_weighted_fusion` (364–435), `compute_weight_coherence` (441–460), `cir_gate_coherence` (292–327), `MultistaticError` (36–56), `FusedSensingFrame` (62–81)
- `v2/crates/wifi-densepose-ruvector/src/viewpoint/fusion.rs` — `MultistaticArray::fuse` (358–436), `FusedEmbedding` (54–66), `ViewpointFusionEvent` (183–219), `FusionError` (109–136)
- `v2/crates/wifi-densepose-bfld/src/event.rs` — `BfldEvent` (28–73), `with_privacy_gating` (79–107), `apply_privacy_gating` (112–117)
- `v2/crates/wifi-densepose-bfld/src/privacy_gate.rs` — `PrivacyGate::demote` (31–75), monotonic demotion invariant
- `v2/crates/wifi-densepose-bfld/src/lib.rs` — `PrivacyClass` (84–94), `as_u8` (114)
- `v2/crates/wifi-densepose-signal/src/ruvsense/cir.rs` — `Cir` (265), `dominant_tap_ratio` (275), `CirEstimator::estimate` (380), `CirConfig::ht20` (164)
- `v2/crates/wifi-densepose-signal/src/ruvsense/multiband.rs` — `MultiBandCsiFrame` (47–57), wrapped by `CalibratedFrame`
- `v2/crates/wifi-densepose-signal/src/ruvsense/calibration.rs` (ADR-135) — `BaselineCalibration`, `CalibrationDeviationScore`, drift_score
- `archive/v1/data/proof/verify.py` — witness proof chain; `fusion_quality_check()` extension
- `archive/v1/data/proof/expected_features.sha256` — hash key `fusion_quality_contradiction_v1` to be added

### External

- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*. — softmax attention weighting reused in `attention_weighted_fusion`; `per_node_weights` is the attention distribution exposed for audit.
- Mardia, K.V. & Jupp, P.E. (2000). *Directional Statistics*. Wiley. — circular phase consensus underlying `PhaseAlignmentFailed` detection (sin/cos pooling in `attention_weighted_fusion`).


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `4fa3847ac`, issue #841): `QualityScore`, `EvidenceRef`, and the canonical `ContradictionFlag`; `MultistaticFuser::fuse_scored()` added additively (does not break `fuse()` or its callers). 6 tests.

**Integration glue -- not yet on the live path:** emission of `CalibrationIdMismatch` / `DriftProfileConflict` / `PhaseAlignmentFailed` once `calibration_id` propagation and the phase-align convergence signal are threaded onto frames; the BFLD witness record emitted on privacy demotion.

**Trust contribution:** sensor *agreement made explicit* -- fusion records the evidence it relied on, and any disagreement automatically tightens the downstream privacy class.

---

## Witness Integrity Review (2026-06-14) — domain-separation fix

A beyond-SOTA security review of `wifi-densepose-engine` (the composition root
that builds the §2.7 trust witness in `witness_of`) found a real **witness
domain-separation gap**, now fixed.

**Finding (witness-gap, HIGH).** `witness_of` concatenated `model_version`,
`calibration_version`, and `privacy_decision` boundary-to-boundary, and the
variable-length `evidence` list carried no explicit count. A string straddling a
field boundary therefore collided with a *different* trust decision —
e.g. a per-room adapter id (ADR-150 §3.4, operator-influenceable) that absorbs
the leading bytes of the calibration epoch (`model="…cal:00a"`, `cal="b"`)
produces the **same** witness as `model="…"`, `cal="cal:00ab"`. Two distinct
privacy-relevant input tuples → one witness defeats the "any privacy-relevant
delta → different witness" guarantee this ADR's §2.7 witness exists to provide.

**Fix.** The witness now (a) prepends a domain tag `ruview.engine.witness.v1`,
(b) writes an explicit 8-byte evidence count, and (c) **length-prefixes every
field** (8-byte LE length ‖ bytes), so field framing is unambiguous regardless
of contents. This is a witness-layout change (all prior witness bytes are
invalidated by design); downstream consumers only assert witness *relationships*
(`assert_ne`/`assert_eq` across runs), not absolute bytes, so nothing breaks.

Pinned by `witness_distinguishes_model_calibration_boundary` and
`witness_distinguishes_evidence_model_boundary` (both fail on the old
concatenation). Witness **determinism** was reviewed and confirmed clean: no
HashMap iteration and no float formatting feed the hash (floats appear only in
the `SemanticState` statement, which is outside the witness).
