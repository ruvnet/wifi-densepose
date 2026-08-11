//! `ruview-witness` — the staged, append-only, hash-linked witness chain.
//!
//! This crate implements **ADR-316** (witness chain), primitive 19 of the
//! ADR-297 perception substrate. Instead of emitting a bare boolean
//! ("person present"), RuView emits a *chain* whose ordered stages record the
//! auditable reasoning behind an output:
//!
//! ```text
//! RF observation ▸ DSP evidence ▸ model inference ▸ independent corroboration
//!               ▸ spatial state ▸ policy decision
//! ```
//!
//! Each stage is a typed record carrying its own [`Confidence`] and its
//! provenance ([`EvidenceLevel`]). The chain is **hash-linked**: each stage
//! binds the hash of the prior stage ([`Stage::prior_hash`]), and the chain
//! carries the recomputed [`WitnessChain::head`] of its terminal stage, so an
//! in-place mutation, a reordering, or a broken link is detectable by
//! [`WitnessChain::verify`] without trusting the emitting host.
//!
//! ## Relationship to sibling ADRs
//!
//! - **ADR-302 ([`ruview_attest`])** roots the chain: the first stage is built
//!   from an authenticated [`VerifiedMeasurement`]
//!   ([`WitnessChain::from_measurement`]), so the whole chain descends from a
//!   verified chain of custody. The stage hash reuses `ruview-attest`'s BLAKE3
//!   ([`ruview_attest::PayloadHash`]) — no separate hash primitive is added.
//! - **ADR-292** contributes [`SourceState`]: a `Synthetic` root can never
//!   present as `LiveVerified`, and it caps the chain's effective evidence
//!   level.
//! - **ADR-299** contributes [`DomainState`] (the `KNOWN → DEGRADED → UNKNOWN`
//!   staleness guard): a low-confidence or out-of-distribution inference is
//!   recorded as such, never silently promoted.
//! - **ADR-318** will attach the real terminal [`PolicyDecision`]; here it is a
//!   faithfully-typed placeholder for the governed action taken (or withheld).
//!
//! ## Honesty discipline (ADR-297 rule 1 / CLAUDE.md)
//!
//! [`Confidence::Unknown`] is a first-class value, never an error. A missing
//! corroboration is recorded as [`Corroboration::None`], never fabricated. The
//! chain's *effective* evidence level is the **minimum** across its stages, so
//! the chain can never claim more than its weakest link.
//!
//! ### Evidence grade
//!
//! Like `ruview-attest`, any guarantee exercised by this crate's tests is
//! **SYNTHETIC / L0** — the fixtures are constructed in code, never captured
//! from silicon. Hash-linking here is tamper-*evident* against in-place
//! mutation; deployment-grade non-repudiation additionally requires the ADR-302
//! per-stage RuField signatures, layered on top of this structure, plus
//! real-hardware evidence.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use ruview_attest::{
    CalibrationRef, DeviceId, PayloadHash, SignedMeasurement, Timestamp, VerifiedMeasurement,
};

/// Width, in bytes, of a stage hash (BLAKE3, via [`ruview_attest::PayloadHash`]).
pub const HASH_LEN: usize = 32;

/// Maximum accepted byte length of any free-text field carried in a stage.
/// Bounds allocation at the (untrusted) construction boundary.
pub const MAX_TEXT_LEN: usize = 256;

/// Domain-separation prefix mixed into every stage's canonical bytes so a stage
/// hash can never collide with a hash computed for another purpose.
const DOMAIN: &[u8] = b"ruview-witness/v1\x00stage\x00";

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure while constructing a stage value from untrusted input.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InputError {
    /// A free-text field exceeded [`MAX_TEXT_LEN`].
    #[error("text field length {got} exceeds maximum {max}", max = MAX_TEXT_LEN)]
    TextTooLong {
        /// The offending length.
        got: usize,
    },
    /// A confidence value was not a finite number in `0.0..=1.0`.
    #[error("confidence {0} is not a finite value in 0.0..=1.0")]
    InvalidConfidence(f32),
}

/// Reason a [`WitnessChain`] operation was rejected. Malformed structure is a
/// hard `Err`, never a panic and never a silently-accepted chain.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ChainError {
    /// A stage whose evidence kind does not permit it here was appended: the
    /// stage order must be strictly increasing (append-only, no reordering).
    #[error("stage {next:?} cannot follow {last:?}: stage order must strictly increase")]
    NotAppendable {
        /// The current terminal stage kind.
        last: StageKind,
        /// The rejected stage kind.
        next: StageKind,
    },
    /// The chain was empty (a chain always roots in an RF observation).
    #[error("chain is empty")]
    Empty,
    /// The root stage was not an RF observation.
    #[error("chain root must be an RF observation, found {0:?}")]
    RootNotObservation(StageKind),
    /// A stage's recorded prior-stage hash did not match the recomputed hash of
    /// its predecessor: a link is broken, missing, or a stage was reordered.
    #[error("broken hash link at stage index {index}")]
    BrokenLink {
        /// Index of the stage whose `prior_hash` did not match.
        index: usize,
    },
    /// Stage order was not strictly increasing (a stage was reordered).
    #[error("non-monotonic stage order at index {index}")]
    NonMonotonicOrder {
        /// Index of the out-of-order stage.
        index: usize,
    },
    /// The terminal stage's recomputed hash did not match the chain head: the
    /// last stage was tampered with in place.
    #[error("terminal stage does not match the recorded chain head (tamper)")]
    TerminalTampered,
}

// ---------------------------------------------------------------------------
// Provenance and confidence primitives
// ---------------------------------------------------------------------------

/// The evidence level of a stage (ADR-282, L0–L5). The chain's effective level
/// is the **minimum** across stages, so the weakest link caps the whole chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EvidenceLevel {
    /// L0 — synthetic / self-referential; no external anchor.
    L0 = 0,
    /// L1.
    L1 = 1,
    /// L2.
    L2 = 2,
    /// L3.
    L3 = 3,
    /// L4.
    L4 = 4,
    /// L5 — strongest anchored evidence.
    L5 = 5,
}

/// The ADR-292 source state of an RF observation. `Unknown` is structurally
/// absent here: a stage that cannot assert a live state records `Synthetic` or
/// `Disconnected`, and a `Synthetic` root can never present as `LiveVerified`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SourceState {
    /// Synthetic input; caps the chain at [`EvidenceLevel::L0`].
    Synthetic = 0,
    /// Live and cryptographically verified (ADR-302).
    LiveVerified = 1,
    /// Live but unverified.
    LiveUnverified = 2,
    /// Live source that has gone stale.
    Stale = 3,
    /// Source disconnected.
    Disconnected = 4,
}

impl SourceState {
    /// Whether this state is the authenticated live state. A `Synthetic` root
    /// answers `false`, upholding the ADR-292 invariant.
    pub fn is_live_verified(&self) -> bool {
        matches!(self, SourceState::LiveVerified)
    }
}

/// The ADR-299 domain-signature gate result for a model inference — the
/// `VALID → DEGRADED → UNKNOWN` staleness guard. Recorded faithfully so a
/// degraded or out-of-distribution inference is never silently promoted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DomainState {
    /// In-distribution; the certificate is valid.
    Known = 0,
    /// Drifting; the capability is degraded and recalibration is due.
    Degraded = 1,
    /// Out of distribution; the answer is UNKNOWN (never a confident class).
    Unknown = 2,
}

/// A stage's confidence. [`Confidence::Unknown`] is a first-class value
/// (ADR-297 rule 1), never an error and never coerced to a number.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Confidence {
    /// A finite confidence in `0.0..=1.0`.
    Known(f32),
    /// The stage has no confidence to report.
    Unknown,
}

impl Confidence {
    /// Construct a known confidence, rejecting non-finite or out-of-range input
    /// at the boundary.
    pub fn known(value: f32) -> Result<Self, InputError> {
        if !value.is_finite() || value < 0.0 || value > 1.0 {
            return Err(InputError::InvalidConfidence(value));
        }
        Ok(Confidence::Known(value))
    }

    /// The unknown confidence.
    pub const fn unknown() -> Self {
        Confidence::Unknown
    }
}

// ---------------------------------------------------------------------------
// Stage kinds and typed per-stage evidence
// ---------------------------------------------------------------------------

/// The ordered kinds of a witness stage. Ordering is the pipeline order; a
/// chain's stages must strictly increase, which is what makes reordering
/// detectable and bounds a chain to at most six stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum StageKind {
    /// The authenticated RF frame envelope (root link).
    RfObservation = 0,
    /// Deterministic DSP features and ADR-137 quality signals.
    DspEvidence = 1,
    /// Model version, raw output, uncertainty, and the ADR-299 gate result.
    ModelInference = 2,
    /// The ADR-300 agreement link (phase 2); "no corroboration" in phase 1.
    IndependentCorroboration = 3,
    /// The ADR-303 ontology entity the inference updated.
    SpatialState = 4,
    /// The terminal governed action (ADR-318).
    PolicyDecision = 5,
}

/// The RF observation stage: the authenticated measurement lineage plus its
/// ADR-292 source state. This is the root link of every chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RfObservation {
    /// The verified chain-of-custody record from `ruview-attest` (ADR-302).
    pub measurement: VerifiedMeasurement,
    /// The ADR-292 source state of the measurement.
    pub source_state: SourceState,
}

impl RfObservation {
    /// Root an observation in a verified measurement and its source state.
    pub fn from_verified(measurement: VerifiedMeasurement, source_state: SourceState) -> Self {
        Self {
            measurement,
            source_state,
        }
    }
}

/// The DSP evidence stage: a deterministic feature descriptor, an SNR-style
/// quality figure, and the ADR-137 quality-gate verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DspEvidence {
    descriptor: String,
    /// Signal-quality figure (e.g. SNR in dB) supporting the features.
    pub snr_db: f32,
    /// Whether the ADR-137 quality gate passed.
    pub quality_ok: bool,
}

impl DspEvidence {
    /// Construct DSP evidence, validating the descriptor length at the boundary.
    pub fn new(
        descriptor: impl Into<String>,
        snr_db: f32,
        quality_ok: bool,
    ) -> Result<Self, InputError> {
        Ok(Self {
            descriptor: checked_text(descriptor.into())?,
            snr_db,
            quality_ok,
        })
    }

    /// The feature descriptor.
    pub fn descriptor(&self) -> &str {
        &self.descriptor
    }
}

/// The model inference stage: which model produced which raw output, with what
/// predictive uncertainty, under which ADR-299 domain gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInference {
    model_version: String,
    label: String,
    /// Predictive uncertainty of the raw output.
    pub uncertainty: f32,
    /// The ADR-299 domain-gate result. `Unknown` records an OOD inference.
    pub domain_state: DomainState,
}

impl ModelInference {
    /// Construct a model inference, validating text fields at the boundary.
    pub fn new(
        model_version: impl Into<String>,
        label: impl Into<String>,
        uncertainty: f32,
        domain_state: DomainState,
    ) -> Result<Self, InputError> {
        Ok(Self {
            model_version: checked_text(model_version.into())?,
            label: checked_text(label.into())?,
            uncertainty,
            domain_state,
        })
    }

    /// The model version string.
    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// The raw output label.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// The independent-corroboration stage (ADR-300). In phase 1 the honest value
/// is [`Corroboration::None`] — a missing corroboration is recorded, never
/// fabricated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Corroboration {
    /// No independent corroboration was available (phase-1 default).
    None,
    /// A second modality agreed, with an agreement score in `0.0..=1.0`.
    Agreed {
        /// The corroborating modality.
        modality: String,
        /// Agreement score.
        score: f32,
    },
    /// A second modality disagreed.
    Disagreed {
        /// The corroborating modality.
        modality: String,
        /// Agreement score.
        score: f32,
    },
}

impl Corroboration {
    /// Construct an `Agreed` corroboration, validating the modality length.
    pub fn agreed(modality: impl Into<String>, score: f32) -> Result<Self, InputError> {
        Ok(Corroboration::Agreed {
            modality: checked_text(modality.into())?,
            score,
        })
    }

    /// Construct a `Disagreed` corroboration, validating the modality length.
    pub fn disagreed(modality: impl Into<String>, score: f32) -> Result<Self, InputError> {
        Ok(Corroboration::Disagreed {
            modality: checked_text(modality.into())?,
            score,
        })
    }
}

/// The kind of ADR-303 ontology entity a stage updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SpatialEntity {
    /// A single observation.
    Observation = 0,
    /// A track (linked observations over time).
    Track = 1,
    /// A discrete event.
    Event = 2,
}

/// The spatial-state stage: the ADR-303 ontology entity the inference updated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialState {
    /// The kind of entity updated.
    pub entity: SpatialEntity,
    entity_id: String,
}

impl SpatialState {
    /// Construct a spatial-state record, validating the entity id length.
    pub fn new(entity: SpatialEntity, entity_id: impl Into<String>) -> Result<Self, InputError> {
        Ok(Self {
            entity,
            entity_id: checked_text(entity_id.into())?,
        })
    }

    /// The entity identifier.
    pub fn entity_id(&self) -> &str {
        &self.entity_id
    }
}

/// The governed action a policy took or withheld (ADR-318 owns the real one).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyAction {
    /// An action was taken.
    Act {
        /// The action identifier.
        action: String,
    },
    /// The action was withheld (e.g. on a degraded/unknown domain).
    Withhold {
        /// Why the action was withheld.
        reason: String,
    },
}

/// The terminal policy-decision stage: the governed action, with the ADR-315
/// capability certificate it relied on (if any).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyDecision {
    /// The governed action taken or withheld.
    pub action: PolicyAction,
    certificate_ref: Option<String>,
}

impl PolicyDecision {
    /// Record a taken action.
    pub fn act(
        action: impl Into<String>,
        certificate_ref: Option<String>,
    ) -> Result<Self, InputError> {
        Self::new(
            PolicyAction::Act {
                action: checked_text(action.into())?,
            },
            certificate_ref,
        )
    }

    /// Record a withheld action.
    pub fn withhold(
        reason: impl Into<String>,
        certificate_ref: Option<String>,
    ) -> Result<Self, InputError> {
        Self::new(
            PolicyAction::Withhold {
                reason: checked_text(reason.into())?,
            },
            certificate_ref,
        )
    }

    fn new(action: PolicyAction, certificate_ref: Option<String>) -> Result<Self, InputError> {
        let certificate_ref = match certificate_ref {
            Some(c) => Some(checked_text(c)?),
            None => None,
        };
        Ok(Self {
            action,
            certificate_ref,
        })
    }

    /// The relied-upon certificate reference, if any.
    pub fn certificate_ref(&self) -> Option<&str> {
        self.certificate_ref.as_deref()
    }
}

/// The typed evidence carried by a stage. Its variant fixes the stage kind, so
/// a stage can never carry evidence inconsistent with its position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StageEvidence {
    /// RF observation (root).
    RfObservation(RfObservation),
    /// DSP evidence.
    DspEvidence(DspEvidence),
    /// Model inference.
    ModelInference(ModelInference),
    /// Independent corroboration.
    IndependentCorroboration(Corroboration),
    /// Spatial state.
    SpatialState(SpatialState),
    /// Policy decision (terminal).
    PolicyDecision(PolicyDecision),
}

impl StageEvidence {
    /// The stage kind this evidence variant belongs to.
    pub fn kind(&self) -> StageKind {
        match self {
            StageEvidence::RfObservation(_) => StageKind::RfObservation,
            StageEvidence::DspEvidence(_) => StageKind::DspEvidence,
            StageEvidence::ModelInference(_) => StageKind::ModelInference,
            StageEvidence::IndependentCorroboration(_) => StageKind::IndependentCorroboration,
            StageEvidence::SpatialState(_) => StageKind::SpatialState,
            StageEvidence::PolicyDecision(_) => StageKind::PolicyDecision,
        }
    }

    fn write_canonical(&self, out: &mut Vec<u8>) {
        match self {
            StageEvidence::RfObservation(o) => {
                out.push(0);
                let m = &o.measurement;
                push_field(out, m.device.as_str().as_bytes());
                out.extend_from_slice(&m.sequence.to_le_bytes());
                out.extend_from_slice(&m.timestamp.0.to_le_bytes());
                push_field(out, &m.payload_hash.0);
                match &m.calibration_ref {
                    Some(c) => {
                        out.push(1);
                        push_field(out, c.as_str().as_bytes());
                    }
                    None => out.push(0),
                }
                out.push(o.source_state as u8);
            }
            StageEvidence::DspEvidence(d) => {
                out.push(1);
                push_field(out, d.descriptor.as_bytes());
                out.extend_from_slice(&d.snr_db.to_le_bytes());
                out.push(d.quality_ok as u8);
            }
            StageEvidence::ModelInference(m) => {
                out.push(2);
                push_field(out, m.model_version.as_bytes());
                push_field(out, m.label.as_bytes());
                out.extend_from_slice(&m.uncertainty.to_le_bytes());
                out.push(m.domain_state as u8);
            }
            StageEvidence::IndependentCorroboration(c) => {
                out.push(3);
                match c {
                    Corroboration::None => out.push(0),
                    Corroboration::Agreed { modality, score } => {
                        out.push(1);
                        push_field(out, modality.as_bytes());
                        out.extend_from_slice(&score.to_le_bytes());
                    }
                    Corroboration::Disagreed { modality, score } => {
                        out.push(2);
                        push_field(out, modality.as_bytes());
                        out.extend_from_slice(&score.to_le_bytes());
                    }
                }
            }
            StageEvidence::SpatialState(s) => {
                out.push(4);
                out.push(s.entity as u8);
                push_field(out, s.entity_id.as_bytes());
            }
            StageEvidence::PolicyDecision(p) => {
                out.push(5);
                match &p.action {
                    PolicyAction::Act { action } => {
                        out.push(0);
                        push_field(out, action.as_bytes());
                    }
                    PolicyAction::Withhold { reason } => {
                        out.push(1);
                        push_field(out, reason.as_bytes());
                    }
                }
                match &p.certificate_ref {
                    Some(c) => {
                        out.push(1);
                        push_field(out, c.as_bytes());
                    }
                    None => out.push(0),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The hash-linked stage
// ---------------------------------------------------------------------------

/// A stage hash: the BLAKE3 (via [`ruview_attest::PayloadHash`]) of a stage's
/// canonical bytes, which include the prior stage's hash. Fixed width so a
/// malformed wire value cannot force an unbounded allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageHash(pub [u8; HASH_LEN]);

/// One stage of a witness chain: a typed [`StageEvidence`] with its
/// [`Confidence`] and [`EvidenceLevel`], hash-linked to the prior stage.
///
/// Fields are readable for inspection but a `Stage` inside a [`WitnessChain`] is
/// only reachable immutably ([`WitnessChain::stages`]); the chain never hands
/// out a mutable stage, upholding the append-only invariant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stage {
    /// The stage kind (equals `evidence.kind()`).
    pub kind: StageKind,
    /// The stage's confidence (may be [`Confidence::Unknown`]).
    pub confidence: Confidence,
    /// The stage's provenance / evidence level.
    pub evidence_level: EvidenceLevel,
    /// The hash of the prior stage, or `None` for the root.
    pub prior_hash: Option<StageHash>,
    /// The typed evidence.
    pub evidence: StageEvidence,
}

impl Stage {
    fn new(
        evidence: StageEvidence,
        confidence: Confidence,
        evidence_level: EvidenceLevel,
        prior_hash: Option<StageHash>,
    ) -> Self {
        Self {
            kind: evidence.kind(),
            confidence,
            evidence_level,
            prior_hash,
            evidence,
        }
    }

    /// The deterministic hash of this stage over its canonical, length-prefixed
    /// bytes — including the prior-stage hash, which is what links the chain.
    pub fn hash(&self) -> StageHash {
        let mut b = Vec::with_capacity(DOMAIN.len() + 64);
        b.extend_from_slice(DOMAIN);
        b.push(self.kind as u8);
        match self.confidence {
            Confidence::Unknown => b.push(0),
            Confidence::Known(v) => {
                b.push(1);
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b.push(self.evidence_level as u8);
        match &self.prior_hash {
            Some(h) => {
                b.push(1);
                b.extend_from_slice(&h.0);
            }
            None => b.push(0),
        }
        self.evidence.write_canonical(&mut b);
        StageHash(PayloadHash::of(&b).0)
    }
}

// ---------------------------------------------------------------------------
// The chain
// ---------------------------------------------------------------------------

/// A verified summary of a chain, returned by [`WitnessChain::verify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainSummary {
    /// Number of stages.
    pub len: usize,
    /// The effective evidence level: the **minimum** across all stages.
    pub effective_level: EvidenceLevel,
    /// The kind of the terminal stage.
    pub terminal_kind: StageKind,
    /// The verified head hash.
    pub head: StageHash,
    /// Whether the terminal stage is a [`StageKind::PolicyDecision`].
    pub finalized: bool,
}

/// A staged, append-only, hash-linked witness chain (ADR-316).
///
/// A chain always roots in an RF observation and grows by strictly-increasing
/// stage kind. Prior stages are immutable: [`WitnessChain::append`] only ever
/// pushes, and the stored [`WitnessChain::head`] binds the terminal stage so
/// even a last-stage in-place mutation is caught by [`WitnessChain::verify`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessChain {
    /// When this chain is an append-only correction of a prior chain, the head
    /// hash of the chain it supersedes.
    correction_of: Option<StageHash>,
    stages: Vec<Stage>,
    head: StageHash,
}

impl WitnessChain {
    /// Root a chain in an RF observation.
    pub fn root(
        observation: RfObservation,
        confidence: Confidence,
        evidence_level: EvidenceLevel,
    ) -> Self {
        let stage = Stage::new(
            StageEvidence::RfObservation(observation),
            confidence,
            evidence_level,
            None,
        );
        let head = stage.hash();
        Self {
            correction_of: None,
            stages: vec![stage],
            head,
        }
    }

    /// Root a chain directly in an authenticated [`VerifiedMeasurement`]
    /// (ADR-302), so the chain descends from a verified chain of custody.
    pub fn from_measurement(
        measurement: VerifiedMeasurement,
        source_state: SourceState,
        confidence: Confidence,
        evidence_level: EvidenceLevel,
    ) -> Self {
        Self::root(
            RfObservation::from_verified(measurement, source_state),
            confidence,
            evidence_level,
        )
    }

    /// Begin an append-only *correction* of `prior`: a fresh chain that
    /// references the superseded chain's head rather than editing it in place.
    pub fn correcting(
        prior: &WitnessChain,
        observation: RfObservation,
        confidence: Confidence,
        evidence_level: EvidenceLevel,
    ) -> Self {
        let mut chain = Self::root(observation, confidence, evidence_level);
        chain.correction_of = Some(prior.head);
        chain
    }

    /// Append a stage. The stage kind (derived from `evidence`) must be strictly
    /// greater than the current terminal kind, so prior stages are never
    /// mutated or reordered. The new stage binds the current head hash.
    pub fn append(
        &mut self,
        evidence: StageEvidence,
        confidence: Confidence,
        evidence_level: EvidenceLevel,
    ) -> Result<(), ChainError> {
        let next = evidence.kind();
        let last = self.stages.last().map(|s| s.kind).ok_or(ChainError::Empty)?;
        if (next as u8) <= (last as u8) {
            return Err(ChainError::NotAppendable { last, next });
        }
        let stage = Stage::new(evidence, confidence, evidence_level, Some(self.head));
        self.head = stage.hash();
        self.stages.push(stage);
        Ok(())
    }

    /// The chain's stages, immutably. There is no mutable accessor: a chain is
    /// append-only.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// The recorded head (terminal-stage) hash.
    pub fn head(&self) -> StageHash {
        self.head
    }

    /// The head hash of a chain this one corrects, if any.
    pub fn correction_of(&self) -> Option<StageHash> {
        self.correction_of
    }

    /// Verify the whole chain: the root is an RF observation, every stage binds
    /// the recomputed hash of its predecessor, stage order strictly increases,
    /// and the terminal stage matches the recorded head. Returns a
    /// [`ChainSummary`] whose effective level is the minimum across stages.
    pub fn verify(&self) -> Result<ChainSummary, ChainError> {
        let first = self.stages.first().ok_or(ChainError::Empty)?;
        if first.kind != StageKind::RfObservation {
            return Err(ChainError::RootNotObservation(first.kind));
        }

        let mut prev_hash: Option<StageHash> = None;
        let mut prev_kind: Option<StageKind> = None;
        let mut effective = EvidenceLevel::L5;

        for (index, stage) in self.stages.iter().enumerate() {
            // Link integrity: the recorded prior hash must equal the recomputed
            // hash of the predecessor (None for the root).
            if stage.prior_hash != prev_hash {
                return Err(ChainError::BrokenLink { index });
            }
            // Monotonic stage order.
            if let Some(pk) = prev_kind {
                if (stage.kind as u8) <= (pk as u8) {
                    return Err(ChainError::NonMonotonicOrder { index });
                }
            }
            if stage.evidence_level < effective {
                effective = stage.evidence_level;
            }
            prev_hash = Some(stage.hash());
            prev_kind = Some(stage.kind);
        }

        // Terminal integrity: catches an in-place mutation of the last stage,
        // which no successor's `prior_hash` binds.
        let head = prev_hash.ok_or(ChainError::Empty)?;
        if head != self.head {
            return Err(ChainError::TerminalTampered);
        }

        let terminal_kind = prev_kind.ok_or(ChainError::Empty)?;
        Ok(ChainSummary {
            len: self.stages.len(),
            effective_level: effective,
            terminal_kind,
            head,
            finalized: terminal_kind == StageKind::PolicyDecision,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate a free-text field length at the boundary.
fn checked_text(text: String) -> Result<String, InputError> {
    if text.len() > MAX_TEXT_LEN {
        return Err(InputError::TextTooLong { got: text.len() });
    }
    Ok(text)
}

/// Append a `u32` little-endian length prefix followed by the bytes, so the
/// canonical encoding is field-unambiguous and serde-format independent.
fn push_field(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

// ---------------------------------------------------------------------------
// Tests (SYNTHETIC / L0 fixtures — constructed in code, no wall clock, no RNG)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_attest::{
        AttestationVerifier, Blake3MacSigner, FreshnessPolicy, SignedMeasurement, TAG_LEN,
    };

    fn verified(seq: u64) -> VerifiedMeasurement {
        VerifiedMeasurement {
            device: DeviceId::new("esp32-node-01").unwrap(),
            sequence: seq,
            timestamp: Timestamp(1000),
            payload_hash: PayloadHash::of(b"csi-frame"),
            calibration_ref: Some(CalibrationRef::new("cal-cert-abc").unwrap()),
        }
    }

    /// Build a full six-stage chain from an L-labelled root.
    fn full_chain(root_level: EvidenceLevel, source: SourceState) -> WitnessChain {
        let mut chain = WitnessChain::from_measurement(
            verified(1),
            source,
            Confidence::known(0.9).unwrap(),
            root_level,
        );
        chain
            .append(
                StageEvidence::DspEvidence(
                    DspEvidence::new("doppler-motion-band", 12.5, true).unwrap(),
                ),
                Confidence::known(0.8).unwrap(),
                EvidenceLevel::L3,
            )
            .unwrap();
        chain
            .append(
                StageEvidence::ModelInference(
                    ModelInference::new("presence-v3", "person", 0.15, DomainState::Known).unwrap(),
                ),
                Confidence::known(0.72).unwrap(),
                EvidenceLevel::L2,
            )
            .unwrap();
        chain
            .append(
                StageEvidence::IndependentCorroboration(Corroboration::None),
                Confidence::Unknown,
                EvidenceLevel::L1,
            )
            .unwrap();
        chain
            .append(
                StageEvidence::SpatialState(
                    SpatialState::new(SpatialEntity::Track, "track-7").unwrap(),
                ),
                Confidence::known(0.7).unwrap(),
                EvidenceLevel::L2,
            )
            .unwrap();
        chain
            .append(
                StageEvidence::PolicyDecision(
                    PolicyDecision::act("dim-lights", Some("cap-cert-9".into())).unwrap(),
                ),
                Confidence::known(0.7).unwrap(),
                EvidenceLevel::L2,
            )
            .unwrap();
        chain
    }

    #[test]
    fn full_chain_verifies() {
        let chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let summary = chain.verify().unwrap();
        assert_eq!(summary.len, 6);
        assert_eq!(summary.terminal_kind, StageKind::PolicyDecision);
        assert!(summary.finalized);
        // Effective level is the minimum across stages (L1 from the
        // no-corroboration stage), never the root's L3.
        assert_eq!(summary.effective_level, EvidenceLevel::L1);
    }

    #[test]
    fn stages_are_in_pipeline_order() {
        let chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let kinds: Vec<StageKind> = chain.stages().iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![
                StageKind::RfObservation,
                StageKind::DspEvidence,
                StageKind::ModelInference,
                StageKind::IndependentCorroboration,
                StageKind::SpatialState,
                StageKind::PolicyDecision,
            ]
        );
    }

    #[test]
    fn root_from_authenticated_measurement() {
        // Sign + verify with ruview-attest, then root the chain in the result.
        let key = [7u8; TAG_LEN];
        let signer = Blake3MacSigner::new(key);
        let device = DeviceId::new("esp32-node-01").unwrap();
        let mut verifier = AttestationVerifier::new(FreshnessPolicy::new(1_000_000_000, 100_000_000));
        verifier.enroll(device.clone(), signer.clone());
        let signed =
            SignedMeasurement::sign(&signer, device, 1, Timestamp(1000), b"csi-frame", None);
        let vm = verifier.verify(&signed, b"csi-frame", Timestamp(1000)).unwrap();

        let chain = WitnessChain::from_measurement(
            vm,
            SourceState::LiveVerified,
            Confidence::known(0.9).unwrap(),
            EvidenceLevel::L4,
        );
        assert!(chain.verify().is_ok());
        match &chain.stages()[0].evidence {
            StageEvidence::RfObservation(o) => {
                assert_eq!(o.measurement.sequence, 1);
                assert!(o.source_state.is_live_verified());
            }
            _ => panic!("root must be an RF observation"),
        }
    }

    #[test]
    fn synthetic_root_caps_effective_level_at_l0() {
        // A synthetic root labelled L0 caps the whole chain regardless of
        // later, higher-labelled stages.
        let chain = full_chain(EvidenceLevel::L0, SourceState::Synthetic);
        let summary = chain.verify().unwrap();
        assert_eq!(summary.effective_level, EvidenceLevel::L0);
        match &chain.stages()[0].evidence {
            StageEvidence::RfObservation(o) => {
                assert_eq!(o.source_state, SourceState::Synthetic);
                assert!(!o.source_state.is_live_verified());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn append_rejects_out_of_order_stage() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        // Terminal is PolicyDecision; nothing can follow.
        let err = chain
            .append(
                StageEvidence::DspEvidence(DspEvidence::new("x", 1.0, true).unwrap()),
                Confidence::Unknown,
                EvidenceLevel::L1,
            )
            .unwrap_err();
        assert!(matches!(err, ChainError::NotAppendable { .. }));

        // A second RF observation is also rejected (kind not strictly greater).
        let mut c2 = WitnessChain::from_measurement(
            verified(1),
            SourceState::LiveVerified,
            Confidence::Unknown,
            EvidenceLevel::L2,
        );
        assert!(matches!(
            c2.append(
                StageEvidence::RfObservation(RfObservation::from_verified(
                    verified(2),
                    SourceState::LiveVerified
                )),
                Confidence::Unknown,
                EvidenceLevel::L2,
            ),
            Err(ChainError::NotAppendable { .. })
        ));
    }

    #[test]
    fn tampered_middle_stage_fails() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        // Mutate a middle stage's confidence in place without re-linking.
        chain.stages[2].confidence = Confidence::known(0.01).unwrap();
        assert!(matches!(chain.verify(), Err(ChainError::BrokenLink { index: 3 })));
    }

    #[test]
    fn tampered_terminal_stage_fails() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let last = chain.stages.len() - 1;
        // No successor binds the terminal stage; the recorded head catches it.
        chain.stages[last].evidence_level = EvidenceLevel::L5;
        assert_eq!(chain.verify(), Err(ChainError::TerminalTampered));
    }

    #[test]
    fn tampered_evidence_content_fails() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        // Rewrite the model label on a middle stage.
        chain.stages[2].evidence = StageEvidence::ModelInference(
            ModelInference::new("presence-v3", "empty-room", 0.15, DomainState::Known).unwrap(),
        );
        assert!(matches!(chain.verify(), Err(ChainError::BrokenLink { index: 3 })));
    }

    #[test]
    fn reordered_stages_fail() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        chain.stages.swap(2, 3);
        // The swap breaks both the hash link and the monotonic order; either
        // way verification must reject it.
        assert!(chain.verify().is_err());
    }

    #[test]
    fn missing_link_fails() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        // Drop a non-root stage's prior-hash link.
        chain.stages[3].prior_hash = None;
        assert!(matches!(chain.verify(), Err(ChainError::BrokenLink { index: 3 })));
    }

    #[test]
    fn dropped_stage_fails() {
        let mut chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        // Remove a middle stage entirely: the follower's link no longer matches
        // and the stored head no longer matches the recomputed terminal.
        chain.stages.remove(3);
        assert!(chain.verify().is_err());
    }

    #[test]
    fn serde_round_trip_preserves_chain() {
        let chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let json = serde_json::to_string(&chain).unwrap();
        let back: WitnessChain = serde_json::from_str(&json).unwrap();
        assert_eq!(chain, back);
        // A deserialized chain still verifies end to end.
        assert!(back.verify().is_ok());
    }

    #[test]
    fn confidence_and_provenance_preserved() {
        let chain = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let json = serde_json::to_string(&chain).unwrap();
        let back: WitnessChain = serde_json::from_str(&json).unwrap();
        for (a, b) in chain.stages().iter().zip(back.stages()) {
            assert_eq!(a.confidence, b.confidence);
            assert_eq!(a.evidence_level, b.evidence_level);
            assert_eq!(a.kind, b.kind);
        }
        // The UNKNOWN corroboration confidence survives faithfully.
        assert_eq!(back.stages()[3].confidence, Confidence::Unknown);
        match &back.stages()[3].evidence {
            StageEvidence::IndependentCorroboration(c) => assert_eq!(*c, Corroboration::None),
            _ => panic!(),
        }
    }

    #[test]
    fn unknown_gate_recorded_faithfully() {
        // An OOD inference is carried as DomainState::Unknown with Unknown
        // confidence — never promoted to a confident class.
        let mut chain = WitnessChain::from_measurement(
            verified(1),
            SourceState::LiveVerified,
            Confidence::known(0.9).unwrap(),
            EvidenceLevel::L3,
        );
        chain
            .append(
                StageEvidence::DspEvidence(DspEvidence::new("band", 3.0, false).unwrap()),
                Confidence::Unknown,
                EvidenceLevel::L1,
            )
            .unwrap();
        chain
            .append(
                StageEvidence::ModelInference(
                    ModelInference::new("presence-v3", "unknown", 0.9, DomainState::Unknown)
                        .unwrap(),
                ),
                Confidence::Unknown,
                EvidenceLevel::L0,
            )
            .unwrap();
        let summary = chain.verify().unwrap();
        assert_eq!(summary.effective_level, EvidenceLevel::L0);
        match &chain.stages()[2].evidence {
            StageEvidence::ModelInference(m) => assert_eq!(m.domain_state, DomainState::Unknown),
            _ => panic!(),
        }
    }

    #[test]
    fn append_only_correction_references_prior() {
        let prior = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let correction = WitnessChain::correcting(
            &prior,
            RfObservation::from_verified(verified(2), SourceState::LiveVerified),
            Confidence::known(0.95).unwrap(),
            EvidenceLevel::L3,
        );
        // The correction is a new chain that references, not edits, the prior.
        assert_eq!(correction.correction_of(), Some(prior.head()));
        assert!(correction.verify().is_ok());
        assert!(prior.verify().is_ok());
        assert_ne!(correction.head(), prior.head());
    }

    #[test]
    fn chain_building_is_deterministic() {
        let a = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        let b = full_chain(EvidenceLevel::L3, SourceState::LiveVerified);
        assert_eq!(a, b);
        assert_eq!(a.head(), b.head());
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    #[test]
    fn confidence_boundary_validation() {
        assert!(Confidence::known(f32::NAN).is_err());
        assert!(Confidence::known(-0.1).is_err());
        assert!(Confidence::known(1.1).is_err());
        assert!(Confidence::known(0.0).is_ok());
        assert!(Confidence::known(1.0).is_ok());
    }

    #[test]
    fn text_boundary_validation() {
        let long = "x".repeat(MAX_TEXT_LEN + 1);
        assert!(matches!(
            DspEvidence::new(long, 1.0, true),
            Err(InputError::TextTooLong { .. })
        ));
        assert!(DspEvidence::new("x".repeat(MAX_TEXT_LEN), 1.0, true).is_ok());
    }

    #[test]
    fn empty_chain_verify_is_error_not_panic() {
        // Constructed only via serde to reach the empty-stages guard.
        let json = r#"{"correction_of":null,"stages":[],"head":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#;
        let chain: WitnessChain = serde_json::from_str(json).unwrap();
        assert_eq!(chain.verify(), Err(ChainError::Empty));
    }
}
