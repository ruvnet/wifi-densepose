//! # `ruview-certify` — signed capability certificates (ADR-318, ADR-300 §1)
//!
//! A [`CapabilityCertificate`] is a bounded, signed attestation that a specific
//! capability (e.g. presence, pose) has been *validated for a specific
//! environment*, for a *bounded* time. RuView must stop making unconditional
//! capability claims: "supports presence" is not a true statement — presence
//! works in some rooms, on some hardware, for some subject dynamics, and fails
//! on a stationary subject at range in an uncalibrated room. The honest unit of
//! the claim is a signed, expiring certificate, never a feature flag.
//!
//! ## What the certificate binds
//!
//! - the **capability** ([`Capability`]);
//! - the **room** ([`SpaceId`], ADR-306) plus the **calibration-certificate
//!   version** (ADR-301) it was validated against;
//! - the **hardware** ([`DeviceId`], ADR-305);
//! - the scored **model** version;
//! - the **calibrated date** the calibration was captured;
//! - the operating **metrics** (`moving_recall`, `stationary_recall`,
//!   `false_presence_per_24h`) sliced from the ADR-304 ledger for **exactly this
//!   context** (never pooled across contexts);
//! - a `valid_until` expiry that is **never open-ended** and **cannot outlive the
//!   calibration validity**;
//! - exactly one [`EvidenceLevel`] (ADR-282) that **cannot exceed the evidence
//!   slice's floor** — a certificate never upgrades the ledger it is minted from;
//! - a **signature** over the canonical serialization ([`ruview_attest`]); an
//!   unsigned certificate is not a valid certificate.
//!
//! ## Honest by construction (ADR-300 rule)
//!
//! - Minting from a slice that reports **no evidence** yields no certificate —
//!   absence of evidence is never a capability.
//! - The evidence level is the ledger floor, never an upgrade.
//! - A certificate minted from a synthetic ledger slice is `L0`/SYNTHETIC by
//!   construction; nothing here invents a MEASURED number.
//! - [`CapabilityCertificate::is_valid`] is *conditional on the live domain
//!   signature* (ADR-302): a certificate over a `DEGRADED`/`UNKNOWN` domain is
//!   not valid, and an expired certificate is not valid — the honest failure is
//!   UNKNOWN, not a best-effort guess.
//!
//! Time is always injected (no wall clock); no randomness; malformed input is a
//! returned error, never a panic; allocation is bounded at every boundary.

#![forbid(unsafe_code)]

use ruview_attest::{DeviceId, Signature, Signer, Verifier};
use ruview_evidence::{EvidenceLevel, EvidenceSlice, SummaryEvidence};
use ruview_ontology::SpaceId;
use serde::{Deserialize, Serialize};
use wifi_densepose_calibration::CalibrationCertificate;

/// Maximum accepted byte length for the model-version identifier. Bounds
/// allocation at the untrusted-input boundary (CLAUDE.md).
pub const MAX_MODEL_LEN: usize = 256;

/// Domain-separation tag for the canonical signing bytes. Distinguishes a
/// capability-certificate signature from any other signed object in the system.
const DOMAIN: &[u8] = b"ruview-certify/CapabilityCertificate/v1";

// ---------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------

/// The phenomenon a certificate is about. A device may only be certified for a
/// capability it is attested to sense (ADR-305/ADR-141); the attestation gate is
/// a phase-2 concern — this phase binds the capability into the signed object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Capability {
    /// Presence / occupancy detection.
    Presence,
    /// Body-pose (DensePose) estimation.
    Pose,
}

impl Capability {
    /// Stable byte tag used inside the canonical serialization. Never `0`, so a
    /// field boundary can never be confused with an absent value.
    const fn tag(self) -> u8 {
        match self {
            Capability::Presence => 1,
            Capability::Pose => 2,
        }
    }
}

/// The live domain-state signature a consumer supplies at validation time
/// (ADR-302). Only `Known` permits a capability; `Degraded`/`Unknown` gate the
/// certificate to invalid — the honest failure is UNKNOWN, not a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DomainState {
    /// The domain is characterized and within its calibration envelope.
    Known,
    /// The domain has drifted or is degraded — no capability.
    Degraded,
    /// The domain is uncharacterized / unknown — no capability.
    Unknown,
}

impl DomainState {
    /// Whether the live domain permits consuming a capability.
    #[must_use]
    pub fn is_known(self) -> bool {
        matches!(self, DomainState::Known)
    }
}

/// The adapter boundary from the real `ruview-ood` gate result to this
/// crate's three-way domain signature (ADR-297/302). `ruview_ood::DomainState`
/// carries a `DomainCause` on `Degraded`/`Unknown`; `is_valid` only needs the
/// three-way outcome, so the cause is dropped here — this is the conversion
/// this crate's own `DomainState` doc comment already described but that
/// previously did not exist anywhere, leaving `ruview-ood`'s live drift result
/// with no path into a certificate check.
impl From<ruview_ood::DomainState> for DomainState {
    fn from(state: ruview_ood::DomainState) -> Self {
        match state {
            ruview_ood::DomainState::Known => DomainState::Known,
            ruview_ood::DomainState::Degraded(_) => DomainState::Degraded,
            ruview_ood::DomainState::Unknown(_) => DomainState::Unknown,
        }
    }
}

#[cfg(test)]
mod domain_state_adapter_tests {
    //! Pins the `ruview-ood` -> `ruview-certify` adapter boundary: a real OOD
    //! gate result must map to the matching certify-side outcome, and — the
    //! load-bearing case — a post-drift `Unknown` from `ood` must actually
    //! invalidate an otherwise-valid certificate through this conversion,
    //! not just when a test hand-constructs `certify::DomainState::Unknown`
    //! directly.
    use super::DomainState;

    #[test]
    fn known_maps_to_known() {
        assert_eq!(DomainState::from(ruview_ood::DomainState::Known), DomainState::Known);
    }

    #[test]
    fn degraded_drops_the_cause_and_maps_to_degraded() {
        assert_eq!(
            DomainState::from(ruview_ood::DomainState::Degraded(
                ruview_ood::DomainCause::ModerateDrift
            )),
            DomainState::Degraded
        );
    }

    #[test]
    fn unknown_drops_the_cause_and_maps_to_unknown() {
        assert_eq!(
            DomainState::from(ruview_ood::DomainState::Unknown(
                ruview_ood::DomainCause::DriftBeyondEnvelope
            )),
            DomainState::Unknown
        );
    }
}

/// The operating metrics frozen onto a certificate, sliced from the ADR-304
/// ledger for one exact context (never a global average).
///
/// `false_presence_per_24h` carries the ledger's context false-positive rate;
/// no per-24h count is invented here — the value is the number the ledger
/// reports for this context, relabelled to the certificate's operating vocab.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatingMetrics {
    /// Recall on moving subjects, `[0, 1]`.
    pub moving_recall: f64,
    /// Recall on stationary subjects, `[0, 1]`.
    pub stationary_recall: f64,
    /// False-presence operating metric (ledger context false-positive rate).
    pub false_presence_per_24h: f64,
}

/// The unsigned content a signature binds: everything a verifier must
/// reconstruct byte-for-byte to check the tag.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CertificateContent {
    /// The certified phenomenon.
    pub capability: Capability,
    /// The room this claim is validated for (ADR-306).
    pub room: SpaceId,
    /// Version of the calibration certificate the validation ran against
    /// (ADR-301). The certificate cannot outlive this calibration.
    pub calibration_version: u64,
    /// Expiry of the calibration certificate (unix seconds); the ceiling on
    /// `valid_until`.
    pub calibration_expires_at_unix_s: i64,
    /// The authenticated device the claim is validated for (ADR-305).
    pub hardware: DeviceId,
    /// The scored model version.
    pub model_version: String,
    /// Capture time of the calibration certificate (unix seconds).
    pub calibrated_date_unix_s: i64,
    /// Operating metrics, sliced from the ledger for this exact context.
    pub metrics: OperatingMetrics,
    /// Explicit expiry (unix seconds); never open-ended, never past the
    /// calibration expiry.
    pub valid_until_unix_s: i64,
    /// Exactly one evidence level; the ledger floor, never an upgrade.
    pub evidence_level: EvidenceLevel,
}

impl CertificateContent {
    /// Deterministic, length-prefixed canonical serialization used as the
    /// signing input. Length prefixes make the encoding unambiguous (no field
    /// can be confused with another) and independent of any serde format, so
    /// two byte-identical contents always sign identically.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            DOMAIN.len() + 128 + self.room.as_str().len() + self.hardware.as_str().len(),
        );
        out.extend_from_slice(DOMAIN);
        out.push(self.capability.tag());
        push_field(&mut out, self.room.as_str().as_bytes());
        out.extend_from_slice(&self.calibration_version.to_le_bytes());
        out.extend_from_slice(&self.calibration_expires_at_unix_s.to_le_bytes());
        push_field(&mut out, self.hardware.as_str().as_bytes());
        push_field(&mut out, self.model_version.as_bytes());
        out.extend_from_slice(&self.calibrated_date_unix_s.to_le_bytes());
        out.extend_from_slice(&self.metrics.moving_recall.to_bits().to_le_bytes());
        out.extend_from_slice(&self.metrics.stationary_recall.to_bits().to_le_bytes());
        out.extend_from_slice(&self.metrics.false_presence_per_24h.to_bits().to_le_bytes());
        out.extend_from_slice(&self.valid_until_unix_s.to_le_bytes());
        out.push(level_byte(self.evidence_level));
        out
    }
}

/// A signed capability certificate: the [`CertificateContent`] together with a
/// signature over its canonical bytes. `signature` is [`None`] for an unsigned
/// certificate, which is never valid (ADR-318 §1).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilityCertificate {
    /// The signed content.
    pub content: CertificateContent,
    /// Tag over [`CertificateContent::canonical_bytes`]; `None` means unsigned.
    pub signature: Option<Signature>,
}

impl CapabilityCertificate {
    /// Wrap content as an **unsigned** certificate. Useful for tests and for
    /// staging content before signing; [`Self::verify`] and [`Self::is_valid`]
    /// both reject it because an unsigned certificate is not a valid
    /// certificate (ADR-318 §1).
    #[must_use]
    pub fn unsigned(content: CertificateContent) -> Self {
        Self {
            content,
            signature: None,
        }
    }

    /// Verify the signature over the canonical bytes. Returns `false` for an
    /// unsigned certificate or a tampered one. This is the cryptographic check;
    /// [`Self::is_valid`] adds the expiry and live-domain gates.
    #[must_use]
    pub fn verify<V: Verifier + ?Sized>(&self, verifier: &V) -> bool {
        match &self.signature {
            Some(sig) => verifier.verify(&self.content.canonical_bytes(), sig),
            None => false,
        }
    }

    /// The consumer gate (ADR-318 §3, ADR-300/ADR-302): the certificate is valid
    /// **iff** it is signed, it has not expired (`now < valid_until`), and the
    /// live domain state is `Known`. A `Degraded`/`Unknown` domain or an expired
    /// or unsigned certificate resolves to *not valid* — the honest UNKNOWN,
    /// never a best-effort guess. This is the crypto-independent gate; call
    /// [`Self::verify`] with the enrolled key for the signature check.
    #[must_use]
    pub fn is_valid(&self, now_unix_s: i64, domain: DomainState) -> bool {
        self.signature.is_some()
            && domain.is_known()
            && now_unix_s < self.content.valid_until_unix_s
    }
}

// ---------------------------------------------------------------------------
// Minting
// ---------------------------------------------------------------------------

/// The inputs to [`mint`], other than the signer and the evidence slice. Owned
/// so the minted certificate freezes its own copy of every bound field.
#[derive(Clone, Debug)]
pub struct MintRequest<'c> {
    /// The phenomenon being certified.
    pub capability: Capability,
    /// The room the claim is validated for.
    pub room: SpaceId,
    /// The authenticated device the claim is validated for.
    pub hardware: DeviceId,
    /// The scored model version.
    pub model_version: String,
    /// The calibration certificate the validation ran against; supplies the
    /// version, calibrated date, and the expiry ceiling.
    pub calibration: &'c CalibrationCertificate,
    /// Requested expiry (unix seconds); must not exceed the calibration expiry.
    pub valid_until_unix_s: i64,
    /// Requested evidence level; must not exceed the slice floor.
    pub evidence_level: EvidenceLevel,
}

/// Mint a signed [`CapabilityCertificate`] from an ADR-304 evidence slice for
/// one `(room, device, model)` context.
///
/// Minting is a pure function over the slice: the metrics are frozen into the
/// signed object. It refuses to issue a certificate unless every honesty
/// invariant holds.
///
/// # Errors
/// - [`CertifyError::NoEvidence`] — the slice reports no evidence for the
///   context; absence of evidence is never a capability.
/// - [`CertifyError::ContextMismatch`] — the slice's context does not match the
///   bound room/hardware/model, so the metrics would not describe the claim.
/// - [`CertifyError::CalibrationRoomMismatch`] — the calibration certificate is
///   for a different room than the claim.
/// - [`CertifyError::EvidenceLevelUpgrade`] — the requested level exceeds the
///   ledger floor (no upgrade).
/// - [`CertifyError::OutlivesCalibration`] — `valid_until` is past the
///   calibration expiry; a certificate cannot outlive its calibration.
/// - [`CertifyError::ModelTooLong`] — the model version exceeds [`MAX_MODEL_LEN`].
pub fn mint<S: Signer + ?Sized>(
    signer: &S,
    request: MintRequest<'_>,
    slice: &EvidenceSlice<'_>,
) -> Result<CapabilityCertificate, CertifyError> {
    // Bound untrusted input at the boundary.
    if request.model_version.len() > MAX_MODEL_LEN {
        return Err(CertifyError::ModelTooLong {
            len: request.model_version.len(),
            max: MAX_MODEL_LEN,
        });
    }

    // Absence of evidence is never a capability (ADR-318 §2).
    let summary = slice.summarize();
    let (floor, agg) = match summary.evidence {
        SummaryEvidence::NoEvidence => return Err(CertifyError::NoEvidence),
        SummaryEvidence::Aggregated { level, metrics, .. } => (level, metrics),
    };

    // The metrics must describe *this* context, or the claim is unbacked.
    let ctx = slice.context();
    if ctx.room != request.room.as_str() {
        return Err(CertifyError::ContextMismatch { field: "room" });
    }
    if ctx.device != request.hardware.as_str() {
        return Err(CertifyError::ContextMismatch { field: "device" });
    }
    if ctx.model_version != request.model_version {
        return Err(CertifyError::ContextMismatch {
            field: "model_version",
        });
    }

    // The calibration certificate must be for the same room as the claim.
    if request.calibration.space_id != request.room.as_str() {
        return Err(CertifyError::CalibrationRoomMismatch);
    }

    // Evidence level is inherited from the ledger and can never be upgraded.
    if request.evidence_level > floor {
        return Err(CertifyError::EvidenceLevelUpgrade {
            requested: request.evidence_level,
            floor,
        });
    }

    // A certificate can never outlive the calibration it was validated against.
    let calibration_expires_at_unix_s = request.calibration.expires_at_unix_s;
    if request.valid_until_unix_s > calibration_expires_at_unix_s {
        return Err(CertifyError::OutlivesCalibration {
            valid_until_unix_s: request.valid_until_unix_s,
            calibration_expires_at_unix_s,
        });
    }

    let content = CertificateContent {
        capability: request.capability,
        room: request.room,
        calibration_version: request.calibration.version,
        calibration_expires_at_unix_s,
        hardware: request.hardware,
        model_version: request.model_version,
        calibrated_date_unix_s: request.calibration.captured_at_unix_s,
        metrics: OperatingMetrics {
            moving_recall: agg.moving_recall,
            stationary_recall: agg.stationary_recall,
            false_presence_per_24h: agg.false_positive_rate,
        },
        valid_until_unix_s: request.valid_until_unix_s,
        evidence_level: request.evidence_level,
    };

    let signature = signer.sign(&content.canonical_bytes());
    Ok(CapabilityCertificate {
        content,
        signature: Some(signature),
    })
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors raised at the minting boundary. No variant panics; a malformed or
/// dishonest request is always a returned error (CLAUDE.md).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CertifyError {
    /// The evidence slice reports no evidence for the context — no capability.
    #[error("no evidence for the context; a certificate cannot be minted")]
    NoEvidence,
    /// The slice's context does not match a bound field.
    #[error("evidence slice context field `{field}` does not match the bound claim")]
    ContextMismatch {
        /// The mismatched field name.
        field: &'static str,
    },
    /// The calibration certificate is for a different room than the claim.
    #[error("calibration certificate room does not match the certified room")]
    CalibrationRoomMismatch,
    /// The requested evidence level exceeds the ledger floor (no upgrade).
    #[error("requested evidence level {requested:?} exceeds ledger floor {floor:?}")]
    EvidenceLevelUpgrade {
        /// The requested (too-high) level.
        requested: EvidenceLevel,
        /// The ledger floor that caps it.
        floor: EvidenceLevel,
    },
    /// `valid_until` is past the calibration expiry.
    #[error(
        "valid_until {valid_until_unix_s} outlives calibration expiry \
         {calibration_expires_at_unix_s}"
    )]
    OutlivesCalibration {
        /// The requested expiry.
        valid_until_unix_s: i64,
        /// The calibration ceiling it exceeded.
        calibration_expires_at_unix_s: i64,
    },
    /// The model version exceeded [`MAX_MODEL_LEN`].
    #[error("model version is {len} bytes, exceeds max {max}")]
    ModelTooLong {
        /// The offending length.
        len: usize,
        /// The maximum accepted length.
        max: usize,
    },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Length-prefixed field push (8-byte LE length + bytes) for unambiguous
/// canonical encoding.
fn push_field(out: &mut Vec<u8>, field: &[u8]) {
    out.extend_from_slice(&(field.len() as u64).to_le_bytes());
    out.extend_from_slice(field);
}

/// Stable byte for an evidence level, ordered `L0 < … < L5`.
fn level_byte(level: EvidenceLevel) -> u8 {
    match level {
        EvidenceLevel::L0 => 0,
        EvidenceLevel::L1 => 1,
        EvidenceLevel::L2 => 2,
        EvidenceLevel::L3 => 3,
        EvidenceLevel::L4 => 4,
        EvidenceLevel::L5 => 5,
    }
}

#[cfg(test)]
mod tests;
