//! The domain-state machine: KNOWN → DEGRADED → UNKNOWN.
//!
//! Implements the ADR-300 staleness guard `VALID → DEGRADED → UNKNOWN` as a
//! **pure** classification over four measured inputs (ADR-302 §1):
//!
//! 1. **domain distance** — [`FingerprintDistance`] of the live fingerprint vs
//!    the certified one (ADR-301 `distance()`);
//! 2. **signal quality** — [`SignalQuality`] (ADR-137 coherence/contradiction
//!    plus per-frame validity);
//! 3. **calibration compatibility** — [`CalibrationCompat`]: is a valid,
//!    non-invalidated, device/space-matched certificate present?
//!
//! (The model's own predictive **uncertainty** — the fourth ADR-302 input — is
//! attached and acted on at the [`crate::InferenceGate`], keeping `classify`'s
//! signature exactly the three-plus-envelope form the phase-1 spec pins.)
//!
//! The transition is monotone escalation (worst signal wins) so a degraded
//! room can never be reported as KNOWN, and hysteresis is provided by keeping
//! the inner (enter-DEGRADED) and outer (enter-UNKNOWN) thresholds distinct so
//! the gate does not flap on drift noise straddling a single line.

use serde::{Deserialize, Serialize};
use wifi_densepose_calibration::certificate::{CompatibilityEnvelope, FingerprintDistance, RoomFingerprint};

use crate::error::{require_unit_interval, Result};

/// The domain-distance primitive (ADR-302 §1): drift of the **live** room
/// fingerprint away from the **certified** reference distribution.
///
/// Reuses the calibration crate's [`FingerprintDistance`] (ADR-301), which
/// already splits drift into an empty-baseline (geometry) component and an
/// occupancy component, so a consumer can distinguish "the room itself changed"
/// from "occupancy statistics changed". This is a thin, documented adapter — no
/// second distance definition is introduced.
///
/// `certified` is the certificate's attested fingerprint; `live` is the
/// currently observed one.
pub fn domain_distance(certified: &RoomFingerprint, live: &RoomFingerprint) -> FingerprintDistance {
    certified.distance(live)
}

/// The specific reason a domain left KNOWN. Always reported alongside the state
/// (ADR-302: "never a bare label").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainCause {
    // --- UNKNOWN-grade causes (hard) ---
    /// No calibration certificate is present for this space/device.
    NoCertificate,
    /// The certificate is past its expiry (stale — ADR-300 staleness guard).
    CertificateExpired,
    /// The certificate's signature did not verify (tamper).
    CertificateTampered,
    /// The certificate was minted by a different signed device (ADR-305).
    DeviceMismatch,
    /// The certificate attests a different space (ADR-306).
    SpaceMismatch,
    /// Empty-baseline / total drift crossed the **outer** envelope threshold —
    /// the room changed materially (furniture, AP channel, geometry).
    DriftBeyondEnvelope,
    /// Signal quality fell below the usability floor — nothing can be trusted.
    SignalUnusable,

    // --- DEGRADED-grade causes (soft) ---
    /// Moderate drift: past the **inner** threshold but within the envelope.
    ModerateDrift,
    /// An ADR-137 contradiction flag was raised (tolerated, but lower-evidence).
    Contradiction,
    /// Signal quality dipped below the KNOWN threshold but above the floor.
    LowSignalQuality,
    /// The model's own predictive uncertainty is elevated (attached at the gate).
    ElevatedUncertainty,
}

impl DomainCause {
    /// A stable machine-readable slug for evidence records (ADR-304).
    pub fn as_str(self) -> &'static str {
        match self {
            DomainCause::NoCertificate => "no_certificate",
            DomainCause::CertificateExpired => "certificate_expired",
            DomainCause::CertificateTampered => "certificate_tampered",
            DomainCause::DeviceMismatch => "device_mismatch",
            DomainCause::SpaceMismatch => "space_mismatch",
            DomainCause::DriftBeyondEnvelope => "drift_beyond_envelope",
            DomainCause::SignalUnusable => "signal_unusable",
            DomainCause::ModerateDrift => "moderate_drift",
            DomainCause::Contradiction => "contradiction",
            DomainCause::LowSignalQuality => "low_signal_quality",
            DomainCause::ElevatedUncertainty => "elevated_uncertainty",
        }
    }
}

/// The gate's decision for one inference (ADR-302 §2).
///
/// `DEGRADED` and `UNKNOWN` always carry the triggering [`DomainCause`]; a bare
/// state is never produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainState {
    /// In-distribution: drift within the envelope, quality high, certificate
    /// valid & compatible. Confident classifications may be returned.
    Known,
    /// A soft threshold was crossed. Classifications are still returned but must
    /// be treated as lower-evidence; carries the specific cause.
    Degraded(DomainCause),
    /// The room changed materially or calibration is absent/stale. RuView stops
    /// returning confident classifications. This is required behavior, not an
    /// error (ADR-300 rule 1).
    Unknown(DomainCause),
}

impl DomainState {
    /// `true` only for [`DomainState::Known`].
    pub fn is_known(self) -> bool {
        matches!(self, DomainState::Known)
    }

    /// `true` for [`DomainState::Unknown`].
    pub fn is_unknown(self) -> bool {
        matches!(self, DomainState::Unknown(_))
    }

    /// `true` for [`DomainState::Degraded`].
    pub fn is_degraded(self) -> bool {
        matches!(self, DomainState::Degraded(_))
    }

    /// The triggering cause, if the domain is not KNOWN.
    pub fn cause(self) -> Option<DomainCause> {
        match self {
            DomainState::Known => None,
            DomainState::Degraded(c) | DomainState::Unknown(c) => Some(c),
        }
    }

    /// Pure classification with the default thresholds (ADR-302 §2). This is the
    /// canonical `classify(distance, envelope, signal_quality, calibration_compat)`
    /// entry point: it takes only measured inputs and returns a state — no clock,
    /// no randomness, no allocation.
    pub fn classify(
        distance: FingerprintDistance,
        envelope: CompatibilityEnvelope,
        signal_quality: SignalQuality,
        calibration_compat: CalibrationCompat,
    ) -> DomainState {
        DomainThresholds::default().classify(distance, envelope, signal_quality, calibration_compat)
    }
}

/// Per-frame signal-quality summary (ADR-137 reuse + per-frame validity).
///
/// `score` folds fusion coherence and per-frame SNR/validity into a single
/// `[0, 1]` health value; `contradiction` mirrors the ADR-137 contradiction
/// flag; `valid` is the per-frame validity bit. Constructed through a validated
/// boundary so a non-finite or out-of-range score can never enter the gate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Combined coherence/SNR health in `[0, 1]` (higher is better).
    pub score: f32,
    /// ADR-137 contradiction flag for this frame.
    pub contradiction: bool,
    /// Per-frame validity bit (a structurally invalid frame is unusable).
    pub valid: bool,
}

impl SignalQuality {
    /// Validated constructor. Rejects a non-finite or out-of-`[0, 1]` score
    /// (bounded-input discipline at the fusion boundary).
    pub fn new(score: f32, contradiction: bool, valid: bool) -> Result<Self> {
        let score = require_unit_interval("signal_quality.score", score)?;
        Ok(Self {
            score,
            contradiction,
            valid,
        })
    }

    /// Derive a quality score from raw ADR-137 signals. `coherence` is clamped
    /// to `[0, 1]`; `snr_db` is mapped through a bounded, monotone squash so a
    /// hostile/NaN SNR cannot poison the score. Never fails — a wholly invalid
    /// input yields a zero score and `valid = false`.
    pub fn from_signals(coherence: f32, snr_db: f32, contradiction: bool, valid: bool) -> Self {
        let coherence = clamp_unit(coherence);
        // Map SNR (dB) into [0, 1]: <=0 dB -> 0, >=30 dB -> 1, linear between.
        let snr_norm = if snr_db.is_finite() {
            (snr_db / 30.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let score = 0.5 * coherence + 0.5 * snr_norm;
        Self {
            score,
            contradiction,
            valid,
        }
    }
}

/// Whether a valid, non-invalidated calibration certificate is present for this
/// space and signed device (ADR-302 §1 input 3). Derived from an ADR-301
/// [`CertificateStatus`](wifi_densepose_calibration::certificate::CertificateStatus)
/// plus space/device identity checks; see [`crate::assess_certificate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationCompat {
    /// A valid certificate, matching space and device, drift within envelope.
    Valid,
    /// Certificate present but drifted beyond its envelope (stale distribution).
    DriftedBeyondEnvelope,
    /// Certificate present but expired.
    Expired,
    /// Certificate signature did not verify.
    Tampered,
    /// Certificate was minted by a different signed device.
    DeviceMismatch,
    /// Certificate attests a different space.
    SpaceMismatch,
    /// No certificate at all for this space/device.
    Absent,
}

impl CalibrationCompat {
    /// `true` only when a fully valid, compatible certificate is present.
    pub fn is_compatible(self) -> bool {
        matches!(self, CalibrationCompat::Valid)
    }

    /// The hard (UNKNOWN-grade) cause this compatibility state implies, if any.
    /// A non-`Valid` compatibility is always a hard failure: a stale, absent,
    /// or mismatched certificate cannot support a KNOWN domain (ADR-302 §3,
    /// "absence of evidence is absence of capability").
    fn hard_cause(self) -> Option<DomainCause> {
        match self {
            CalibrationCompat::Valid => None,
            CalibrationCompat::DriftedBeyondEnvelope => Some(DomainCause::DriftBeyondEnvelope),
            CalibrationCompat::Expired => Some(DomainCause::CertificateExpired),
            CalibrationCompat::Tampered => Some(DomainCause::CertificateTampered),
            CalibrationCompat::DeviceMismatch => Some(DomainCause::DeviceMismatch),
            CalibrationCompat::SpaceMismatch => Some(DomainCause::SpaceMismatch),
            CalibrationCompat::Absent => Some(DomainCause::NoCertificate),
        }
    }
}

/// The gate's calibration thresholds (ADR-302 §2). These are the "calibration
/// parameters, reported with each decision" the ADR requires — not baked-in
/// magic numbers. All are validated at construction.
///
/// Hysteresis is expressed as the gap between the inner (enter-DEGRADED) and
/// outer (enter-UNKNOWN) drift lines: `inner = envelope.max_total_drift *
/// inner_drift_fraction`, strictly below the outer envelope, so drift noise
/// straddling one line cannot flap KNOWN⇄UNKNOWN directly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DomainThresholds {
    /// Fraction of the envelope's `max_total_drift` at which drift enters
    /// DEGRADED. In `[0, 1)` so the inner line stays strictly inside the outer.
    pub inner_drift_fraction: f32,
    /// Minimum signal-quality score to remain KNOWN. Below it (but at/above the
    /// floor) → DEGRADED.
    pub quality_known_min: f32,
    /// Usability floor. Below it the frame is unusable → UNKNOWN.
    pub quality_floor: f32,
}

impl Default for DomainThresholds {
    fn default() -> Self {
        // Conservative phase-1 defaults; consumers tune per space/model.
        Self {
            inner_drift_fraction: 0.6,
            quality_known_min: 0.6,
            quality_floor: 0.3,
        }
    }
}

impl DomainThresholds {
    /// Validated constructor. Enforces `0 <= floor <= known_min <= 1`, and
    /// `inner_drift_fraction` in `[0, 1)`, so the inner drift line is always
    /// strictly below the outer envelope (bounded-input discipline).
    pub fn new(inner_drift_fraction: f32, quality_known_min: f32, quality_floor: f32) -> Result<Self> {
        if !inner_drift_fraction.is_finite() || !(0.0..1.0).contains(&inner_drift_fraction) {
            return Err(crate::error::OodError::InvalidParameter {
                field: "inner_drift_fraction",
                reason: format!("must be finite in [0, 1), got {inner_drift_fraction}"),
            });
        }
        let quality_known_min = require_unit_interval("quality_known_min", quality_known_min)?;
        let quality_floor = require_unit_interval("quality_floor", quality_floor)?;
        if quality_floor > quality_known_min {
            return Err(crate::error::OodError::InvalidParameter {
                field: "quality_floor",
                reason: format!(
                    "floor {quality_floor} must not exceed known_min {quality_known_min}"
                ),
            });
        }
        Ok(Self {
            inner_drift_fraction,
            quality_known_min,
            quality_floor,
        })
    }

    /// Pure classification (ADR-300 staleness guard `VALID → DEGRADED →
    /// UNKNOWN`). Monotone escalation: the first matching hard cause wins
    /// UNKNOWN; otherwise the first matching soft cause wins DEGRADED; else
    /// KNOWN. Deterministic, allocation-free, no clock.
    pub fn classify(
        self,
        distance: FingerprintDistance,
        envelope: CompatibilityEnvelope,
        signal_quality: SignalQuality,
        calibration_compat: CalibrationCompat,
    ) -> DomainState {
        // --- Hard failures → UNKNOWN (checked first; certificate before drift) ---
        if let Some(cause) = calibration_compat.hard_cause() {
            return DomainState::Unknown(cause);
        }
        let outer = envelope.max_total_drift;
        // A non-finite live distance is treated as maximal drift, never a panic.
        if !distance.total.is_finite() || distance.total > outer {
            return DomainState::Unknown(DomainCause::DriftBeyondEnvelope);
        }
        if !signal_quality.valid || signal_quality.score < self.quality_floor {
            return DomainState::Unknown(DomainCause::SignalUnusable);
        }

        // --- Soft failures → DEGRADED (drift first, then quality signals) ---
        let inner = outer * self.inner_drift_fraction;
        if distance.total > inner {
            return DomainState::Degraded(DomainCause::ModerateDrift);
        }
        if signal_quality.contradiction {
            return DomainState::Degraded(DomainCause::Contradiction);
        }
        if signal_quality.score < self.quality_known_min {
            return DomainState::Degraded(DomainCause::LowSignalQuality);
        }

        DomainState::Known
    }
}

/// Clamp into `[0, 1]`, mapping non-finite to `0.0` (worst). Shared helper so no
/// untrusted float can escape the unit interval without panicking.
pub(crate) fn clamp_unit(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
