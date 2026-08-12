//! The inference gate (ADR-302 §2, ADR-300 rule 1).
//!
//! Every inference passes through the gate. It:
//!
//! 1. classifies the domain from distance + envelope + signal quality +
//!    calibration compatibility;
//! 2. attaches the model's own predictive **uncertainty** (the fourth ADR-302
//!    input), escalating a KNOWN domain to DEGRADED when uncertainty is
//!    elevated;
//! 3. **suppresses the confident class** when the domain is not KNOWN — an
//!    UNKNOWN domain returns no class, a first-class value rather than a
//!    confidently-wrong label (ADR-300 rule 1);
//! 4. emits a [`RecalibrationRequest`] whenever the state is DEGRADED or
//!    UNKNOWN — a *signal*, never an action; recalibration itself is out of
//!    scope for this crate (ADR-300 staleness guard).

use serde::{Deserialize, Serialize};
use wifi_densepose_calibration::certificate::{CompatibilityEnvelope, FingerprintDistance};

use crate::domain::{clamp_unit, CalibrationCompat, DomainCause, DomainState, DomainThresholds, SignalQuality};
use crate::error::{require_unit_interval, Result};

/// A model head's proposed inference, before gating. `class` is the model's
/// candidate label of any type; `confidence`/`uncertainty` are its own scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Inference<C> {
    /// The model's candidate class/label.
    pub class: C,
    /// The model's reported confidence in `[0, 1]` (sanitized at the gate).
    pub confidence: f32,
    /// The model's predictive uncertainty in `[0, 1]` (sanitized at the gate).
    pub uncertainty: f32,
}

impl<C> Inference<C> {
    /// Construct an inference. Confidence/uncertainty are stored as given and
    /// sanitized (clamped, NaN → worst) when the gate consumes them, so a
    /// hostile model score cannot escape `[0, 1]` downstream.
    pub fn new(class: C, confidence: f32, uncertainty: f32) -> Self {
        Self {
            class,
            confidence,
            uncertainty,
        }
    }
}

/// How urgently recalibration is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecalibrationUrgency {
    /// DEGRADED: recommended — the domain still supports flagged inferences.
    Recommended,
    /// UNKNOWN: required — confident inference is suspended until re-cal.
    Required,
}

/// A signal that recalibration should be triggered (ADR-302 §2 / ADR-300
/// staleness guard). This crate **emits** the request; it never performs
/// recalibration (that is ADR-301's job). Carries the triggering cause so the
/// caller can route it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecalibrationRequest {
    /// Why recalibration is being requested.
    pub reason: DomainCause,
    /// How urgent the request is.
    pub urgency: RecalibrationUrgency,
}

/// The fully-contextualized result of gating one inference. Carries the domain
/// state, all four input measurements, and either a (flagged) class or none —
/// so downstream consumers (ADR-304 evidence engine) get the whole decision,
/// never a bare label.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatedInference<C> {
    /// The domain state (KNOWN / DEGRADED / UNKNOWN + cause).
    pub state: DomainState,
    /// Live-vs-certified domain distance (ADR-302 input 1).
    pub distance: FingerprintDistance,
    /// Signal quality (ADR-302 input 2).
    pub signal_quality: SignalQuality,
    /// Calibration compatibility (ADR-302 input 3).
    pub calibration_compat: CalibrationCompat,
    /// Model predictive uncertainty, sanitized to `[0, 1]` (ADR-302 input 4).
    pub uncertainty: f32,
    /// The returned class. `None` in UNKNOWN — the confident label is
    /// suppressed (ADR-300 rule 1). `Some` in KNOWN and DEGRADED (flagged).
    pub class: Option<C>,
    /// Sanitized confidence, present iff a class is returned.
    pub confidence: Option<f32>,
    /// A recalibration signal, present iff the state is DEGRADED or UNKNOWN.
    pub recalibration: Option<RecalibrationRequest>,
}

impl<C> GatedInference<C> {
    /// `true` iff a confident class survived the gate (only in KNOWN).
    pub fn is_confident(&self) -> bool {
        self.state.is_known() && self.class.is_some()
    }
}

/// The shared OOD gate every inference routes through (ADR-302 §2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InferenceGate {
    thresholds: DomainThresholds,
    /// Max uncertainty tolerated while KNOWN; above it, a KNOWN domain is
    /// escalated to DEGRADED (the fourth ADR-302 input acting on the state).
    max_uncertainty_known: f32,
}

impl Default for InferenceGate {
    fn default() -> Self {
        Self {
            thresholds: DomainThresholds::default(),
            max_uncertainty_known: 0.5,
        }
    }
}

impl InferenceGate {
    /// Validated constructor. `max_uncertainty_known` must be finite in `[0, 1]`.
    pub fn new(thresholds: DomainThresholds, max_uncertainty_known: f32) -> Result<Self> {
        let max_uncertainty_known = require_unit_interval("max_uncertainty_known", max_uncertainty_known)?;
        Ok(Self {
            thresholds,
            max_uncertainty_known,
        })
    }

    /// The thresholds in effect (reported with each decision per ADR-302 §2).
    pub fn thresholds(&self) -> DomainThresholds {
        self.thresholds
    }

    /// Gate one inference. Pure: deterministic in its inputs, no clock, no
    /// randomness, bounded allocation. Consumes `inference` (the class is moved
    /// into the result or dropped when suppressed).
    ///
    /// Behavior:
    /// - KNOWN → class + confidence returned, no recalibration signal;
    /// - DEGRADED → class + confidence returned **flagged**, recalibration
    ///   *recommended*;
    /// - UNKNOWN → class suppressed (`None`), recalibration *required*.
    pub fn evaluate<C>(
        &self,
        inference: Inference<C>,
        distance: FingerprintDistance,
        envelope: CompatibilityEnvelope,
        signal_quality: SignalQuality,
        calibration_compat: CalibrationCompat,
    ) -> GatedInference<C> {
        let uncertainty = clamp_unit(inference.uncertainty);

        let mut state = self
            .thresholds
            .classify(distance, envelope, signal_quality, calibration_compat);

        // Fourth input: elevated uncertainty escalates a KNOWN domain to
        // DEGRADED. It never *upgrades* a state — worst signal always wins.
        if state.is_known() && uncertainty > self.max_uncertainty_known {
            state = DomainState::Degraded(DomainCause::ElevatedUncertainty);
        }

        let confidence = clamp_unit(inference.confidence);
        let (class, confidence, recalibration) = match state {
            DomainState::Known => (Some(inference.class), Some(confidence), None),
            DomainState::Degraded(reason) => (
                Some(inference.class),
                Some(confidence),
                Some(RecalibrationRequest {
                    reason,
                    urgency: RecalibrationUrgency::Recommended,
                }),
            ),
            // ADR-300 rule 1: no confident class in UNKNOWN. The class is
            // dropped, not returned with lowered confidence.
            DomainState::Unknown(reason) => (
                None,
                None,
                Some(RecalibrationRequest {
                    reason,
                    urgency: RecalibrationUrgency::Required,
                }),
            ),
        };

        GatedInference {
            state,
            distance,
            signal_quality,
            calibration_compat,
            uncertainty,
            class,
            confidence,
            recalibration,
        }
    }
}
