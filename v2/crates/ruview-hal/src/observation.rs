//! The HAL observation (ADR-320 §2): a canonical observation plus HAL context.
//!
//! [`HalObservation`] wraps the canonical ontology
//! [`Observation`](ruview_ontology::Observation) — reusing it rather than
//! reinventing a per-crate shape (ADR-300 rule 3) — and adds the two pieces the
//! HAL boundary contributes: the [`Modality`] the measurement came through and
//! a per-observation [`Uncertainty`]. The ontology `Observation` already
//! carries the mandatory `EvidenceLevel` and `SemanticProvenance`, so those
//! travel with the fact and are surfaced here by delegating accessors — never
//! duplicated or allowed to diverge.

use serde::{Deserialize, Serialize};

use ruview_ontology::{EvidenceLevel, Observation, SemanticProvenance, SensorId};

use crate::modality::Modality;

/// A confidence value that is either a bounded scalar or first-class UNKNOWN.
///
/// UNKNOWN is a value, never an error (ADR-300 rule 1): a source that cannot
/// quantify its confidence says so and stays legible rather than defaulting to
/// a confident number.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// No confidence can be assigned.
    Unknown,
    /// A confidence in the closed unit interval `[0.0, 1.0]`.
    Known(f64),
}

impl Confidence {
    /// Construct a `Known` confidence, clamping into `[0.0, 1.0]`. A non-finite
    /// input (NaN/inf) collapses to [`Confidence::Unknown`] rather than
    /// propagating a poisoned value.
    #[must_use]
    pub fn known(value: f64) -> Self {
        if value.is_finite() {
            Self::Known(value.clamp(0.0, 1.0))
        } else {
            Self::Unknown
        }
    }

    /// True when this is [`Confidence::Unknown`].
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// Per-observation uncertainty carried alongside the canonical observation.
///
/// `degraded` distinguishes a *legitimately* unquantifiable source (`degraded
/// = false`) from one whose raw input was malformed and yielded a best-effort
/// UNKNOWN placeholder (`degraded = true`). Neither is an error.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Uncertainty {
    /// The confidence, or UNKNOWN.
    pub confidence: Confidence,
    /// True when the observation is an UNKNOWN placeholder produced from
    /// malformed / out-of-bounds raw input rather than a real measurement.
    pub degraded: bool,
}

impl Uncertainty {
    /// A first-class UNKNOWN with a bounded, non-degraded source (e.g. an
    /// event-driven sensor that simply does not quantify confidence).
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            confidence: Confidence::Unknown,
            degraded: false,
        }
    }

    /// An UNKNOWN produced because the raw input was malformed or exceeded the
    /// adapter's bounds. Flagged `degraded` so downstream fusion can weight or
    /// drop it, but still a well-formed observation, not a panic or error.
    #[must_use]
    pub fn degraded() -> Self {
        Self {
            confidence: Confidence::Unknown,
            degraded: true,
        }
    }

    /// A quantified uncertainty from a valid sample.
    #[must_use]
    pub fn known(confidence: f64) -> Self {
        Self {
            confidence: Confidence::known(confidence),
            degraded: false,
        }
    }

    /// True when the confidence is UNKNOWN (for any reason).
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        self.confidence.is_unknown()
    }
}

/// A canonical observation as it crosses the HAL boundary.
///
/// The inner [`Observation`] is the single downstream representation; `modality`
/// and `uncertainty` are the HAL's added context. A camera-derived and a
/// CSI-derived `HalObservation` are the same type with different provenance and
/// evidence — neither is lifted to the other's grade (CLAUDE.md: never present
/// WiFi sensing as camera-grade).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HalObservation {
    /// The modality this measurement was sensed through.
    pub modality: Modality,
    /// Per-observation uncertainty (possibly UNKNOWN).
    pub uncertainty: Uncertainty,
    /// The canonical ontology observation this HAL sample maps onto.
    pub observation: Observation,
}

impl HalObservation {
    /// The evidence level of the underlying observation (ADR-282).
    #[must_use]
    pub fn evidence_level(&self) -> EvidenceLevel {
        self.observation.evidence_level
    }

    /// The provenance of the underlying observation.
    #[must_use]
    pub fn provenance(&self) -> &SemanticProvenance {
        &self.observation.provenance
    }

    /// The authenticated sensor identity that produced this observation.
    #[must_use]
    pub fn sensor(&self) -> &SensorId {
        &self.observation.sensor
    }

    /// True when this observation carries UNKNOWN uncertainty.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        self.uncertainty.is_unknown()
    }

    /// True when this observation was produced by a synthetic source, marked by
    /// its provenance calibration handle. A synthetic observation can never
    /// alias to a measured/calibrated one (ADR-279 invariant 6): the reference
    /// adapters always emit [`EvidenceLevel::L0`] with a synthetic calibration
    /// handle, which cannot reach the corroborated/calibrated levels
    /// (`>= L2`).
    #[must_use]
    pub fn is_synthetic(&self) -> bool {
        self.observation.provenance.calibration_version == crate::SYNTHETIC_CALIBRATION
    }
}
