//! Candidate sensor actions the scheduler ranks (ADR-311 §1).
//!
//! **SYNTHETIC / L0 scaffold (ADR-282).** An [`ExpectedReduction`] is a *model
//! prediction* of how much a not-yet-taken measurement would shrink the fused
//! covariance — in a fielded system it comes from the ADR-312 RF-twin forward
//! model evaluated against the ADR-308 covariance. It is never a measured
//! quantity: a value-of-information estimate made *before* paying for the
//! measurement. No accuracy claim is made.

use serde::{Deserialize, Serialize};

use ruview_hal::Modality;
use ruview_ontology::SensorId;

use crate::cost::Cost;

/// The predicted uncertainty reduction of taking one candidate measurement,
/// with UNKNOWN as a first-class value (ADR-297 rule 1).
///
/// A candidate whose informativeness the forward model cannot predict is
/// [`ExpectedReduction::Unknown`] — it is **not** silently treated as zero. The
/// scheduler's [`UnknownPolicy`](crate::UnknownPolicy) decides whether such a
/// candidate is probed (to *learn* its informativeness) or deferred; either way
/// the choice is explicit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedReduction {
    /// A predicted, non-negative uncertainty reduction on the ADR-299 objective.
    Known(f64),
    /// The forward model cannot predict this candidate's informativeness.
    Unknown,
}

impl ExpectedReduction {
    /// Construct a [`Known`](ExpectedReduction::Known) reduction from a raw
    /// prediction, sanitizing at the boundary: a non-finite prediction becomes
    /// [`Unknown`](ExpectedReduction::Unknown) (honest, per rule 1), and a
    /// negative prediction — uncertainty cannot be *increased* by sampling — is
    /// clamped to `0.0`.
    #[must_use]
    pub fn known(raw: f64) -> Self {
        if !raw.is_finite() {
            Self::Unknown
        } else {
            Self::Known(raw.max(0.0))
        }
    }

    /// The predicted reduction if known, else `None`.
    #[must_use]
    pub fn value(&self) -> Option<f64> {
        match self {
            Self::Known(v) => Some(*v),
            Self::Unknown => None,
        }
    }

    /// True when the informativeness is unknown.
    #[must_use]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }
}

/// One candidate sensor action the scheduler may spend budget on.
///
/// It names the radio/[`Modality`] to sample, the modelled
/// [`ExpectedReduction`] of doing so, and the [`Cost`] triple it would consume.
/// [`cycles_since_sampled`](SensorAction::cycles_since_sampled) is caller-
/// supplied staleness that feeds the sampling floor — the scheduler is a pure
/// function of its inputs and holds no cross-cycle state of its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorAction {
    /// The authenticated sensor identity (ADR-302) this action would sample.
    pub sensor: SensorId,
    /// The sensing modality of that sensor (ADR-317).
    pub modality: Modality,
    /// Modelled expected uncertainty reduction of taking the measurement.
    pub expected_reduction: ExpectedReduction,
    /// Modelled cost triple the action would consume.
    pub cost: Cost,
    /// Cycles since this sensor was last sampled, supplied by the caller. Feeds
    /// the sampling floor so a currently-low-value sensor is not starved into
    /// permanent blindness (ADR-311 §2). `0` means "sampled last cycle".
    #[serde(default)]
    pub cycles_since_sampled: u32,
}

impl SensorAction {
    /// Convenience constructor with `cycles_since_sampled = 0`.
    #[must_use]
    pub fn new(
        sensor: SensorId,
        modality: Modality,
        expected_reduction: ExpectedReduction,
        cost: Cost,
    ) -> Self {
        Self {
            sensor,
            modality,
            expected_reduction,
            cost,
            cycles_since_sampled: 0,
        }
    }
}
