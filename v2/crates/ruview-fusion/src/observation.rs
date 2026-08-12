//! The fusion input: a canonical HAL observation plus its presence claim
//! (ADR-311 §1).
//!
//! Fusion consumes authenticated, ontology-typed observations. A
//! [`HalObservation`] carries the modality, evidence level, sensor identity,
//! container, and provenance (the canonical ADR-306 vocabulary, reused rather
//! than reinvented — ADR-300 rule 3); a [`PresenceObservation`] pairs it with
//! that sensor's [`Claim`] about whether its container is occupied. Keeping the
//! claim separate from the HAL frame lets the engine gate on the observation's
//! own health: a malformed / degraded HAL observation abstains no matter what
//! number it reports, and a source that cannot quantify presence says
//! [`Claim::Unknown`] rather than defaulting to a confident value (ADR-300
//! rule 1).

use serde::{Deserialize, Serialize};

use ruview_hal::HalObservation;

use crate::estimate::Estimate;

/// A single sensor's occupancy claim for the container its observation is in.
///
/// UNKNOWN is a first-class value here, never an error: a source may quantify
/// its belief ([`Claim::Estimated`]) or explicitly abstain ([`Claim::Unknown`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "claim", rename_all = "snake_case")]
pub enum Claim {
    /// A quantified presence claim: probability of occupancy and its variance.
    Estimated {
        /// Probability of occupancy in `[0.0, 1.0]`.
        probability: f64,
        /// Variance of the estimate; strictly positive.
        variance: f64,
    },
    /// The source abstains — it contributes provenance but no numeric estimate.
    Unknown,
}

impl Claim {
    /// Construct a quantified claim, validating the numbers at the boundary. A
    /// non-finite value or a non-positive variance cannot form a weightable
    /// estimate, so the claim degrades to [`Claim::Unknown`] rather than
    /// erroring or poisoning the fusion; the probability is clamped to
    /// `[0.0, 1.0]`.
    #[must_use]
    pub fn estimated(probability: f64, variance: f64) -> Self {
        match Estimate::new(probability, variance) {
            Some(e) => Self::Estimated {
                probability: e.probability,
                variance: e.variance,
            },
            None => Self::Unknown,
        }
    }

    /// The weightable [`Estimate`] this claim carries, or `None` when it
    /// abstains or its numbers are not weightable.
    #[must_use]
    pub fn estimate(&self) -> Option<Estimate> {
        match *self {
            Self::Estimated {
                probability,
                variance,
            } => Estimate::new(probability, variance),
            Self::Unknown => None,
        }
    }
}

/// One fusion input: a canonical HAL observation and its presence claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PresenceObservation {
    /// The canonical HAL observation (modality, evidence, sensor, container,
    /// provenance, and per-observation uncertainty).
    pub hal: HalObservation,
    /// This sensor's occupancy claim for its container.
    pub claim: Claim,
}

impl PresenceObservation {
    /// Build a fusion input with a quantified claim (validated; see
    /// [`Claim::estimated`]).
    #[must_use]
    pub fn estimated(hal: HalObservation, probability: f64, variance: f64) -> Self {
        Self {
            claim: Claim::estimated(probability, variance),
            hal,
        }
    }

    /// Build a fusion input whose source abstains ([`Claim::Unknown`]).
    #[must_use]
    pub fn unknown(hal: HalObservation) -> Self {
        Self {
            hal,
            claim: Claim::Unknown,
        }
    }

    /// The usable presence estimate this observation contributes, or `None` when
    /// it abstains.
    ///
    /// An observation abstains when its HAL frame is `degraded` (malformed /
    /// out-of-bounds raw input — the HAL already flagged it UNKNOWN) or when its
    /// claim is not a weightable estimate. Abstaining observations still carry
    /// their provenance into the fused state; they simply do not move the fused
    /// value. A merely *high-variance* claim is **not** abstaining — it
    /// contributes, but is down-weighted by inverse-variance.
    #[must_use]
    pub fn usable_estimate(&self) -> Option<Estimate> {
        if self.hal.uncertainty.degraded {
            return None;
        }
        self.claim.estimate()
    }
}
