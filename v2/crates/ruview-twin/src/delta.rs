//! The load-bearing operation: `delta(observed, expected)` (ADR-315 §2).
//!
//! **SYNTHETIC / L0.** A [`TwinDelta`] is a *model-relative* statement: how far a
//! supplied observation set sits from the twin's own predicted distributions,
//! measured against the twin's own modelled variance. It is **not** a detection,
//! and asserts no accuracy (ADR-315 evidence discipline, ADR-300). A change that
//! is large relative to the modelled variance is a *candidate physical change*
//! to be corroborated, never a confident claim.
//!
//! Consistent with ADR-300 rule 1, a link the twin cannot evaluate — unknown
//! prediction, or an observation for a link outside the twin — is reported as
//! [`LinkDeltaStatus::Unknown`], excluded from the aggregate magnitude, never an
//! error.

use serde::{Deserialize, Serialize};

use crate::predict::{predict_link, ExpectedDistribution, UnknownReason};
use crate::twin::{LinkId, RfTwin};

/// Default significance threshold (standard deviations). A link whose absolute
/// deviation exceeds this many modelled standard deviations is flagged as
/// deviating. `3.0` ≈ a conventional 3-sigma gate; it is a model gate, not a
/// calibrated false-alarm rate.
pub const DEFAULT_SIGNIFICANCE_THRESHOLD: f64 = 3.0;

/// One observed observable value for a link (e.g. a measured mean RSSI, dBm).
/// The value is caller-supplied; this crate never samples a clock or sensor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkObservation {
    /// The link this observation is for.
    pub link: LinkId,
    /// The observed value, in the observable's units (dBm for RSSI).
    pub value: f64,
}

/// A supplied set of link observations to compare against the twin.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ObservationSet {
    /// The observations. Order is not significant; duplicate links use the first.
    pub observations: Vec<LinkObservation>,
}

impl ObservationSet {
    /// An empty observation set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one observation and return `self` for chaining.
    #[must_use]
    pub fn with(mut self, link: LinkId, value: f64) -> Self {
        self.observations.push(LinkObservation { link, value });
        self
    }

    /// Build the observation set that exactly reproduces a twin's own predicted
    /// means — the *zero-delta* reference. Links the twin cannot predict are
    /// omitted (they would only surface as UNKNOWN).
    #[must_use]
    pub fn from_twin_prediction(twin: &RfTwin) -> Self {
        let mut observations = Vec::new();
        for link in twin.links() {
            if let ExpectedDistribution::Known { mean, .. } = predict_link(twin, &link) {
                observations.push(LinkObservation { link, value: mean });
            }
        }
        Self { observations }
    }
}

/// Outcome for a single link in a delta computation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum LinkDeltaStatus {
    /// The link was evaluated against a known distribution.
    Evaluated {
        /// Supplied observed value.
        observed: f64,
        /// Twin's modelled mean.
        expected_mean: f64,
        /// Twin's modelled variance (dB²).
        expected_variance: f64,
        /// `observed - expected_mean`, signed.
        deviation: f64,
        /// `|deviation| / sqrt(variance)`, i.e. standard deviations. `None` when
        /// variance is zero (significance is undefined, reported as UNKNOWN-ish
        /// rather than infinite).
        #[serde(skip_serializing_if = "Option::is_none")]
        significance: Option<f64>,
    },
    /// The link could not be evaluated; first-class UNKNOWN (ADR-300 rule 1).
    Unknown {
        /// Why it is unknown.
        reason: UnknownReason,
    },
}

/// The delta for one link.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkDelta {
    /// The link.
    pub link: LinkId,
    /// Its outcome.
    pub status: LinkDeltaStatus,
}

/// The typed result of `delta(observed, expected)` over an observation set.
///
/// **SYNTHETIC / L0.** `total_magnitude` and `deviating_links` are model-relative
/// summaries, not a detection or accuracy claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwinDelta {
    /// Twin baseline version this delta is relative to (ADR-315 §2).
    pub baseline_version: u64,
    /// Significance threshold (standard deviations) used to flag deviating links.
    pub significance_threshold: f64,
    /// L2 norm of evaluated per-link deviations — the overall magnitude of the
    /// change against the twin. `0.0` exactly when every evaluated link matches
    /// its prediction.
    pub total_magnitude: f64,
    /// Largest per-link significance among evaluated links (`0.0` if none).
    pub max_significance: f64,
    /// Links whose significance meets or exceeds the threshold — *which* links
    /// deviate.
    pub deviating_links: Vec<LinkId>,
    /// Links present in the observation set but not evaluable (UNKNOWN).
    pub unknown_links: Vec<LinkId>,
    /// Per-link detail for every observation supplied.
    pub per_link: Vec<LinkDelta>,
}

impl TwinDelta {
    /// True when no evaluated link deviates beyond the threshold. UNKNOWN links
    /// do not count as changes (they are reported separately).
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.deviating_links.is_empty() && self.total_magnitude == 0.0
    }
}

/// Compute the delta of a supplied observation set against the twin's predicted
/// distributions, using [`DEFAULT_SIGNIFICANCE_THRESHOLD`].
#[must_use]
pub fn compute_delta(twin: &RfTwin, observed: &ObservationSet) -> TwinDelta {
    compute_delta_with_threshold(twin, observed, DEFAULT_SIGNIFICANCE_THRESHOLD)
}

/// Compute the delta with an explicit significance threshold (standard
/// deviations). Deterministic and allocation-bounded by the observation count.
#[must_use]
pub fn compute_delta_with_threshold(
    twin: &RfTwin,
    observed: &ObservationSet,
    significance_threshold: f64,
) -> TwinDelta {
    let mut per_link = Vec::with_capacity(observed.observations.len());
    let mut deviating_links = Vec::new();
    let mut unknown_links = Vec::new();
    let mut sum_sq = 0.0_f64;
    let mut max_significance = 0.0_f64;

    for obs in &observed.observations {
        match predict_link(twin, &obs.link) {
            ExpectedDistribution::Known {
                mean, variance, ..
            } => {
                let deviation = obs.value - mean;
                let significance = if variance > 0.0 {
                    let sig = deviation.abs() / variance.sqrt();
                    if sig > max_significance {
                        max_significance = sig;
                    }
                    if sig >= significance_threshold {
                        deviating_links.push(obs.link.clone());
                    }
                    Some(sig)
                } else {
                    // Zero modelled variance: significance is undefined. A
                    // non-zero deviation still contributes to magnitude, but we
                    // do not fabricate an infinite significance.
                    None
                };
                sum_sq += deviation * deviation;
                per_link.push(LinkDelta {
                    link: obs.link.clone(),
                    status: LinkDeltaStatus::Evaluated {
                        observed: obs.value,
                        expected_mean: mean,
                        expected_variance: variance,
                        deviation,
                        significance,
                    },
                });
            }
            ExpectedDistribution::Unknown { reason } => {
                unknown_links.push(obs.link.clone());
                per_link.push(LinkDelta {
                    link: obs.link.clone(),
                    status: LinkDeltaStatus::Unknown { reason },
                });
            }
        }
    }

    TwinDelta {
        baseline_version: twin.version,
        significance_threshold,
        total_magnitude: sum_sq.sqrt(),
        max_significance,
        deviating_links,
        unknown_links,
        per_link,
    }
}
