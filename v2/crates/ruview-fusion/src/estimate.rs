//! The presence estimate and its uncertainty-aware combination (ADR-311 §2).
//!
//! An [`Estimate`] is a single sensor's belief about zone occupancy expressed as
//! a probability with a variance. Estimates combine by **inverse-variance
//! weighting** — the standard optimal linear combination of independent
//! Gaussian estimates (equivalently a product of Gaussians / a static Kalman
//! update): a low-variance (confident) estimate dominates and a high-variance
//! (uncertain) one is down-weighted, and combining agreeing estimates *lowers*
//! the fused variance (the belief sharpens). This is deliberately **not** a
//! naive mean, which would ignore how certain each source is and could never
//! sharpen (ADR-311: "uncertainty-weighted ... not a silently averaged value").

use serde::{Deserialize, Serialize};

/// A presence estimate: the probability of occupancy and its variance.
///
/// `probability` is bounded to `[0.0, 1.0]` and `variance` is strictly
/// positive and finite — both enforced by [`Estimate::new`], so a malformed
/// estimate can never enter the fusion arithmetic (it is rejected at the
/// boundary and the source abstains instead).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    /// Probability of occupancy, in the closed unit interval `[0.0, 1.0]`.
    pub probability: f64,
    /// Variance of the estimate; strictly positive. Smaller ⇒ more certain.
    pub variance: f64,
}

impl Estimate {
    /// Construct a validated estimate, or `None` when the inputs cannot form a
    /// weightable estimate (non-finite value, or variance `<= 0`). A NaN/inf
    /// probability or a zero/negative variance is rejected rather than
    /// propagated as a poisoned weight; the probability is clamped into
    /// `[0.0, 1.0]`.
    #[must_use]
    pub fn new(probability: f64, variance: f64) -> Option<Self> {
        if !probability.is_finite() || !variance.is_finite() || variance <= 0.0 {
            return None;
        }
        Some(Self {
            probability: probability.clamp(0.0, 1.0),
            variance,
        })
    }

    /// The precision (inverse variance) — the weight this estimate carries in an
    /// inverse-variance combination.
    #[must_use]
    pub fn precision(&self) -> f64 {
        1.0 / self.variance
    }
}

/// The result of inverse-variance combination over a non-empty set of
/// estimates: the fused mean/variance plus the reduced chi-square disagreement
/// statistic used to detect irreconcilable conflict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Combined {
    /// The inverse-variance-weighted mean probability, clamped to `[0.0, 1.0]`.
    pub probability: f64,
    /// The fused variance `1 / Σ precision` — never larger than the smallest
    /// contributing variance, so agreeing estimates sharpen the belief.
    pub variance: f64,
    /// Reduced chi-square `Σ wᵢ (pᵢ − mean)² / dof` (dof = `n − 1`, floored at
    /// 1). Near 0 when sources agree relative to their stated uncertainty;
    /// large when they disagree by more than that uncertainty allows.
    pub reduced_chi_square: f64,
}

/// Combine independent presence estimates by inverse-variance weighting.
///
/// Returns `None` for an empty input (there is nothing to fuse — the caller
/// resolves that to UNKNOWN / insufficient coverage). For a single estimate the
/// fused mean and variance are that estimate's own (pass-through) and the
/// disagreement statistic is 0.
#[must_use]
pub fn combine(estimates: &[Estimate]) -> Option<Combined> {
    if estimates.is_empty() {
        return None;
    }
    let precision_sum: f64 = estimates.iter().map(Estimate::precision).sum();
    // precision_sum is strictly positive because every Estimate has variance > 0.
    let mean = estimates
        .iter()
        .map(|e| e.probability * e.precision())
        .sum::<f64>()
        / precision_sum;
    let variance = 1.0 / precision_sum;
    let chi_square: f64 = estimates
        .iter()
        .map(|e| e.precision() * (e.probability - mean).powi(2))
        .sum();
    let dof = (estimates.len() - 1).max(1) as f64;
    Some(Combined {
        probability: mean.clamp(0.0, 1.0),
        variance,
        reduced_chi_square: chi_square / dof,
    })
}
