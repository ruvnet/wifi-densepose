//! Coarse, non-reversible appearance features (ADR-304 §3, privacy boundary).
//!
//! A [`CoarseFeature`] is the appearance channel used for short-horizon track
//! continuity (the ADR-303/ADR-304 `CsiFingerprint` analogue). Its type is the
//! privacy enforcement point:
//!
//! - **Coarse.** Raw values are quantized into a handful of buckets
//!   ([`COARSE_LEVELS`]), so fine structure that could serve as a biometric is
//!   discarded at construction.
//! - **Non-reversible.** Quantization is lossy and there is no de-quantizer:
//!   the original values cannot be recovered from a `CoarseFeature`.
//! - **Bounded.** Dimension is capped at [`MAX_FEATURE_DIM`], bounding
//!   allocation on untrusted input.
//! - **Carries no civil identifier.** The type holds only opaque bucket indices
//!   — no name, account, MAC, phone, or other join key exists in the schema.

use serde::{Deserialize, Serialize};

use crate::error::TrackError;

/// Maximum accepted coarse-feature dimension. Bounds allocation.
pub const MAX_FEATURE_DIM: usize = 16;

/// Number of coarse quantization buckets per component (a 3-bit coarse code).
/// Deliberately small so the feature is non-identifying.
pub const COARSE_LEVELS: u8 = 8;

/// A bounded, coarse, non-reversible appearance descriptor.
///
/// Construct via [`CoarseFeature::quantize`]. Two features are compared with an
/// L1 distance over aligned buckets; features of differing dimension are treated
/// as maximally distant (non-comparable) rather than panicking.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoarseFeature {
    /// Opaque coarse bucket indices, each in `0..COARSE_LEVELS`.
    bins: Vec<u8>,
}

impl CoarseFeature {
    /// Quantize raw components (each expected in `[0.0, 1.0]`, clamped
    /// otherwise) into coarse buckets.
    ///
    /// Rejects an empty or over-long vector at the boundary; never panics on
    /// NaN/inf (those clamp to the nearest bucket edge).
    pub fn quantize(raw: &[f64]) -> Result<Self, TrackError> {
        if raw.is_empty() {
            return Err(TrackError::EmptyFeature);
        }
        if raw.len() > MAX_FEATURE_DIM {
            return Err(TrackError::FeatureTooLarge {
                dim: raw.len(),
                max: MAX_FEATURE_DIM,
            });
        }
        let top = i64::from(COARSE_LEVELS) - 1;
        let bins = raw
            .iter()
            .map(|&v| {
                // NaN maps to 0 via the failed comparison in clamp guards below.
                let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
                let bucket = (c * f64::from(COARSE_LEVELS)).floor() as i64;
                bucket.clamp(0, top) as u8
            })
            .collect();
        Ok(Self { bins })
    }

    /// The number of coarse components.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.bins.len()
    }

    /// Borrow the opaque bucket indices (for tests / serialization checks).
    #[must_use]
    pub fn bins(&self) -> &[u8] {
        &self.bins
    }

    /// The maximum possible [`distance`](Self::distance) for this dimension —
    /// used to normalize the gate. Always finite.
    #[must_use]
    pub fn max_distance(&self) -> f64 {
        self.bins.len() as f64 * f64::from(COARSE_LEVELS - 1)
    }

    /// L1 distance over aligned buckets. Differing dimensions are non-comparable
    /// and return the larger side's maximum distance (treated as far apart) so a
    /// dimension mismatch can never masquerade as a close match.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        if self.bins.len() != other.bins.len() {
            return self.max_distance().max(other.max_distance());
        }
        self.bins
            .iter()
            .zip(&other.bins)
            .map(|(&a, &b)| f64::from(a.abs_diff(b)))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_is_coarse_and_bounded() {
        let f = CoarseFeature::quantize(&[0.0, 0.5, 1.0]).unwrap();
        assert_eq!(f.dim(), 3);
        // Every bucket is within the coarse range.
        assert!(f.bins().iter().all(|&b| b < COARSE_LEVELS));
        // 0.5 lands in the middle bucket, not at an extreme.
        assert_eq!(f.bins()[0], 0);
        assert_eq!(f.bins()[2], COARSE_LEVELS - 1);
    }

    #[test]
    fn quantize_is_lossy_non_reversible() {
        // Two nearby-but-distinct raw values collapse to the same bucket:
        // information is destroyed, so the original is unrecoverable.
        let a = CoarseFeature::quantize(&[0.01]).unwrap();
        let b = CoarseFeature::quantize(&[0.10]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_empty_and_overlong() {
        assert_eq!(CoarseFeature::quantize(&[]), Err(TrackError::EmptyFeature));
        let long = vec![0.5; MAX_FEATURE_DIM + 1];
        assert!(matches!(
            CoarseFeature::quantize(&long),
            Err(TrackError::FeatureTooLarge { .. })
        ));
    }

    #[test]
    fn nan_and_inf_do_not_panic() {
        let f = CoarseFeature::quantize(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
        assert_eq!(f.bins(), &[0, COARSE_LEVELS - 1, 0]);
    }

    #[test]
    fn distance_symmetric_and_mismatch_is_far() {
        let a = CoarseFeature::quantize(&[0.0, 0.0]).unwrap();
        let b = CoarseFeature::quantize(&[1.0, 1.0]).unwrap();
        assert_eq!(a.distance(&b), b.distance(&a));
        assert!(a.distance(&b) > 0.0);
        let c = CoarseFeature::quantize(&[0.0]).unwrap();
        assert!(a.distance(&c) >= a.max_distance());
    }
}
