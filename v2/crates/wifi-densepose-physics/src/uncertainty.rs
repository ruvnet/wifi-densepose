//! Covariance-derived movement weights.

use wifi_densepose_core::SymmetricCovariance3;

/// Larger uncertainty permits more motion, bounded away from zero and one.
#[must_use]
pub fn movement_weight(covariance: SymmetricCovariance3) -> f32 {
    (covariance.trace() / 0.03).clamp(0.05, 0.95)
}
