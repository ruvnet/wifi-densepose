//! Boundary errors for the OOD gate.
//!
//! Errors are raised only when *configuration* input is malformed (a threshold
//! outside its valid range, a non-finite quality score). Runtime domain
//! ambiguity is **never** an error: it is the first-class [`DomainState::Unknown`]
//! value (ADR-300 rule 1). Nothing in this crate panics on malformed runtime
//! input.
//!
//! [`DomainState::Unknown`]: crate::DomainState::Unknown

use thiserror::Error;

/// Errors from constructing OOD configuration values at their boundary.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum OodError {
    /// A configuration value was non-finite or outside its documented range.
    #[error("invalid OOD parameter '{field}': {reason}")]
    InvalidParameter {
        /// The offending field.
        field: &'static str,
        /// Why it was rejected (value + expected range).
        reason: String,
    },
}

/// Convenience result alias for boundary-validated constructors.
pub type Result<T> = core::result::Result<T, OodError>;

/// Validate that `value` is finite and within `[0, 1]`, or return a boundary
/// error naming `field`. Shared by every bounded `[0, 1]` config field so the
/// discipline is identical at each boundary.
pub(crate) fn require_unit_interval(field: &'static str, value: f32) -> Result<f32> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(OodError::InvalidParameter {
            field,
            reason: format!("must be finite in [0, 1], got {value}"),
        });
    }
    Ok(value)
}
