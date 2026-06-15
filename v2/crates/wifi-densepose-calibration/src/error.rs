//! Error types for the calibration pipeline.

use thiserror::Error;

/// Errors surfaced by the per-room calibration & training pipeline (ADR-151).
#[derive(Debug, Error)]
pub enum CalibrationError {
    /// An anchor was recorded with zero frames.
    #[error("anchor '{0}' captured no frames")]
    EmptyAnchor(String),

    /// The enrollment session is missing anchors required to train a specialist.
    #[error("enrollment incomplete: missing anchors {missing:?}")]
    IncompleteEnrollment {
        /// Labels still required.
        missing: Vec<String>,
    },

    /// A frame did not match the expected tier geometry.
    #[error("frame geometry mismatch: {0}")]
    Geometry(String),

    /// Not enough samples to fit a specialist.
    #[error("insufficient samples for '{kind}': have {have}, need {need}")]
    InsufficientSamples {
        /// Specialist kind.
        kind: String,
        /// Samples available.
        have: usize,
        /// Samples required.
        need: usize,
    },

    /// Serialization / persistence failure.
    #[error("serialization error: {0}")]
    Serde(String),

    /// The specialist bank was trained against a different baseline and is stale.
    #[error("bank is STALE: trained against baseline {trained}, current is {current}")]
    StaleBaseline {
        /// Baseline id the bank was trained against.
        trained: String,
        /// Current baseline id.
        current: String,
    },
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, CalibrationError>;
