//! Boundary-validation errors (ADR-307).
//!
//! These cover *malformed input* only. Association **uncertainty** is never an
//! error: an ambiguous or unmatched detection is reported as a first-class
//! [`Association::Unknown`](crate::Association) outcome (ADR-300 rule 1), not a
//! `Result::Err`.

use ruview_ontology::IdError;
use thiserror::Error;

/// Reasons a detection or a manager operation is rejected at the boundary.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TrackError {
    /// A position component was NaN or infinite.
    #[error("position component is not finite")]
    NonFinitePosition,
    /// The coarse feature vector was empty.
    #[error("coarse feature vector must not be empty")]
    EmptyFeature,
    /// The coarse feature vector exceeded [`MAX_FEATURE_DIM`](crate::MAX_FEATURE_DIM).
    #[error("feature dimension {dim} exceeds maximum {max}")]
    FeatureTooLarge {
        /// Supplied dimension.
        dim: usize,
        /// Enforced maximum.
        max: usize,
    },
    /// A minted pseudonym / track id failed ontology id validation. This is an
    /// internal invariant (the manager mints `track_N`/`person_N`) and only
    /// surfaces if the counter overflows the id-length bound.
    #[error("invalid minted identifier: {0}")]
    Id(#[from] IdError),
    /// A referenced track id is not held by the manager.
    #[error("unknown track id")]
    UnknownTrack,
}
