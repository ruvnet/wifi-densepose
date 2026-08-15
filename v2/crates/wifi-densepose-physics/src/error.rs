//! Typed internal errors. Public frame failures are mapped to abstention reasons.

/// Construction/configuration errors that cannot be represented as a frame result.
#[derive(Debug, thiserror::Error)]
pub enum PhysicsError {
    /// Invalid trusted configuration.
    #[error("invalid physics configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// A correction authorization is empty or bound to another configuration.
    #[error("invalid correction authorization: {0}")]
    InvalidCorrectionAuthorization(&'static str),
}
