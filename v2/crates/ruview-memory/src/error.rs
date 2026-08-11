//! Boundary errors (ADR-309 / CLAUDE.md — validate untrusted input, never
//! panic).
//!
//! Malformed input yields one of these typed errors; nothing here panics.

use thiserror::Error;

/// Errors raised at the spatial-memory input boundaries.
#[derive(Clone, Debug, PartialEq, Error)]
pub enum MemoryError {
    /// A configuration field was out of its valid domain.
    #[error("invalid config: {what}")]
    InvalidConfig {
        /// Human-readable reason.
        what: &'static str,
    },
    /// A supplied value was non-finite (`NaN`/`inf`).
    #[error("non-finite value: {what}")]
    NonFiniteValue {
        /// Which value.
        what: &'static str,
    },
    /// Occupancy was negative.
    #[error("occupancy must be >= 0, got {value}")]
    NegativeOccupancy {
        /// The rejected value.
        value: f64,
    },
    /// More zones than [`MAX_ZONES`](crate::MAX_ZONES).
    #[error("too many zones (max {max})")]
    TooManyZones {
        /// The enforced maximum.
        max: usize,
    },
    /// More links in a zone than [`MAX_LINKS_PER_ZONE`](crate::MAX_LINKS_PER_ZONE).
    #[error("too many links in a zone (max {max})")]
    TooManyLinks {
        /// The enforced maximum.
        max: usize,
    },
    /// More modality channels than
    /// [`MAX_MODALITY_CHANNELS`](crate::MAX_MODALITY_CHANNELS).
    #[error("too many modality channels (max {max})")]
    TooManyChannels {
        /// The enforced maximum.
        max: usize,
    },
    /// A modality channel id exceeded
    /// [`MAX_CHANNEL_ID_LEN`](crate::MAX_CHANNEL_ID_LEN).
    #[error("modality channel id length {len} exceeds maximum {max}")]
    ChannelIdTooLong {
        /// Actual length in bytes.
        len: usize,
        /// The enforced maximum.
        max: usize,
    },
}
