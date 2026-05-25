//! Unified error type for `homecore-hap`.

use thiserror::Error;

/// Errors produced by the HAP bridge and its sub-components.
#[derive(Debug, Error)]
pub enum HapError {
    #[error("entity not found: {0}")]
    EntityNotFound(String),

    #[error("entity {entity_id} cannot be mapped to a HAP accessory type: {reason}")]
    UnmappableEntity { entity_id: String, reason: String },

    #[error("accessory already registered: {0}")]
    AlreadyRegistered(String),

    #[error("mDNS advertiser error: {0}")]
    MdnsError(String),

    #[error("bridge not running")]
    NotRunning,
}
