//! Unified error type for `homecore-hap`.

use std::path::PathBuf;
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

    #[error("pairing store error: {0}")]
    PairingStore(String),

    #[error("invalid pairing record: {0}")]
    InvalidPairingRecord(String),

    #[error("controller pairing already exists: {0}")]
    PairingAlreadyExists(String),

    #[error("controller pairing not found: {0}")]
    PairingNotFound(String),

    #[error("maximum controller pairings reached")]
    PairingCapacity,

    #[error("insecure permissions on {path}: mode {mode:o}; expected no group/other access")]
    InsecurePermissions { path: PathBuf, mode: u32 },

    #[error("invalid HAP session transition from {from} to {to}")]
    InvalidSessionTransition {
        from: &'static str,
        to: &'static str,
    },

    #[error("HAP protocol error: {0}")]
    Protocol(String),

    #[error("HAP server error: {0}")]
    Server(String),
}
