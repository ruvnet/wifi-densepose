//! `PluginError` — typed error enum for the homecore-plugins crate.

use thiserror::Error;

/// Errors produced by the HOMECORE plugin system.
#[derive(Debug, Error)]
pub enum PluginError {
    /// The plugin manifest JSON is missing required fields or is malformed.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// A plugin with this ID is already loaded in the registry.
    #[error("plugin already loaded: {0}")]
    AlreadyLoaded(String),

    /// No plugin with this ID is loaded in the registry.
    #[error("plugin not found: {0}")]
    NotFound(String),

    /// The plugin runtime failed to spawn or execute the plugin.
    #[error("runtime error: {0}")]
    RuntimeError(String),

    /// The plugin's `setup` hook returned an error.
    #[error("plugin setup failed: {0}")]
    SetupFailed(String),

    /// The plugin failed signature/integrity verification (ADR-162 P4):
    /// hash mismatch, bad signature, untrusted publisher, or unsigned
    /// module under a non-dev trust policy.
    #[error("plugin signature rejected: {0}")]
    SignatureRejected(String),

    /// A plugin attempted a host call (e.g. `hc_state_set`) on an entity
    /// it did not declare in `homecore_permissions` (ADR-162 P5 authority
    /// isolation).
    #[error("plugin permission denied: {0}")]
    PermissionDenied(String),

    /// The plugin's `unload` hook returned an error.
    #[error("plugin unload failed: {0}")]
    UnloadFailed(String),

    /// IO error (manifest file not found, WASM binary missing, etc.).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
