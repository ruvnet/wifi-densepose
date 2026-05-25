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

    /// The plugin's `unload` hook returned an error.
    #[error("plugin unload failed: {0}")]
    UnloadFailed(String),

    /// IO error (manifest file not found, WASM binary missing, etc.).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
