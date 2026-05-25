//! `HomeCorePlugin` trait + `PluginId` newtype.
//!
//! Every first-party and third-party HOMECORE integration must implement
//! `HomeCorePlugin`. P1 provides an in-process native Rust implementation;
//! the WASM ABI wrapper (which maps the WASM exports `setup_entry`,
//! `call_service_handler`, `receive_event` to this trait) lands in P2.

use std::fmt;

use async_trait::async_trait;
use homecore::HomeCore;

use crate::error::PluginError;

/// Unique identifier for a loaded plugin — mirrors the `domain` field of
/// the plugin's `PluginManifest` (e.g. `"mqtt"`, `"homecore_lights"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PluginId(pub String);

impl PluginId {
    /// Create a new `PluginId` from any string-like value.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the inner domain string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PluginId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lifecycle trait that every HOMECORE integration must implement.
///
/// Implementing types are passed to [`PluginRuntime::load`]; the runtime
/// calls these methods at the appropriate lifecycle points.
///
/// # Async
/// Both methods are `async` to allow network / IO initialisation without
/// blocking the Tokio runtime. The `async_trait` macro erases the `impl`
/// return type so it works in trait objects.
#[async_trait]
pub trait HomeCorePlugin: Send + Sync + 'static {
    /// Called once when the plugin's config entry is being set up.
    ///
    /// The plugin receives a reference to the `HomeCore` runtime and should
    /// register its entities, services, and event subscriptions here.
    async fn setup(&self, hc: HomeCore) -> Result<(), PluginError>;

    /// Called when the plugin is being removed from the registry.
    ///
    /// The plugin should clean up subscriptions and deregister its entities.
    async fn unload(&self) -> Result<(), PluginError>;
}
