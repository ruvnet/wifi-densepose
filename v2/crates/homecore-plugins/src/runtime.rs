//! `PluginRuntime` trait + `InProcessRuntime` (P1).
//!
//! Abstracts over Wasmtime (P2, `--features wasmtime`) and native in-process
//! Rust plugins (P1, always-on). A third backend, wasm3 (P3), will provide
//! interpretation mode for constrained hardware.
//!
//! # Architecture
//!
//! ```text
//! PluginRegistry
//!       │
//!       ▼
//! PluginRuntime  ◄─── InProcessRuntime  (P1, native Rust, <1 µs call)
//!                ◄─── WasmtimeRuntime   (P2, Cranelift JIT, ~5 ms cold start)
//!                ◄─── Wasm3Runtime      (P3, interpreter, ~50 kB, Pi Zero)
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use homecore::HomeCore;

use crate::error::PluginError;
use crate::manifest::PluginManifest;
use crate::plugin::{HomeCorePlugin, PluginId};

/// A loaded plugin handle — returned by [`PluginRuntime::load`].
pub struct LoadedPlugin {
    pub id: PluginId,
    pub manifest: PluginManifest,
    /// Underlying plugin instance (boxed trait object).
    pub(crate) instance: Arc<dyn HomeCorePlugin>,
}

impl LoadedPlugin {
    /// Delegate to the inner plugin's `setup` method.
    pub async fn setup(&self, hc: HomeCore) -> Result<(), PluginError> {
        self.instance.setup(hc).await
    }

    /// Delegate to the inner plugin's `unload` method.
    pub async fn unload(&self) -> Result<(), PluginError> {
        self.instance.unload().await
    }
}

/// Abstraction over the WASM (and native) plugin execution environment.
///
/// P2 will supply a `WasmtimeRuntime` that compiles `.wasm` bytes with
/// Cranelift; P3 adds a `Wasm3Runtime` for constrained targets. Both will
/// implement this trait so the registry is runtime-agnostic.
#[async_trait]
pub trait PluginRuntime: Send + Sync + 'static {
    /// Load a plugin from a boxed [`HomeCorePlugin`] implementation and a
    /// parsed `PluginManifest`. Returns a `LoadedPlugin` handle.
    async fn load(
        &self,
        id: PluginId,
        manifest: PluginManifest,
        plugin: Arc<dyn HomeCorePlugin>,
    ) -> Result<LoadedPlugin, PluginError>;
}

/// Native in-process runtime — loads first-party Rust plugins directly.
///
/// No WASM compilation; no sandbox. Intended for first-party plugins
/// (RuView MQTT bridge, presence sensor, etc.) that are compiled into the
/// HOMECORE binary and therefore trusted. Third-party / community plugins
/// must use the `WasmtimeRuntime` (P2) for isolation.
pub struct InProcessRuntime;

#[async_trait]
impl PluginRuntime for InProcessRuntime {
    async fn load(
        &self,
        id: PluginId,
        manifest: PluginManifest,
        plugin: Arc<dyn HomeCorePlugin>,
    ) -> Result<LoadedPlugin, PluginError> {
        Ok(LoadedPlugin {
            id,
            manifest,
            instance: plugin,
        })
    }
}

// ── Feature-gated Wasmtime stub (P2) ──────────────────────────────────────

#[cfg(feature = "wasmtime")]
pub mod wasmtime_rt {
    //! Wasmtime JIT runtime — P2 stub.
    //!
    //! This module intentionally does not compile to a usable runtime yet.
    //! It exists so that `cargo check --features wasmtime` exercises the
    //! dependency graph and catches obvious breakage early.
    //!
    //! Full implementation tracked in ADR-128 §7 P2.

    use super::*;

    /// Wasmtime-backed plugin runtime (Cranelift JIT on Pi 5 and x86_64).
    /// Not yet implemented — P2 work.
    pub struct WasmtimeRuntime;

    #[async_trait]
    impl PluginRuntime for WasmtimeRuntime {
        async fn load(
            &self,
            _id: PluginId,
            _manifest: PluginManifest,
            _plugin: Arc<dyn HomeCorePlugin>,
        ) -> Result<LoadedPlugin, PluginError> {
            Err(PluginError::RuntimeError(
                "WasmtimeRuntime is not yet implemented (ADR-128 P2)".into(),
            ))
        }
    }
}
