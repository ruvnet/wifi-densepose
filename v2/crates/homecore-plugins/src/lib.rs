//! HOMECORE-PLUGINS — WASM integration plugin system.
//!
//! Implements [ADR-128](../../docs/adr/ADR-128-homecore-integration-plugin-system.md)
//! P1 scaffold: manifest parsing, the `HomeCorePlugin` async trait, the
//! `PluginRuntime` abstraction, and the `PluginRegistry`.
//!
//! ## What's here (P1)
//!
//! - [`manifest`] — `PluginManifest`: superset of HA `manifest.json`; serde
//!   round-trip + required-field validation.
//! - [`plugin`] — `HomeCorePlugin` async trait, `PluginId` newtype.
//! - [`runtime`] — `PluginRuntime` trait + `InProcessRuntime` (native Rust,
//!   first-party plugins compiled into the binary).
//! - [`registry`] — `PluginRegistry<R>`: load / unload / list plugins.
//! - [`error`] — `PluginError` typed error enum.
//!
//! ## Runtime scope
//!
//! - `WasmtimeRuntime` (`--features wasmtime`) provides the Cranelift sandbox.
//! - Native Rust plugins are compiled into the server; arbitrary native
//!   dynamic libraries are not loaded.
//! - The host ABI is deliberately narrow and resource-bounded.
//!
//! ## Now enforced (ADR-162)
//!
//! - **Ed25519 signature + SHA-256 integrity verification (P4)** — see
//!   [`verify`]: the plugin load path hashes the real `.wasm` bytes, checks
//!   the manifest `wasm_module_hash`, verifies `wasm_module_sig` against
//!   `publisher_key`, and enforces a [`verify::PluginPolicy`] allowlist.
//! - **Permission / authority isolation (P5)** — see [`permissions`]: a
//!   plugin's `hc_state_set` writes are gated against the entity domains/
//!   globs it declared in `homecore_permissions`.
//!
//! ## Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `wasmtime` | off | Wasmtime Cranelift JIT runtime (P2) |

pub mod error;
pub mod discovery;
pub mod host_abi;
pub mod manifest;
pub mod permissions;
pub mod plugin;
pub mod registry;
pub mod runtime;
pub mod verify;

#[cfg(feature = "wasmtime")]
pub mod wasmtime_runtime;

pub use error::PluginError;
pub use discovery::{discover_plugins, DiscoveredPlugin, DiscoveryLimits};
pub use host_abi::{ConfigEntryJson, StateChangedEventJson};
pub use manifest::{IotClass, IntegrationType, PluginManifest};
pub use permissions::PermissionSet;
pub use plugin::{HomeCorePlugin, PluginId};
pub use registry::PluginRegistry;
pub use runtime::{InProcessRuntime, LoadedPlugin, PluginRuntime};
pub use verify::{verify_module, PluginPolicy};

#[cfg(feature = "wasmtime")]
pub use wasmtime_runtime::{WasmPlugin, WasmtimeRuntime};

#[cfg(test)]
mod tests;
