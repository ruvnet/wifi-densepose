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
//! ## What's NOT here yet (deferred)
//!
//! - `WasmtimeRuntime` (P2, `--features wasmtime`): Cranelift JIT sandbox on
//!   Pi 5 / x86_64. The runtime-selection question (Wasmtime vs wasm3) is still
//!   open (ADR-128 §8) and will be resolved in Q2 before P2 begins.
//! - Host ABI wiring: `hc_state_get`, `hc_state_set`, `hc_event_fire`, etc.
//!   (P2 — requires ADR-127 state machine API freeze first).
//! - Config entry lifecycle + hot-load (P3).
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
//! | `wasm3` | off | wasm3 interpreter runtime for constrained hardware (P3) |

pub mod error;
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
