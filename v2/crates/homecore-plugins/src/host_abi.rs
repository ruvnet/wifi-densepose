//! Host ABI — the public on-the-wire memory format between the HOMECORE host
//! and every WASM plugin.
//!
//! # Overview
//!
//! HOMECORE uses **JSON over UTF-8 linear memory** for all host↔guest data.
//! This matches HA's JSON-everywhere convention and makes call payloads
//! inspectable in debuggers without a schema file. Each `hc_*` host function
//! and each guest export uses the same pointer + length convention:
//!
//! ```text
//!   host calls alloc(size) → ptr  (exported by guest)
//!   host writes UTF-8 bytes into guest linear memory at [ptr, ptr+size)
//!   host calls the guest export with (ptr: i32, len: i32)
//!   guest reads and JSON-decodes the slice
//!   guest writes its reply via hc_state_set / hc_log / etc. (host imports)
//!   host calls dealloc(ptr, size) when finished   (exported by guest)
//! ```
//!
//! # Wire types
//!
//! | Call | Direction | JSON schema |
//! |------|-----------|-------------|
//! | `hc_state_get` reply | host → caller | `{"entity_id":"…","state":"…","attributes":{…}}` or null bytes (not found) |
//! | `hc_state_set` args | guest → host | `(entity_id, state, attrs)` as 3 separate ptr/len pairs; each is a UTF-8 string or JSON object |
//! | `hc_log` args | guest → host | `(level: i32, msg)` where level 0=debug 1=info 2=warn 3=error |
//! | `hc_state_subscribe` | guest → host | entity_id UTF-8 string |
//! | `setup_entry` | host → guest | `{"entry_id":"…","domain":"…","data":{}}` (ConfigEntry JSON) |
//! | `receive_event` | host → guest | `{"event_type":"state_changed","entity_id":"…","new_state":"…"}` |
//!
//! # Memory layout guarantees
//!
//! - Buffers are **always** valid UTF-8 (JSON subset).
//! - Maximum buffer size is **64 KiB** (65,536 bytes). Larger payloads must
//!   be split by the caller; the host rejects oversized writes with a WASM
//!   trap. This bound is enforced in [`write_guest_buf`].
//! - The host **never** holds a guest memory pointer across a WASM call
//!   boundary. Pointers are only valid for the duration of a single call.
//!
//! # `hc_state_subscribe` semantics
//!
//! A plugin calls `hc_state_subscribe(eid_ptr, eid_len)` once per entity it
//! wants to track. Subsequent state changes for that entity arrive via a
//! `receive_event` call with event_type `"state_changed"`.
//!
//! Subscriptions are held for the lifetime of the plugin instance.

/// Maximum number of bytes the host will write into a single guest buffer.
/// Plugins may safely size their `alloc` buffers at this ceiling.
pub const MAX_ABI_BUFFER_BYTES: usize = 65_536;

/// JSON payload passed to `setup_entry` when a config entry is set up.
///
/// Serialises to HA-compat `ConfigEntry` JSON.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ConfigEntryJson {
    pub entry_id: String,
    pub domain: String,
    pub title: String,
    pub data: serde_json::Value,
}

impl ConfigEntryJson {
    /// Construct a minimal config entry for test / bootstrap use.
    pub fn bootstrap(domain: &str) -> Self {
        Self {
            entry_id: uuid::Uuid::new_v4().to_string(),
            domain: domain.to_owned(),
            title: domain.to_owned(),
            data: serde_json::json!({}),
        }
    }
}

/// JSON payload for `receive_event` — `state_changed` variant.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct StateChangedEventJson {
    pub event_type: String,
    pub entity_id: String,
    pub new_state: Option<String>,
    pub attributes: serde_json::Value,
}

impl StateChangedEventJson {
    /// Construct a `state_changed` event payload.
    pub fn state_changed(
        entity_id: &str,
        new_state: Option<&str>,
        attributes: serde_json::Value,
    ) -> Self {
        Self {
            event_type: "state_changed".to_owned(),
            entity_id: entity_id.to_owned(),
            new_state: new_state.map(str::to_owned),
            attributes,
        }
    }
}

/// Log levels for `hc_log`.
#[repr(i32)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl LogLevel {
    /// Convert from the i32 wire value. Unknown values map to `Warn`.
    pub fn from_i32(n: i32) -> Self {
        match n {
            0 => LogLevel::Debug,
            1 => LogLevel::Info,
            3 => LogLevel::Error,
            _ => LogLevel::Warn,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
}
