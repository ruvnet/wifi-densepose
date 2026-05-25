//! HOMECORE example WASM plugin — proves the ADR-128 P2 host ABI round-trip.
//!
//! # Behaviour
//!
//! This plugin monitors `sensor.test_temp` and controls
//! `binary_sensor.test_alert` based on the temperature reading:
//!
//! - `sensor.test_temp` > 25 → set `binary_sensor.test_alert` to `"on"`
//! - `sensor.test_temp` < 20 → set `binary_sensor.test_alert` to `"off"`
//! - Between 20 and 25 → no change (hysteresis dead-band)
//!
//! # ABI
//!
//! The plugin is compiled to `wasm32-unknown-unknown` and exposes the three
//! exports required by the HOMECORE host ABI (ADR-128 §5.2):
//!
//! | Export | Signature | Called when |
//! |--------|-----------|-------------|
//! | `plugin_setup` | `(ptr:i32, len:i32) → i32` | Config entry set up |
//! | `plugin_handle_state_changed` | `(ptr:i32, len:i32) → i32` | State change event |
//! | `alloc` | `(size:i32) → i32` | Host needs a guest buffer |
//! | `dealloc` | `(ptr:i32, size:i32)` | Host frees a guest buffer |
//!
//! # Wire format
//!
//! All payloads are **UTF-8 JSON** delivered via length-prefixed linear
//! memory pointers. See `abi.rs` for the guest-side helpers and
//! `homecore-plugins/src/host_abi.rs` for the authoritative spec.

mod abi;

// Re-export alloc/dealloc so the host can find them.
pub use abi::{alloc, dealloc};

// ── Entity IDs ─────────────────────────────────────────────────────────────

const TEMP_SENSOR: &str = "sensor.test_temp";
const ALERT_SENSOR: &str = "binary_sensor.test_alert";

// ── Thresholds ─────────────────────────────────────────────────────────────

const HIGH_THRESH: f64 = 25.0; // above → alert on
const LOW_THRESH: f64 = 20.0; // below → alert off

// ── Plugin exports ──────────────────────────────────────────────────────────

/// `plugin_setup(config_entry_ptr: i32, config_entry_len: i32) → i32`
///
/// Called once by the host when the config entry is set up. Subscribes to
/// `sensor.test_temp` state changes so the host will deliver them via
/// `plugin_handle_state_changed`.
///
/// Returns 0 on success, negative on error.
#[no_mangle]
pub unsafe extern "C" fn plugin_setup(_ptr: i32, _len: i32) -> i32 {
    // Subscribe to temperature sensor state changes.
    let sub_result = abi::hc_state_subscribe(
        TEMP_SENSOR.as_ptr() as i32,
        TEMP_SENSOR.len() as i32,
    );
    if sub_result != 0 {
        return -1;
    }
    abi::log_info("homecore-plugin-example: setup complete, subscribed to sensor.test_temp");
    0
}

/// `plugin_handle_state_changed(event_ptr: i32, event_len: i32) → i32`
///
/// Called by the host whenever a subscribed entity changes state.
/// The payload is a JSON object:
/// `{"event_type":"state_changed","entity_id":"…","new_state":"…","attributes":{}}`
///
/// Returns 0 on success, negative on error.
#[no_mangle]
pub unsafe extern "C" fn plugin_handle_state_changed(ptr: i32, len: i32) -> i32 {
    if len <= 0 || len as usize > abi::MAX_ABI_BUFFER_BYTES {
        return -1;
    }

    // Read the event JSON from linear memory.
    let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    let json_str = match std::str::from_utf8(slice) {
        Ok(s) => s,
        Err(_) => return -2,
    };

    // Parse the event JSON.
    let entity_id = extract_json_string(json_str, "entity_id");
    let new_state_raw = extract_json_string(json_str, "new_state");

    // Only act on sensor.test_temp.
    match entity_id.as_deref() {
        Some(e) if e == TEMP_SENSOR => {}
        _ => return 0,
    };

    let new_state = match new_state_raw {
        Some(s) => s,
        None => return 0,
    };

    // Parse the temperature value.
    let temp: f64 = match new_state.parse::<f64>() {
        Ok(t) => t,
        Err(_) => return 0, // not a number — ignore
    };

    // Apply threshold logic with hysteresis dead-band.
    if temp > HIGH_THRESH {
        abi::set_state(ALERT_SENSOR, "on", "{}");
        abi::log_info("homecore-plugin-example: temp > 25, alert ON");
    } else if temp < LOW_THRESH {
        abi::set_state(ALERT_SENSOR, "off", "{}");
        abi::log_info("homecore-plugin-example: temp < 20, alert OFF");
    }
    // Dead-band: 20 <= temp <= 25, no change.

    0
}

// ── Minimal JSON field extraction ──────────────────────────────────────────

/// Extract a string value for `key` from a flat JSON object string.
/// Returns `Some(value)` if found, `None` otherwise.
/// Only handles simple `"key":"value"` pairs at the top level.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":\"", key);
    let start = json.find(&needle)? + needle.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}
