//! Generated OpenTelemetry semantic-convention name constants for
//! RuView's curated telemetry (event names and attribute keys).
//!
//! GENERATED from `semconv/registry/` by `weaver registry generate`.
//! Do not edit by hand: change the registry or the template at
//! `templates/registry/rust/`, then regenerate (the exact command CI
//! runs — note `--future`, matching `weaver registry check --future`)
//! from the repository root and commit the result:
//!
//! ```text
//! weaver registry generate rust v2/crates/wifi-densepose-sensing-server/src \
//!     -t templates -r semconv/registry --future
//! cargo fmt -p wifi-densepose-sensing-server
//! ```
//!
//! The CI `semconv` workflow fails if this file drifts from the registry.

/// The semantic-conventions schema URL these constants were generated from —
/// the registry manifest's `schema_url`, which carries the conventions
/// version. Attach it to a telemetry resource so consumers can resolve the
/// schema.
pub const SCHEMA_URL: &str =
    "https://raw.githubusercontent.com/ruvnet/RuView/main/semconv/schema/ruview-0.1.0.yaml";

// Attribute keys.

/// `ruview.csi.frames_total` attribute key.
pub const RUVIEW_CSI_FRAMES_TOTAL: &str = "ruview.csi.frames_total";

/// `ruview.csi.nodes_active` attribute key.
pub const RUVIEW_CSI_NODES_ACTIVE: &str = "ruview.csi.nodes_active";

/// `ruview.csi.source` attribute key.
pub const RUVIEW_CSI_SOURCE: &str = "ruview.csi.source";

/// `ruview.inference.confidence` attribute key.
pub const RUVIEW_INFERENCE_CONFIDENCE: &str = "ruview.inference.confidence";

/// `ruview.model.id` attribute key.
pub const RUVIEW_MODEL_ID: &str = "ruview.model.id";

/// `ruview.motion.level` attribute key.
pub const RUVIEW_MOTION_LEVEL: &str = "ruview.motion.level";

/// `ruview.node.id` attribute key.
pub const RUVIEW_NODE_ID: &str = "ruview.node.id";

/// `ruview.persons.count` attribute key.
pub const RUVIEW_PERSONS_COUNT: &str = "ruview.persons.count";

/// `ruview.presence.state` attribute key.
pub const RUVIEW_PRESENCE_STATE: &str = "ruview.presence.state";

/// `ruview.vitals.breathing_confidence` attribute key.
pub const RUVIEW_VITALS_BREATHING_CONFIDENCE: &str = "ruview.vitals.breathing_confidence";

/// `ruview.vitals.breathing_rate_bpm` attribute key.
pub const RUVIEW_VITALS_BREATHING_RATE_BPM: &str = "ruview.vitals.breathing_rate_bpm";

/// `ruview.vitals.heart_rate_bpm` attribute key.
pub const RUVIEW_VITALS_HEART_RATE_BPM: &str = "ruview.vitals.heart_rate_bpm";

/// `ruview.vitals.heartbeat_confidence` attribute key.
pub const RUVIEW_VITALS_HEARTBEAT_CONFIDENCE: &str = "ruview.vitals.heartbeat_confidence";

/// Every attribute key registered for curated RuView events.
pub const ATTRIBUTE_KEYS: &[&str] = &[
    RUVIEW_CSI_FRAMES_TOTAL,
    RUVIEW_CSI_NODES_ACTIVE,
    RUVIEW_CSI_SOURCE,
    RUVIEW_INFERENCE_CONFIDENCE,
    RUVIEW_MODEL_ID,
    RUVIEW_MOTION_LEVEL,
    RUVIEW_NODE_ID,
    RUVIEW_PERSONS_COUNT,
    RUVIEW_PRESENCE_STATE,
    RUVIEW_VITALS_BREATHING_CONFIDENCE,
    RUVIEW_VITALS_BREATHING_RATE_BPM,
    RUVIEW_VITALS_HEART_RATE_BPM,
    RUVIEW_VITALS_HEARTBEAT_CONFIDENCE,
];

// Log event names (each instrumented `tracing` call site names its event
// with one of these so the exported Logs signal stays registry-backed).

/// `ruview.csi.stats` log event name.
pub const EVENT_RUVIEW_CSI_STATS: &str = "ruview.csi.stats";

/// `ruview.fall.detected` log event name.
pub const EVENT_RUVIEW_FALL_DETECTED: &str = "ruview.fall.detected";

/// `ruview.model.loaded` log event name.
pub const EVENT_RUVIEW_MODEL_LOADED: &str = "ruview.model.loaded";

/// `ruview.mqtt.error` log event name.
pub const EVENT_RUVIEW_MQTT_ERROR: &str = "ruview.mqtt.error";

/// `ruview.node.offline` log event name.
pub const EVENT_RUVIEW_NODE_OFFLINE: &str = "ruview.node.offline";

/// `ruview.node.online` log event name.
pub const EVENT_RUVIEW_NODE_ONLINE: &str = "ruview.node.online";

/// `ruview.presence.changed` log event name.
pub const EVENT_RUVIEW_PRESENCE_CHANGED: &str = "ruview.presence.changed";

/// `ruview.vitals.estimate` log event name.
pub const EVENT_RUVIEW_VITALS_ESTIMATE: &str = "ruview.vitals.estimate";

/// Every curated event name in the generated registry.
pub const EVENT_NAMES: &[&str] = &[
    EVENT_RUVIEW_CSI_STATS,
    EVENT_RUVIEW_FALL_DETECTED,
    EVENT_RUVIEW_MODEL_LOADED,
    EVENT_RUVIEW_MQTT_ERROR,
    EVENT_RUVIEW_NODE_OFFLINE,
    EVENT_RUVIEW_NODE_ONLINE,
    EVENT_RUVIEW_PRESENCE_CHANGED,
    EVENT_RUVIEW_VITALS_ESTIMATE,
];

#[cfg(test)]
mod tests {
    use super::{ATTRIBUTE_KEYS, EVENT_NAMES};

    #[test]
    fn instrumentation_uses_only_registered_ruview_literals() {
        let sources = [include_str!("main.rs"), include_str!("mqtt/publisher.rs")];

        for source in sources {
            let mut rest = source;
            while let Some(start) = rest.find("\"ruview.") {
                let value = &rest[start + 1..];
                let end = value
                    .find('"')
                    .expect("ruview string literal must have a closing quote");
                let literal = &value[..end];
                assert!(
                    ATTRIBUTE_KEYS.contains(&literal) || EVENT_NAMES.contains(&literal),
                    "instrumentation literal `{literal}` is absent from semconv/registry"
                );
                rest = &value[end + 1..];
            }
        }
    }
}
