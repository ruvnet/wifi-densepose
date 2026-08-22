//! Home Assistant MQTT auto-discovery payload publisher. ADR-122 §2.1.
//!
//! Generates the JSON config messages HA expects on
//! `homeassistant/<type>/<unique_id>/config` to auto-create the six BFLD
//! entities. Class-gated identically to the state-topic router
//! (`mqtt_topics.rs`): `identity_risk` discovery is only published at exactly
//! `PrivacyClass::Anonymous`.
//!
//! Discovery payloads should be published **once per node session**, retained
//! by the broker (`retain = true`) so HA finds them on next start. The
//! `RumqttPublisher` exposes a `with_retain(true)` builder for this; the
//! state-topic loop must keep `retain = false` to avoid stale-state flapping.

#![cfg(feature = "std")]

use crate::mqtt_topics::{Publish, TopicMessage};
use crate::PrivacyClass;

/// Bootstrap helper: render the per-node HA-DISCO config payloads and forward
/// each through `publisher`. Returns the count published, or short-circuits
/// on the first publisher error.
///
/// Typical bootstrap pattern combining iter 25's `Arc<Mutex<P>>` adapter and
/// iter 23's retain-aware `RumqttPublisher`:
///
/// ```ignore
/// use std::sync::{Arc, Mutex};
/// use wifi_densepose_bfld::{
///     publish_discovery, BfldConfig, BfldPipeline, BfldPipelineHandle,
///     PrivacyClass, RumqttPublisher,
/// };
/// use rumqttc::MqttOptions;
///
/// let opts = MqttOptions::new("seed-01", ("broker.local", 1883));
/// let (retained_pub, _conn) = RumqttPublisher::connect(opts.clone(), 64);
/// let mut retained_pub = retained_pub.with_retain(true);
/// publish_discovery(&mut retained_pub, "seed-01", PrivacyClass::Anonymous)?;
///
/// let (state_pub, _conn) = RumqttPublisher::connect(opts, 64);
/// let pipeline = BfldPipeline::new(BfldConfig::new("seed-01"));
/// let handle = BfldPipelineHandle::spawn(pipeline, state_pub);
/// // handle.send(...) from now on
/// # Ok::<(), rumqttc::ClientError>(())
/// ```
pub fn publish_discovery<P: Publish>(
    publisher: &mut P,
    node_id: &str,
    class: PrivacyClass,
) -> Result<usize, P::Error> {
    let mut count = 0;
    for msg in render_discovery_payloads(node_id, class) {
        publisher.publish(&msg)?;
        count += 1;
    }
    Ok(count)
}

/// Render every HA-DISCO config message for the given node at `class`. Returns
/// an empty `Vec` for classes < `Anonymous` (HA doesn't see raw / derived).
#[must_use]
pub fn render_discovery_payloads(node_id: &str, class: PrivacyClass) -> Vec<TopicMessage> {
    if class.as_u8() < PrivacyClass::Anonymous.as_u8() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(6);

    out.push(config_message(
        "binary_sensor",
        node_id,
        "presence",
        "BFLD Presence",
        Some("occupancy"),
        None,
        None,
    ));
    out.push(config_message(
        "sensor",
        node_id,
        "motion",
        "BFLD Motion",
        None,
        None,
        Some("diagnostic"),
    ));
    out.push(config_message(
        "sensor",
        node_id,
        "person_count",
        "BFLD Person Count",
        None,
        Some("people"),
        None,
    ));
    out.push(config_message(
        "sensor",
        node_id,
        "zone_activity",
        "BFLD Zone Activity",
        None,
        None,
        Some("diagnostic"),
    ));
    out.push(config_message(
        "sensor",
        node_id,
        "confidence",
        "BFLD Confidence",
        None,
        None,
        Some("diagnostic"),
    ));

    // identity_risk discovery only at class 2. Class 3 computes but doesn't
    // publish — therefore HA should not even see the entity exist.
    if class == PrivacyClass::Anonymous {
        out.push(config_message(
            "sensor",
            node_id,
            "identity_risk",
            "BFLD Identity Risk",
            None,
            None,
            Some("diagnostic"),
        ));
    }

    out
}

fn config_message(
    ha_type: &str,
    node_id: &str,
    entity: &str,
    name: &str,
    device_class: Option<&str>,
    unit_of_measurement: Option<&str>,
    entity_category: Option<&str>,
) -> TopicMessage {
    let unique_id = format!("{node_id}_bfld_{entity}");
    let topic = format!("homeassistant/{ha_type}/{unique_id}/config");
    let state_topic = format!("ruview/{node_id}/bfld/{entity}/state");
    let availability_topic_str = crate::availability::availability_topic(node_id);

    let mut payload = String::with_capacity(384);
    payload.push('{');
    push_str_field(&mut payload, "name", name, true);
    push_str_field(&mut payload, "unique_id", &unique_id, false);
    push_str_field(&mut payload, "state_topic", &state_topic, false);
    // Availability — every entity inherits the device-level offline marker.
    push_str_field(&mut payload, "availability_topic", &availability_topic_str, false);
    push_str_field(
        &mut payload,
        "payload_available",
        crate::availability::PAYLOAD_AVAILABLE,
        false,
    );
    push_str_field(
        &mut payload,
        "payload_not_available",
        crate::availability::PAYLOAD_NOT_AVAILABLE,
        false,
    );
    if let Some(dc) = device_class {
        push_str_field(&mut payload, "device_class", dc, false);
    }
    if let Some(unit) = unit_of_measurement {
        push_str_field(&mut payload, "unit_of_measurement", unit, false);
    }
    if let Some(cat) = entity_category {
        push_str_field(&mut payload, "entity_category", cat, false);
    }
    payload.push_str(",\"device\":{");
    push_str_field(&mut payload, "identifiers", node_id, true);
    push_str_field(
        &mut payload,
        "name",
        &format!("RuView Seed {node_id}"),
        false,
    );
    push_str_field(&mut payload, "model", "BFLD", false);
    push_str_field(&mut payload, "manufacturer", "RuView", false);
    payload.push('}');
    payload.push('}');

    TopicMessage { topic, payload }
}

fn push_str_field(out: &mut String, key: &str, value: &str, first: bool) {
    if !first {
        out.push(',');
    }
    out.push('"');
    out.push_str(key);
    out.push_str("\":\"");
    // Minimal JSON escaping for the values BFLD controls — node_id is ASCII
    // alphanumeric + dash by convention, names are operator-controlled. A
    // future iter can swap to serde_json::to_string for full escape coverage.
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let escape = format!("\\u{:04x}", c as u32);
                out.push_str(&escape);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
