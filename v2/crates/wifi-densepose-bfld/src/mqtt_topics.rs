//! MQTT topic router. ADR-122 §2.2.
//!
//! Pure-function module that maps a [`BfldEvent`] into a list of per-entity
//! MQTT topic + payload pairs. No broker dependency lives here — the actual
//! `publish` call is a thin wrapper around `Client::publish(topic, payload)`
//! once a broker integration lands (deferred to a follow-up iter).
//!
//! Topic shape (ADR-122 §2.2):
//!
//! ```text
//! ruview/<node_id>/bfld/presence/state          # class >= 2
//! ruview/<node_id>/bfld/motion/state            # class >= 2
//! ruview/<node_id>/bfld/person_count/state      # class >= 2
//! ruview/<node_id>/bfld/zone_activity/state     # class >= 2 (when zone_id set)
//! ruview/<node_id>/bfld/confidence/state        # class >= 2
//! ruview/<node_id>/bfld/identity_risk/state     # class == 2 only
//! ```
//!
//! `raw` (class-1) and `availability` topics are intentionally not yet emitted
//! by this router; they belong to the broker-connection lifecycle, not to the
//! per-event publish loop.

#![cfg(feature = "std")]

use crate::{BfldEvent, PrivacyClass};

/// Per-topic MQTT message ready to feed into `Client::publish(topic, payload)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicMessage {
    /// Full MQTT topic, e.g. `ruview/seed-01/bfld/presence/state`.
    pub topic: String,
    /// UTF-8 payload bytes — single JSON scalar (`true`, `0.72`, `"living_room"`)
    /// or a compact JSON object for diagnostics.
    pub payload: String,
}

impl TopicMessage {
    /// Build a topic of the form `ruview/<node_id>/bfld/<suffix>/state`.
    #[must_use]
    pub fn ruview_topic(node_id: &str, entity: &str) -> String {
        let mut s = String::with_capacity(7 + node_id.len() + 6 + entity.len() + 6);
        s.push_str("ruview/");
        s.push_str(node_id);
        s.push_str("/bfld/");
        s.push_str(entity);
        s.push_str("/state");
        s
    }
}

/// Abstract MQTT publisher boundary. The crate ships only the trait + a
/// capture-impl for tests; the production rumqttc-backed impl lands in a
/// follow-up iter behind a `mqtt` feature gate.
///
/// `publish` is synchronous so callers can hold a `&mut self` without an
/// async runtime; the rumqttc wrapper drives a tokio task internally.
pub trait Publish {
    /// Error type — typically the broker's transport error.
    type Error;
    /// Publish a single rendered message. Implementations may buffer.
    fn publish(&mut self, msg: &TopicMessage) -> Result<(), Self::Error>;
}

/// Capture-impl for unit tests. Stores every published message in order.
#[derive(Debug, Default)]
pub struct CapturePublisher {
    /// Every `publish()` call appends to this vec.
    pub published: Vec<TopicMessage>,
}

impl Publish for CapturePublisher {
    type Error = core::convert::Infallible;
    fn publish(&mut self, msg: &TopicMessage) -> Result<(), Self::Error> {
        self.published.push(msg.clone());
        Ok(())
    }
}

/// Forward `Publish` through a shared `Arc<Mutex<P>>` so a publisher owned by
/// a worker thread can still be inspected by the test or operator after the
/// fact. Lock-poisoning is treated as a panic — there is no recovery story.
impl<P: Publish> Publish for std::sync::Arc<std::sync::Mutex<P>> {
    type Error = P::Error;
    fn publish(&mut self, msg: &TopicMessage) -> Result<(), Self::Error> {
        self.lock()
            .expect("BFLD publish: inner publisher Mutex poisoned")
            .publish(msg)
    }
}

/// Publish every topic message rendered from `event`. Returns the number of
/// messages actually published (zero for Raw / Derived class events). Errors
/// short-circuit — the publisher state at error time may have partial output.
pub fn publish_event<P: Publish>(
    publisher: &mut P,
    event: &BfldEvent,
) -> Result<usize, P::Error> {
    let mut count = 0;
    for msg in render_events(event) {
        publisher.publish(&msg)?;
        count += 1;
    }
    Ok(count)
}

/// Render an event into the per-entity MQTT messages it should publish. Returns
/// an empty vec for events that fail the class gate (e.g., raw class 0).
#[must_use]
pub fn render_events(event: &BfldEvent) -> Vec<TopicMessage> {
    let class_byte = event.privacy_class.as_u8();
    if class_byte < PrivacyClass::Anonymous.as_u8() {
        // Raw + Derived stay local — never published on the public topic tree.
        return Vec::new();
    }

    let mut out = Vec::with_capacity(6);
    let node = &event.node_id;

    out.push(TopicMessage {
        topic: TopicMessage::ruview_topic(node, "presence"),
        payload: if event.presence { "true".into() } else { "false".into() },
    });
    out.push(TopicMessage {
        topic: TopicMessage::ruview_topic(node, "motion"),
        payload: format!("{:.6}", event.motion),
    });
    out.push(TopicMessage {
        topic: TopicMessage::ruview_topic(node, "person_count"),
        payload: format!("{}", event.person_count),
    });
    out.push(TopicMessage {
        topic: TopicMessage::ruview_topic(node, "confidence"),
        payload: format!("{:.6}", event.confidence),
    });

    if let Some(zone) = &event.zone_id {
        // Emit a JSON string so consumers can distinguish "no zone" (omitted)
        // from "single-zone deployment" (always the same zone string). The zone
        // name is operator-controlled; escape JSON metacharacters so a name
        // containing a quote or backslash cannot produce malformed/injected
        // JSON. Mirrors ha_discovery.rs::push_str_field's escaping.
        out.push(TopicMessage {
            topic: TopicMessage::ruview_topic(node, "zone_activity"),
            payload: json_string_literal(zone),
        });
    }

    // Identity risk is only published at exactly class 2 (Anonymous). Class 3
    // (Restricted) computes the score internally but never emits it.
    if class_byte == PrivacyClass::Anonymous.as_u8() {
        if let Some(score) = event.identity_risk_score {
            out.push(TopicMessage {
                topic: TopicMessage::ruview_topic(node, "identity_risk"),
                payload: format!("{score:.6}"),
            });
        }
    }

    out
}

/// Wrap `value` in JSON double-quote delimiters, escaping the metacharacters
/// that would otherwise break out of the string literal (`"`, `\`, control
/// chars, and the bare `\n`/`\r`/`\t` whitespace). Kept in lockstep with
/// `ha_discovery::push_str_field` so state-topic and discovery payloads escape
/// identically.
fn json_string_literal(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
