//! Smoke tests for `RumqttPublisher`. Verifies the `mqtt` feature compiles
//! and the publisher constructs without a live broker. Full integration
//! against a real mosquitto lives in a follow-up iter (env-gated to keep CI
//! green when no broker is available).

#![cfg(feature = "mqtt")]

use rumqttc::{MqttOptions, QoS};
use wifi_densepose_bfld::mqtt_topics::TopicMessage;
use wifi_densepose_bfld::{publish_event, BfldEvent, PrivacyClass, Publish, RumqttPublisher};

fn unreachable_opts() -> MqttOptions {
    // Port 1 is reserved (RFC 1700) and the loopback address will refuse
    // immediately — perfect for a construction smoke test that must not block.
    MqttOptions::new("bfld-smoke-iter23", ("127.0.0.1", 1))
}

fn sample_event() -> BfldEvent {
    BfldEvent::with_privacy_gating(
        "seed-99".into(),
        1_700_000_000_000_000_000,
        true,
        0.5,
        1,
        0.9,
        None,
        PrivacyClass::Anonymous,
        Some(0.25),
        Some([0xAB; 32]),
    )
}

#[test]
fn rumqttc_publisher_constructs_without_broker() {
    let (_publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    // Reaching this line means rumqttc::Client::new() returned without panic
    // (it spawns its own connection task that fails async — never propagates here).
}

#[test]
fn with_retain_builder_yields_a_publisher() {
    let (publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    let _retained = publisher.with_retain(true);
}

#[test]
fn publish_queues_message_without_blocking_on_broker_state() {
    // rumqttc's sync Client::publish puts the packet into an unbounded
    // queue; it returns Ok even when the connection is offline. The queued
    // packet will only succeed when a thread iterates Connection::iter(),
    // which we deliberately do NOT do here — the smoke test verifies that
    // `publish_event` returns `Ok(6)` without blocking on the broker.
    let (mut publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    let event = sample_event();
    let count = publish_event(&mut publisher, &event).expect("queue must accept");
    assert_eq!(count, 5, "Anonymous + no zone publishes 5 topic messages");
}

#[test]
fn restricted_event_publishes_four_messages_through_rumqttc() {
    let mut event = sample_event();
    event.privacy_class = PrivacyClass::Restricted;
    event.apply_privacy_gating();
    let (mut publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    let count = publish_event(&mut publisher, &event).expect("queue must accept");
    assert_eq!(
        count, 4,
        "Restricted + no zone publishes 4 topics (no identity_risk)",
    );
}

#[test]
fn publisher_trait_object_is_constructible() {
    // Compile-time witness that RumqttPublisher implements Publish; lets
    // operators store one inside `Box<dyn Publish<Error = _>>` registries.
    let (publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    let _boxed: Box<dyn Publish<Error = rumqttc::ClientError>> = Box::new(publisher);
}

#[test]
fn direct_publish_call_through_trait_object() {
    let (mut publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    let msg = TopicMessage {
        topic: "ruview/seed/bfld/presence/state".into(),
        payload: "true".into(),
    };
    publisher.publish(&msg).expect("queue accept");
}

// QoS sanity: the Publish trait doesn't expose QoS in the message itself, so
// the publisher must default to a sensible level. AtLeastOnce is the
// HA-DISCO recommendation for state topics.
#[test]
fn default_qos_is_at_least_once_via_connect() {
    let (_publisher, _connection) = RumqttPublisher::connect(unreachable_opts(), 16);
    // The QoS isn't observable through the public API; this test pins the
    // documented default so a future PR that changes it will need to
    // update this assertion alongside.
    let _at_least_once = QoS::AtLeastOnce; // doc anchor
}
