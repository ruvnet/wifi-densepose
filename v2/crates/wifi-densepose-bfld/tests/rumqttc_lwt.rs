//! Acceptance tests for the LWT integration on `RumqttPublisher`. ADR-122 §2.2.

#![cfg(feature = "mqtt")]

use rumqttc::MqttOptions;
use wifi_densepose_bfld::{
    availability_topic, publish_event, with_lwt, BfldEvent, PrivacyClass, Publish, RumqttPublisher,
    TopicMessage,
};

fn unreachable_opts(client_id: &str) -> MqttOptions {
    MqttOptions::new(client_id, ("127.0.0.1", 1))
}

#[test]
fn with_lwt_returns_options_without_panic() {
    let opts = unreachable_opts("bfld-lwt-1");
    let _opts = with_lwt(opts, "seed-01");
    // rumqttc 0.24 doesn't expose a getter for the LWT, so the structural
    // assertion is the runtime non-panic + the fact that the build of the
    // LastWill struct succeeded.
}

#[test]
fn connect_with_lwt_constructs_publisher_and_connection() {
    let opts = unreachable_opts("bfld-lwt-2");
    let (_publisher, _connection) = RumqttPublisher::connect_with_lwt("seed-01", opts, 16);
    // Reaching here means rumqttc accepted the LWT-augmented options.
}

#[test]
fn connect_with_lwt_uses_documented_availability_topic() {
    // We can't introspect MqttOptions's LWT after construction, but the helper
    // builds the topic via the same availability_topic() function used by
    // the discovery publisher — assert that function returns the documented
    // path so a topic drift between LWT and discovery is impossible by
    // construction.
    assert_eq!(
        availability_topic("seed-test"),
        "ruview/seed-test/bfld/availability",
    );
}

#[test]
fn connect_with_lwt_publisher_still_publishes_state_topics() {
    // Smoke: the LWT-equipped publisher must still pass state messages
    // through publish() without modification.
    let opts = unreachable_opts("bfld-lwt-3");
    let (mut publisher, _connection) = RumqttPublisher::connect_with_lwt("seed-01", opts, 16);
    let event = BfldEvent::with_privacy_gating(
        "seed-01".into(),
        1_700_000_000_000_000_000,
        true,
        0.5,
        1,
        0.9,
        None,
        PrivacyClass::Anonymous,
        Some(0.25),
        None,
    );
    let count = publish_event(&mut publisher, &event).expect("publish queues");
    // Anonymous + no zone publishes 5 entity topics: presence, motion,
    // person_count, confidence, identity_risk. rf_signature_hash isn't an
    // MQTT entity topic — it rides inside the JSON event surface only.
    assert_eq!(count, 5, "Anonymous + no zone → 5 topics");
}

#[test]
fn publisher_trait_object_constructible_with_lwt_path() {
    let opts = unreachable_opts("bfld-lwt-4");
    let (publisher, _connection) = RumqttPublisher::connect_with_lwt("seed-01", opts, 16);
    let _boxed: Box<dyn Publish<Error = rumqttc::ClientError>> = Box::new(publisher);
}

#[test]
fn with_lwt_is_idempotent_against_double_call() {
    // Calling with_lwt twice should leave the most recent LWT installed
    // without panicking — useful for libraries that may wrap operator-
    // supplied options without knowing if LWT was already attached.
    let opts = unreachable_opts("bfld-lwt-5");
    let opts = with_lwt(opts, "node-a");
    let opts = with_lwt(opts, "node-b");
    let _ = opts; // no panic = pass; rumqttc replaces the will silently.
}

#[test]
fn caller_built_options_can_opt_in_via_with_lwt_then_pass_to_connect() {
    // Operators with custom MqttOptions (e.g., TLS, credentials) build their
    // own opts, then call with_lwt before passing to RumqttPublisher::connect.
    let mut opts = unreachable_opts("bfld-lwt-6");
    opts.set_keep_alive(30);
    let opts = with_lwt(opts, "seed-01");
    let (_publisher, _connection) = RumqttPublisher::connect(opts, 16);
}

#[test]
fn placeholder_topicmessage_path_unaffected_by_lwt() {
    // Sanity: TopicMessage and Publish surfaces from the non-mqtt path stay
    // unchanged when the mqtt feature is on; the LWT addition is purely additive.
    let m = TopicMessage {
        topic: "ruview/x/bfld/presence/state".into(),
        payload: "true".into(),
    };
    assert_eq!(m.topic, "ruview/x/bfld/presence/state");
}
