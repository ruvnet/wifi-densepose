//! Acceptance tests for ADR-122 §2.2 — MQTT topic routing.

#![cfg(feature = "std")]

use wifi_densepose_bfld::{render_events, BfldEvent, PrivacyClass, TopicMessage};

fn sample_event(class: PrivacyClass, with_zone: bool) -> BfldEvent {
    BfldEvent::with_privacy_gating(
        "seed-01".into(),
        1_700_000_000_000_000_000,
        true,
        0.72,
        2,
        0.91,
        if with_zone { Some("living_room".into()) } else { None },
        class,
        Some(0.34),
        Some([0xAB; 32]),
    )
}

fn topics_for(class: PrivacyClass) -> Vec<String> {
    render_events(&sample_event(class, true))
        .into_iter()
        .map(|m| m.topic)
        .collect()
}

// --- topic shape ---------------------------------------------------------

#[test]
fn topic_format_is_ruview_node_bfld_entity_state() {
    let t = TopicMessage::ruview_topic("seed-42", "presence");
    assert_eq!(t, "ruview/seed-42/bfld/presence/state");
}

#[test]
fn anonymous_class_publishes_six_topics_with_zone() {
    let topics = topics_for(PrivacyClass::Anonymous);
    assert_eq!(topics.len(), 6, "got {topics:?}");
    let expected: Vec<&str> = vec![
        "ruview/seed-01/bfld/presence/state",
        "ruview/seed-01/bfld/motion/state",
        "ruview/seed-01/bfld/person_count/state",
        "ruview/seed-01/bfld/confidence/state",
        "ruview/seed-01/bfld/zone_activity/state",
        "ruview/seed-01/bfld/identity_risk/state",
    ];
    for t in &expected {
        assert!(topics.contains(&t.to_string()), "missing topic {t}");
    }
}

#[test]
fn anonymous_class_without_zone_omits_zone_activity_topic() {
    let topics: Vec<String> = render_events(&sample_event(PrivacyClass::Anonymous, false))
        .into_iter()
        .map(|m| m.topic)
        .collect();
    assert!(!topics.iter().any(|t| t.contains("zone_activity")));
    assert_eq!(topics.len(), 5);
}

// --- class-gated routing -------------------------------------------------

#[test]
fn restricted_class_omits_identity_risk_topic() {
    let topics = topics_for(PrivacyClass::Restricted);
    assert!(
        !topics.iter().any(|t| t.contains("identity_risk")),
        "Restricted (class 3) must NOT publish identity_risk: {topics:?}",
    );
    // Other entities still present.
    assert!(topics.iter().any(|t| t.contains("presence")));
    assert!(topics.iter().any(|t| t.contains("motion")));
}

#[test]
fn raw_and_derived_classes_publish_nothing() {
    // Raw (0) and Derived (1) are local-only / research — never on the
    // public topic tree.
    let raw = render_events(&sample_event(PrivacyClass::Raw, true));
    assert!(raw.is_empty(), "Raw class must publish nothing");
    let derived = render_events(&sample_event(PrivacyClass::Derived, true));
    assert!(derived.is_empty(), "Derived class must publish nothing");
}

// --- payload shape -------------------------------------------------------

#[test]
fn presence_payload_is_lowercase_json_bool() {
    let msgs = render_events(&sample_event(PrivacyClass::Anonymous, false));
    let pres = msgs
        .iter()
        .find(|m| m.topic.contains("presence"))
        .expect("presence topic");
    assert_eq!(pres.payload, "true");
}

#[test]
fn motion_payload_is_fixed_precision_decimal() {
    let msgs = render_events(&sample_event(PrivacyClass::Anonymous, false));
    let motion = msgs
        .iter()
        .find(|m| m.topic.contains("motion"))
        .expect("motion topic");
    assert_eq!(motion.payload, "0.720000");
}

#[test]
fn person_count_payload_is_bare_integer() {
    let msgs = render_events(&sample_event(PrivacyClass::Anonymous, false));
    let pc = msgs
        .iter()
        .find(|m| m.topic.contains("person_count"))
        .expect("person_count topic");
    assert_eq!(pc.payload, "2");
}

#[test]
fn zone_payload_is_json_string_with_quotes() {
    let msgs = render_events(&sample_event(PrivacyClass::Anonymous, true));
    let zone = msgs
        .iter()
        .find(|m| m.topic.contains("zone_activity"))
        .expect("zone_activity topic");
    assert_eq!(zone.payload, "\"living_room\"");
}

#[test]
fn zone_payload_escapes_json_metacharacters() {
    // A zone name containing a double-quote or backslash must not break out of
    // the JSON string literal it is emitted into. ha_discovery.rs already
    // escapes operator-controlled strings via push_str_field; render_events
    // must do the same for parity so the state-topic payload is always valid
    // JSON that Home Assistant can parse.
    let ev = BfldEvent::with_privacy_gating(
        "seed-01".into(),
        0,
        true,
        0.1,
        1,
        0.9,
        Some(r#"living"room\back"#.into()),
        PrivacyClass::Anonymous,
        None,
        None,
    );
    let msgs = render_events(&ev);
    let zone = msgs
        .iter()
        .find(|m| m.topic.contains("zone_activity"))
        .expect("zone_activity topic");
    // Expected: the inner quote and backslash are backslash-escaped, wrapped in
    // one pair of unescaped delimiter quotes -> a single valid JSON string.
    assert_eq!(zone.payload, r#""living\"room\\back""#);
    // And it must parse as JSON back to the original zone string.
    let parsed: String = serde_json::from_str(&zone.payload).expect("valid JSON string");
    assert_eq!(parsed, r#"living"room\back"#);
}

#[test]
fn identity_risk_payload_is_fixed_precision_decimal() {
    let msgs = render_events(&sample_event(PrivacyClass::Anonymous, false));
    let risk = msgs
        .iter()
        .find(|m| m.topic.contains("identity_risk"))
        .expect("identity_risk topic");
    assert_eq!(risk.payload, "0.340000");
}
