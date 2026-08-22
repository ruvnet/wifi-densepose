//! Live-broker integration test for `RumqttPublisher`. ADR-122 §2.2 end-to-end.
//!
//! **Skipped silently when `BFLD_MQTT_BROKER` is unset**, so CI runs that lack
//! a broker stay green. Locally:
//!
//! ```text
//! scoop install mosquitto
//! mosquitto -v -c mosquitto-allow-anon.conf &
//! BFLD_MQTT_BROKER=tcp://localhost:1883 \
//!   cargo test -p wifi-densepose-bfld --features mqtt --test mosquitto_integration
//! ```
//!
//! Test discipline (per `feedback_mqtt_integration_test_patterns` memory):
//! - per-test unique `client_id` (current nanosecond timestamp suffix)
//! - subscriber eventloop pumped until SubAck arrives before publishing
//! - explicit `wait_for_n_messages` with timeout — never `loop { iter.recv() }`

#![cfg(feature = "mqtt")]

use std::env;
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rumqttc::{Client, Event, Incoming, MqttOptions, Packet, QoS};
use wifi_densepose_bfld::{publish_event, BfldEvent, PrivacyClass, RumqttPublisher};

const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(5);
const RECEIVE_TIMEOUT: Duration = Duration::from_secs(10);

fn broker_env() -> Option<(String, u16)> {
    let raw = env::var("BFLD_MQTT_BROKER").ok()?;
    let raw = raw.strip_prefix("tcp://").unwrap_or(&raw);
    let mut parts = raw.splitn(2, ':');
    let host = parts.next()?.to_string();
    let port: u16 = parts.next().unwrap_or("1883").parse().ok()?;
    Some((host, port))
}

fn unique_client_id(prefix: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}-{nanos}")
}

fn sample_event(node_id: &str) -> BfldEvent {
    BfldEvent::with_privacy_gating(
        node_id.into(),
        1_700_000_000_000_000_000,
        true,
        0.62,
        2,
        0.88,
        Some("test_zone".into()),
        PrivacyClass::Anonymous,
        Some(0.34),
        Some([0xAB; 32]),
    )
}

/// Spawn a subscriber + a pump thread. Returns the receiver of incoming
/// `(topic, payload)` pairs and a oneshot signalling SubAck arrival.
fn spawn_subscriber(
    host: &str,
    port: u16,
    topic_filter: &str,
) -> (Receiver<(String, String)>, Receiver<()>) {
    let mut opts = MqttOptions::new(unique_client_id("bfld-sub"), (host, port));
    opts.set_keep_alive(5);
    let (client, mut connection) = Client::builder(opts).capacity(64).build();
    client
        .subscribe(topic_filter, QoS::AtLeastOnce)
        .expect("subscribe enqueue");

    let (incoming_tx, incoming_rx) = channel();
    let (suback_tx, suback_rx) = channel();
    thread::spawn(move || {
        // rumqttc-v4-next stops the connection once every request sender is
        // dropped. Keep the subscriber client alive for as long as its pump
        // thread runs; otherwise the broker sees a clean disconnect directly
        // after SUBACK and no subsequent publications can be delivered.
        let _client_guard = client;
        for notification in connection.iter() {
            match notification {
                Ok(Event::Incoming(Packet::SubAck(_))) => {
                    let _ = suback_tx.send(());
                }
                Ok(Event::Incoming(Incoming::Publish(p))) => {
                    let topic = String::from_utf8_lossy(&p.topic).to_string();
                    let payload = String::from_utf8_lossy(&p.payload).to_string();
                    if incoming_tx.send((topic, payload)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
                _ => {}
            }
        }
    });
    (incoming_rx, suback_rx)
}

fn collect_messages(
    rx: &Receiver<(String, String)>,
    expected_count: usize,
    timeout: Duration,
) -> Vec<(String, String)> {
    let deadline = Instant::now() + timeout;
    let mut out = Vec::with_capacity(expected_count);
    while out.len() < expected_count {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok(msg) => out.push(msg),
            Err(RecvTimeoutError::Timeout) => break,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    out
}

#[test]
fn live_broker_anonymous_event_roundtrips_all_six_topics() {
    let Some((host, port)) = broker_env() else {
        eprintln!(
            "BFLD_MQTT_BROKER unset — skipping live mosquitto roundtrip test. \
             Set e.g. BFLD_MQTT_BROKER=tcp://localhost:1883 to enable."
        );
        return;
    };

    let node_id = unique_client_id("seed");
    let filter = format!("ruview/{node_id}/bfld/+/state");

    // Subscriber first so it's ready before the publisher sends.
    let (incoming_rx, suback_rx) = spawn_subscriber(&host, port, &filter);
    suback_rx
        .recv_timeout(SUBSCRIBE_TIMEOUT)
        .expect("SubAck within 5s");

    // Publisher with its own connection. Spawn a thread iterating the
    // Connection so publishes actually reach the broker.
    let mut opts = MqttOptions::new(unique_client_id("bfld-pub"), (host.as_str(), port));
    opts.set_keep_alive(5);
    let (mut publisher, mut pub_connection) = RumqttPublisher::connect(opts, 64);
    thread::spawn(move || {
        for _ in pub_connection.iter() { /* drain protocol events */ }
    });

    // Give the publisher a brief moment to complete CONNECT before publish.
    thread::sleep(Duration::from_millis(200));

    let event = sample_event(&node_id);
    let count = publish_event(&mut publisher, &event).expect("queue publish");
    assert_eq!(count, 6, "Anonymous + zone publishes 6 topics");

    let messages = collect_messages(&incoming_rx, 6, RECEIVE_TIMEOUT);
    assert_eq!(
        messages.len(),
        6,
        "broker delivered {} of 6 expected messages",
        messages.len(),
    );

    // Topic correctness — every expected entity must appear exactly once.
    let topics: Vec<&str> = messages.iter().map(|(t, _)| t.as_str()).collect();
    for entity in [
        "presence",
        "motion",
        "person_count",
        "confidence",
        "zone_activity",
        "identity_risk",
    ] {
        assert!(
            topics
                .iter()
                .any(|t| t == &format!("ruview/{node_id}/bfld/{entity}/state").as_str()),
            "missing entity {entity} in delivered topics {topics:?}",
        );
    }
}

#[test]
fn live_broker_restricted_event_omits_identity_risk() {
    let Some((host, port)) = broker_env() else {
        eprintln!("BFLD_MQTT_BROKER unset — skipping");
        return;
    };

    let node_id = unique_client_id("seed-r");
    let filter = format!("ruview/{node_id}/bfld/+/state");

    let (incoming_rx, suback_rx) = spawn_subscriber(&host, port, &filter);
    suback_rx
        .recv_timeout(SUBSCRIBE_TIMEOUT)
        .expect("SubAck within 5s");

    let mut opts = MqttOptions::new(unique_client_id("bfld-pub-r"), (host.as_str(), port));
    opts.set_keep_alive(5);
    let (mut publisher, mut pub_connection) = RumqttPublisher::connect(opts, 64);
    thread::spawn(move || for _ in pub_connection.iter() {});
    thread::sleep(Duration::from_millis(200));

    let mut event = sample_event(&node_id);
    event.privacy_class = PrivacyClass::Restricted;
    event.apply_privacy_gating();
    publish_event(&mut publisher, &event).expect("queue publish");

    // Expect 5 messages: 6 entities minus identity_risk.
    let messages = collect_messages(&incoming_rx, 6, Duration::from_secs(3));
    assert_eq!(messages.len(), 5);
    assert!(
        !messages.iter().any(|(t, _)| t.contains("identity_risk")),
        "Restricted class must not publish identity_risk topic, got {messages:?}",
    );
}
