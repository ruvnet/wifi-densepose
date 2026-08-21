//! `RumqttPublisher` — production [`Publish`] impl backed by `rumqttc`.
//! ADR-122 §2.2 broker integration.
//!
//! Gated on `feature = "mqtt"`. The sync `rumqttc::Client` is used so the
//! `Publish` trait's sync method signature is honored without a tokio runtime.
//! The companion `rumqttc::Connection` returned by [`RumqttPublisher::connect`]
//! must be pumped by the caller (typically on a dedicated thread) to drive
//! the MQTT protocol — published messages remain queued until the connection
//! sends them.
//!
//! ```ignore
//! use std::thread;
//! use wifi_densepose_bfld::{publish_event, RumqttPublisher};
//! use rumqttc::MqttOptions;
//!
//! let opts = MqttOptions::new("seed-01", ("broker.local", 1883));
//! let (mut publisher, mut connection) = RumqttPublisher::connect(opts, 100);
//! thread::spawn(move || for _ in connection.iter() { /* drain */ });
//! // ... build BfldEvent ...
//! publish_event(&mut publisher, &event).expect("mqtt publish");
//! ```

#![cfg(feature = "mqtt")]

use rumqttc::{Client, Connection, LastWill, MqttOptions, PublishOptions, QoS};

use crate::availability::{availability_topic, PAYLOAD_NOT_AVAILABLE};
use crate::mqtt_topics::{Publish, TopicMessage};

/// Sync MQTT publisher wrapping [`rumqttc::Client`].
pub struct RumqttPublisher {
    client: Client,
    qos: QoS,
    retain: bool,
}

impl RumqttPublisher {
    /// Wrap an existing `Client` at the supplied QoS. `retain = false` matches
    /// HA-DISCO state-topic semantics (retained payloads cause stale-state
    /// flapping on broker reconnect). For availability-style topics callers
    /// should construct a separate publisher with `retain = true`.
    #[must_use]
    pub const fn new(client: Client, qos: QoS) -> Self {
        Self {
            client,
            qos,
            retain: false,
        }
    }

    /// Toggle the per-publisher `retain` flag.
    #[must_use]
    pub const fn with_retain(mut self, retain: bool) -> Self {
        self.retain = retain;
        self
    }

    /// Build a publisher + an unpumped `Connection`. Caller is responsible
    /// for spawning a thread that iterates the connection (typical pattern
    /// shown in the module-level doc example).
    #[must_use]
    pub fn connect(opts: MqttOptions, capacity: usize) -> (Self, Connection) {
        let (client, connection) = Client::builder(opts).capacity(capacity).build();
        (Self::new(client, QoS::AtLeastOnce), connection)
    }

    /// Like [`Self::connect`] but also configures the MQTT Last Will and
    /// Testament so the broker auto-publishes `"offline"` on
    /// `ruview/<node_id>/bfld/availability` (retained, QoS 1) when the
    /// publisher's TCP session drops without a clean DISCONNECT.
    ///
    /// Pairs with [`crate::publish_availability_online`] — call that on first
    /// CONNECT to set `"online"`; the LWT covers the disconnect path.
    #[must_use]
    pub fn connect_with_lwt(
        node_id: &str,
        opts: MqttOptions,
        capacity: usize,
    ) -> (Self, Connection) {
        let opts = with_lwt(opts, node_id);
        Self::connect(opts, capacity)
    }
}

/// Mutate `opts` to attach the BFLD availability LWT. Public so callers that
/// build their own `MqttOptions` (custom tls, credentials, etc.) can still
/// opt in to the LWT without using `connect_with_lwt`.
#[must_use]
pub fn with_lwt(mut opts: MqttOptions, node_id: &str) -> MqttOptions {
    // LastWill::new takes (topic, message, qos, retain).
    // retain = true so HA sees "offline" on next start even if the session
    // dropped while HA was down.
    let will = LastWill::new(
        availability_topic(node_id),
        PAYLOAD_NOT_AVAILABLE,
        QoS::AtLeastOnce,
        true,
    );
    opts.set_last_will(will);
    opts
}

impl Publish for RumqttPublisher {
    type Error = rumqttc::ClientError;

    fn publish(&mut self, msg: &TopicMessage) -> Result<(), Self::Error> {
        self.client
            .publish(
                &msg.topic,
                msg.payload.as_bytes(),
                PublishOptions::new(self.qos).retain(self.retain),
            )
    }
}
