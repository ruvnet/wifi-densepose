//! MQTT connection lifecycle + topic publication (ADR-115 §2 / §3.5 / §3.6).
//!
//! Gated behind `--features mqtt` because it pulls in `rumqttc`. The
//! consumer is the broadcast channel `sensing-server` already writes to
//! in `main.rs` (the same channel the WebSocket handler subscribes to —
//! see ADR-115 §1 for the message types).
//!
//! ## Lifecycle
//!
//! 1. **Connect**: build [`rumqttc::MqttOptions`] from [`MqttConfig`],
//!    install LWT on every entity's availability topic, set keepalive.
//! 2. **Discovery**: emit one retained discovery `config` topic per
//!    enabled entity per known node. Re-emit every `refresh_secs`.
//! 3. **Availability heartbeat**: publish `online` retained on every
//!    availability topic on connect, and re-publish every 30 s so HA can
//!    detect zombie sessions.
//! 4. **State publication**: subscribe to the broadcast channel; for
//!    each inbound message project it into a [`VitalsSnapshot`], pass
//!    through the privacy filter, gate by [`RateLimiter`], encode via
//!    [`StateEncoder`], publish.
//!
//! ## Reconnect strategy
//!
//! `rumqttc::EventLoop` reconnects automatically with backoff. After a
//! successful reconnect we re-publish discovery (retained config topics
//! survive at the broker, but a fresh HA install that came online after
//! we last refreshed needs them) and reset the rate limiter so the
//! first post-reconnect sample emits promptly.

use std::sync::Arc;
use std::time::{Duration, Instant};

use rumqttc::{AsyncClient, ClientError, EventLoop, MqttOptions, QoS, Transport};
use tokio::sync::broadcast;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use super::config::{MqttConfig, TlsConfig};
use super::discovery::{DiscoveryBuilder, EntityKind};
use super::state::{RateLimiter, StateEncoder, StateMessage, VitalsSnapshot};

/// Heartbeat cadence for availability re-publication (per §3.6).
const AVAILABILITY_HEARTBEAT: Duration = Duration::from_secs(30);

/// Build a `rumqttc::MqttOptions` from validated [`MqttConfig`].
fn build_mqtt_options(cfg: &MqttConfig) -> MqttOptions {
    let mut opts = MqttOptions::new(&cfg.client_id, &cfg.host, cfg.port);
    opts.set_keep_alive(Duration::from_secs(30));
    opts.set_clean_session(true);

    if let (Some(u), Some(p)) = (cfg.username.as_deref(), cfg.password.as_deref()) {
        opts.set_credentials(u, p);
    } else if let Some(u) = cfg.username.as_deref() {
        opts.set_credentials(u, "");
    }

    if !matches!(cfg.tls, TlsConfig::Off) {
        // We always use rustls (matches `ureq` in this crate). The
        // specific cert / CA wiring is done by the runtime constructor;
        // here we just flip the transport.
        opts.set_transport(Transport::tls_with_default_config());
    }

    opts
}

/// One node's per-entity availability topics, pre-computed at startup so
/// the heartbeat loop doesn't allocate per tick.
struct NodeAvailability {
    online_topics: Vec<String>,
}

impl NodeAvailability {
    fn for_builder(b: &DiscoveryBuilder<'_>, entities: &[EntityKind]) -> Self {
        let online_topics = entities
            .iter()
            .map(|e| b.availability_topic(*e))
            .collect();
        Self { online_topics }
    }
}

/// Spawn the MQTT publisher background task. Returns the join handle so
/// the caller can `await` it on shutdown. Errors during connection are
/// retried internally by `rumqttc::EventLoop`.
pub fn spawn(
    cfg: Arc<MqttConfig>,
    builder_owned: OwnedDiscoveryBuilder,
    state_rx: broadcast::Receiver<VitalsSnapshot>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        run(cfg, builder_owned, state_rx).await;
    })
}

/// Owned twin of [`DiscoveryBuilder`] so the publisher task doesn't need
/// to borrow from a stack frame the user holds. Cloned cheaply per
/// reconnect.
#[derive(Debug, Clone)]
pub struct OwnedDiscoveryBuilder {
    pub discovery_prefix: String,
    pub node_id: String,
    pub node_friendly_name: Option<String>,
    pub sw_version: String,
    pub model: String,
    pub via_device: Option<String>,
}

impl OwnedDiscoveryBuilder {
    pub fn as_borrowed(&self) -> DiscoveryBuilder<'_> {
        DiscoveryBuilder {
            discovery_prefix: &self.discovery_prefix,
            node_id: &self.node_id,
            node_friendly_name: self.node_friendly_name.as_deref(),
            sw_version: &self.sw_version,
            model: &self.model,
            via_device: self.via_device.as_deref(),
        }
    }

    /// Derive a per-node builder from this base (issue #898). Each physical
    /// RuView node must surface as its own Home-Assistant device — the base
    /// builder's `node_id` (the MQTT client id) is replaced with the actual
    /// node id, giving a distinct `wifi_densepose_<node>` device identifier
    /// and a per-node friendly name, instead of collapsing every node into a
    /// single hard-coded device.
    pub fn for_node(&self, node_id: &str) -> OwnedDiscoveryBuilder {
        OwnedDiscoveryBuilder {
            discovery_prefix: self.discovery_prefix.clone(),
            node_id: node_id.to_string(),
            node_friendly_name: Some(format!("RuView node {node_id}")),
            sw_version: self.sw_version.clone(),
            model: self.model.clone(),
            via_device: self.via_device.clone(),
        }
    }
}

/// Core run loop. Pumps the broadcast channel + the MQTT event loop in
/// the same `select!` so we never block one on the other.
async fn run(
    cfg: Arc<MqttConfig>,
    builder_owned: OwnedDiscoveryBuilder,
    mut state_rx: broadcast::Receiver<VitalsSnapshot>,
) {
    let opts = build_mqtt_options(&cfg);
    let (client, mut eventloop): (AsyncClient, EventLoop) = AsyncClient::new(opts, 256);

    let entities = DiscoveryBuilder::enabled_entities(
        cfg.privacy_mode,
        cfg.publish_pose,
        &[], // no_semantic — wire from cli::Args in P3.5
    );

    // #898: one Home-Assistant device per node. Discovery + availability are
    // published lazily the first time a snapshot for a given node_id arrives;
    // each node's builder + availability are retained here for heartbeats and
    // the offline LWT. (Previously a single hard-coded builder collapsed every
    // node into one device.)
    let mut nodes: std::collections::HashMap<String, (OwnedDiscoveryBuilder, NodeAvailability)> =
        std::collections::HashMap::new();

    let mut rate_limiter = RateLimiter::new();
    let mut last_heartbeat = Instant::now();
    let mut last_refresh = Instant::now();
    let start_instant = Instant::now();

    info!(
        host = %cfg.host,
        port = cfg.port,
        prefix = %cfg.discovery_prefix,
        entities = entities.len(),
        privacy = cfg.privacy_mode,
        "[mqtt] publisher started",
    );

    loop {
        tokio::select! {
            biased;

            // Pump the rumqttc event loop. Errors trigger automatic
            // reconnect; we just log and continue.
            ev = eventloop.poll() => {
                match ev {
                    Ok(_) => {}
                    Err(e) => {
                        error!("[mqtt] event loop error, will reconnect: {e}");
                        rate_limiter.reset();
                        // Brief backoff before next poll attempt.
                        tokio::time::sleep(Duration::from_millis(500)).await;
                    }
                }
            }

            // Periodic heartbeat / discovery refresh.
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                if last_heartbeat.elapsed() >= AVAILABILITY_HEARTBEAT {
                    for (_, na) in nodes.values() {
                        if let Err(e) = publish_availability(&client, na, "online").await {
                            warn!("[mqtt] heartbeat publish failed: {e}");
                        }
                    }
                    last_heartbeat = Instant::now();
                }
                if last_refresh.elapsed() >= Duration::from_secs(cfg.refresh_secs) {
                    for (nb, _) in nodes.values() {
                        if let Err(e) =
                            publish_all_discovery(&client, &nb.as_borrowed(), &entities).await
                        {
                            warn!("[mqtt] discovery refresh failed: {e}");
                        }
                    }
                    last_refresh = Instant::now();
                }
            }

            // Inbound state snapshot from the rest of sensing-server.
            recv = state_rx.recv() => {
                match recv {
                    Ok(snap) => {
                        let elapsed = start_instant.elapsed();
                        // #898: on first sight of a node_id, publish that
                        // node's discovery + availability; then route its
                        // state to per-node topics.
                        if !nodes.contains_key(&snap.node_id) {
                            let nb = builder_owned.for_node(&snap.node_id);
                            let borrowed = nb.as_borrowed();
                            if let Err(e) =
                                publish_all_discovery(&client, &borrowed, &entities).await
                            {
                                warn!("[mqtt] node {} discovery failed: {e}", snap.node_id);
                            }
                            let na = NodeAvailability::for_builder(&borrowed, &entities);
                            if let Err(e) = publish_availability(&client, &na, "online").await {
                                warn!("[mqtt] node {} availability failed: {e}", snap.node_id);
                            }
                            nodes.insert(snap.node_id.clone(), (nb, na));
                        }
                        let borrowed = nodes[&snap.node_id].0.as_borrowed();
                        publish_snapshot(&client, &borrowed, &snap, &cfg, &mut rate_limiter, elapsed).await;
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("[mqtt] lagged behind broadcast by {n} messages — dropped");
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        info!("[mqtt] broadcast channel closed, draining");
                        // Publish offline for every known node before exit.
                        for (_, na) in nodes.values() {
                            let _ = publish_availability(&client, na, "offline").await;
                        }
                        let _ = client.disconnect().await;
                        return;
                    }

                }
            }
        }
    }
}

async fn publish_all_discovery(
    client: &AsyncClient,
    b: &DiscoveryBuilder<'_>,
    entities: &[EntityKind],
) -> Result<(), ClientError> {
    for &e in entities {
        let cfg = b.build(e);
        let topic = b.config_topic(e);
        let payload = serde_json::to_string(&cfg).expect("discovery payload always serialises");
        client.publish(&topic, QoS::AtLeastOnce, true, payload).await?;
    }
    Ok(())
}

async fn publish_availability(
    client: &AsyncClient,
    avail: &NodeAvailability,
    state: &str,
) -> Result<(), ClientError> {
    for topic in &avail.online_topics {
        client.publish(topic, QoS::AtLeastOnce, true, state).await?;
    }
    Ok(())
}

async fn publish_snapshot(
    client: &AsyncClient,
    b: &DiscoveryBuilder<'_>,
    snap: &VitalsSnapshot,
    cfg: &MqttConfig,
    rl: &mut RateLimiter,
    elapsed: Duration,
) {
    let encoder = StateEncoder { builder: b };

    // Binary: presence (change-only — caller is responsible for detecting
    // change, but we always publish here because broadcast already debounces
    // and HA will dedup retained equal values harmlessly).
    if let Some(m) = encoder.boolean(EntityKind::Presence, snap.presence) {
        let _ = publish_state(client, &m).await;
    }

    // Event: fall.
    if snap.fall_detected {
        if let Some(m) = encoder.event(
            EntityKind::FallDetected,
            "fall_detected",
            snap.timestamp_ms,
            Some(snap.vital_confidence),
        ) {
            let _ = publish_state(client, &m).await;
        }
    }

    // Numeric rate-limited entities.
    for (entity, allowed) in [
        (EntityKind::PersonCount, rl.allow(EntityKind::PersonCount, elapsed, &cfg.rates)),
        (EntityKind::HeartRate, !cfg.privacy_mode && rl.allow(EntityKind::HeartRate, elapsed, &cfg.rates)),
        (EntityKind::BreathingRate, !cfg.privacy_mode && rl.allow(EntityKind::BreathingRate, elapsed, &cfg.rates)),
        (EntityKind::MotionLevel, rl.allow(EntityKind::MotionLevel, elapsed, &cfg.rates)),
        (EntityKind::MotionEnergy, rl.allow(EntityKind::MotionEnergy, elapsed, &cfg.rates)),
        (EntityKind::PresenceScore, rl.allow(EntityKind::PresenceScore, elapsed, &cfg.rates)),
        (EntityKind::Rssi, rl.allow(EntityKind::Rssi, elapsed, &cfg.rates)),
    ] {
        if !allowed {
            continue;
        }
        if let Some(m) = encoder.numeric(entity, snap) {
            let _ = publish_state(client, &m).await;
        }
    }
}

async fn publish_state(client: &AsyncClient, m: &StateMessage) -> Result<(), ClientError> {
    let qos = match m.qos {
        0 => QoS::AtMostOnce,
        1 => QoS::AtLeastOnce,
        _ => QoS::ExactlyOnce,
    };
    client.publish(&m.topic, qos, m.retain, m.payload.clone()).await
}

#[cfg(test)]
mod per_node_device_tests {
    //! Issue #898 — each physical node must surface as its own Home-Assistant
    //! device, not collapse into one hard-coded device.
    use super::*;

    fn base() -> OwnedDiscoveryBuilder {
        OwnedDiscoveryBuilder {
            discovery_prefix: "homeassistant".into(),
            node_id: "wifi-densepose-1".into(),
            node_friendly_name: Some("RuView".into()),
            sw_version: "0.0.0".into(),
            model: "test".into(),
            via_device: None,
        }
    }

    fn device_identifiers(b: &OwnedDiscoveryBuilder) -> Vec<String> {
        b.as_borrowed().build(EntityKind::Presence).device.identifiers
    }

    #[test]
    fn for_node_overrides_node_id_and_friendly_name() {
        let n = base().for_node("node-A");
        assert_eq!(n.node_id, "node-A");
        assert_eq!(n.node_friendly_name.as_deref(), Some("RuView node node-A"));
    }

    #[test]
    fn distinct_nodes_yield_distinct_ha_device_identifiers() {
        let b = base();
        let a = device_identifiers(&b.for_node("node-A"));
        let c = device_identifiers(&b.for_node("node-B"));
        assert_eq!(a, vec!["wifi_densepose_node-A".to_string()]);
        assert_eq!(c, vec!["wifi_densepose_node-B".to_string()]);
        assert_ne!(a, c, "#898: two nodes must not collapse into one device");
    }

    #[test]
    fn single_node_keeps_a_stable_identity() {
        // Two snapshots from the same node map to the same device.
        let b = base();
        assert_eq!(
            device_identifiers(&b.for_node("node-7")),
            device_identifiers(&b.for_node("node-7"))
        );
    }
}
