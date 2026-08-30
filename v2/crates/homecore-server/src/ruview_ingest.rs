//! Privacy-bounded RuView semantic ingest for HomeCore.
//!
//! This boundary deliberately consumes only the two Apple-Home projection
//! endpoints. It never requests or deserializes CSI, CIR, pose, camera, LiDAR,
//! identity, or waveform payloads. Valid snapshots become ordinary HomeCore
//! binary sensors, so the existing state-change listener is the only path into
//! the optional HAP bridge.

use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result};
use homecore::{Context, EntityId, HomeCore};
use reqwest::{Client, StatusCode, Url};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::watch;
use tracing::{info, warn};

const MAX_BODY_BYTES: usize = 64 * 1024;
const MAX_NODE_ID_BYTES: usize = 64;
const MAX_SOURCE_BYTES: usize = 128;
const MAX_PERSONS: u64 = 1_000;
const MAX_MOTION: f64 = 1_000.0;
const MOTION_THRESHOLD: f64 = 0.1;

const FORBIDDEN_FIELDS: &[&str] = &[
    "raw_csi",
    "csi",
    "cir",
    "rf_tensors",
    "recordings",
    "pose",
    "pose_frames",
    "camera",
    "lidar",
    "vital_waveforms",
    "identity",
    "identity_observations",
    "identity_risk_score",
    "soul_match_probability",
    "rf_signature_hash",
];

/// Secret wrapper whose `Debug` output can safely appear in derived configs.
#[derive(Clone)]
pub(crate) struct Secret(String);

impl Secret {
    pub(crate) fn new(value: String) -> Result<Self> {
        if value.trim().is_empty() || value.len() > 8 * 1024 {
            anyhow::bail!("RuView read token must be non-empty and at most 8192 bytes");
        }
        Ok(Self(value))
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Secret([REDACTED])")
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RuViewIngestConfig {
    pub(crate) base_url: Url,
    pub(crate) node_id: String,
    pub(crate) token: Secret,
    pub(crate) poll_interval: Duration,
    pub(crate) timeout: Duration,
    pub(crate) max_staleness: Duration,
}

impl RuViewIngestConfig {
    pub(crate) fn validate(
        base_url: &str,
        node_id: String,
        token: String,
        poll_interval: Duration,
        timeout: Duration,
        max_staleness: Duration,
    ) -> Result<Self> {
        let mut base_url = Url::parse(base_url).context("invalid RuView base URL")?;
        if !matches!(base_url.scheme(), "http" | "https") {
            anyhow::bail!("RuView base URL must use http or https");
        }
        if base_url.scheme() == "http" {
            let is_loopback = base_url.host_str().is_some_and(|host| {
                host.eq_ignore_ascii_case("localhost")
                    || host
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|address| address.is_loopback())
            });
            if !is_loopback {
                anyhow::bail!(
                    "RuView HTTP is permitted only for a loopback origin; use HTTPS remotely"
                );
            }
        }
        if !base_url.username().is_empty() || base_url.password().is_some() {
            anyhow::bail!("RuView base URL must not contain credentials");
        }
        base_url.set_query(None);
        base_url.set_fragment(None);
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }
        validate_node_id(&node_id)?;
        if !(Duration::from_millis(250)..=Duration::from_secs(60)).contains(&poll_interval) {
            anyhow::bail!("RuView poll interval must be between 250ms and 60s");
        }
        if !(Duration::from_millis(100)..=Duration::from_secs(30)).contains(&timeout) {
            anyhow::bail!("RuView timeout must be between 100ms and 30s");
        }
        if !(Duration::from_millis(500)..=Duration::from_secs(300)).contains(&max_staleness) {
            anyhow::bail!("RuView max staleness must be between 500ms and 5m");
        }
        Ok(Self {
            base_url,
            node_id,
            token: Secret::new(token)?,
            poll_interval,
            timeout,
            max_staleness,
        })
    }
}

pub(crate) struct RuViewIngestRuntime {
    stop_tx: watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl RuViewIngestRuntime {
    pub(crate) async fn shutdown(self) {
        let _ = self.stop_tx.send(true);
        let _ = self.task.await;
    }
}

pub(crate) fn start(hc: HomeCore, config: RuViewIngestConfig) -> RuViewIngestRuntime {
    let (stop_tx, stop_rx) = watch::channel(false);
    let task = tokio::spawn(run(hc, config, stop_rx));
    RuViewIngestRuntime { stop_tx, task }
}

async fn run(hc: HomeCore, config: RuViewIngestConfig, mut stop_rx: watch::Receiver<bool>) {
    let client = match Client::builder()
        .timeout(config.timeout)
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            warn!(%error, "RuView ingest client initialization failed");
            return;
        }
    };
    info!(
        base_url = %config.base_url,
        node_id = %config.node_id,
        poll_ms = config.poll_interval.as_millis(),
        "RuView semantic ingest enabled"
    );
    let mut delay = Duration::ZERO;
    loop {
        if delay > Duration::ZERO {
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                changed = stop_rx.changed() => {
                    if changed.is_err() || *stop_rx.borrow() { break; }
                }
            }
        }
        if *stop_rx.borrow() {
            break;
        }
        match poll_once(&client, &config).await {
            Ok(snapshot) => {
                apply_snapshot(&hc, &config.node_id, snapshot);
                delay = config.poll_interval;
            }
            Err(error) => {
                remove_entities(&hc, &config.node_id);
                warn!(node_id = %config.node_id, %error, "RuView evidence unavailable; HomeCore projection removed");
                delay = if delay.is_zero() {
                    config.poll_interval
                } else {
                    delay.saturating_mul(2).min(Duration::from_secs(60))
                };
            }
        }
    }
    remove_entities(&hc, &config.node_id);
    info!(node_id = %config.node_id, "RuView semantic ingest stopped");
}

#[derive(Debug)]
struct Snapshot {
    timestamp_ms: u64,
    occupancy: bool,
    motion: bool,
    unexpected_occupancy: bool,
    unrecognized_activity_pattern: bool,
    privacy_class: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VitalsResponse {
    node_id: Value,
    timestamp_ms: u64,
    presence: bool,
    n_persons: u64,
    confidence: f64,
    breathing_rate_bpm: Option<f64>,
    heartrate_bpm: Option<f64>,
    motion: f64,
    source: String,
    privacy_class: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticResponse {
    node_id: Value,
    privacy_class: u8,
    events: SemanticEvents,
    redacted_fields: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticEvents {
    unknown_presence: SemanticEvent,
    unexpected_occupancy: SemanticEvent,
    unrecognized_activity_pattern: SemanticEvent,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticEvent {
    active: bool,
    source: String,
    ts: u64,
}

async fn poll_once(client: &Client, config: &RuViewIngestConfig) -> Result<Snapshot> {
    let vitals_url = endpoint(config, "vitals")?;
    let semantic_url = endpoint(config, "semantic-events")?;
    let (vitals, semantic) = tokio::try_join!(
        fetch_json(client, vitals_url, &config.token),
        fetch_json(client, semantic_url, &config.token),
    )?;
    reject_forbidden_fields(&vitals)?;
    reject_forbidden_fields(&semantic)?;
    let vitals: VitalsResponse = serde_json::from_value(vitals).context("invalid vitals schema")?;
    let semantic: SemanticResponse =
        serde_json::from_value(semantic).context("invalid semantic-events schema")?;
    let vitals_node = node_id_string(&vitals.node_id)?;
    let semantic_node = node_id_string(&semantic.node_id)?;
    if vitals_node != config.node_id || semantic_node != config.node_id {
        anyhow::bail!("RuView response node_id does not match configured node");
    }
    validate_privacy(vitals.privacy_class)?;
    validate_privacy(semantic.privacy_class)?;
    if vitals.privacy_class != semantic.privacy_class {
        anyhow::bail!("RuView endpoint privacy classes disagree");
    }
    validate_finite("confidence", vitals.confidence, 0.0, 1.0)?;
    validate_finite("motion", vitals.motion, 0.0, MAX_MOTION)?;
    if vitals.n_persons > MAX_PERSONS {
        anyhow::bail!("n_persons exceeds bounded range");
    }
    validate_optional_rate("breathing_rate_bpm", vitals.breathing_rate_bpm)?;
    validate_optional_rate("heartrate_bpm", vitals.heartrate_bpm)?;
    validate_source(&vitals.source)?;
    for event in [
        &semantic.events.unknown_presence,
        &semantic.events.unexpected_occupancy,
        &semantic.events.unrecognized_activity_pattern,
    ] {
        validate_source(&event.source)?;
        validate_fresh(event.ts, config.max_staleness)?;
    }
    validate_fresh(vitals.timestamp_ms, config.max_staleness)?;
    let event_ts = semantic.events.unknown_presence.ts;
    if event_ts.abs_diff(vitals.timestamp_ms) > config.max_staleness.as_millis() as u64 {
        anyhow::bail!("RuView endpoints are not temporally aligned");
    }
    if semantic.events.unknown_presence.active != vitals.presence {
        anyhow::bail!("RuView endpoint occupancy evidence disagrees");
    }
    // This is a required proof of server-side redaction, not permission to
    // ingest any of the named fields.
    for required in [
        "identity_risk_score",
        "soul_match_probability",
        "rf_signature_hash",
    ] {
        if !semantic
            .redacted_fields
            .iter()
            .any(|field| field == required)
        {
            anyhow::bail!("semantic-events response lacks required redaction declaration");
        }
    }
    Ok(Snapshot {
        timestamp_ms: vitals.timestamp_ms,
        occupancy: vitals.presence,
        motion: vitals.motion > MOTION_THRESHOLD,
        unexpected_occupancy: semantic.events.unexpected_occupancy.active,
        unrecognized_activity_pattern: semantic.events.unrecognized_activity_pattern.active,
        privacy_class: vitals.privacy_class,
    })
}

async fn fetch_json(client: &Client, url: Url, token: &Secret) -> Result<Value> {
    let mut response = client
        .get(url)
        .bearer_auth(token.expose())
        .header(reqwest::header::ACCEPT, "application/json")
        .send()
        .await
        .context("RuView request failed")?;
    if response.status() != StatusCode::OK {
        anyhow::bail!("RuView returned HTTP {}", response.status());
    }
    if let Some(length) = response.content_length() {
        if length > MAX_BODY_BYTES as u64 {
            anyhow::bail!("RuView response exceeds size limit");
        }
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed reading RuView response")?
    {
        if body.len().saturating_add(chunk.len()) > MAX_BODY_BYTES {
            anyhow::bail!("RuView response exceeds size limit");
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body).context("RuView response is not valid JSON")
}

fn endpoint(config: &RuViewIngestConfig, kind: &str) -> Result<Url> {
    config
        .base_url
        .join(&format!("api/v1/{kind}/{}/latest", config.node_id))
        .context("failed to construct RuView endpoint URL")
}

fn apply_snapshot(hc: &HomeCore, node_id: &str, snapshot: Snapshot) {
    let attributes = |device_class: &str, friendly_name: &str| {
        json!({
            "device_class": device_class,
            "friendly_name": friendly_name,
            "integration": "ruview",
            "node_id": node_id,
            "observed_at_ms": snapshot.timestamp_ms,
            "privacy_class": snapshot.privacy_class,
            "source": "ruview_semantic_projection",
        })
    };
    set_binary(
        hc,
        &entity_id(node_id, "occupancy"),
        snapshot.occupancy,
        attributes("occupancy", "RuView Occupancy"),
    );
    set_binary(
        hc,
        &entity_id(node_id, "motion"),
        snapshot.motion,
        attributes("motion", "RuView Motion"),
    );
    set_binary(
        hc,
        &entity_id(node_id, "unexpected_occupancy"),
        snapshot.unexpected_occupancy,
        attributes("occupancy", "RuView Unexpected Occupancy"),
    );
    set_binary(
        hc,
        &entity_id(node_id, "unrecognized_activity_pattern"),
        snapshot.unrecognized_activity_pattern,
        attributes("motion", "RuView Unrecognized Activity Pattern"),
    );
}

fn set_binary(hc: &HomeCore, id: &EntityId, active: bool, attributes: Value) {
    hc.states().set(
        id.clone(),
        if active { "on" } else { "off" },
        attributes,
        Context::with_user("homecore.ruview_ingest"),
    );
}

fn remove_entities(hc: &HomeCore, node_id: &str) {
    for suffix in [
        "occupancy",
        "motion",
        "unexpected_occupancy",
        "unrecognized_activity_pattern",
    ] {
        hc.states().remove(&entity_id(node_id, suffix));
    }
}

fn entity_id(node_id: &str, suffix: &str) -> EntityId {
    let slug: String = node_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    EntityId::parse(format!("binary_sensor.ruview_{slug}_{suffix}"))
        .expect("validated node id always produces a valid bounded entity id")
}

fn validate_node_id(node_id: &str) -> Result<()> {
    if node_id.is_empty()
        || node_id.len() > MAX_NODE_ID_BYTES
        || !node_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | ':'))
    {
        anyhow::bail!("RuView node id must be 1..=64 ASCII letters, digits, '-', '_', or ':'");
    }
    Ok(())
}

fn node_id_string(value: &Value) -> Result<String> {
    let node_id = match value {
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        _ => anyhow::bail!("RuView node_id must be a string or integer"),
    };
    validate_node_id(&node_id)?;
    Ok(node_id)
}

fn validate_privacy(value: u8) -> Result<()> {
    if matches!(value, 2 | 3) {
        Ok(())
    } else {
        anyhow::bail!("RuView privacy_class must be P2 or P3")
    }
}

fn validate_finite(name: &str, value: f64, min: f64, max: f64) -> Result<()> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        anyhow::bail!("RuView {name} is not finite and bounded")
    }
}

fn validate_optional_rate(name: &str, value: Option<f64>) -> Result<()> {
    if let Some(value) = value {
        validate_finite(name, value, 0.0, 300.0)?;
    }
    Ok(())
}

fn validate_source(source: &str) -> Result<()> {
    if source.is_empty() || source.len() > MAX_SOURCE_BYTES || source.chars().any(char::is_control)
    {
        anyhow::bail!("RuView source is empty, oversized, or contains control characters");
    }
    Ok(())
}

fn validate_fresh(timestamp_ms: u64, max_staleness: Duration) -> Result<()> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis() as u64;
    let future_tolerance = Duration::from_secs(5).as_millis() as u64;
    if timestamp_ms > now.saturating_add(future_tolerance)
        || now.saturating_sub(timestamp_ms) > max_staleness.as_millis() as u64
    {
        anyhow::bail!("RuView evidence is stale or implausibly future-dated");
    }
    Ok(())
}

fn reject_forbidden_fields(value: &Value) -> Result<()> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                let normalized = key.to_ascii_lowercase();
                if FORBIDDEN_FIELDS
                    .iter()
                    .any(|blocked| normalized == *blocked)
                {
                    anyhow::bail!("RuView response contains a prohibited field");
                }
                reject_forbidden_fields(value)?;
            }
        }
        Value::Array(values) => {
            for value in values {
                reject_forbidden_fields(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{extract::Path, http::HeaderMap, routing::get, Json, Router};
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tokio::sync::Mutex;

    #[derive(Clone)]
    struct MockState {
        mode: Arc<Mutex<&'static str>>,
        saw_auth: Arc<Mutex<bool>>,
    }

    async fn mock_vitals(
        Path(node): Path<String>,
        axum::extract::State(state): axum::extract::State<MockState>,
        headers: HeaderMap,
    ) -> (StatusCode, Json<Value>) {
        *state.saw_auth.lock().await = headers
            .get(reqwest::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            == Some("Bearer read-only-secret");
        if *state.mode.lock().await == "offline" {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "offline"})),
            );
        }
        let ts = now_ms();
        let mut body = json!({
            "node_id": node, "timestamp_ms": ts, "presence": true,
            "n_persons": 1, "confidence": 0.9, "breathing_rate_bpm": 18.0,
            "heartrate_bpm": 72.0, "motion": 0.5, "source": "esp32-csi",
            "privacy_class": 2
        });
        match *state.mode.lock().await {
            "stale" => body["timestamp_ms"] = json!(ts - 60_000),
            "forbidden" => body["pose_frames"] = json!([]),
            _ => {}
        }
        (StatusCode::OK, Json(body))
    }

    async fn mock_semantic(
        Path(node): Path<String>,
        axum::extract::State(state): axum::extract::State<MockState>,
    ) -> (StatusCode, Json<Value>) {
        if *state.mode.lock().await == "offline" {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": "offline"})),
            );
        }
        let ts = now_ms();
        (
            StatusCode::OK,
            Json(json!({
                "node_id": node, "privacy_class": 2,
                "events": {
                    "unknown_presence": {"active": true, "source": "accepted_csi_presence", "ts": ts},
                    "unexpected_occupancy": {"active": true, "source": "schedule_policy", "ts": ts},
                    "unrecognized_activity_pattern": {"active": false, "source": "anomaly_policy_unconfigured", "ts": ts}
                },
                "redacted_fields": ["identity_risk_score", "soul_match_probability", "rf_signature_hash"]
            })),
        )
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    async fn mock_server(mode: &'static str) -> (Url, MockState, tokio::task::JoinHandle<()>) {
        let state = MockState {
            mode: Arc::new(Mutex::new(mode)),
            saw_auth: Arc::new(Mutex::new(false)),
        };
        let app = Router::new()
            .route("/api/v1/vitals/:node/latest", get(mock_vitals))
            .route("/api/v1/semantic-events/:node/latest", get(mock_semantic))
            .with_state(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, state, task)
    }

    fn config(url: &Url) -> RuViewIngestConfig {
        RuViewIngestConfig::validate(
            url.as_str(),
            "7".into(),
            "read-only-secret".into(),
            Duration::from_millis(250),
            Duration::from_secs(1),
            Duration::from_secs(5),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn authoritative_projection_sets_only_allowed_binary_entities() {
        let (url, state, task) = mock_server("ok").await;
        let hc = HomeCore::new();
        let snapshot = poll_once(&Client::new(), &config(&url)).await.unwrap();
        apply_snapshot(&hc, "7", snapshot);
        assert_eq!(hc.states().len(), 4);
        assert_eq!(
            hc.states().get(&entity_id("7", "occupancy")).unwrap().state,
            "on"
        );
        assert_eq!(
            hc.states().get(&entity_id("7", "motion")).unwrap().state,
            "on"
        );
        assert_eq!(
            hc.states()
                .get(&entity_id("7", "unexpected_occupancy"))
                .unwrap()
                .state,
            "on"
        );
        assert!(*state.saw_auth.lock().await);
        let serialized = serde_json::to_string(&hc.states().all()).unwrap();
        for forbidden in FORBIDDEN_FIELDS {
            assert!(!serialized.contains(forbidden));
        }
        task.abort();
    }

    #[tokio::test]
    async fn stale_or_forbidden_evidence_is_rejected() {
        for mode in ["stale", "forbidden"] {
            let (url, _state, task) = mock_server(mode).await;
            assert!(poll_once(&Client::new(), &config(&url)).await.is_err());
            task.abort();
        }
    }

    #[tokio::test]
    async fn runtime_removes_projection_when_upstream_goes_offline() {
        let (url, state, task) = mock_server("ok").await;
        let hc = HomeCore::new();
        let runtime = start(hc.clone(), config(&url));
        tokio::time::timeout(Duration::from_secs(2), async {
            while hc.states().len() != 4 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        *state.mode.lock().await = "offline";
        tokio::time::timeout(Duration::from_secs(2), async {
            while !hc.states().is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        runtime.shutdown().await;
        task.abort();
    }

    #[cfg(feature = "hap-server")]
    #[tokio::test]
    async fn projected_entities_use_the_existing_hap_mapper() {
        use homecore_hap::{EntityToAccessoryMapper, HapAccessoryType};

        let (url, _state, task) = mock_server("ok").await;
        let hc = HomeCore::new();
        let snapshot = poll_once(&Client::new(), &config(&url)).await.unwrap();
        apply_snapshot(&hc, "7", snapshot);
        let occupancy = hc.states().get(&entity_id("7", "occupancy")).unwrap();
        let motion = hc.states().get(&entity_id("7", "motion")).unwrap();
        let threshold = hc
            .states()
            .get(&entity_id("7", "unexpected_occupancy"))
            .unwrap();
        let activity = hc
            .states()
            .get(&entity_id("7", "unrecognized_activity_pattern"))
            .unwrap();
        assert_eq!(
            EntityToAccessoryMapper::map(&occupancy.entity_id, &occupancy)
                .unwrap()
                .accessory_type,
            HapAccessoryType::OccupancySensor
        );
        assert_eq!(
            EntityToAccessoryMapper::map(&motion.entity_id, &motion)
                .unwrap()
                .accessory_type,
            HapAccessoryType::MotionSensor
        );
        assert_eq!(
            EntityToAccessoryMapper::map(&threshold.entity_id, &threshold)
                .unwrap()
                .accessory_type,
            HapAccessoryType::OccupancySensor
        );
        assert_eq!(
            EntityToAccessoryMapper::map(&activity.entity_id, &activity)
                .unwrap()
                .accessory_type,
            HapAccessoryType::MotionSensor
        );
        task.abort();
    }

    #[test]
    fn configuration_is_opt_in_bounded_and_redacts_secret() {
        let cfg = RuViewIngestConfig::validate(
            "https://ruview.example",
            "aa:bb".into(),
            "secret".into(),
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(10),
        )
        .unwrap();
        assert!(!format!("{cfg:?}").contains("secret"));
        assert!(RuViewIngestConfig::validate(
            "file:///tmp/x",
            "7".into(),
            "secret".into(),
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(10),
        )
        .is_err());
        assert!(RuViewIngestConfig::validate(
            "http://ruview.example",
            "7".into(),
            "secret".into(),
            Duration::from_secs(1),
            Duration::from_secs(2),
            Duration::from_secs(10),
        )
        .is_err());
    }
}
