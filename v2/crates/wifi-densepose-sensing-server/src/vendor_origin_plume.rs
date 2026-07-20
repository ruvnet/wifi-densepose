//! Capability-safe ADR-270 adapters for Origin AI and Plume/OpenSync.
//!
//! Origin publishes derived sensing results through contract-gated APIs and
//! webhooks. Public documentation does not define stable paths, so paths are
//! supplied by the contracted deployment configuration. OpenSync exports RF
//! and network telemetry; neither adapter promotes scalar data to complex CSI.

use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use wifi_densepose_hardware::vendor_rf::{
    ProviderAvailability, ProviderDescriptor, RfCapability, VendorEventError, VendorId,
    VendorRfEvent, VendorRfProvider,
};

const MAX_PAYLOAD_BYTES: usize = 256 * 1024;
const MAX_EVENTS_PER_PAYLOAD: usize = 256;
const MAX_ENDPOINT_LEN: usize = 2048;
const MAX_ENV_NAME_LEN: usize = 128;

/// A request plan deliberately containing a credential *reference*, not a secret.
#[derive(Debug, Clone, PartialEq)]
pub struct VendorRequest {
    pub method: &'static str,
    pub endpoint: String,
    pub headers: BTreeMap<String, String>,
    pub credential_env: Option<String>,
    pub body: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OriginAiConfig {
    /// Contract-provided HTTPS sensing-server base URL.
    pub base_url: String,
    /// Contract-provided relative event API path. No public path is assumed.
    pub event_path: String,
    /// Name of the environment variable holding the partner bearer token.
    pub token_env: String,
}

impl OriginAiConfig {
    pub fn validate(&self) -> Result<(), VendorEventError> {
        validate_https_base(&self.base_url)?;
        validate_relative_path(&self.event_path)?;
        validate_env_name(&self.token_env)
    }

    /// Constructs a GET plan. The caller resolves `credential_env` at execution time.
    pub fn events_request(&self) -> Result<VendorRequest, VendorEventError> {
        self.validate()?;
        Ok(VendorRequest {
            method: "GET",
            endpoint: join_endpoint(&self.base_url, &self.event_path),
            headers: BTreeMap::from([("accept".into(), "application/json".into())]),
            credential_env: Some(self.token_env.clone()),
            body: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlumeOpenSyncConfig {
    /// HTTPS northbound/sandbox URL supplied by the operator.
    pub base_url: String,
    /// Relative endpoint accepting OVSDB JSON-RPC for this deployment.
    pub ovsdb_path: String,
    /// Optional environment variable used by a protected northbound endpoint.
    pub token_env: Option<String>,
}

impl PlumeOpenSyncConfig {
    pub fn validate(&self) -> Result<(), VendorEventError> {
        validate_https_base(&self.base_url)?;
        validate_relative_path(&self.ovsdb_path)?;
        if let Some(name) = &self.token_env {
            validate_env_name(name)?;
        }
        Ok(())
    }

    /// Builds a read-only OVSDB `select` transaction for an allow-listed table.
    pub fn select_request(&self, table: &str) -> Result<VendorRequest, VendorEventError> {
        self.validate()?;
        if !matches!(
            table,
            "Wifi_Radio_State" | "Wifi_VIF_State" | "Wifi_Associated_Clients"
        ) {
            return Err(VendorEventError::InvalidPayload);
        }
        Ok(VendorRequest {
            method: "POST",
            endpoint: join_endpoint(&self.base_url, &self.ovsdb_path),
            headers: BTreeMap::from([
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ]),
            credential_env: self.token_env.clone(),
            body: Some(json!({
                "jsonrpc": "2.0",
                "id": "ruview-read-only",
                "method": "transact",
                "params": ["Open_vSwitch", {"op": "select", "table": table, "where": []}]
            })),
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct OriginAiProvider;

impl OriginAiProvider {
    pub fn synthetic_fixture(seed: u64, count: usize) -> Vec<VendorRfEvent> {
        let count = count.min(MAX_EVENTS_PER_PAYLOAD);
        (0..count)
            .map(|index| {
                let state = splitmix64(seed.wrapping_add(index as u64));
                let motion = (state & 1) as f64;
                let confidence = 0.70 + ((state >> 8) % 300) as f64 / 1000.0;
                VendorRfEvent {
                    vendor: VendorId::OriginAi,
                    capability: RfCapability::DerivedSensing,
                    sequence: index as u64,
                    timestamp_us: 1_700_000_000_000_000 + index as u64 * 100_000,
                    source_id: format!("origin-sim-{:08x}", seed as u32),
                    synthetic: true,
                    metrics: BTreeMap::from([
                        ("motion".into(), motion),
                        ("confidence".into(), confidence),
                    ]),
                    label: Some(if motion == 1.0 { "motion" } else { "clear" }.into()),
                }
            })
            .collect()
    }
}

impl VendorRfProvider for OriginAiProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            vendor: VendorId::OriginAi,
            capabilities: vec![RfCapability::DerivedSensing],
            availability: ProviderAvailability::ContractRequired,
            hardware_validated: false,
            reason: "Origin partner API/SDK access is contract-gated; adapter accepts derived sensing only"
                .into(),
        }
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        check_payload(payload)?;
        let envelope: OriginEnvelope = decode_json(payload)?;
        if envelope.events.is_empty() || envelope.events.len() > MAX_EVENTS_PER_PAYLOAD {
            return Err(VendorEventError::InvalidPayload);
        }
        envelope
            .events
            .into_iter()
            .map(|event| event.into_vendor_event(&self.descriptor()))
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginEnvelope {
    events: Vec<OriginEvent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginEvent {
    sequence: u64,
    timestamp_us: u64,
    source_id: String,
    kind: OriginKind,
    confidence: f64,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    label: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OriginKind {
    Motion,
    Occupancy,
    Presence,
    Fall,
    BreathingRate,
}

impl OriginEvent {
    fn into_vendor_event(
        self,
        descriptor: &ProviderDescriptor,
    ) -> Result<VendorRfEvent, VendorEventError> {
        if self.timestamp_us == 0 || !(0.0..=1.0).contains(&self.confidence) {
            return Err(VendorEventError::InvalidPayload);
        }
        let (metric, requires_value) = match self.kind {
            OriginKind::Motion => ("motion", false),
            OriginKind::Occupancy => ("occupancy", false),
            OriginKind::Presence => ("presence", false),
            OriginKind::Fall => ("fall", false),
            OriginKind::BreathingRate => ("breathing_rate_bpm", true),
        };
        let value = self.value.ok_or(VendorEventError::InvalidPayload)?;
        if !value.is_finite()
            || (requires_value && !(1.0..=120.0).contains(&value))
            || (!requires_value && value != 0.0 && value != 1.0)
        {
            return Err(VendorEventError::InvalidPayload);
        }
        let event = VendorRfEvent {
            vendor: VendorId::OriginAi,
            capability: RfCapability::DerivedSensing,
            sequence: self.sequence,
            timestamp_us: self.timestamp_us,
            source_id: self.source_id,
            synthetic: false,
            metrics: BTreeMap::from([
                (metric.into(), value),
                ("confidence".into(), self.confidence),
            ]),
            label: self.label,
        };
        event.validate(descriptor)?;
        Ok(event)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PlumeOpenSyncProvider;

impl PlumeOpenSyncProvider {
    pub fn synthetic_fixture(seed: u64, count: usize) -> Vec<VendorRfEvent> {
        let count = count.min(MAX_EVENTS_PER_PAYLOAD);
        (0..count)
            .map(|index| {
                let state = splitmix64(seed.wrapping_add(index as u64));
                VendorRfEvent {
                    vendor: VendorId::Plume,
                    capability: RfCapability::RfTelemetry,
                    sequence: index as u64,
                    timestamp_us: 1_700_000_000_000_000 + index as u64 * 250_000,
                    source_id: format!("opensync-sim-{:08x}", seed as u32),
                    synthetic: true,
                    metrics: BTreeMap::from([
                        ("rssi_dbm".into(), -30.0 - (state % 55) as f64),
                        ("channel".into(), 1.0 + ((state >> 8) % 165) as f64),
                        (
                            "noise_floor_dbm".into(),
                            -100.0 + ((state >> 16) % 12) as f64,
                        ),
                    ]),
                    label: None,
                }
            })
            .collect()
    }
}

impl VendorRfProvider for PlumeOpenSyncProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        ProviderDescriptor {
            vendor: VendorId::Plume,
            capabilities: vec![RfCapability::RfTelemetry],
            availability: ProviderAvailability::CredentialsRequired,
            hardware_validated: false,
            reason: "OpenSync radio/client telemetry only; Plume Sense is a separate gated service"
                .into(),
        }
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        check_payload(payload)?;
        let envelope: OpenSyncEnvelope = decode_json(payload)?;
        if envelope.observations.is_empty() || envelope.observations.len() > MAX_EVENTS_PER_PAYLOAD
        {
            return Err(VendorEventError::InvalidPayload);
        }
        envelope
            .observations
            .into_iter()
            .map(|event| event.into_vendor_event(&self.descriptor()))
            .collect()
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenSyncEnvelope {
    observations: Vec<OpenSyncObservation>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenSyncObservation {
    sequence: u64,
    timestamp_us: u64,
    source_id: String,
    rssi_dbm: f64,
    channel: u16,
    #[serde(default)]
    noise_floor_dbm: Option<f64>,
    #[serde(default)]
    tx_rate_mbps: Option<f64>,
    #[serde(default)]
    rx_rate_mbps: Option<f64>,
    #[serde(default)]
    clients: Option<u16>,
}

impl OpenSyncObservation {
    fn into_vendor_event(
        self,
        descriptor: &ProviderDescriptor,
    ) -> Result<VendorRfEvent, VendorEventError> {
        if self.timestamp_us == 0
            || !(-127.0..=0.0).contains(&self.rssi_dbm)
            || self.channel == 0
            || self.channel > 233
        {
            return Err(VendorEventError::InvalidPayload);
        }
        let mut metrics = BTreeMap::from([
            ("rssi_dbm".into(), self.rssi_dbm),
            ("channel".into(), self.channel as f64),
        ]);
        insert_optional(
            &mut metrics,
            "noise_floor_dbm",
            self.noise_floor_dbm,
            -127.0,
            0.0,
        )?;
        insert_optional(
            &mut metrics,
            "tx_rate_mbps",
            self.tx_rate_mbps,
            0.0,
            100_000.0,
        )?;
        insert_optional(
            &mut metrics,
            "rx_rate_mbps",
            self.rx_rate_mbps,
            0.0,
            100_000.0,
        )?;
        if let Some(clients) = self.clients {
            metrics.insert("clients".into(), clients as f64);
        }
        let event = VendorRfEvent {
            vendor: VendorId::Plume,
            capability: RfCapability::RfTelemetry,
            sequence: self.sequence,
            timestamp_us: self.timestamp_us,
            source_id: self.source_id,
            synthetic: false,
            metrics,
            label: None,
        };
        event.validate(descriptor)?;
        Ok(event)
    }
}

fn insert_optional(
    metrics: &mut BTreeMap<String, f64>,
    key: &str,
    value: Option<f64>,
    minimum: f64,
    maximum: f64,
) -> Result<(), VendorEventError> {
    if let Some(value) = value {
        if !value.is_finite() || !(minimum..=maximum).contains(&value) {
            return Err(VendorEventError::InvalidPayload);
        }
        metrics.insert(key.into(), value);
    }
    Ok(())
}

fn check_payload(payload: &[u8]) -> Result<(), VendorEventError> {
    if payload.is_empty() || payload.len() > MAX_PAYLOAD_BYTES {
        Err(VendorEventError::InvalidPayload)
    } else {
        Ok(())
    }
}

fn decode_json<T: for<'de> Deserialize<'de>>(payload: &[u8]) -> Result<T, VendorEventError> {
    serde_json::from_slice(payload).map_err(|error| {
        let message = error.to_string();
        VendorEventError::MalformedPayload(message.chars().take(160).collect())
    })
}

fn validate_https_base(value: &str) -> Result<(), VendorEventError> {
    let authority = value
        .strip_prefix("https://")
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_default();
    if value.len() > MAX_ENDPOINT_LEN
        || authority.is_empty()
        || authority.contains('@')
        || authority.chars().any(char::is_whitespace)
        || value.contains(['\r', '\n', '#', '?'])
    {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), VendorEventError> {
    if value.len() > MAX_ENDPOINT_LEN
        || !value.starts_with('/')
        || value.starts_with("//")
        || value.contains(['\r', '\n', '#', '?'])
        || value.split('/').any(|segment| segment == "..")
    {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(())
}

fn validate_env_name(value: &str) -> Result<(), VendorEventError> {
    if value.is_empty()
        || value.len() > MAX_ENV_NAME_LEN
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
        || value.as_bytes()[0].is_ascii_digit()
    {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(())
}

fn join_endpoint(base: &str, path: &str) -> String {
    format!("{}{}", base.trim_end_matches('/'), path)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wifi_densepose_hardware::vendor_rf::MAX_VENDOR_TEXT_LEN;

    #[test]
    fn descriptors_are_honest_about_capabilities_and_access() {
        let origin = OriginAiProvider.descriptor();
        assert_eq!(origin.capabilities, vec![RfCapability::DerivedSensing]);
        assert_eq!(origin.availability, ProviderAvailability::ContractRequired);
        let plume = PlumeOpenSyncProvider.descriptor();
        assert_eq!(plume.capabilities, vec![RfCapability::RfTelemetry]);
        assert_eq!(
            plume.availability,
            ProviderAvailability::CredentialsRequired
        );
        assert!(!origin.capabilities.contains(&RfCapability::ComplexCsi));
        assert!(!plume.capabilities.contains(&RfCapability::ComplexCsi));
    }

    #[test]
    fn origin_decodes_only_derived_sensing() {
        let payload = br#"{"events":[{"sequence":7,"timestamp_us":42,"source_id":"zone-a","kind":"occupancy","confidence":0.91,"value":1,"label":"occupied"}]}"#;
        let events = OriginAiProvider.decode(payload).unwrap();
        assert_eq!(events[0].capability, RfCapability::DerivedSensing);
        assert_eq!(events[0].metrics["occupancy"], 1.0);
        assert!(!events[0].synthetic);
    }

    #[test]
    fn origin_rejects_unknown_raw_csi_and_bad_values() {
        let raw = br#"{"events":[{"sequence":1,"timestamp_us":2,"source_id":"z","kind":"motion","confidence":1,"value":1,"raw_csi":[[1,2]]}]}"#;
        assert!(OriginAiProvider.decode(raw).is_err());
        let bad_confidence = br#"{"events":[{"sequence":1,"timestamp_us":2,"source_id":"z","kind":"motion","confidence":1.1,"value":1}]}"#;
        assert_eq!(
            OriginAiProvider.decode(bad_confidence),
            Err(VendorEventError::InvalidPayload)
        );
    }

    #[test]
    fn plume_decodes_telemetry_without_inventing_csi() {
        let payload = br#"{"observations":[{"sequence":9,"timestamp_us":10,"source_id":"pod-a","rssi_dbm":-54,"channel":36,"noise_floor_dbm":-94,"tx_rate_mbps":1200,"clients":4}]}"#;
        let events = PlumeOpenSyncProvider.decode(payload).unwrap();
        assert_eq!(events[0].capability, RfCapability::RfTelemetry);
        assert_eq!(events[0].metrics["rssi_dbm"], -54.0);
        assert!(!events[0].metrics.contains_key("csi"));
    }

    #[test]
    fn malformed_empty_oversized_and_unbounded_arrays_fail_closed() {
        assert!(OriginAiProvider.decode(b"{").is_err());
        assert_eq!(
            OriginAiProvider.decode(b""),
            Err(VendorEventError::InvalidPayload)
        );
        assert_eq!(
            PlumeOpenSyncProvider.decode(&vec![b' '; MAX_PAYLOAD_BYTES + 1]),
            Err(VendorEventError::InvalidPayload)
        );
        let event = r#"{"sequence":1,"timestamp_us":2,"source_id":"p","rssi_dbm":-50,"channel":1}"#;
        let payload = format!("{{\"observations\":[{}]}}", vec![event; 257].join(","));
        assert_eq!(
            PlumeOpenSyncProvider.decode(payload.as_bytes()),
            Err(VendorEventError::InvalidPayload)
        );
    }

    #[test]
    fn telemetry_ranges_are_enforced() {
        let payload = br#"{"observations":[{"sequence":1,"timestamp_us":2,"source_id":"p","rssi_dbm":4,"channel":36}]}"#;
        assert_eq!(
            PlumeOpenSyncProvider.decode(payload),
            Err(VendorEventError::InvalidPayload)
        );
    }

    #[test]
    fn fixtures_are_deterministic_bounded_and_marked_synthetic() {
        let a = OriginAiProvider::synthetic_fixture(19, 4);
        assert_eq!(a, OriginAiProvider::synthetic_fixture(19, 4));
        assert!(a.iter().all(|event| event.synthetic));
        let p = PlumeOpenSyncProvider::synthetic_fixture(23, usize::MAX);
        assert_eq!(p.len(), MAX_EVENTS_PER_PAYLOAD);
        assert!(p.iter().all(|event| event.synthetic));
        assert_eq!(p, PlumeOpenSyncProvider::synthetic_fixture(23, usize::MAX));
    }

    #[test]
    fn request_plans_reference_secrets_without_embedding_them() {
        let origin = OriginAiConfig {
            base_url: "https://partner.example".into(),
            event_path: "/contract/v1/events".into(),
            token_env: "ORIGIN_AI_TOKEN".into(),
        }
        .events_request()
        .unwrap();
        assert_eq!(
            origin.endpoint,
            "https://partner.example/contract/v1/events"
        );
        assert_eq!(origin.credential_env.as_deref(), Some("ORIGIN_AI_TOKEN"));
        assert!(format!("{origin:?}").find("Bearer ").is_none());

        let plume = PlumeOpenSyncConfig {
            base_url: "https://sandbox.example".into(),
            ovsdb_path: "/ovsdb".into(),
            token_env: Some("OPENSYNC_TOKEN".into()),
        }
        .select_request("Wifi_Radio_State")
        .unwrap();
        assert_eq!(plume.body.as_ref().unwrap()["method"], "transact");
        assert_eq!(plume.credential_env.as_deref(), Some("OPENSYNC_TOKEN"));
    }

    #[test]
    fn request_validation_rejects_injection_and_write_tables() {
        let config = PlumeOpenSyncConfig {
            base_url: "https://sandbox.example".into(),
            ovsdb_path: "/ovsdb".into(),
            token_env: None,
        };
        assert_eq!(
            config.select_request("AWLAN_Node"),
            Err(VendorEventError::InvalidPayload)
        );
        let bad = OriginAiConfig {
            base_url: "http://insecure.example".into(),
            event_path: "/events\r\nx: y".into(),
            token_env: "token".into(),
        };
        assert_eq!(bad.events_request(), Err(VendorEventError::InvalidPayload));
    }

    #[test]
    fn labels_and_source_ids_obey_shared_contract_bounds() {
        let label = "x".repeat(MAX_VENDOR_TEXT_LEN + 1);
        let payload = format!(
            "{{\"events\":[{{\"sequence\":1,\"timestamp_us\":2,\"source_id\":\"z\",\"kind\":\"motion\",\"confidence\":1,\"value\":1,\"label\":\"{label}\"}}]}}"
        );
        assert_eq!(
            OriginAiProvider.decode(payload.as_bytes()),
            Err(VendorEventError::InvalidPayload)
        );
    }
}
