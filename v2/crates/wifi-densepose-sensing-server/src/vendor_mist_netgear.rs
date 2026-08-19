//! ADR-270 adapters for Mist/Juniper and NETGEAR Insight.
//!
//! These cloud APIs expose client RF telemetry and location/network context.
//! They do not expose complex channel state information (CSI), and this module
//! deliberately cannot construct a `ComplexCsi` event.

use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt;
use wifi_densepose_hardware::vendor_rf::{
    ProviderAvailability, ProviderDescriptor, RfCapability, VendorEventError, VendorId,
    VendorRfEvent, VendorRfProvider,
};

pub const MAX_VENDOR_PAYLOAD_BYTES: usize = 1024 * 1024;
pub const MAX_EVENTS_PER_PAGE: usize = 1_000;
pub const MAX_CURSOR_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistRegion {
    Global,
    Europe,
    AsiaPacific,
    Australia,
}

impl MistRegion {
    pub const fn base_url(self) -> &'static str {
        match self {
            Self::Global => "https://api.mist.com",
            Self::Europe => "https://api.eu.mist.com",
            Self::AsiaPacific => "https://api.ac2.mist.com",
            Self::Australia => "https://api.gc1.mist.com",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetgearRegion {
    NorthAmerica,
    Europe,
    Australia,
}

impl NetgearRegion {
    pub const fn base_url(self) -> &'static str {
        match self {
            Self::NorthAmerica => "https://insight.netgear.com",
            Self::Europe => "https://eu.insight.netgear.com",
            Self::Australia => "https://au.insight.netgear.com",
        }
    }
}

/// An authentication value whose `Debug` output is always redacted.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretToken(String);

impl SecretToken {
    pub fn new(value: impl Into<String>) -> Result<Self, VendorEventError> {
        let value = value.into();
        if value.is_empty() || value.len() > 4_096 || value.chars().any(char::is_control) {
            return Err(VendorEventError::InvalidPayload);
        }
        Ok(Self(value))
    }

    /// Intended only for constructing an HTTP authorization header.
    pub fn expose_for_header(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretToken([REDACTED])")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct VendorRequestConfig {
    pub base_url: &'static str,
    pub path: String,
    pub cursor: Option<String>,
    pub token: SecretToken,
}

impl VendorRequestConfig {
    pub fn validate(&self) -> Result<(), VendorEventError> {
        if !self.base_url.starts_with("https://")
            || !self.path.starts_with('/')
            || self.path.contains("..")
            || self.path.chars().any(char::is_control)
            || self.cursor.as_deref().is_some_and(|cursor| {
                cursor.is_empty()
                    || cursor.len() > MAX_CURSOR_BYTES
                    || cursor
                        .chars()
                        .any(|c| c.is_control() || c == '&' || c == '?' || c == '#')
            })
        {
            return Err(VendorEventError::InvalidPayload);
        }
        Ok(())
    }
}

impl fmt::Debug for VendorRequestConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VendorRequestConfig")
            .field("base_url", &self.base_url)
            .field("path", &self.path)
            .field("cursor", &self.cursor.as_ref().map(|_| "[PRESENT]"))
            .field("token", &"[REDACTED]")
            .finish()
    }
}

pub fn mist_request(
    region: MistRegion,
    site_id: &str,
    cursor: Option<String>,
    token: SecretToken,
) -> Result<VendorRequestConfig, VendorEventError> {
    validate_identifier(site_id)?;
    let request = VendorRequestConfig {
        base_url: region.base_url(),
        path: format!("/api/v1/sites/{site_id}/stats/clients"),
        cursor,
        token,
    };
    request.validate()?;
    Ok(request)
}

pub fn netgear_request(
    region: NetgearRegion,
    location_id: &str,
    cursor: Option<String>,
    token: SecretToken,
) -> Result<VendorRequestConfig, VendorEventError> {
    validate_identifier(location_id)?;
    let request = VendorRequestConfig {
        base_url: region.base_url(),
        path: format!("/api/v1/locations/{location_id}/clients"),
        cursor,
        token,
    };
    request.validate()?;
    Ok(request)
}

fn validate_identifier(value: &str) -> Result<(), VendorEventError> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_'))
    {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedVendorPage {
    pub events: Vec<VendorRfEvent>,
    pub next_cursor: Option<String>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MistProvider;

impl MistProvider {
    pub fn decode_page(&self, payload: &[u8]) -> Result<DecodedVendorPage, VendorEventError> {
        decode_page(
            payload,
            VendorId::Mist,
            mist_descriptor(),
            parse_mist_record,
        )
    }
}

impl VendorRfProvider for MistProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        mist_descriptor()
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        Ok(self.decode_page(payload)?.events)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NetgearInsightProvider;

impl NetgearInsightProvider {
    pub fn decode_page(&self, payload: &[u8]) -> Result<DecodedVendorPage, VendorEventError> {
        decode_page(
            payload,
            VendorId::Netgear,
            netgear_descriptor(),
            parse_netgear_record,
        )
    }
}

impl VendorRfProvider for NetgearInsightProvider {
    fn descriptor(&self) -> ProviderDescriptor {
        netgear_descriptor()
    }

    fn decode(&self, payload: &[u8]) -> Result<Vec<VendorRfEvent>, VendorEventError> {
        Ok(self.decode_page(payload)?.events)
    }
}

pub fn mist_descriptor() -> ProviderDescriptor {
    ProviderDescriptor {
        vendor: VendorId::Mist,
        capabilities: vec![RfCapability::RfTelemetry, RfCapability::NetworkOnly],
        availability: ProviderAvailability::CredentialsRequired,
        hardware_validated: false,
        reason: "Mist cloud client RF telemetry and location context; never CSI".into(),
    }
}

pub fn netgear_descriptor() -> ProviderDescriptor {
    ProviderDescriptor {
        vendor: VendorId::Netgear,
        capabilities: vec![RfCapability::RfTelemetry, RfCapability::NetworkOnly],
        availability: ProviderAvailability::CredentialsRequired,
        hardware_validated: false,
        reason: "NETGEAR Insight client RF and network telemetry; never CSI".into(),
    }
}

type RecordParser = fn(&Map<String, Value>, usize) -> Result<VendorRfEvent, VendorEventError>;

fn decode_page(
    payload: &[u8],
    vendor: VendorId,
    descriptor: ProviderDescriptor,
    parser: RecordParser,
) -> Result<DecodedVendorPage, VendorEventError> {
    if payload.is_empty() || payload.len() > MAX_VENDOR_PAYLOAD_BYTES {
        return Err(VendorEventError::InvalidPayload);
    }
    let root: Value = serde_json::from_slice(payload)
        .map_err(|e| VendorEventError::MalformedPayload(e.to_string()))?;
    let (records, next_cursor) = extract_records_and_cursor(&root)?;
    if records.is_empty() || records.len() > MAX_EVENTS_PER_PAGE {
        return Err(VendorEventError::InvalidPayload);
    }

    let mut events = Vec::with_capacity(records.len());
    for (index, value) in records.iter().enumerate() {
        let object = value.as_object().ok_or(VendorEventError::InvalidPayload)?;
        let event = parser(object, index)?;
        if event.vendor != vendor || event.capability != RfCapability::RfTelemetry {
            return Err(VendorEventError::CapabilityMismatch);
        }
        event.validate(&descriptor)?;
        events.push(event);
    }
    Ok(DecodedVendorPage {
        events,
        next_cursor,
    })
}

fn extract_records_and_cursor(
    root: &Value,
) -> Result<(&[Value], Option<String>), VendorEventError> {
    if let Some(records) = root.as_array() {
        return Ok((records, None));
    }
    let object = root.as_object().ok_or(VendorEventError::InvalidPayload)?;
    let records = ["results", "data", "clients", "events", "items"]
        .iter()
        .find_map(|key| object.get(*key).and_then(Value::as_array))
        .ok_or(VendorEventError::InvalidPayload)?;
    let cursor_value = object
        .get("next_cursor")
        .or_else(|| object.get("nextPageToken"))
        .or_else(|| object.get("next_page_token"))
        .or_else(|| {
            object
                .get("pagination")
                .and_then(Value::as_object)
                .and_then(|p| p.get("next").or_else(|| p.get("cursor")))
        });
    let next_cursor = match cursor_value {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if valid_cursor(value) => Some(value.clone()),
        Some(_) => return Err(VendorEventError::InvalidPayload),
    };
    Ok((records, next_cursor))
}

fn valid_cursor(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_CURSOR_BYTES
        && !value
            .chars()
            .any(|c| c.is_control() || c == '&' || c == '?' || c == '#')
}

#[derive(Debug, Deserialize)]
struct MistRecord {
    #[serde(alias = "client_id", alias = "mac")]
    id: String,
    #[serde(default, alias = "last_seen", alias = "lastSeen")]
    timestamp: Option<Value>,
    #[serde(default, alias = "rssi_dbm")]
    rssi: Option<f64>,
    #[serde(default)]
    snr: Option<f64>,
    #[serde(default)]
    channel: Option<f64>,
    #[serde(default)]
    x: Option<f64>,
    #[serde(default)]
    y: Option<f64>,
    #[serde(default)]
    site_id: Option<String>,
    #[serde(default)]
    ap_id: Option<String>,
    #[serde(default, alias = "event_type", alias = "type")]
    event: Option<String>,
}

fn parse_mist_record(
    object: &Map<String, Value>,
    index: usize,
) -> Result<VendorRfEvent, VendorEventError> {
    let record: MistRecord = serde_json::from_value(Value::Object(object.clone()))
        .map_err(|e| VendorEventError::MalformedPayload(e.to_string()))?;
    validate_source(&record.id)?;
    validate_optional_text(record.site_id.as_deref())?;
    validate_optional_text(record.ap_id.as_deref())?;
    validate_optional_text(record.event.as_deref())?;
    let timestamp_us = parse_timestamp(record.timestamp.as_ref())?;
    let mut metrics = BTreeMap::new();
    push_metric(&mut metrics, "rssi_dbm", record.rssi, -150.0, 20.0)?;
    push_metric(&mut metrics, "snr_db", record.snr, -50.0, 100.0)?;
    push_metric(&mut metrics, "channel", record.channel, 1.0, 7_000.0)?;
    push_metric(&mut metrics, "x_m", record.x, -1_000_000.0, 1_000_000.0)?;
    push_metric(&mut metrics, "y_m", record.y, -1_000_000.0, 1_000_000.0)?;
    if metrics.is_empty() {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(VendorRfEvent {
        vendor: VendorId::Mist,
        capability: RfCapability::RfTelemetry,
        sequence: stable_sequence(&record.id, timestamp_us, index),
        timestamp_us,
        source_id: record.id,
        synthetic: false,
        metrics,
        label: context_label(&[record.site_id, record.ap_id, record.event])?,
    })
}

#[derive(Debug, Deserialize)]
struct NetgearRecord {
    #[serde(
        alias = "clientId",
        alias = "mac",
        alias = "macAddress",
        alias = "deviceId"
    )]
    id: String,
    #[serde(default, alias = "lastSeen", alias = "observedAt")]
    timestamp: Option<Value>,
    #[serde(default, alias = "signalStrength", alias = "rssi_dbm")]
    rssi: Option<f64>,
    #[serde(default, alias = "signalToNoiseRatio")]
    snr: Option<f64>,
    #[serde(default)]
    channel: Option<f64>,
    #[serde(default, alias = "txRateMbps", alias = "tx_rate")]
    tx_rate_mbps: Option<f64>,
    #[serde(default, alias = "rxRateMbps", alias = "rx_rate")]
    rx_rate_mbps: Option<f64>,
    #[serde(default, alias = "locationId")]
    location_id: Option<String>,
    #[serde(default, alias = "accessPointId", alias = "apId")]
    ap_id: Option<String>,
    #[serde(default, alias = "ssidName")]
    ssid: Option<String>,
}

fn parse_netgear_record(
    object: &Map<String, Value>,
    index: usize,
) -> Result<VendorRfEvent, VendorEventError> {
    let record: NetgearRecord = serde_json::from_value(Value::Object(object.clone()))
        .map_err(|e| VendorEventError::MalformedPayload(e.to_string()))?;
    validate_source(&record.id)?;
    validate_optional_text(record.location_id.as_deref())?;
    validate_optional_text(record.ap_id.as_deref())?;
    validate_optional_text(record.ssid.as_deref())?;
    let timestamp_us = parse_timestamp(record.timestamp.as_ref())?;
    let mut metrics = BTreeMap::new();
    push_metric(&mut metrics, "rssi_dbm", record.rssi, -150.0, 20.0)?;
    push_metric(&mut metrics, "snr_db", record.snr, -50.0, 100.0)?;
    push_metric(&mut metrics, "channel", record.channel, 1.0, 7_000.0)?;
    push_metric(
        &mut metrics,
        "tx_rate_mbps",
        record.tx_rate_mbps,
        0.0,
        100_000.0,
    )?;
    push_metric(
        &mut metrics,
        "rx_rate_mbps",
        record.rx_rate_mbps,
        0.0,
        100_000.0,
    )?;
    if metrics.is_empty() {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(VendorRfEvent {
        vendor: VendorId::Netgear,
        capability: RfCapability::RfTelemetry,
        sequence: stable_sequence(&record.id, timestamp_us, index),
        timestamp_us,
        source_id: record.id,
        synthetic: false,
        metrics,
        label: context_label(&[record.location_id, record.ap_id, record.ssid])?,
    })
}

fn parse_timestamp(value: Option<&Value>) -> Result<u64, VendorEventError> {
    let value = value.ok_or(VendorEventError::InvalidPayload)?;
    if let Some(text) = value.as_str() {
        if let Ok(raw) = text.parse::<u64>() {
            return normalize_integer_timestamp(raw);
        }
        if let Ok(raw) = text.parse::<f64>() {
            return normalize_fractional_timestamp(raw);
        }
        let parsed = chrono::DateTime::parse_from_rfc3339(text)
            .map_err(|_| VendorEventError::InvalidPayload)?;
        return u64::try_from(parsed.timestamp_micros())
            .map_err(|_| VendorEventError::InvalidPayload);
    }
    if let Some(raw) = value.as_u64() {
        return normalize_integer_timestamp(raw);
    }
    normalize_fractional_timestamp(value.as_f64().ok_or(VendorEventError::InvalidPayload)?)
}

fn normalize_integer_timestamp(raw: u64) -> Result<u64, VendorEventError> {
    if raw == 0 {
        return Err(VendorEventError::InvalidPayload);
    }
    // Normalize seconds, milliseconds, or microseconds to microseconds.
    if raw < 10_000_000_000 {
        raw.checked_mul(1_000_000)
            .ok_or(VendorEventError::InvalidPayload)
    } else if raw < 10_000_000_000_000 {
        raw.checked_mul(1_000)
            .ok_or(VendorEventError::InvalidPayload)
    } else if raw < 10_000_000_000_000_000 {
        Ok(raw)
    } else {
        Err(VendorEventError::InvalidPayload)
    }
}

fn normalize_fractional_timestamp(raw: f64) -> Result<u64, VendorEventError> {
    if !raw.is_finite() || raw <= 0.0 {
        return Err(VendorEventError::InvalidPayload);
    }
    let micros = if raw < 10_000_000_000.0 {
        raw * 1_000_000.0
    } else if raw < 10_000_000_000_000.0 {
        raw * 1_000.0
    } else if raw < 10_000_000_000_000_000.0 {
        raw
    } else {
        return Err(VendorEventError::InvalidPayload);
    };
    if !micros.is_finite() || micros > u64::MAX as f64 {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(micros.round() as u64)
}

fn push_metric(
    metrics: &mut BTreeMap<String, f64>,
    name: &str,
    value: Option<f64>,
    minimum: f64,
    maximum: f64,
) -> Result<(), VendorEventError> {
    if let Some(value) = value {
        if !value.is_finite() || !(minimum..=maximum).contains(&value) {
            return Err(VendorEventError::InvalidPayload);
        }
        metrics.insert(name.into(), value);
    }
    Ok(())
}

fn validate_source(value: &str) -> Result<(), VendorEventError> {
    if value.is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(())
}

fn validate_optional_text(value: Option<&str>) -> Result<(), VendorEventError> {
    if value.is_some_and(|v| v.is_empty() || v.len() > 256 || v.chars().any(char::is_control)) {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok(())
}

fn context_label(parts: &[Option<String>]) -> Result<Option<String>, VendorEventError> {
    let label = parts
        .iter()
        .filter_map(Option::as_deref)
        .collect::<Vec<_>>()
        .join("/");
    if label.len() > 256 {
        return Err(VendorEventError::InvalidPayload);
    }
    Ok((!label.is_empty()).then_some(label))
}

fn stable_sequence(source: &str, timestamp_us: u64, index: usize) -> u64 {
    // FNV-1a gives a deterministic correlation key without process-random state.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in source
        .bytes()
        .chain(timestamp_us.to_le_bytes())
        .chain((index as u64).to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIST_FIXTURE: &[u8] = br#"{
      "results": [
        {"id":"aa:bb:cc:dd:ee:ff","timestamp":1710000000,"rssi":-47,"snr":31,"channel":44,"x":12.5,"y":8.25,"site_id":"site-1","ap_id":"ap-7"},
        {"client_id":"station-2","last_seen":1710000000123,"rssi_dbm":-62,"event_type":"client-info"}
      ],
      "next_cursor":"page-2"
    }"#;

    const NETGEAR_FIXTURE: &[u8] = br#"{
      "data": [
        {"clientId":"client-1","observedAt":"1710000000000000","signalStrength":-53,"signalToNoiseRatio":24,"channel":149,"txRateMbps":866.7,"rxRateMbps":721.2,"locationId":"office","accessPointId":"ap-2","ssidName":"lab"}
      ],
      "pagination":{"next":"cursor-2"}
    }"#;

    #[test]
    fn descriptors_are_honest_and_valid() {
        for descriptor in [mist_descriptor(), netgear_descriptor()] {
            descriptor.validate().unwrap();
            assert!(!descriptor.hardware_validated);
            assert_eq!(
                descriptor.availability,
                ProviderAvailability::CredentialsRequired
            );
            assert!(!descriptor.capabilities.contains(&RfCapability::ComplexCsi));
        }
    }

    #[test]
    fn mist_rest_and_webhook_fixture_is_deterministic() {
        let provider = MistProvider;
        let first = provider.decode_page(MIST_FIXTURE).unwrap();
        let second = provider.decode_page(MIST_FIXTURE).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.next_cursor.as_deref(), Some("page-2"));
        assert_eq!(first.events.len(), 2);
        assert_eq!(first.events[0].timestamp_us, 1_710_000_000_000_000);
        assert_eq!(first.events[1].timestamp_us, 1_710_000_000_123_000);
        assert_eq!(first.events[0].metrics["x_m"], 12.5);
        assert!(!first.events[0].synthetic);
    }

    #[test]
    fn netgear_page_fixture_normalizes_aliases() {
        let page = NetgearInsightProvider.decode_page(NETGEAR_FIXTURE).unwrap();
        assert_eq!(page.next_cursor.as_deref(), Some("cursor-2"));
        assert_eq!(page.events.len(), 1);
        let event = &page.events[0];
        assert_eq!(event.vendor, VendorId::Netgear);
        assert_eq!(event.capability, RfCapability::RfTelemetry);
        assert_eq!(event.metrics["tx_rate_mbps"], 866.7);
        assert_eq!(event.label.as_deref(), Some("office/ap-2/lab"));
    }

    #[test]
    fn top_level_array_is_supported_without_pagination() {
        let payload = br#"[{"mac":"a","timestamp":1710000000,"rssi":-40}]"#;
        let page = MistProvider.decode_page(payload).unwrap();
        assert_eq!(page.events.len(), 1);
        assert_eq!(page.next_cursor, None);
    }

    #[test]
    fn payload_and_page_bounds_are_enforced() {
        assert_eq!(
            MistProvider.decode(&vec![b' '; MAX_VENDOR_PAYLOAD_BYTES + 1]),
            Err(VendorEventError::InvalidPayload)
        );
        let values = (0..=MAX_EVENTS_PER_PAGE)
            .map(|_| serde_json::json!({"id":"a","timestamp":1710000000,"rssi":-40}))
            .collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&values).unwrap();
        assert_eq!(
            MistProvider.decode(&bytes),
            Err(VendorEventError::InvalidPayload)
        );
    }

    #[test]
    fn missing_identity_timestamp_or_metrics_fails_closed() {
        for payload in [
            br#"[{"timestamp":1710000000,"rssi":-40}]"#.as_slice(),
            br#"[{"id":"a","rssi":-40}]"#.as_slice(),
            br#"[{"id":"a","timestamp":1710000000}]"#.as_slice(),
        ] {
            assert!(MistProvider.decode(payload).is_err());
        }
    }

    #[test]
    fn invalid_metric_cursor_and_record_schema_fail_closed() {
        assert!(MistProvider
            .decode(br#"[{"id":"a","timestamp":1710000000,"rssi":999}]"#)
            .is_err());
        assert!(NetgearInsightProvider.decode(br#"{"data":[42]}"#).is_err());
        assert!(MistProvider
            .decode_page(
                br#"{"results":[{"id":"a","timestamp":1710000000,"rssi":-40}],"next_cursor":"x&admin=true"}"#
            )
            .is_err());
    }

    #[test]
    fn request_configuration_is_regional_validated_and_redacted() {
        let token = SecretToken::new("super-secret").unwrap();
        let mist = mist_request(
            MistRegion::Europe,
            "site_1",
            Some("page-2".into()),
            token.clone(),
        )
        .unwrap();
        assert_eq!(mist.base_url, "https://api.eu.mist.com");
        assert_eq!(mist.path, "/api/v1/sites/site_1/stats/clients");
        let netgear = netgear_request(NetgearRegion::Australia, "location-7", None, token).unwrap();
        assert_eq!(netgear.base_url, "https://au.insight.netgear.com");
        assert!(!format!("{netgear:?}").contains("super-secret"));
        assert!(!format!("{:?}", netgear.token).contains("super-secret"));
        assert!(mist_request(
            MistRegion::Global,
            "../other-site",
            None,
            SecretToken::new("token").unwrap()
        )
        .is_err());
    }

    #[test]
    fn token_rejects_header_injection() {
        assert!(SecretToken::new("token\r\nX-Injected: yes").is_err());
        assert!(SecretToken::new("").is_err());
    }

    #[test]
    fn timestamp_units_are_normalized_and_extremes_rejected() {
        assert_eq!(
            parse_timestamp(Some(&serde_json::json!(1_710_000_000))).unwrap(),
            1_710_000_000_000_000
        );
        assert_eq!(
            parse_timestamp(Some(&serde_json::json!(1_710_000_000_123_u64))).unwrap(),
            1_710_000_000_123_000
        );
        assert!(parse_timestamp(Some(&serde_json::json!(0))).is_err());
        assert!(parse_timestamp(Some(&serde_json::json!(-1))).is_err());
        assert!(parse_timestamp(Some(&serde_json::json!(u64::MAX))).is_err());
        assert_eq!(
            parse_timestamp(Some(&serde_json::json!("2024-03-09T16:00:00Z"))).unwrap(),
            1_710_000_000_000_000
        );
        assert_eq!(
            parse_timestamp(Some(&serde_json::json!(1710000000.25))).unwrap(),
            1_710_000_000_250_000
        );
    }
}
