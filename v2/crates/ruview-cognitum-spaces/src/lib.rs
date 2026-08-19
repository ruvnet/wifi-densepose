//! Cognitum Spaces read client (ADR-325).
//!
//! This crate consumes tenant-scoped semantic P2/P3 state only. It never
//! uploads raw CSI/CIR, RF tensors, pose frames, vital waveforms, recordings,
//! or identity observations, and it exposes no action method.

use std::time::Duration;

use reqwest::redirect::Policy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_SPACES: usize = 100;
const MAX_JSON_DEPTH: usize = 16;
const MAX_JSON_NODES: usize = 10_000;
const MAX_STRING_BYTES: usize = 4096;
const REQUIRED_EXCLUSIONS: [&str; 7] = [
    "raw_csi",
    "cir",
    "rf_tensors",
    "recordings",
    "pose_frames",
    "vital_waveforms",
    "identity_observations",
];

#[derive(Clone)]
pub enum Credential {
    OAuth(String),
    ApiKey(String),
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OAuth(_) => f.write_str("OAuth(<redacted>)"),
            Self::ApiKey(_) => f.write_str("ApiKey(<redacted>)"),
        }
    }
}

impl Credential {
    pub fn oauth(token: impl Into<String>) -> Result<Self, Error> {
        secret(Self::OAuth, token.into())
    }

    pub fn api_key(key: impl Into<String>) -> Result<Self, Error> {
        let key = key.into();
        if !key.starts_with("cog_") || key.len() == 4 {
            return Err(Error::InvalidCredential);
        }
        secret(Self::ApiKey, key)
    }
}

fn secret(make: impl FnOnce(String) -> Credential, value: String) -> Result<Credential, Error> {
    if value.is_empty()
        || value.len() > 16_384
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        return Err(Error::InvalidCredential);
    }
    Ok(make(value))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Spaces base URL must be HTTPS (HTTP is allowed only on loopback)")]
    InsecureUrl,
    #[error("invalid Spaces base URL")]
    InvalidUrl,
    #[error("invalid or empty credential")]
    InvalidCredential,
    #[error("invalid Spaces request: {0}")]
    InvalidRequest(String),
    #[error("Spaces request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("Spaces rejected the credential ({0})")]
    Authentication(u16),
    #[error("Spaces returned HTTP {0}")]
    Http(u16),
    #[error("Spaces response is too large")]
    ResponseTooLarge,
    #[error("Spaces response is not JSON")]
    ContentType,
    #[error("Spaces response violates the semantic boundary: {0}")]
    InvalidResponse(String),
}

#[derive(Clone, Debug)]
pub struct Client {
    base: Url,
    endpoint: Url,
    credential: Credential,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base_url: &str, credential: Credential) -> Result<Self, Error> {
        let base = Url::parse(base_url).map_err(|_| Error::InvalidUrl)?;
        let loopback = base
            .host_str()
            .is_some_and(|host| host == "localhost" || host == "127.0.0.1" || host == "::1");
        if base.scheme() != "https" && !(base.scheme() == "http" && loopback) {
            return Err(Error::InsecureUrl);
        }
        if !base.username().is_empty()
            || base.password().is_some()
            || base.query().is_some()
            || base.fragment().is_some()
        {
            return Err(Error::InvalidUrl);
        }
        let endpoint = base.join("/v1/spaces").map_err(|_| Error::InvalidUrl)?;
        let http = reqwest::Client::builder()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .user_agent(concat!(
                "ruview-cognitum-spaces/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()?;
        Ok(Self {
            base,
            endpoint,
            credential,
            http,
        })
    }

    pub async fn list(&self) -> Result<SpacesResponse, Error> {
        let body = self.get(self.endpoint.clone()).await?;
        decode(&body)
    }

    /// Read one stable page from the versioned Cognitum spatial hierarchy.
    /// This is a read-only method; the client exposes no publisher, approval,
    /// command, or actuator operation.
    pub async fn list_spatial(
        &self,
        kind: SpatialKind,
        page: &PageRequest,
    ) -> Result<SpatialResponse, Error> {
        page.validate()?;
        if matches!(self.credential, Credential::ApiKey(_)) && page.workspace_id.is_none() {
            return Err(Error::InvalidRequest(
                "API-key spatial reads require a workspace id".into(),
            ));
        }
        let mut endpoint = self
            .base
            .join(&format!("/v1/spatial/{}", kind.as_str()))
            .map_err(|_| Error::InvalidUrl)?;
        {
            let mut query = endpoint.query_pairs_mut();
            query.append_pair("limit", &page.limit.to_string());
            if let Some(cursor) = &page.cursor {
                query.append_pair("cursor", cursor);
            }
            if let Some(workspace_id) = &page.workspace_id {
                query.append_pair("workspaceId", workspace_id);
            }
        }
        let body = self.get(endpoint).await?;
        decode_spatial(&body, kind)
    }

    async fn get(&self, endpoint: Url) -> Result<Vec<u8>, Error> {
        let mut request = self.http.get(endpoint).header("Accept", "application/json");
        request = match &self.credential {
            Credential::OAuth(token) => request.bearer_auth(token),
            Credential::ApiKey(key) => request.header("X-API-Key", key),
        };
        let mut response = request.send().await?;
        let status = response.status();
        if status.as_u16() == 401 || status.as_u16() == 403 {
            return Err(Error::Authentication(status.as_u16()));
        }
        if !status.is_success() {
            return Err(Error::Http(status.as_u16()));
        }
        let is_json = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .is_some_and(|m| m.trim().eq_ignore_ascii_case("application/json"))
            });
        if !is_json {
            return Err(Error::ContentType);
        }
        if response
            .content_length()
            .is_some_and(|n| n > MAX_RESPONSE_BYTES as u64)
        {
            return Err(Error::ResponseTooLarge);
        }
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
                return Err(Error::ResponseTooLarge);
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }
}

/// Versioned resource collections available from `/v1/spatial`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpatialKind {
    Sites,
    Buildings,
    Floors,
    Spaces,
    Zones,
    Entities,
    Events,
    Alerts,
}

impl SpatialKind {
    /// Stable wire path segment.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sites => "sites",
            Self::Buildings => "buildings",
            Self::Floors => "floors",
            Self::Spaces => "spaces",
            Self::Zones => "zones",
            Self::Entities => "entities",
            Self::Events => "events",
            Self::Alerts => "alerts",
        }
    }
}

/// Bounded stable-page request. OAuth derives its workspace from the signed
/// token; the optional workspace id exists only for the legacy API-key path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageRequest {
    pub limit: u8,
    pub cursor: Option<String>,
    pub workspace_id: Option<String>,
}

impl Default for PageRequest {
    fn default() -> Self {
        Self {
            limit: 50,
            cursor: None,
            workspace_id: None,
        }
    }
}

impl PageRequest {
    fn validate(&self) -> Result<(), Error> {
        if self.limit == 0 || self.limit > 100 {
            return Err(Error::InvalidRequest("limit must be from 1 to 100".into()));
        }
        if self.cursor.as_ref().is_some_and(|value| {
            value.is_empty() || value.len() > 512 || value.chars().any(char::is_control)
        }) {
            return Err(Error::InvalidRequest("cursor is invalid".into()));
        }
        if self
            .workspace_id
            .as_ref()
            .is_some_and(|value| !is_uuid(value))
        {
            return Err(Error::InvalidRequest("workspace id must be a UUID".into()));
        }
        Ok(())
    }
}

fn is_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 36
        && [8, 13, 18, 23].iter().all(|&index| bytes[index] == b'-')
        && matches!(bytes[14], b'1'..=b'8')
        && matches!(bytes[19].to_ascii_lowercase(), b'8' | b'9' | b'a' | b'b')
        && bytes
            .iter()
            .enumerate()
            .all(|(index, byte)| [8, 13, 18, 23].contains(&index) || byte.is_ascii_hexdigit())
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 120
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b':' | b'-'))
}

fn valid_timestamp(value: &str) -> bool {
    chrono::DateTime::parse_from_rfc3339(value).is_ok()
}

fn optional_id_valid(value: Option<&str>) -> bool {
    value.is_none_or(valid_id)
}

/// One versioned P2/P3 hierarchy/event/alert page.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpatialResponse {
    pub object: String,
    pub kind: SpatialKind,
    pub schema_version: String,
    pub data: Vec<SpatialResource>,
    pub next_cursor: Option<String>,
    pub boundary: DataBoundary,
}

/// Common bounded spatial resource. Kind-specific fields stay in `attributes`;
/// tenant/workspace and lineage fields remain typed and independently checked.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpatialResource {
    pub id: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub kind: SpatialKind,
    pub schema_version: String,
    pub privacy: String,
    pub message_id: String,
    pub event_sequence: u64,
    pub version: u64,
    pub site_id: Option<String>,
    pub building_id: Option<String>,
    pub floor_id: Option<String>,
    pub space_id: Option<String>,
    pub zone_id: Option<String>,
    pub name: Option<String>,
    pub entity_type: Option<String>,
    pub identity_mode: Option<String>,
    pub event_type: Option<String>,
    pub alert_type: Option<String>,
    pub severity: Option<String>,
    pub status: Option<String>,
    #[serde(default)]
    pub related_event_ids: Vec<String>,
    pub observed_at: String,
    pub expires_at: Option<String>,
    pub retention_expires_at: Option<String>,
    pub confidence: Option<f64>,
    pub created_at: Option<String>,
    pub updated_at: Option<String>,
    #[serde(default)]
    pub attributes: Value,
    #[serde(default)]
    pub provenance: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpacesResponse {
    pub object: String,
    pub data: Vec<Space>,
    pub boundary: DataBoundary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Space {
    pub id: String,
    pub tenant_id: String,
    pub workspace_id: Option<String>,
    pub site_id: String,
    pub name: String,
    pub version: u64,
    pub privacy: String,
    pub status: String,
    pub connection: String,
    pub state: SemanticState,
    pub provenance: Value,
    pub hardware: Value,
    pub data_boundary: Value,
    #[serde(default)]
    pub observed_at: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticState {
    pub occupancy: Option<u64>,
    pub confidence: Option<f64>,
    pub observed_at: Option<String>,
    pub freshness_ms: Option<u64>,
    pub classification: String,
    pub uncertainty: Value,
    pub evidence: Vec<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataBoundary {
    pub authoritative_state: String,
    pub cloud_role: String,
    pub excluded: Vec<String>,
}

pub fn decode(bytes: &[u8]) -> Result<SpacesResponse, Error> {
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(Error::ResponseTooLarge);
    }
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|_| Error::InvalidResponse("malformed JSON".into()))?;
    validate_value(&value, 0)?;
    let response: SpacesResponse = serde_json::from_value(value)
        .map_err(|e| Error::InvalidResponse(format!("schema mismatch: {e}")))?;
    if response.object != "list" || response.data.len() > MAX_SPACES {
        return Err(Error::InvalidResponse("invalid list envelope".into()));
    }
    if response.boundary.authoritative_state != "HomeCore Edge"
        || REQUIRED_EXCLUSIONS.iter().any(|required| {
            !response
                .boundary
                .excluded
                .iter()
                .any(|excluded| excluded == required)
        })
    {
        return Err(Error::InvalidResponse(
            "incomplete edge privacy boundary".into(),
        ));
    }
    for space in &response.data {
        if space.id.is_empty()
            || space.tenant_id.is_empty()
            || space.site_id.is_empty()
            || space.name.is_empty()
        {
            return Err(Error::InvalidResponse(
                "space identity is incomplete".into(),
            ));
        }
        if !matches!(space.privacy.as_str(), "P2" | "P3") || space.state.classification != "P2" {
            return Err(Error::InvalidResponse("non-semantic privacy class".into()));
        }
        if space
            .state
            .confidence
            .is_some_and(|v| !v.is_finite() || !(0.0..=1.0).contains(&v))
        {
            return Err(Error::InvalidResponse("invalid confidence".into()));
        }
    }
    Ok(response)
}

/// Decode and independently enforce one `/v1/spatial/{kind}` page.
pub fn decode_spatial(bytes: &[u8], expected_kind: SpatialKind) -> Result<SpatialResponse, Error> {
    if bytes.len() > MAX_RESPONSE_BYTES {
        return Err(Error::ResponseTooLarge);
    }
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|_| Error::InvalidResponse("malformed JSON".into()))?;
    validate_value(&value, 0)?;
    let response: SpatialResponse = serde_json::from_value(value)
        .map_err(|error| Error::InvalidResponse(format!("spatial schema mismatch: {error}")))?;
    if response.object != "list"
        || response.kind != expected_kind
        || response.schema_version != "1.0"
        || response.data.len() > MAX_SPACES
    {
        return Err(Error::InvalidResponse(
            "invalid spatial list envelope".into(),
        ));
    }
    if response.boundary.authoritative_state != "HomeCore Edge"
        || REQUIRED_EXCLUSIONS.iter().any(|required| {
            !response
                .boundary
                .excluded
                .iter()
                .any(|excluded| excluded == required)
        })
    {
        return Err(Error::InvalidResponse(
            "incomplete edge privacy boundary".into(),
        ));
    }
    if response.next_cursor.as_ref().is_some_and(|cursor| {
        cursor.is_empty() || cursor.len() > 512 || cursor.chars().any(char::is_control)
    }) {
        return Err(Error::InvalidResponse("invalid next cursor".into()));
    }
    for record in &response.data {
        if !valid_id(&record.id)
            || record.tenant_id.is_empty()
            || !is_uuid(&record.workspace_id)
            || record.kind != expected_kind
            || record.schema_version != "1.0"
            || !valid_id(&record.message_id)
            || record.version == 0
            || !valid_timestamp(&record.observed_at)
            || record
                .expires_at
                .as_deref()
                .is_some_and(|value| !valid_timestamp(value))
            || record
                .retention_expires_at
                .as_deref()
                .is_some_and(|value| !valid_timestamp(value))
            || record
                .created_at
                .as_deref()
                .is_some_and(|value| !valid_timestamp(value))
            || record
                .updated_at
                .as_deref()
                .is_some_and(|value| !valid_timestamp(value))
            || !optional_id_valid(record.site_id.as_deref())
            || !optional_id_valid(record.building_id.as_deref())
            || !optional_id_valid(record.floor_id.as_deref())
            || !optional_id_valid(record.space_id.as_deref())
            || !optional_id_valid(record.zone_id.as_deref())
            || record.related_event_ids.len() > 32
            || record.related_event_ids.iter().any(|id| !valid_id(id))
            || record
                .related_event_ids
                .iter()
                .enumerate()
                .any(|(index, id)| record.related_event_ids[..index].contains(id))
            || !record.attributes.is_object()
            || !record.provenance.is_object()
        {
            return Err(Error::InvalidResponse(
                "spatial resource identity is incomplete".into(),
            ));
        }
        if let Some(expires_at) = record.expires_at.as_deref() {
            let observed = chrono::DateTime::parse_from_rfc3339(&record.observed_at)
                .map_err(|_| Error::InvalidResponse("invalid observed timestamp".into()))?;
            let expires = chrono::DateTime::parse_from_rfc3339(expires_at)
                .map_err(|_| Error::InvalidResponse("invalid expiry timestamp".into()))?;
            if expires <= observed {
                return Err(Error::InvalidResponse(
                    "expiry must follow observation".into(),
                ));
            }
        }
        if !matches!(record.privacy.as_str(), "P2" | "P3") {
            return Err(Error::InvalidResponse("non-semantic privacy class".into()));
        }
        if record
            .confidence
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(Error::InvalidResponse("invalid confidence".into()));
        }
        if matches!(record.kind, SpatialKind::Buildings | SpatialKind::Floors)
            && record.site_id.as_deref().is_none_or(str::is_empty)
        {
            return Err(Error::InvalidResponse(
                "spatial parent is incomplete".into(),
            ));
        }
        if matches!(record.kind, SpatialKind::Spaces)
            && (record.site_id.as_deref().is_none_or(str::is_empty)
                || record.building_id.as_deref().is_none_or(str::is_empty)
                || record.floor_id.as_deref().is_none_or(str::is_empty))
        {
            return Err(Error::InvalidResponse(
                "spatial parent is incomplete".into(),
            ));
        }
        if matches!(
            record.kind,
            SpatialKind::Zones | SpatialKind::Entities | SpatialKind::Events | SpatialKind::Alerts
        ) && (record.site_id.as_deref().is_none_or(str::is_empty)
            || record.space_id.as_deref().is_none_or(str::is_empty))
        {
            return Err(Error::InvalidResponse(
                "spatial parent is incomplete".into(),
            ));
        }
        if record.kind == SpatialKind::Entities
            && (!matches!(
                record.entity_type.as_deref(),
                Some("sensor" | "person" | "object" | "track")
            ) || matches!(record.entity_type.as_deref(), Some("person" | "track"))
                && record.identity_mode.as_deref() != Some("anonymous"))
        {
            return Err(Error::InvalidResponse(
                "entity privacy contract is invalid".into(),
            ));
        }
        if record.kind == SpatialKind::Events
            && record.event_type.as_deref().is_none_or(str::is_empty)
        {
            return Err(Error::InvalidResponse("event type is missing".into()));
        }
        if record.kind == SpatialKind::Alerts
            && (record.alert_type.as_deref().is_none_or(str::is_empty)
                || !matches!(
                    record.severity.as_deref(),
                    Some("info" | "warning" | "critical")
                )
                || !matches!(
                    record.status.as_deref(),
                    Some("open" | "acknowledged" | "resolved")
                ))
        {
            return Err(Error::InvalidResponse("alert contract is invalid".into()));
        }
    }
    Ok(response)
}

fn validate_value(value: &Value, depth: usize) -> Result<(), Error> {
    let mut nodes = 0;
    validate_value_inner(value, depth, &mut nodes)
}

fn validate_value_inner(value: &Value, depth: usize, nodes: &mut usize) -> Result<(), Error> {
    *nodes = nodes.saturating_add(1);
    if *nodes > MAX_JSON_NODES {
        return Err(Error::InvalidResponse(
            "JSON structure exceeds node bound".into(),
        ));
    }
    if depth > MAX_JSON_DEPTH {
        return Err(Error::InvalidResponse("JSON nesting is too deep".into()));
    }
    match value {
        Value::String(s) if s.len() > MAX_STRING_BYTES => {
            return Err(Error::InvalidResponse("string exceeds bound".into()));
        }
        Value::Array(items) if items.len() > 1000 => {
            return Err(Error::InvalidResponse("array exceeds bound".into()));
        }
        Value::Array(items) => {
            for item in items {
                validate_value_inner(item, depth + 1, nodes)?;
            }
        }
        Value::Object(map) => {
            if map.len() > 128 {
                return Err(Error::InvalidResponse("object exceeds bound".into()));
            }
            for (key, item) in map {
                if key.len() > MAX_STRING_BYTES {
                    return Err(Error::InvalidResponse("object key exceeds bound".into()));
                }
                let normalized: String = key
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect();
                if matches!(
                    normalized.as_str(),
                    "csi"
                        | "rawcsi"
                        | "channelstateinformation"
                        | "cir"
                        | "rawcir"
                        | "channelimpulseresponse"
                        | "rftensor"
                        | "rftensors"
                        | "packetcapture"
                        | "packetcaptures"
                        | "pcap"
                        | "recording"
                        | "recordings"
                        | "audiorecording"
                        | "videorecording"
                        | "poseframe"
                        | "poseframes"
                        | "skeleton"
                        | "keypoints"
                        | "vitalwaveform"
                        | "vitalwaveforms"
                        | "heartratewaveform"
                        | "identityobservation"
                        | "identityobservations"
                        | "biometric"
                        | "biometrics"
                        | "face"
                        | "faces"
                        | "faceembedding"
                ) {
                    return Err(Error::InvalidResponse(format!(
                        "forbidden raw field: {key}"
                    )));
                }
                validate_value_inner(item, depth + 1, nodes)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Vec<u8> {
        br#"{"object":"list","data":[{"id":"room-1","tenantId":"tenant-1","workspaceId":"workspace-1","siteId":"site-1","name":"Room","version":1,"privacy":"P2","status":"live","connection":"connected","state":{"occupancy":1,"confidence":0.9,"observedAt":"2026-08-17T00:00:00Z","freshnessMs":5,"classification":"P2","uncertainty":null,"evidence":[]},"provenance":{},"hardware":{},"dataBoundary":{},"observedAt":"2026-08-17T00:00:00Z","expiresAt":null}],"boundary":{"authoritativeState":"HomeCore Edge","cloudRole":"tenant-scoped semantic synchronization","excluded":["raw_csi","cir","rf_tensors","recordings","pose_frames","vital_waveforms","identity_observations"]}}"#.to_vec()
    }

    fn valid_spatial() -> Vec<u8> {
        br#"{"object":"list","kind":"spaces","schemaVersion":"1.0","data":[{"id":"room-1","tenantId":"tenant-1","workspaceId":"22222222-2222-4222-8222-222222222222","kind":"spaces","schemaVersion":"1.0","privacy":"P2","messageId":"message-1","eventSequence":7,"version":1,"siteId":"site-1","buildingId":"building-1","floorId":"floor-1","spaceId":null,"zoneId":null,"name":"Room","observedAt":"2026-08-19T12:00:00Z","expiresAt":null,"retentionExpiresAt":null,"confidence":0.8,"attributes":{"occupancy":2},"provenance":{"witnessDigest":"abc"}}],"nextCursor":null,"boundary":{"authoritativeState":"HomeCore Edge","cloudRole":"tenant/workspace-scoped semantic synchronization","excluded":["raw_csi","cir","rf_tensors","recordings","pose_frames","vital_waveforms","identity_observations"]}}"#.to_vec()
    }

    #[test]
    fn accepts_bounded_semantic_state() {
        assert_eq!(decode(&valid()).unwrap().data.len(), 1);
    }

    #[test]
    fn rejects_raw_fields_anywhere() {
        let mut value: Value = serde_json::from_slice(&valid()).unwrap();
        value["data"][0]["state"]["raw_csi"] = Value::String("secret".into());
        assert!(matches!(
            decode(&serde_json::to_vec(&value).unwrap()),
            Err(Error::InvalidResponse(_))
        ));
    }

    #[test]
    fn rejects_missing_confidence_as_a_number_outside_bounds() {
        let mut value: Value = serde_json::from_slice(&valid()).unwrap();
        value["data"][0]["state"]["confidence"] = Value::from(2.0);
        assert!(matches!(
            decode(&serde_json::to_vec(&value).unwrap()),
            Err(Error::InvalidResponse(_))
        ));
    }

    #[test]
    fn rejects_incomplete_privacy_boundary() {
        let mut value: Value = serde_json::from_slice(&valid()).unwrap();
        value["boundary"]["excluded"] = serde_json::json!(["raw_csi"]);
        assert!(matches!(
            decode(&serde_json::to_vec(&value).unwrap()),
            Err(Error::InvalidResponse(_))
        ));
    }

    #[test]
    fn rejects_credential_bearing_urls_and_whitespace_secrets() {
        let credential = Credential::oauth("token").unwrap();
        assert!(matches!(
            Client::new("https://user:pass@api.cognitum.one", credential),
            Err(Error::InvalidUrl)
        ));
        assert!(matches!(
            Credential::oauth("token with spaces"),
            Err(Error::InvalidCredential)
        ));
    }

    #[test]
    fn credentials_are_redacted() {
        let c = Credential::oauth("secret-token").unwrap();
        assert!(!format!("{c:?}").contains("secret-token"));
    }

    #[test]
    fn accepts_versioned_spatial_pages() {
        let response = decode_spatial(&valid_spatial(), SpatialKind::Spaces).unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].event_sequence, 7);
    }

    #[test]
    fn spatial_page_is_bound_to_requested_kind_and_parents() {
        assert!(decode_spatial(&valid_spatial(), SpatialKind::Events).is_err());
        let mut value: Value = serde_json::from_slice(&valid_spatial()).unwrap();
        value["data"][0]["floorId"] = Value::Null;
        assert!(matches!(
            decode_spatial(&serde_json::to_vec(&value).unwrap(), SpatialKind::Spaces),
            Err(Error::InvalidResponse(_))
        ));
    }

    #[test]
    fn spatial_page_rejects_cross_boundary_payload_and_bad_workspace() {
        let mut raw: Value = serde_json::from_slice(&valid_spatial()).unwrap();
        raw["data"][0]["attributes"]["pose_frames"] = serde_json::json!([1]);
        assert!(decode_spatial(&serde_json::to_vec(&raw).unwrap(), SpatialKind::Spaces).is_err());

        let mut workspace: Value = serde_json::from_slice(&valid_spatial()).unwrap();
        workspace["data"][0]["workspaceId"] = Value::String("not-a-uuid".into());
        assert!(decode_spatial(
            &serde_json::to_vec(&workspace).unwrap(),
            SpatialKind::Spaces
        )
        .is_err());

        let mut timestamp: Value = serde_json::from_slice(&valid_spatial()).unwrap();
        timestamp["data"][0]["observedAt"] = Value::String("not-a-timestamp".into());
        assert!(decode_spatial(
            &serde_json::to_vec(&timestamp).unwrap(),
            SpatialKind::Spaces
        )
        .is_err());

        let mut alias: Value = serde_json::from_slice(&valid_spatial()).unwrap();
        alias["data"][0]["attributes"]["packet_captures"] = serde_json::json!([1]);
        assert!(decode_spatial(&serde_json::to_vec(&alias).unwrap(), SpatialKind::Spaces).is_err());
    }

    #[test]
    fn page_request_is_bounded_and_api_key_needs_workspace() {
        assert!(PageRequest {
            limit: 0,
            ..PageRequest::default()
        }
        .validate()
        .is_err());
        assert!(PageRequest {
            limit: 50,
            cursor: Some("x".repeat(513)),
            workspace_id: None,
        }
        .validate()
        .is_err());
        assert!(PageRequest {
            workspace_id: Some("22222222-2222-4222-8222-222222222222".into()),
            ..PageRequest::default()
        }
        .validate()
        .is_ok());
        assert!(PageRequest {
            workspace_id: Some("22222222-2222-7222-8222-222222222222".into()),
            ..PageRequest::default()
        }
        .validate()
        .is_ok());
    }
}
