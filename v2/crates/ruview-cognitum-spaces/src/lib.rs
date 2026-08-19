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
            endpoint,
            credential,
            http,
        })
    }

    pub async fn list(&self) -> Result<SpacesResponse, Error> {
        let mut request = self
            .http
            .get(self.endpoint.clone())
            .header("Accept", "application/json");
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
        decode(&body)
    }
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

fn validate_value(value: &Value, depth: usize) -> Result<(), Error> {
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
                validate_value(item, depth + 1)?;
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
                    "rawcsi"
                        | "cir"
                        | "rftensors"
                        | "recordings"
                        | "poseframes"
                        | "vitalwaveforms"
                        | "identityobservations"
                ) {
                    return Err(Error::InvalidResponse(format!(
                        "forbidden raw field: {key}"
                    )));
                }
                validate_value(item, depth + 1)?;
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
}
