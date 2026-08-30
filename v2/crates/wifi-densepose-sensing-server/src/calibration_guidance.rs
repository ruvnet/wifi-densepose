//! Optional, bounded calibration guidance via Cognitum.
//!
//! The mobile client never receives the service credential. Operators inject
//! `COGNITUM_API_KEY` into this trusted runtime (for example from GCP Secret
//! Manager), and this module sends only allowlisted aggregate state.

use std::collections::VecDeque;
use std::error::Error as _;
use std::fmt;
use std::io::Read;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{rejection::JsonRejection, Extension};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const COGNITUM_COMPLETIONS_URL: &str = "https://api.cognitum.one/v1/chat/completions";
const COGNITUM_MODEL: &str = "cognitum-low";
const MAX_RESPONSE_BYTES: u64 = 64 * 1024;
const MAX_GUIDANCE_BYTES: usize = 8_000;
const MAX_REASONS: usize = 8;
const MAX_REASON_LEN: usize = 64;
const MAX_CONCURRENT_REQUESTS: usize = 2;
const MAX_REQUESTS_PER_WINDOW: usize = 12;
const RATE_WINDOW: Duration = Duration::from_secs(60);
// The shipped mobile REST client aborts after five seconds. Keep both the
// transport and task guards below that so a client does not time out first and
// then retry a still-running, cost-bearing call.
const UPSTREAM_DEADLINE: Duration = Duration::from_secs(4);
const TASK_DEADLINE: Duration = Duration::from_millis(4_500);
const IDEMPOTENCY_DOMAIN: &[u8] = b"ruview.calibration-guidance.request.v1\0";
const ALLOWED_REASONS: &[&str] = &[
    "need_room",
    "need_live_nodes",
    "need_placement",
    "weak_signal",
    "stale_node",
    "need_baseline",
    "baseline_collecting",
    "need_verification",
    "ready_to_save",
];

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationGuidanceRequest {
    pub step: CalibrationStep,
    pub active_node_count: u8,
    pub stale_node_count: u8,
    pub placed_node_count: u8,
    pub rssi_strong_count: u8,
    pub rssi_fair_count: u8,
    pub rssi_weak_count: u8,
    pub baseline_state: BaselineState,
    pub baseline_progress_pct: u8,
    pub local_gate_reasons: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStep {
    Config,
    Nodes,
    Baseline,
    Save,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineState {
    None,
    Collecting,
    Fresh,
    Stale,
    Failed,
}

#[derive(Debug, Serialize)]
pub struct CalibrationGuidanceResponse {
    pub guidance: String,
    pub request_id: String,
    pub model: String,
    pub resolved_tier: String,
    pub escalated: bool,
    pub cap_degraded: bool,
    pub price_usd: f64,
}

#[derive(Debug, Deserialize)]
struct CognitumResponse {
    model: String,
    choices: Vec<CognitumChoice>,
    x_cognitum: CognitumReceipt,
}

#[derive(Debug, Deserialize)]
struct CognitumChoice {
    index: u32,
    message: CognitumMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize)]
struct CognitumMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct CognitumReceipt {
    request_id: String,
    resolved_model: String,
    resolved_tier: String,
    escalated: bool,
    cap_degraded: bool,
    price_usd: f64,
}

/// Startup failures intentionally carry no credential text or upstream body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationGuidanceConfigError {
    AuthenticationRequired,
    InvalidCredential,
}

impl fmt::Display for CalibrationGuidanceConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::AuthenticationRequired => {
                "Cognitum calibration guidance requires RuView API authentication"
            }
            Self::InvalidCredential => "Cognitum calibration guidance credential is invalid",
        })
    }
}

impl std::error::Error for CalibrationGuidanceConfigError {}

/// Cloneable broker state installed as an Axum extension. The key is private,
/// has no `Debug` implementation, and is never included in an error or log.
#[derive(Clone)]
pub struct CalibrationGuidanceBroker {
    inner: Arc<BrokerInner>,
}

struct BrokerInner {
    api_key: Option<Box<str>>,
    endpoint: Box<str>,
    agent: ureq::Agent,
    concurrency: Arc<Semaphore>,
    recent_requests: Mutex<VecDeque<Instant>>,
    max_requests_per_window: usize,
    rate_window: Duration,
    task_deadline: Duration,
}

impl CalibrationGuidanceBroker {
    /// Load the optional server-owned key once at startup. A configured key with
    /// authentication disabled is a fatal configuration error: an open LAN API
    /// must never become a spending oracle.
    pub fn from_env(auth_enabled: bool) -> Result<Self, CalibrationGuidanceConfigError> {
        let api_key = match std::env::var("COGNITUM_API_KEY") {
            Ok(value) if value.is_empty() => None,
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(CalibrationGuidanceConfigError::InvalidCredential)
            }
        };
        Self::from_optional_key(api_key, auth_enabled)
    }

    fn from_optional_key(
        api_key: Option<String>,
        auth_enabled: bool,
    ) -> Result<Self, CalibrationGuidanceConfigError> {
        if api_key.is_some() && !auth_enabled {
            return Err(CalibrationGuidanceConfigError::AuthenticationRequired);
        }
        if api_key.as_deref().is_some_and(|key| !valid_api_key(key)) {
            return Err(CalibrationGuidanceConfigError::InvalidCredential);
        }
        Ok(Self::new(
            api_key,
            COGNITUM_COMPLETIONS_URL.to_owned(),
            UPSTREAM_DEADLINE,
            TASK_DEADLINE,
            MAX_CONCURRENT_REQUESTS,
            MAX_REQUESTS_PER_WINDOW,
            RATE_WINDOW,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        api_key: Option<String>,
        endpoint: String,
        upstream_deadline: Duration,
        task_deadline: Duration,
        max_concurrent_requests: usize,
        max_requests_per_window: usize,
        rate_window: Duration,
    ) -> Self {
        let connect_deadline = upstream_deadline.min(Duration::from_secs(2));
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(connect_deadline)
            .timeout(upstream_deadline)
            .redirects(0)
            .build();
        Self {
            inner: Arc::new(BrokerInner {
                api_key: api_key.map(String::into_boxed_str),
                endpoint: endpoint.into_boxed_str(),
                agent,
                concurrency: Arc::new(Semaphore::new(max_concurrent_requests)),
                recent_requests: Mutex::new(VecDeque::new()),
                max_requests_per_window,
                rate_window,
                task_deadline,
            }),
        }
    }

    pub fn is_configured(&self) -> bool {
        self.inner.api_key.is_some()
    }

    fn api_key(&self) -> Result<&str, GuidanceError> {
        self.inner
            .api_key
            .as_deref()
            .ok_or(GuidanceError::NOT_CONFIGURED)
    }

    fn task_deadline(&self) -> Duration {
        self.inner.task_deadline
    }

    fn try_admit(&self) -> Result<OwnedSemaphorePermit, GuidanceError> {
        let permit = Arc::clone(&self.inner.concurrency)
            .try_acquire_owned()
            .map_err(|_| GuidanceError::BUSY)?;
        let now = Instant::now();
        let mut recent = self
            .inner
            .recent_requests
            .lock()
            .map_err(|_| GuidanceError::INTERNAL)?;
        while recent.front().is_some_and(|started| {
            now.saturating_duration_since(*started) >= self.inner.rate_window
        }) {
            recent.pop_front();
        }
        if recent.len() >= self.inner.max_requests_per_window {
            return Err(GuidanceError::RATE_LIMITED);
        }
        recent.push_back(now);
        Ok(permit)
    }

    #[cfg(test)]
    fn for_test(
        endpoint: String,
        upstream_deadline: Duration,
        max_concurrent_requests: usize,
        max_requests_per_window: usize,
    ) -> Self {
        Self::new(
            Some(format!("cog_{}", "a".repeat(64))),
            endpoint,
            upstream_deadline,
            upstream_deadline.saturating_add(Duration::from_millis(100)),
            max_concurrent_requests,
            max_requests_per_window,
            Duration::from_secs(60),
        )
    }
}

#[derive(Debug, Serialize)]
struct GuidanceErrorEnvelope {
    code: &'static str,
    message: &'static str,
}

#[derive(Debug, Clone, Copy)]
struct GuidanceError {
    status: StatusCode,
    code: &'static str,
    message: &'static str,
}

impl GuidanceError {
    const NOT_CONFIGURED: Self = Self::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "calibration_guidance_not_configured",
        "Cognitum calibration guidance is not configured",
    );
    const BUSY: Self = Self::new(
        StatusCode::TOO_MANY_REQUESTS,
        "calibration_guidance_busy",
        "Calibration guidance is already processing the maximum number of requests",
    );
    const RATE_LIMITED: Self = Self::new(
        StatusCode::TOO_MANY_REQUESTS,
        "calibration_guidance_rate_limited",
        "Calibration guidance request limit reached; try again later",
    );
    const UPSTREAM_RATE_LIMITED: Self = Self::new(
        StatusCode::TOO_MANY_REQUESTS,
        "cognitum_rate_limited",
        "Cognitum guidance request limit reached; try again later",
    );
    const UPSTREAM_TIMEOUT: Self = Self::new(
        StatusCode::GATEWAY_TIMEOUT,
        "cognitum_timeout",
        "Cognitum guidance timed out",
    );
    const UPSTREAM_AUTH: Self = Self::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "cognitum_credential_rejected",
        "Cognitum guidance credential was rejected",
    );
    const UPSTREAM_SCOPE: Self = Self::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "cognitum_scope_required",
        "Cognitum guidance credential lacks the required low-tier scope",
    );
    const UPSTREAM_REJECTED: Self = Self::new(
        StatusCode::BAD_GATEWAY,
        "cognitum_request_rejected",
        "Cognitum rejected the bounded guidance request",
    );
    const UPSTREAM_UNAVAILABLE: Self = Self::new(
        StatusCode::BAD_GATEWAY,
        "cognitum_unavailable",
        "Cognitum guidance is temporarily unavailable",
    );
    const INVALID_RESPONSE: Self = Self::new(
        StatusCode::BAD_GATEWAY,
        "cognitum_invalid_response",
        "Cognitum returned invalid guidance",
    );
    const INTERNAL: Self = Self::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "calibration_guidance_internal_error",
        "Calibration guidance failed",
    );

    const fn new(status: StatusCode, code: &'static str, message: &'static str) -> Self {
        Self {
            status,
            code,
            message,
        }
    }

    const fn invalid_request(message: &'static str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_calibration_guidance_request",
            message,
        )
    }
}

impl IntoResponse for GuidanceError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(GuidanceErrorEnvelope {
                code: self.code,
                message: self.message,
            }),
        )
            .into_response()
    }
}

fn valid_api_key(api_key: &str) -> bool {
    api_key.len() == 68
        && api_key.starts_with("cog_")
        && api_key[4..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_request(request: &CalibrationGuidanceRequest) -> Result<(), &'static str> {
    let node_counts = [
        request.active_node_count,
        request.stale_node_count,
        request.placed_node_count,
        request.rssi_strong_count,
        request.rssi_fair_count,
        request.rssi_weak_count,
    ];
    if node_counts.iter().any(|count| *count > 32) {
        return Err("Node count exceeds the calibration summary limit");
    }
    let rssi_total = u16::from(request.rssi_strong_count)
        + u16::from(request.rssi_fair_count)
        + u16::from(request.rssi_weak_count);
    if rssi_total != u16::from(request.active_node_count) {
        return Err("RSSI band counts must equal the active node count");
    }
    if request.placed_node_count > request.active_node_count {
        return Err("Placed node count must not exceed the active node count");
    }
    if request.baseline_progress_pct > 100 {
        return Err("Baseline progress must be between 0 and 100");
    }
    if request.local_gate_reasons.len() > MAX_REASONS
        || request.local_gate_reasons.iter().any(|reason| {
            reason.is_empty()
                || reason.len() > MAX_REASON_LEN
                || reason.chars().any(char::is_control)
                || !ALLOWED_REASONS.contains(&reason.as_str())
        })
    {
        return Err("Local gate reasons exceed the calibration summary limit");
    }
    Ok(())
}

fn bounded_receipt_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn valid_guidance_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_GUIDANCE_BYTES
        && !value
            .chars()
            .any(|character| character.is_control() && !matches!(character, '\n' | '\r' | '\t'))
}

fn parse_cognitum_response(
    value: CognitumResponse,
) -> Result<CalibrationGuidanceResponse, GuidanceError> {
    if value.model != COGNITUM_MODEL || value.choices.len() != 1 {
        return Err(GuidanceError::INVALID_RESPONSE);
    }
    let choice = value
        .choices
        .into_iter()
        .next()
        .ok_or(GuidanceError::INVALID_RESPONSE)?;
    let guidance = choice.message.content.trim().to_owned();
    if choice.index != 0
        || choice.finish_reason != "stop"
        || !valid_guidance_text(&guidance)
        || !bounded_receipt_text(&value.x_cognitum.request_id)
        || value.x_cognitum.resolved_model != COGNITUM_MODEL
        || value.x_cognitum.resolved_tier != "low"
        || value.x_cognitum.escalated
        || value.x_cognitum.cap_degraded
        || !value.x_cognitum.price_usd.is_finite()
        || value.x_cognitum.price_usd < 0.0
    {
        return Err(GuidanceError::INVALID_RESPONSE);
    }
    Ok(CalibrationGuidanceResponse {
        guidance,
        request_id: value.x_cognitum.request_id,
        model: value.x_cognitum.resolved_model,
        resolved_tier: value.x_cognitum.resolved_tier,
        escalated: value.x_cognitum.escalated,
        cap_degraded: value.x_cognitum.cap_degraded,
        price_usd: value.x_cognitum.price_usd,
    })
}

fn build_cognitum_body(request: &CalibrationGuidanceRequest) -> serde_json::Value {
    let prompt = serde_json::json!({
        "schema": "ruview.calibration-guidance-summary.v1",
        "step": request.step,
        "active_node_count": request.active_node_count,
        "stale_node_count": request.stale_node_count,
        "placed_node_count": request.placed_node_count,
        "rssi_bands": {
            "strong": request.rssi_strong_count,
            "fair": request.rssi_fair_count,
            "weak": request.rssi_weak_count,
        },
        "baseline_state": request.baseline_state,
        "baseline_progress_pct": request.baseline_progress_pct,
        "local_gate_reasons": request.local_gate_reasons,
    });
    serde_json::json!({
        "model": COGNITUM_MODEL,
        "stream": false,
        "cache": false,
        "reasoning": { "enabled": false },
        "temperature": 0.1,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": "You are RuView installation guidance. Use only the anonymous aggregate summary. Give at most three short, ordered placement or calibration actions. Preserve local gate results, state uncertainty, never claim hardware validation, never infer room geometry, identity, pose, health, or measurements absent from the summary."
            },
            { "role": "user", "content": prompt.to_string() }
        ]
    })
}

/// Cognitum namespaces idempotency records by the authenticated API key. Hashing
/// the complete outbound body makes retries stable without exposing a room or
/// node identifier; any material summary or prompt-contract change gets a new
/// key. An unchanged summary intentionally replays the same advisory receipt.
fn idempotency_key_for_body(body: &serde_json::Value) -> Result<String, GuidanceError> {
    let bytes = serde_json::to_vec(body).map_err(|_| GuidanceError::INTERNAL)?;
    let mut hasher = Sha256::new();
    hasher.update(IDEMPOTENCY_DOMAIN);
    hasher.update(bytes);
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut key = String::with_capacity(14 + digest.len() * 2);
    key.push_str("ruview-cal-v1-");
    for byte in digest {
        key.push(HEX[(byte >> 4) as usize] as char);
        key.push(HEX[(byte & 0x0f) as usize] as char);
    }
    if key.len() > 128
        || !key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(GuidanceError::INTERNAL);
    }
    Ok(key)
}

fn upstream_error(error: ureq::Error) -> GuidanceError {
    match error {
        ureq::Error::Status(401 | 403, _) => GuidanceError::UPSTREAM_AUTH,
        ureq::Error::Status(402, _) => GuidanceError::UPSTREAM_SCOPE,
        ureq::Error::Status(408 | 504, _) => GuidanceError::UPSTREAM_TIMEOUT,
        ureq::Error::Status(429, _) => GuidanceError::UPSTREAM_RATE_LIMITED,
        ureq::Error::Status(400 | 404 | 405 | 409 | 415 | 422, _) => {
            GuidanceError::UPSTREAM_REJECTED
        }
        ureq::Error::Transport(transport) if transport_is_timeout(&transport) => {
            GuidanceError::UPSTREAM_TIMEOUT
        }
        ureq::Error::Status(_, _) | ureq::Error::Transport(_) => {
            GuidanceError::UPSTREAM_UNAVAILABLE
        }
    }
}

fn transport_is_timeout(transport: &ureq::Transport) -> bool {
    let mut source = transport.source();
    while let Some(error) = source {
        if error.downcast_ref::<std::io::Error>().is_some_and(|error| {
            matches!(
                error.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            )
        }) {
            return true;
        }
        source = error.source();
    }
    false
}

fn call_cognitum(
    broker: &CalibrationGuidanceBroker,
    request: CalibrationGuidanceRequest,
) -> Result<CalibrationGuidanceResponse, GuidanceError> {
    validate_request(&request).map_err(GuidanceError::invalid_request)?;
    let api_key = broker.api_key()?;
    let body = build_cognitum_body(&request);
    let idempotency_key = idempotency_key_for_body(&body)?;
    let response = broker
        .inner
        .agent
        .post(&broker.inner.endpoint)
        .set("Accept", "application/json")
        .set("Content-Type", "application/json")
        .set("X-API-Key", api_key)
        .set("Idempotency-Key", &idempotency_key)
        .send_json(body)
        .map_err(upstream_error)?;
    if !response
        .header("content-type")
        .is_some_and(|value| value.to_ascii_lowercase().starts_with("application/json"))
    {
        return Err(GuidanceError::INVALID_RESPONSE);
    }
    let mut text = String::new();
    response
        .into_reader()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_string(&mut text)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::TimedOut {
                GuidanceError::UPSTREAM_TIMEOUT
            } else {
                GuidanceError::INVALID_RESPONSE
            }
        })?;
    if text.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(GuidanceError::INVALID_RESPONSE);
    }
    let value: CognitumResponse =
        serde_json::from_str(&text).map_err(|_| GuidanceError::INVALID_RESPONSE)?;
    parse_cognitum_response(value)
}

pub async fn handler(
    Extension(broker): Extension<CalibrationGuidanceBroker>,
    payload: Result<Json<CalibrationGuidanceRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(request) => request,
        Err(_) => {
            return GuidanceError::invalid_request("Calibration guidance request is invalid")
                .into_response()
        }
    };
    if let Err(message) = validate_request(&request) {
        return GuidanceError::invalid_request(message).into_response();
    }
    if !broker.is_configured() {
        return GuidanceError::NOT_CONFIGURED.into_response();
    }
    let permit = match broker.try_admit() {
        Ok(permit) => permit,
        Err(error) => return error.into_response(),
    };
    let task_deadline = broker.task_deadline();
    let task_broker = broker.clone();
    let task = tokio::task::spawn_blocking(move || {
        // Keep the permit in the blocking task. If the async request is
        // cancelled or times out, detached upstream work still consumes a slot
        // until its own transport deadline fires.
        let _permit = permit;
        call_cognitum(&task_broker, request)
    });
    match tokio::time::timeout(task_deadline, task).await {
        Ok(Ok(Ok(response))) => (StatusCode::OK, Json(response)).into_response(),
        Ok(Ok(Err(error))) => error.into_response(),
        Ok(Err(_)) => GuidanceError::INTERNAL.into_response(),
        Err(_) => GuidanceError::UPSTREAM_TIMEOUT.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use axum::routing::post;
    use axum::Router;
    use std::io::Write as _;
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;
    use tower::ServiceExt;

    fn spawn_mock_upstream(
        response_body: String,
        response_delay: Duration,
    ) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (request_tx, request_rx) = mpsc::channel();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(1)))
                .unwrap();
            let mut request_bytes = Vec::new();
            let mut chunk = [0_u8; 2_048];
            loop {
                let count = stream.read(&mut chunk).unwrap_or(0);
                if count == 0 {
                    break;
                }
                request_bytes.extend_from_slice(&chunk[..count]);
                let Some(header_end) = request_bytes
                    .windows(4)
                    .position(|part| part == b"\r\n\r\n")
                else {
                    continue;
                };
                let header_end = header_end + 4;
                let headers = String::from_utf8_lossy(&request_bytes[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                if request_bytes.len() >= header_end + content_length {
                    break;
                }
            }
            let _ = request_tx.send(String::from_utf8_lossy(&request_bytes).into_owned());
            if !response_delay.is_zero() {
                thread::sleep(response_delay);
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            let _ = stream.write_all(response.as_bytes());
        });
        (format!("http://{address}/v1/chat/completions"), request_rx)
    }

    fn request() -> CalibrationGuidanceRequest {
        CalibrationGuidanceRequest {
            step: CalibrationStep::Nodes,
            active_node_count: 3,
            stale_node_count: 0,
            placed_node_count: 2,
            rssi_strong_count: 1,
            rssi_fair_count: 2,
            rssi_weak_count: 0,
            baseline_state: BaselineState::None,
            baseline_progress_pct: 0,
            local_gate_reasons: vec!["need_placement".to_owned()],
        }
    }

    fn valid_response_value() -> serde_json::Value {
        serde_json::json!({
            "model": "cognitum-low",
            "choices": [{
                "index": 0,
                "message": { "content": "1. Move the weak node away from metal." },
                "finish_reason": "stop"
            }],
            "x_cognitum": {
                "request_id": "req-1",
                "resolved_model": "cognitum-low",
                "resolved_tier": "low",
                "escalated": false,
                "cap_degraded": false,
                "price_usd": 0.00042
            }
        })
    }

    fn valid_response() -> CognitumResponse {
        serde_json::from_value(valid_response_value()).unwrap()
    }

    #[test]
    fn configured_key_requires_ruview_auth_and_errors_redact_it() {
        let secret = format!("cog_{}", "b".repeat(64));
        let error = CalibrationGuidanceBroker::from_optional_key(Some(secret.clone()), false)
            .err()
            .unwrap();
        assert_eq!(
            error,
            CalibrationGuidanceConfigError::AuthenticationRequired
        );
        assert!(!error.to_string().contains(&secret));
    }

    #[test]
    fn invalid_key_fails_startup_without_echoing_secret() {
        let error = CalibrationGuidanceBroker::from_optional_key(
            Some("not-a-cognitum-key".to_owned()),
            true,
        )
        .err()
        .unwrap();
        assert_eq!(error, CalibrationGuidanceConfigError::InvalidCredential);
        assert!(!error.to_string().contains("not-a-cognitum-key"));
    }

    #[test]
    fn rejects_inconsistent_or_unbounded_summary() {
        let mut value = request();
        value.rssi_fair_count = 1;
        assert!(validate_request(&value).is_err());
        value.rssi_fair_count = 2;
        value.local_gate_reasons = vec!["x".repeat(MAX_REASON_LEN + 1)];
        assert!(validate_request(&value).is_err());
    }

    #[test]
    fn exact_low_tier_receipt_is_required() {
        let parsed = parse_cognitum_response(valid_response()).unwrap();
        assert_eq!(parsed.resolved_tier, "low");
        assert!(!parsed.escalated);
        assert!(!parsed.cap_degraded);
        assert_eq!(parsed.price_usd, 0.00042);

        let mut wrong_tier = valid_response();
        wrong_tier.x_cognitum.resolved_tier = "mid".to_owned();
        assert!(parse_cognitum_response(wrong_tier).is_err());
        let mut wrong_model = valid_response();
        wrong_model.x_cognitum.resolved_model = "cognitum-mid".to_owned();
        assert!(parse_cognitum_response(wrong_model).is_err());
        let mut escalated = valid_response();
        escalated.x_cognitum.escalated = true;
        assert!(parse_cognitum_response(escalated).is_err());
        let mut degraded = valid_response();
        degraded.x_cognitum.cap_degraded = true;
        assert!(parse_cognitum_response(degraded).is_err());
        let mut negative_price = valid_response();
        negative_price.x_cognitum.price_usd = -0.01;
        assert!(parse_cognitum_response(negative_price).is_err());
        let mut non_finite_price = valid_response();
        non_finite_price.x_cognitum.price_usd = f64::INFINITY;
        assert!(parse_cognitum_response(non_finite_price).is_err());
    }

    #[test]
    fn idempotency_key_is_stable_bounded_and_body_sensitive() {
        let first_body = build_cognitum_body(&request());
        let first = idempotency_key_for_body(&first_body).unwrap();
        assert_eq!(first, idempotency_key_for_body(&first_body).unwrap());
        assert!(first.len() <= 128);
        assert!(first
            .bytes()
            .all(|byte| { byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-') }));

        let mut changed = request();
        changed.placed_node_count = 3;
        assert_ne!(
            first,
            idempotency_key_for_body(&build_cognitum_body(&changed)).unwrap()
        );
    }

    #[test]
    fn forwards_exact_bounded_body_and_stable_idempotency_header() {
        let response_body = serde_json::json!({
            "model": "cognitum-low",
            "choices": [{
                "index": 0,
                "message": { "content": "1. Move the weak node away from metal." },
                "finish_reason": "stop"
            }],
            "x_cognitum": {
                "request_id": "req-wire-1",
                "resolved_model": "cognitum-low",
                "resolved_tier": "low",
                "escalated": false,
                "cap_degraded": false,
                "price_usd": 0.00042
            }
        })
        .to_string();
        let (endpoint, captured) = spawn_mock_upstream(response_body, Duration::ZERO);
        let broker = CalibrationGuidanceBroker::for_test(endpoint, Duration::from_secs(1), 1, 4);
        let receipt = call_cognitum(&broker, request()).unwrap();
        assert_eq!(receipt.request_id, "req-wire-1");

        let raw_request = captured.recv_timeout(Duration::from_secs(1)).unwrap();
        let (raw_headers, raw_body) = raw_request.split_once("\r\n\r\n").unwrap();
        let normalized_headers = raw_headers.to_ascii_lowercase();
        assert!(normalized_headers.contains("x-api-key: cog_"));
        assert!(normalized_headers.contains("idempotency-key: ruview-cal-v1-"));
        let body: serde_json::Value = serde_json::from_str(raw_body).unwrap();
        assert_eq!(body["model"], "cognitum-low");
        assert_eq!(body["stream"], false);
        assert_eq!(body["cache"], false);
        assert_eq!(body["reasoning"]["enabled"], false);
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn hard_transport_deadline_maps_to_stable_timeout() {
        let (endpoint, _captured) = spawn_mock_upstream(
            valid_response_value().to_string(),
            Duration::from_millis(200),
        );
        let broker = CalibrationGuidanceBroker::for_test(endpoint, Duration::from_millis(30), 1, 4);
        let started = Instant::now();
        let error = call_cognitum(&broker, request()).unwrap_err();
        assert_eq!(error.code, GuidanceError::UPSTREAM_TIMEOUT.code);
        assert!(started.elapsed() < Duration::from_millis(180));
    }

    #[test]
    fn concurrency_and_rate_limits_fail_closed() {
        let broker = CalibrationGuidanceBroker::for_test(
            "http://127.0.0.1:9".to_owned(),
            Duration::from_millis(50),
            1,
            1,
        );
        let permit = broker.try_admit().unwrap();
        assert_eq!(
            broker.try_admit().unwrap_err().code,
            GuidanceError::BUSY.code
        );
        drop(permit);
        assert_eq!(
            broker.try_admit().unwrap_err().code,
            GuidanceError::RATE_LIMITED.code
        );
    }

    #[tokio::test]
    async fn unconfigured_broker_returns_stable_bounded_error_envelope() {
        let broker = CalibrationGuidanceBroker::from_optional_key(None, false).unwrap();
        let app = Router::new()
            .route("/api/v1/calibration/guidance", post(handler))
            .layer(Extension(broker));
        let response = app
            .oneshot(
                Request::post("/api/v1/calibration/guidance")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&request()).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 1_024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "code": "calibration_guidance_not_configured",
                "message": "Cognitum calibration guidance is not configured"
            })
        );
    }
}
