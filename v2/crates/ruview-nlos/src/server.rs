//! Authenticated read-only NLOS HTTP and WebSocket surface.
//!
//! Browsers exchange a bearer token for a single-use, short-lived WebSocket
//! ticket because the browser WebSocket API cannot set an Authorization header.
//! Native clients may authenticate the upgrade directly with a bearer header.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;

use crate::protocol::TrackEnvelope;
use crate::{MAX_WIRE_BYTES, TRACK_SCHEMA_V1};

const TICKET_TTL_MS: u64 = 30_000;
const SESSION_TTL_MS: u64 = 60 * 60 * 1_000;
const MAX_TICKETS: usize = 1_024;
const MAX_HISTORY: usize = 64;

/// Thread-safe authenticated publication hub.
#[derive(Clone)]
pub struct NlosHub {
    inner: Arc<HubInner>,
}

struct HubInner {
    bearer_digest: [u8; 32],
    session_id: String,
    latest: RwLock<Option<TrackEnvelope>>,
    history: RwLock<VecDeque<TrackEnvelope>>,
    publish_order: tokio::sync::Mutex<()>,
    last_published_sequence: tokio::sync::Mutex<Option<u64>>,
    tickets: Mutex<BTreeMap<String, Ticket>>,
    broadcast: broadcast::Sender<TrackEnvelope>,
    allowed_origin: Option<axum::http::HeaderValue>,
}

#[derive(Clone, Copy)]
struct Ticket {
    expires_at_unix_ms: u64,
    origin_digest: Option<[u8; 32]>,
}

impl NlosHub {
    /// Create a hub. The bearer token is hashed immediately and never retained.
    pub fn new(bearer_token: &str, session_id: impl Into<String>) -> Result<Self, ServerError> {
        if bearer_token.len() < 32
            || bearer_token.len() > 512
            || bearer_token
                .bytes()
                .any(|byte| !(0x21..=0x7e).contains(&byte))
        {
            return Err(ServerError::WeakToken);
        }
        let session_id = session_id.into();
        if session_id.is_empty()
            || session_id.len() > 64
            || session_id
                .bytes()
                .any(|b| !(b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':')))
        {
            return Err(ServerError::InvalidSession);
        }
        let (broadcast, _) = broadcast::channel(64);
        Ok(Self {
            inner: Arc::new(HubInner {
                bearer_digest: sha256(bearer_token.as_bytes()),
                session_id,
                latest: RwLock::new(None),
                history: RwLock::new(VecDeque::with_capacity(MAX_HISTORY)),
                publish_order: tokio::sync::Mutex::new(()),
                last_published_sequence: tokio::sync::Mutex::new(None),
                tickets: Mutex::new(BTreeMap::new()),
                broadcast,
                allowed_origin: None,
            }),
        })
    }

    /// Permit one exact browser origin for ticket exchange. Wildcards,
    /// credentials, paths, fragments, and cleartext non-loopback origins are
    /// rejected. Call this before cloning the hub.
    pub fn with_allowed_origin(mut self, origin: &str) -> Result<Self, ServerError> {
        let value = validate_origin(origin)?;
        let Some(inner) = Arc::get_mut(&mut self.inner) else {
            return Err(ServerError::Internal);
        };
        inner.allowed_origin = Some(value);
        Ok(self)
    }

    /// Publish one validated, bounded, monotonically ordered envelope.
    pub async fn publish(&self, envelope: TrackEnvelope) -> Result<(), ServerError> {
        envelope.validate()?;
        if envelope.session_id != self.inner.session_id {
            return Err(ServerError::SessionMismatch);
        }
        let encoded = serde_json::to_vec(&envelope).map_err(|_| ServerError::Serialization)?;
        if encoded.len() > MAX_WIRE_BYTES {
            return Err(ServerError::FrameTooLarge);
        }
        let now = now_unix_ms();
        if envelope.expires_at_unix_ms <= now
            || envelope.captured_at_unix_ms > now.saturating_add(1_000)
        {
            return Err(ServerError::StaleOrFuture);
        }
        // Serialize the latest/history/broadcast transition so concurrent
        // publishers cannot expose sequence N+1 before N in another surface.
        let _publish_order = self.inner.publish_order.lock().await;
        let mut last_sequence = self.inner.last_published_sequence.lock().await;
        if last_sequence.is_some_and(|previous| previous >= envelope.sequence) {
            return Err(ServerError::ReplayOrOutOfOrder);
        }
        *last_sequence = Some(envelope.sequence);
        drop(last_sequence);
        let mut latest = self.inner.latest.write().await;
        *latest = Some(envelope.clone());
        drop(latest);
        let mut history = self.inner.history.write().await;
        history.retain(|item| item.expires_at_unix_ms > now);
        while history.len() >= MAX_HISTORY {
            history.pop_front();
        }
        history.push_back(envelope.clone());
        drop(history);
        let _ = self.inner.broadcast.send(envelope);
        Ok(())
    }

    async fn purge_expired(&self, now: u64) {
        let _publish_order = self.inner.publish_order.lock().await;
        let mut latest = self.inner.latest.write().await;
        if latest
            .as_ref()
            .is_some_and(|item| item.expires_at_unix_ms <= now)
        {
            *latest = None;
        }
        drop(latest);
        self.inner
            .history
            .write()
            .await
            .retain(|item| item.expires_at_unix_ms > now);
    }

    /// Build the read-only API router. No permissive CORS layer is installed.
    pub fn router(self) -> Router {
        let router = Router::new()
            .route("/health", get(health))
            .route("/api/v1/nlos/latest", get(latest))
            .route("/api/v1/nlos/tracks", get(tracks))
            .route("/api/v1/nlos/ws-ticket", post(issue_ws_ticket))
            .route("/api/v1/nlos/ws", get(websocket))
            .layer(RequestBodyLimitLayer::new(8 * 1024))
            .with_state(self.clone());
        if let Some(origin) = self.inner.allowed_origin.clone() {
            router.layer(
                CorsLayer::new()
                    .allow_origin(origin)
                    .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
                    .allow_headers([
                        axum::http::header::AUTHORIZATION,
                        axum::http::header::CONTENT_TYPE,
                    ]),
            )
        } else {
            // No CORS headers means browsers remain same-origin by default.
            router
        }
    }

    fn bearer_authorized(&self, headers: &HeaderMap) -> bool {
        let Some(value) = headers.get(axum::http::header::AUTHORIZATION) else {
            return false;
        };
        let Ok(value) = value.to_str() else {
            return false;
        };
        let Some(token) = value.strip_prefix("Bearer ") else {
            return false;
        };
        constant_time_eq(&sha256(token.as_bytes()), &self.inner.bearer_digest)
    }

    fn issue_ticket(
        &self,
        now: u64,
        origin_digest: Option<[u8; 32]>,
    ) -> Result<(String, u64), ServerError> {
        let mut tickets = self
            .inner
            .tickets
            .lock()
            .map_err(|_| ServerError::Internal)?;
        tickets.retain(|_, ticket| ticket.expires_at_unix_ms > now);
        if tickets.len() >= MAX_TICKETS {
            return Err(ServerError::TicketCapacity);
        }
        let mut random = [0_u8; 32];
        getrandom::getrandom(&mut random).map_err(|_| ServerError::Entropy)?;
        let ticket = lowercase_hex(&random);
        let expires = now.saturating_add(TICKET_TTL_MS);
        tickets.insert(
            ticket.clone(),
            Ticket {
                expires_at_unix_ms: expires,
                origin_digest,
            },
        );
        Ok((ticket, expires))
    }

    fn consume_ticket(&self, raw: &str, now: u64, origin_digest: Option<[u8; 32]>) -> bool {
        if raw.len() != 64
            || !raw
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return false;
        }
        let Ok(mut tickets) = self.inner.tickets.lock() else {
            return false;
        };
        tickets.remove(raw).is_some_and(|ticket| {
            ticket.expires_at_unix_ms >= now && ticket.origin_digest == origin_digest
        })
    }

    fn request_origin_digest(&self, headers: &HeaderMap) -> Result<Option<[u8; 32]>, ServerError> {
        let actual = headers.get(axum::http::header::ORIGIN);
        if let Some(expected) = &self.inner.allowed_origin {
            if actual != Some(expected) {
                return Err(ServerError::InvalidOrigin);
            }
        }
        Ok(actual.map(|value| sha256(value.as_bytes())))
    }
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "ruview-nlos",
    })
}

async fn latest(State(hub): State<NlosHub>, headers: HeaderMap) -> Response {
    if !hub.bearer_authorized(&headers) {
        return unauthorized();
    }
    hub.purge_expired(now_unix_ms()).await;
    match hub.inner.latest.read().await.clone() {
        Some(frame) => sensitive(Json(frame).into_response()),
        None => sensitive(StatusCode::NO_CONTENT.into_response()),
    }
}

async fn tracks(State(hub): State<NlosHub>, headers: HeaderMap) -> Response {
    if !hub.bearer_authorized(&headers) {
        return unauthorized();
    }
    hub.purge_expired(now_unix_ms()).await;
    let history: Vec<_> = hub.inner.history.read().await.iter().cloned().collect();
    sensitive(Json(history).into_response())
}

async fn issue_ws_ticket(State(hub): State<NlosHub>, headers: HeaderMap) -> Response {
    if !hub.bearer_authorized(&headers) {
        return unauthorized();
    }
    let now = now_unix_ms();
    let origin_digest = match hub.request_origin_digest(&headers) {
        Ok(value) => value,
        Err(_) => return sensitive(StatusCode::FORBIDDEN.into_response()),
    };
    let (ticket, expires_at_unix_ms) = match hub.issue_ticket(now, origin_digest) {
        Ok(result) => result,
        Err(_) => return sensitive(StatusCode::SERVICE_UNAVAILABLE.into_response()),
    };
    let host = headers
        .get(axum::http::header::HOST)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<axum::http::uri::Authority>().ok())
        .map_or_else(|| "127.0.0.1:8787".to_owned(), |value| value.to_string());
    let forwarded = headers
        .get("x-forwarded-proto")
        .and_then(|value| value.to_str().ok());
    let scheme = if matches!(forwarded, Some("https" | "wss")) {
        "wss"
    } else {
        "ws"
    };
    sensitive(
        Json(WsTicketResponse {
            schema: "ruview.nlos.ws-ticket.v1",
            web_socket_url: format!("{scheme}://{host}/api/v1/nlos/ws?ticket={ticket}"),
            expires_at_unix_ms,
        })
        .into_response(),
    )
}

async fn websocket(
    State(hub): State<NlosHub>,
    headers: HeaderMap,
    Query(query): Query<WsQuery>,
    upgrade: WebSocketUpgrade,
) -> Response {
    let now = now_unix_ms();
    let origin_digest = headers
        .get(axum::http::header::ORIGIN)
        .map(|value| sha256(value.as_bytes()));
    let ticket_ok = query
        .ticket
        .as_deref()
        .is_some_and(|ticket| hub.consume_ticket(ticket, now, origin_digest));
    if !ticket_ok && !hub.bearer_authorized(&headers) {
        return unauthorized();
    }
    upgrade
        .protocols([TRACK_SCHEMA_V1])
        .max_message_size(8 * 1024)
        .max_frame_size(8 * 1024)
        .on_upgrade(move |socket| websocket_session(hub, socket))
}

async fn websocket_session(hub: NlosHub, mut socket: WebSocket) {
    // Subscribe before reading the retained latest value. The sequence
    // watermark below de-duplicates a publication that lands in between.
    let mut receiver = hub.inner.broadcast.subscribe();
    let session_expires_at_unix_ms = now_unix_ms().saturating_add(SESSION_TTL_MS);
    let authenticated = Authenticated {
        schema: "ruview.nlos.authenticated.v1",
        session_id: hub.inner.session_id.clone(),
        expires_at_unix_ms: session_expires_at_unix_ms,
    };
    let Ok(payload) = serde_json::to_string(&authenticated) else {
        return;
    };
    if socket.send(Message::Text(payload)).await.is_err() {
        return;
    }
    let mut last_sent_sequence = None;
    if let Some(latest) = hub
        .inner
        .latest
        .read()
        .await
        .clone()
        .filter(|frame| frame.expires_at_unix_ms > now_unix_ms())
    {
        let Ok(payload) = serde_json::to_string(&latest) else {
            return;
        };
        if socket.send(Message::Text(payload)).await.is_err() {
            return;
        }
        last_sent_sequence = Some(latest.sequence);
    }
    let session_expiry = tokio::time::sleep(std::time::Duration::from_millis(SESSION_TTL_MS));
    tokio::pin!(session_expiry);
    loop {
        tokio::select! {
            message = receiver.recv() => {
                let frame = match message {
                    Ok(frame) => frame,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                };
                if last_sent_sequence.is_some_and(|last| frame.sequence <= last) {
                    continue;
                }
                let Ok(payload) = serde_json::to_string(&frame) else { break; };
                if socket.send(Message::Text(payload)).await.is_err() { break; }
                last_sent_sequence = Some(frame.sequence);
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    Some(Ok(Message::Ping(value))) => {
                        if socket.send(Message::Pong(value)).await.is_err() { break; }
                    }
                    _ => {}
                }
            }
            _ = &mut session_expiry => {
                let _ = socket.send(Message::Close(None)).await;
                break;
            }
        }
    }
}

fn unauthorized() -> Response {
    sensitive(
        (
            StatusCode::UNAUTHORIZED,
            Json(ApiError {
                error: "unauthorized",
            }),
        )
            .into_response(),
    )
}

fn sensitive(mut response: Response) -> Response {
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    response
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis() as u64)
}

fn sha256(value: &[u8]) -> [u8; 32] {
    Sha256::digest(value).into()
}

fn constant_time_eq(left: &[u8; 32], right: &[u8; 32]) -> bool {
    left.iter()
        .zip(right.iter())
        .fold(0_u8, |difference, (a, b)| difference | (a ^ b))
        == 0
}

fn lowercase_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn validate_origin(origin: &str) -> Result<axum::http::HeaderValue, ServerError> {
    if origin.len() > 2_048 || origin.chars().any(|value| matches!(value, '#' | '?' | '@')) {
        return Err(ServerError::InvalidOrigin);
    }
    let uri: axum::http::Uri = origin.parse().map_err(|_| ServerError::InvalidOrigin)?;
    let scheme = uri.scheme_str().ok_or(ServerError::InvalidOrigin)?;
    let authority = uri.authority().ok_or(ServerError::InvalidOrigin)?;
    if uri
        .path_and_query()
        .is_some_and(|path| path.as_str() != "/")
    {
        return Err(ServerError::InvalidOrigin);
    }
    let host = authority.host();
    let loopback = matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]");
    if scheme != "https" && !(scheme == "http" && loopback) {
        return Err(ServerError::InvalidOrigin);
    }
    origin.parse().map_err(|_| ServerError::InvalidOrigin)
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    service: &'static str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WsTicketResponse {
    schema: &'static str,
    web_socket_url: String,
    expires_at_unix_ms: u64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Authenticated {
    schema: &'static str,
    session_id: String,
    expires_at_unix_ms: u64,
}

#[derive(Deserialize)]
struct WsQuery {
    ticket: Option<String>,
}

#[derive(Serialize)]
struct ApiError {
    error: &'static str,
}

/// Server security or publication failure.
#[derive(Debug, Error)]
pub enum ServerError {
    /// Token is outside the shared visible-ASCII length boundary.
    #[error("bearer token must contain 32 to 512 visible ASCII characters")]
    WeakToken,
    /// Session id is empty or too long.
    #[error("invalid server session id")]
    InvalidSession,
    /// Track frame contract failure.
    #[error(transparent)]
    Contract(#[from] crate::protocol::ContractError),
    /// Encoded frame exceeded 256 KiB.
    #[error("track frame exceeds the wire-size bound")]
    FrameTooLarge,
    /// Sequence moved backwards or repeated.
    #[error("replayed or out-of-order track frame")]
    ReplayOrOutOfOrder,
    /// Frame was already expired or materially future-dated at publication.
    #[error("stale or future-dated track frame")]
    StaleOrFuture,
    /// Publisher session did not match the authenticated server session.
    #[error("track frame session does not match server session")]
    SessionMismatch,
    /// Random ticket generation failed.
    #[error("secure entropy unavailable")]
    Entropy,
    /// Too many unexpired tickets exist.
    #[error("ticket capacity reached")]
    TicketCapacity,
    /// Lock poisoning or another internal failure.
    #[error("internal server failure")]
    Internal,
    /// JSON serialization failed.
    #[error("frame serialization failed")]
    Serialization,
    /// Browser origin was not one exact HTTPS origin or loopback development origin.
    #[error("invalid allowed browser origin")]
    InvalidOrigin,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn ticket_is_single_use_and_expiring() {
        let hub = NlosHub::new(&"x".repeat(32), "s1").unwrap();
        let origin = Some(sha256(b"https://app.example.test"));
        let (ticket, expires) = hub.issue_ticket(1_000, origin).unwrap();
        assert!(!hub.consume_ticket(&ticket, expires, None));
        let (ticket, expires) = hub.issue_ticket(1_500, origin).unwrap();
        assert!(hub.consume_ticket(&ticket, expires, origin));
        assert!(!hub.consume_ticket(&ticket, expires, origin));
        let (ticket, expires) = hub.issue_ticket(2_000, origin).unwrap();
        assert!(!hub.consume_ticket(&ticket, expires + 1, origin));
    }

    #[test]
    fn token_digest_comparison_is_exact() {
        let hub = NlosHub::new(&"a".repeat(32), "s1").unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {}", "a".repeat(32)).parse().unwrap(),
        );
        assert!(hub.bearer_authorized(&headers));
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Bearer {}", "b".repeat(32)).parse().unwrap(),
        );
        assert!(!hub.bearer_authorized(&headers));
    }

    #[test]
    fn browser_origin_is_exact_and_https_except_loopback() {
        assert!(NlosHub::new(&"x".repeat(32), "s1")
            .unwrap()
            .with_allowed_origin("https://app.example.test")
            .is_ok());
        assert!(NlosHub::new(&"x".repeat(32), "s1")
            .unwrap()
            .with_allowed_origin("http://127.0.0.1:8081")
            .is_ok());
        for invalid in [
            "*",
            "http://app.example.test",
            "https://user@example.test",
            "https://example.test/path",
        ] {
            assert!(NlosHub::new(&"x".repeat(32), "s1")
                .unwrap()
                .with_allowed_origin(invalid)
                .is_err());
        }
    }

    #[tokio::test]
    async fn ticket_endpoint_requires_bearer_and_returns_strict_contract() {
        let token = "z".repeat(32);
        let router = NlosHub::new(&token, "s1").unwrap().router();
        let unauthorized = router
            .clone()
            .oneshot(
                Request::post("/api/v1/nlos/ws-ticket")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let authorized = router
            .oneshot(
                Request::post("/api/v1/nlos/ws-ticket")
                    .header("host", "127.0.0.1:8787")
                    .header("authorization", format!("Bearer {token}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(authorized.status(), StatusCode::OK);
        assert_eq!(
            authorized.headers().get(axum::http::header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store, max-age=0"))
        );
        let bytes = authorized.into_body().collect().await.unwrap().to_bytes();
        assert!(bytes.len() < 8 * 1024);
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["schema"], "ruview.nlos.ws-ticket.v1");
        assert!(value["webSocketUrl"]
            .as_str()
            .unwrap()
            .starts_with("ws://127.0.0.1:8787/api/v1/nlos/ws?ticket="));
        assert!(value["expiresAtUnixMs"].as_u64().is_some());
    }

    #[tokio::test]
    async fn exact_origin_preflight_is_allowed_without_wildcard() {
        let router = NlosHub::new(&"x".repeat(32), "s1")
            .unwrap()
            .with_allowed_origin("https://app.example.test")
            .unwrap()
            .router();
        let response = router
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/v1/nlos/ws-ticket")
                    .header("origin", "https://app.example.test")
                    .header("access-control-request-method", "POST")
                    .header("access-control-request-headers", "authorization")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .unwrap(),
            "https://app.example.test"
        );
        assert_ne!(
            response
                .headers()
                .get("access-control-allow-origin")
                .unwrap(),
            "*"
        );
    }

    #[tokio::test]
    async fn configured_origin_is_required_for_ticket_issue() {
        let token = "z".repeat(32);
        let router = NlosHub::new(&token, "s1")
            .unwrap()
            .with_allowed_origin("https://app.example.test")
            .unwrap()
            .router();
        let forbidden = router
            .clone()
            .oneshot(
                Request::post("/api/v1/nlos/ws-ticket")
                    .header("host", "nlos.example.test")
                    .header("authorization", format!("Bearer {token}"))
                    .header("origin", "https://other.example.test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);
        assert_eq!(
            forbidden.headers().get(axum::http::header::CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store, max-age=0"))
        );
    }

    #[tokio::test]
    async fn publisher_rejects_session_switch_and_sequence_replay() {
        let fixture = include_str!("../tests/fixtures/track_synthetic.json");
        let mut frame: TrackEnvelope = serde_json::from_str(fixture).unwrap();
        let now = now_unix_ms();
        frame.captured_at_unix_ms = now;
        frame.expires_at_unix_ms = now + 250;
        let hub = NlosHub::new(&"x".repeat(32), frame.session_id.clone()).unwrap();
        hub.publish(frame.clone()).await.unwrap();
        assert!(matches!(
            hub.publish(frame.clone()).await,
            Err(ServerError::ReplayOrOutOfOrder)
        ));
        let mut wrong_session = frame;
        wrong_session.session_id = "another-session".into();
        wrong_session.sequence += 1;
        assert!(matches!(
            hub.publish(wrong_session).await,
            Err(ServerError::SessionMismatch)
        ));
    }

    #[tokio::test]
    async fn concurrent_publication_keeps_history_monotonic() {
        let fixture = include_str!("../tests/fixtures/track_synthetic.json");
        let mut frame: TrackEnvelope = serde_json::from_str(fixture).unwrap();
        let now = now_unix_ms();
        frame.captured_at_unix_ms = now;
        frame.expires_at_unix_ms = now + 250;
        let hub = NlosHub::new(&"x".repeat(32), frame.session_id.clone()).unwrap();
        let barrier = Arc::new(tokio::sync::Barrier::new(32));
        let mut tasks = Vec::new();
        for sequence in 1..=32_u64 {
            let hub = hub.clone();
            let barrier = barrier.clone();
            let mut candidate = frame.clone();
            candidate.sequence = sequence;
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                hub.publish(candidate).await.ok()
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }

        let history = hub.inner.history.read().await;
        let sequences: Vec<_> = history.iter().map(|item| item.sequence).collect();
        assert!(!sequences.is_empty());
        assert!(sequences.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            hub.inner
                .latest
                .read()
                .await
                .as_ref()
                .map(|item| item.sequence),
            sequences.last().copied()
        );
    }

    #[tokio::test]
    async fn expired_tracks_are_removed_from_all_retention_surfaces() {
        let fixture = include_str!("../tests/fixtures/track_synthetic.json");
        let mut frame: TrackEnvelope = serde_json::from_str(fixture).unwrap();
        let now = now_unix_ms();
        frame.captured_at_unix_ms = now;
        frame.expires_at_unix_ms = now + 5;
        let hub = NlosHub::new(&"x".repeat(32), frame.session_id.clone()).unwrap();
        hub.publish(frame).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        // Public read surfaces call the same lazy purge before returning data;
        // no unbounded per-frame timer queue is retained by the hub.
        hub.purge_expired(now_unix_ms()).await;
        assert!(hub.inner.latest.read().await.is_none());
        assert!(hub.inner.history.read().await.is_empty());
    }
}
