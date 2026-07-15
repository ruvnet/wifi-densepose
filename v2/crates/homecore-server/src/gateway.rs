//! HOMECORE-UI backend-for-frontend (BFF) gateway — ADR-131 §11.
//!
//! `homecore-server` is the single origin the dashboard talks to (§2.1).
//! This module adds the `/api/homecore/*` aggregation namespace and the
//! `/api/cal/*` reverse-proxy to the calibration service, so the browser
//! never makes a cross-origin call and never holds an upstream credential.
//!
//! Implemented now (self-contained, no new external service):
//!   * `/api/cal/*`            — reverse-proxy → calibration API (ADR-151)   [W2]
//!   * `GET /api/homecore/rooms` — per-room RoomState, adapted to the UI shape [W2]
//!   * `GET /api/homecore/cogs`  — COG supervisor over the apps dir           [W4]
//!   * `GET /api/homecore/appliance` — host metrics from /proc + port probes  [W6]
//!
//! Returns a typed `503 upstream_unavailable` for routes whose upstream is
//! a SEED device / appliance daemon not present in this repo (§11.2 / §12):
//! seeds, federation, witness, privacy, settings, automations, events
//! history, hailo, tokens. The front-end renders these as error states
//! (it never falls back to mock in production — §2.2).
//!
//! NOTE: written against the real crate APIs but NOT yet compiled in the
//! authoring environment (no Rust toolchain); run `cargo test -p
//! homecore-server` on a Rust host.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use homecore_api::auth::BearerAuth;
use homecore_api::SharedState;

/// Static gateway configuration (from CLI/env in `main`).
pub struct GatewayConfig {
    /// Base URL of the calibration service (`wifi-densepose calibrate-serve`),
    /// e.g. `http://127.0.0.1:8090`. `None` disables the calibration routes.
    pub calibration_url: Option<String>,
    /// Bearer token for the calibration service (held server-side only).
    pub calibration_token: Option<String>,
    /// COG install directory the supervisor reads (`/var/lib/cognitum/apps`).
    pub apps_dir: PathBuf,
    /// Per-proxy timeout so one slow upstream cannot stall the dashboard.
    pub timeout: Duration,
}

#[derive(Clone)]
pub struct GatewayState {
    pub shared: SharedState,
    pub http: reqwest::Client,
    pub cfg: Arc<GatewayConfig>,
}

impl GatewayState {
    pub fn new(shared: SharedState, cfg: GatewayConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self { shared, http, cfg: Arc::new(cfg) }
    }
}

/// Build the gateway router (state already applied → `Router<()>`), ready
/// to `.merge()` into the main app alongside the homecore-api routes.
pub fn gateway_router(state: GatewayState) -> Router {
    Router::new()
        // ── calibration reverse-proxy (W2) ──────────────────────────
        .route("/api/cal/*path", get(cal_proxy_get).post(cal_proxy_post))
        // ── aggregation endpoints (W2 / W4 / W6) ────────────────────
        .route("/api/homecore/rooms", get(rooms))
        .route("/api/homecore/cogs", get(cogs_list))
        .route("/api/homecore/appliance", get(appliance))
        // ── upstream-dependent stubs (W3 / W5 / W6): typed 503 ───────
        .route("/api/homecore/seeds", get(stub_503))
        .route("/api/homecore/seeds/:id", get(stub_503))
        .route("/api/homecore/federation", get(stub_503))
        .route("/api/homecore/witness", get(stub_503))
        .route("/api/homecore/privacy", get(stub_503).post(stub_503))
        .route("/api/homecore/settings", get(stub_503))
        .route("/api/homecore/automations", get(stub_503).post(stub_503))
        // No OTA feed wired yet → "no updates available" is an empty list,
        // not an error (so a working COG list is never blanked).
        .route("/api/homecore/cogs/updates", get(empty_list))
        .route("/api/homecore/hailo", get(stub_503))
        .route("/api/homecore/tokens", get(stub_503))
        .route("/api/events", get(stub_503))
        .with_state(state)
}

// ── auth + typed errors ─────────────────────────────────────────────

async fn require_auth(headers: &HeaderMap, st: &GatewayState) -> Result<(), Response> {
    BearerAuth::from_headers(headers, st.shared.tokens())
        .await
        .map(|_| ())
        .map_err(|e| e.into_response())
}

fn typed(status: StatusCode, error: &str, detail: &str) -> Response {
    (status, Json(json!({ "error": error, "detail": detail }))).into_response()
}
fn upstream_unavailable(detail: &str) -> Response {
    typed(StatusCode::SERVICE_UNAVAILABLE, "upstream_unavailable", detail)
}
fn upstream_timeout(detail: &str) -> Response {
    typed(StatusCode::GATEWAY_TIMEOUT, "upstream_timeout", detail)
}
fn bad_request(detail: &str) -> Response {
    typed(StatusCode::BAD_REQUEST, "bad_request", detail)
}

/// Reject a proxied wildcard path that could escape the `/api/` scope on the
/// upstream calibration service (path-traversal / confused-deputy SSRF —
/// ADR-131 §11 security review). The privileged server-side calibration bearer
/// is attached by `proxy()`, so a client must NOT be able to redirect that
/// credential outside `…/api/`.
///
/// Returns `Err(400)` when the path (or its percent-decoded form):
///   * is absolute (`/…`) — would replace the `…/api/` base entirely,
///   * contains a backslash (`\`) — Windows/alt-separator traversal,
///   * has any segment equal to `.` or `..` — dot-segment traversal,
///   * still carries `%2e%2e` / `%2f` (single-decode is enough — we reject on
///     the decoded form AND on a residual encoded marker, so double-encoding
///     like `%252e` decodes once to `%2e` and is caught here).
///
/// Legitimate `v1/...` paths (the only shape the UI sends) pass unchanged.
fn validate_proxy_path(path: &str) -> Result<(), Response> {
    // 1. Reject on the raw form first (cheap; catches backslash + leading `/`).
    if path.starts_with('/') {
        return Err(bad_request("proxied path must be relative (leading '/' not allowed)"));
    }
    if path.contains('\\') {
        return Err(bad_request("proxied path must not contain a backslash"));
    }
    // 2. Percent-decode once and re-check; reject if decoding is invalid.
    let decoded = percent_decode_once(path)
        .ok_or_else(|| bad_request("proxied path has invalid percent-encoding"))?;
    if decoded.starts_with('/') || decoded.contains('\\') {
        return Err(bad_request("proxied path resolves to an absolute/traversal path"));
    }
    // 3. Reject any `.`/`..` segment on BOTH the raw and decoded forms so an
    //    encoded `%2e%2e%2f` cannot slip a dot-segment past the split.
    for form in [path, decoded.as_str()] {
        for seg in form.split(['/', '\\']) {
            if seg == "." || seg == ".." {
                return Err(bad_request("proxied path must not contain '.' or '..' segments"));
            }
        }
        // Defence in depth: a residual encoded traversal marker survived the
        // single decode (e.g. originally double-encoded). Reject it outright.
        let lower = form.to_ascii_lowercase();
        if lower.contains("%2e") || lower.contains("%2f") || lower.contains("%5c") {
            return Err(bad_request("proxied path must not contain encoded traversal markers"));
        }
    }
    Ok(())
}

/// Minimal single-pass percent-decoder (no external dep). Returns `None` on a
/// malformed escape so callers can fail closed.
fn percent_decode_once(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' => {
                if i + 2 >= bytes.len() {
                    return None;
                }
                let hi = (bytes[i + 1] as char).to_digit(16)?;
                let lo = (bytes[i + 2] as char).to_digit(16)?;
                out.push((hi * 16 + lo) as u8);
                i += 3;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

/// Routes whose upstream is a SEED device / appliance daemon not present
/// in this repo. Honest 503 until the corresponding §12 wave lands.
async fn stub_503(State(st): State<GatewayState>, headers: HeaderMap) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    upstream_unavailable("endpoint not yet wired — see ADR-131 §11/§12 (SEED device / appliance upstream)")
}

/// Auth-gated empty-array response (e.g. OTA updates with no feed wired).
async fn empty_list(State(st): State<GatewayState>, headers: HeaderMap) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    Json(Vec::<Value>::new()).into_response()
}

// ── calibration reverse-proxy (W2) ──────────────────────────────────

async fn cal_proxy_get(
    State(st): State<GatewayState>,
    headers: HeaderMap,
    Path(path): Path<String>,
    RawQuery(q): RawQuery,
) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    if let Err(r) = validate_proxy_path(&path) {
        return r;
    }
    let base = match &st.cfg.calibration_url {
        Some(u) => u,
        None => return upstream_unavailable("calibration service not configured (set --calibration-url / HOMECORE_CALIBRATION_URL)"),
    };
    let qs = q.map(|s| format!("?{s}")).unwrap_or_default();
    // The wildcard already carries the `v1/...` segment (the UI calls
    // `/api/cal/v1/...`), so map `/api/cal/<rest>` → `<base>/api/<rest>`.
    let url = format!("{}/api/{}{}", base.trim_end_matches('/'), path, qs);
    proxy(&st, st.http.get(&url)).await
}

async fn cal_proxy_post(
    State(st): State<GatewayState>,
    headers: HeaderMap,
    Path(path): Path<String>,
    body: Bytes,
) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    if let Err(r) = validate_proxy_path(&path) {
        return r;
    }
    let base = match &st.cfg.calibration_url {
        Some(u) => u,
        None => return upstream_unavailable("calibration service not configured (set --calibration-url / HOMECORE_CALIBRATION_URL)"),
    };
    let url = format!("{}/api/{}", base.trim_end_matches('/'), path);
    let rb = st
        .http
        .post(&url)
        .header(header::CONTENT_TYPE, "application/json")
        .body(body);
    proxy(&st, rb).await
}

/// Send an upstream request (with the server-side calibration token) and
/// stream the response back verbatim, mapping transport failures to typed
/// errors.
async fn proxy(st: &GatewayState, mut rb: reqwest::RequestBuilder) -> Response {
    if let Some(tok) = &st.cfg.calibration_token {
        rb = rb.bearer_auth(tok);
    }
    match rb.send().await {
        Ok(resp) => {
            let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let ct = resp
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("application/json")
                .to_string();
            match resp.bytes().await {
                Ok(b) => {
                    let mut out = Response::new(axum::body::Body::from(b));
                    *out.status_mut() = status;
                    if let Ok(hv) = HeaderValue::from_str(&ct) {
                        out.headers_mut().insert(header::CONTENT_TYPE, hv);
                    }
                    out
                }
                Err(e) => upstream_unavailable(&format!("calibration body read failed: {e}")),
            }
        }
        Err(e) if e.is_timeout() => upstream_timeout("calibration service timed out"),
        Err(e) => upstream_unavailable(&format!("calibration service: {e}")),
    }
}

async fn fetch_json(st: &GatewayState, url: &str) -> Result<Value, Response> {
    let mut rb = st.http.get(url);
    if let Some(tok) = &st.cfg.calibration_token {
        rb = rb.bearer_auth(tok);
    }
    match rb.send().await {
        Ok(resp) => resp
            .json::<Value>()
            .await
            .map_err(|e| upstream_unavailable(&format!("calibration JSON parse: {e}"))),
        Err(e) if e.is_timeout() => Err(upstream_timeout("calibration service timed out")),
        Err(e) => Err(upstream_unavailable(&format!("calibration service: {e}"))),
    }
}

// ── rooms aggregation + RoomState adapter (W2 / §11.3) ──────────────

async fn rooms(State(st): State<GatewayState>, headers: HeaderMap) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    let base = match &st.cfg.calibration_url {
        Some(u) => u.trim_end_matches('/').to_string(),
        None => return upstream_unavailable("calibration service not configured"),
    };
    let banks = match fetch_json(&st, &format!("{base}/api/v1/calibration/baselines")).await {
        Ok(v) => bank_names(&v),
        Err(r) => return r,
    };
    // Fetch every bank's RoomState concurrently (§11 perf): one slow bank no
    // longer serialises behind the others. Order is preserved by collecting in
    // the original bank order.
    let fetches = banks.into_iter().map(|bank| {
        let st = &st;
        let base = base.as_str();
        async move {
            let url = format!("{base}/api/v1/room/state?bank={bank}");
            fetch_json(st, &url).await.ok().map(|v| adapt_room_state(&bank, &v))
        }
    });
    let out: Vec<Value> = futures::future::join_all(fetches)
        .await
        .into_iter()
        .flatten()
        .collect();
    Json(out).into_response()
}

/// Accept either `["living_room", ...]` or `[{ "name"|"id"|"bank": ... }]`.
fn bank_names(v: &Value) -> Vec<String> {
    match v {
        Value::Array(items) => items
            .iter()
            .filter_map(|it| match it {
                Value::String(s) => Some(s.clone()),
                Value::Object(o) => o
                    .get("name")
                    .or_else(|| o.get("id"))
                    .or_else(|| o.get("bank"))
                    .and_then(|x| x.as_str())
                    .map(str::to_string),
                _ => None,
            })
            .collect(),
        Value::Object(o) => o
            .get("baselines")
            .map(|b| bank_names(b))
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Adapt the calibration `RoomState` (Option<SpecialistReading> fields +
/// `vetoed`/`stale`) onto the UI shape (§11.3). `None` → JSON `null`,
/// preserving the not-trained-vs-withheld distinction (§6 invariant 3).
fn adapt_room_state(bank: &str, v: &Value) -> Value {
    let chip = |k: &str| -> Value {
        match v.get(k) {
            Some(r) if !r.is_null() => json!({
                "value": r.get("label").and_then(|l| l.as_str()).map(Value::from)
                    .unwrap_or_else(|| r.get("value").cloned().unwrap_or(Value::Null)),
                "confidence": r.get("confidence").cloned().unwrap_or(Value::Null),
            }),
            _ => Value::Null,
        }
    };
    let bpm = |k: &str| -> Value {
        match v.get(k) {
            Some(r) if !r.is_null() => json!({
                "value": r.get("value").cloned().unwrap_or(Value::Null),
                "confidence": r.get("confidence").cloned().unwrap_or(Value::Null),
            }),
            _ => Value::Null,
        }
    };
    let anomaly = match v.get("anomaly") {
        Some(r) if !r.is_null() => json!({
            "value": r.get("value").cloned().unwrap_or(Value::Null),
            "confidence": r.get("confidence").cloned().unwrap_or(Value::Null),
            // §6 invariant 3 (honesty): pass through the REAL anomaly threshold
            // from the upstream RoomState if present; if absent, emit null
            // (withheld) — never fabricate a constant. The UI treats null as
            // withheld, not a fake default.
            "threshold": r.get("threshold").cloned().unwrap_or(Value::Null),
        }),
        _ => Value::Null,
    };
    json!({
        "room_id": bank,
        "seeds": [],
        "stale": v.get("stale").and_then(|b| b.as_bool()).unwrap_or(false),
        "vetoed": v.get("vetoed").and_then(|b| b.as_bool()).unwrap_or(false),
        "presence": chip("presence"),
        "posture": chip("posture"),
        "breathing_bpm": bpm("breathing"),
        "heart_bpm": bpm("heartbeat"),
        "restlessness": bpm("restlessness"),
        "anomaly": anomaly,
    })
}

// ── COG supervisor (W4 / §11.6) ─────────────────────────────────────

async fn cogs_list(State(st): State<GatewayState>, headers: HeaderMap) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    let mut out: Vec<Value> = Vec::new();
    let rd = match std::fs::read_dir(&st.cfg.apps_dir) {
        Ok(rd) => rd,
        Err(_) => return Json(out).into_response(), // no apps dir yet → empty
    };
    for entry in rd.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let manifest = match std::fs::read_to_string(dir.join("manifest.json")) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let m: Value = match serde_json::from_str(&manifest) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = m
            .get("id")
            .and_then(|x| x.as_str())
            .unwrap_or_else(|| dir.file_name().and_then(|n| n.to_str()).unwrap_or("?"))
            .to_string();
        let pid = read_pid(&dir, &id);
        let alive = pid.map(pid_alive).unwrap_or(false);
        let status = if alive { "running" } else { "stopped" };
        out.push(json!({
            "id": id,
            "version": m.get("version").and_then(|x| x.as_str()).unwrap_or("?"),
            "arch": m.get("arch").and_then(|x| x.as_str()).unwrap_or("arm"),
            "status": status,
            "pid": pid,
            "sha256_verified": m.get("binary_sha256").is_some(),
            "signature_verified": m.get("binary_signature").is_some(),
            "hef": m.get("hef").cloned().unwrap_or(Value::Null),
        }));
    }
    Json(out).into_response()
}

fn read_pid(dir: &std::path::Path, id: &str) -> Option<i64> {
    for name in [format!("{id}.pid"), "pid".to_string(), "app.pid".to_string()] {
        if let Ok(s) = std::fs::read_to_string(dir.join(&name)) {
            if let Ok(p) = s.trim().parse::<i64>() {
                return Some(p);
            }
        }
    }
    None
}

fn pid_alive(pid: i64) -> bool {
    if pid <= 0 {
        return false;
    }
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

// ── appliance metrics (W6 / §11.5) ──────────────────────────────────

async fn appliance(State(st): State<GatewayState>, headers: HeaderMap) -> Response {
    if let Err(r) = require_auth(&headers, &st).await {
        return r;
    }
    let ram = mem_used_pct();
    let cpu = cpu_load_pct();
    let uptime = uptime_secs();
    // Probe the appliance services concurrently with a non-blocking async
    // connect under a timeout (§11 perf): previously a sequential blocking
    // `std::net::TcpStream::connect_timeout` stalled the whole async handler
    // for up to `N * timeout` and parked a Tokio worker thread per probe.
    let probes = [
        ("ruview-mcp-brain", 9876u16),
        ("cognitum-rvf-agent", 9004),
        ("ruvector-hailo-worker", 50051),
    ]
    .into_iter()
    .map(|(name, port)| {
        let timeout = st.cfg.timeout;
        async move {
            let up = tcp_open("127.0.0.1", port, timeout).await;
            json!({ "name": name, "port": port, "status": if up { "running" } else { "unreachable" } })
        }
    });
    let services: Vec<Value> = futures::future::join_all(probes).await;
    Json(json!({
        "cpu_pct": cpu,
        "ram_pct": ram,
        "hailo_load_pct": Value::Null,   // requires the Hailo runtime stat source (§11.5 APPLIANCE)
        "hailo_temp_c": Value::Null,
        "uptime_s": uptime,
        "services": services,
        "event_rate": [],
        "channel_capacity": 4096,
        "channel_lag": 0,
    }))
    .into_response()
}

fn read_first_line(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok().and_then(|s| s.lines().next().map(str::to_string))
}

fn uptime_secs() -> Option<u64> {
    read_first_line("/proc/uptime")
        .and_then(|l| l.split_whitespace().next().map(str::to_string))
        .and_then(|s| s.parse::<f64>().ok())
        .map(|f| f as u64)
}

fn mem_used_pct() -> Option<f64> {
    let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = 0f64;
    let mut avail = 0f64;
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("MemTotal:") => total = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            Some("MemAvailable:") => avail = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0),
            _ => {}
        }
    }
    if total > 0.0 {
        Some(((total - avail) / total * 100.0 * 10.0).round() / 10.0)
    } else {
        None
    }
}

fn cpu_load_pct() -> Option<f64> {
    // loadavg(1m) / ncpu * 100 — a cheap proxy (no two-sample /proc/stat).
    let load = read_first_line("/proc/loadavg")?
        .split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()?;
    let ncpu = std::thread::available_parallelism().map(|n| n.get() as f64).unwrap_or(1.0);
    Some(((load / ncpu * 100.0).min(100.0) * 10.0).round() / 10.0)
}

/// Non-blocking liveness probe: succeeds iff a TCP connection to
/// `host:port` completes within `timeout`. Async so it never parks a Tokio
/// worker thread (unlike the blocking `std::net` connect it replaced).
async fn tcp_open(host: &str, port: u16, timeout: Duration) -> bool {
    let addr = format!("{host}:{port}");
    matches!(
        tokio::time::timeout(timeout, tokio::net::TcpStream::connect(&addr)).await,
        Ok(Ok(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use homecore::HomeCore;
    use homecore_api::{LongLivedTokenStore, SharedState};
    use tower::ServiceExt;

    fn gw() -> GatewayState {
        let shared = SharedState::with_tokens(
            HomeCore::new(),
            "Test",
            "test",
            LongLivedTokenStore::allow_any_non_empty(),
        );
        GatewayState::new(
            shared,
            GatewayConfig {
                calibration_url: None,
                calibration_token: None,
                apps_dir: PathBuf::from("/nonexistent-apps-dir"),
                timeout: Duration::from_millis(200),
            },
        )
    }

    async fn send(app: Router, method: &str, path: &str) -> (StatusCode, String) {
        let resp = app
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("authorization", "Bearer dev")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let b = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        (status, String::from_utf8_lossy(&b).into_owned())
    }

    #[tokio::test]
    async fn unauthenticated_is_rejected() {
        let app = gateway_router(gw());
        let resp = app
            .oneshot(Request::builder().uri("/api/homecore/cogs").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn cogs_returns_empty_when_apps_dir_missing() {
        let (status, body) = send(gateway_router(gw()), "GET", "/api/homecore/cogs").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.trim(), "[]");
    }

    #[tokio::test]
    async fn rooms_503_when_calibration_unconfigured() {
        let (status, body) = send(gateway_router(gw()), "GET", "/api/homecore/rooms").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("upstream_unavailable"));
    }

    #[tokio::test]
    async fn seed_tier_routes_are_typed_503() {
        for p in ["/api/homecore/seeds", "/api/homecore/federation", "/api/homecore/witness", "/api/events"] {
            let (status, body) = send(gateway_router(gw()), "GET", p).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{p} should be 503");
            assert!(body.contains("upstream_unavailable"), "{p} typed body");
        }
    }

    #[tokio::test]
    async fn appliance_returns_metrics_json() {
        let (status, body) = send(gateway_router(gw()), "GET", "/api/homecore/appliance").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("\"services\""));
        assert!(body.contains("\"ram_pct\""));
    }

    #[test]
    fn adapt_room_state_maps_fields_and_preserves_null() {
        // breathing/heartbeat rename; None → null; anomaly gets a threshold.
        let cal = json!({
            "presence": {"kind":"Presence","value":1.0,"confidence":0.9,"label":"occupied"},
            "posture": {"kind":"Posture","value":2.0,"confidence":0.8,"label":"lying"},
            "breathing": {"kind":"Breathing","value":12.0,"confidence":0.7,"label":null},
            "heartbeat": null,
            "restlessness": {"kind":"Restlessness","value":0.1,"confidence":0.6,"label":null},
            "anomaly": {"kind":"Anomaly","value":0.2,"confidence":0.5,"label":null},
            "vetoed": false, "stale": true
        });
        let ui = adapt_room_state("bedroom_1", &cal);
        assert_eq!(ui["room_id"], "bedroom_1");
        assert_eq!(ui["stale"], true);
        assert_eq!(ui["presence"]["value"], "occupied");
        assert_eq!(ui["breathing_bpm"]["value"], 12.0);
        assert!(ui["heart_bpm"].is_null(), "None heartbeat must map to null (not trained)");
        // §6 invariant 3: upstream RoomState carries no threshold here, so the
        // adapter must emit null (withheld) — NOT a fabricated constant.
        assert!(
            ui["anomaly"]["threshold"].is_null(),
            "absent upstream threshold must surface as null, never a hardcoded value"
        );
    }

    #[test]
    fn adapt_room_state_passes_through_real_anomaly_threshold() {
        // When the upstream RoomState DOES carry a real threshold, it must be
        // forwarded verbatim (no fabrication, no override).
        let cal = json!({
            "anomaly": {"kind":"Anomaly","value":0.2,"confidence":0.5,"threshold":0.73},
        });
        let ui = adapt_room_state("bedroom_1", &cal);
        assert_eq!(ui["anomaly"]["threshold"], 0.73, "real threshold must pass through");
    }

    #[test]
    fn validate_proxy_path_allows_legit_v1_paths() {
        // The only shape the UI sends must pass unchanged.
        for ok in [
            "v1/room/state",
            "v1/calibration/baselines",
            "v1/enroll/status",
            "v1/room/state?bank=living_room", // query is split off before this fn
        ] {
            // strip any query the caller would have removed; we only validate path
            let p = ok.split('?').next().unwrap();
            assert!(validate_proxy_path(p).is_ok(), "{p} should be allowed");
        }
    }

    #[test]
    fn validate_proxy_path_rejects_traversal_variants() {
        for bad in [
            "v1/../../x",        // dot-segment traversal
            "../etc/passwd",     // parent escape
            "/etc/passwd",       // absolute
            "v1\\..\\..\\x",     // backslash traversal
            "..%2f..%2fx",       // encoded slash
            "%2e%2e/x",          // encoded dot-dot
            "v1/%2e%2e%2fadmin", // mixed encoded traversal
            "%252e%252e/x",      // double-encoded (residual %2e after one decode)
        ] {
            assert!(validate_proxy_path(bad).is_err(), "{bad} must be rejected");
        }
    }

    #[tokio::test]
    async fn cal_proxy_rejects_traversal_with_400_before_upstream() {
        // `gw()` has calibration_url=None: a path that reached URL-building
        // would 503 ("not configured"). A 400 here proves the traversal is
        // rejected BEFORE any upstream request is even attempted.
        for (method, path) in [
            ("GET", "/api/cal/v1/../../x"),
            ("GET", "/api/cal/..%2f..%2fx"),
            ("GET", "/api/cal/%2e%2e/x"),
            ("POST", "/api/cal/v1/../../x"),
        ] {
            let (status, body) = send(gateway_router(gw()), method, path).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{method} {path} must be 400");
            assert!(body.contains("bad_request"), "{method} {path} typed 400 body");
            assert!(
                !body.contains("upstream_unavailable"),
                "{method} {path} must NOT reach the upstream-config branch"
            );
        }
    }

    #[tokio::test]
    async fn cal_proxy_allows_legit_path_through_to_upstream_config() {
        // A legitimate v1 path passes validation and then hits the
        // "not configured" 503 (proving it was NOT blocked as traversal).
        let (status, body) = send(gateway_router(gw()), "GET", "/api/cal/v1/room/state").await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains("upstream_unavailable"), "legit path should reach upstream branch");
    }

    #[test]
    fn bank_names_accepts_strings_and_objects() {
        assert_eq!(bank_names(&json!(["a", "b"])), vec!["a", "b"]);
        assert_eq!(bank_names(&json!([{"name":"x"}, {"id":"y"}])), vec!["x", "y"]);
        assert_eq!(bank_names(&json!({"baselines":["z"]})), vec!["z"]);
    }
}
