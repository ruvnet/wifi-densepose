//! Opt-in bearer-token auth for the sensing-server HTTP API (#443).
//!
//! When the `RUVIEW_API_TOKEN` environment variable is set, every request
//! whose path begins with `/api/v1/` must carry a matching
//! `Authorization: Bearer <token>` header, otherwise the server responds with
//! `401 Unauthorized`. When the env var is unset (or empty), the middleware is
//! a no-op and the API stays unauthenticated — preserving the long-standing
//! LAN-only deployment posture documented in the issue. This is a binary,
//! deployment-time switch with **no default authentication change**.
//!
//! Endpoints outside `/api/v1/*` (`/health*`, `/ws/sensing`, the static `/ui/*`
//! mount, `/`) are intentionally **not** gated:
//! * `/health*` is the liveness/readiness probe that orchestrators hit
//!   anonymously;
//! * `/ws/sensing` and `/ui/*` are served to local browsers that can't easily
//!   inject headers — the sensitive control plane is the `/api/v1/*` tree, and
//!   that is what this layer protects.
//!
//! The header check uses a length-then-byte constant-time compare to avoid
//! leaking the token through timing.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{header::AUTHORIZATION, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Environment variable that gates the middleware. Unset / empty ⇒ auth off.
pub const API_TOKEN_ENV: &str = "RUVIEW_API_TOKEN";

/// Path prefix the middleware protects when auth is enabled.
pub const PROTECTED_PREFIX: &str = "/api/v1/";

/// Cheap, cloneable handle to the configured token (or `None`).
#[derive(Debug, Clone, Default)]
pub struct AuthState {
    /// The expected bearer token, if any. `None` ⇒ middleware is a no-op.
    token: Option<Arc<String>>,
}

impl AuthState {
    /// Build an [`AuthState`] from an explicit string. Empty ⇒ disabled.
    pub fn from_token(t: impl Into<String>) -> Self {
        let s = t.into();
        if s.is_empty() {
            AuthState { token: None }
        } else {
            AuthState {
                token: Some(Arc::new(s)),
            }
        }
    }

    /// Read [`API_TOKEN_ENV`] from the process environment. Returns
    /// `AuthState { token: None }` when the variable is unset or empty.
    pub fn from_env() -> Self {
        match std::env::var(API_TOKEN_ENV) {
            Ok(s) if !s.is_empty() => AuthState::from_token(s),
            _ => AuthState::default(),
        }
    }

    /// Whether the middleware will enforce auth on `/api/v1/*` requests.
    pub fn is_enabled(&self) -> bool {
        self.token.is_some()
    }
}

/// Constant-time byte slice equality. Returns `false` immediately on length
/// mismatch (lengths are not secret here — both sides are fixed tokens).
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Axum middleware: enforces `Authorization: Bearer <token>` on `/api/v1/*`
/// requests when [`AuthState::is_enabled`] returns `true`. Wires up via
/// [`axum::middleware::from_fn_with_state`].
pub async fn require_bearer(
    State(auth): State<AuthState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = auth.token.clone() else {
        return next.run(request).await;
    };
    if !request.uri().path().starts_with(PROTECTED_PREFIX) {
        return next.run(request).await;
    }
    let supplied = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        // RFC 6750 §2.1 / RFC 7235 §2.1: the auth-scheme ("Bearer") is
        // case-insensitive. Match it as such (and tolerate extra leading
        // whitespace before the token) so a correct token isn't rejected
        // just because a client sent `bearer`/`BEARER`. The token compare
        // below stays exact + constant-time.
        .and_then(|s| {
            let (scheme, token) = s.split_once(' ')?;
            scheme
                .eq_ignore_ascii_case("Bearer")
                .then(|| token.trim_start())
        });
    let ok = supplied
        .map(|s| ct_eq(s.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);
    if ok {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            "missing or invalid bearer token (set Authorization: Bearer <RUVIEW_API_TOKEN>)\n",
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    fn ok_handler() -> Router {
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/api/v1/info", get(|| async { "ok" }))
            .route("/api/v1/sensitive", axum::routing::post(|| async { "ok" }))
            .route("/ui/index.html", get(|| async { "<html/>" }))
    }

    fn wrap(auth: AuthState) -> Router {
        ok_handler().layer(axum::middleware::from_fn_with_state(auth, require_bearer))
    }

    async fn status(router: Router, method: &str, path: &str, auth: Option<&str>) -> StatusCode {
        let mut req = Request::builder()
            .method(method)
            .uri(path)
            .body(Body::empty())
            .unwrap();
        if let Some(t) = auth {
            req.headers_mut()
                .insert(AUTHORIZATION, format!("Bearer {t}").parse().unwrap());
        }
        router.oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn middleware_is_no_op_when_token_unset() {
        let r = wrap(AuthState::default());
        assert_eq!(
            status(r.clone(), "GET", "/api/v1/info", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(r.clone(), "POST", "/api/v1/sensitive", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(r.clone(), "GET", "/health", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(r, "GET", "/ui/index.html", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn enabled_blocks_api_without_bearer() {
        let r = wrap(AuthState::from_token("s3cr3t"));
        assert_eq!(
            status(r.clone(), "GET", "/api/v1/info", None).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(r, "POST", "/api/v1/sensitive", None).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn accepts_case_insensitive_bearer_scheme() {
        // RFC 6750 §2.1 / RFC 7235 §2.1: the auth-scheme is case-insensitive.
        // A correct token must authenticate regardless of scheme casing or
        // extra whitespace; a wrong token must still be rejected.
        async fn req_status(auth_value: &str) -> StatusCode {
            let r = wrap(AuthState::from_token("s3cr3t"));
            let mut req = Request::builder()
                .method("GET")
                .uri("/api/v1/info")
                .body(Body::empty())
                .unwrap();
            req.headers_mut()
                .insert(AUTHORIZATION, auth_value.parse().unwrap());
            r.oneshot(req).await.unwrap().status()
        }
        assert_eq!(req_status("Bearer s3cr3t").await, StatusCode::OK);
        assert_eq!(req_status("bearer s3cr3t").await, StatusCode::OK);
        assert_eq!(req_status("BEARER s3cr3t").await, StatusCode::OK);
        assert_eq!(req_status("Bearer  s3cr3t").await, StatusCode::OK); // extra space
        // Scheme leniency must NOT weaken the token check.
        assert_eq!(req_status("bearer nope").await, StatusCode::UNAUTHORIZED);
        assert_eq!(req_status("Basic s3cr3t").await, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn enabled_blocks_api_with_wrong_bearer() {
        let r = wrap(AuthState::from_token("s3cr3t"));
        assert_eq!(
            status(r.clone(), "GET", "/api/v1/info", Some("nope")).await,
            StatusCode::UNAUTHORIZED
        );
        // Wrong scheme (Basic / token) — only "Bearer <token>" is accepted.
        let mut req = Request::builder()
            .method("GET")
            .uri("/api/v1/info")
            .body(Body::empty())
            .unwrap();
        req.headers_mut()
            .insert(AUTHORIZATION, "Basic s3cr3t".parse().unwrap());
        assert_eq!(
            r.oneshot(req).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn enabled_allows_api_with_correct_bearer() {
        let r = wrap(AuthState::from_token("s3cr3t"));
        assert_eq!(
            status(r.clone(), "GET", "/api/v1/info", Some("s3cr3t")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(r, "POST", "/api/v1/sensitive", Some("s3cr3t")).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn enabled_never_gates_paths_outside_api_v1() {
        let r = wrap(AuthState::from_token("s3cr3t"));
        // Even with auth ON, `/health` and `/ui/*` are reachable without a token:
        // orchestrator probes and the local UI need to load unchallenged.
        assert_eq!(
            status(r.clone(), "GET", "/health", None).await,
            StatusCode::OK
        );
        assert_eq!(
            status(r, "GET", "/ui/index.html", None).await,
            StatusCode::OK
        );
    }

    #[test]
    fn ct_eq_basics() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"ab")); // length mismatch
        assert!(!ct_eq(b"", b"x"));
        assert!(ct_eq(b"", b""));
    }

    #[test]
    fn from_env_treats_empty_as_disabled() {
        // Avoid touching the real env in a thread-shared test — exercise the
        // string ctor directly with the same trim logic.
        assert!(!AuthState::from_token("").is_enabled());
        assert!(AuthState::from_token("x").is_enabled());
    }

    #[test]
    fn protected_prefix_and_env_constants_are_stable() {
        // These are documented in the issue body and the README; keep them locked.
        assert_eq!(API_TOKEN_ENV, "RUVIEW_API_TOKEN");
        assert_eq!(PROTECTED_PREFIX, "/api/v1/");
    }
}
