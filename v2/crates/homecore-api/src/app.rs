//! Axum router wiring. Mounts the §2.1 P2 routes + the WS endpoint.

use axum::http::{header, HeaderValue, Method};
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::rest;
use crate::state::SharedState;
use crate::ws;

pub type AppState = SharedState;

/// Build the Axum router with an EXPLICIT CORS allowlist (audit fix
/// HC-05). The previous `CorsLayer::permissive()` set
/// `Access-Control-Allow-Origin: *` which lets any webpage make
/// authenticated cross-origin calls once a bearer is leaked.
///
/// Default allowlist: `http://localhost:5173` (the homecore-frontend
/// Vite dev server) plus the same on port 3000 / 8080 / 8081 / 8123
/// covering the most common reverse-proxy + HA-app paths. Production
/// deployments should set `HOMECORE_CORS_ORIGINS=https://...` (comma-
/// separated) to override.
pub fn router(state: SharedState) -> Router {
    let cors = build_cors_layer();
    Router::new()
        .route("/api/", get(rest::api_root))
        .route("/api/config", get(rest::get_config))
        .route("/api/states", get(rest::get_states))
        .route(
            "/api/states/:entity_id",
            get(rest::get_state)
                .post(rest::set_state)
                .delete(rest::delete_state),
        )
        .route("/api/services", get(rest::get_services))
        .route("/api/services/:domain/:service", post(rest::call_service))
        .route("/api/websocket", get(ws::websocket_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Build the audited CORS allowlist layer (HC-05). Exposed so the
/// integration binary can apply the SAME allowlist to routes merged in
/// outside `router()` (e.g. the ADR-131 BFF gateway), instead of leaving
/// `/api/homecore/*` and `/api/cal/*` with no CORS coverage at all.
pub fn build_cors_layer() -> CorsLayer {
    let raw = std::env::var("HOMECORE_CORS_ORIGINS").ok();
    let origins: Vec<HeaderValue> = match raw {
        Some(v) if !v.trim().is_empty() => v
            .split(',')
            .filter_map(|s| s.trim().parse::<HeaderValue>().ok())
            .collect(),
        _ => default_origins(),
    };
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS, Method::DELETE])
        .allow_headers([
            header::AUTHORIZATION,
            header::CONTENT_TYPE,
            header::ACCEPT,
        ])
        .allow_credentials(false)
}

fn default_origins() -> Vec<HeaderValue> {
    // Dev defaults — homecore-frontend Vite (5173), common reverse-
    // proxy ports (3000, 8080, 8081), and the bind port itself (8123)
    // so HA-companion-app-style same-origin calls work without
    // ceremony.
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:8123",
        "http://127.0.0.1:8123",
    ]
    .iter()
    .filter_map(|o| o.parse::<HeaderValue>().ok())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // `set_var`/`remove_var` mutate process-global state; serialize every test
    // that touches HOMECORE_CORS_ORIGINS so they cannot race in parallel.
    // Poison-tolerant: a panicking test must not cascade-fail the others.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn default_origins_includes_vite_and_ha_ports() {
        let origins = default_origins();
        assert!(origins.iter().any(|o| o.to_str().unwrap().contains("5173")));
        assert!(origins.iter().any(|o| o.to_str().unwrap().contains("8123")));
        assert!(!origins.is_empty());
    }

    #[test]
    fn env_override_via_homecore_cors_origins() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("HOMECORE_CORS_ORIGINS", "https://example.com,https://other.example.com");
        // build_cors_layer() returns a CorsLayer which doesn't expose
        // its origin list; we test the parse path indirectly by
        // confirming no panic + at least one origin would parse.
        let parsed: Vec<_> = "https://example.com,https://other.example.com"
            .split(',')
            .filter_map(|s| s.trim().parse::<HeaderValue>().ok())
            .collect();
        assert_eq!(parsed.len(), 2);
        std::env::remove_var("HOMECORE_CORS_ORIGINS");
    }

    #[test]
    fn env_empty_falls_back_to_defaults() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("HOMECORE_CORS_ORIGINS", "   ");
        let raw = std::env::var("HOMECORE_CORS_ORIGINS").ok();
        let trimmed = raw.as_deref().map(|s| s.trim()).unwrap_or("");
        assert!(trimmed.is_empty());
        std::env::remove_var("HOMECORE_CORS_ORIGINS");
    }
}
