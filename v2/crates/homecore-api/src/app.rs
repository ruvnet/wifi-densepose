//! Axum router wiring. Mounts the §2.1 P2 routes + the WS endpoint.

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::rest;
use crate::state::SharedState;
use crate::ws;

pub type AppState = SharedState;

/// Build the Axum router. The `state` is cloned into each handler at
/// call time via `State<SharedState>`.
pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/api/", get(rest::api_root))
        .route("/api/config", get(rest::get_config))
        .route("/api/states", get(rest::get_states))
        .route("/api/states/:entity_id", get(rest::get_state).post(rest::set_state))
        .route("/api/services", get(rest::get_services))
        .route("/api/services/:domain/:service", post(rest::call_service))
        .route("/api/websocket", get(ws::websocket_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
