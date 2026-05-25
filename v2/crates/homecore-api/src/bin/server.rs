//! `homecore-api-server` binary. Boots a HomeCore runtime and serves
//! the HA-compat REST + WS API on `:8123`.
//!
//! P1: bare-minimum bring-up. No persistence, no plugins, no auth
//! beyond "any non-empty bearer". Useful for `curl` smoke tests of
//! the wire format from the existing HA companion app:
//!
//!     cargo run -p homecore-api --bin homecore-api-server
//!     curl -H "Authorization: Bearer test" http://127.0.0.1:8123/api/

use homecore::HomeCore;
use homecore_api::{router, SharedState, DEFAULT_PORT};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug,homecore_api=debug".into()),
        )
        .init();

    let homecore = HomeCore::new();
    let state = SharedState::new(homecore);
    let app = router(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], DEFAULT_PORT));
    tracing::info!("HOMECORE-API listening on http://{addr}  (HA-compat /api + /api/websocket)");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
