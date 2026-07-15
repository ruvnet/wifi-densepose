//! `homecore-server` — the HOMECORE integration binary.
//!
//! Boots one process that exposes the full HA-compat surface:
//!
//!   - HomeCore runtime (state machine + event bus + service registry)
//!   - SQLite recorder writing every state_changed event
//!   - REST + WebSocket API on :8123 (HA wire-compat)
//!   - Plugin runtime (InProcessRuntime by default; Wasmtime with --features wasmtime)
//!   - Automation engine subscribed to the state machine
//!   - Assist pipeline (intent recognizer + handler set)
//!   - HAP bridge surface (accessories registered via the API)
//!
//! Run with:
//!
//!     cargo run -p homecore-server --bin homecore-server -- --bind 0.0.0.0:8123
//!
//! All-feature build with ruvector + wasmtime:
//!
//!     cargo run -p homecore-server --features ruvector,wasmtime -- ...

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tracing::{info, warn};

use homecore::{Context, EntityId, HomeCore, ServiceCall, ServiceError, ServiceName};
use homecore::service::FnHandler;
use homecore_api::{build_cors_layer, router, LongLivedTokenStore, SharedState};
use homecore_assist::pipeline::default_pipeline;
use homecore_assist::RegexIntentRecognizer;
use homecore_automation::AutomationEngine;
use homecore_hap::{bridge::HapBridge, mdns::HapServiceRecord};
use homecore_plugins::{InProcessRuntime, PluginRegistry};
use homecore_recorder::Recorder;

use axum::Router;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

mod gateway;
use gateway::{GatewayConfig, GatewayState};

/// Compile-time default location of the HOMECORE-UI assets (ADR-131).
/// Works in dev/CI; the appliance overrides with `--ui-dir` /
/// `HOMECORE_UI_DIR`.
const DEFAULT_UI_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/ui");

#[derive(Parser, Debug, Clone)]
#[command(name = "homecore-server", version)]
struct Cli {
    /// Bind address for the HA-compat REST + WS API.
    #[arg(long, env = "HOMECORE_BIND", default_value = "0.0.0.0:8123")]
    bind: SocketAddr,

    /// Directory of the HOMECORE-UI dashboard assets, served at
    /// `/homecore` (ADR-131). Empty string disables the UI mount.
    #[arg(long, env = "HOMECORE_UI_DIR", default_value = DEFAULT_UI_DIR)]
    ui_dir: String,

    /// Base URL of the calibration service (`wifi-densepose calibrate-serve`),
    /// reverse-proxied by the BFF gateway at `/api/cal/*` (ADR-131 §11).
    /// Unset → calibration/room endpoints return a typed 503.
    #[arg(long, env = "HOMECORE_CALIBRATION_URL")]
    calibration_url: Option<String>,

    /// Bearer token for the calibration service (held server-side only,
    /// never exposed to the browser — ADR-131 §11.10).
    #[arg(long, env = "HOMECORE_CALIBRATION_TOKEN")]
    calibration_token: Option<String>,

    /// COG install directory the gateway's supervisor reads (ADR-131 §11.6).
    #[arg(long, env = "HOMECORE_APPS_DIR", default_value = "/var/lib/cognitum/apps")]
    apps_dir: String,

    /// Per-upstream proxy timeout in milliseconds (ADR-131 §11.1).
    #[arg(long, env = "HOMECORE_GATEWAY_TIMEOUT_MS", default_value_t = 2000)]
    gateway_timeout_ms: u64,

    /// SQLite recorder DB path. Use `:memory:` for an ephemeral run.
    #[arg(long, env = "HOMECORE_DB", default_value = "sqlite::memory:")]
    db: String,

    /// Friendly location name surfaced via `/api/config`.
    #[arg(long, env = "HOMECORE_LOCATION", default_value = "Home")]
    location_name: String,

    /// Disable the SQLite recorder for low-resource deployments.
    #[arg(long)]
    no_recorder: bool,

    /// Skip the boot-time entity seeding (10 demo entities including
    /// 4 RuView-derived sensors). Use this when wiring real
    /// integrations that will populate the state machine themselves.
    #[arg(long)]
    no_seed_entities: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();

    info!("HOMECORE booting — bind={}, db={}, location={:?}",
          cli.bind, cli.db, cli.location_name);

    // ── 1. HomeCore runtime ─────────────────────────────────────────
    let hc = HomeCore::new();
    info!("HomeCore state machine + event bus + service registry online");

    // Seed a representative set of built-in services so the web UI
    // and HA-wire-compat clients see a populated /api/services on
    // first boot. These are no-op handlers (they just echo back the
    // call as JSON for observability) — integrations override them
    // by registering the same ServiceName later.
    seed_default_services(&hc).await;

    // Seed 10 representative entities so the web UI's Dashboard +
    // States pages have content out of the box. Operators registering
    // real integrations / plugins overwrite these by writing the same
    // entity_id with new values. Opt out with `--no-seed-entities`.
    if !cli.no_seed_entities {
        seed_default_entities(&hc);
    } else {
        info!("Entity seeding disabled by --no-seed-entities");
    }

    // ── 2. Recorder (optional) ──────────────────────────────────────
    if !cli.no_recorder {
        match Recorder::open(&cli.db).await {
            Ok(recorder) => {
                let recorder = Arc::new(recorder);
                let rec_clone = Arc::clone(&recorder);
                let mut state_rx = hc.states().subscribe();
                tokio::spawn(async move {
                    while let Ok(event) = state_rx.recv().await {
                        if let Err(e) = rec_clone.record_state(&event).await {
                            warn!("recorder write failed: {e}");
                        }
                    }
                });
                info!("Recorder open at {} — state_changed events being persisted", cli.db);
            }
            Err(e) => {
                warn!("Recorder failed to open ({e}) — continuing without persistence");
            }
        }
    } else {
        info!("Recorder disabled by --no-recorder");
    }

    // ── 3. Plugin runtime ───────────────────────────────────────────
    let plugin_runtime = InProcessRuntime;
    let plugin_registry: PluginRegistry<InProcessRuntime> = PluginRegistry::new(plugin_runtime);
    info!("Plugin registry ready (runtime: InProcess; Wasmtime available with --features wasmtime)");
    let _ = plugin_registry; // wired-but-empty at boot; integrations register here

    // ── 4. Automation engine ────────────────────────────────────────
    // Construct AND start the engine (HC-WS-03, ADR-161). `start()`
    // spawns the state-change event loop + the 1 Hz wall-clock timer
    // task so state/numeric/event AND time triggers all fire. The
    // engine is kept alive for the process lifetime (it is moved into a
    // long-lived binding); its background tasks run until the HomeCore
    // broadcast channel closes at shutdown. No automations are loaded at
    // boot yet (YAML loader is P-next); integrations register via
    // `engine.register(..)`.
    let automation_engine = AutomationEngine::new(hc.clone());
    let _automation_task = automation_engine.start();
    info!(
        "Automation engine started ({} automations registered) — \
         state/numeric/event + time triggers active",
        automation_engine.len()
    );

    // ── 5. Assist pipeline ──────────────────────────────────────────
    let recognizer = RegexIntentRecognizer::new();
    let pipeline = default_pipeline(recognizer);
    info!("Assist pipeline ready (5 built-in intent handlers via default_pipeline)");
    let _ = pipeline; // wired-but-idle at boot; voice WS plugs in here

    // ── 6. HAP bridge surface ───────────────────────────────────────
    let hap_record = HapServiceRecord {
        instance_name: "HOMECORE".to_string(),
        port: 51826,
        setup_code: "123-45-678".to_string(),
        device_id: "AA:BB:CC:DD:EE:FF".to_string(),
    };
    let hap_bridge = HapBridge::new(hap_record);
    info!("HAP bridge surface ready ({} accessories registered)", hap_bridge.running_accessories().len());
    let _ = hap_bridge;

    // ── 7. REST + WS API ────────────────────────────────────────────
    // Token provisioning closes audit findings HC-01/HC-02. If
    // HOMECORE_TOKENS is set in the env, populate the store from
    // its comma-separated list. Otherwise fall back to DEV mode
    // (warn-on-each-request) so existing smoke tests still work.
    let tokens = if std::env::var("HOMECORE_TOKENS").map(|v| !v.trim().is_empty()).unwrap_or(false) {
        let s = LongLivedTokenStore::from_env();
        let n = s.len().await;
        info!("LongLivedTokenStore provisioned with {} bearer token(s) from HOMECORE_TOKENS", n);
        s
    } else {
        warn!("HOMECORE_TOKENS not set — token store in DEV mode (any non-empty bearer accepted). Provision real tokens before exposing to the network.");
        LongLivedTokenStore::allow_any_non_empty()
    };
    let api_state = SharedState::with_tokens(
        hc.clone(),
        cli.location_name,
        env!("CARGO_PKG_VERSION"),
        tokens,
    );
    // BFF gateway (ADR-131 §11): single-origin aggregation of the
    // calibration API + SEED/appliance tiers. Shares the same token store
    // for auth; upstream credentials stay server-side.
    let gw = GatewayState::new(
        api_state.clone(),
        GatewayConfig {
            calibration_url: cli.calibration_url.clone(),
            calibration_token: cli.calibration_token.clone(),
            apps_dir: std::path::PathBuf::from(&cli.apps_dir),
            timeout: std::time::Duration::from_millis(cli.gateway_timeout_ms),
        },
    );
    // Merge the HA-compat API + UI mount with the BFF gateway, THEN apply the
    // audited CORS allowlist + request tracing to the WHOLE surface. The
    // gateway routes (`/api/homecore/*`, `/api/cal/*`) are merged in outside
    // `router()`'s own layers, so without this outer layer they would have NO
    // CORS coverage and would not be traced (ADR-131 §11 review). Applying CORS
    // again to the homecore-api routes is idempotent.
    let app = build_app(api_state, &cli.ui_dir)
        .merge(gateway::gateway_router(gw))
        .layer(build_cors_layer())
        .layer(TraceLayer::new_for_http());
    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    info!("HOMECORE-API listening on http://{} (HA-compat /api + /api/websocket)", cli.bind);
    info!(
        "HOMECORE BFF gateway active: /api/homecore/* + /api/cal/* (calibration_url={:?})",
        cli.calibration_url
    );
    if !cli.ui_dir.trim().is_empty() {
        info!("HOMECORE-UI (ADR-131) served at http://{}/homecore/ from {}", cli.bind, cli.ui_dir);
    } else {
        info!("HOMECORE-UI mount disabled (--ui-dir empty)");
    }

    // Run forever (until SIGINT). axum::serve handles graceful shutdown.
    axum::serve(listener, app).await?;
    Ok(())
}

/// Assemble the full HTTP surface: the HA-compat REST + WS router
/// (ADR-130) plus the HOMECORE-UI static mount at `/homecore` (ADR-131).
/// Split out from `main` so it is exercised by the integration tests.
fn build_app(api_state: SharedState, ui_dir: &str) -> Router {
    let app = router(api_state);
    if ui_dir.trim().is_empty() {
        return app;
    }
    // ServeDir serves index.html for the directory root, so /homecore/
    // returns the dashboard and /homecore/js/... /homecore/css/... map
    // straight onto the asset tree the relative <link>/<script> tags use.
    app.nest_service("/homecore", ServeDir::new(ui_dir))
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,homecore=debug,homecore_server=debug,tower_http=info".into()),
        )
        .init();
}

/// Register a representative set of built-in services so `/api/services`
/// is non-empty on first boot. Each handler simply echoes the call back
/// as a JSON acknowledgement — integrations override these by
/// re-registering the same `ServiceName` with a real handler later.
///
/// The set covers the HA wire-compat "starter pack" (homeassistant /
/// light / switch / scene / automation domains) plus a `homecore.*`
/// domain so operators can see HOMECORE-native services distinguished
/// from the HA-compat ones.
async fn seed_default_services(hc: &HomeCore) {
    let echo = || FnHandler(|call: ServiceCall| async move {
        Ok(serde_json::json!({
            "called": format!("{}.{}", call.name.domain, call.name.service),
            "service_data": call.data,
            "acknowledged": true,
        }))
    });

    let svcs = [
        // Conventional HA wire-compat services
        ("homeassistant", "restart"),
        ("homeassistant", "stop"),
        ("homeassistant", "reload_core_config"),
        ("light", "turn_on"),
        ("light", "turn_off"),
        ("light", "toggle"),
        ("switch", "turn_on"),
        ("switch", "turn_off"),
        ("switch", "toggle"),
        ("scene", "apply"),
        ("automation", "trigger"),
        // HOMECORE-native services
        ("homecore", "ping"),
        ("homecore", "snapshot_state"),
    ];

    for (domain, service) in svcs {
        hc.services()
            .register(ServiceName::new(domain, service), echo())
            .await;
    }

    let count = hc.services().registered_services().await.len();
    let _ = ServiceError::NotRegistered { domain: String::new(), service: String::new() };
    info!("Service registry seeded with {} default service(s)", count);
}

/// Register 10 representative entities so a fresh `--db :memory:`
/// boot has content for the web UI. Mirrors `scripts/homecore-seed.sh`
/// — when both are run the script just overwrites these values, so
/// they stay in sync.
fn seed_default_entities(hc: &HomeCore) {
    let entities: Vec<(&str, &str, serde_json::Value)> = vec![
        ("sensor.living_room_presence", "false", serde_json::json!({
            "friendly_name": "Living Room Presence", "device_class": "occupancy",
            "source": "RuView ESP32-C6 BFLD"
        })),
        ("sensor.living_room_motion_score", "0.0", serde_json::json!({
            "friendly_name": "Living Room Motion Score", "unit_of_measurement": "score",
            "icon": "mdi:motion-sensor"
        })),
        ("sensor.bedroom_breathing_rate", "14.5", serde_json::json!({
            "friendly_name": "Bedroom Breathing Rate", "unit_of_measurement": "BPM",
            "device_class": "frequency", "source": "Seeed MR60BHA2 mmWave"
        })),
        ("sensor.bedroom_heart_rate", "68.0", serde_json::json!({
            "friendly_name": "Bedroom Heart Rate", "unit_of_measurement": "BPM",
            "device_class": "frequency", "source": "Seeed MR60BHA2 mmWave"
        })),
        ("light.kitchen_ceiling", "on", serde_json::json!({
            "friendly_name": "Kitchen Ceiling", "brightness": 230,
            "color_temp_kelvin": 4000, "supported_color_modes": ["color_temp"]
        })),
        ("light.living_room_lamp", "off", serde_json::json!({
            "friendly_name": "Living Room Lamp", "brightness": 0,
            "supported_color_modes": ["brightness"]
        })),
        ("switch.coffee_maker", "off", serde_json::json!({
            "friendly_name": "Coffee Maker", "device_class": "outlet"
        })),
        ("binary_sensor.front_door", "off", serde_json::json!({
            "friendly_name": "Front Door", "device_class": "door"
        })),
        ("climate.thermostat", "heat", serde_json::json!({
            "friendly_name": "Thermostat", "current_temperature": 21.5,
            "temperature": 22.0, "hvac_modes": ["off", "heat", "cool", "auto"],
            "supported_features": 387
        })),
        ("sensor.air_quality_index", "42", serde_json::json!({
            "friendly_name": "Air Quality Index", "unit_of_measurement": "AQI",
            "device_class": "aqi"
        })),
    ];

    for (id, state, attrs) in entities {
        match EntityId::parse(id) {
            Ok(eid) => {
                hc.states().set(eid, state, attrs, Context::new());
            }
            Err(e) => warn!("seed_default_entities: bad entity_id {id}: {e}"),
        }
    }

    let _ = ServiceCall {
        name: ServiceName::new("homecore", "noop"),
        data: serde_json::json!({}),
        context: Context::new(),
    };
    let total = hc.states().all().len();
    info!("State machine seeded with {} default entit{}", total,
          if total == 1 { "y" } else { "ies" });
}

#[cfg(test)]
mod ui_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use homecore::HomeCore;
    use homecore_api::{LongLivedTokenStore, SharedState};
    use tower::ServiceExt; // for `oneshot`

    fn test_state() -> SharedState {
        SharedState::with_tokens(
            HomeCore::new(),
            "Test".to_string(),
            "test",
            LongLivedTokenStore::allow_any_non_empty(),
        )
    }

    async fn get(app: Router, path: &str) -> (StatusCode, String) {
        let resp = app
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
            .await
            .unwrap();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn ui_index_is_served_at_homecore() {
        let app = build_app(test_state(), DEFAULT_UI_DIR);
        let (status, body) = get(app, "/homecore/").await;
        assert_eq!(status, StatusCode::OK, "GET /homecore/ should serve index.html");
        assert!(body.contains("HOMECORE"), "index.html should mention HOMECORE");
        assert!(body.contains("./js/app.js"), "index.html should bootstrap app.js");
    }

    #[tokio::test]
    async fn ui_design_tokens_are_served() {
        let app = build_app(test_state(), DEFAULT_UI_DIR);
        let (status, body) = get(app, "/homecore/css/tokens.css").await;
        assert_eq!(status, StatusCode::OK);
        // §3.1 invariant: the exact production palette must be present.
        assert!(body.contains("#4ecdc4"), "--cyan token must be present");
        assert!(body.contains("--purple"), "--purple token must be present");
    }

    #[tokio::test]
    async fn ui_panels_are_served() {
        let app = build_app(test_state(), DEFAULT_UI_DIR);
        for p in ["dashboard", "rooms", "calibration", "fleet", "seed-detail",
                  "entities", "cogs", "events", "audit", "settings"] {
            let (status, _) = get(app.clone(), &format!("/homecore/js/panels/{p}.js")).await;
            assert_eq!(status, StatusCode::OK, "panel {p}.js should be served");
        }
    }

    #[tokio::test]
    async fn api_still_works_alongside_ui_mount() {
        let app = build_app(test_state(), DEFAULT_UI_DIR);
        // `GET /api/` is auth-gated (HC-API-AUTH-01); send a bearer.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/")
                    .header("authorization", "Bearer dev")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert_eq!(status, StatusCode::OK, "the HA-compat API must coexist with the UI mount");
        assert!(body.contains("API running"));
    }

    #[tokio::test]
    async fn ui_mount_can_be_disabled() {
        let app = build_app(test_state(), "");
        let (status, _) = get(app, "/homecore/").await;
        assert_eq!(status, StatusCode::NOT_FOUND, "empty --ui-dir disables the mount");
    }

    /// Build the SAME merged + layered surface `main()` serves: API + UI mount
    /// + BFF gateway, with the audited CORS allowlist + tracing applied to the
    /// whole thing. Used to prove the gateway routes are CORS-covered.
    fn full_app(state: SharedState) -> Router {
        use crate::gateway::{GatewayConfig, GatewayState};
        let gw = GatewayState::new(
            state.clone(),
            GatewayConfig {
                calibration_url: None,
                calibration_token: None,
                apps_dir: std::path::PathBuf::from("/nonexistent-apps-dir"),
                timeout: std::time::Duration::from_millis(200),
            },
        );
        build_app(state, "")
            .merge(crate::gateway::gateway_router(gw))
            .layer(homecore_api::build_cors_layer())
            .layer(TraceLayer::new_for_http())
    }

    #[tokio::test]
    async fn gateway_routes_are_cors_covered_after_merge() {
        // A CORS preflight from the Vite dev origin must succeed (echo the
        // allowed origin) for a GATEWAY route — proving the outer CORS layer
        // covers the merged routes, not just the homecore-api ones.
        let app = full_app(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/homecore/appliance")
                    .header("origin", "http://localhost:5173")
                    .header("access-control-request-method", "GET")
                    .header("access-control-request-headers", "authorization")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // CORS preflight handled by the layer → 2xx with the origin echoed back.
        assert!(
            resp.status().is_success(),
            "gateway preflight should succeed, got {}",
            resp.status()
        );
        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            allow_origin, "http://localhost:5173",
            "gateway route must echo the allowlisted dev origin"
        );
    }
}
