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

use homecore::HomeCore;
use homecore_api::{router, LongLivedTokenStore, SharedState};
use homecore_assist::pipeline::default_pipeline;
use homecore_assist::RegexIntentRecognizer;
use homecore_automation::AutomationEngine;
use homecore_hap::{bridge::HapBridge, mdns::HapServiceRecord};
use homecore_plugins::{InProcessRuntime, PluginRegistry};
use homecore_recorder::Recorder;

#[derive(Parser, Debug, Clone)]
#[command(name = "homecore-server", version)]
struct Cli {
    /// Bind address for the HA-compat REST + WS API.
    #[arg(long, env = "HOMECORE_BIND", default_value = "0.0.0.0:8123")]
    bind: SocketAddr,

    /// SQLite recorder DB path. Use `:memory:` for an ephemeral run.
    #[arg(long, env = "HOMECORE_DB", default_value = "sqlite::memory:")]
    db: String,

    /// Friendly location name surfaced via `/api/config`.
    #[arg(long, env = "HOMECORE_LOCATION", default_value = "Home")]
    location_name: String,

    /// Disable the SQLite recorder for low-resource deployments.
    #[arg(long)]
    no_recorder: bool,
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
    let _automation_engine = AutomationEngine::new(hc.clone());
    info!("Automation engine ready (no automations loaded yet)");

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
    let app = router(api_state);
    let listener = tokio::net::TcpListener::bind(cli.bind).await?;
    info!("HOMECORE-API listening on http://{} (HA-compat /api + /api/websocket)", cli.bind);

    // Run forever (until SIGINT). axum::serve handles graceful shutdown.
    axum::serve(listener, app).await?;
    Ok(())
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,homecore=debug,homecore_server=debug,tower_http=info".into()),
        )
        .init();
}
