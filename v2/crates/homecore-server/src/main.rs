//! `homecore-server` — the HOMECORE integration binary.
//!
//! Boots one process that exposes the full HA-compat surface:
//!
//!   - HomeCore runtime (state machine + event bus + service registry)
//!   - SQLite recorder writing every state_changed event
//!   - REST + WebSocket API on :8123 (HA wire-compat)
//!   - Automation engine subscribed to the state machine
//!   - Assist pipeline (intent recognizer + handler set)
//!
//! Run with:
//!
//!     cargo run -p homecore-server --bin homecore-server -- --bind 0.0.0.0:8123
//!
//! All-feature build with ruvector + wasmtime:
//!
//!     cargo run -p homecore-server --features ruvector,wasmtime -- ...

use std::net::SocketAddr;

use anyhow::Result;
use clap::Parser;
use tracing::{info, warn};

use homecore::service::FnHandler;
use homecore::{Context, EntityId, HomeCore, ServiceCall, ServiceError, ServiceName};
use homecore_api::{build_cors_layer, router, LongLivedTokenStore, SharedState};
use homecore_assist::{
    AssistPipeline, HassCancelAll, HassLightSet, HassNevermind, HassTurnOff, HassTurnOn,
    RegexIntentRecognizer,
};
use homecore_automation::AutomationEngine;
use homecore_recorder::{Recorder, RecorderListener};

use axum::Router;
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

mod gateway;
mod hap;
mod plugins;
mod restore;
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
    #[arg(
        long,
        env = "HOMECORE_APPS_DIR",
        default_value = "/var/lib/cognitum/apps"
    )]
    apps_dir: String,

    /// Per-upstream proxy timeout in milliseconds (ADR-131 §11.1).
    #[arg(long, env = "HOMECORE_GATEWAY_TIMEOUT_MS", default_value_t = 2000)]
    gateway_timeout_ms: u64,

    /// SQLite recorder DB path. Use `:memory:` for an ephemeral run.
    #[arg(long, env = "HOMECORE_DB", default_value = "sqlite://homecore.db")]
    db: String,

    /// HOMECORE registry storage directory restored at startup.
    #[arg(
        long,
        env = "HOMECORE_STORAGE_DIR",
        default_value = ".homecore/storage"
    )]
    storage_dir: std::path::PathBuf,

    /// Maximum registry rows and latest entity states restored at startup.
    #[arg(long, env = "HOMECORE_RESTORE_LIMIT", default_value_t = 100_000)]
    restore_limit: usize,

    /// Friendly location name surfaced via `/api/config`.
    #[arg(long, env = "HOMECORE_LOCATION", default_value = "Home")]
    location_name: String,

    /// Disable the SQLite recorder for low-resource deployments.
    #[arg(long)]
    no_recorder: bool,

    /// Explicitly allow any non-empty bearer token. Development only.
    #[arg(long, env = "HOMECORE_INSECURE_DEV_AUTH", default_value_t = false)]
    insecure_dev_auth: bool,

    /// Seed synthetic demo entities. Disabled by default so simulated
    /// biometric readings are never mistaken for live sensor data.
    #[arg(long)]
    seed_demo_entities: bool,

    /// Optional Home Assistant-style automations YAML file to load at boot.
    #[arg(long, env = "HOMECORE_AUTOMATIONS")]
    automations: Option<std::path::PathBuf>,

    /// Explicit directories containing packaged WebAssembly plugins.
    #[arg(
        long = "plugin-dir",
        env = "HOMECORE_PLUGIN_DIRS",
        value_delimiter = ','
    )]
    plugin_dirs: Vec<std::path::PathBuf>,

    /// Base64 Ed25519 publisher keys trusted to sign WebAssembly packages.
    #[arg(
        long = "plugin-trusted-publisher",
        env = "HOMECORE_PLUGIN_TRUSTED_PUBLISHERS",
        value_delimiter = ','
    )]
    plugin_trusted_publishers: Vec<String>,

    /// Permit unsigned WebAssembly plugins. Unsafe; development only.
    #[arg(long, env = "HOMECORE_PLUGIN_ALLOW_UNSIGNED", default_value_t = false)]
    plugin_allow_unsigned: bool,

    /// Bind address for the optional HomeKit Accessory Protocol server.
    /// The server remains disabled unless this option is supplied.
    #[arg(long, env = "HOMECORE_HAP_BIND")]
    hap_bind: Option<SocketAddr>,

    /// Stable six-octet HAP accessory identifier (for example
    /// `AA:BB:CC:DD:EE:FF`). Required when HAP is enabled.
    #[arg(long, env = "HOMECORE_HAP_DEVICE_ID")]
    hap_device_id: Option<String>,

    /// LAN address published in the HAP mDNS record. Required when HAP is enabled.
    #[arg(long, env = "HOMECORE_HAP_ADVERTISE_ADDR")]
    hap_advertise_addr: Option<std::net::IpAddr>,

    /// DNS hostname published by mDNS for the HAP bridge.
    #[arg(long, env = "HOMECORE_HAP_HOSTNAME", default_value = "homecore")]
    hap_hostname: String,

    /// HAP discovery instance shown to controller applications.
    #[arg(
        long,
        env = "HOMECORE_HAP_INSTANCE_NAME",
        default_value = "HOMECORE Bridge"
    )]
    hap_instance_name: String,

    /// Durable controller pairing database.
    #[arg(
        long,
        env = "HOMECORE_HAP_PAIRING_STORE",
        default_value = ".homecore/hap/pairings.json"
    )]
    hap_pairing_store: std::path::PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    let has_tokens = std::env::var("HOMECORE_TOKENS")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false);
    if !has_tokens && !cli.insecure_dev_auth {
        anyhow::bail!(
            "HOMECORE_TOKENS is required; use --insecure-dev-auth only for isolated development"
        );
    }
    let tokens = if has_tokens {
        let store = LongLivedTokenStore::from_env();
        info!(
            "Provisioned {} bearer token(s) from HOMECORE_TOKENS",
            store.len().await
        );
        store
    } else {
        warn!(
            "Insecure development authentication enabled: any non-empty bearer token is accepted"
        );
        LongLivedTokenStore::allow_any_non_empty()
    };

    info!(
        "HOMECORE booting — bind={}, db={}, location={:?}",
        cli.bind, cli.db, cli.location_name
    );

    // ── 1. HomeCore runtime ─────────────────────────────────────────
    let hc = HomeCore::new();
    info!("HomeCore state machine + event bus + service registry online");

    let recorder = if cli.no_recorder {
        None
    } else {
        match open_recorder(&cli.db).await {
            Ok(recorder) => Some(recorder),
            Err(error) => {
                warn!("Recorder failed to open ({error}) — continuing without persistence");
                None
            }
        }
    };
    let restored =
        restore::restore_startup(&hc, recorder.as_ref(), &cli.storage_dir, cli.restore_limit).await;
    info!(
        entities = restored.entity_entries,
        devices = restored.device_entries,
        states = restored.states,
        truncated = restored.truncated,
        "Startup restoration complete"
    );
    for warning in restored.warnings {
        warn!("{warning}");
    }

    // Seed a representative set of built-in services so the web UI
    // and HA-wire-compat clients see a populated /api/services on
    // first boot. These are no-op handlers (they just echo back the
    // call as JSON for observability) — integrations override them
    // by registering the same ServiceName later.
    register_builtin_services(&hc).await;

    // Seed 10 representative entities so the web UI's Dashboard +
    // States pages have content out of the box. Operators registering
    // real integrations / plugins overwrite these by writing the same
    // entity_id with new values. Opt out with `--no-seed-entities`.
    if cli.seed_demo_entities {
        seed_default_entities(&hc);
    } else {
        info!("Synthetic demo entities disabled (use --seed-demo-entities to opt in)");
    }

    // ── 2. Recorder (optional) ──────────────────────────────────────
    if let Some(recorder) = recorder {
        let _recorder_task = RecorderListener::new(hc.states(), recorder).spawn();
        info!(
            "Recorder open at {} — state_changed events being persisted",
            cli.db
        );
    } else {
        info!("Recorder unavailable or disabled");
    }

    // ── 3. Plugin runtime ───────────────────────────────────────────
    let server_plugins = plugins::ServerPlugins::start(
        hc.clone(),
        plugins::PluginConfig {
            directories: cli.plugin_dirs.clone(),
            trusted_publishers: cli.plugin_trusted_publishers.clone(),
            allow_unsigned: cli.plugin_allow_unsigned,
            limits: homecore_plugins::DiscoveryLimits::default(),
        },
    )
    .await?;

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
    if let Some(path) = &cli.automations {
        let raw = tokio::fs::read_to_string(path).await?;
        let automations: Vec<homecore_automation::Automation> = serde_yaml::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("invalid automations file {}: {e}", path.display()))?;
        for automation in automations {
            automation_engine.register(automation);
        }
    }
    let _automation_task = automation_engine.start();
    info!(
        "Automation engine started ({} automations registered) — \
         state/numeric/event + time triggers active",
        automation_engine.len()
    );

    // ── 5. Assist pipeline ──────────────────────────────────────────
    // ── 6. HAP bridge surface ───────────────────────────────────────
    // ── 7. REST + WS API ────────────────────────────────────────────
    // Token provisioning closes audit findings HC-01/HC-02. If
    // HOMECORE_TOKENS is set in the env, populate the store from
    // its comma-separated list. Otherwise fall back to DEV mode
    // (warn-on-each-request) so existing smoke tests still work.
    let api_state = SharedState::with_tokens(
        hc.clone(),
        cli.location_name,
        env!("CARGO_PKG_VERSION"),
        tokens,
    );
    // BFF gateway (ADR-131 §11): single-origin aggregation of the
    // calibration API + SEED/appliance tiers. Shares the same token store
    // for auth; upstream credentials stay server-side.
    let assist = build_assist_pipeline().await?;
    info!(
        "Assist intent endpoint ready with {} handlers",
        assist.handler_count()
    );
    let hap_runtime = hap::start(
        &hc,
        hap::HapRuntimeConfig {
            bind_addr: cli.hap_bind,
            device_id: cli.hap_device_id.clone(),
            advertise_addr: cli.hap_advertise_addr,
            hostname: cli.hap_hostname.clone(),
            instance_name: cli.hap_instance_name.clone(),
            pairing_store: cli.hap_pairing_store.clone(),
        },
    )
    .await?;
    let gw = GatewayState::with_assist(
        api_state.clone(),
        GatewayConfig {
            calibration_url: cli.calibration_url.clone(),
            calibration_token: cli.calibration_token.clone(),
            apps_dir: std::path::PathBuf::from(&cli.apps_dir),
            timeout: std::time::Duration::from_millis(cli.gateway_timeout_ms),
        },
        assist,
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
    info!(
        "HOMECORE-API listening on http://{} (HA-compat /api + /api/websocket)",
        cli.bind
    );
    info!(
        "HOMECORE BFF gateway active: /api/homecore/* + /api/cal/* (calibration_url={:?})",
        cli.calibration_url
    );
    if !cli.ui_dir.trim().is_empty() {
        info!(
            "HOMECORE-UI (ADR-131) served at http://{}/homecore/ from {}",
            cli.bind, cli.ui_dir
        );
    } else {
        info!("HOMECORE-UI mount disabled (--ui-dir empty)");
    }

    let shutdown_hc = hc.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            if let Err(error) = tokio::signal::ctrl_c().await {
                warn!("failed to install Ctrl-C handler: {error}");
            }
            shutdown_hc
                .bus()
                .fire_system(homecore::SystemEvent::HomeCoreStop);
            info!("Shutdown requested; draining active HTTP connections");
        })
        .await?;
    hap_runtime.shutdown().await?;
    server_plugins.shutdown().await;
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

#[cfg(not(feature = "ruvector"))]
async fn open_recorder(path: &str) -> anyhow::Result<Recorder> {
    Ok(Recorder::open(path).await?)
}

async fn build_assist_pipeline() -> anyhow::Result<AssistPipeline<RegexIntentRecognizer>> {
    let recognizer = RegexIntentRecognizer::new();
    recognizer
        .register(
            "HassTurnOn",
            r"^turn on (?:the )?(?P<entity_id>[a-z0-9_]+\.[a-z0-9_]+)$",
            "*",
        )
        .await?;
    recognizer
        .register(
            "HassTurnOff",
            r"^turn off (?:the )?(?P<entity_id>[a-z0-9_]+\.[a-z0-9_]+)$",
            "*",
        )
        .await?;
    recognizer
        .register(
            "HassLightSet",
            r"^set (?P<entity_id>light\.[a-z0-9_]+) to (?P<brightness>[0-9]{1,3})$",
            "*",
        )
        .await?;
    recognizer
        .register("HassNevermind", r"^(?:never ?mind|cancel that)$", "*")
        .await?;
    recognizer
        .register("HassCancelAll", r"^cancel all automations$", "*")
        .await?;

    let mut pipeline = AssistPipeline::new(recognizer);
    pipeline.register_handler(HassTurnOn);
    pipeline.register_handler(HassTurnOff);
    pipeline.register_handler(HassLightSet);
    pipeline.register_handler(HassNevermind);
    pipeline.register_handler(HassCancelAll);
    Ok(pipeline)
}

#[cfg(feature = "ruvector")]
async fn open_recorder(path: &str) -> anyhow::Result<Recorder> {
    use homecore_recorder::{RuvectorSemanticIndex, SemanticIndex};
    use tokio::sync::RwLock;

    let index = RuvectorSemanticIndex::new(100_000)
        .map_err(|error| anyhow::anyhow!("failed to initialize ruvector index: {error}"))?;
    let semantic: std::sync::Arc<RwLock<dyn SemanticIndex>> =
        std::sync::Arc::new(RwLock::new(index));
    Ok(Recorder::open_with_index(path, semantic).await?)
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "info,homecore=debug,homecore_server=debug,tower_http=info".into()
            }),
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
async fn register_builtin_services(hc: &HomeCore) {
    hc.services()
        .register(
            ServiceName::new("homecore", "ping"),
            FnHandler(|_call| async { Ok(serde_json::json!({ "pong": true })) }),
        )
        .await;

    let snapshot_states = hc.states().clone();
    hc.services()
        .register(
            ServiceName::new("homecore", "snapshot_state"),
            FnHandler(move |_call| {
                let states = snapshot_states.clone();
                async move { Ok(serde_json::to_value(states.all()).unwrap_or_default()) }
            }),
        )
        .await;

    for domain in ["homeassistant", "light", "switch"] {
        for action in ["turn_on", "turn_off", "toggle"] {
            let states = hc.states().clone();
            let required_domain = (domain != "homeassistant").then(|| domain.to_owned());
            hc.services()
                .register(
                    ServiceName::new(domain, action),
                    FnHandler(move |call: ServiceCall| {
                        let states = states.clone();
                        let required_domain = required_domain.clone();
                        async move {
                            let ids = service_entity_ids(&call.data)?;
                            let mut changed = Vec::with_capacity(ids.len());
                            for id in ids {
                                if let Some(domain) = required_domain.as_deref() {
                                    if id.domain() != domain {
                                        return Err(ServiceError::HandlerFailed(format!(
                                            "{}.{} only accepts {domain} entities",
                                            call.name.domain, call.name.service
                                        )));
                                    }
                                }
                                let current = states.get(&id).ok_or_else(|| {
                                    ServiceError::HandlerFailed(format!(
                                        "entity not found: {}",
                                        id.as_str()
                                    ))
                                })?;
                                let next = match call.name.service.as_str() {
                                    "turn_on" => "on",
                                    "turn_off" => "off",
                                    "toggle" if current.state == "on" => "off",
                                    "toggle" => "on",
                                    _ => unreachable!("only registered actions are dispatched"),
                                };
                                let mut attributes = current.attributes.clone();
                                if next == "on" {
                                    if let (Some(object), Some(brightness)) =
                                        (attributes.as_object_mut(), call.data.get("brightness"))
                                    {
                                        object.insert("brightness".into(), brightness.clone());
                                    }
                                }
                                states.set(
                                    id.clone(),
                                    next,
                                    attributes,
                                    Context::child_of(&call.context),
                                );
                                changed.push(id.as_str().to_owned());
                            }
                            Ok(serde_json::json!({ "changed": changed }))
                        }
                    }),
                )
                .await;
        }
    }

    info!(
        "Registered {} executable built-in services",
        hc.services().registered_services().await.len()
    );
}

fn service_entity_ids(data: &serde_json::Value) -> Result<Vec<EntityId>, ServiceError> {
    let value = data
        .get("entity_id")
        .or_else(|| {
            data.get("target")
                .and_then(|target| target.get("entity_id"))
        })
        .ok_or_else(|| ServiceError::HandlerFailed("missing entity_id".into()))?;
    let raw_ids: Vec<&str> = match value {
        serde_json::Value::String(value) => vec![value],
        serde_json::Value::Array(values) => values
            .iter()
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    ServiceError::HandlerFailed("entity_id array must contain strings".into())
                })
            })
            .collect::<Result<_, _>>()?,
        _ => {
            return Err(ServiceError::HandlerFailed(
                "entity_id must be a string or string array".into(),
            ));
        }
    };
    if raw_ids.is_empty() {
        return Err(ServiceError::HandlerFailed(
            "entity_id array cannot be empty".into(),
        ));
    }
    raw_ids
        .into_iter()
        .map(|value| {
            EntityId::parse(value).map_err(|error| ServiceError::HandlerFailed(error.to_string()))
        })
        .collect()
}

/// Register 10 representative entities so a fresh `--db :memory:`
/// boot has content for the web UI. Mirrors `scripts/homecore-seed.sh`
/// — when both are run the script just overwrites these values, so
/// they stay in sync.
fn seed_default_entities(hc: &HomeCore) {
    let entities: Vec<(&str, &str, serde_json::Value)> = vec![
        (
            "sensor.living_room_presence",
            "false",
            serde_json::json!({
                "friendly_name": "Living Room Presence", "device_class": "occupancy",
                "source": "RuView ESP32-C6 BFLD"
            }),
        ),
        (
            "sensor.living_room_motion_score",
            "0.0",
            serde_json::json!({
                "friendly_name": "Living Room Motion Score", "unit_of_measurement": "score",
                "icon": "mdi:motion-sensor"
            }),
        ),
        (
            "sensor.bedroom_breathing_rate",
            "14.5",
            serde_json::json!({
                "friendly_name": "Bedroom Breathing Rate", "unit_of_measurement": "BPM",
                "device_class": "frequency", "source": "Seeed MR60BHA2 mmWave"
            }),
        ),
        (
            "sensor.bedroom_heart_rate",
            "68.0",
            serde_json::json!({
                "friendly_name": "Bedroom Heart Rate", "unit_of_measurement": "BPM",
                "device_class": "frequency", "source": "Seeed MR60BHA2 mmWave"
            }),
        ),
        (
            "light.kitchen_ceiling",
            "on",
            serde_json::json!({
                "friendly_name": "Kitchen Ceiling", "brightness": 230,
                "color_temp_kelvin": 4000, "supported_color_modes": ["color_temp"]
            }),
        ),
        (
            "light.living_room_lamp",
            "off",
            serde_json::json!({
                "friendly_name": "Living Room Lamp", "brightness": 0,
                "supported_color_modes": ["brightness"]
            }),
        ),
        (
            "switch.coffee_maker",
            "off",
            serde_json::json!({
                "friendly_name": "Coffee Maker", "device_class": "outlet"
            }),
        ),
        (
            "binary_sensor.front_door",
            "off",
            serde_json::json!({
                "friendly_name": "Front Door", "device_class": "door"
            }),
        ),
        (
            "climate.thermostat",
            "heat",
            serde_json::json!({
                "friendly_name": "Thermostat", "current_temperature": 21.5,
                "temperature": 22.0, "hvac_modes": ["off", "heat", "cool", "auto"],
                "supported_features": 387
            }),
        ),
        (
            "sensor.air_quality_index",
            "42",
            serde_json::json!({
                "friendly_name": "Air Quality Index", "unit_of_measurement": "AQI",
                "device_class": "aqi"
            }),
        ),
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
    info!(
        "State machine seeded with {} default entit{}",
        total,
        if total == 1 { "y" } else { "ies" }
    );
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
        assert_eq!(
            status,
            StatusCode::OK,
            "GET /homecore/ should serve index.html"
        );
        assert!(
            body.contains("HOMECORE"),
            "index.html should mention HOMECORE"
        );
        assert!(
            body.contains("./js/app.js"),
            "index.html should bootstrap app.js"
        );
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
        for p in [
            "dashboard",
            "rooms",
            "calibration",
            "fleet",
            "seed-detail",
            "entities",
            "cogs",
            "events",
            "audit",
            "settings",
        ] {
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
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let body = String::from_utf8_lossy(&bytes);
        assert_eq!(
            status,
            StatusCode::OK,
            "the HA-compat API must coexist with the UI mount"
        );
        assert!(body.contains("API running"));
    }

    #[tokio::test]
    async fn ui_mount_can_be_disabled() {
        let app = build_app(test_state(), "");
        let (status, _) = get(app, "/homecore/").await;
        assert_eq!(
            status,
            StatusCode::NOT_FOUND,
            "empty --ui-dir disables the mount"
        );
    }

    /// Build the SAME merged + layered surface `main()` serves: API + UI mount
    /// + BFF gateway, with the audited CORS allowlist + tracing applied to the
    ///   whole thing. Used to prove the gateway routes are CORS-covered.
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

    #[tokio::test]
    async fn builtin_light_service_changes_real_state() {
        let hc = HomeCore::new();
        let id = EntityId::parse("light.kitchen").unwrap();
        hc.states().set(
            id.clone(),
            "off",
            serde_json::json!({"friendly_name": "Kitchen"}),
            Context::new(),
        );
        register_builtin_services(&hc).await;

        let result = hc
            .services()
            .call(ServiceCall {
                name: ServiceName::new("light", "turn_on"),
                data: serde_json::json!({"entity_id": "light.kitchen", "brightness": 123}),
                context: Context::new(),
            })
            .await
            .unwrap();

        assert_eq!(result["changed"][0], "light.kitchen");
        let state = hc.states().get(&id).unwrap();
        assert_eq!(state.state, "on");
        assert_eq!(state.attributes["brightness"], 123);
    }

    #[tokio::test]
    async fn unsupported_builtin_service_is_not_success_shaped() {
        let hc = HomeCore::new();
        register_builtin_services(&hc).await;
        let result = hc
            .services()
            .call(ServiceCall {
                name: ServiceName::new("homeassistant", "restart"),
                data: serde_json::json!({}),
                context: Context::new(),
            })
            .await;
        assert!(matches!(result, Err(ServiceError::NotRegistered { .. })));
    }
}
