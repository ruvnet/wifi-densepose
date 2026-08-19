//! Server ownership for plugin discovery, setup, event dispatch, and teardown.
//!
//! Native Rust plugins are never discovered from disk. They must be compiled
//! into this binary and exposed through [`NativePluginRegistration`]. External
//! packages are WebAssembly only and require the `wasmtime` feature.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, Result};
use homecore::HomeCore;
use homecore_plugins::{
    DiscoveryLimits, HomeCorePlugin, InProcessRuntime, PluginError, PluginManifest, PluginRegistry,
    StateChangedEventJson,
};
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

#[cfg(feature = "wasmtime")]
use homecore_plugins::{
    discover_plugins, ConfigEntryJson, PluginId, PluginPolicy, WasmPlugin, WasmtimeRuntime,
};

/// Factory entry for a trusted plugin linked into the server at compile time.
type NativePluginFactory = fn() -> Result<(PluginManifest, Arc<dyn HomeCorePlugin>), PluginError>;

pub struct NativePluginRegistration {
    pub create: NativePluginFactory,
}

/// This is intentionally empty in the generic server. Appliance builds can
/// add first-party registrations here; arbitrary Rust cdylibs are unsupported.
const NATIVE_PLUGINS: &[NativePluginRegistration] = &[];

#[derive(Debug, Clone)]
pub struct PluginConfig {
    pub directories: Vec<PathBuf>,
    pub trusted_publishers: Vec<String>,
    pub allow_unsigned: bool,
    #[cfg_attr(not(feature = "wasmtime"), allow(dead_code))]
    pub limits: DiscoveryLimits,
}

pub struct ServerPlugins {
    native: Arc<PluginRegistry<InProcessRuntime>>,
    dispatcher: JoinHandle<()>,
    #[cfg(feature = "wasmtime")]
    wasm: Vec<(PluginId, WasmPlugin)>,
}

impl ServerPlugins {
    /// Start the complete plugin subsystem. Any configured package that fails
    /// discovery, trust verification, compilation, or setup aborts startup.
    pub async fn start(hc: HomeCore, config: PluginConfig) -> Result<Self> {
        let native = Arc::new(PluginRegistry::new(InProcessRuntime));
        for registration in NATIVE_PLUGINS {
            let (manifest, plugin) =
                (registration.create)().context("compiled-in native plugin factory failed")?;
            native
                .load(manifest, plugin, hc.clone())
                .await
                .context("compiled-in native plugin setup failed")?;
        }

        #[cfg(not(feature = "wasmtime"))]
        if !config.directories.is_empty() {
            anyhow::bail!(
                "plugin directories were configured, but homecore-server was built without \
                 --features wasmtime"
            );
        }
        #[cfg(not(feature = "wasmtime"))]
        if config.allow_unsigned || !config.trusted_publishers.is_empty() {
            anyhow::bail!(
                "plugin trust options were configured, but homecore-server was built without \
                 --features wasmtime"
            );
        }

        #[cfg(feature = "wasmtime")]
        let wasm = load_wasm_plugins(&hc, &config).await?;

        let mut receiver = hc.states().subscribe();
        let dispatch_native = native.clone();
        #[cfg(feature = "wasmtime")]
        let dispatch_wasm = wasm.clone();
        let dispatcher = tokio::spawn(async move {
            loop {
                let change = match receiver.recv().await {
                    Ok(change) => change,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                        warn!(
                            skipped,
                            "plugin state dispatcher lagged; events were dropped"
                        );
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                };
                let event = StateChangedEventJson::state_changed(
                    change.entity_id.as_str(),
                    change.new_state.as_ref().map(|state| state.state.as_str()),
                    change
                        .new_state
                        .as_ref()
                        .map(|state| state.attributes.clone())
                        .unwrap_or_else(|| serde_json::json!({})),
                );
                for (id, error) in dispatch_native.state_changed(&event).await {
                    error!(plugin = %id, %error, "native plugin state_changed failed");
                }
                #[cfg(feature = "wasmtime")]
                for (id, plugin) in &dispatch_wasm {
                    if !plugin
                        .subscriptions()
                        .iter()
                        .any(|entity| entity == change.entity_id.as_str())
                    {
                        continue;
                    }
                    let plugin = plugin.clone();
                    let id = id.clone();
                    let event = StateChangedEventJson::state_changed(
                        &event.entity_id,
                        event.new_state.as_deref(),
                        event.attributes.clone(),
                    );
                    match tokio::task::spawn_blocking(move || plugin.call_state_changed(&event))
                        .await
                    {
                        Ok(Ok(0)) => {}
                        Ok(Ok(code)) => {
                            error!(plugin = %id, code, "WASM state_changed returned failure")
                        }
                        Ok(Err(error)) => {
                            error!(plugin = %id, %error, "WASM state_changed trapped")
                        }
                        Err(error) => error!(plugin = %id, %error, "WASM dispatch task failed"),
                    }
                }
            }
        });

        #[cfg(feature = "wasmtime")]
        info!(
            native = NATIVE_PLUGINS.len(),
            wasm = wasm.len(),
            "plugin subsystem started"
        );
        #[cfg(not(feature = "wasmtime"))]
        info!(native = NATIVE_PLUGINS.len(), "plugin subsystem started");
        Ok(Self {
            native,
            dispatcher,
            #[cfg(feature = "wasmtime")]
            wasm,
        })
    }

    /// Stop dispatch first, then tear down WASM and native plugins in reverse
    /// deterministic order. All teardown failures are reported.
    pub async fn shutdown(self) {
        self.dispatcher.abort();
        let _ = self.dispatcher.await;
        #[cfg(feature = "wasmtime")]
        for (id, plugin) in self.wasm.into_iter().rev() {
            match tokio::task::spawn_blocking(move || plugin.call_teardown()).await {
                Ok(Ok(0)) => {}
                Ok(Ok(code)) => error!(plugin = %id, code, "WASM teardown returned failure"),
                Ok(Err(error)) => error!(plugin = %id, %error, "WASM teardown trapped"),
                Err(error) => error!(plugin = %id, %error, "WASM teardown task failed"),
            }
        }
        for (id, error) in self.native.shutdown().await {
            error!(plugin = %id, %error, "native plugin teardown failed");
        }
    }
}

#[cfg(feature = "wasmtime")]
async fn load_wasm_plugins(
    hc: &HomeCore,
    config: &PluginConfig,
) -> Result<Vec<(PluginId, WasmPlugin)>> {
    let policy = if config.allow_unsigned {
        warn!("INSECURE plugin policy enabled: unsigned plugins may execute");
        PluginPolicy::AllowUnsigned
    } else {
        let keys: Vec<_> = config
            .trusted_publishers
            .iter()
            .map(String::as_str)
            .collect();
        PluginPolicy::trusted(&keys).context("invalid trusted plugin publisher key")?
    };
    let discovered =
        discover_plugins(&config.directories, config.limits).context("plugin discovery failed")?;
    let runtime = WasmtimeRuntime::new().context("Wasmtime initialization failed")?;
    let mut loaded: Vec<(PluginId, WasmPlugin)> = Vec::with_capacity(discovered.len());
    for package in discovered {
        let id = PluginId::new(&package.manifest.domain);
        let bytes = package
            .read_module(config.limits)
            .with_context(|| format!("failed reading plugin `{id}`"))?;
        let plugin = runtime
            .load_plugin(&package.manifest, &bytes, hc.clone(), &policy)
            .with_context(|| format!("plugin `{id}` rejected before setup"))?;
        let config_entry = serde_json::to_string(&ConfigEntryJson::bootstrap(id.as_str()))?;
        let setup_plugin = plugin.clone();
        let result = tokio::task::spawn_blocking(move || setup_plugin.call_setup(&config_entry))
            .await
            .with_context(|| format!("plugin `{id}` setup task failed"))?
            .with_context(|| format!("plugin `{id}` setup trapped"))?;
        if result != 0 {
            let _ = plugin.call_teardown();
            teardown_wasm(loaded).await;
            anyhow::bail!("plugin `{id}` setup returned failure code {result}");
        }
        info!(
            plugin = %id,
            package = %package.package_dir.display(),
            "signed WASM plugin loaded"
        );
        loaded.push((id, plugin));
    }
    Ok(loaded)
}

#[cfg(feature = "wasmtime")]
async fn teardown_wasm(plugins: Vec<(PluginId, WasmPlugin)>) {
    for (_, plugin) in plugins.into_iter().rev() {
        let _ = tokio::task::spawn_blocking(move || plugin.call_teardown()).await;
    }
}
