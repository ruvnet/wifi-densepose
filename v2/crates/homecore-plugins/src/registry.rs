//! `PluginRegistry` — load, unload, and list HOMECORE plugins.
//!
//! The registry is runtime-agnostic: it accepts any type that implements
//! [`PluginRuntime`] and delegates load/unload to it. This allows swapping
//! the `InProcessRuntime` (P1) for a `WasmtimeRuntime` (P2) without
//! changing registry code.

use std::collections::HashMap;
use std::sync::Arc;

use homecore::HomeCore;
use tokio::sync::RwLock;

use crate::error::PluginError;
use crate::manifest::PluginManifest;
use crate::plugin::{HomeCorePlugin, PluginId};
use crate::runtime::{LoadedPlugin, PluginRuntime};

/// Holds all loaded plugins keyed by `PluginId`.
///
/// Thread-safe via `RwLock` — concurrent reads are cheap; writes (load /
/// unload) take an exclusive lock only while mutating the map.
pub struct PluginRegistry<R: PluginRuntime> {
    runtime: R,
    plugins: RwLock<HashMap<PluginId, LoadedPlugin>>,
}

impl<R: PluginRuntime> PluginRegistry<R> {
    /// Create an empty registry backed by `runtime`.
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            plugins: RwLock::new(HashMap::new()),
        }
    }

    /// Load a plugin, call its `setup` hook, and insert it into the registry.
    ///
    /// Returns `PluginError::AlreadyLoaded` if a plugin with the same ID is
    /// already registered.
    pub async fn load(
        &self,
        manifest: PluginManifest,
        plugin: Arc<dyn HomeCorePlugin>,
        hc: HomeCore,
    ) -> Result<PluginId, PluginError> {
        let id = PluginId::new(&manifest.domain);

        {
            let guard = self.plugins.read().await;
            if guard.contains_key(&id) {
                return Err(PluginError::AlreadyLoaded(id.to_string()));
            }
        }

        let loaded = self
            .runtime
            .load(id.clone(), manifest, plugin)
            .await?;

        loaded
            .setup(hc)
            .await
            .map_err(|e| PluginError::SetupFailed(e.to_string()))?;

        self.plugins.write().await.insert(id.clone(), loaded);
        Ok(id)
    }

    /// Unload a plugin by ID, calling its `unload` hook first.
    ///
    /// Returns `PluginError::NotFound` if the plugin was not loaded.
    pub async fn unload(&self, id: &PluginId) -> Result<(), PluginError> {
        let loaded = {
            let mut guard = self.plugins.write().await;
            guard
                .remove(id)
                .ok_or_else(|| PluginError::NotFound(id.to_string()))?
        };

        loaded
            .unload()
            .await
            .map_err(|e| PluginError::UnloadFailed(e.to_string()))?;

        Ok(())
    }

    /// Return a snapshot of currently loaded plugin IDs and their manifest domains.
    pub async fn list(&self) -> Vec<(PluginId, String)> {
        let guard = self.plugins.read().await;
        guard
            .iter()
            .map(|(id, lp)| (id.clone(), lp.manifest.domain.clone()))
            .collect()
    }

    /// Return `true` if a plugin with this ID is loaded.
    pub async fn contains(&self, id: &PluginId) -> bool {
        self.plugins.read().await.contains_key(id)
    }
}
