//! `PluginRegistry` — load, unload, and list HOMECORE plugins.
//!
//! The registry is runtime-agnostic: it accepts any type that implements
//! [`PluginRuntime`] and delegates load/unload to it. This allows swapping
//! the `InProcessRuntime` (P1) for a `WasmtimeRuntime` (P2) without
//! changing registry code.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use homecore::HomeCore;
use tokio::sync::RwLock;

use crate::error::PluginError;
use crate::manifest::PluginManifest;
use crate::plugin::{HomeCorePlugin, PluginId};
use crate::runtime::{LoadedPlugin, PluginRuntime};
use crate::StateChangedEventJson;

/// Holds all loaded plugins keyed by `PluginId`.
///
/// Thread-safe via `RwLock` — concurrent reads are cheap; writes (load /
/// unload) take an exclusive lock only while mutating the map.
pub struct PluginRegistry<R: PluginRuntime> {
    runtime: R,
    plugins: RwLock<BTreeMap<PluginId, LoadedPlugin>>,
    loading: RwLock<BTreeSet<PluginId>>,
}

impl<R: PluginRuntime> PluginRegistry<R> {
    /// Create an empty registry backed by `runtime`.
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            plugins: RwLock::new(BTreeMap::new()),
            loading: RwLock::new(BTreeSet::new()),
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
            let mut loading = self.loading.write().await;
            if guard.contains_key(&id) || !loading.insert(id.clone()) {
                return Err(PluginError::AlreadyLoaded(id.to_string()));
            }
        }

        let result = self
            .runtime
            .load(id.clone(), manifest, plugin)
            .await;
        let loaded = match result {
            Ok(loaded) => loaded,
            Err(error) => {
                self.loading.write().await.remove(&id);
                return Err(error);
            }
        };
        if let Err(error) = loaded.setup(hc).await {
            // Best-effort rollback for partially initialized plugins.
            let _ = loaded.unload().await;
            self.loading.write().await.remove(&id);
            return Err(PluginError::SetupFailed(error.to_string()));
        }
        self.plugins.write().await.insert(id.clone(), loaded);
        self.loading.write().await.remove(&id);
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

        if let Err(error) = loaded.unload().await {
            // Keep the handle reachable so teardown can be retried.
            self.plugins.write().await.insert(id.clone(), loaded);
            return Err(PluginError::UnloadFailed(error.to_string()));
        }

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

    /// Dispatch a state change in stable plugin-domain order.
    pub async fn state_changed(&self, event: &StateChangedEventJson) -> Vec<(PluginId, PluginError)> {
        let guard = self.plugins.read().await;
        let mut errors = Vec::new();
        for (id, plugin) in guard.iter() {
            if let Err(error) = plugin.state_changed(event).await {
                errors.push((id.clone(), error));
            }
        }
        errors
    }

    /// Tear down all plugins in reverse domain order. Failures are collected
    /// while remaining plugins still receive teardown.
    pub async fn shutdown(&self) -> Vec<(PluginId, PluginError)> {
        let ids: Vec<_> = self.plugins.read().await.keys().cloned().rev().collect();
        let mut errors = Vec::new();
        for id in ids {
            if let Err(error) = self.unload(&id).await {
                errors.push((id, error));
            }
        }
        errors
    }
}
