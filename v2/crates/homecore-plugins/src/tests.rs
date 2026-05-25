//! Unit tests for homecore-plugins P1 scaffold.
//!
//! Covers: manifest parse + round-trip, manifest field validation,
//! PluginRegistry load/unload/list/duplicate, InProcessRuntime,
//! and PluginError variants.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use homecore::HomeCore;
    use tokio::sync::Mutex;

    use crate::error::PluginError;
    use crate::manifest::PluginManifest;
    use crate::plugin::{HomeCorePlugin, PluginId};
    use crate::registry::PluginRegistry;
    use crate::runtime::InProcessRuntime;

    // ── Test double ────────────────────────────────────────────────────────

    /// Minimal plugin that records setup/unload calls.
    struct TestPlugin {
        pub setup_called: Mutex<bool>,
        pub unload_called: Mutex<bool>,
    }

    impl TestPlugin {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                setup_called: Mutex::new(false),
                unload_called: Mutex::new(false),
            })
        }
    }

    #[async_trait]
    impl HomeCorePlugin for TestPlugin {
        async fn setup(&self, _hc: HomeCore) -> Result<(), PluginError> {
            *self.setup_called.lock().await = true;
            Ok(())
        }

        async fn unload(&self) -> Result<(), PluginError> {
            *self.unload_called.lock().await = true;
            Ok(())
        }
    }

    fn minimal_manifest(domain: &str) -> PluginManifest {
        PluginManifest {
            domain: domain.into(),
            name: "Test Plugin".into(),
            version: "1.0.0".into(),
            documentation: None,
            iot_class: None,
            config_flow: false,
            integration_type: None,
            dependencies: vec![],
            requirements: vec![],
            wasm_module: None,
            wasm_module_hash: None,
            wasm_module_sig: None,
            publisher_key: None,
            min_homecore_version: None,
            host_imports_required: vec![],
            homecore_permissions: vec![],
            cog_id: None,
        }
    }

    // ── Manifest tests ─────────────────────────────────────────────────────

    #[test]
    fn manifest_parse_round_trip() {
        let json = r#"{
            "domain": "mqtt",
            "name": "MQTT",
            "version": "2025.1.0",
            "iot_class": "local_push",
            "config_flow": true,
            "dependencies": [],
            "requirements": [],
            "wasm_module": "mqtt.wasm",
            "homecore_permissions": ["state:write:sensor.*"]
        }"#;

        let m = PluginManifest::parse_json(json).expect("should parse");
        assert_eq!(m.domain, "mqtt");
        assert_eq!(m.version, "2025.1.0");
        assert!(m.config_flow);
        assert_eq!(m.homecore_permissions, vec!["state:write:sensor.*"]);

        // round-trip: serialize back to JSON and re-parse
        let serialised = serde_json::to_string(&m).expect("should serialise");
        let m2 = PluginManifest::parse_json(&serialised).expect("round-trip should parse");
        assert_eq!(m.domain, m2.domain);
        assert_eq!(m.version, m2.version);
    }

    #[test]
    fn manifest_rejects_empty_domain() {
        let json = r#"{"domain":"","name":"X","version":"1.0.0"}"#;
        let err = PluginManifest::parse_json(json).unwrap_err();
        assert!(
            err.to_string().contains("domain"),
            "error should mention domain: {err}"
        );
    }

    #[test]
    fn manifest_rejects_missing_domain() {
        let json = r#"{"name":"X","version":"1.0.0"}"#;
        // serde will fill domain as "" due to missing field → validation rejects
        let err = PluginManifest::parse_json(json).unwrap_err();
        // Either a serde error (missing field) or a validation error is acceptable
        let s = err.to_string();
        assert!(!s.is_empty(), "should produce a non-empty error");
    }

    #[test]
    fn manifest_rejects_empty_version() {
        let json = r#"{"domain":"lights","name":"Lights","version":""}"#;
        let err = PluginManifest::parse_json(json).unwrap_err();
        assert!(
            err.to_string().contains("version"),
            "error should mention version: {err}"
        );
    }

    // ── Registry + InProcessRuntime tests ─────────────────────────────────

    #[tokio::test]
    async fn registry_load_and_list() {
        let hc = HomeCore::new();
        let registry = PluginRegistry::new(InProcessRuntime);
        let plugin = TestPlugin::new();
        let manifest = minimal_manifest("lights");

        let id = registry
            .load(manifest, plugin.clone(), hc)
            .await
            .expect("load should succeed");

        assert_eq!(id.as_str(), "lights");
        assert!(*plugin.setup_called.lock().await, "setup should have been called");

        let listing = registry.list().await;
        assert_eq!(listing.len(), 1);
        assert_eq!(listing[0].0.as_str(), "lights");
    }

    #[tokio::test]
    async fn registry_unload_removes_plugin() {
        let hc = HomeCore::new();
        let registry = PluginRegistry::new(InProcessRuntime);
        let plugin = TestPlugin::new();

        let id = registry
            .load(minimal_manifest("switch"), plugin.clone(), hc)
            .await
            .expect("load should succeed");

        registry.unload(&id).await.expect("unload should succeed");
        assert!(*plugin.unload_called.lock().await, "unload should have been called");
        assert_eq!(registry.list().await.len(), 0);
    }

    #[tokio::test]
    async fn registry_rejects_duplicate_load() {
        let hc1 = HomeCore::new();
        let hc2 = HomeCore::new();
        let registry = PluginRegistry::new(InProcessRuntime);

        registry
            .load(minimal_manifest("sensor"), TestPlugin::new(), hc1)
            .await
            .expect("first load should succeed");

        let err = registry
            .load(minimal_manifest("sensor"), TestPlugin::new(), hc2)
            .await
            .unwrap_err();

        assert!(
            matches!(err, PluginError::AlreadyLoaded(_)),
            "expected AlreadyLoaded, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn registry_unload_unknown_plugin_returns_not_found() {
        let registry = PluginRegistry::new(InProcessRuntime);
        let id = PluginId::new("nonexistent");
        let err = registry.unload(&id).await.unwrap_err();
        assert!(
            matches!(err, PluginError::NotFound(_)),
            "expected NotFound, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn in_process_runtime_setup_called() {
        let hc = HomeCore::new();
        let registry = PluginRegistry::new(InProcessRuntime);
        let plugin = TestPlugin::new();

        registry
            .load(minimal_manifest("climate"), plugin.clone(), hc)
            .await
            .expect("load should succeed");

        assert!(
            *plugin.setup_called.lock().await,
            "InProcessRuntime must call setup"
        );
    }

    // ── Error display ──────────────────────────────────────────────────────

    #[test]
    fn error_display_variants() {
        let e1 = PluginError::AlreadyLoaded("mqtt".into());
        assert!(e1.to_string().contains("mqtt"));

        let e2 = PluginError::NotFound("climate".into());
        assert!(e2.to_string().contains("climate"));

        let e3 = PluginError::RuntimeError("boom".into());
        assert!(e3.to_string().contains("boom"));
    }
}
