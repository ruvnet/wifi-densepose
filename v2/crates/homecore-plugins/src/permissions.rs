//! Plugin authority / capability isolation (ADR-162, P5).
//!
//! Wasmtime already gives a plugin **memory** isolation — it cannot read
//! another plugin's linear memory. It does NOT, by itself, stop a plugin
//! from using a host import to write any entity it likes. Before this fix
//! `hc_state_set` happily let any plugin write `lock.front_door` or
//! `alarm_control_panel.*`, and the manifest's `homecore_permissions`
//! claims were parsed but **never consulted** (ADR-161 deferred P5).
//!
//! This module adds **authority isolation**: a plugin may only write
//! entities its manifest declared. The host import consults a
//! [`PermissionSet`] before applying any state write and returns a typed
//! error to the guest (it does **not** panic the host) on a violation.
//!
//! ## Permission grammar
//!
//! Each entry in `homecore_permissions` is one of:
//!
//!   * a bare entity glob — `"light.*"`, `"light.kitchen"`, `"*"`;
//!   * the explicit capability form `"state:write:<glob>"` (the form the
//!     ADR-128 manifest doc shows), e.g. `"state:write:sensor.*"`.
//!
//! A glob supports a single trailing `*` (HA-style domain wildcards:
//! `light.*` matches every `light` entity) and a leading-or-bare `*`
//! (`*` = everything). Exact strings match exactly. A plugin with **no**
//! `state:write` entries can write **nothing** — the secure default.

use crate::manifest::PluginManifest;

/// The set of entity-write permissions a plugin holds, distilled from its
/// manifest `homecore_permissions` at load time.
#[derive(Debug, Clone, Default)]
pub struct PermissionSet {
    /// Glob patterns the plugin may write (state:write authority). Empty =
    /// the plugin may write nothing.
    write_globs: Vec<String>,
}

impl PermissionSet {
    /// Build a permission set from a manifest's `homecore_permissions`.
    ///
    /// Only `state:write` authority is modelled here (the host import this
    /// gates is `hc_state_set`). A bare glob (`"light.*"`) is treated as a
    /// write grant; the explicit `"state:write:<glob>"` form is also
    /// accepted. Other capability strings (`state:read:*`, future verbs)
    /// are ignored for write-gating purposes.
    pub fn from_manifest(manifest: &PluginManifest) -> Self {
        let mut write_globs = Vec::new();
        for claim in &manifest.homecore_permissions {
            let claim = claim.trim();
            if let Some(glob) = claim.strip_prefix("state:write:") {
                write_globs.push(glob.trim().to_string());
            } else if claim.starts_with("state:read:") {
                // read authority — not relevant to write gating.
            } else if !claim.is_empty() {
                // Bare glob — treat as a write grant.
                write_globs.push(claim.to_string());
            }
        }
        Self { write_globs }
    }

    /// An all-allowing set (equivalent to a `"*"` grant). Used by the
    /// legacy permission-free `WasmtimeRuntime::load_wasm` path so existing
    /// callers/tests that do not supply a manifest keep working; the
    /// permission-gated path uses [`Self::from_manifest`].
    pub fn allow_all() -> Self {
        Self {
            write_globs: vec!["*".to_string()],
        }
    }

    /// May this plugin write the given entity id (e.g. `"light.kitchen"`)?
    pub fn may_write(&self, entity_id: &str) -> bool {
        self.write_globs.iter().any(|g| glob_matches(g, entity_id))
    }

    /// Number of write-grant globs (0 = can write nothing).
    pub fn write_grant_count(&self) -> usize {
        self.write_globs.len()
    }
}

/// Match `entity_id` against a single glob pattern.
///
/// Supported forms:
///   * `"*"`              → matches anything.
///   * `"light.*"`        → trailing wildcard: any id with the `light.` prefix.
///   * `"light.kitchen"`  → exact match.
fn glob_matches(pattern: &str, entity_id: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return entity_id.starts_with(prefix);
    }
    pattern == entity_id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_with(perms: &[&str]) -> PluginManifest {
        PluginManifest {
            domain: "p".into(),
            name: "P".into(),
            version: "1".into(),
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
            homecore_permissions: perms.iter().map(|s| s.to_string()).collect(),
            cog_id: None,
        }
    }

    #[test]
    fn domain_glob_allows_same_domain_only() {
        let ps = PermissionSet::from_manifest(&manifest_with(&["light.*"]));
        assert!(ps.may_write("light.kitchen"));
        assert!(ps.may_write("light.bedroom"));
        assert!(!ps.may_write("lock.front_door"));
        assert!(!ps.may_write("alarm_control_panel.home"));
    }

    #[test]
    fn no_permissions_can_write_nothing() {
        let ps = PermissionSet::from_manifest(&manifest_with(&[]));
        assert_eq!(ps.write_grant_count(), 0);
        assert!(!ps.may_write("light.kitchen"));
        assert!(!ps.may_write("sensor.temp"));
    }

    #[test]
    fn explicit_state_write_form_is_honored() {
        let ps = PermissionSet::from_manifest(&manifest_with(&["state:write:sensor.*"]));
        assert!(ps.may_write("sensor.temp"));
        assert!(!ps.may_write("light.kitchen"));
    }

    #[test]
    fn read_grants_do_not_confer_write() {
        let ps = PermissionSet::from_manifest(&manifest_with(&["state:read:lock.*"]));
        assert!(!ps.may_write("lock.front_door"));
    }

    #[test]
    fn exact_entity_grant_is_scoped() {
        let ps = PermissionSet::from_manifest(&manifest_with(&["light.kitchen"]));
        assert!(ps.may_write("light.kitchen"));
        assert!(!ps.may_write("light.bedroom"));
    }

    #[test]
    fn wildcard_grants_everything() {
        let ps = PermissionSet::from_manifest(&manifest_with(&["*"]));
        assert!(ps.may_write("lock.front_door"));
    }
}
