//! Plugin manifest — superset of HA's `manifest.json`.
//!
//! See ADR-128 §3 for the full field list. Fields present in HA's schema
//! are preserved verbatim. HOMECORE-specific fields are marked `[HOMECORE]`.

use serde::{Deserialize, Serialize};

use crate::error::PluginError;

/// Coarse-grained permission claim string (glob pattern).
/// Example: `"state:write:sensor.*"`.
pub type PermissionClaim = String;

/// HA `iot_class` values (non-exhaustive — HA adds new classes over time).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IotClass {
    LocalPush,
    LocalPolling,
    CloudPush,
    CloudPolling,
    AssumedState,
    Calculated,
    #[serde(other)]
    Other,
}

/// HOMECORE integration type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegrationType {
    Integration,
    Helper,
    Entity,
    #[serde(other)]
    Other,
}

/// Parsed and validated plugin manifest.
///
/// Serialises to/from HA-compatible `manifest.json`. HOMECORE-only fields
/// are `Option<…>` so that a plain HA manifest is a valid (native-only)
/// HOMECORE manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique integration domain identifier (e.g. `"mqtt"`).
    pub domain: String,

    /// Human-readable integration name.
    pub name: String,

    /// SemVer-ish version string (HA uses calendar-versioning, e.g. `"2025.1.0"`).
    pub version: String,

    /// Optional documentation URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documentation: Option<String>,

    /// HA `iot_class` — how the integration communicates with the device.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iot_class: Option<IotClass>,

    /// Whether this integration ships a UI config flow.
    #[serde(default)]
    pub config_flow: bool,

    /// HOMECORE integration type (optional, defaults to Integration).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integration_type: Option<IntegrationType>,

    /// Intra-HOMECORE dependencies (other plugin domains this one requires).
    #[serde(default)]
    pub dependencies: Vec<String>,

    /// External package requirements — kept for schema compat, ignored in HOMECORE
    /// (WASM modules carry their own static deps, no pip).
    #[serde(default)]
    pub requirements: Vec<String>,

    // ── [HOMECORE] fields ──────────────────────────────────────────────────

    /// [HOMECORE] Relative path to the `.wasm` binary (absent for native plugins).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wasm_module: Option<String>,

    /// [HOMECORE] `sha256:<hex>` hash of the wasm binary.
    ///
    /// **(P4 — ENFORCED, ADR-162):** `verify::verify_module` computes the
    /// SHA-256 of the real `.wasm` bytes on load and rejects the module if
    /// it does not equal this hash (tamper detection). See [`crate::verify`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wasm_module_hash: Option<String>,

    /// [HOMECORE] Ed25519 signature of the wasm binary hash (`ed25519:<base64>`).
    ///
    /// **(P4 — ENFORCED, ADR-162):** verified against `publisher_key` over
    /// the SHA-256 module digest before instantiation. A bad/forged/absent
    /// signature is rejected under the secure trust policy (the
    /// `cog-ha-matter::witness_signing` Ed25519 pattern is reused).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wasm_module_sig: Option<String>,

    /// [HOMECORE] Ed25519 public key of the plugin publisher.
    ///
    /// **(P4 — ENFORCED, ADR-162):** used to verify `wasm_module_sig`, and
    /// checked against the host's [`crate::verify::PluginPolicy`] trust
    /// allowlist — an unknown publisher is rejected by the secure default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub publisher_key: Option<String>,

    /// [HOMECORE] Minimum HOMECORE version required by this plugin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_homecore_version: Option<String>,

    /// [HOMECORE] Subset of host functions the WASM module imports.
    #[serde(default)]
    pub host_imports_required: Vec<String>,

    /// [HOMECORE] Coarse-grained permission claims (glob patterns).
    ///
    /// **(P5 — ENFORCED, ADR-162):** `state:write:<glob>` (or a bare entity
    /// glob like `light.*`) grants are parsed into a
    /// [`crate::permissions::PermissionSet` ] and consulted by the
    /// `hc_state_set` host import. A plugin can no longer write an entity it
    /// did not declare; a plugin with no write grants can write nothing.
    #[serde(default)]
    pub homecore_permissions: Vec<PermissionClaim>,

    /// [HOMECORE] Seed app registry cog ID for distribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cog_id: Option<String>,
}

impl PluginManifest {
    /// Parse a `manifest.json` JSON string and validate required fields.
    ///
    /// Required fields: `domain`, `name`, `version`.
    pub fn parse_json(s: &str) -> Result<Self, PluginError> {
        let m: Self = serde_json::from_str(s)
            .map_err(|e| PluginError::InvalidManifest(e.to_string()))?;
        m.validate()?;
        Ok(m)
    }

    fn validate(&self) -> Result<(), PluginError> {
        if self.domain.trim().is_empty() {
            return Err(PluginError::InvalidManifest(
                "manifest `domain` must not be empty".into(),
            ));
        }
        if self.name.trim().is_empty() {
            return Err(PluginError::InvalidManifest(
                "manifest `name` must not be empty".into(),
            ));
        }
        if self.version.trim().is_empty() {
            return Err(PluginError::InvalidManifest(
                "manifest `version` must not be empty".into(),
            ));
        }
        Ok(())
    }
}
