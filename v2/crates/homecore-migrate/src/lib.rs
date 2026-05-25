//! homecore-migrate — Migration tooling from Python Home Assistant.
//!
//! Implements [ADR-134](../../docs/adr/ADR-134-homecore-migration-from-python-ha.md)
//! (referenced via ADR-126 §4, series map row ADR-134 HOMECORE-MIGRATE).
//!
//! ## P1 scope
//!
//! - [`storage`] — `HaStorageDir`, `HaStorageEnvelope`; `read_envelope(path)`
//! - [`storage_format`] — versioned format parsers (`v13`); unknown minor_version → hard error
//! - [`entity_registry`] — `core.entity_registry` → `Vec<homecore::EntityEntry>`
//! - [`device_registry`] — `core.device_registry` → `Vec<DeviceImport>` (P1 stub)
//! - [`config_entries`] — `core.config_entries` diagnostic (count + domain list; P2 converts)
//! - [`secrets`] — `secrets.yaml` → `HashMap<String, String>`
//! - [`automations`] — `automations.yaml` count + ID list (P2 converts)
//! - [`cli`] — `clap`-derived subcommand types shared between `src/main.rs` and tests
//!
//! ## What is NOT here yet (deferred to P2+)
//!
//! - Conversion of `config_entries` to HOMECORE plugin manifests
//! - Conversion of `automations.yaml` to `homecore-automation` YAML
//! - Side-by-side runtime mode (requires `homecore-recorder`, ADR-132)
//! - `!secret` reference resolution in non-secrets YAML files

pub mod automations;
pub mod cli;
pub mod config_entries;
pub mod device_registry;
pub mod entity_registry;
pub mod secrets;
pub mod storage;
pub mod storage_format;

/// Crate-level error type. Each module exposes `MigrateError` variants.
#[derive(Debug, thiserror::Error)]
pub enum MigrateError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("JSON parse error in {path}: {source}")]
    JsonParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("YAML parse error in {path}: {source}")]
    YamlParse {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },

    /// Fired when the outer `{version, minor_version}` envelope version is
    /// known but the `minor_version` is not supported by any compiled parser.
    /// Per ADR-134 §6 Q5: hard error on unknown minor_version.
    #[error(
        "unsupported schema version in {file}: \
         version={version} minor_version={minor_version}. \
         Upgrade homecore-migrate or downgrade HA to a supported release."
    )]
    UnsupportedSchemaVersion {
        file: String,
        version: u32,
        minor_version: u32,
    },

    #[error("missing required field '{field}' in {context}")]
    MissingField { field: String, context: String },

    #[error("entity_id parse error: {0}")]
    EntityId(#[from] homecore::EntityIdError),
}
