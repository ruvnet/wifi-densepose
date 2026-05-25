//! HA `.storage/` directory abstraction and the outer storage envelope.
//!
//! Every file in `.storage/` shares the same outer JSON shape:
//!
//! ```json
//! {
//!   "version": 1,
//!   "minor_version": 3,
//!   "key": "core.entity_registry",
//!   "data": { ... }
//! }
//! ```
//!
//! `read_envelope` reads and validates this outer wrapper. The `data` field is
//! left as `serde_json::Value` — version-specific parsers in `storage_format`
//! are responsible for further deserialization.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::MigrateError;

/// Points to a HA `.storage/` directory.
#[derive(Clone, Debug)]
pub struct HaStorageDir {
    pub path: PathBuf,
}

impl HaStorageDir {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Returns the full path to a named storage file.
    pub fn file_path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

/// The outer JSON envelope that wraps every HA `.storage/*.json` file.
/// Source: `homeassistant/helpers/storage.py` `Store._write_data`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HaStorageEnvelope {
    pub version: u32,
    /// Introduced in HA 2022.x for backwards-compatible schema additions.
    #[serde(default)]
    pub minor_version: u32,
    pub key: String,
    /// Inner payload. Parsed by versioned format-specific code.
    pub data: serde_json::Value,
}

/// Read and deserialize a `.storage/*.json` envelope from `path`.
///
/// Returns `MigrateError::Io` if the file cannot be read, or
/// `MigrateError::JsonParse` if the JSON is malformed.
pub fn read_envelope(path: &Path) -> Result<HaStorageEnvelope, MigrateError> {
    let raw = std::fs::read_to_string(path).map_err(|e| MigrateError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    serde_json::from_str(&raw).map_err(|e| MigrateError::JsonParse {
        path: path.display().to_string(),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WELL_FORMED: &str = r#"{
        "version": 1,
        "minor_version": 3,
        "key": "core.entity_registry",
        "data": {"entities": []}
    }"#;

    #[test]
    fn envelope_parses_well_formed() {
        let env: HaStorageEnvelope = serde_json::from_str(WELL_FORMED).unwrap();
        assert_eq!(env.version, 1);
        assert_eq!(env.minor_version, 3);
        assert_eq!(env.key, "core.entity_registry");
        assert!(env.data.get("entities").is_some());
    }

    #[test]
    fn envelope_missing_minor_version_defaults_to_zero() {
        let json = r#"{"version": 1, "key": "core.config_entries", "data": {}}"#;
        let env: HaStorageEnvelope = serde_json::from_str(json).unwrap();
        assert_eq!(env.minor_version, 0);
    }

    #[test]
    fn envelope_rejects_malformed_json() {
        let result = serde_json::from_str::<HaStorageEnvelope>("not json");
        assert!(result.is_err());
    }
}
