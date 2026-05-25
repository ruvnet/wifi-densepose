//! Parser for `core.entity_registry` (HA storage schema v1, minor_version 1–13).
//!
//! Reads the `.storage/core.entity_registry` file and converts it into a
//! `Vec<homecore::EntityEntry>` that can be loaded directly into the HOMECORE
//! in-memory entity registry.
//!
//! Schema as of HA 2025.1 (minor_version=13):
//! ```json
//! {
//!   "version": 1, "minor_version": 13, "key": "core.entity_registry",
//!   "data": {
//!     "entities": [
//!       {
//!         "entity_id": "light.kitchen",
//!         "unique_id": "hue_lamp_42",
//!         "platform": "hue",
//!         "name": "Kitchen lamp",
//!         "disabled_by": null,
//!         "area_id": "kitchen",
//!         "device_id": "abc123",
//!         "entity_category": null,
//!         "config_entry_id": "ce_001"
//!       }
//!     ]
//!   }
//! }
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};

use homecore::{registry::DisabledBy, EntityCategory, EntityEntry, EntityId};

use crate::{
    storage::read_envelope,
    storage_format::v13,
    MigrateError,
};

// Key used by `inspect` subcommand when scanning the directory.
#[allow(dead_code)]
const FILE_KEY: &str = "core.entity_registry";

/// Raw HA entity registry data block (the `data` field in the envelope).
#[derive(Debug, Deserialize)]
struct HaEntityRegistryData {
    entities: Vec<HaEntityRow>,
    /// Deleted-entity tombstones (ignored in P1 — forwarded as Q5 note).
    #[serde(default)]
    #[allow(dead_code)]
    deleted_entities: Vec<serde_json::Value>,
}

/// A single row from `data.entities`.
#[derive(Debug, Serialize, Deserialize)]
struct HaEntityRow {
    entity_id: String,
    #[serde(default)]
    unique_id: Option<String>,
    platform: String,
    /// User-set display name (separate from HA-integration default name).
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    disabled_by: Option<HaDisabledBy>,
    #[serde(default)]
    area_id: Option<String>,
    #[serde(default)]
    device_id: Option<String>,
    #[serde(default)]
    entity_category: Option<HaEntityCategory>,
    #[serde(default)]
    config_entry_id: Option<String>,
    // Fields present in v13 that we capture but do not yet map to HOMECORE.
    // Forwarded as Q5 items.
    #[serde(default)]
    hidden_by: Option<String>,        // v13: "user" | "integration"
    #[serde(default)]
    has_entity_name: Option<bool>,    // v13: HA naming convention flag
    #[serde(default)]
    original_name: Option<String>,    // v13: integration-provided default name
    #[serde(default)]
    icon: Option<String>,             // v13: mdi:xxx icon override
    #[serde(default)]
    original_icon: Option<String>,    // v13: integration-provided icon
    #[serde(default)]
    aliases: Option<Vec<String>>,     // v13: user-set aliases for voice assist
    #[serde(default)]
    capabilities: Option<serde_json::Value>, // v13: integration-specific caps
    #[serde(default)]
    supported_features: Option<u64>,  // v13: bitmask
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum HaDisabledBy {
    User,
    Integration,
    ConfigEntry,
    Device,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum HaEntityCategory {
    Config,
    Diagnostic,
    #[serde(other)]
    Unknown,
}

fn map_disabled_by(v: Option<HaDisabledBy>) -> Option<DisabledBy> {
    v.and_then(|d| match d {
        HaDisabledBy::User => Some(DisabledBy::User),
        HaDisabledBy::Integration => Some(DisabledBy::Integration),
        HaDisabledBy::ConfigEntry => Some(DisabledBy::ConfigEntry),
        HaDisabledBy::Device => Some(DisabledBy::Device),
        HaDisabledBy::Unknown => None,
    })
}

fn map_entity_category(v: Option<HaEntityCategory>) -> Option<EntityCategory> {
    v.and_then(|c| match c {
        HaEntityCategory::Config => Some(EntityCategory::Config),
        HaEntityCategory::Diagnostic => Some(EntityCategory::Diagnostic),
        HaEntityCategory::Unknown => None,
    })
}

/// Read `core.entity_registry` from `path` and return HOMECORE entries.
///
/// Errors:
/// - `MigrateError::Io` if the file cannot be read
/// - `MigrateError::JsonParse` if the JSON is malformed
/// - `MigrateError::UnsupportedSchemaVersion` if minor_version is not 1–13
/// - `MigrateError::EntityId` if any `entity_id` string is invalid
pub fn read_entity_registry(path: &Path) -> Result<Vec<EntityEntry>, MigrateError> {
    let env = read_envelope(path)?;
    let file_str = path.display().to_string();
    v13::require_supported(&file_str, env.version, env.minor_version)?;

    let data: HaEntityRegistryData =
        serde_json::from_value(env.data).map_err(|e| MigrateError::JsonParse {
            path: file_str.clone(),
            source: e,
        })?;

    let mut entries = Vec::with_capacity(data.entities.len());
    for row in data.entities {
        let entity_id = EntityId::parse(&row.entity_id)?;
        entries.push(EntityEntry {
            entity_id,
            unique_id: row.unique_id,
            platform: row.platform,
            name: row.name,
            disabled_by: map_disabled_by(row.disabled_by),
            area_id: row.area_id,
            device_id: row.device_id,
            entity_category: map_entity_category(row.entity_category),
            config_entry_id: row.config_entry_id,
        });
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_fixture(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f
    }

    const FIXTURE_V13: &str = r#"{
        "version": 1,
        "minor_version": 13,
        "key": "core.entity_registry",
        "data": {
            "entities": [
                {
                    "entity_id": "light.kitchen",
                    "unique_id": "hue_lamp_42",
                    "platform": "hue",
                    "name": "Kitchen lamp",
                    "disabled_by": null,
                    "area_id": "kitchen",
                    "device_id": "abc123",
                    "entity_category": null,
                    "config_entry_id": "ce_001"
                },
                {
                    "entity_id": "sensor.bedroom_temperature",
                    "unique_id": "zigbee_temp_01",
                    "platform": "zha",
                    "name": null,
                    "disabled_by": "integration",
                    "area_id": null,
                    "device_id": "dev_02",
                    "entity_category": "diagnostic",
                    "config_entry_id": "ce_002",
                    "hidden_by": null,
                    "has_entity_name": true,
                    "original_name": "Temperature",
                    "aliases": ["room temp"],
                    "supported_features": 0
                }
            ],
            "deleted_entities": []
        }
    }"#;

    #[test]
    fn parses_v13_entity_registry() {
        let f = write_fixture(FIXTURE_V13);
        let entries = read_entity_registry(f.path()).unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn entity_fields_round_trip_correctly() {
        let f = write_fixture(FIXTURE_V13);
        let entries = read_entity_registry(f.path()).unwrap();
        let light = entries.iter().find(|e| e.entity_id.as_str() == "light.kitchen").unwrap();
        assert_eq!(light.unique_id.as_deref(), Some("hue_lamp_42"));
        assert_eq!(light.platform, "hue");
        assert_eq!(light.name.as_deref(), Some("Kitchen lamp"));
        assert!(light.disabled_by.is_none());
        assert_eq!(light.area_id.as_deref(), Some("kitchen"));
        assert_eq!(light.device_id.as_deref(), Some("abc123"));
        assert!(light.entity_category.is_none());
        assert_eq!(light.config_entry_id.as_deref(), Some("ce_001"));
    }

    #[test]
    fn disabled_by_maps_to_homecore() {
        let f = write_fixture(FIXTURE_V13);
        let entries = read_entity_registry(f.path()).unwrap();
        let sensor = entries
            .iter()
            .find(|e| e.entity_id.as_str() == "sensor.bedroom_temperature")
            .unwrap();
        assert_eq!(sensor.disabled_by, Some(DisabledBy::Integration));
        assert_eq!(sensor.entity_category, Some(EntityCategory::Diagnostic));
    }

    #[test]
    fn unknown_minor_version_raises_error() {
        let json = r#"{
            "version": 1, "minor_version": 99,
            "key": "core.entity_registry",
            "data": {"entities": [], "deleted_entities": []}
        }"#;
        let f = write_fixture(json);
        let err = read_entity_registry(f.path()).unwrap_err();
        assert!(
            matches!(err, MigrateError::UnsupportedSchemaVersion { minor_version: 99, .. }),
            "got: {err}"
        );
        let msg = err.to_string();
        assert!(msg.contains("minor_version=99"), "{msg}");
    }
}
