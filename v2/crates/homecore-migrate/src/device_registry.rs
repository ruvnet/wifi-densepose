//! Parser for `core.device_registry` (HA storage schema v1, minor_version 1–13).
//!
//! P1: deserializes the envelope and returns `Vec<DeviceImport>`.
//! HOMECORE's device registry isn't fully wired yet (ADR-127 §2.5 deferred
//! to P2), so `DeviceImport` is a staging type for the future hand-off.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{storage::read_envelope, storage_format::v13, MigrateError};

/// Staging type for a device imported from HA. Not yet wired to HOMECORE's
/// device registry (ADR-127 §2.5 — deferred to P2).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceImport {
    pub id: String,
    pub config_entries: Vec<String>,
    #[serde(default)]
    pub manufacturer: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    /// `identifiers` — list of `[integration, id]` pairs. Preserved as raw
    /// JSON for P2 consumption; not yet mapped to HOMECORE DeviceEntry.
    #[serde(default)]
    pub identifiers: Vec<Vec<String>>,
    #[serde(default)]
    pub connections: Vec<Vec<String>>,
    #[serde(default)]
    pub via_device_id: Option<String>,
    #[serde(default)]
    pub area_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HaDeviceRegistryData {
    devices: Vec<DeviceImport>,
    /// Deleted device tombstones — ignored in P1.
    #[serde(default)]
    #[allow(dead_code)]
    deleted_devices: Vec<serde_json::Value>,
}

/// Read `core.device_registry` from `path` and return the raw import list.
pub fn read_device_registry(path: &Path) -> Result<Vec<DeviceImport>, MigrateError> {
    let env = read_envelope(path)?;
    let file_str = path.display().to_string();
    v13::require_supported(&file_str, env.version, env.minor_version)?;

    let data: HaDeviceRegistryData =
        serde_json::from_value(env.data).map_err(|e| MigrateError::JsonParse {
            path: file_str,
            source: e,
        })?;
    Ok(data.devices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const FIXTURE: &str = r#"{
        "version": 1,
        "minor_version": 13,
        "key": "core.device_registry",
        "data": {
            "devices": [
                {
                    "id": "dev_abc",
                    "config_entries": ["ce_001"],
                    "manufacturer": "Philips",
                    "model": "Hue Bridge",
                    "name": "Philips Hue Bridge",
                    "identifiers": [["hue", "001788FFFE3D4B13"]],
                    "connections": [["mac", "00:17:88:ff:fe:3d:4b:13"]],
                    "via_device_id": null,
                    "area_id": null
                }
            ],
            "deleted_devices": []
        }
    }"#;

    #[test]
    fn parses_device_registry() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(FIXTURE.as_bytes()).unwrap();
        let devices = read_device_registry(f.path()).unwrap();
        assert_eq!(devices.len(), 1);
        let d = &devices[0];
        assert_eq!(d.id, "dev_abc");
        assert_eq!(d.manufacturer.as_deref(), Some("Philips"));
        assert_eq!(d.identifiers, vec![vec!["hue", "001788FFFE3D4B13"]]);
    }
}
