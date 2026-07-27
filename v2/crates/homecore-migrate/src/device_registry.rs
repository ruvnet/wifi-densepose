//! Conversion for HA `core.device_registry` schema v1/minor 1-13.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use homecore::DeviceEntry;
use serde::Deserialize;

use crate::{
    storage::{read_envelope, write_json_atomic},
    storage_format::v13,
    MigrateError,
};

const FILE_KEY: &str = "core.device_registry";

#[derive(Debug, Deserialize)]
struct HaDeviceRegistryData {
    devices: Vec<HaDeviceRow>,
    #[serde(default)]
    deleted_devices: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct HaDeviceRow {
    id: String,
    #[serde(default)]
    config_entries: HashSet<String>,
    #[serde(default)]
    identifiers: HashSet<(String, String)>,
    #[serde(default)]
    connections: HashSet<(String, String)>,
    #[serde(default)]
    manufacturer: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    name_by_user: Option<String>,
    #[serde(default)]
    sw_version: Option<String>,
    #[serde(default)]
    hw_version: Option<String>,
    #[serde(default)]
    serial_number: Option<String>,
    #[serde(default)]
    via_device_id: Option<String>,
    #[serde(default)]
    area_id: Option<String>,
    #[serde(default)]
    entry_type: Option<String>,
    #[serde(default)]
    disabled_by: Option<String>,
    #[serde(default)]
    configuration_url: Option<String>,
    #[serde(default)]
    labels: HashSet<String>,
    #[serde(default)]
    primary_config_entry: Option<String>,
    #[serde(default, flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

impl From<HaDeviceRow> for DeviceEntry {
    fn from(row: HaDeviceRow) -> Self {
        Self {
            id: row.id,
            config_entries: row.config_entries,
            identifiers: row.identifiers,
            connections: row.connections,
            manufacturer: row.manufacturer,
            model: row.model,
            model_id: row.model_id,
            name: row.name,
            name_by_user: row.name_by_user,
            sw_version: row.sw_version,
            hw_version: row.hw_version,
            serial_number: row.serial_number,
            via_device_id: row.via_device_id,
            area_id: row.area_id,
            entry_type: row.entry_type,
            disabled_by: row.disabled_by,
            configuration_url: row.configuration_url,
            labels: row.labels,
            primary_config_entry: row.primary_config_entry,
            extra: row.extra,
        }
    }
}

pub fn read_device_registry(path: &Path) -> Result<Vec<DeviceEntry>, MigrateError> {
    let env = read_envelope(path)?;
    let file = path.display().to_string();
    v13::require_supported(&file, env.version, env.minor_version)?;
    if env.key != FILE_KEY {
        return Err(MigrateError::UnexpectedStorageKey {
            path: file,
            expected: FILE_KEY.to_owned(),
            actual: env.key,
        });
    }
    let data: HaDeviceRegistryData =
        serde_json::from_value(env.data).map_err(|source| MigrateError::JsonParse {
            path: path.display().to_string(),
            source,
        })?;
    let _preserved_tombstone_count = data.deleted_devices.len();
    Ok(data.devices.into_iter().map(DeviceEntry::from).collect())
}

pub fn write_device_registry(
    storage_dir: &Path,
    devices: &[DeviceEntry],
) -> Result<PathBuf, MigrateError> {
    write_device_registry_with(storage_dir, devices, false)
}

/// As [`write_device_registry`], but `force = true` atomically replaces an
/// existing destination instead of refusing — the escape hatch for
/// re-running an import after fixing a bad source row.
pub fn write_device_registry_with(
    storage_dir: &Path,
    devices: &[DeviceEntry],
    force: bool,
) -> Result<PathBuf, MigrateError> {
    let target = storage_dir.join(FILE_KEY);
    let payload = serde_json::json!({
        "version": 1,
        "minor_version": 13,
        "key": FILE_KEY,
        "data": {
            "devices": devices,
            "deleted_devices": []
        }
    });
    write_json_atomic(&target, &payload, force)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const FIXTURE: &str = r#"{
      "version":1,"minor_version":13,"key":"core.device_registry",
      "data":{"devices":[{
        "id":"dev_abc","config_entries":["ce_001"],
        "manufacturer":"Philips","model":"Hue Bridge","model_id":"BSB002",
        "name":"Hue","name_by_user":"Downstairs Hue",
        "sw_version":"1.2","hw_version":"3","serial_number":"SN42",
        "identifiers":[["hue","001788FFFE3D4B13"]],
        "connections":[["mac","00:17:88:ff:fe:3d:4b:13"]],
        "via_device_id":"gateway","area_id":"living_room",
        "entry_type":"service","disabled_by":"user",
        "configuration_url":"http://hue.local","labels":["lighting"],
        "primary_config_entry":"ce_001","created_at":1735689600.0
      }],"deleted_devices":[]}
    }"#;

    #[test]
    fn all_supported_fields_round_trip() {
        let mut source = NamedTempFile::new().unwrap();
        source.write_all(FIXTURE.as_bytes()).unwrap();
        let devices = read_device_registry(source.path()).unwrap();
        let destination = tempfile::tempdir().unwrap();
        let path = write_device_registry(destination.path(), &devices).unwrap();
        let imported = read_device_registry(&path).unwrap();
        assert_eq!(imported, devices);
        assert_eq!(imported[0].serial_number.as_deref(), Some("SN42"));
        assert!(imported[0].labels.contains("lighting"));
        assert_eq!(imported[0].extra["created_at"], 1735689600.0);
    }

    #[test]
    fn destination_is_never_overwritten() {
        let mut source = NamedTempFile::new().unwrap();
        source.write_all(FIXTURE.as_bytes()).unwrap();
        let devices = read_device_registry(source.path()).unwrap();
        let destination = tempfile::tempdir().unwrap();
        write_device_registry(destination.path(), &devices).unwrap();
        let error = write_device_registry(destination.path(), &[]).unwrap_err();
        assert!(error.to_string().contains("refusing to overwrite"));
        assert_eq!(
            read_device_registry(&destination.path().join(FILE_KEY))
                .unwrap()
                .len(),
            1
        );
    }

    /// `force = true` is the operator escape hatch: re-running an import
    /// after fixing a bad source row (or re-importing after further HA-side
    /// changes) must not require manually deleting prior output first.
    #[test]
    fn force_overwrites_existing_destination() {
        let mut source = NamedTempFile::new().unwrap();
        source.write_all(FIXTURE.as_bytes()).unwrap();
        let devices = read_device_registry(source.path()).unwrap();
        let destination = tempfile::tempdir().unwrap();
        write_device_registry(destination.path(), &devices).unwrap();
        write_device_registry_with(destination.path(), &[], true).unwrap();
        assert_eq!(
            read_device_registry(&destination.path().join(FILE_KEY))
                .unwrap()
                .len(),
            0
        );
    }
}
