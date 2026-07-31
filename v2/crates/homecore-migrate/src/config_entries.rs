//! Lossless conversion of HA `core.config_entries` into HOMECORE storage.
//!
//! HOMECORE cannot yet execute arbitrary HA integrations. Every source row is
//! therefore retained verbatim while portable identity/config fields are
//! projected for future plugin setup. Typed warnings make the unsupported
//! surface machine-readable instead of silently dropping it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    storage::{read_envelope, write_json_atomic, HaStorageEnvelope},
    MigrateError,
};

const SOURCE_KEY: &str = "core.config_entries";
pub const DESTINATION_KEY: &str = "homecore.config_entries";
const MAX_SUPPORTED_MINOR: u32 = 4;
const HOMECORE_CONFIG_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HomeCoreConfigEntry {
    pub entry_id: String,
    pub domain: String,
    pub title: String,
    #[serde(default)]
    pub data: serde_json::Value,
    /// Exact HA row, including unsupported and future fields.
    pub source: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "code", rename_all = "snake_case")]
pub enum MigrationWarning {
    UnsupportedDomain { entry_id: String, domain: String },
    UnsupportedField { entry_id: String, field: String },
    UnsupportedRootField { field: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HomeCoreConfigData {
    pub source_version: u32,
    pub source_minor_version: u32,
    /// Exact non-entry fields from the HA `data` object.
    #[serde(default)]
    pub source_extra: BTreeMap<String, serde_json::Value>,
    pub entries: Vec<HomeCoreConfigEntry>,
    pub warnings: Vec<MigrationWarning>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HomeCoreConfigEnvelope {
    pub version: u32,
    pub minor_version: u32,
    pub key: String,
    pub data: HomeCoreConfigData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfigEntriesSummary {
    pub count: usize,
    pub domains: Vec<String>,
    pub warning_count: usize,
    pub destination: Option<PathBuf>,
}

fn validate_source(env: &HaStorageEnvelope, path: &Path) -> Result<(), MigrateError> {
    if env.version != 1 || env.minor_version > MAX_SUPPORTED_MINOR {
        return Err(MigrateError::UnsupportedSchemaVersion {
            file: path.display().to_string(),
            version: env.version,
            minor_version: env.minor_version,
        });
    }
    if env.key != SOURCE_KEY {
        return Err(MigrateError::UnexpectedStorageKey {
            path: path.display().to_string(),
            expected: SOURCE_KEY.to_owned(),
            actual: env.key.clone(),
        });
    }
    Ok(())
}

pub fn convert_config_entries(path: &Path) -> Result<HomeCoreConfigEnvelope, MigrateError> {
    let env = read_envelope(path)?;
    validate_source(&env, path)?;
    let entries = env
        .data
        .get("entries")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| MigrateError::MissingField {
            field: "entries".to_owned(),
            context: path.display().to_string(),
        })?;
    let source_extra: BTreeMap<String, serde_json::Value> = env
        .data
        .as_object()
        .into_iter()
        .flat_map(|object| object.iter())
        .filter(|(field, _)| field.as_str() != "entries")
        .map(|(field, value)| (field.clone(), value.clone()))
        .collect();

    let portable = BTreeSet::from(["entry_id", "domain", "title", "data"]);
    let mut converted = Vec::with_capacity(entries.len());
    let mut warnings = source_extra
        .keys()
        .map(|field| MigrationWarning::UnsupportedRootField {
            field: field.clone(),
        })
        .collect::<Vec<_>>();
    for (index, source) in entries.iter().enumerate() {
        let object = source
            .as_object()
            .ok_or_else(|| MigrateError::MissingField {
                field: "object row".to_owned(),
                context: format!("{} data.entries[{index}]", path.display()),
            })?;
        let string_field = |field: &str| -> Result<String, MigrateError> {
            object
                .get(field)
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| MigrateError::MissingField {
                    field: field.to_owned(),
                    context: format!("{} data.entries[{index}]", path.display()),
                })
        };
        let entry_id = string_field("entry_id")?;
        let domain = string_field("domain")?;
        let title = object
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(&domain)
            .to_owned();
        let data = object
            .get("data")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));

        // No HA domain is implicitly claimed executable by HOMECORE. The
        // source is retained so a matching plugin can consume it later.
        warnings.push(MigrationWarning::UnsupportedDomain {
            entry_id: entry_id.clone(),
            domain: domain.clone(),
        });
        for field in object
            .keys()
            .filter(|field| !portable.contains(field.as_str()))
        {
            warnings.push(MigrationWarning::UnsupportedField {
                entry_id: entry_id.clone(),
                field: field.clone(),
            });
        }
        converted.push(HomeCoreConfigEntry {
            entry_id,
            domain,
            title,
            data,
            source: source.clone(),
        });
    }
    warnings.sort_by_key(|warning| serde_json::to_string(warning).unwrap_or_default());

    Ok(HomeCoreConfigEnvelope {
        version: HOMECORE_CONFIG_VERSION,
        minor_version: 0,
        key: DESTINATION_KEY.to_owned(),
        data: HomeCoreConfigData {
            source_version: env.version,
            source_minor_version: env.minor_version,
            source_extra,
            entries: converted,
            warnings,
        },
    })
}

pub fn write_config_entries(
    storage_dir: &Path,
    envelope: &HomeCoreConfigEnvelope,
) -> Result<PathBuf, MigrateError> {
    write_config_entries_with(storage_dir, envelope, false)
}

/// As [`write_config_entries`], but `force = true` atomically replaces an
/// existing destination instead of refusing — the escape hatch for
/// re-running an import after fixing a bad source row.
pub fn write_config_entries_with(
    storage_dir: &Path,
    envelope: &HomeCoreConfigEnvelope,
    force: bool,
) -> Result<PathBuf, MigrateError> {
    let target = storage_dir.join(DESTINATION_KEY);
    write_json_atomic(&target, envelope, force)
}

pub fn read_homecore_config_entries(path: &Path) -> Result<HomeCoreConfigEnvelope, MigrateError> {
    let raw = std::fs::read_to_string(path).map_err(|source| MigrateError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let envelope: HomeCoreConfigEnvelope =
        serde_json::from_str(&raw).map_err(|source| MigrateError::JsonParse {
            path: path.display().to_string(),
            source,
        })?;
    if envelope.version != HOMECORE_CONFIG_VERSION || envelope.minor_version != 0 {
        return Err(MigrateError::UnsupportedSchemaVersion {
            file: path.display().to_string(),
            version: envelope.version,
            minor_version: envelope.minor_version,
        });
    }
    if envelope.key != DESTINATION_KEY {
        return Err(MigrateError::UnexpectedStorageKey {
            path: path.display().to_string(),
            expected: DESTINATION_KEY.to_owned(),
            actual: envelope.key,
        });
    }
    Ok(envelope)
}

pub fn inspect_config_entries(path: &Path) -> Result<ConfigEntriesSummary, MigrateError> {
    let converted = convert_config_entries(path)?;
    let domains: BTreeMap<&str, usize> =
        converted
            .data
            .entries
            .iter()
            .fold(BTreeMap::new(), |mut map, entry| {
                *map.entry(entry.domain.as_str()).or_default() += 1;
                map
            });
    Ok(ConfigEntriesSummary {
        count: converted.data.entries.len(),
        domains: domains.keys().map(|value| (*value).to_owned()).collect(),
        warning_count: converted.data.warnings.len(),
        destination: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const FIXTURE: &str = r#"{
      "version":1,"minor_version":2,"key":"core.config_entries",
      "data":{"future_root":{"kept":true},"entries":[{
        "domain":"future_hub","entry_id":"ce_001","title":"Future Hub",
        "source":"user","state":"loaded","data":{"host":"10.0.0.2"},
        "options":{"scan":true},"future_field":{"nested":[1,2,3]}
      }]}
    }"#;

    fn fixture() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(FIXTURE.as_bytes()).unwrap();
        file
    }

    #[test]
    fn unknown_domain_and_fields_are_lossless_with_typed_warnings() {
        let source = fixture();
        let converted = convert_config_entries(source.path()).unwrap();
        assert_eq!(
            converted.data.entries[0].source["future_field"]["nested"][2],
            3
        );
        assert_eq!(converted.data.source_extra["future_root"]["kept"], true);
        assert!(converted.data.warnings.iter().any(|warning| matches!(
            warning,
            MigrationWarning::UnsupportedDomain { domain, .. } if domain == "future_hub"
        )));
        assert!(converted.data.warnings.iter().any(|warning| matches!(
            warning,
            MigrationWarning::UnsupportedField { field, .. } if field == "future_field"
        )));

        let dir = tempfile::tempdir().unwrap();
        let path = write_config_entries(dir.path(), &converted).unwrap();
        let restored = read_homecore_config_entries(&path).unwrap();
        assert_eq!(restored, converted);
    }

    #[test]
    fn unknown_destination_version_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(DESTINATION_KEY);
        std::fs::write(
            &path,
            r#"{"version":99,"minor_version":0,"key":"homecore.config_entries","data":{"source_version":1,"source_minor_version":1,"entries":[],"warnings":[]}}"#,
        )
        .unwrap();
        assert!(matches!(
            read_homecore_config_entries(&path),
            Err(MigrateError::UnsupportedSchemaVersion { version: 99, .. })
        ));
    }

    #[test]
    fn malformed_entry_is_an_error_not_a_partial_write() {
        let mut source = NamedTempFile::new().unwrap();
        source
            .write_all(
                br#"{"version":1,"minor_version":1,"key":"core.config_entries","data":{"entries":[{"domain":"hue"}]}}"#,
            )
            .unwrap();
        assert!(matches!(
            convert_config_entries(source.path()),
            Err(MigrateError::MissingField { .. })
        ));
    }
}
