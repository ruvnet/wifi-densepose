//! Parser for `core.config_entries` (HA storage schema v1, minor_version varies).
//!
//! Per ADR-134 §6 Q5, `.storage/core.config_entries` format is undocumented
//! and version-gated. P1 reads the envelope and emits:
//!   - count of config entries
//!   - list of integration domains represented
//!
//! Conversion to HOMECORE plugin manifests is P2.
//!
//! Note: `config_entries` uses a different `minor_version` track from
//! `entity_registry`. As of HA 2025.1 it is typically minor_version=1 or 2.
//! We accept any minor_version ≤ MAX_SUPPORTED_MINOR and hard-error above it.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{storage::read_envelope, MigrateError};

/// Maximum `minor_version` we claim to understand for config_entries.
const MAX_SUPPORTED_MINOR: u32 = 4;

/// Diagnostic summary produced by P1 inspection.
#[derive(Clone, Debug, Serialize)]
pub struct ConfigEntriesSummary {
    pub count: usize,
    pub domains: Vec<String>,
}

/// Minimal fields we read from each config-entry row.
#[derive(Debug, Deserialize)]
struct HaConfigEntryRow {
    domain: String,
    #[allow(dead_code)]
    entry_id: String,
    /// Title shown in HA UI (informational only in P1).
    #[serde(default)]
    #[allow(dead_code)]
    title: Option<String>,
    /// Source of the entry: "user" | "discovery" | "import" etc.
    #[serde(default)]
    #[allow(dead_code)]
    source: Option<String>,
    /// State: "loaded" | "setup_error" etc.
    #[serde(default)]
    #[allow(dead_code)]
    state: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HaConfigEntriesData {
    entries: Vec<HaConfigEntryRow>,
}

/// Read `core.config_entries` from `path` and return a diagnostic summary.
pub fn inspect_config_entries(path: &Path) -> Result<ConfigEntriesSummary, MigrateError> {
    let env = read_envelope(path)?;
    let file_str = path.display().to_string();

    // config_entries has version=1 and minor_version in 1..MAX_SUPPORTED_MINOR.
    if env.version != 1 || env.minor_version > MAX_SUPPORTED_MINOR {
        return Err(MigrateError::UnsupportedSchemaVersion {
            file: file_str.clone(),
            version: env.version,
            minor_version: env.minor_version,
        });
    }

    let data: HaConfigEntriesData =
        serde_json::from_value(env.data).map_err(|e| MigrateError::JsonParse {
            path: file_str,
            source: e,
        })?;

    let mut domains: Vec<String> = data.entries.iter().map(|e| e.domain.clone()).collect();
    domains.sort();
    domains.dedup();

    Ok(ConfigEntriesSummary {
        count: data.entries.len(),
        domains,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const FIXTURE: &str = r#"{
        "version": 1,
        "minor_version": 1,
        "key": "core.config_entries",
        "data": {
            "entries": [
                {"domain": "hue", "entry_id": "ce_001", "title": "Philips Hue", "source": "user", "state": "loaded"},
                {"domain": "zha",  "entry_id": "ce_002", "title": "ZHA",         "source": "user", "state": "loaded"},
                {"domain": "hue",  "entry_id": "ce_003", "title": "Hue 2",       "source": "user", "state": "setup_error"}
            ]
        }
    }"#;

    #[test]
    fn inspect_emits_count_and_domains() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(FIXTURE.as_bytes()).unwrap();
        let summary = inspect_config_entries(f.path()).unwrap();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.domains, vec!["hue", "zha"]);
    }

    #[test]
    fn unknown_minor_version_hard_errors() {
        let json = r#"{
            "version": 1, "minor_version": 99,
            "key": "core.config_entries",
            "data": {"entries": []}
        }"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        let err = inspect_config_entries(f.path()).unwrap_err();
        assert!(matches!(
            err,
            MigrateError::UnsupportedSchemaVersion { minor_version: 99, .. }
        ));
    }
}
