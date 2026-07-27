//! Bounded startup restoration for registries and latest recorder states.

use std::path::Path;

use homecore::{DeviceEntry, EntityEntry, HomeCore};
use homecore_migrate::storage::read_envelope;
use homecore_recorder::{Recorder, RestoreWarning};
use serde::de::DeserializeOwned;

pub const MAX_REGISTRY_ENTRIES: usize = 100_000;
pub const MAX_REGISTRY_FILE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestorePhase {
    EntityRegistry,
    DeviceRegistry,
    States,
}

#[derive(Debug, Default)]
pub struct StartupRestoreReport {
    pub entity_entries: usize,
    pub device_entries: usize,
    pub states: usize,
    pub warnings: Vec<String>,
    pub truncated: bool,
    pub phases: Vec<RestorePhase>,
}

pub async fn restore_startup(
    homecore: &HomeCore,
    recorder: Option<&Recorder>,
    storage_dir: &Path,
    limit: usize,
) -> StartupRestoreReport {
    let limit = limit.min(MAX_REGISTRY_ENTRIES);
    let mut report = StartupRestoreReport::default();

    report.phases.push(RestorePhase::EntityRegistry);
    let entities_path = storage_dir.join("core.entity_registry");
    if entities_path.exists() {
        match load_rows::<EntityEntry>(&entities_path, "entities", limit, |entry| {
            entry.entity_id.as_str().to_owned()
        }) {
            Ok(batch) => {
                report.truncated |= batch.truncated;
                report.warnings.extend(batch.warnings);
                for entry in batch.rows {
                    homecore.entities().register(entry).await;
                    report.entity_entries += 1;
                }
            }
            Err(error) => report.warnings.push(error),
        }
    }

    report.phases.push(RestorePhase::DeviceRegistry);
    let devices_path = storage_dir.join("core.device_registry");
    if devices_path.exists() {
        match load_rows::<DeviceEntry>(&devices_path, "devices", limit, |entry| entry.id.clone()) {
            Ok(batch) => {
                report.truncated |= batch.truncated;
                report.warnings.extend(batch.warnings);
                for entry in batch.rows {
                    homecore.devices().register(entry).await;
                    report.device_entries += 1;
                }
            }
            Err(error) => report.warnings.push(error),
        }
    }

    report.phases.push(RestorePhase::States);
    if let Some(recorder) = recorder {
        match recorder.restore_latest(homecore.states(), limit).await {
            Ok(state_report) => {
                report.states = state_report.restored;
                report.truncated |= state_report.truncated;
                report
                    .warnings
                    .extend(state_report.warnings.into_iter().map(format_state_warning));
            }
            Err(error) => report
                .warnings
                .push(format!("state restore failed: {error}")),
        }
    }
    report
}

struct RowBatch<T> {
    rows: Vec<T>,
    warnings: Vec<String>,
    truncated: bool,
}

fn load_rows<T: DeserializeOwned>(
    path: &Path,
    field: &str,
    limit: usize,
    key: impl Fn(&T) -> String,
) -> Result<RowBatch<T>, String> {
    let file_size = std::fs::metadata(path)
        .map_err(|error| format!("{}: {error}", path.display()))?
        .len();
    if file_size > MAX_REGISTRY_FILE_BYTES {
        return Err(format!(
            "{}: registry file is {file_size} bytes, exceeding the {} byte startup bound",
            path.display(),
            MAX_REGISTRY_FILE_BYTES
        ));
    }
    let envelope = read_envelope(path).map_err(|error| error.to_string())?;
    if envelope.version != 1 || envelope.minor_version > 13 {
        return Err(format!(
            "{}: unsupported registry version {}.{}",
            path.display(),
            envelope.version,
            envelope.minor_version
        ));
    }
    let values = envelope
        .data
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{}: missing data.{field} array", path.display()))?;
    let scan_limit = limit.saturating_add(1).min(MAX_REGISTRY_ENTRIES + 1);
    let mut rows = Vec::with_capacity(values.len().min(scan_limit));
    let mut warnings = Vec::new();
    for (index, value) in values.iter().take(scan_limit).enumerate() {
        match serde_json::from_value::<T>(value.clone()) {
            Ok(row) => rows.push(row),
            Err(error) => warnings.push(format!(
                "{}: isolated malformed data.{field}[{index}]: {error}",
                path.display()
            )),
        }
    }
    rows.sort_by_key(&key);
    let truncated = values.len() > limit || rows.len() > limit;
    rows.truncate(limit);
    Ok(RowBatch {
        rows,
        warnings,
        truncated,
    })
}

fn format_state_warning(warning: RestoreWarning) -> String {
    format!("isolated malformed recorder row: {warning:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::{Context, EntityId};
    use homecore_recorder::Recorder;

    #[tokio::test]
    async fn restores_registries_before_states_and_isolates_bad_rows() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("core.entity_registry"),
            r#"{"version":1,"minor_version":13,"key":"core.entity_registry","data":{"entities":[
              {"entity_id":"sensor.good","unique_id":null,"platform":"test","name":null,"disabled_by":null,"area_id":null,"device_id":"dev1","entity_category":null,"config_entry_id":null},
              {"entity_id":"INVALID","platform":"bad"}
            ]}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("core.device_registry"),
            r#"{"version":1,"minor_version":13,"key":"core.device_registry","data":{"devices":[
              {"id":"dev1","config_entries":[],"identifiers":[],"connections":[],"manufacturer":null,"model":null,"model_id":null,"name":"Device","name_by_user":null,"sw_version":null,"hw_version":null,"serial_number":null,"via_device_id":null,"area_id":null,"entry_type":null,"disabled_by":null,"configuration_url":null,"labels":[],"primary_config_entry":null}
            ]}}"#,
        )
        .unwrap();
        let recorder = Recorder::open("sqlite::memory:").await.unwrap();
        let source = HomeCore::new();
        source.states().set(
            EntityId::parse("sensor.good").unwrap(),
            "on",
            serde_json::json!({"restorable": true}),
            Context::new(),
        );
        let mut receiver = source.states().subscribe();
        // Force one recorder row from a real state event.
        source.states().set(
            EntityId::parse("sensor.good").unwrap(),
            "off",
            serde_json::json!({"restorable": true}),
            Context::new(),
        );
        recorder
            .record_state(&receiver.recv().await.unwrap())
            .await
            .unwrap();

        let target = HomeCore::new();
        let report =
            restore_startup(&target, Some(&recorder), dir.path(), MAX_REGISTRY_ENTRIES).await;
        assert_eq!(
            report.phases,
            vec![
                RestorePhase::EntityRegistry,
                RestorePhase::DeviceRegistry,
                RestorePhase::States
            ]
        );
        assert_eq!(report.entity_entries, 1);
        assert_eq!(report.device_entries, 1);
        assert_eq!(report.states, 1);
        assert!(!report.warnings.is_empty());
        let state = target
            .states()
            .get(&EntityId::parse("sensor.good").unwrap())
            .unwrap();
        assert!(state.context.is_restoration());
    }
}
