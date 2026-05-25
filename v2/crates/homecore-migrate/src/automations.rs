//! Parser for `automations.yaml`.
//!
//! P1: reads the YAML, validates the top-level structure, and emits a count
//! plus the list of automation IDs/aliases.
//!
//! Conversion to `homecore-automation` YAML format is deferred to P2.
//!
//! HA `automations.yaml` is a YAML sequence of automation objects:
//!
//! ```yaml
//! - id: '1620000000001'
//!   alias: "Turn on lights at sunset"
//!   trigger: [...]
//!   condition: []
//!   action: [...]
//! - id: '1620000000002'
//!   alias: "Turn off lights at midnight"
//!   trigger: [...]
//!   action: [...]
//! ```

use std::path::Path;

use serde::Deserialize;

use crate::MigrateError;

/// Diagnostic summary of `automations.yaml`.
#[derive(Clone, Debug)]
pub struct AutomationsSummary {
    pub count: usize,
    /// `(id, alias)` pairs. `id` defaults to an empty string if absent.
    pub automations: Vec<AutomationIdent>,
}

/// Minimal identifying info for a single automation.
#[derive(Clone, Debug)]
pub struct AutomationIdent {
    pub id: String,
    pub alias: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HaAutomationRow {
    #[serde(default)]
    id: String,
    #[serde(default)]
    alias: Option<String>,
    // All other fields (trigger, condition, action, mode, etc.) ignored in P1.
    #[allow(dead_code)]
    #[serde(flatten)]
    _rest: serde_json::Value,
}

/// Read `automations.yaml` from `path` and return a summary.
pub fn read_automations(path: &Path) -> Result<AutomationsSummary, MigrateError> {
    let raw = std::fs::read_to_string(path).map_err(|e| MigrateError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    if raw.trim().is_empty() {
        return Ok(AutomationsSummary { count: 0, automations: vec![] });
    }

    let rows: Vec<HaAutomationRow> =
        serde_yaml::from_str(&raw).map_err(|e| MigrateError::YamlParse {
            path: path.display().to_string(),
            source: e,
        })?;

    let automations = rows
        .iter()
        .map(|r| AutomationIdent { id: r.id.clone(), alias: r.alias.clone() })
        .collect::<Vec<_>>();

    Ok(AutomationsSummary { count: rows.len(), automations })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    const FIXTURE: &str = r#"
- id: '1620000000001'
  alias: "Turn on lights at sunset"
  trigger:
    - platform: sun
      event: sunset
  action:
    - service: light.turn_on
      target:
        entity_id: light.living_room

- id: '1620000000002'
  alias: "Turn off lights at midnight"
  trigger:
    - platform: time
      at: "00:00:00"
  action:
    - service: light.turn_off
      target:
        entity_id: all
"#;

    #[test]
    fn parses_automation_count_and_ids() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(FIXTURE.as_bytes()).unwrap();
        let summary = read_automations(f.path()).unwrap();
        assert_eq!(summary.count, 2);
        assert_eq!(summary.automations.len(), 2);
        assert_eq!(summary.automations[0].id, "1620000000001");
        assert_eq!(
            summary.automations[0].alias.as_deref(),
            Some("Turn on lights at sunset")
        );
        assert_eq!(summary.automations[1].id, "1620000000002");
    }

    #[test]
    fn empty_automations_returns_zero_count() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(b"").unwrap();
        let summary = read_automations(f.path()).unwrap();
        assert_eq!(summary.count, 0);
    }
}
