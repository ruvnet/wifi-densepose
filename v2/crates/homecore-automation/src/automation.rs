//! `Automation` — the parsed representation of one HA automation YAML block.
//!
//! Mirrors HA's `AutomationConfig` / `AutomationEntity`. Deserialized from
//! YAML via serde; validated at construction time by the engine.

use serde::{Deserialize, Serialize};

use crate::action::Action;
use crate::condition::Condition;
use crate::trigger::Trigger;

/// Script run mode. Mirrors HA's `ScriptRunMode` (`script/__init__.py`).
///
/// Controls what happens when a second trigger fires while the automation
/// is already running.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Only one instance runs at a time. If already running, the new
    /// trigger is silently dropped (HA default).
    #[default]
    Single,
    /// Kill the running instance and start a fresh one.
    Restart,
    /// Queue new triggers; execute sequentially when the prior run finishes.
    Queued,
    /// Allow unlimited concurrent runs.
    Parallel,
    /// Same as `Single` but also skips the first trigger (rarely used).
    IgnoreFirst,
}

/// A parsed automation. Cheap to clone — all heaps are `Arc`-free vecs of
/// enums; the engine holds `Arc<Automation>` copies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Automation {
    /// Unique identifier. HA auto-assigns a 32-char hex ID if omitted.
    pub id: String,

    /// Human-readable alias shown in the HA UI.
    #[serde(default)]
    pub alias: Option<String>,

    /// Optional free-text description.
    #[serde(default)]
    pub description: Option<String>,

    /// Whether the automation is enabled. Disabled automations are loaded
    /// but their triggers are not evaluated.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Script run mode.
    #[serde(default)]
    pub mode: RunMode,

    /// Maximum concurrent runs when mode is `Queued` or `Parallel`.
    #[serde(default)]
    pub max: Option<usize>,

    /// One or more trigger definitions. At least one must be present.
    pub trigger: Vec<Trigger>,

    /// Optional conditions — all must pass before actions run.
    #[serde(default)]
    pub condition: Vec<Condition>,

    /// Action sequence to execute when triggered + conditions pass.
    pub action: Vec<Action>,
}

fn default_enabled() -> bool {
    true
}

impl Automation {
    /// Minimal constructor for tests.
    pub fn new(
        id: impl Into<String>,
        triggers: Vec<Trigger>,
        actions: Vec<Action>,
    ) -> Self {
        Self {
            id: id.into(),
            alias: None,
            description: None,
            enabled: true,
            mode: RunMode::Single,
            max: None,
            trigger: triggers,
            condition: vec![],
            action: actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trigger::Trigger;

    #[test]
    fn run_mode_defaults_to_single() {
        let a = Automation::new("test.1", vec![Trigger::Event { event_type: "t".into() }], vec![]);
        assert_eq!(a.mode, RunMode::Single);
    }

    #[test]
    fn automation_enabled_by_default() {
        let a = Automation::new("test.2", vec![], vec![]);
        assert!(a.enabled);
    }

    #[test]
    fn run_mode_roundtrip_yaml() {
        // RunMode is a plain string enum; deserialize from a bare YAML string.
        let mode: RunMode = serde_yaml::from_str("restart").unwrap();
        assert_eq!(mode, RunMode::Restart);
    }
}
