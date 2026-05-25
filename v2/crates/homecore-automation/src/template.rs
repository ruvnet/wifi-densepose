//! MiniJinja-based template environment with HA-compatible globals.
//!
//! ADR-129 §2.1 — P1 ships four HA globals: `states()`, `state_attr()`,
//! `is_state()`, `now()`. The `utcnow()`, `as_timestamp()`, `distance()`,
//! and `iif()` globals plus custom filters land in P2.

use std::sync::Arc;

use chrono::Utc;
use minijinja::{Environment, Value};

use homecore::{EntityId, StateMachine};

use crate::error::AutomationError;

/// MiniJinja environment pre-loaded with HA-compatible globals.
///
/// Constructed once per `AutomationEngine` and shared via `Arc`. The
/// globals close over an `Arc<StateMachine>` so every template render
/// sees the live current state.
pub struct TemplateEnvironment {
    env: Environment<'static>,
}

impl TemplateEnvironment {
    /// Build a new environment backed by the given state machine.
    pub fn new(states: Arc<StateMachine>) -> Self {
        let mut env = Environment::new();

        // --- states(entity_id) ---
        // Returns the current state string of an entity, or "unavailable".
        let states_sm = Arc::clone(&states);
        env.add_global(
            "states",
            Value::from_function(move |entity_id: String| -> String {
                EntityId::parse(&entity_id)
                    .ok()
                    .and_then(|eid| states_sm.get(&eid))
                    .map(|s| s.state.clone())
                    .unwrap_or_else(|| "unavailable".into())
            }),
        );

        // --- state_attr(entity_id, attribute) ---
        // Returns an attribute value as a JSON string, or empty string.
        let attr_sm = Arc::clone(&states);
        env.add_global(
            "state_attr",
            Value::from_function(move |entity_id: String, attr: String| -> String {
                EntityId::parse(&entity_id)
                    .ok()
                    .and_then(|eid| attr_sm.get(&eid))
                    .and_then(|s| s.attributes.get(&attr).cloned())
                    .map(|v| match v {
                        serde_json::Value::String(s) => s,
                        other => other.to_string(),
                    })
                    .unwrap_or_default()
            }),
        );

        // --- is_state(entity_id, state) ---
        // Returns true if the entity's current state matches the given value.
        let is_state_sm = Arc::clone(&states);
        env.add_global(
            "is_state",
            Value::from_function(move |entity_id: String, expected: String| -> bool {
                EntityId::parse(&entity_id)
                    .ok()
                    .and_then(|eid| is_state_sm.get(&eid))
                    .map(|s| s.state == expected)
                    .unwrap_or(false)
            }),
        );

        // --- now() ---
        // Returns the current UTC datetime as an ISO 8601 string.
        // HA returns a Python datetime; MiniJinja returns a string which
        // templates can further format with the `strftime` filter.
        env.add_global(
            "now",
            Value::from_function(|| -> String {
                Utc::now().format("%Y-%m-%dT%H:%M:%S%.6f+00:00").to_string()
            }),
        );

        Self { env }
    }

    /// Render a template string and return the string output.
    pub fn render(&self, template_str: &str) -> Result<String, AutomationError> {
        // Wrap bare expressions like `{{ states('light.kitchen') }}`
        // in a minimal template wrapper.
        let tmpl = self
            .env
            .template_from_str(template_str)
            .map_err(|e| AutomationError::TemplateRender(e.to_string()))?;
        tmpl.render(())
            .map_err(|e| AutomationError::TemplateRender(e.to_string()))
    }

    /// Render a template and interpret the output as a boolean.
    /// "true", "1", "yes", "on" → true. Everything else → false.
    pub fn render_bool(&self, template_str: &str) -> Result<bool, AutomationError> {
        let raw = self.render(template_str)?;
        let v = raw.trim().to_ascii_lowercase();
        Ok(matches!(v.as_str(), "true" | "1" | "yes" | "on"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::{Context, EntityId, StateMachine};
    use std::sync::Arc;

    fn sm_with(entity_id: &str, state: &str, attrs: serde_json::Value) -> Arc<StateMachine> {
        let sm = Arc::new(StateMachine::new());
        sm.set(EntityId::parse(entity_id).unwrap(), state, attrs, Context::new());
        sm
    }

    #[test]
    fn states_global_returns_current_state() {
        let sm = sm_with("light.kitchen", "on", serde_json::json!({}));
        let env = TemplateEnvironment::new(sm);
        let out = env.render("{{ states('light.kitchen') }}").unwrap();
        assert_eq!(out.trim(), "on");
    }

    #[test]
    fn states_global_unknown_entity_returns_unavailable() {
        let sm = Arc::new(StateMachine::new());
        let env = TemplateEnvironment::new(sm);
        let out = env.render("{{ states('sensor.unknown') }}").unwrap();
        assert_eq!(out.trim(), "unavailable");
    }

    #[test]
    fn state_attr_returns_attribute_value() {
        let sm = sm_with(
            "light.kitchen",
            "on",
            serde_json::json!({"brightness": 200}),
        );
        let env = TemplateEnvironment::new(sm);
        let out = env.render("{{ state_attr('light.kitchen', 'brightness') }}").unwrap();
        assert_eq!(out.trim(), "200");
    }

    #[test]
    fn is_state_global_true_when_matches() {
        let sm = sm_with("switch.fan", "on", serde_json::json!({}));
        let env = TemplateEnvironment::new(sm);
        let out = env.render("{{ is_state('switch.fan', 'on') }}").unwrap();
        assert_eq!(out.trim(), "true");
    }

    #[test]
    fn is_state_global_false_when_no_match() {
        let sm = sm_with("switch.fan", "off", serde_json::json!({}));
        let env = TemplateEnvironment::new(sm);
        let out = env.render("{{ is_state('switch.fan', 'on') }}").unwrap();
        assert_eq!(out.trim(), "false");
    }

    #[test]
    fn now_global_returns_timestamp_string() {
        let sm = Arc::new(StateMachine::new());
        let env = TemplateEnvironment::new(sm);
        let out = env.render("{{ now() }}").unwrap();
        // Should be an ISO 8601 datetime string containing 'T'
        assert!(out.contains('T'), "now() returned: {out}");
    }

    #[test]
    fn render_bool_true_values() {
        let sm = Arc::new(StateMachine::new());
        let env = TemplateEnvironment::new(sm);
        for tmpl in &["true", "1", "yes", "on"] {
            let result = env.render_bool(tmpl).unwrap();
            assert!(result, "expected true for: {tmpl}");
        }
    }

    #[test]
    fn render_bool_false_for_other() {
        let sm = Arc::new(StateMachine::new());
        let env = TemplateEnvironment::new(sm);
        assert!(!env.render_bool("false").unwrap());
        assert!(!env.render_bool("0").unwrap());
        assert!(!env.render_bool("off").unwrap());
    }
}
