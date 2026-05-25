//! `Condition` enum + async evaluation.
//!
//! Mirrors HA's 7 condition types. P1 ships: `state`, `numeric_state`,
//! `template`, `and`, `or`, `not`. Time/zone/sun/device land in P2.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use homecore::{EntityId, StateMachine};

use crate::template::TemplateEnvironment;

/// Context passed to condition evaluation. Holds a snapshot of the state
/// machine and the optional template evaluator.
#[derive(Clone)]
pub struct EvalContext {
    pub states: Arc<StateMachine>,
    pub template_env: Option<Arc<TemplateEnvironment>>,
}

impl EvalContext {
    pub fn new(states: Arc<StateMachine>) -> Self {
        Self { states, template_env: None }
    }

    pub fn with_templates(states: Arc<StateMachine>, env: Arc<TemplateEnvironment>) -> Self {
        Self { states, template_env: Some(env) }
    }
}

/// Condition configuration. Deserialized from YAML `condition:` blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "condition", rename_all = "snake_case")]
pub enum Condition {
    /// Entity state equals a specific value.
    State {
        entity_id: EntityId,
        state: String,
    },
    /// Entity numeric state satisfies threshold bounds.
    NumericState {
        entity_id: EntityId,
        #[serde(default)]
        above: Option<f64>,
        #[serde(default)]
        below: Option<f64>,
    },
    /// Jinja2 template evaluates to truthy.
    Template {
        value_template: String,
    },
    /// All child conditions must be true (logical AND).
    And {
        conditions: Vec<Condition>,
    },
    /// At least one child condition must be true (logical OR).
    Or {
        conditions: Vec<Condition>,
    },
    /// Inner condition must be false (logical NOT).
    Not {
        conditions: Vec<Condition>,
    },
}

impl Condition {
    /// Evaluate this condition against the provided context.
    ///
    /// Uses `Box::pin` for recursive variants (And/Or/Not) to satisfy the
    /// Rust requirement that recursive async fns introduce indirection.
    pub fn evaluate<'a>(&'a self, ctx: &'a EvalContext) -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send + 'a>> {
        Box::pin(async move {
            match self {
                Condition::State { entity_id, state } => {
                    ctx.states
                        .get(entity_id)
                        .map_or(false, |s| s.state == *state)
                }
                Condition::NumericState { entity_id, above, below } => {
                    let value: Option<f64> = ctx
                        .states
                        .get(entity_id)
                        .and_then(|s| s.state.parse().ok());
                    match value {
                        None => false,
                        Some(v) => {
                            above.map_or(true, |a| v > a) && below.map_or(true, |b| v < b)
                        }
                    }
                }
                Condition::Template { value_template } => {
                    if let Some(env) = &ctx.template_env {
                        match env.render_bool(value_template) {
                            Ok(v) => v,
                            Err(_) => false,
                        }
                    } else {
                        false
                    }
                }
                Condition::And { conditions } => {
                    for c in conditions {
                        if !c.evaluate(ctx).await {
                            return false;
                        }
                    }
                    true
                }
                Condition::Or { conditions } => {
                    for c in conditions {
                        if c.evaluate(ctx).await {
                            return true;
                        }
                    }
                    false
                }
                Condition::Not { conditions } => {
                    for c in conditions {
                        if c.evaluate(ctx).await {
                            return false;
                        }
                    }
                    true
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::{Context, EntityId, StateMachine};
    use std::sync::Arc;

    fn sm_with(entity_id: &str, state: &str) -> Arc<StateMachine> {
        let sm = Arc::new(StateMachine::new());
        sm.set(
            EntityId::parse(entity_id).unwrap(),
            state,
            serde_json::json!({}),
            Context::new(),
        );
        sm
    }

    #[tokio::test]
    async fn state_condition_matches() {
        let sm = sm_with("light.kitchen", "on");
        let ctx = EvalContext::new(sm);
        let cond = Condition::State {
            entity_id: EntityId::parse("light.kitchen").unwrap(),
            state: "on".into(),
        };
        assert!(cond.evaluate(&ctx).await);
    }

    #[tokio::test]
    async fn state_condition_no_match() {
        let sm = sm_with("light.kitchen", "off");
        let ctx = EvalContext::new(sm);
        let cond = Condition::State {
            entity_id: EntityId::parse("light.kitchen").unwrap(),
            state: "on".into(),
        };
        assert!(!cond.evaluate(&ctx).await);
    }

    #[tokio::test]
    async fn numeric_condition_above() {
        let sm = sm_with("sensor.temperature", "28");
        let ctx = EvalContext::new(sm);
        let cond = Condition::NumericState {
            entity_id: EntityId::parse("sensor.temperature").unwrap(),
            above: Some(25.0),
            below: None,
        };
        assert!(cond.evaluate(&ctx).await);
    }

    #[tokio::test]
    async fn and_combinator_all_true() {
        let sm = Arc::new(StateMachine::new());
        sm.set(EntityId::parse("light.a").unwrap(), "on", serde_json::json!({}), Context::new());
        sm.set(EntityId::parse("light.b").unwrap(), "on", serde_json::json!({}), Context::new());
        let ctx = EvalContext::new(sm);
        let cond = Condition::And {
            conditions: vec![
                Condition::State { entity_id: EntityId::parse("light.a").unwrap(), state: "on".into() },
                Condition::State { entity_id: EntityId::parse("light.b").unwrap(), state: "on".into() },
            ],
        };
        assert!(cond.evaluate(&ctx).await);
    }

    #[tokio::test]
    async fn and_combinator_one_false() {
        let sm = Arc::new(StateMachine::new());
        sm.set(EntityId::parse("light.a").unwrap(), "on", serde_json::json!({}), Context::new());
        sm.set(EntityId::parse("light.b").unwrap(), "off", serde_json::json!({}), Context::new());
        let ctx = EvalContext::new(sm);
        let cond = Condition::And {
            conditions: vec![
                Condition::State { entity_id: EntityId::parse("light.a").unwrap(), state: "on".into() },
                Condition::State { entity_id: EntityId::parse("light.b").unwrap(), state: "on".into() },
            ],
        };
        assert!(!cond.evaluate(&ctx).await);
    }

    #[tokio::test]
    async fn or_combinator_one_true() {
        let sm = Arc::new(StateMachine::new());
        sm.set(EntityId::parse("light.a").unwrap(), "off", serde_json::json!({}), Context::new());
        sm.set(EntityId::parse("light.b").unwrap(), "on", serde_json::json!({}), Context::new());
        let ctx = EvalContext::new(sm);
        let cond = Condition::Or {
            conditions: vec![
                Condition::State { entity_id: EntityId::parse("light.a").unwrap(), state: "on".into() },
                Condition::State { entity_id: EntityId::parse("light.b").unwrap(), state: "on".into() },
            ],
        };
        assert!(cond.evaluate(&ctx).await);
    }

    #[tokio::test]
    async fn not_condition_inverts() {
        let sm = sm_with("light.kitchen", "off");
        let ctx = EvalContext::new(sm);
        let cond = Condition::Not {
            conditions: vec![
                Condition::State {
                    entity_id: EntityId::parse("light.kitchen").unwrap(),
                    state: "on".into(),
                },
            ],
        };
        assert!(cond.evaluate(&ctx).await);
    }
}
