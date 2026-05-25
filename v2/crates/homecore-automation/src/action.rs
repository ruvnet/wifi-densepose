//! `Action` enum and async execution.
//!
//! Implements the ADR-129 P1 action set: `service_call`, `delay`, `scene`,
//! `wait_for_trigger`, `choose`. Complex variants (parallel, repeat, if,
//! stop, fire_event, wait_template) land in P2.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::sleep;

use homecore::{Context, HomeCore, ServiceCall, ServiceName};

use crate::error::AutomationError;

/// Runtime context passed into action execution.
pub struct ExecutionContext {
    /// HOMECORE handle — provides service registry + state machine.
    pub hc: HomeCore,
    /// Causality context for service calls triggered by this automation.
    pub context: Context,
    /// Automation ID for tracing/logging.
    pub automation_id: String,
}

impl ExecutionContext {
    pub fn new(hc: HomeCore, automation_id: impl Into<String>) -> Self {
        Self {
            hc,
            context: Context::new(),
            automation_id: automation_id.into(),
        }
    }
}

/// Action configuration. Deserialized from YAML `action:` blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Action {
    /// Call a HOMECORE service.
    ServiceCall {
        domain: String,
        service: String,
        #[serde(default)]
        data: serde_json::Value,
    },
    /// Pause execution for a fixed duration (ISO 8601 or seconds float).
    Delay {
        /// Delay in seconds.
        seconds: f64,
    },
    /// Activate a named scene entity.
    Scene {
        scene: String,
    },
    /// Block until one of the listed triggers fires (or timeout).
    WaitForTrigger {
        timeout_seconds: Option<f64>,
    },
    /// Conditional branching — first matching branch wins.
    Choose {
        choices: Vec<ChoiceBranch>,
        #[serde(default)]
        default: Vec<Action>,
    },
}

/// A single branch in a `Choose` action.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChoiceBranch {
    pub conditions: Vec<serde_yaml::Value>,
    pub sequence: Vec<Action>,
}

impl Action {
    /// Execute this action using the provided context.
    ///
    /// Returns a JSON value (may be `null`) for callers that chain
    /// `wait_for_trigger` / `set_variable` patterns (P2).
    ///
    /// Uses `Box::pin` for recursive variants (Choose) to satisfy the
    /// Rust requirement that recursive async fns introduce indirection.
    pub fn execute<'a>(
        &'a self,
        ctx: &'a mut ExecutionContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value, AutomationError>> + Send + 'a>> {
        Box::pin(async move {
            match self {
                Action::ServiceCall { domain, service, data } => {
                    let call = ServiceCall {
                        name: ServiceName::new(domain.clone(), service.clone()),
                        data: data.clone(),
                        context: ctx.context.clone(),
                    };
                    let result = ctx.hc.services().call(call).await?;
                    Ok(result)
                }
                Action::Delay { seconds } => {
                    let dur = Duration::from_secs_f64(*seconds);
                    sleep(dur).await;
                    Ok(serde_json::Value::Null)
                }
                Action::Scene { scene } => {
                    // Scene activation maps to homeassistant.turn_on with entity_id = scene
                    let call = ServiceCall {
                        name: ServiceName::new("homeassistant", "turn_on"),
                        data: serde_json::json!({ "entity_id": scene }),
                        context: ctx.context.clone(),
                    };
                    let result = ctx.hc.services().call(call).await?;
                    Ok(result)
                }
                Action::WaitForTrigger { timeout_seconds } => {
                    // P1 stub — just sleeps for the timeout duration if specified.
                    // Full trigger subscription lands in P2.
                    if let Some(secs) = timeout_seconds {
                        sleep(Duration::from_secs_f64(*secs)).await;
                    }
                    Ok(serde_json::Value::Null)
                }
                Action::Choose { choices: _, default } => {
                    // P1 stub — condition evaluation for choices lands in P2;
                    // for now, fall through to default branch.
                    for a in default {
                        a.execute(ctx).await?;
                    }
                    Ok(serde_json::Value::Null)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::{HomeCore, ServiceCall, ServiceError, ServiceName};
    use homecore::service::FnHandler;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn service_call_action_fires_handler() {
        let hc = HomeCore::new();
        let log: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
        let log2 = Arc::clone(&log);
        hc.services()
            .register(
                ServiceName::new("light", "turn_on"),
                FnHandler(move |call: ServiceCall| {
                    let log3 = Arc::clone(&log2);
                    async move {
                        log3.lock().unwrap().push(call.data.clone());
                        Ok(call.data)
                    }
                }),
            )
            .await;

        let action = Action::ServiceCall {
            domain: "light".into(),
            service: "turn_on".into(),
            data: serde_json::json!({"brightness": 255}),
        };
        let mut exec_ctx = ExecutionContext::new(hc, "test_auto");
        let res = action.execute(&mut exec_ctx).await.unwrap();
        assert_eq!(res["brightness"], 255);
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn delay_action_completes() {
        let hc = HomeCore::new();
        let mut exec_ctx = ExecutionContext::new(hc, "test_auto");
        let action = Action::Delay { seconds: 0.001 };
        let result = action.execute(&mut exec_ctx).await.unwrap();
        assert!(result.is_null());
    }

    #[tokio::test]
    async fn service_call_unregistered_returns_error() {
        let hc = HomeCore::new();
        let mut exec_ctx = ExecutionContext::new(hc, "test_auto");
        let action = Action::ServiceCall {
            domain: "light".into(),
            service: "turn_on".into(),
            data: serde_json::json!({}),
        };
        let err = action.execute(&mut exec_ctx).await.unwrap_err();
        assert!(matches!(err, AutomationError::ServiceCall(ServiceError::NotRegistered { .. })));
    }
}
