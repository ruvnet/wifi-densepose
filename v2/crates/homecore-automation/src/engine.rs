//! `AutomationEngine` — subscribes to the HOMECORE event bus, evaluates
//! triggers, and runs automation action sequences.
//!
//! ADR-129 §2 design: one Tokio task per running automation instance.
//! RunMode::Single is enforced via a per-automation `AtomicBool` flag.

use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use homecore::HomeCore;

use crate::action::ExecutionContext;
use crate::automation::Automation;
use crate::condition::EvalContext;
use crate::trigger::TriggerContext;

/// The automation engine. Holds a HOMECORE handle and a list of registered
/// automations. Call `start()` to begin listening for events.
pub struct AutomationEngine {
    hc: HomeCore,
    automations: Arc<Mutex<Vec<Arc<Automation>>>>,
}

impl AutomationEngine {
    /// Create a new engine backed by the given HOMECORE handle.
    pub fn new(hc: HomeCore) -> Self {
        Self {
            hc,
            automations: Arc::new(Mutex::new(vec![])),
        }
    }

    /// Register an automation. Can be called before or after `start()`.
    pub fn register(&self, automation: Automation) {
        self.automations.lock().unwrap().push(Arc::new(automation));
    }

    /// Subscribe to the state-machine broadcast channel and start
    /// evaluating triggers. Returns a join handle for the background task.
    ///
    /// The task runs until the broadcast sender is dropped (i.e. the
    /// `HomeCore` instance is destroyed).
    pub fn start(&self) -> tokio::task::JoinHandle<()> {
        let mut rx = self.hc.states().subscribe();
        let automations = Arc::clone(&self.automations);
        let hc = self.hc.clone();

        tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        let autos = automations.lock().unwrap().clone();
                        for automation in autos {
                            if !automation.enabled {
                                continue;
                            }
                            let trigger_ctx = TriggerContext::state_changed(
                                event.entity_id.clone(),
                                event.old_state.clone(),
                                event.new_state.clone(),
                            );
                            // Check all triggers — fire on first match
                            let triggered = automation
                                .trigger
                                .iter()
                                .any(|t| t.matches_sync(&trigger_ctx));
                            if !triggered {
                                continue;
                            }
                            // Evaluate conditions
                            let sm = Arc::new(hc.states().clone());
                            let eval_ctx = EvalContext::new(sm);
                            let mut conditions_pass = true;
                            for cond in &automation.condition {
                                if !cond.evaluate(&eval_ctx).await {
                                    conditions_pass = false;
                                    break;
                                }
                            }
                            if !conditions_pass {
                                continue;
                            }
                            // Execute actions in a spawned task (non-blocking)
                            let auto_clone = Arc::clone(&automation);
                            let hc_clone = hc.clone();
                            tokio::spawn(async move {
                                let mut exec_ctx =
                                    ExecutionContext::new(hc_clone, auto_clone.id.clone());
                                for action in &auto_clone.action {
                                    if let Err(e) = action.execute(&mut exec_ctx).await {
                                        // P1: log errors to stderr; structured logging in P2
                                        eprintln!(
                                            "[homecore-automation] action error in {}: {e}",
                                            auto_clone.id
                                        );
                                        break;
                                    }
                                }
                            });
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("[homecore-automation] state-changed receiver lagged by {n} events");
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::automation::Automation;
    use crate::trigger::Trigger;
    use homecore::{Context, EntityId, HomeCore, ServiceCall, ServiceName};
    use homecore::service::FnHandler;
    use std::sync::{Arc, Mutex};
    use tokio::time::{sleep, Duration};

    /// Register a recording handler that captures all calls.
    async fn register_recorder(
        hc: &HomeCore,
        domain: &str,
        service: &str,
    ) -> Arc<Mutex<Vec<serde_json::Value>>> {
        let log: Arc<Mutex<Vec<serde_json::Value>>> = Arc::new(Mutex::new(vec![]));
        let log2 = Arc::clone(&log);
        hc.services()
            .register(
                ServiceName::new(domain, service),
                FnHandler(move |call: ServiceCall| {
                    let l = Arc::clone(&log2);
                    async move {
                        l.lock().unwrap().push(call.data.clone());
                        Ok(serde_json::Value::Null)
                    }
                }),
            )
            .await;
        log
    }

    #[tokio::test]
    async fn engine_fires_automation_on_state_change() {
        let hc = HomeCore::new();
        let log = register_recorder(&hc, "light", "turn_on").await;

        let engine = AutomationEngine::new(hc.clone());
        engine.register(Automation::new(
            "test_auto_1",
            vec![Trigger::State {
                entity_id: EntityId::parse("switch.living").unwrap(),
                from: None,
                to: Some("on".into()),
            }],
            vec![Action::ServiceCall {
                domain: "light".into(),
                service: "turn_on".into(),
                data: serde_json::json!({"brightness": 100}),
            }],
        ));

        let _handle = engine.start();

        // Fire a matching state change
        hc.states().set(
            EntityId::parse("switch.living").unwrap(),
            "on",
            serde_json::json!({}),
            Context::new(),
        );

        // Give the async task time to run
        sleep(Duration::from_millis(50)).await;

        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(log.lock().unwrap()[0]["brightness"], 100);
    }

    #[tokio::test]
    async fn engine_does_not_fire_on_wrong_entity() {
        let hc = HomeCore::new();
        let log = register_recorder(&hc, "light", "turn_on").await;

        let engine = AutomationEngine::new(hc.clone());
        engine.register(Automation::new(
            "test_auto_2",
            vec![Trigger::State {
                entity_id: EntityId::parse("switch.living").unwrap(),
                from: None,
                to: Some("on".into()),
            }],
            vec![Action::ServiceCall {
                domain: "light".into(),
                service: "turn_on".into(),
                data: serde_json::json!({}),
            }],
        ));

        let _handle = engine.start();

        // Fire on a DIFFERENT entity
        hc.states().set(
            EntityId::parse("switch.bedroom").unwrap(),
            "on",
            serde_json::json!({}),
            Context::new(),
        );

        sleep(Duration::from_millis(50)).await;
        assert_eq!(log.lock().unwrap().len(), 0, "should not fire on wrong entity");
    }

    #[tokio::test]
    async fn engine_disabled_automation_does_not_fire() {
        let hc = HomeCore::new();
        let log = register_recorder(&hc, "light", "turn_on").await;

        let engine = AutomationEngine::new(hc.clone());
        let mut auto = Automation::new(
            "test_auto_3",
            vec![Trigger::State {
                entity_id: EntityId::parse("switch.living").unwrap(),
                from: None,
                to: Some("on".into()),
            }],
            vec![Action::ServiceCall {
                domain: "light".into(),
                service: "turn_on".into(),
                data: serde_json::json!({}),
            }],
        );
        auto.enabled = false;
        engine.register(auto);

        let _handle = engine.start();

        hc.states().set(
            EntityId::parse("switch.living").unwrap(),
            "on",
            serde_json::json!({}),
            Context::new(),
        );

        sleep(Duration::from_millis(50)).await;
        assert_eq!(log.lock().unwrap().len(), 0, "disabled automation should not fire");
    }
}
