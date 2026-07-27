//! `HomeCore` runtime coordinator. Mirrors `homeassistant.core.HomeAssistant`.
//!
//! Cheap to clone — all internals are `Arc`-shared so tasks can each
//! hold their own `HomeCore` handle without coordination overhead.

use std::sync::Arc;

use crate::bus::EventBus;
use crate::registry::EntityRegistry;
use crate::service::ServiceRegistry;
use crate::state::StateMachine;

#[derive(Clone)]
pub struct HomeCore {
    inner: Arc<HomeCoreInner>,
}

struct HomeCoreInner {
    pub bus: EventBus,
    pub states: StateMachine,
    pub services: ServiceRegistry,
    pub entities: EntityRegistry,
}

impl HomeCore {
    pub fn new() -> Self {
        let bus = EventBus::new();
        Self {
            inner: Arc::new(HomeCoreInner {
                states: StateMachine::with_event_bus(bus.clone()),
                services: ServiceRegistry::with_event_bus(bus.clone()),
                bus,
                entities: EntityRegistry::new(),
            }),
        }
    }

    pub fn bus(&self) -> &EventBus {
        &self.inner.bus
    }

    pub fn states(&self) -> &StateMachine {
        &self.inner.states
    }

    pub fn services(&self) -> &ServiceRegistry {
        &self.inner.services
    }

    pub fn entities(&self) -> &EntityRegistry {
        &self.inner.entities
    }
}

impl Default for HomeCore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityId;
    use crate::event::{Context, SystemEvent};
    use crate::service::{FnHandler, ServiceCall, ServiceName};

    #[tokio::test]
    async fn end_to_end_set_then_get() {
        let hc = HomeCore::new();
        let id = EntityId::parse("light.kitchen").unwrap();
        hc.states().set(
            id.clone(),
            "on",
            serde_json::json!({"brightness": 200}),
            Context::new(),
        );
        let snap = hc.states().get(&id).unwrap();
        assert_eq!(snap.state, "on");
        assert_eq!(snap.attributes["brightness"], 200);
    }

    #[tokio::test]
    async fn state_changes_are_published_on_shared_system_bus() {
        let hc = HomeCore::new();
        let mut rx = hc.bus().subscribe_system();
        let id = EntityId::parse("light.kitchen").unwrap();

        hc.states()
            .set(id.clone(), "on", serde_json::json!({}), Context::new());

        let event = rx.recv().await.unwrap();
        match event {
            SystemEvent::StateChanged(change) => {
                assert_eq!(change.entity_id, id);
                assert_eq!(change.new_state.unwrap().state, "on");
            }
            other => panic!("expected StateChanged, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn service_calls_are_published_on_shared_system_bus() {
        let hc = HomeCore::new();
        let service = ServiceName::new("light", "turn_on");
        hc.services()
            .register(
                service.clone(),
                FnHandler(|_| async { Ok(serde_json::json!({})) }),
            )
            .await;
        let mut rx = hc.bus().subscribe_system();

        hc.services()
            .call(ServiceCall {
                name: service,
                data: serde_json::json!({"brightness": 42}),
                context: Context::new(),
            })
            .await
            .unwrap();

        match rx.recv().await.unwrap() {
            SystemEvent::ServiceCalled {
                domain,
                service,
                data,
                ..
            } => {
                assert_eq!(domain, "light");
                assert_eq!(service, "turn_on");
                assert_eq!(data["brightness"], 42);
            }
            other => panic!("expected ServiceCalled, got {other:?}"),
        }
    }
}
