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
        Self {
            inner: Arc::new(HomeCoreInner {
                bus: EventBus::new(),
                states: StateMachine::new(),
                services: ServiceRegistry::new(),
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
    use crate::event::Context;

    #[tokio::test]
    async fn end_to_end_set_then_get() {
        let hc = HomeCore::new();
        let id = EntityId::parse("light.kitchen").unwrap();
        hc.states().set(id.clone(), "on", serde_json::json!({"brightness": 200}), Context::new());
        let snap = hc.states().get(&id).unwrap();
        assert_eq!(snap.state, "on");
        assert_eq!(snap.attributes["brightness"], 200);
    }
}
