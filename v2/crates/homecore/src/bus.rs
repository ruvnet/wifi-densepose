//! Event bus — typed system events + untyped domain events.
//!
//! ADR-127 §2.2: HA's single dict-typed event channel becomes two:
//! - typed `SystemEvent` channel for known shapes (recorder, automation)
//! - untyped `DomainEvent` channel for arbitrary integration events
//!
//! Capacity 4,096 on both. Lagged receivers must re-sync (recorder
//! re-reads current state; automation re-evaluates triggers).

use std::sync::Arc;

use tokio::sync::broadcast;

use crate::event::{DomainEvent, SystemEvent};

pub const EVENT_CHANNEL_CAPACITY: usize = 4096;

#[derive(Clone)]
pub struct EventBus {
    inner: Arc<EventBusInner>,
}

struct EventBusInner {
    system_tx: broadcast::Sender<SystemEvent>,
    domain_tx: broadcast::Sender<DomainEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (system_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let (domain_tx, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        Self {
            inner: Arc::new(EventBusInner { system_tx, domain_tx }),
        }
    }

    pub fn subscribe_system(&self) -> broadcast::Receiver<SystemEvent> {
        self.inner.system_tx.subscribe()
    }

    pub fn subscribe_domain(&self) -> broadcast::Receiver<DomainEvent> {
        self.inner.domain_tx.subscribe()
    }

    /// Fire a typed system event. Returns the number of active
    /// receivers (zero is fine).
    pub fn fire_system(&self, event: SystemEvent) -> usize {
        self.inner.system_tx.send(event).unwrap_or(0)
    }

    /// Fire an untyped domain event. Mirrors `hass.bus.async_fire`.
    pub fn fire_domain(&self, event: DomainEvent) -> usize {
        self.inner.domain_tx.send(event).unwrap_or(0)
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Context;

    #[tokio::test]
    async fn fire_system_reaches_subscriber() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe_system();
        bus.fire_system(SystemEvent::HomeCoreStarted);
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, SystemEvent::HomeCoreStarted));
    }

    #[tokio::test]
    async fn fire_domain_reaches_subscriber() {
        let bus = EventBus::new();
        let mut rx = bus.subscribe_domain();
        bus.fire_domain(DomainEvent::new(
            "ruview_csi_frame",
            serde_json::json!({"frame_id": 42}),
            Context::new(),
        ));
        let event = rx.recv().await.unwrap();
        assert_eq!(event.event_type, "ruview_csi_frame");
        assert_eq!(event.event_data["frame_id"], 42);
    }
}
