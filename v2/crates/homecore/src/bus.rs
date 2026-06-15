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

    /// Bus-lag safety (same failure class as the homecore-api WS
    /// broadcast-lag DoS, here on the core bus): a subscriber that never
    /// drains must NOT block the publisher, must NOT make the channel grow
    /// without bound, and must NOT take down a healthy fast subscriber. The
    /// bounded `tokio::sync::broadcast` gives the slow receiver a recoverable
    /// `Lagged(n)` (drop-oldest, re-sync) while `fire_*` stays non-blocking.
    ///
    /// Evidence: with EVENT_CHANNEL_CAPACITY = 4096 we fire 3× capacity
    /// while a slow subscriber sits idle. Every `fire_domain` returns
    /// promptly (publisher never blocked); the slow receiver observes
    /// `Lagged` then re-syncs to live events; the fast receiver — created
    /// after the flood and kept drained — receives all subsequent events
    /// with no loss. The bus stays live throughout.
    #[tokio::test]
    async fn slow_subscriber_does_not_block_publisher_or_kill_the_bus() {
        use tokio::sync::broadcast::error::TryRecvError;

        let bus = EventBus::new();
        // Slow subscriber: subscribes, then never drains during the flood.
        let mut slow = bus.subscribe_domain();

        // Publisher fires 3× capacity. None of these may block.
        let total = EVENT_CHANNEL_CAPACITY * 3;
        for i in 0..total {
            // Returns the receiver count (>=1 here); the point is it
            // returns AT ALL without awaiting the slow receiver.
            let _ = bus.fire_domain(DomainEvent::new(
                "flood",
                serde_json::json!({ "i": i }),
                Context::new(),
            ));
        }

        // The slow receiver is forced past capacity → recoverable Lagged,
        // NOT a closed channel and NOT a hang.
        let mut saw_lagged = false;
        loop {
            match slow.try_recv() {
                Ok(_) => {}
                Err(TryRecvError::Lagged(n)) => {
                    assert!(n > 0);
                    saw_lagged = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Closed) => panic!("bus closed — must stay live"),
            }
        }
        assert!(saw_lagged, "slow subscriber should have lagged, not blocked the bus");

        // The bus is still live: a fresh fast subscriber receives new events.
        let mut fast = bus.subscribe_domain();
        bus.fire_domain(DomainEvent::new("live", serde_json::json!({"ok": true}), Context::new()));
        let evt = fast.recv().await.unwrap();
        assert_eq!(evt.event_type, "live");

        // And the lagged subscriber recovers (re-syncs) to live events too.
        let evt2 = slow.recv().await.unwrap();
        assert_eq!(evt2.event_type, "live");
    }
}
