//! Concurrent state machine — the heart of HOMECORE.
//!
//! Mirrors `homeassistant.core.StateMachine`. Differences from HA per
//! ADR-127 §2.1:
//!
//! - DashMap shard-locked instead of one asyncio.Lock for the whole map
//! - Writers atomically replace `Arc<State>` entries; readers get
//!   zero-copy clones
//! - State changes fan out via a tokio broadcast channel (capacity
//!   4,096); slow subscribers get `Lagged` and must re-sync from the
//!   current map
//!
//! ## NOT in P1 (deferred to P2+)
//!
//! - `async_set_internal` schema validation
//! - Bulk delete of an entire domain (`async_remove_domain`)
//! - Restore-state on startup from the recorder (ADR-132)

use std::sync::Arc;

use chrono::Utc;
use dashmap::DashMap;
use tokio::sync::broadcast;

use crate::entity::{EntityId, State};
use crate::event::{Context, StateChangedEvent};

/// Broadcast channel capacity for state-changed events. 4,096 events
/// at 20 Hz per entity covers ~3 minutes of backlog for a single hot
/// entity. Slow subscribers must re-sync from the current map.
pub const STATE_CHANGED_CHANNEL_CAPACITY: usize = 4096;

/// The state machine. Cheap to clone (one `Arc`) — pass copies to as
/// many tasks as you like.
#[derive(Clone)]
pub struct StateMachine {
    inner: Arc<StateMachineInner>,
}

struct StateMachineInner {
    states: DashMap<EntityId, Arc<State>>,
    tx: broadcast::Sender<StateChangedEvent>,
}

impl StateMachine {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(STATE_CHANGED_CHANNEL_CAPACITY);
        Self {
            inner: Arc::new(StateMachineInner {
                states: DashMap::with_capacity(256),
                tx,
            }),
        }
    }

    /// Subscribe to state-changed events. Each subscriber gets an
    /// independent receiver; capacity is shared. Falling behind by
    /// 4,096 events yields `RecvError::Lagged(n)`.
    pub fn subscribe(&self) -> broadcast::Receiver<StateChangedEvent> {
        self.inner.tx.subscribe()
    }

    /// Read a state. Returns `None` if the entity is unknown.
    /// Zero-copy: caller gets an `Arc<State>` clone.
    pub fn get(&self, entity_id: &EntityId) -> Option<Arc<State>> {
        self.inner.states.get(entity_id).map(|s| Arc::clone(&s))
    }

    /// Write a state. Fires a `state_changed` broadcast even on the
    /// first write (old_state = None). HA semantics: only fires if the
    /// state string OR attributes changed; pure no-op writes are
    /// suppressed.
    ///
    /// Returns the new state snapshot.
    pub fn set(
        &self,
        entity_id: EntityId,
        new_state: impl Into<String>,
        attributes: serde_json::Value,
        context: Context,
    ) -> Arc<State> {
        let new_state_str = new_state.into();
        let old = self.inner.states.get(&entity_id).map(|r| Arc::clone(&*r));

        let next = match &old {
            Some(prev) => Arc::new(prev.next(new_state_str.clone(), attributes.clone(), context)),
            None => Arc::new(State::new(entity_id.clone(), new_state_str.clone(), attributes.clone(), context)),
        };

        // HA suppresses no-op writes (same state + same attributes).
        // We follow the same rule to keep the broadcast channel quiet.
        let is_noop = match &old {
            Some(prev) => prev.state == new_state_str && prev.attributes == attributes,
            None => false,
        };

        self.inner.states.insert(entity_id.clone(), Arc::clone(&next));

        if !is_noop {
            let event = StateChangedEvent {
                entity_id,
                old_state: old,
                new_state: Some(Arc::clone(&next)),
                fired_at: Utc::now(),
            };
            // err = no receivers; that's fine, write still committed.
            let _ = self.inner.tx.send(event);
        }
        next
    }

    /// Remove a state. Fires `state_changed` with `new_state = None`.
    pub fn remove(&self, entity_id: &EntityId) -> Option<Arc<State>> {
        let removed = self.inner.states.remove(entity_id).map(|(_, s)| s);
        if let Some(old) = &removed {
            let event = StateChangedEvent {
                entity_id: entity_id.clone(),
                old_state: Some(Arc::clone(old)),
                new_state: None,
                fired_at: Utc::now(),
            };
            let _ = self.inner.tx.send(event);
        }
        removed
    }

    /// Snapshot all current states. Allocates a new Vec — useful for
    /// the REST GET /api/states path (ADR-130).
    pub fn all(&self) -> Vec<Arc<State>> {
        self.inner.states.iter().map(|r| Arc::clone(r.value())).collect()
    }

    /// Snapshot all states whose entity_id matches a domain prefix.
    /// Mirrors HA's `hass.states.async_all(domain)`.
    pub fn all_by_domain(&self, domain: &str) -> Vec<Arc<State>> {
        self.inner
            .states
            .iter()
            .filter(|r| r.key().domain() == domain)
            .map(|r| Arc::clone(r.value()))
            .collect()
    }

    /// Number of entities currently tracked.
    pub fn len(&self) -> usize {
        self.inner.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.states.len() == 0
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> EntityId {
        EntityId::parse(s).unwrap()
    }

    #[tokio::test]
    async fn set_writes_and_fires() {
        let sm = StateMachine::new();
        let mut rx = sm.subscribe();
        sm.set(id("light.kitchen"), "on", serde_json::json!({"brightness": 200}), Context::new());
        let evt = rx.recv().await.unwrap();
        assert_eq!(evt.entity_id.as_str(), "light.kitchen");
        assert!(evt.old_state.is_none());
        assert_eq!(evt.new_state.as_ref().unwrap().state, "on");
    }

    #[tokio::test]
    async fn noop_writes_are_suppressed() {
        let sm = StateMachine::new();
        sm.set(id("light.k"), "on", serde_json::json!({}), Context::new());
        let mut rx = sm.subscribe();
        // Same state + same attributes → no event.
        sm.set(id("light.k"), "on", serde_json::json!({}), Context::new());
        let try_recv = tokio::time::timeout(std::time::Duration::from_millis(50), rx.recv()).await;
        assert!(try_recv.is_err(), "expected no event for no-op write");
    }

    #[tokio::test]
    async fn attribute_only_change_fires_but_preserves_last_changed() {
        let sm = StateMachine::new();
        let s1 = sm.set(id("sensor.t"), "20", serde_json::json!({"unit": "C"}), Context::new());
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let s2 = sm.set(id("sensor.t"), "20", serde_json::json!({"unit": "F"}), Context::new());
        assert_eq!(s1.last_changed, s2.last_changed);
        assert!(s2.last_updated > s1.last_updated);
    }

    #[test]
    fn all_by_domain_filters() {
        let sm = StateMachine::new();
        sm.set(id("light.a"), "on", serde_json::json!({}), Context::new());
        sm.set(id("light.b"), "off", serde_json::json!({}), Context::new());
        sm.set(id("sensor.t"), "20", serde_json::json!({}), Context::new());
        assert_eq!(sm.all_by_domain("light").len(), 2);
        assert_eq!(sm.all_by_domain("sensor").len(), 1);
        assert_eq!(sm.all().len(), 3);
    }

    #[tokio::test]
    async fn remove_fires_with_no_new_state() {
        let sm = StateMachine::new();
        sm.set(id("light.k"), "on", serde_json::json!({}), Context::new());
        let mut rx = sm.subscribe();
        sm.remove(&id("light.k"));
        let evt = rx.recv().await.unwrap();
        assert!(evt.new_state.is_none());
        assert!(evt.old_state.is_some());
    }
}
