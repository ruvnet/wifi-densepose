//! Typed system events + untyped domain events + Context.
//!
//! Mirrors `homeassistant.core.EventBus` + `homeassistant.const.EVENT_*`
//! constants. ADR-127 §2.2 splits HA's single dict-typed event channel
//! into two: a typed system channel (zero-allocation read path) and a
//! json-blob domain channel (for arbitrary integration-fired events).

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entity::{EntityId, State};

/// Well-known HA event-type string constants.
///
/// Mirrors `homeassistant/const.py` `EVENT_*` constants. Used by
/// integrations that fire untyped [`DomainEvent`]s.
#[non_exhaustive]
pub struct EventType;

impl EventType {
    pub const STATE_CHANGED: &'static str = "state_changed";
    pub const SERVICE_REGISTERED: &'static str = "service_registered";
    pub const SERVICE_REMOVED: &'static str = "service_removed";
    pub const CALL_SERVICE: &'static str = "call_service";
    pub const COMPONENT_LOADED: &'static str = "component_loaded";
    pub const PLATFORM_DISCOVERED: &'static str = "platform_discovered";
    pub const HOMEASSISTANT_START: &'static str = "homeassistant_start";
    pub const HOMEASSISTANT_STARTED: &'static str = "homeassistant_started";
    pub const HOMEASSISTANT_STOP: &'static str = "homeassistant_stop";
    pub const HOMEASSISTANT_FINAL_WRITE: &'static str = "homeassistant_final_write";
    pub const HOMEASSISTANT_CLOSE: &'static str = "homeassistant_close";
}

/// Causality context for a state change or service call.
///
/// Mirrors `homeassistant.core.Context`. Used by automations to detect
/// loops ("don't re-fire on a state change my own automation caused")
/// and by the recorder (ADR-132) to attribute changes to users.
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub struct Context {
    pub id: Uuid,
    pub user_id: Option<String>,
    pub parent_id: Option<Uuid>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_user(user_id: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id: Some(user_id.into()),
            parent_id: None,
        }
    }

    pub fn child_of(parent: &Context) -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id: parent.user_id.clone(),
            parent_id: Some(parent.id),
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            user_id: None,
            parent_id: None,
        }
    }
}

/// Typed enum of system events. Subscribers that only care about a
/// specific shape (the recorder, the websocket subscriber) can match on
/// the variant without going through `serde_json::Value`.
#[derive(Clone, Debug)]
pub enum SystemEvent {
    StateChanged(StateChangedEvent),
    ServiceRegistered { domain: String, service: String },
    ServiceRemoved { domain: String, service: String },
    ComponentLoaded { component: String },
    HomeCoreStart,
    HomeCoreStarted,
    HomeCoreStop,
}

/// State-change event payload. Carries the old and new snapshots so a
/// subscriber doesn't need to read the state machine again to learn
/// what changed.
///
/// Mirrors HA's event_data `{ entity_id, old_state, new_state }`.
#[derive(Clone, Debug)]
pub struct StateChangedEvent {
    pub entity_id: EntityId,
    pub old_state: Option<Arc<State>>,
    pub new_state: Option<Arc<State>>,
    pub fired_at: DateTime<Utc>,
}

/// Untyped event fired by integrations. Mirrors HA's
/// `EventBus.async_fire(event_type, event_data)`.
#[derive(Clone, Debug)]
pub struct DomainEvent {
    pub event_type: String,
    pub event_data: serde_json::Value,
    pub origin: EventOrigin,
    pub context: Context,
    pub fired_at: DateTime<Utc>,
}

/// Where an event originated. Mirrors HA's `EventOrigin` enum (`local`
/// vs `remote`).
#[derive(Clone, Debug, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub enum EventOrigin {
    Local,
    Remote,
}

impl DomainEvent {
    pub fn new(
        event_type: impl Into<String>,
        event_data: serde_json::Value,
        context: Context,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            event_data,
            origin: EventOrigin::Local,
            context,
            fired_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_child_inherits_user_id() {
        let parent = Context::with_user("alice");
        let child = Context::child_of(&parent);
        assert_eq!(child.user_id.as_deref(), Some("alice"));
        assert_eq!(child.parent_id, Some(parent.id));
        assert_ne!(child.id, parent.id);
    }

    #[test]
    fn event_type_constants_match_ha_names() {
        // These string values are wire-format with HA — must match
        // exactly so ADR-130 can serve a wire-compat WebSocket API.
        assert_eq!(EventType::STATE_CHANGED, "state_changed");
        assert_eq!(EventType::HOMEASSISTANT_START, "homeassistant_start");
    }
}
