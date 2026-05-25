//! Entity ID newtype + immutable state snapshot type.
//!
//! Mirrors `homeassistant/core.py` `State` and the `entity_id` string
//! validation that every public HA call performs.
//!
//! ## EntityId validation (ADR-127 §2.1 + Q1)
//!
//! HA accepts unicode entity IDs since 2024.3. HOMECORE P1 accepts the
//! ASCII subset `[a-z0-9_]+\.[a-z0-9_]+` and rejects everything else
//! with a clear error. Unicode acceptance is deferred to P2 once the
//! Q1 strictness decision is made (see ADR-127 §8).

use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::event::Context;

/// Validated `domain.name` entity identifier.
///
/// Construct via [`EntityId::parse`] or [`EntityId::new`]; both validate
/// against the format `[a-z0-9_]+\.[a-z0-9_]+`. Custom `Serialize` /
/// `Deserialize` round-trips as a plain JSON string (matching HA's wire
/// format) and re-validates on deserialize so invalid IDs from disk
/// fail at load time rather than at first use.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct EntityId(Arc<str>);

impl Serialize for EntityId {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for EntityId {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        EntityId::parse(s).map_err(serde::de::Error::custom)
    }
}

impl EntityId {
    /// Validates and constructs an `EntityId`. Returns
    /// [`EntityIdError`] if the input is not `domain.name` shape with
    /// ASCII lowercase / digits / underscore in each segment.
    pub fn parse(s: impl Into<String>) -> Result<Self, EntityIdError> {
        let s: String = s.into();
        let (domain, name) = s
            .split_once('.')
            .ok_or_else(|| EntityIdError::MissingDot(s.clone()))?;
        if domain.is_empty() {
            return Err(EntityIdError::EmptyDomain(s));
        }
        if name.is_empty() {
            return Err(EntityIdError::EmptyName(s));
        }
        for ch in domain.chars().chain(name.chars()) {
            if !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_') {
                return Err(EntityIdError::InvalidChar { entity_id: s, ch });
            }
        }
        Ok(Self(Arc::from(s)))
    }

    /// Same as [`Self::parse`] but takes a `&str` and returns
    /// `Result<&'static EntityId, ...>` for constant entity IDs known
    /// at compile time. Used by ADR-128 plugins to register fixed-name
    /// services like `homeassistant.restart`.
    pub fn new(s: &str) -> Result<Self, EntityIdError> {
        Self::parse(s.to_owned())
    }

    /// Returns the `domain` part (everything before the first `.`).
    pub fn domain(&self) -> &str {
        self.0.split_once('.').map(|(d, _)| d).unwrap_or(&self.0)
    }

    /// Returns the `name` part (everything after the first `.`).
    pub fn name(&self) -> &str {
        self.0.split_once('.').map(|(_, n)| n).unwrap_or("")
    }

    /// Underlying string view.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EntityId({})", self.0)
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Error, Debug, Clone, Eq, PartialEq)]
pub enum EntityIdError {
    #[error("entity_id {0:?} is missing the required '.' between domain and name")]
    MissingDot(String),
    #[error("entity_id {0:?} has an empty domain segment")]
    EmptyDomain(String),
    #[error("entity_id {0:?} has an empty name segment")]
    EmptyName(String),
    #[error("entity_id {entity_id:?} contains invalid character {ch:?} — only [a-z0-9_] allowed (HA-compat ASCII subset; see ADR-127 §Q1)")]
    InvalidChar { entity_id: String, ch: char },
}

/// Immutable state snapshot for one entity at one moment in time.
///
/// Mirrors `homeassistant.core.State`. Reader-cloneable via `Arc<State>`;
/// writers atomically replace the entry in the `DashMap` so observers
/// never see a partial mutation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub entity_id: EntityId,
    pub state: String,
    /// Attribute bag — accepts whatever JSON the integration emits.
    /// Mirrors HA's `Dict[str, Any]` attribute model.
    pub attributes: serde_json::Value,
    /// When the `state` field last changed value. Only bumped if the
    /// new state string differs from the old; attribute-only updates
    /// preserve this timestamp.
    pub last_changed: DateTime<Utc>,
    /// When this snapshot was written. Bumped on every `set` call,
    /// including attribute-only updates.
    pub last_updated: DateTime<Utc>,
    /// Causality context — links state changes to the user / automation
    /// / service call that originated them. Mirrors HA's `Context`.
    pub context: Context,
}

impl State {
    /// Construct a fresh state snapshot at `now`.
    pub fn new(
        entity_id: EntityId,
        state: impl Into<String>,
        attributes: serde_json::Value,
        context: Context,
    ) -> Self {
        let now = Utc::now();
        Self {
            entity_id,
            state: state.into(),
            attributes,
            last_changed: now,
            last_updated: now,
            context,
        }
    }

    /// Construct the next state snapshot. If the new `state` string
    /// equals the prior `state`, `last_changed` is preserved.
    pub fn next(
        &self,
        new_state: impl Into<String>,
        new_attributes: serde_json::Value,
        context: Context,
    ) -> Self {
        let new_state = new_state.into();
        let now = Utc::now();
        let last_changed = if new_state == self.state {
            self.last_changed
        } else {
            now
        };
        Self {
            entity_id: self.entity_id.clone(),
            state: new_state,
            attributes: new_attributes,
            last_changed,
            last_updated: now,
            context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_id_parses_valid() {
        let e = EntityId::parse("light.living_room").unwrap();
        assert_eq!(e.domain(), "light");
        assert_eq!(e.name(), "living_room");
        assert_eq!(e.as_str(), "light.living_room");
    }

    #[test]
    fn entity_id_rejects_missing_dot() {
        assert!(matches!(
            EntityId::parse("light_living_room"),
            Err(EntityIdError::MissingDot(_))
        ));
    }

    #[test]
    fn entity_id_rejects_uppercase() {
        let err = EntityId::parse("light.LivingRoom").unwrap_err();
        match err {
            EntityIdError::InvalidChar { ch, .. } => assert_eq!(ch, 'L'),
            other => panic!("expected InvalidChar, got {other:?}"),
        }
    }

    #[test]
    fn entity_id_rejects_unicode() {
        // ADR-127 §Q1 — P1 is strict ASCII. Unicode acceptance deferred.
        assert!(EntityId::parse("light.küche").is_err());
    }

    #[test]
    fn state_next_preserves_last_changed_when_state_unchanged() {
        let id = EntityId::parse("sensor.temp").unwrap();
        let s1 = State::new(id.clone(), "20.0", serde_json::json!({}), Context::default());
        std::thread::sleep(std::time::Duration::from_millis(2));
        let s2 = s1.next("20.0", serde_json::json!({"updated": true}), Context::default());
        assert_eq!(s1.last_changed, s2.last_changed);
        assert!(s2.last_updated > s1.last_updated);
    }

    #[test]
    fn state_next_bumps_last_changed_when_state_changes() {
        let id = EntityId::parse("sensor.temp").unwrap();
        let s1 = State::new(id, "20.0", serde_json::json!({}), Context::default());
        std::thread::sleep(std::time::Duration::from_millis(2));
        let s2 = s1.next("21.0", serde_json::json!({}), Context::default());
        assert!(s2.last_changed > s1.last_changed);
    }
}
