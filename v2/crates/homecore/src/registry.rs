//! In-memory entity registry (P1). Persistence to
//! `.homecore/storage/core.entity_registry` lands in P2.
//!
//! Schema fields mirror HA `core.entity_registry` v13 per ADR-127 §2.4.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::entity::EntityId;

/// Entity category enum. Mirrors HA `homeassistant.helpers.entity.EntityCategory`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityCategory {
    Config,
    Diagnostic,
}

/// Source that disabled an entity. Mirrors HA `disabled_by` enum.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisabledBy {
    User,
    Integration,
    ConfigEntry,
    Device,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityEntry {
    pub entity_id: EntityId,
    pub unique_id: Option<String>,
    pub platform: String,
    /// User-set display name. None means "use the entity's default name".
    pub name: Option<String>,
    pub disabled_by: Option<DisabledBy>,
    pub area_id: Option<String>,
    pub device_id: Option<String>,
    pub entity_category: Option<EntityCategory>,
    pub config_entry_id: Option<String>,
}

#[derive(Clone)]
pub struct EntityRegistry {
    entries: Arc<RwLock<HashMap<EntityId, EntityEntry>>>,
}

impl EntityRegistry {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, entry: EntityEntry) {
        self.entries.write().await.insert(entry.entity_id.clone(), entry);
    }

    pub async fn get(&self, entity_id: &EntityId) -> Option<EntityEntry> {
        self.entries.read().await.get(entity_id).cloned()
    }

    pub async fn remove(&self, entity_id: &EntityId) -> Option<EntityEntry> {
        self.entries.write().await.remove(entity_id)
    }

    pub async fn all(&self) -> Vec<EntityEntry> {
        self.entries.read().await.values().cloned().collect()
    }

    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }
}

impl Default for EntityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_and_read() {
        let reg = EntityRegistry::new();
        let id = EntityId::parse("light.kitchen").unwrap();
        reg.register(EntityEntry {
            entity_id: id.clone(),
            unique_id: Some("hue_lamp_42".into()),
            platform: "hue".into(),
            name: Some("Kitchen lamp".into()),
            disabled_by: None,
            area_id: Some("kitchen".into()),
            device_id: None,
            entity_category: None,
            config_entry_id: None,
        })
        .await;
        let got = reg.get(&id).await.unwrap();
        assert_eq!(got.platform, "hue");
        assert_eq!(got.name.as_deref(), Some("Kitchen lamp"));
    }

    #[tokio::test]
    async fn disabled_by_round_trips_via_serde() {
        let entry = EntityEntry {
            entity_id: EntityId::parse("sensor.x").unwrap(),
            unique_id: None,
            platform: "test".into(),
            name: None,
            disabled_by: Some(DisabledBy::Integration),
            area_id: None,
            device_id: None,
            entity_category: Some(EntityCategory::Diagnostic),
            config_entry_id: None,
        };
        let json = serde_json::to_string(&entry).unwrap();
        // HA wire format uses snake_case for the disabled_by enum.
        assert!(json.contains("\"disabled_by\":\"integration\""));
        assert!(json.contains("\"entity_category\":\"diagnostic\""));
        let back: EntityEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.disabled_by, Some(DisabledBy::Integration));
    }
}
