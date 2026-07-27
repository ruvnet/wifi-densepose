//! In-memory entity and device registries. Durable files are loaded by
//! `homecore-server` during bounded startup restoration.
//!
//! Schema fields mirror HA `core.entity_registry` v13 per ADR-127 §2.4.

use std::collections::{BTreeMap, HashMap, HashSet};
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

/// Physical-device metadata persisted in `core.device_registry`.
///
/// The fields track the HA v13 registry surface used by HOMECORE. Identifier
/// and connection pairs are sets because their order is not semantically
/// meaningful in HA.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceEntry {
    pub id: String,
    #[serde(default)]
    pub config_entries: HashSet<String>,
    #[serde(default)]
    pub identifiers: HashSet<(String, String)>,
    #[serde(default)]
    pub connections: HashSet<(String, String)>,
    pub manufacturer: Option<String>,
    pub model: Option<String>,
    pub model_id: Option<String>,
    pub name: Option<String>,
    pub name_by_user: Option<String>,
    pub sw_version: Option<String>,
    pub hw_version: Option<String>,
    pub serial_number: Option<String>,
    pub via_device_id: Option<String>,
    pub area_id: Option<String>,
    pub entry_type: Option<String>,
    pub disabled_by: Option<String>,
    pub configuration_url: Option<String>,
    #[serde(default)]
    pub labels: HashSet<String>,
    pub primary_config_entry: Option<String>,
    /// Forward-compatible device fields from newer HA v13-compatible rows.
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
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
        self.entries
            .write()
            .await
            .insert(entry.entity_id.clone(), entry);
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

    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}

impl Default for EntityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct DeviceRegistry {
    entries: Arc<RwLock<HashMap<String, DeviceEntry>>>,
}

impl DeviceRegistry {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn register(&self, entry: DeviceEntry) {
        self.entries.write().await.insert(entry.id.clone(), entry);
    }

    pub async fn get(&self, id: &str) -> Option<DeviceEntry> {
        self.entries.read().await.get(id).cloned()
    }

    pub async fn all(&self) -> Vec<DeviceEntry> {
        self.entries.read().await.values().cloned().collect()
    }

    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}

impl Default for DeviceRegistry {
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
