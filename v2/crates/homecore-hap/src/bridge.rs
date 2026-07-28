//! `HapBridge` — owns the set of HOMECORE entities exposed as HAP accessories.
//!
//! The bridge owns mappings and their event stream. The feature-gated network
//! lifecycle is started separately with `start_server`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use homecore::entity::EntityId;
use tokio::sync::broadcast;

use crate::accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};
use crate::error::HapError;
use crate::mapping::{AccessoryMapping, EntityToAccessoryMapper};
use crate::mdns::{HapServiceRecord, MdnsAdvertiser, NullAdvertiser};

/// One registered HAP accessory — an entity + its last-known mapping.
#[derive(Debug, Clone)]
pub struct ExposedAccessory {
    pub entity_id: EntityId,
    pub accessory_type: HapAccessoryType,
    pub mapping: AccessoryMapping,
}

/// A characteristic snapshot emitted after a registered entity changes.
#[derive(Debug, Clone)]
pub struct CharacteristicEvent {
    pub entity_id: EntityId,
    pub accessory_type: HapAccessoryType,
    pub characteristics: Vec<(HapCharacteristic, HapCharacteristicValue)>,
}

struct BridgeInner {
    accessories: HashMap<EntityId, ExposedAccessory>,
}

/// HOMECORE-to-HAP accessory bridge state.
///
/// Call [`HapBridge::add_accessory`] to register entities and
/// [`HapBridge::running_accessories`] to read back what is currently
/// registered. Use `start_server` for the bounded TCP lifecycle.
#[derive(Clone)]
pub struct HapBridge {
    inner: Arc<RwLock<BridgeInner>>,
    advertiser: Arc<dyn MdnsAdvertiser>,
    events: broadcast::Sender<CharacteristicEvent>,
    pub service_record: HapServiceRecord,
}

impl HapBridge {
    /// Create a bridge with the given service record and a `NullAdvertiser`.
    pub fn new(service_record: HapServiceRecord) -> Self {
        Self::with_advertiser(service_record, Arc::new(NullAdvertiser))
    }

    /// Create a bridge with a custom `MdnsAdvertiser`.
    pub fn with_advertiser(
        service_record: HapServiceRecord,
        advertiser: Arc<dyn MdnsAdvertiser>,
    ) -> Self {
        let (events, _) = broadcast::channel(128);
        Self {
            inner: Arc::new(RwLock::new(BridgeInner {
                accessories: HashMap::new(),
            })),
            advertiser,
            events,
            service_record,
        }
    }

    /// Register an entity as a HAP accessory.
    ///
    /// The entity's current mapping is computed from `state`; call
    /// `update_accessory` on each `StateChanged` event to keep it fresh.
    ///
    /// Returns `HapError::AlreadyRegistered` if the entity is already
    /// registered. Call `remove_accessory` first to replace it.
    pub fn add_accessory(
        &self,
        entity_id: &EntityId,
        state: &homecore::entity::State,
    ) -> Result<(), HapError> {
        let mapping = EntityToAccessoryMapper::map(entity_id, state)?;
        let accessory_type = mapping.accessory_type;
        let exposed = ExposedAccessory {
            entity_id: entity_id.clone(),
            accessory_type,
            mapping,
        };
        let mut inner = self.inner.write().unwrap();
        if inner.accessories.contains_key(entity_id) {
            return Err(HapError::AlreadyRegistered(entity_id.as_str().to_owned()));
        }
        inner.accessories.insert(entity_id.clone(), exposed);
        tracing::debug!(entity = %entity_id, ?accessory_type, "HAP accessory registered");
        Ok(())
    }

    /// Remove a registered accessory.
    ///
    /// Returns `HapError::EntityNotFound` if the entity was not registered.
    pub fn remove_accessory(&self, entity_id: &EntityId) -> Result<(), HapError> {
        let mut inner = self.inner.write().unwrap();
        if inner.accessories.remove(entity_id).is_none() {
            return Err(HapError::EntityNotFound(entity_id.as_str().to_owned()));
        }
        tracing::debug!(entity = %entity_id, "HAP accessory removed");
        Ok(())
    }

    /// Refresh a registered accessory and notify event subscribers.
    pub fn update_accessory(
        &self,
        entity_id: &EntityId,
        state: &homecore::entity::State,
    ) -> Result<(), HapError> {
        let mapping = EntityToAccessoryMapper::map(entity_id, state)?;
        let accessory_type = mapping.accessory_type;
        {
            let mut inner = self.inner.write().unwrap();
            let accessory = inner
                .accessories
                .get_mut(entity_id)
                .ok_or_else(|| HapError::EntityNotFound(entity_id.as_str().to_owned()))?;
            accessory.accessory_type = accessory_type;
            accessory.mapping = mapping.clone();
        }
        let _ = self.events.send(CharacteristicEvent {
            entity_id: entity_id.clone(),
            accessory_type,
            characteristics: mapping.characteristics,
        });
        Ok(())
    }

    /// Subscribe to bounded characteristic updates. Lagging receivers receive
    /// Tokio's explicit `Lagged` error and must resynchronize from a snapshot.
    pub fn subscribe_events(&self) -> broadcast::Receiver<CharacteristicEvent> {
        self.events.subscribe()
    }

    /// Snapshot all currently registered accessories.
    pub fn running_accessories(&self) -> Vec<ExposedAccessory> {
        self.inner
            .read()
            .unwrap()
            .accessories
            .values()
            .cloned()
            .collect()
    }

    /// Number of registered accessories.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().accessories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start advertisement only.
    ///
    /// This legacy lifecycle does not bind a TCP listener. New integrations
    /// should call `start_server`, which advertises only after binding.
    pub async fn start(&self) -> Result<(), HapError> {
        self.advertiser.advertise(&self.service_record).await?;
        tracing::info!(
            instance = %self.service_record.instance_name,
            port = self.service_record.port,
            "HAP advertisement started without a TCP server"
        );
        Ok(())
    }

    /// Graceful shutdown — retracts mDNS advertisement.
    pub async fn stop(&self) -> Result<(), HapError> {
        self.advertiser
            .retract(&self.service_record.instance_name)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::entity::{EntityId, State};
    use homecore::event::Context;

    fn make_bridge() -> HapBridge {
        HapBridge::new(HapServiceRecord::bridge(
            "RuView Sense",
            51826,
            "AA:BB:CC:DD:EE:FF",
        ))
    }

    fn light_state(name: &str, on: bool, brightness: u8) -> (EntityId, State) {
        let eid = EntityId::parse(format!("light.{name}")).unwrap();
        let attrs = serde_json::json!({"brightness": brightness});
        let s = State::new(
            eid.clone(),
            if on { "on" } else { "off" },
            attrs,
            Context::default(),
        );
        (eid, s)
    }

    #[test]
    fn add_remove_roundtrip() {
        let bridge = make_bridge();
        let (eid, s) = light_state("kitchen", true, 200);

        assert!(bridge.is_empty());
        bridge.add_accessory(&eid, &s).unwrap();
        assert_eq!(bridge.len(), 1);

        let acc = bridge.running_accessories();
        assert_eq!(acc.len(), 1);
        assert_eq!(acc[0].entity_id, eid);
        assert_eq!(acc[0].accessory_type, HapAccessoryType::Lightbulb);

        bridge.remove_accessory(&eid).unwrap();
        assert!(bridge.is_empty());
    }

    #[test]
    fn add_duplicate_returns_error() {
        let bridge = make_bridge();
        let (eid, s) = light_state("kitchen", true, 200);
        bridge.add_accessory(&eid, &s).unwrap();
        let err = bridge.add_accessory(&eid, &s).unwrap_err();
        assert!(matches!(err, HapError::AlreadyRegistered(_)));
    }

    #[test]
    fn remove_nonexistent_returns_error() {
        let bridge = make_bridge();
        let eid = EntityId::parse("light.ghost").unwrap();
        let err = bridge.remove_accessory(&eid).unwrap_err();
        assert!(matches!(err, HapError::EntityNotFound(_)));
    }

    #[tokio::test]
    async fn update_emits_characteristic_event() {
        let bridge = make_bridge();
        let (eid, initial) = light_state("kitchen", false, 10);
        bridge.add_accessory(&eid, &initial).unwrap();
        let mut events = bridge.subscribe_events();
        let (_, updated) = light_state("kitchen", true, 200);
        bridge.update_accessory(&eid, &updated).unwrap();

        let event = events.recv().await.unwrap();
        assert_eq!(event.entity_id, eid);
        assert!(event
            .characteristics
            .contains(&(HapCharacteristic::On, HapCharacteristicValue::Bool(true))));
    }

    #[tokio::test]
    async fn start_stop_with_null_advertiser() {
        let bridge = make_bridge();
        bridge.start().await.unwrap();
        bridge.stop().await.unwrap();
    }
}
