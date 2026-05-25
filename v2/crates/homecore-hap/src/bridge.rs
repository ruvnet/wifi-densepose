//! `HapBridge` — owns the set of HOMECORE entities exposed as HAP accessories.
//!
//! P1 does not start a real HAP-1.1 server; it ships the API surface so other
//! crates (and P2's `hap-server` feature) can register accessories and query
//! their current mapping. The actual mDNS + HAP pairing is gated to P2.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use homecore::entity::EntityId;

use crate::accessory::HapAccessoryType;
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

struct BridgeInner {
    accessories: HashMap<EntityId, ExposedAccessory>,
}

/// The P1 HAP bridge.
///
/// Call [`HapBridge::add_accessory`] to register entities and
/// [`HapBridge::running_accessories`] to read back what is currently
/// registered. In P2, `start()` will spawn the `hap` server task.
#[derive(Clone)]
pub struct HapBridge {
    inner: Arc<RwLock<BridgeInner>>,
    advertiser: Arc<dyn MdnsAdvertiser>,
    pub service_record: HapServiceRecord,
}

impl HapBridge {
    /// Create a bridge with the given service record and a `NullAdvertiser`
    /// (P1 default — real mDNS lands in P2).
    pub fn new(service_record: HapServiceRecord) -> Self {
        Self::with_advertiser(service_record, Arc::new(NullAdvertiser))
    }

    /// Create a bridge with a custom `MdnsAdvertiser` (used in tests and P2).
    pub fn with_advertiser(
        service_record: HapServiceRecord,
        advertiser: Arc<dyn MdnsAdvertiser>,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(BridgeInner { accessories: HashMap::new() })),
            advertiser,
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

    /// Snapshot all currently registered accessories.
    pub fn running_accessories(&self) -> Vec<ExposedAccessory> {
        self.inner.read().unwrap().accessories.values().cloned().collect()
    }

    /// Number of registered accessories.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().accessories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// P2 stub — will start the HAP-1.1 server + mDNS advertisement.
    /// In P1 this only fires the null advertiser.
    pub async fn start(&self) -> Result<(), HapError> {
        self.advertiser.advertise(&self.service_record).await?;
        tracing::info!(
            instance = %self.service_record.instance_name,
            port = self.service_record.port,
            "HapBridge started (P1 — no real HAP server; mDNS stub only)"
        );
        Ok(())
    }

    /// Graceful shutdown — retracts mDNS advertisement.
    pub async fn stop(&self) -> Result<(), HapError> {
        self.advertiser.retract(&self.service_record.instance_name).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::entity::{EntityId, State};
    use homecore::event::Context;

    fn make_bridge() -> HapBridge {
        HapBridge::new(HapServiceRecord {
            instance_name: "RuView Sense".into(),
            port: 51826,
            setup_code: "111-22-333".into(),
            device_id: "AA:BB:CC:DD:EE:FF".into(),
        })
    }

    fn light_state(name: &str, on: bool, brightness: u8) -> (EntityId, State) {
        let eid = EntityId::parse(&format!("light.{name}")).unwrap();
        let attrs = serde_json::json!({"brightness": brightness});
        let s = State::new(eid.clone(), if on { "on" } else { "off" }, attrs, Context::default());
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
    async fn start_stop_with_null_advertiser() {
        let bridge = make_bridge();
        bridge.start().await.unwrap();
        bridge.stop().await.unwrap();
    }
}
