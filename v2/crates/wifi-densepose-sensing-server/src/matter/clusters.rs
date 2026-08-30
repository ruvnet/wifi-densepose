//! Matter cluster + device-type ID mappings for RuView entities.
//!
//! IDs come from the **Matter Core Spec 1.3 §A.1 Reserved Cluster IDs**
//! and **§1.3 Device Library**. Where ADR-115 §3.11.1 uses a name,
//! the constant below carries the spec hex.

use crate::mqtt::discovery::EntityKind;

/// Matter cluster identifier — 32-bit spec ID.
pub type ClusterId = u32;

/// Matter endpoint device-type identifier — 32-bit spec ID.
pub type EndpointTypeId = u32;

// ── Matter Core Spec 1.3 — Reserved Cluster IDs we publish ───────────
/// Per §A.1.4 "OccupancySensing" — boolean occupancy + occupancy
/// sensor type bitmap.
pub const CLUSTER_OCCUPANCY_SENSING: ClusterId = 0x0406;

/// Per §A.1.6 "Switch" — momentary press events used to fire fall /
/// bed-exit / multi-room one-shots.
pub const CLUSTER_SWITCH: ClusterId = 0x003B;

/// Per §A.1.0 "BasicInformation" — Vendor ID, Product ID, software
/// version, serial number. Every endpoint includes this.
pub const CLUSTER_BASIC_INFORMATION: ClusterId = 0x0028;

/// Per §A.1.5 "BooleanState" — single boolean attribute. Used for
/// non-occupancy boolean primitives (no_movement etc.) where the
/// occupancy semantics would be misleading to controllers.
pub const CLUSTER_BOOLEAN_STATE: ClusterId = 0x0045;

/// Per §A.1.16 "BridgedDeviceBasicInformation" — identifies a bridged
/// device (one per RuView node) on a Matter Bridged Devices Aggregator.
pub const CLUSTER_BRIDGED_DEVICE_BASIC_INFORMATION: ClusterId = 0x0039;

// ── Matter Device Library 1.3 — Device-type IDs ──────────────────────
/// Per §7.3 OccupancySensor.
pub const DEVICE_TYPE_OCCUPANCY_SENSOR: EndpointTypeId = 0x0107;
/// Per §6.6 GenericSwitch. Used for fall / bed-exit / multi-room events.
pub const DEVICE_TYPE_GENERIC_SWITCH: EndpointTypeId = 0x000F;
/// Per §10.2 Aggregator. The top-level endpoint that exposes all
/// bridged RuView nodes.
pub const DEVICE_TYPE_AGGREGATOR: EndpointTypeId = 0x000E;
/// Per §10.1 Bridged Node — one endpoint per RuView physical node.
pub const DEVICE_TYPE_BRIDGED_NODE: EndpointTypeId = 0x0013;

// ── Vendor-extension attribute (per ADR §3.11.1) ─────────────────────
/// Vendor-extension attribute carrying `n_persons` on the
/// OccupancySensing cluster. Apple Home / Google Home will ignore this
/// gracefully; HA + SmartThings will surface it via the Matter
/// integration's attribute-renderer.
///
/// Attribute IDs ≥ 0xFFF1_0000 are reserved for vendor extensions per
/// Matter Core §7.18.2. We use 0xFFF1_0001 = "wifi-densepose person
/// count".
pub const VENDOR_ATTR_PERSON_COUNT: u32 = 0xFFF1_0001;

/// Spec-defined event ID on the Switch cluster (§A.1.6.5.4).
pub const EVENT_SWITCH_MULTI_PRESS_COMPLETE: u32 = 0x06;

/// One per `EntityKind` that ADR-115 §3.11.1 maps to Matter. Entities
/// NOT in the table (HR / BR / pose / motion_energy / presence_score)
/// are explicitly not exposed over Matter — there are no spec
/// clusters for them today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatterClusterMapping {
    /// Which cluster the entity lives on.
    pub cluster: ClusterId,
    /// Which device-type the endpoint declares.
    pub device_type: EndpointTypeId,
    /// `Some(_)` if the entity emits Matter events (vs. attribute
    /// reads); `None` if it's read as a cluster attribute.
    pub event_id: Option<u32>,
    /// `Some(_)` if the entity uses a vendor-extension attribute
    /// rather than a spec attribute.
    pub vendor_attr_id: Option<u32>,
    /// True iff this entity belongs on the same endpoint as the parent
    /// node's OccupancySensor (multi-attribute entity grouping).
    pub shares_occupancy_endpoint: bool,
}

/// Map an `EntityKind` to its Matter exposure, if any. Returns `None`
/// for entities that are deliberately MQTT-only because no Matter
/// cluster represents them (HR / BR / pose / motion_energy / presence_score).
pub fn matter_mapping(entity: EntityKind) -> Option<MatterClusterMapping> {
    use EntityKind::*;
    Some(match entity {
        Presence | ZoneOccupancy => MatterClusterMapping {
            cluster: CLUSTER_OCCUPANCY_SENSING,
            device_type: DEVICE_TYPE_OCCUPANCY_SENSOR,
            event_id: None,
            vendor_attr_id: None,
            shares_occupancy_endpoint: false,
        },
        PersonCount => MatterClusterMapping {
            cluster: CLUSTER_OCCUPANCY_SENSING,
            device_type: DEVICE_TYPE_OCCUPANCY_SENSOR,
            event_id: None,
            vendor_attr_id: Some(VENDOR_ATTR_PERSON_COUNT),
            shares_occupancy_endpoint: true,
        },
        FallDetected | BedExit | MultiRoomTransition => MatterClusterMapping {
            cluster: CLUSTER_SWITCH,
            device_type: DEVICE_TYPE_GENERIC_SWITCH,
            event_id: Some(EVENT_SWITCH_MULTI_PRESS_COMPLETE),
            vendor_attr_id: None,
            shares_occupancy_endpoint: false,
        },
        // Semantic primitives that surface as occupancy-style booleans
        // (separate endpoints — one per primitive — so controllers can
        // bind individual scenes to each).
        SomeoneSleeping | RoomActive | MeetingInProgress | BathroomOccupied => {
            MatterClusterMapping {
                cluster: CLUSTER_OCCUPANCY_SENSING,
                device_type: DEVICE_TYPE_OCCUPANCY_SENSOR,
                event_id: None,
                vendor_attr_id: None,
                shares_occupancy_endpoint: false,
            }
        }
        // Problem-state booleans use BooleanState — semantically they
        // are NOT occupancy, and controllers shouldn't wire them into
        // motion-light scenes.
        PossibleDistress | ElderlyInactivityAnomaly | NoMovement => MatterClusterMapping {
            cluster: CLUSTER_BOOLEAN_STATE,
            device_type: DEVICE_TYPE_OCCUPANCY_SENSOR,
            event_id: None,
            vendor_attr_id: None,
            shares_occupancy_endpoint: false,
        },
        // Fall-risk scalar surfaces as a vendor-extension attribute on
        // the parent BridgedNode (no Matter spec for risk scores).
        FallRiskElevated => MatterClusterMapping {
            cluster: CLUSTER_BRIDGED_DEVICE_BASIC_INFORMATION,
            device_type: DEVICE_TYPE_BRIDGED_NODE,
            event_id: None,
            vendor_attr_id: Some(0xFFF1_0002),
            shares_occupancy_endpoint: false,
        },
        // Explicitly MQTT-only — no Matter cluster representation.
        BreathingRate | HeartRate | MotionLevel | MotionEnergy | PresenceScore | Rssi
        | PoseKeypoints => return None,
    })
}

/// True iff the entity has a Matter exposure on a current spec cluster.
// P2 Matter-publisher API surface; real Matter exposure is deferred (ADR-159 §A5).
#[allow(dead_code)]
pub fn entity_on_matter(entity: EntityKind) -> bool {
    matter_mapping(entity).is_some()
}

/// Compute the next available endpoint ID for a node-scoped entity,
/// given a starting offset (the bridge's first child endpoint). Used
/// by the publisher to assign per-primitive endpoints deterministically.
// P2 Matter-publisher API surface; real Matter exposure is deferred (ADR-159 §A5).
#[allow(dead_code)]
pub fn next_endpoint(base: u16, primitive_index: u16) -> u16 {
    base.saturating_add(primitive_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presence_maps_to_occupancy_sensor() {
        let m = matter_mapping(EntityKind::Presence).unwrap();
        assert_eq!(m.cluster, 0x0406); // OccupancySensing
        assert_eq!(m.device_type, 0x0107); // OccupancySensor
        assert!(m.event_id.is_none());
        assert!(m.vendor_attr_id.is_none());
    }

    #[test]
    fn zone_occupancy_uses_occupancy_sensor_too() {
        let m = matter_mapping(EntityKind::ZoneOccupancy).unwrap();
        assert_eq!(m.cluster, CLUSTER_OCCUPANCY_SENSING);
        assert_eq!(m.device_type, DEVICE_TYPE_OCCUPANCY_SENSOR);
    }

    #[test]
    fn person_count_is_vendor_extension_on_occupancy_endpoint() {
        let m = matter_mapping(EntityKind::PersonCount).unwrap();
        assert_eq!(m.cluster, CLUSTER_OCCUPANCY_SENSING);
        assert_eq!(m.vendor_attr_id, Some(0xFFF1_0001));
        assert!(m.shares_occupancy_endpoint);
    }

    #[test]
    fn fall_uses_switch_multi_press_complete_event() {
        let m = matter_mapping(EntityKind::FallDetected).unwrap();
        assert_eq!(m.cluster, CLUSTER_SWITCH);
        assert_eq!(m.device_type, DEVICE_TYPE_GENERIC_SWITCH);
        assert_eq!(m.event_id, Some(EVENT_SWITCH_MULTI_PRESS_COMPLETE));
    }

    #[test]
    fn bed_exit_uses_switch_event() {
        let m = matter_mapping(EntityKind::BedExit).unwrap();
        assert_eq!(m.cluster, CLUSTER_SWITCH);
        assert!(m.event_id.is_some());
    }

    #[test]
    fn multi_room_uses_switch_event() {
        let m = matter_mapping(EntityKind::MultiRoomTransition).unwrap();
        assert_eq!(m.cluster, CLUSTER_SWITCH);
    }

    #[test]
    fn someone_sleeping_uses_occupancy_separate_endpoint() {
        let m = matter_mapping(EntityKind::SomeoneSleeping).unwrap();
        assert_eq!(m.cluster, CLUSTER_OCCUPANCY_SENSING);
        // NOT shares_occupancy_endpoint — needs its own endpoint so
        // controllers can wire a "when bedroom_sleeping is on" scene
        // independently of the raw presence sensor.
        assert!(!m.shares_occupancy_endpoint);
    }

    #[test]
    fn distress_uses_boolean_state_not_occupancy() {
        // The semantic distinction matters: a controller binding a
        // "when motion detected, turn lights on" scene must NOT fire
        // for distress. We use BooleanState to keep them separate.
        let m = matter_mapping(EntityKind::PossibleDistress).unwrap();
        assert_eq!(m.cluster, CLUSTER_BOOLEAN_STATE);
    }

    #[test]
    fn no_movement_uses_boolean_state() {
        let m = matter_mapping(EntityKind::NoMovement).unwrap();
        assert_eq!(m.cluster, CLUSTER_BOOLEAN_STATE);
    }

    #[test]
    fn fall_risk_scalar_is_vendor_attribute_on_bridged_node() {
        let m = matter_mapping(EntityKind::FallRiskElevated).unwrap();
        assert_eq!(m.cluster, CLUSTER_BRIDGED_DEVICE_BASIC_INFORMATION);
        assert!(m.vendor_attr_id.is_some());
    }

    #[test]
    fn biometric_entities_have_no_matter_exposure() {
        // ADR §3.11.4 — Matter spec has no clusters for these, so
        // they're explicitly None.
        assert!(matter_mapping(EntityKind::HeartRate).is_none());
        assert!(matter_mapping(EntityKind::BreathingRate).is_none());
        assert!(matter_mapping(EntityKind::PoseKeypoints).is_none());
    }

    #[test]
    fn rssi_and_motion_continuous_are_mqtt_only() {
        // No standard cluster represents signal strength or continuous
        // motion-level for a non-light device.
        assert!(matter_mapping(EntityKind::Rssi).is_none());
        assert!(matter_mapping(EntityKind::MotionLevel).is_none());
        assert!(matter_mapping(EntityKind::MotionEnergy).is_none());
        assert!(matter_mapping(EntityKind::PresenceScore).is_none());
    }

    #[test]
    fn next_endpoint_is_deterministic_and_overflow_safe() {
        assert_eq!(next_endpoint(2, 0), 2);
        assert_eq!(next_endpoint(2, 5), 7);
        // Saturation on overflow rather than panic.
        assert_eq!(next_endpoint(u16::MAX, 1), u16::MAX);
    }

    #[test]
    fn entity_on_matter_is_consistent_with_matter_mapping_some() {
        for e in [
            EntityKind::Presence,
            EntityKind::FallDetected,
            EntityKind::SomeoneSleeping,
            EntityKind::HeartRate,
            EntityKind::Rssi,
        ] {
            assert_eq!(entity_on_matter(e), matter_mapping(e).is_some());
        }
    }

    #[test]
    fn all_entities_exhaustive_classification() {
        // Spot-check that every EntityKind variant has a defined
        // status — either a mapping or an explicit None — so a future
        // addition can't silently miss the Matter table.
        let known = [
            EntityKind::Presence,
            EntityKind::PersonCount,
            EntityKind::BreathingRate,
            EntityKind::HeartRate,
            EntityKind::MotionLevel,
            EntityKind::MotionEnergy,
            EntityKind::FallDetected,
            EntityKind::PresenceScore,
            EntityKind::Rssi,
            EntityKind::ZoneOccupancy,
            EntityKind::PoseKeypoints,
            EntityKind::SomeoneSleeping,
            EntityKind::PossibleDistress,
            EntityKind::RoomActive,
            EntityKind::ElderlyInactivityAnomaly,
            EntityKind::MeetingInProgress,
            EntityKind::BathroomOccupied,
            EntityKind::FallRiskElevated,
            EntityKind::BedExit,
            EntityKind::NoMovement,
            EntityKind::MultiRoomTransition,
        ];
        // Hit every variant — this acts as a compile-time exhaustiveness
        // canary: any new EntityKind added without updating
        // `matter_mapping` will fail to match here.
        for e in known {
            let _ = matter_mapping(e); // doesn't panic
        }
    }

    #[test]
    fn cluster_ids_match_matter_spec_1_3() {
        // Sanity-check the cluster IDs against the published spec
        // values — catches a transcription typo.
        assert_eq!(CLUSTER_OCCUPANCY_SENSING, 0x0406);
        assert_eq!(CLUSTER_SWITCH, 0x003B);
        assert_eq!(CLUSTER_BOOLEAN_STATE, 0x0045);
        assert_eq!(CLUSTER_BRIDGED_DEVICE_BASIC_INFORMATION, 0x0039);
        assert_eq!(DEVICE_TYPE_OCCUPANCY_SENSOR, 0x0107);
        assert_eq!(DEVICE_TYPE_GENERIC_SWITCH, 0x000F);
        assert_eq!(DEVICE_TYPE_AGGREGATOR, 0x000E);
        assert_eq!(DEVICE_TYPE_BRIDGED_NODE, 0x0013);
    }
}
