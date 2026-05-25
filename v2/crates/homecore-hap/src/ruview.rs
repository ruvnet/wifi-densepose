//! RuView sensing primitives → HAP characteristic mapping (ADR-125 §2.1.d).
//!
//! Per ADR-125, RuView's privacy-class-2/3 events map to HomeKit primitives
//! as semantic ambient signals, not surveillance events:
//!
//! | RuView primitive | HAP service | Rationale |
//! |-----------------|-------------|-----------|
//! | `edge_vitals.presence` | OccupancySensor | Anonymous presence = occupancy |
//! | `edge_vitals.motion` | MotionSensor | Motion burst |
//! | `edge_vitals.fall_detected` | LeakSensor | HA convention: abnormal events |
//! | `edge_vitals.breathing_present` | OccupancySensor | Sleep-room occupancy |
//!
//! Raw `identity_risk_score`, `rf_signature_hash`, and class-0 BFI data are
//! **never** mapped. Structural invariant I1 (ADR-118 §2.2) is enforced here.

use crate::accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};
use crate::mapping::AccessoryMapping;

/// Parsed RuView edge vitals event from the sensing-server.
///
/// All fields are class-2 (Anonymous) or class-3 (Restricted) derived signals.
/// Raw BFI / `identity_risk_score` / `rf_signature_hash` are intentionally
/// absent — they must not cross the HAP boundary per ADR-125 §2.2.
#[derive(Debug, Clone, Default)]
pub struct EdgeVitals {
    /// True if at least one person is present in the sensing zone.
    pub presence: bool,
    /// True if motion was detected in the last sensing window.
    pub motion: bool,
    /// True if a fall event was detected (latched, 5 s cooldown).
    pub fall_detected: bool,
    /// True if rhythmic breathing is detected (sleep-room occupancy signal).
    pub breathing_present: bool,
    /// Optional ambient temperature reading (°C), forwarded if available
    /// from a co-located temperature sensor.
    pub ambient_temp_c: Option<f64>,
}

/// Maps `EdgeVitals` to a `Vec<AccessoryMapping>` — one per RuView primitive
/// that should be exposed as a distinct HAP service (child accessory).
pub struct RuViewToHapMapper;

impl RuViewToHapMapper {
    /// Convert a `EdgeVitals` snapshot to HAP accessory mappings.
    ///
    /// Always returns mappings for presence, motion, and fall; the ambient
    /// temperature mapping is only emitted when `ambient_temp_c` is `Some`.
    pub fn map(vitals: &EdgeVitals) -> Vec<AccessoryMapping> {
        let mut out = Vec::with_capacity(4);

        // Presence → OccupancySensor
        out.push(AccessoryMapping {
            accessory_type: HapAccessoryType::OccupancySensor,
            characteristics: vec![(
                HapCharacteristic::OccupancyDetected,
                HapCharacteristicValue::UInt8(if vitals.presence || vitals.breathing_present { 1 } else { 0 }),
            )],
        });

        // Motion → MotionSensor
        out.push(AccessoryMapping {
            accessory_type: HapAccessoryType::MotionSensor,
            characteristics: vec![(
                HapCharacteristic::MotionDetected,
                HapCharacteristicValue::Bool(vitals.motion),
            )],
        });

        // Fall detected → LeakSensor (HA homekit_controller convention for
        // "abnormal event" — not a literal water leak, but an automation-
        // triggerable threshold event, per ADR-125 §2.1.d).
        out.push(AccessoryMapping {
            accessory_type: HapAccessoryType::LeakSensor,
            characteristics: vec![(
                HapCharacteristic::LeakDetected,
                HapCharacteristicValue::UInt8(if vitals.fall_detected { 1 } else { 0 }),
            )],
        });

        // Optional temperature
        if let Some(temp) = vitals.ambient_temp_c {
            out.push(AccessoryMapping {
                accessory_type: HapAccessoryType::TemperatureSensor,
                characteristics: vec![(
                    HapCharacteristic::CurrentTemperature,
                    HapCharacteristicValue::Float(temp),
                )],
            });
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};

    #[test]
    fn presence_true_maps_to_occupancy_detected_1() {
        let vitals = EdgeVitals { presence: true, ..Default::default() };
        let mappings = RuViewToHapMapper::map(&vitals);
        let occ = mappings.iter().find(|m| m.accessory_type == HapAccessoryType::OccupancySensor).unwrap();
        assert!(occ.characteristics.contains(&(
            HapCharacteristic::OccupancyDetected,
            HapCharacteristicValue::UInt8(1)
        )));
    }

    #[test]
    fn fall_detected_maps_to_leak_sensor() {
        let vitals = EdgeVitals { fall_detected: true, ..Default::default() };
        let mappings = RuViewToHapMapper::map(&vitals);
        let leak = mappings.iter().find(|m| m.accessory_type == HapAccessoryType::LeakSensor).unwrap();
        assert!(leak.characteristics.contains(&(
            HapCharacteristic::LeakDetected,
            HapCharacteristicValue::UInt8(1)
        )));
    }

    #[test]
    fn motion_false_maps_correctly() {
        let vitals = EdgeVitals { motion: false, ..Default::default() };
        let mappings = RuViewToHapMapper::map(&vitals);
        let mot = mappings.iter().find(|m| m.accessory_type == HapAccessoryType::MotionSensor).unwrap();
        assert!(mot.characteristics.contains(&(
            HapCharacteristic::MotionDetected,
            HapCharacteristicValue::Bool(false)
        )));
    }

    #[test]
    fn ambient_temp_emits_temperature_mapping() {
        let vitals = EdgeVitals { ambient_temp_c: Some(22.5), ..Default::default() };
        let mappings = RuViewToHapMapper::map(&vitals);
        let temp = mappings.iter().find(|m| m.accessory_type == HapAccessoryType::TemperatureSensor);
        assert!(temp.is_some());
    }

    #[test]
    fn no_ambient_temp_omits_temperature_mapping() {
        let vitals = EdgeVitals { ambient_temp_c: None, ..Default::default() };
        let mappings = RuViewToHapMapper::map(&vitals);
        assert!(mappings.iter().all(|m| m.accessory_type != HapAccessoryType::TemperatureSensor));
    }

    #[test]
    fn breathing_present_triggers_occupancy() {
        let vitals = EdgeVitals { presence: false, breathing_present: true, ..Default::default() };
        let mappings = RuViewToHapMapper::map(&vitals);
        let occ = mappings.iter().find(|m| m.accessory_type == HapAccessoryType::OccupancySensor).unwrap();
        assert!(occ.characteristics.contains(&(
            HapCharacteristic::OccupancyDetected,
            HapCharacteristicValue::UInt8(1)
        )));
    }
}
