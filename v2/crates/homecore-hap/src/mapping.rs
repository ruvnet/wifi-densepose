//! HOMECORE entity → HAP accessory type + characteristic value mapping.
//!
//! Mirrors the HA `homekit` integration's mapping table
//! (homeassistant/components/homekit/type_*.py) for the entity domains and
//! device classes handled in P1.

use serde_json::Value;

use homecore::entity::{EntityId, State};

use crate::accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};
use crate::error::HapError;

/// Result of mapping one HOMECORE entity state to the HAP layer.
#[derive(Debug, Clone)]
pub struct AccessoryMapping {
    /// HAP service type to advertise for this entity.
    pub accessory_type: HapAccessoryType,
    /// Characteristic key/value pairs to set on the HAP service.
    pub characteristics: Vec<(HapCharacteristic, HapCharacteristicValue)>,
}

/// Maps a HOMECORE entity `(EntityId, State)` pair to a `HapAccessoryType`
/// and its current characteristic values.
///
/// Rule table (mirrors HA homekit_controller mapping):
///
/// | Domain | device_class | HAP service |
/// |--------|-------------|-------------|
/// | `light` | — | Lightbulb |
/// | `switch` | — | Switch |
/// | `binary_sensor` | `occupancy` | OccupancySensor |
/// | `binary_sensor` | `motion` | MotionSensor |
/// | `binary_sensor` | `door` / `window` | ContactSensor |
/// | `sensor` | — + unit=°C/°F | TemperatureSensor |
/// | `sensor` | — + unit=% (humidity) | HumiditySensor |
/// | `cover` (door) | — | Door |
/// | `lock` | — | Lock |
pub struct EntityToAccessoryMapper;

impl EntityToAccessoryMapper {
    /// Map a HOMECORE entity to its HAP representation.
    ///
    /// Returns `HapError::UnmappableEntity` for domains that have no
    /// defined HAP mapping (e.g. `automation`, `input_boolean`).
    pub fn map(entity_id: &EntityId, state: &State) -> Result<AccessoryMapping, HapError> {
        match entity_id.domain() {
            "light" => Self::map_light(state),
            "switch" => Self::map_switch(state),
            "binary_sensor" => Self::map_binary_sensor(entity_id, state),
            "sensor" => Self::map_sensor(entity_id, state),
            "cover" => Self::map_cover(state),
            "lock" => Self::map_lock(state),
            other => Err(HapError::UnmappableEntity {
                entity_id: entity_id.as_str().to_owned(),
                reason: format!("domain '{other}' has no HAP mapping in P1"),
            }),
        }
    }

    fn map_light(state: &State) -> Result<AccessoryMapping, HapError> {
        let on = state.state == "on";
        let mut chars = vec![(HapCharacteristic::On, HapCharacteristicValue::Bool(on))];
        if let Some(b) = state.attributes.get("brightness").and_then(Value::as_u64) {
            chars.push((
                HapCharacteristic::Brightness,
                HapCharacteristicValue::UInt8(b.min(255) as u8),
            ));
        }
        Ok(AccessoryMapping { accessory_type: HapAccessoryType::Lightbulb, characteristics: chars })
    }

    fn map_switch(state: &State) -> Result<AccessoryMapping, HapError> {
        let on = state.state == "on";
        Ok(AccessoryMapping {
            accessory_type: HapAccessoryType::Switch,
            characteristics: vec![(HapCharacteristic::On, HapCharacteristicValue::Bool(on))],
        })
    }

    fn map_binary_sensor(
        entity_id: &EntityId,
        state: &State,
    ) -> Result<AccessoryMapping, HapError> {
        let detected = state.state == "on";
        let device_class = state
            .attributes
            .get("device_class")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();

        // Also check name heuristics for device_class-less entities.
        let name = entity_id.name();
        let is_occupancy = device_class == "occupancy" || name.contains("occupancy") || name.contains("presence");
        let is_motion = device_class == "motion" || name.contains("motion");
        let is_door = device_class == "door" || device_class == "window";

        if is_occupancy {
            return Ok(AccessoryMapping {
                accessory_type: HapAccessoryType::OccupancySensor,
                characteristics: vec![(
                    HapCharacteristic::OccupancyDetected,
                    HapCharacteristicValue::UInt8(if detected { 1 } else { 0 }),
                )],
            });
        }
        if is_motion {
            return Ok(AccessoryMapping {
                accessory_type: HapAccessoryType::MotionSensor,
                characteristics: vec![(
                    HapCharacteristic::MotionDetected,
                    HapCharacteristicValue::Bool(detected),
                )],
            });
        }
        if is_door {
            return Ok(AccessoryMapping {
                accessory_type: HapAccessoryType::ContactSensor,
                characteristics: vec![(
                    HapCharacteristic::ContactSensorState,
                    HapCharacteristicValue::UInt8(if detected { 1 } else { 0 }),
                )],
            });
        }
        // Fallback: treat as motion sensor
        Ok(AccessoryMapping {
            accessory_type: HapAccessoryType::MotionSensor,
            characteristics: vec![(
                HapCharacteristic::MotionDetected,
                HapCharacteristicValue::Bool(detected),
            )],
        })
    }

    fn map_sensor(entity_id: &EntityId, state: &State) -> Result<AccessoryMapping, HapError> {
        let unit = state
            .attributes
            .get("unit_of_measurement")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let name = entity_id.name();

        let is_temp = unit == "°C" || unit == "°F" || unit == "C" || unit == "F"
            || name.contains("temp") || name.contains("temperature");
        let is_humidity = unit == "%" && (name.contains("humid") || name.contains("rh"));

        if is_temp {
            let temp: f64 = state.state.parse().unwrap_or(0.0);
            return Ok(AccessoryMapping {
                accessory_type: HapAccessoryType::TemperatureSensor,
                characteristics: vec![(
                    HapCharacteristic::CurrentTemperature,
                    HapCharacteristicValue::Float(temp),
                )],
            });
        }
        if is_humidity {
            let hum: f64 = state.state.parse().unwrap_or(0.0);
            return Ok(AccessoryMapping {
                accessory_type: HapAccessoryType::HumiditySensor,
                characteristics: vec![(
                    HapCharacteristic::CurrentRelativeHumidity,
                    HapCharacteristicValue::Float(hum),
                )],
            });
        }
        Err(HapError::UnmappableEntity {
            entity_id: entity_id.as_str().to_owned(),
            reason: "sensor unit/name not recognised as temperature or humidity".into(),
        })
    }

    fn map_cover(state: &State) -> Result<AccessoryMapping, HapError> {
        let door_state: u8 = match state.state.as_str() {
            "open" => 0,
            "opening" => 2,
            "closing" => 3,
            _ => 1, // closed
        };
        Ok(AccessoryMapping {
            accessory_type: HapAccessoryType::Door,
            characteristics: vec![(
                HapCharacteristic::CurrentDoorState,
                HapCharacteristicValue::UInt8(door_state),
            )],
        })
    }

    fn map_lock(state: &State) -> Result<AccessoryMapping, HapError> {
        let lock_state: u8 = match state.state.as_str() {
            "unlocked" => 0,
            "locked" => 1,
            _ => 3, // unknown
        };
        Ok(AccessoryMapping {
            accessory_type: HapAccessoryType::Lock,
            characteristics: vec![(
                HapCharacteristic::LockCurrentState,
                HapCharacteristicValue::UInt8(lock_state),
            )],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use homecore::entity::{EntityId, State};
    use homecore::event::Context;

    fn state(id: &str, st: &str, attrs: serde_json::Value) -> (EntityId, State) {
        let eid = EntityId::parse(id).unwrap();
        let s = State::new(eid.clone(), st, attrs, Context::default());
        (eid, s)
    }

    #[test]
    fn light_kitchen_on_with_brightness() {
        let (eid, s) = state(
            "light.kitchen",
            "on",
            serde_json::json!({"brightness": 200}),
        );
        let mapping = EntityToAccessoryMapper::map(&eid, &s).unwrap();
        assert_eq!(mapping.accessory_type, HapAccessoryType::Lightbulb);
        assert!(mapping.characteristics.contains(&(
            HapCharacteristic::On,
            HapCharacteristicValue::Bool(true)
        )));
        assert!(mapping.characteristics.contains(&(
            HapCharacteristic::Brightness,
            HapCharacteristicValue::UInt8(200)
        )));
    }

    #[test]
    fn binary_sensor_occupancy_device_class() {
        let (eid, s) = state(
            "binary_sensor.kitchen_presence",
            "on",
            serde_json::json!({"device_class": "occupancy"}),
        );
        let mapping = EntityToAccessoryMapper::map(&eid, &s).unwrap();
        assert_eq!(mapping.accessory_type, HapAccessoryType::OccupancySensor);
        assert!(mapping.characteristics.contains(&(
            HapCharacteristic::OccupancyDetected,
            HapCharacteristicValue::UInt8(1)
        )));
    }

    #[test]
    fn sensor_outdoor_temp_celsius() {
        let (eid, s) = state(
            "sensor.outdoor_temp",
            "21.5",
            serde_json::json!({"unit_of_measurement": "°C"}),
        );
        let mapping = EntityToAccessoryMapper::map(&eid, &s).unwrap();
        assert_eq!(mapping.accessory_type, HapAccessoryType::TemperatureSensor);
        assert!(mapping.characteristics.contains(&(
            HapCharacteristic::CurrentTemperature,
            HapCharacteristicValue::Float(21.5)
        )));
    }

    #[test]
    fn unmappable_domain_returns_error() {
        let (eid, s) = state("automation.morning", "on", serde_json::json!({}));
        assert!(EntityToAccessoryMapper::map(&eid, &s).is_err());
    }
}
