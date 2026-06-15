//! ASTM F3411 Remote ID — **Basic ID message only** (ADR-159 §A3).
//!
//! Only the Basic ID message (`encode_basic_id`) is implemented. The
//! Location/Vector message is **not** encoded yet because the drone position is
//! tracked in a local NED frame (north/east metres relative to a takeoff datum),
//! and a compliant Location/Vector message requires WGS84 latitude/longitude.
//! Broadcasting NED metres in lat/lon fields would emit physically-impossible
//! coordinates (e.g. "latitude = 12.4 metres"), so we deliberately keep the
//! drone position in honest `drone_north_m` / `drone_east_m` fields until a real
//! local-tangent-plane NED→WGS84 transform (with an operator datum) lands. See
//! the `ACCEPTED-FUTURE` note in ADR-159 §A3.

use crate::types::DroneState;
use serde::{Deserialize, Serialize};

/// Remote ID broadcast state for one drone.
///
/// Drone position is stored as **NED metres** (`drone_north_m` / `drone_east_m`)
/// relative to the operator/takeoff datum — *not* WGS84 lat/lon — because no
/// datum-anchored geodetic transform is wired yet. The operator position is true
/// WGS84 (it comes from the operator's GNSS, not the local frame).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteIdBroadcast {
    pub uas_id: [u8; 20],        // 20-byte UAS ID (ANSI/CTA-2063-A)
    /// Operator latitude (WGS84 degrees) — real geodetic position.
    pub operator_lat: f64,
    /// Operator longitude (WGS84 degrees) — real geodetic position.
    pub operator_lon: f64,
    /// Drone north offset in **metres** from the operator/takeoff datum (NED x).
    /// NOT a latitude. See module docs — Location/Vector encoding is deferred
    /// until a real NED→WGS84 transform exists.
    pub drone_north_m: f64,
    /// Drone east offset in **metres** from the operator/takeoff datum (NED y).
    /// NOT a longitude.
    pub drone_east_m: f64,
    pub altitude_msl_m: f32,
    pub speed_ms: f32,
    pub heading_deg: f32,
    pub timestamp_ms: u64,
    pub emergency_status: bool,
}

impl RemoteIdBroadcast {
    pub fn new(uas_id: [u8; 20]) -> Self {
        Self {
            uas_id,
            operator_lat: 0.0,
            operator_lon: 0.0,
            drone_north_m: 0.0,
            drone_east_m: 0.0,
            altitude_msl_m: 0.0,
            speed_ms: 0.0,
            heading_deg: 0.0,
            timestamp_ms: 0,
            emergency_status: false,
        }
    }

    /// Update from a drone state and operator position.
    ///
    /// The drone position is stored as honest NED metres — we do **not** fake a
    /// lat/lon from a local-frame offset. The operator position is true WGS84.
    pub fn update(&mut self, state: &DroneState, operator_pos: (f64, f64)) {
        // NED metres, stored as-is in metre-typed fields (no fabricated geodetic
        // coordinates). A future Location/Vector encoder must transform these
        // through a datum-anchored NED→WGS84 projection before broadcast.
        self.drone_north_m = state.position.x; // NED x = north offset, metres
        self.drone_east_m = state.position.y; // NED y = east offset, metres
        self.altitude_msl_m = state.altitude_agl_m as f32;
        self.speed_ms = state.velocity.magnitude() as f32;
        self.heading_deg = state.heading_rad.to_degrees() as f32;
        self.timestamp_ms = state.timestamp_ms;
        self.operator_lat = operator_pos.0;
        self.operator_lon = operator_pos.1;
    }

    /// Encode a 25-byte ASTM F3411 Basic ID message.
    /// Format: [message_type(1)] [id_type(1)] [uas_id(20)] [reserved(3)]
    pub fn encode_basic_id(&self) -> [u8; 25] {
        let mut buf = [0u8; 25];
        buf[0] = 0x00; // Message type: Basic ID
        buf[1] = 0x01; // ID type: Serial Number
        buf[2..22].copy_from_slice(&self.uas_id);
        // bytes 22-24: reserved
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_basic_id_length() {
        let rid = RemoteIdBroadcast::new([0x41u8; 20]);
        let buf = rid.encode_basic_id();
        assert_eq!(buf.len(), 25);
        assert_eq!(buf[1], 0x01); // ID type: serial number
    }

    #[test]
    fn test_uas_id_in_encoded_buffer() {
        let mut id = [0u8; 20];
        id[0] = 0xFF;
        let rid = RemoteIdBroadcast::new(id);
        let buf = rid.encode_basic_id();
        assert_eq!(buf[2], 0xFF);
    }

    /// ADR-159 §A3 — a known NED offset must land in honest **metre** fields,
    /// never in WGS84 lat/lon fields (which would broadcast physically-impossible
    /// coordinates like "latitude = 37.5 m"). Fails on old code, where the same
    /// values were stored into `drone_lat`/`drone_lon`.
    #[test]
    fn test_ned_offset_stored_as_metres_not_latlon() {
        use crate::types::{DroneState, NodeId, Position3D};

        let mut state = DroneState::default_at_origin(NodeId(7));
        // 37.5 m north, -12.0 m east of the takeoff datum.
        state.position = Position3D {
            x: 37.5,
            y: -12.0,
            z: 5.0,
        };
        let mut rid = RemoteIdBroadcast::new([0x41u8; 20]);
        // Operator at a real WGS84 fix (San Francisco-ish).
        rid.update(&state, (37.7749, -122.4194));

        // Drone offset is honest NED metres.
        assert_eq!(rid.drone_north_m, 37.5);
        assert_eq!(rid.drone_east_m, -12.0);

        // Operator position is the real geodetic fix and is plausibly a lat/lon.
        assert!((-90.0..=90.0).contains(&rid.operator_lat));
        assert!((-180.0..=180.0).contains(&rid.operator_lon));
        assert!((rid.operator_lat - 37.7749).abs() < 1e-9);

        // The drone NED metres would have been an out-of-range "latitude" only
        // if a value happened to exceed 90 — but the contract is the field name
        // itself: these are metres, not degrees. A future Location/Vector
        // encoder must project them through a real NED→WGS84 transform.
    }
}
