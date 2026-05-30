//! Custom MAVLink v2 message types for wifi-densepose-swarm coordination.
//!
//! Message IDs follow MAVLink custom dialect convention (50000+).
//! All messages are signed via `security::mavlink_signing::MavlinkSigner`.

use serde::{Deserialize, Serialize};
use crate::types::{NodeId, Position3D, CsiDetection};

/// MAVLink message ID base for swarm custom dialect.
pub const SWARM_DIALECT_BASE: u32 = 50000;

/// Message IDs for swarm custom messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwarmMsgId {
    /// Swarm node kinematic state broadcast (50000).
    NodeState = 50000,
    /// CSI detection report from sensing payload (50001).
    CsiReport = 50001,
    /// Task assignment from cluster head to worker (50002).
    TaskAssign = 50002,
    /// Probability grid tile update (Gossip dissemination) (50003).
    GridTileUpdate = 50003,
    /// Cluster head heartbeat + Raft term (50004).
    ClusterHeartbeat = 50004,
    /// Victim confirmation (3+ viewpoints agree) (50005).
    VictimConfirmed = 50005,
}

/// SWARM_NODE_STATE (50000): broadcast by each drone every 100 ms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmNodeState {
    /// Sending node ID.
    pub node_id: u32,
    /// North position in local NED frame (m × 1000 = mm).
    pub pos_north_mm: i32,
    /// East position (mm).
    pub pos_east_mm: i32,
    /// Down position (mm, negative = above ground).
    pub pos_down_mm: i32,
    /// Speed m/s × 100.
    pub speed_cm_s: u16,
    /// Heading degrees × 100 (0–36000).
    pub heading_cdeg: u16,
    /// Battery percent × 10 (0–1000).
    pub battery_10th_pct: u16,
    /// Link quality 0–255 (255 = perfect).
    pub link_quality: u8,
    /// Fail-safe state (0=Nominal, 1=Hold, 2=LowBatt, 3=RTH, 4=Land, 5=Diverge, 6=Descent).
    pub failsafe_state: u8,
    /// Timestamp ms (wraps at u32 max, ~49 days).
    pub timestamp_ms: u32,
}

impl SwarmNodeState {
    pub fn from_drone_state(state: &crate::types::DroneState, failsafe: u8) -> Self {
        Self {
            node_id: state.id.0,
            pos_north_mm: (state.position.x * 1000.0) as i32,
            pos_east_mm: (state.position.y * 1000.0) as i32,
            pos_down_mm: (state.position.z * 1000.0) as i32,
            speed_cm_s: (state.velocity.magnitude() * 100.0) as u16,
            heading_cdeg: ((state.heading_rad.to_degrees().rem_euclid(360.0)) * 100.0) as u16,
            battery_10th_pct: (state.battery_pct * 10.0) as u16,
            link_quality: (state.link_quality * 255.0) as u8,
            failsafe_state: failsafe,
            timestamp_ms: state.timestamp_ms as u32,
        }
    }

    /// Encode to 20-byte MAVLink payload (fixed-length for efficiency).
    pub fn encode(&self) -> [u8; 20] {
        let mut buf = [0u8; 20];
        buf[0..4].copy_from_slice(&self.node_id.to_le_bytes());
        buf[4..8].copy_from_slice(&self.pos_north_mm.to_le_bytes());
        buf[8..12].copy_from_slice(&self.pos_east_mm.to_le_bytes());
        buf[12..16].copy_from_slice(&self.pos_down_mm.to_le_bytes());
        buf[16] = self.failsafe_state;
        buf[17] = self.link_quality;
        buf[18..20].copy_from_slice(&self.battery_10th_pct.to_le_bytes());
        buf
    }

    /// Decode from 20-byte MAVLink payload.
    pub fn decode(buf: &[u8; 20]) -> Self {
        Self {
            node_id: u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            pos_north_mm: i32::from_le_bytes(buf[4..8].try_into().unwrap()),
            pos_east_mm: i32::from_le_bytes(buf[8..12].try_into().unwrap()),
            pos_down_mm: i32::from_le_bytes(buf[12..16].try_into().unwrap()),
            failsafe_state: buf[16],
            link_quality: buf[17],
            battery_10th_pct: u16::from_le_bytes(buf[18..20].try_into().unwrap()),
            speed_cm_s: 0,
            heading_cdeg: 0,
            timestamp_ms: 0,
        }
    }
}

/// SWARM_CSI_REPORT (50001): sent by sensing payload when detection confidence > threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCsiReport {
    pub node_id: u32,
    pub confidence_u8: u8,       // confidence × 255
    pub has_position: bool,
    pub victim_north_mm: i32,    // estimated victim position
    pub victim_east_mm: i32,
    pub victim_down_mm: i32,
    pub timestamp_ms: u32,
}

impl SwarmCsiReport {
    pub fn from_detection(det: &CsiDetection) -> Self {
        let (n, e, d) = det.victim_position
            .map(|p| ((p.x * 1000.0) as i32, (p.y * 1000.0) as i32, (p.z * 1000.0) as i32))
            .unwrap_or((0, 0, 0));
        Self {
            node_id: det.drone_id.0,
            confidence_u8: (det.confidence * 255.0) as u8,
            has_position: det.victim_position.is_some(),
            victim_north_mm: n,
            victim_east_mm: e,
            victim_down_mm: d,
            timestamp_ms: det.timestamp_ms as u32,
        }
    }

    pub fn to_detection(&self) -> CsiDetection {
        CsiDetection {
            drone_id: NodeId(self.node_id),
            confidence: self.confidence_u8 as f32 / 255.0,
            victim_position: if self.has_position {
                Some(Position3D {
                    x: self.victim_north_mm as f64 / 1000.0,
                    y: self.victim_east_mm as f64 / 1000.0,
                    z: self.victim_down_mm as f64 / 1000.0,
                })
            } else {
                None
            },
            timestamp_ms: self.timestamp_ms as u64,
        }
    }
}

/// SWARM_CLUSTER_HEARTBEAT (50004): Raft leader heartbeat.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmClusterHeartbeat {
    pub leader_id: u32,
    pub raft_term: u64,
    pub cluster_size: u8,
    pub active_drones: u8,
    pub mission_phase: u8,       // 0=Systematic, 1=ProbabilisticPursuit, 2=Convergence
    pub timestamp_ms: u32,
}

/// SWARM_VICTIM_CONFIRMED (50005): 3+ viewpoints confirm victim location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmVictimConfirmed {
    pub victim_id: u8,           // sequential victim counter
    pub victim_north_mm: i32,
    pub victim_east_mm: i32,
    pub victim_down_mm: i32,
    pub uncertainty_mm: u16,     // localization uncertainty in mm
    pub contributing_drones: u8, // bitmask (drone 0 = bit 0)
    pub fused_confidence_u8: u8,
    pub timestamp_ms: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DroneState, NodeId, Velocity3D};

    fn make_state() -> DroneState {
        DroneState {
            id: NodeId(3),
            position: Position3D { x: 100.5, y: 200.25, z: -30.0 },
            velocity: Velocity3D { vx: 5.0, vy: 0.0, vz: 0.0 },
            heading_rad: std::f64::consts::PI / 4.0,
            altitude_agl_m: 30.0,
            battery_pct: 78.5,
            link_quality: 0.92,
            timestamp_ms: 12345,
        }
    }

    #[test]
    fn test_node_state_encode_decode_roundtrip() {
        let state = make_state();
        let msg = SwarmNodeState::from_drone_state(&state, 0);
        let encoded = msg.encode();
        let decoded = SwarmNodeState::decode(&encoded);
        assert_eq!(decoded.node_id, 3);
        assert_eq!(decoded.pos_north_mm, 100500);  // 100.5 m × 1000
        assert_eq!(decoded.failsafe_state, 0);
    }

    #[test]
    fn test_csi_report_roundtrip() {
        let det = CsiDetection {
            drone_id: NodeId(1),
            confidence: 0.85,
            victim_position: Some(Position3D { x: 50.0, y: 75.0, z: 0.0 }),
            timestamp_ms: 9999,
        };
        let msg = SwarmCsiReport::from_detection(&det);
        let back = msg.to_detection();
        assert!((back.confidence - 0.85).abs() < 0.01, "confidence roundtrip");
        let vp = back.victim_position.unwrap();
        assert!((vp.x - 50.0).abs() < 0.001);
        assert!((vp.y - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_battery_encoding() {
        let mut state = make_state();
        state.battery_pct = 50.0;
        let msg = SwarmNodeState::from_drone_state(&state, 0);
        assert_eq!(msg.battery_10th_pct, 500); // 50% × 10
    }
}
