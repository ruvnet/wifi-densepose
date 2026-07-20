//! Transceiver-geometry records (ADR-152 §2.1.1, extends ADR-151 Stage 2).
//!
//! PerceptAlign (ADR-152 F1) diagnosed "coordinate overfitting": pose heads
//! trained without an explicit layout model memorise the deployment-specific
//! transceiver geometry and break in unseen rooms. The first, cheap half of
//! the fix is to *record* the geometry at enrollment so every specialist bank
//! knows the layout it was trained under.
//!
//! This module is the record only. The learned geometry *embeddings* that
//! condition specialist heads (ADR-152 §2.1.2) are out of scope until the
//! ADR-151 P6 LoRA heads exist — statistical specialists ignore geometry.
//!
//! Every field is optional **by design**: geometry is captured when the
//! operator knows it (tape measure, checkerboard calibration, installer
//! floor plan) and omitted when they don't. An all-unknown record is still
//! useful — it pins down *which* nodes existed and that geometry was not
//! measured, rather than leaving the question open.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Estimated node position in the room frame (meters).
///
/// The room frame is whatever frame the recording `method` defines (e.g. a
/// tape-measure origin at a room corner, or the shared 3D frame of the
/// two-checkerboard alignment, ADR-152 §2.1.3). Consistency *within* one
/// enrollment is what matters; there is no global frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PositionEstimate {
    /// X coordinate (meters).
    pub x_m: f32,
    /// Y coordinate (meters).
    pub y_m: f32,
    /// Z coordinate / height (meters).
    pub z_m: f32,
}

/// Antenna boresight orientation (radians, room frame).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AntennaOrientation {
    /// Azimuth from the room frame's +X axis, counter-clockwise (radians).
    pub azimuth_rad: f32,
    /// Elevation above the horizontal plane (radians).
    pub elevation_rad: f32,
}

fn unknown_method() -> String {
    "unknown".to_string()
}

/// Per-node transceiver geometry recorded at enrollment (ADR-152 §2.1.1).
///
/// Stored in the [`EnrollmentSession`](crate::EnrollmentSession) event log and
/// snapshotted into the [`SpecialistBank`](crate::SpecialistBank), so a bank
/// always carries the layout it was trained under. Schema-versioned: banks and
/// sessions persisted before this record existed deserialize with no geometry
/// (serde defaults), same pattern as `PresenceSpecialist::mean_dist_threshold`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeGeometry {
    /// Node this record describes (same id space as the multistatic fusion).
    pub node_id: u8,
    /// Estimated position, if measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<PositionEstimate>,
    /// Antenna orientation, if measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<AntennaOrientation>,
    /// Known distances to other nodes (node_id → meters). Empty = not measured.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub distances_m: BTreeMap<u8, f32>,
    /// How the geometry was obtained — free-form provenance, e.g.
    /// `"tape-measure"`, `"checkerboard"`, `"floor-plan"`, `"unknown"`.
    #[serde(default = "unknown_method")]
    pub method: String,
}

impl NodeGeometry {
    /// A record with everything unknown except the node id.
    pub fn unknown(node_id: u8) -> Self {
        Self::new(node_id, "unknown")
    }

    /// A record with no measurements yet, tagged with its provenance method.
    pub fn new(node_id: u8, method: impl Into<String>) -> Self {
        Self {
            node_id,
            position: None,
            orientation: None,
            distances_m: BTreeMap::new(),
            method: method.into(),
        }
    }

    /// Set the position estimate (builder style).
    pub fn with_position(mut self, x_m: f32, y_m: f32, z_m: f32) -> Self {
        self.position = Some(PositionEstimate { x_m, y_m, z_m });
        self
    }

    /// Set the antenna orientation (builder style).
    pub fn with_orientation(mut self, azimuth_rad: f32, elevation_rad: f32) -> Self {
        self.orientation = Some(AntennaOrientation {
            azimuth_rad,
            elevation_rad,
        });
        self
    }

    /// Record a known distance to another node (builder style).
    pub fn with_distance(mut self, other_node_id: u8, meters: f32) -> Self {
        self.distances_m.insert(other_node_id, meters);
        self
    }

    /// `true` when nothing beyond the node id was measured.
    pub fn is_unmeasured(&self) -> bool {
        self.position.is_none() && self.orientation.is_none() && self.distances_m.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_record_roundtrips() {
        let g = NodeGeometry::new(1, "tape-measure")
            .with_position(0.5, 2.0, 1.2)
            .with_orientation(std::f32::consts::FRAC_PI_2, 0.0)
            .with_distance(2, 3.4);
        let json = serde_json::to_string(&g).unwrap();
        let back: NodeGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, g);
        assert_eq!(back.distances_m.get(&2), Some(&3.4));
        assert!(!back.is_unmeasured());
    }

    #[test]
    fn all_optional_empty_roundtrips() {
        let g = NodeGeometry::unknown(7);
        assert!(g.is_unmeasured());
        let json = serde_json::to_string(&g).unwrap();
        // Optional fields must be omitted, not serialized as null/empty.
        assert!(!json.contains("position"));
        assert!(!json.contains("orientation"));
        assert!(!json.contains("distances_m"));
        let back: NodeGeometry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, g);
        assert_eq!(back.method, "unknown");
    }

    #[test]
    fn minimal_json_defaults_cleanly() {
        // A record written by a producer that only knew the node id.
        let g: NodeGeometry = serde_json::from_str(r#"{"node_id":3}"#).unwrap();
        assert_eq!(g.node_id, 3);
        assert!(g.is_unmeasured());
        assert_eq!(g.method, "unknown");
    }
}
