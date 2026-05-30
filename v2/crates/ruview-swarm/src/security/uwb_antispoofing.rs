//! UWB-based GPS anti-spoofing: cross-validates GPS position against UWB ranging.

use crate::types::{NodeId, Position3D};

/// Cross-validates GPS against UWB ranging to neighbours.
pub struct UwbAntiSpoofing {
    /// Tolerance for GPS vs UWB distance discrepancy, metres.
    pub tolerance_m: f64,
    /// Minimum number of UWB neighbours required for a valid cross-check.
    pub min_neighbors: usize,
}

impl UwbAntiSpoofing {
    pub fn new(tolerance_m: f64, min_neighbors: usize) -> Self {
        Self { tolerance_m, min_neighbors }
    }

    /// Returns `true` if the GPS position is consistent with UWB ranging data.
    pub fn is_gps_valid(
        &self,
        gps_position: &Position3D,
        uwb_ranges: &[(NodeId, f64)],
        neighbor_gps: &[(NodeId, Position3D)],
    ) -> bool {
        if uwb_ranges.len() < self.min_neighbors {
            // Not enough UWB anchors to validate — allow through with warning
            return true;
        }

        let validated_count = uwb_ranges
            .iter()
            .filter_map(|(id, uwb_dist)| {
                neighbor_gps
                    .iter()
                    .find(|(nid, _)| nid == id)
                    .map(|(_, ngps)| {
                        let gps_dist = gps_position.distance_to(ngps);
                        (gps_dist - uwb_dist).abs() <= self.tolerance_m
                    })
            })
            .filter(|&ok| ok)
            .count();

        // Require majority of ranges to be consistent
        validated_count * 2 >= uwb_ranges.len()
    }
}

impl Default for UwbAntiSpoofing {
    fn default() -> Self {
        Self::new(2.0, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_gps_valid() {
        let anti = UwbAntiSpoofing::new(2.0, 2);
        let gps = Position3D { x: 0.0, y: 0.0, z: 0.0 };
        let n1_pos = Position3D { x: 10.0, y: 0.0, z: 0.0 };
        let n2_pos = Position3D { x: 0.0, y: 10.0, z: 0.0 };
        let uwb_ranges = vec![(NodeId(1), 10.0), (NodeId(2), 10.0)];
        let neighbor_gps = vec![(NodeId(1), n1_pos), (NodeId(2), n2_pos)];
        assert!(anti.is_gps_valid(&gps, &uwb_ranges, &neighbor_gps));
    }

    #[test]
    fn test_spoofed_gps_invalid() {
        let anti = UwbAntiSpoofing::new(2.0, 2);
        // GPS claims (0,0) but UWB says drone is 50m from both neighbours
        let gps = Position3D { x: 0.0, y: 0.0, z: 0.0 };
        let n1_pos = Position3D { x: 10.0, y: 0.0, z: 0.0 };
        let n2_pos = Position3D { x: 0.0, y: 10.0, z: 0.0 };
        // UWB reports 50m but GPS only shows 10m — spoof detected
        let uwb_ranges = vec![(NodeId(1), 50.0), (NodeId(2), 50.0)];
        let neighbor_gps = vec![(NodeId(1), n1_pos), (NodeId(2), n2_pos)];
        assert!(!anti.is_gps_valid(&gps, &uwb_ranges, &neighbor_gps));
    }
}
