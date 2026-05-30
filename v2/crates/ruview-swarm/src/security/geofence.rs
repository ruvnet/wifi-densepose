//! Geofence: polygon boundary with hard/soft margins.

use crate::types::Position3D;
use serde::{Deserialize, Serialize};

/// Polygon geofence with altitude bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geofence {
    /// Polygon vertices (x, y) in local NED metres.
    pub boundary: Vec<(f64, f64)>,
    pub min_altitude_m: f64,
    pub max_altitude_m: f64,
    /// Hard margin: triggers RTH immediately.
    pub hard_margin_m: f64,
    /// Soft margin: triggers warning + speed reduction.
    pub soft_margin_m: f64,
}

/// Result of a geofence check.
#[derive(Debug, Clone, PartialEq)]
pub enum GeofenceResult {
    Safe,
    SoftWarning { distance_to_boundary_m: f64 },
    HardBreach,
}

impl Geofence {
    /// Check a position against this geofence.
    pub fn check(&self, pos: &Position3D) -> GeofenceResult {
        let altitude_m = -pos.z; // NED: negative z = altitude above ground

        // Altitude check
        if altitude_m < self.min_altitude_m || altitude_m > self.max_altitude_m {
            return GeofenceResult::HardBreach;
        }

        let inside = self.point_in_polygon(pos.x, pos.y);
        let dist = self.distance_to_boundary(pos.x, pos.y);

        if !inside {
            return GeofenceResult::HardBreach;
        }

        if dist <= self.hard_margin_m {
            GeofenceResult::HardBreach
        } else if dist <= self.soft_margin_m {
            GeofenceResult::SoftWarning { distance_to_boundary_m: dist }
        } else {
            GeofenceResult::Safe
        }
    }

    /// Ray-casting algorithm: even number of crossings = outside.
    fn point_in_polygon(&self, x: f64, y: f64) -> bool {
        let n = self.boundary.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = self.boundary[i];
            let (xj, yj) = self.boundary[j];
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Minimum distance from (x, y) to any boundary edge.
    fn distance_to_boundary(&self, x: f64, y: f64) -> f64 {
        let n = self.boundary.len();
        if n == 0 {
            return f64::INFINITY;
        }
        let mut min_dist = f64::INFINITY;
        let mut j = n - 1;
        for i in 0..n {
            let (ax, ay) = self.boundary[j];
            let (bx, by) = self.boundary[i];
            let dist = point_to_segment_dist(x, y, ax, ay, bx, by);
            if dist < min_dist {
                min_dist = dist;
            }
            j = i;
        }
        min_dist
    }
}

fn point_to_segment_dist(px: f64, py: f64, ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-12 {
        return ((px - ax).powi(2) + (py - ay).powi(2)).sqrt();
    }
    let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let cx = ax + t * dx;
    let cy = ay + t * dy;
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square_fence() -> Geofence {
        Geofence {
            boundary: vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
            min_altitude_m: 0.0,
            max_altitude_m: 120.0,
            hard_margin_m: 10.0,
            soft_margin_m: 25.0,
        }
    }

    #[test]
    fn test_centre_is_safe() {
        let f = square_fence();
        let pos = Position3D { x: 50.0, y: 50.0, z: -30.0 };
        assert_eq!(f.check(&pos), GeofenceResult::Safe);
    }

    #[test]
    fn test_outside_is_hard_breach() {
        let f = square_fence();
        let pos = Position3D { x: 150.0, y: 50.0, z: -30.0 };
        assert_eq!(f.check(&pos), GeofenceResult::HardBreach);
    }

    #[test]
    fn test_near_edge_is_soft_warning() {
        let f = square_fence();
        // 15m from boundary → beyond hard (10m) but within soft (25m)
        let pos = Position3D { x: 15.0, y: 50.0, z: -30.0 };
        assert!(matches!(f.check(&pos), GeofenceResult::SoftWarning { .. }));
    }

    #[test]
    fn test_altitude_breach() {
        let f = square_fence();
        let pos = Position3D { x: 50.0, y: 50.0, z: -200.0 }; // 200m altitude
        assert_eq!(f.check(&pos), GeofenceResult::HardBreach);
    }
}
