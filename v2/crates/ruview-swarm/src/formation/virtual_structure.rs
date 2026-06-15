//! Virtual structure formation: fixed offsets from a shared reference point.

use crate::types::{NodeId, Position3D};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Offsets from a shared reference point for each drone in the formation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualStructure {
    /// NodeId → (dx, dy, dz) offset in metres from the reference.
    pub offsets: HashMap<NodeId, (f64, f64, f64)>,
}

impl VirtualStructure {
    /// Create a rectangular grid formation with `n` drones, spaced `spacing_m` apart.
    pub fn grid_formation(n: usize, spacing_m: f64) -> Self {
        let cols = (n as f64).sqrt().ceil() as usize;
        let mut offsets = HashMap::new();
        for i in 0..n {
            let row = i / cols;
            let col = i % cols;
            offsets.insert(
                NodeId(i as u32),
                (row as f64 * spacing_m, col as f64 * spacing_m, 0.0),
            );
        }
        Self { offsets }
    }

    /// Create a circular formation with `n` drones evenly distributed.
    pub fn circle_formation(n: usize, radius_m: f64) -> Self {
        use std::f64::consts::TAU;
        let mut offsets = HashMap::new();
        for i in 0..n {
            let angle = TAU * i as f64 / n as f64;
            offsets.insert(
                NodeId(i as u32),
                (radius_m * angle.cos(), radius_m * angle.sin(), 0.0),
            );
        }
        Self { offsets }
    }

    /// Compute target position for a node, applying its offset from `reference`.
    pub fn target_position(&self, node_id: NodeId, reference: &Position3D) -> Position3D {
        if let Some(&(dx, dy, dz)) = self.offsets.get(&node_id) {
            Position3D {
                x: reference.x + dx,
                y: reference.y + dy,
                z: reference.z + dz,
            }
        } else {
            *reference
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_formation_4_drones() {
        let vs = VirtualStructure::grid_formation(4, 5.0);
        assert_eq!(vs.offsets.len(), 4);
        let ref_pos = Position3D { x: 100.0, y: 200.0, z: -30.0 };
        let p = vs.target_position(NodeId(0), &ref_pos);
        assert!((p.x - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_circle_formation() {
        let vs = VirtualStructure::circle_formation(4, 10.0);
        let ref_pos = Position3D::zero();
        let p = vs.target_position(NodeId(0), &ref_pos);
        // Node 0 at angle 0: x = 10, y = 0
        assert!((p.x - 10.0).abs() < 1e-6);
        assert!(p.y.abs() < 1e-6);
    }
}
