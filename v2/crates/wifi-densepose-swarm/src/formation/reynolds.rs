//! Reynolds flocking: separation, alignment, cohesion.

use crate::types::{NodeId, Position3D, Velocity3D};
use serde::{Deserialize, Serialize};

/// Parameters for Reynolds boid rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReynoldsParams {
    pub separation_dist_m: f64,
    pub separation_weight: f64,
    pub alignment_weight: f64,
    pub cohesion_weight: f64,
    pub k_neighbors: usize,
}

impl Default for ReynoldsParams {
    fn default() -> Self {
        Self {
            separation_dist_m: 3.0,
            separation_weight: 1.5,
            alignment_weight: 1.0,
            cohesion_weight: 0.8,
            k_neighbors: 7,
        }
    }
}

impl ReynoldsParams {
    /// Compute a desired velocity delta for `node_id` based on the three Reynolds rules.
    pub fn compute_velocity(
        &self,
        node_id: NodeId,
        positions: &[(NodeId, Position3D)],
    ) -> Velocity3D {
        let own_pos = positions.iter().find(|(id, _)| *id == node_id).map(|(_, p)| *p);
        let own_pos = match own_pos {
            Some(p) => p,
            None => return Velocity3D::default(),
        };

        // Sort neighbours by distance, take k nearest.
        let mut neighbours: Vec<(f64, &Position3D)> = positions
            .iter()
            .filter(|(id, _)| *id != node_id)
            .map(|(_, p)| (own_pos.distance_to(p), p))
            .collect();
        neighbours.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        neighbours.truncate(self.k_neighbors);

        if neighbours.is_empty() {
            return Velocity3D::default();
        }

        let n = neighbours.len() as f64;

        // --- Separation: steer away from too-close neighbours ---
        let (mut sep_x, mut sep_y, mut sep_z) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (dist, p) in &neighbours {
            if *dist < self.separation_dist_m && *dist > 1e-6 {
                let factor = (self.separation_dist_m - *dist) / self.separation_dist_m;
                sep_x += (own_pos.x - p.x) / dist * factor;
                sep_y += (own_pos.y - p.y) / dist * factor;
                sep_z += (own_pos.z - p.z) / dist * factor;
            }
        }

        // --- Cohesion: steer toward average position ---
        let (avg_x, avg_y, avg_z) = neighbours
            .iter()
            .fold((0.0, 0.0, 0.0), |(ax, ay, az), (_, p)| (ax + p.x, ay + p.y, az + p.z));
        let coh_x = (avg_x / n) - own_pos.x;
        let coh_y = (avg_y / n) - own_pos.y;
        let coh_z = (avg_z / n) - own_pos.z;

        // Combine rules (alignment omitted in position-only mode — no velocity info here).
        let vx = self.separation_weight * sep_x + self.cohesion_weight * coh_x;
        let vy = self.separation_weight * sep_y + self.cohesion_weight * coh_y;
        let vz = self.separation_weight * sep_z + self.cohesion_weight * coh_z;

        Velocity3D { vx, vy, vz }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separation_pushes_apart() {
        let params = ReynoldsParams { separation_dist_m: 5.0, ..Default::default() };
        let positions = vec![
            (NodeId(0), Position3D { x: 0.0, y: 0.0, z: 0.0 }),
            (NodeId(1), Position3D { x: 1.0, y: 0.0, z: 0.0 }), // too close
        ];
        let vel = params.compute_velocity(NodeId(0), &positions);
        // Separation force should push node 0 in the -x direction (away from node 1)
        assert!(vel.vx < 0.0);
    }

    #[test]
    fn test_no_neighbours_returns_zero() {
        let params = ReynoldsParams::default();
        let positions = vec![(NodeId(0), Position3D::zero())];
        let vel = params.compute_velocity(NodeId(0), &positions);
        assert!((vel.vx.abs() + vel.vy.abs()) < 1e-9);
    }
}
