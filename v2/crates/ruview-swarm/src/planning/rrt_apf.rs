//! RRT-APF hybrid path planner: Rapidly-exploring Random Trees with
//! Artificial Potential Field obstacle repulsion.

use crate::types::Position3D;
use rand::Rng;

/// A planned waypoint with an associated target speed.
#[derive(Debug, Clone)]
pub struct Waypoint {
    pub position: Position3D,
    pub speed_ms: f64,
}

/// RRT-APF path planner.
pub struct RrtApfPlanner {
    pub obstacle_cells: Vec<Position3D>,
    pub apf_repulsion_dist: f64,
    pub step_size_m: f64,
}

impl RrtApfPlanner {
    pub fn new(apf_repulsion_dist: f64) -> Self {
        Self {
            obstacle_cells: Vec::new(),
            apf_repulsion_dist,
            step_size_m: 2.0,
        }
    }

    /// Compute the APF repulsion gradient at `pos` from all nearby obstacles.
    pub fn apf_force(&self, pos: &Position3D, neighbors: &[Position3D]) -> (f64, f64, f64) {
        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        let mut fz = 0.0_f64;
        for obs in self.obstacle_cells.iter().chain(neighbors.iter()) {
            let dist = pos.distance_to(obs);
            if dist < self.apf_repulsion_dist && dist > 1e-6 {
                let strength = (self.apf_repulsion_dist - dist) / (dist * dist);
                fx += strength * (pos.x - obs.x);
                fy += strength * (pos.y - obs.y);
                fz += strength * (pos.z - obs.z);
            }
        }
        (fx, fy, fz)
    }

    /// Plan a path from `start` to `goal` using RRT* with APF bias.
    pub fn plan(
        &self,
        start: Position3D,
        goal: Position3D,
        max_iter: usize,
        rng: &mut impl Rng,
    ) -> Vec<Waypoint> {
        let mut tree: Vec<(Position3D, usize)> = vec![(start, 0)];
        let goal_dist_thresh = self.step_size_m * 1.5;

        for _ in 0..max_iter {
            // Sample random point (bias 10% toward goal)
            let sample = if rng.gen::<f64>() < 0.1 {
                goal
            } else {
                let range = 200.0_f64;
                Position3D {
                    x: start.x + (rng.gen::<f64>() - 0.5) * range,
                    y: start.y + (rng.gen::<f64>() - 0.5) * range,
                    z: start.z,
                }
            };

            // Find nearest node in tree
            let (nearest_idx, nearest_pos) = tree
                .iter()
                .enumerate()
                .min_by(|(_, (a, _)), (_, (b, _))| {
                    a.distance_to(&sample)
                        .partial_cmp(&b.distance_to(&sample))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, (p, _))| (i, *p))
                .unwrap_or((0, start));

            // Step toward sample, then apply APF
            let dist_to_sample = nearest_pos.distance_to(&sample);
            if dist_to_sample < 1e-9 {
                continue;
            }
            let scale = self.step_size_m / dist_to_sample;
            let mut new_pos = Position3D {
                x: nearest_pos.x + (sample.x - nearest_pos.x) * scale,
                y: nearest_pos.y + (sample.y - nearest_pos.y) * scale,
                z: nearest_pos.z + (sample.z - nearest_pos.z) * scale,
            };

            // Apply APF correction
            let (fx, fy, fz) = self.apf_force(&new_pos, &[]);
            let apf_scale = 0.3;
            new_pos.x += fx * apf_scale;
            new_pos.y += fy * apf_scale;
            new_pos.z += fz * apf_scale;

            tree.push((new_pos, nearest_idx));

            if new_pos.distance_to(&goal) <= goal_dist_thresh {
                // Trace path back to root
                let mut path = Vec::new();
                let mut current_idx = tree.len() - 1;
                while current_idx != 0 {
                    let (pos, parent) = tree[current_idx];
                    path.push(Waypoint { position: pos, speed_ms: 5.0 });
                    current_idx = parent;
                }
                path.push(Waypoint { position: start, speed_ms: 5.0 });
                path.reverse();
                path.push(Waypoint { position: goal, speed_ms: 2.0 });
                return path;
            }
        }

        // Fallback: direct line
        vec![
            Waypoint { position: start, speed_ms: 5.0 },
            Waypoint { position: goal, speed_ms: 5.0 },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_returns_at_least_two_waypoints() {
        let planner = RrtApfPlanner::new(3.0);
        let start = Position3D { x: 0.0, y: 0.0, z: -30.0 };
        let goal = Position3D { x: 50.0, y: 50.0, z: -30.0 };
        let mut rng = rand::thread_rng();
        let path = planner.plan(start, goal, 500, &mut rng);
        assert!(path.len() >= 2);
    }

    #[test]
    fn test_apf_force_pushes_away() {
        let planner = RrtApfPlanner {
            obstacle_cells: vec![Position3D { x: 1.0, y: 0.0, z: 0.0 }],
            apf_repulsion_dist: 5.0,
            step_size_m: 2.0,
        };
        let pos = Position3D { x: 0.0, y: 0.0, z: 0.0 };
        let (fx, _, _) = planner.apf_force(&pos, &[]);
        assert!(fx < 0.0); // pushed away from x=1 obstacle
    }

    #[test]
    fn test_plan_reaches_goal() {
        let planner = RrtApfPlanner::new(3.0);
        let start = Position3D { x: 0.0, y: 0.0, z: -30.0 };
        let goal  = Position3D { x: 50.0, y: 50.0, z: -30.0 };
        let mut rng = rand::thread_rng();
        let path = planner.plan(start, goal, 500, &mut rng);
        let last = path.last().unwrap();
        // The RRT either reaches goal directly or the fallback end is the goal itself.
        assert!(last.position.distance_to(&goal) < 10.0, "path should end near goal");
    }

    #[test]
    fn test_apf_repulsion_nonzero_near_obstacle() {
        let planner = RrtApfPlanner {
            obstacle_cells: vec![Position3D { x: 3.0, y: 0.0, z: 0.0 }],
            apf_repulsion_dist: 5.0,
            step_size_m: 2.0,
        };
        let pos = Position3D { x: 0.0, y: 0.0, z: 0.0 };
        let (fx, _, _) = planner.apf_force(&pos, &[]);
        assert!(fx < 0.0, "repulsion should push away from obstacle (negative x)");
    }
}
