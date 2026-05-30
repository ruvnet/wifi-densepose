use crate::types::{DroneState, NodeId, Position3D, GridCell, CsiDetection};

/// Local observation vector for a single drone agent.
/// Feeds into the MAPPO actor network.
///
/// Dimension breakdown:
///   - own_state:             9  (pos xyz, vel xyz, heading, battery, link_quality)
///   - neighbor_relative_pos: 18 (K=6 neighbours × 3 floats each)
///   - grid_tile:             25 (5×5 cell victim probabilities)
///   - csi_reading:            5 (confidence, est pos xyz, has_detection flag)
///   - task_encoding:          7 (target xyz, deadline_norm, task_type one-hot × 3)
///
///   TOTAL:                   64
#[derive(Debug, Clone)]
pub struct LocalObservation {
    /// Own state: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, heading, battery, link_quality]
    pub own_state: [f32; 9],
    /// K=6 nearest-neighbour relative positions: [dx, dy, dz] × 6 = 18 floats
    pub neighbor_relative_pos: [f32; 18],
    /// 5×5 grid tile centred on drone position: victim_probability × 25
    pub grid_tile: [f32; 25],
    /// CSI reading: [confidence, est_x, est_y, est_z, has_detection]
    pub csi_reading: [f32; 5],
    /// Current task: [target_x, target_y, target_z, deadline_norm, task_type_one_hot × 3]
    pub task_encoding: [f32; 7],
}

impl LocalObservation {
    pub const DIM: usize = 9 + 18 + 25 + 5 + 7; // = 64

    /// Return an observation with all fields zeroed.
    pub fn zeros() -> Self {
        Self {
            own_state: [0.0; 9],
            neighbor_relative_pos: [0.0; 18],
            grid_tile: [0.0; 25],
            csi_reading: [0.0; 5],
            task_encoding: [0.0; 7],
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(Self::DIM);
        v.extend_from_slice(&self.own_state);
        v.extend_from_slice(&self.neighbor_relative_pos);
        v.extend_from_slice(&self.grid_tile);
        v.extend_from_slice(&self.csi_reading);
        v.extend_from_slice(&self.task_encoding);
        v
    }

    pub fn from_state(
        state: &DroneState,
        neighbors: &[(NodeId, Position3D)],
        grid_tile: [[GridCell; 5]; 5],
        csi_detection: Option<&crate::types::CsiDetection>,
        task_target: Option<&Position3D>,
    ) -> Self {
        let own_state = [
            state.position.x as f32 / 1000.0,  // normalised to km
            state.position.y as f32 / 1000.0,
            state.position.z as f32 / 100.0,
            state.velocity.vx as f32 / 20.0,   // normalised to max speed
            state.velocity.vy as f32 / 20.0,
            state.velocity.vz as f32 / 5.0,
            state.heading_rad as f32 / std::f32::consts::PI,
            state.battery_pct / 100.0,
            state.link_quality,
        ];

        let mut neighbor_relative_pos = [0.0f32; 18];
        for (i, (_, pos)) in neighbors.iter().take(6).enumerate() {
            let base = i * 3;
            neighbor_relative_pos[base]     = (pos.x - state.position.x) as f32 / 100.0;
            neighbor_relative_pos[base + 1] = (pos.y - state.position.y) as f32 / 100.0;
            neighbor_relative_pos[base + 2] = (pos.z - state.position.z) as f32 / 10.0;
        }

        let mut grid_flat = [0.0f32; 25];
        for (r, row) in grid_tile.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                grid_flat[r * 5 + c] = cell.victim_probability;
            }
        }

        let csi_reading = if let Some(det) = csi_detection {
            let vp = det.victim_position.unwrap_or(state.position);
            [det.confidence, (vp.x / 100.0) as f32, (vp.y / 100.0) as f32, (vp.z / 10.0) as f32, 1.0]
        } else {
            [0.0, 0.0, 0.0, 0.0, 0.0]
        };

        let task_encoding: [f32; 7] = if let Some(target) = task_target {
            [
                (target.x / 100.0) as f32,
                (target.y / 100.0) as f32,
                (target.z / 10.0) as f32,
                1.0,  // deadline_norm: placeholder
                1.0, 0.0, 0.0, // task_type one-hot: CoverCell
            ]
        } else {
            [0.0f32; 7]
        };

        Self {
            own_state,
            neighbor_relative_pos,
            grid_tile: grid_flat,
            csi_reading,
            task_encoding,
        }
    }

    /// Build an observation from a drone state without a pre-computed grid tile.
    /// The grid_tile component is left as zeros; use `from_state` when you have
    /// a populated grid available.
    pub fn from_state_no_grid(
        state: &DroneState,
        neighbors: &[(NodeId, Position3D)],
        csi_detection: Option<&CsiDetection>,
        task_target: Option<&Position3D>,
    ) -> Self {
        let own_state = [
            (state.position.x / 1000.0) as f32,
            (state.position.y / 1000.0) as f32,
            (state.position.z / 100.0) as f32,
            (state.velocity.vx / 20.0) as f32,
            (state.velocity.vy / 20.0) as f32,
            (state.velocity.vz / 5.0) as f32,
            (state.heading_rad / std::f64::consts::PI) as f32,
            state.battery_pct / 100.0,
            state.link_quality,
        ];

        let mut neighbor_relative_pos = [0.0f32; 18];
        for (i, (_, pos)) in neighbors.iter().take(6).enumerate() {
            let base = i * 3;
            neighbor_relative_pos[base]   = ((pos.x - state.position.x) / 100.0) as f32;
            neighbor_relative_pos[base+1] = ((pos.y - state.position.y) / 100.0) as f32;
            neighbor_relative_pos[base+2] = ((pos.z - state.position.z) / 10.0) as f32;
        }

        let csi_reading = match csi_detection {
            Some(det) => {
                let vp = det.victim_position.unwrap_or(state.position);
                [det.confidence, (vp.x / 100.0) as f32, (vp.y / 100.0) as f32, (vp.z / 10.0) as f32, 1.0]
            }
            None => [0.0; 5],
        };

        let task_encoding: [f32; 7] = match task_target {
            Some(t) => [(t.x / 100.0) as f32, (t.y / 100.0) as f32, (t.z / 10.0) as f32, 1.0, 1.0, 0.0, 0.0],
            None => [0.0; 7],
        };

        Self {
            own_state,
            neighbor_relative_pos,
            grid_tile: [0.0; 25],
            csi_reading,
            task_encoding,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DroneState, NodeId};

    #[test]
    fn observation_dimension() {
        assert_eq!(LocalObservation::DIM, 64);
    }

    #[test]
    fn to_vec_length() {
        let obs = LocalObservation {
            own_state: [0.0; 9],
            neighbor_relative_pos: [0.0; 18],
            grid_tile: [0.0; 25],
            csi_reading: [0.0; 5],
            task_encoding: [0.0; 7],
        };
        assert_eq!(obs.to_vec().len(), LocalObservation::DIM);
    }

    #[test]
    fn from_state_produces_correct_dim() {
        let state = DroneState::default_at_origin(NodeId(0));
        let grid = [[GridCell::default(); 5]; 5];
        let obs = LocalObservation::from_state(&state, &[], grid, None, None);
        assert_eq!(obs.to_vec().len(), LocalObservation::DIM);
    }

    #[test]
    fn test_observation_dim() {
        let obs = LocalObservation::zeros();
        assert_eq!(obs.to_vec().len(), LocalObservation::DIM);
    }

    #[test]
    fn test_from_state_battery_normalised() {
        use crate::types::Velocity3D;
        let state = DroneState {
            id: NodeId(0),
            position: Default::default(),
            velocity: Velocity3D::default(),
            heading_rad: 0.0,
            altitude_agl_m: 30.0,
            battery_pct: 75.0,
            link_quality: 0.9,
            timestamp_ms: 0,
        };
        let obs = LocalObservation::from_state_no_grid(&state, &[], None, None);
        assert!((obs.own_state[7] - 0.75).abs() < 1e-4, "battery should be normalised to 0.75");
    }
}
