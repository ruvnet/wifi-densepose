//! A-MAPPO heterogeneous-role attention for sensor vs relay swarm nodes.
//!
//! Addresses four edge cases in heterogeneous swarms:
//! 1. Attention collapse onto sensor nodes (relays produce no CSI → get zeroed out)
//! 2. Variable neighbor cardinality (sensor clusters bunch, relays spread)
//! 3. Flocking↔triangulation geometry tension (gated by role)
//! 4. Relay→cluster-head handoff non-stationarity (role-dropout)
//!
//! Pure Rust — compiled in every build (no `train`/candle dependency).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Sensor,
    Relay,
    ClusterHead,
}

impl NodeRole {
    /// One-hot role embedding appended to attention keys.
    pub fn embedding(&self) -> [f32; 3] {
        match self {
            NodeRole::Sensor => [1.0, 0.0, 0.0],
            NodeRole::Relay => [0.0, 1.0, 0.0],
            NodeRole::ClusterHead => [0.0, 0.0, 1.0],
        }
    }
}

pub struct RoleAttention {
    /// Minimum attention weight floor for relay nodes (prevents collapse).
    pub relay_floor: f32,
    /// Temperature for softmax.
    pub temperature: f32,
}

impl Default for RoleAttention {
    fn default() -> Self {
        Self { relay_floor: 0.05, temperature: 1.0 }
    }
}

impl RoleAttention {
    /// Compute role-aware attention weights over neighbors.
    /// `scores`: raw attention logits per neighbor. `roles`: each neighbor's role.
    /// Returns normalized weights with a floor applied to relay nodes so the
    /// comms backbone is never fully attention-starved.
    pub fn weights(&self, scores: &[f32], roles: &[NodeRole]) -> Vec<f32> {
        if scores.is_empty() {
            return vec![];
        }
        // Softmax with temperature
        let max = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores
            .iter()
            .map(|s| ((s - max) / self.temperature).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        let mut w: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        // Apply relay floor
        for (wi, role) in w.iter_mut().zip(roles.iter()) {
            if *role == NodeRole::Relay && *wi < self.relay_floor {
                *wi = self.relay_floor;
            }
        }
        // Renormalize
        let s: f32 = w.iter().sum();
        if s > 0.0 {
            for wi in w.iter_mut() {
                *wi /= s;
            }
        }
        w
    }

    /// Role-segmented attention: separate sensor-pool and relay-pool so a flat
    /// softmax over k-nearest (mostly same-role) doesn't break.
    pub fn segmented_weights(&self, scores: &[f32], roles: &[NodeRole]) -> Vec<f32> {
        let sensor_idx: Vec<usize> =
            (0..roles.len()).filter(|&i| roles[i] != NodeRole::Relay).collect();
        let relay_idx: Vec<usize> =
            (0..roles.len()).filter(|&i| roles[i] == NodeRole::Relay).collect();
        let mut out = vec![0.0f32; scores.len()];
        // Each pool gets a fixed share of the attention mass (if both populated).
        let pools = [(&sensor_idx, 0.6f32), (&relay_idx, 0.4f32)];
        let active_pools = pools.iter().filter(|(idx, _)| !idx.is_empty()).count();
        for (idx, mass) in pools.iter() {
            if idx.is_empty() {
                continue;
            }
            let pool_mass = if active_pools == 1 { 1.0 } else { *mass };
            let pool_scores: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
            let max = pool_scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = pool_scores
                .iter()
                .map(|s| ((s - max) / self.temperature).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            for (k, &i) in idx.iter().enumerate() {
                out[i] = pool_mass * exps[k] / sum;
            }
        }
        out
    }
}

/// Reward modifier protecting triangulation baseline geometry (ADR-148 §4.2).
/// Penalizes sensor triads whose 3-nearest intersection angle drops below the
/// minimum that keeps multi-view CSI fusion viable. Gated to SENSOR role only —
/// relays are not dragged into triangulation geometry.
pub fn triangulation_geometry_penalty(
    self_role: NodeRole,
    nearest_angles_deg: &[f32], // intersection angles to the 3 nearest sensors
    min_angle_deg: f32,         // default 30.0
    penalty: f32,               // e.g. -5.0
) -> f32 {
    if self_role != NodeRole::Sensor {
        return 0.0;
    }
    let below = nearest_angles_deg
        .iter()
        .filter(|&&a| a < min_angle_deg)
        .count();
    below as f32 * penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_floor_prevents_collapse() {
        let attn = RoleAttention { relay_floor: 0.1, temperature: 1.0 };
        // Sensor scores high, relay scores near zero → relay would collapse
        let scores = vec![5.0, 5.0, -10.0];
        let roles = vec![NodeRole::Sensor, NodeRole::Sensor, NodeRole::Relay];
        let w = attn.weights(&scores, &roles);
        assert!(w[2] >= 0.09, "relay weight {} should respect floor", w[2]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "weights must sum to 1, got {}", sum);
    }

    #[test]
    fn test_segmented_splits_pools() {
        let attn = RoleAttention::default();
        let scores = vec![1.0, 1.0, 1.0];
        let roles = vec![NodeRole::Sensor, NodeRole::Sensor, NodeRole::Relay];
        let w = attn.segmented_weights(&scores, &roles);
        let relay_mass = w[2];
        assert!(relay_mass > 0.3 && relay_mass < 0.5, "relay pool ~0.4 mass, got {}", relay_mass);
    }

    #[test]
    fn test_triangulation_penalty_sensor_only() {
        // Relay: no penalty even with bad geometry
        assert_eq!(
            triangulation_geometry_penalty(NodeRole::Relay, &[10.0, 15.0, 20.0], 30.0, -5.0),
            0.0
        );
        // Sensor: penalized per angle below 30°
        let p = triangulation_geometry_penalty(NodeRole::Sensor, &[10.0, 15.0, 40.0], 30.0, -5.0);
        assert_eq!(p, -10.0, "two angles below 30° → 2 × -5.0");
    }

    #[test]
    fn test_role_embedding_onehot() {
        assert_eq!(NodeRole::Sensor.embedding(), [1.0, 0.0, 0.0]);
        assert_eq!(NodeRole::Relay.embedding(), [0.0, 1.0, 0.0]);
    }
}
