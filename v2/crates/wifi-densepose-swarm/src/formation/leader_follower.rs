//! Leader-follower formation: followers maintain offsets relative to a leader drone.

use crate::types::{NodeId, Position3D};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Leader-follower formation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderFollower {
    pub leader_id: NodeId,
    /// Follower → (dx, dy, dz) offset from leader's position.
    pub offsets: HashMap<NodeId, (f64, f64, f64)>,
}

impl LeaderFollower {
    pub fn new(leader_id: NodeId) -> Self {
        Self {
            leader_id,
            offsets: HashMap::new(),
        }
    }

    pub fn add_follower(&mut self, follower: NodeId, offset: (f64, f64, f64)) {
        self.offsets.insert(follower, offset);
    }

    /// Compute target position for a node given current drone positions.
    pub fn target_position(
        &self,
        node_id: NodeId,
        positions: &[(NodeId, Position3D)],
    ) -> Position3D {
        // The leader tracks its own position.
        if node_id == self.leader_id {
            return positions
                .iter()
                .find(|(id, _)| *id == self.leader_id)
                .map(|(_, p)| *p)
                .unwrap_or_default();
        }
        let leader_pos = positions
            .iter()
            .find(|(id, _)| *id == self.leader_id)
            .map(|(_, p)| *p)
            .unwrap_or_default();

        if let Some(&(dx, dy, dz)) = self.offsets.get(&node_id) {
            Position3D {
                x: leader_pos.x + dx,
                y: leader_pos.y + dy,
                z: leader_pos.z + dz,
            }
        } else {
            leader_pos
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_follower_tracks_leader() {
        let mut lf = LeaderFollower::new(NodeId(0));
        lf.add_follower(NodeId(1), (-5.0, 0.0, 0.0));
        let positions = vec![
            (NodeId(0), Position3D { x: 10.0, y: 20.0, z: -30.0 }),
        ];
        let target = lf.target_position(NodeId(1), &positions);
        assert!((target.x - 5.0).abs() < 1e-6);
        assert!((target.y - 20.0).abs() < 1e-6);
    }
}
