//! Mesh topology: maintains a live view of all drone nodes.

use crate::types::{DroneState, NodeId};
use std::collections::HashMap;

/// Hierarchical-mesh topology view.
pub struct MeshTopology {
    pub nodes: HashMap<NodeId, DroneState>,
    pub cluster_head: Option<NodeId>,
}

impl MeshTopology {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            cluster_head: None,
        }
    }

    /// Upsert a node's state.
    pub fn update_node(&mut self, state: DroneState) {
        self.nodes.insert(state.id, state);
    }

    /// Remove a node (e.g. on dropout).
    pub fn remove_node(&mut self, id: &NodeId) {
        self.nodes.remove(id);
        if self.cluster_head == Some(*id) {
            self.cluster_head = None;
        }
    }

    /// All active nodes (sorted by id for determinism).
    pub fn active_nodes(&self) -> Vec<&DroneState> {
        let mut v: Vec<_> = self.nodes.values().collect();
        v.sort_by_key(|s| s.id.0);
        v
    }

    /// Returns the `k` nearest nodes to `from`, sorted ascending by distance.
    pub fn nearest_k(&self, from: NodeId, k: usize) -> Vec<NodeId> {
        if let Some(origin) = self.nodes.get(&from) {
            let mut distances: Vec<(f64, NodeId)> = self
                .nodes
                .iter()
                .filter(|(&id, _)| id != from)
                .map(|(&id, s)| (origin.position.distance_to(&s.position), id))
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            distances.truncate(k);
            distances.into_iter().map(|(_, id)| id).collect()
        } else {
            vec![]
        }
    }
}

impl Default for MeshTopology {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Position3D;

    #[test]
    fn test_nearest_k() {
        let mut topo = MeshTopology::new();
        let mut s0 = DroneState::default_at_origin(NodeId(0));
        s0.position = Position3D { x: 0.0, y: 0.0, z: 0.0 };
        let mut s1 = DroneState::default_at_origin(NodeId(1));
        s1.position = Position3D { x: 10.0, y: 0.0, z: 0.0 };
        let mut s2 = DroneState::default_at_origin(NodeId(2));
        s2.position = Position3D { x: 5.0, y: 0.0, z: 0.0 };
        topo.update_node(s0);
        topo.update_node(s1);
        topo.update_node(s2);
        let nearest = topo.nearest_k(NodeId(0), 1);
        assert_eq!(nearest, vec![NodeId(2)]);
    }
}
