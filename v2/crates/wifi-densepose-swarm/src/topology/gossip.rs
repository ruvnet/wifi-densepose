//! Gossip-based state dissemination for the swarm.

use crate::types::NodeId;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// A gossip-propagated state value with versioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipState<T: Clone> {
    pub value: T,
    pub version: u64,
    pub origin: NodeId,
    pub timestamp_ms: u64,
}

impl<T: Clone> GossipState<T> {
    pub fn new(value: T, origin: NodeId, timestamp_ms: u64) -> Self {
        Self { value, version: 1, origin, timestamp_ms }
    }

    /// Last-write-wins merge: higher version wins; ties go to higher origin id.
    pub fn merge(a: GossipState<T>, b: GossipState<T>) -> GossipState<T> {
        if a.version > b.version {
            a
        } else if b.version > a.version {
            b
        } else if a.origin.0 >= b.origin.0 {
            a
        } else {
            b
        }
    }

    /// Increment the version (call when mutating a local copy before gossiping).
    pub fn bump(&mut self) {
        self.version += 1;
    }

    /// Choose `fanout` random peer IDs to spread this state to, excluding the
    /// local node and the origin to avoid trivial loops.
    pub fn spread(
        &self,
        fanout: usize,
        all_peers: &[NodeId],
        local_id: NodeId,
        rng: &mut impl rand::Rng,
    ) -> Vec<NodeId> {
        let mut candidates: Vec<NodeId> = all_peers
            .iter()
            .copied()
            .filter(|&n| n != local_id && n != self.origin)
            .collect();
        candidates.shuffle(rng);
        candidates.truncate(fanout);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_higher_version_wins() {
        let a: GossipState<u32> = GossipState { value: 1, version: 2, origin: NodeId(1), timestamp_ms: 0 };
        let b: GossipState<u32> = GossipState { value: 2, version: 5, origin: NodeId(2), timestamp_ms: 0 };
        let merged = GossipState::merge(a, b);
        assert_eq!(merged.value, 2);
    }

    #[test]
    fn test_merge_tie_higher_origin_wins() {
        let a: GossipState<u32> = GossipState { value: 10, version: 3, origin: NodeId(5), timestamp_ms: 0 };
        let b: GossipState<u32> = GossipState { value: 20, version: 3, origin: NodeId(2), timestamp_ms: 0 };
        let merged = GossipState::merge(a, b);
        assert_eq!(merged.value, 10); // origin 5 > 2
    }
}
