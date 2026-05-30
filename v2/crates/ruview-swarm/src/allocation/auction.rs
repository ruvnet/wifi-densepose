//! Contract-net (auction) task allocation.

use crate::types::{DroneState, NodeId, SwarmTask, TaskId};
use std::collections::HashMap;

/// A bid submitted by a node for a task.
#[derive(Debug, Clone)]
pub struct Bid {
    pub node_id: NodeId,
    pub task_id: TaskId,
    /// Lower score = more capable/willing. Computed by the bidding node.
    pub score: f32,
}

/// Auction-based task allocator.
pub struct AuctionAllocator {
    pub pending_tasks: HashMap<TaskId, SwarmTask>,
    pub bids: HashMap<TaskId, Vec<Bid>>,
    pub timeout_ms: u64,
}

impl AuctionAllocator {
    pub fn new(timeout_ms: u64) -> Self {
        Self {
            pending_tasks: HashMap::new(),
            bids: HashMap::new(),
            timeout_ms,
        }
    }

    /// Announce a new task (add to pending pool).
    pub fn announce_task(&mut self, task: SwarmTask) {
        let id = task.id;
        self.pending_tasks.insert(id, task);
        self.bids.entry(id).or_default();
    }

    /// Accept a bid for a pending task.
    pub fn submit_bid(&mut self, bid: Bid) {
        if self.pending_tasks.contains_key(&bid.task_id) {
            self.bids.entry(bid.task_id).or_default().push(bid);
        }
    }

    /// Resolve all pending tasks: assign each to the best bidder.
    /// Returns a list of (TaskId, winning NodeId) pairs.
    pub fn resolve(&mut self) -> Vec<(TaskId, NodeId)> {
        let mut results = Vec::new();
        let task_ids: Vec<TaskId> = self.pending_tasks.keys().copied().collect();

        for task_id in task_ids {
            let winner = self
                .bids
                .get(&task_id)
                .and_then(|bids| {
                    bids.iter()
                        .min_by(|a, b| {
                            a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|b| b.node_id)
                });

            if let Some(winner_id) = winner {
                if let Some(task) = self.pending_tasks.get_mut(&task_id) {
                    task.assigned_to = Some(winner_id);
                }
                results.push((task_id, winner_id));
                self.bids.remove(&task_id);
            }
        }

        // Clean up resolved tasks
        for (tid, _) in &results {
            self.pending_tasks.remove(tid);
        }

        results
    }

    /// Compute a bid score heuristic for a node given a task.
    /// Returns a score ∈ [0, ∞): lower is better.
    pub fn compute_bid_score(node: &DroneState, task: &SwarmTask) -> f32 {
        let dist = node.position.distance_to(&task.target) as f32;
        let battery_penalty = (100.0 - node.battery_pct) / 100.0;
        let link_penalty = 1.0 - node.link_quality;
        let priority_bonus = 1.0 - task.priority.clamp(0.0, 1.0);
        dist / 100.0 + battery_penalty * 0.3 + link_penalty * 0.2 + priority_bonus * 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Position3D, SwarmTask, TaskId, TaskKind};

    fn make_task(id: u64) -> SwarmTask {
        SwarmTask {
            id: TaskId(id),
            kind: TaskKind::ReturnToHome,
            priority: 0.5,
            target: Position3D::zero(),
            deadline_ms: None,
            assigned_to: None,
        }
    }

    #[test]
    fn test_auction_assigns_best_bidder() {
        let mut alloc = AuctionAllocator::new(1000);
        let task = make_task(1);
        alloc.announce_task(task);
        alloc.submit_bid(Bid { node_id: NodeId(1), task_id: TaskId(1), score: 0.8 });
        alloc.submit_bid(Bid { node_id: NodeId(2), task_id: TaskId(1), score: 0.3 });
        let results = alloc.resolve();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, NodeId(2)); // lower score wins
    }
}
