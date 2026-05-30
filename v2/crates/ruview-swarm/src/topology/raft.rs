//! Raft-based cluster-head election for drone swarms.

use crate::types::{DroneState, NodeId};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the Raft consensus engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    pub election_timeout_ms: u64,
    pub heartbeat_ms: u64,
    pub min_battery_pct: f32,
    pub min_link_quality: f32,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout_ms: 300,
            heartbeat_ms: 100,
            min_battery_pct: 20.0,
            min_link_quality: 0.4,
        }
    }
}

/// Role within the Raft cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

/// A log entry stored by the Raft leader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub data: Vec<u8>,
}

/// Messages exchanged between Raft peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftMessage {
    RequestVote {
        term: u64,
        candidate_id: NodeId,
        last_log_index: u64,
        last_log_term: u64,
    },
    VoteGranted {
        term: u64,
        voter_id: NodeId,
        granted: bool,
    },
    AppendEntries {
        term: u64,
        leader_id: NodeId,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    AppendEntriesAck {
        term: u64,
        follower_id: NodeId,
        success: bool,
        match_index: u64,
    },
}

/// A Raft node driving cluster-head election within a swarm cluster.
pub struct RaftNode {
    pub id: NodeId,
    pub role: RaftRole,
    pub current_term: u64,
    pub voted_for: Option<NodeId>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub config: RaftConfig,
    /// Votes received as candidate.
    votes_received: u32,
    /// Elapsed time since last heartbeat/election-timeout reset (ms).
    elapsed_since_last_event_ms: u64,
}

impl RaftNode {
    pub fn new(id: NodeId, config: RaftConfig) -> Self {
        Self {
            id,
            role: RaftRole::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            config,
            votes_received: 0,
            elapsed_since_last_event_ms: 0,
        }
    }

    /// Check whether a drone is eligible to become cluster head.
    pub fn is_eligible_leader(state: &DroneState, config: &RaftConfig) -> bool {
        state.battery_pct >= config.min_battery_pct
            && state.link_quality >= config.min_link_quality
    }

    /// Drive the Raft state machine by one time step.
    /// Returns a message to broadcast if an election event fires.
    pub fn tick(&mut self, elapsed: Duration, peers: &[DroneState]) -> Option<RaftMessage> {
        let elapsed_ms = elapsed.as_millis() as u64;
        self.elapsed_since_last_event_ms += elapsed_ms;

        match self.role {
            RaftRole::Leader => {
                if self.elapsed_since_last_event_ms >= self.config.heartbeat_ms {
                    self.elapsed_since_last_event_ms = 0;
                    let last_index = self.log.len() as u64;
                    let last_term = self.log.last().map(|e| e.term).unwrap_or(0);
                    return Some(RaftMessage::AppendEntries {
                        term: self.current_term,
                        leader_id: self.id,
                        prev_log_index: last_index,
                        prev_log_term: last_term,
                        entries: vec![],
                        leader_commit: self.commit_index,
                    });
                }
                None
            }
            RaftRole::Follower | RaftRole::Candidate => {
                if self.elapsed_since_last_event_ms >= self.config.election_timeout_ms {
                    self.elapsed_since_last_event_ms = 0;
                    self.current_term += 1;
                    self.role = RaftRole::Candidate;
                    self.voted_for = Some(self.id);
                    self.votes_received = 1;

                    let last_index = self.log.len() as u64;
                    let last_term = self.log.last().map(|e| e.term).unwrap_or(0);
                    let quorum = (peers.len() / 2 + 1) as u32;
                    // Immediately win if quorum of 1 (single node)
                    if quorum <= 1 {
                        self.role = RaftRole::Leader;
                    }
                    return Some(RaftMessage::RequestVote {
                        term: self.current_term,
                        candidate_id: self.id,
                        last_log_index: last_index,
                        last_log_term: last_term,
                    });
                }
                None
            }
        }
    }

    /// Process an incoming Raft message and optionally produce a reply.
    pub fn handle_message(&mut self, msg: RaftMessage) -> Option<RaftMessage> {
        match msg {
            RaftMessage::RequestVote { term, candidate_id, .. } => {
                if term > self.current_term {
                    self.current_term = term;
                    self.role = RaftRole::Follower;
                    self.voted_for = None;
                }
                let vote_granted = term >= self.current_term
                    && (self.voted_for.is_none() || self.voted_for == Some(candidate_id));
                if vote_granted {
                    self.voted_for = Some(candidate_id);
                    self.elapsed_since_last_event_ms = 0;
                }
                Some(RaftMessage::VoteGranted {
                    term: self.current_term,
                    voter_id: self.id,
                    granted: vote_granted,
                })
            }
            RaftMessage::VoteGranted { term, granted, .. } => {
                if term == self.current_term && self.role == RaftRole::Candidate && granted {
                    self.votes_received += 1;
                    // Assume we know how many peers there are via a simple threshold
                    // The caller is responsible for passing all peer votes
                }
                None
            }
            RaftMessage::AppendEntries { term, leader_id: _, entries, leader_commit, .. } => {
                if term >= self.current_term {
                    self.current_term = term;
                    self.role = RaftRole::Follower;
                    self.voted_for = None;
                    self.elapsed_since_last_event_ms = 0;
                    for entry in entries {
                        self.log.push(entry);
                    }
                    if leader_commit > self.commit_index {
                        self.commit_index = leader_commit.min(self.log.len() as u64);
                    }
                    let match_index = self.log.len() as u64;
                    return Some(RaftMessage::AppendEntriesAck {
                        term: self.current_term,
                        follower_id: self.id,
                        success: true,
                        match_index,
                    });
                }
                Some(RaftMessage::AppendEntriesAck {
                    term: self.current_term,
                    follower_id: self.id,
                    success: false,
                    match_index: self.log.len() as u64,
                })
            }
            RaftMessage::AppendEntriesAck { .. } => None,
        }
    }

    /// Promote to leader once quorum reached. Called by orchestrator.
    pub fn try_promote(&mut self, cluster_size: usize) {
        if self.role == RaftRole::Candidate {
            let quorum = (cluster_size / 2 + 1) as u32;
            if self.votes_received >= quorum {
                self.role = RaftRole::Leader;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DroneState;

    #[test]
    fn test_eligibility_check() {
        let config = RaftConfig::default();
        let mut state = DroneState::default_at_origin(NodeId(1));
        state.battery_pct = 50.0;
        state.link_quality = 0.9;
        assert!(RaftNode::is_eligible_leader(&state, &config));

        state.battery_pct = 5.0;
        assert!(!RaftNode::is_eligible_leader(&state, &config));
    }

    #[test]
    fn test_election_starts_after_timeout() {
        let config = RaftConfig { election_timeout_ms: 100, ..Default::default() };
        let mut node = RaftNode::new(NodeId(1), config);
        let result = node.tick(Duration::from_millis(200), &[]);
        assert!(result.is_some());
        assert_eq!(node.role, RaftRole::Leader); // single node wins immediately
    }
}
