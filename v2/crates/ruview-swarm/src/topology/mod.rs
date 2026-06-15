//! Swarm topology: Raft consensus, gossip dissemination, mesh management.

// NOTE: Raft consensus is ITAR-controlled (USML Category VIII(h)(12)).
// Gossip and mesh are ungated — they are not controlled technologies.
#[cfg(feature = "itar-unrestricted")]
pub mod raft;
pub mod gossip;
pub mod mesh;

#[cfg(feature = "itar-unrestricted")]
pub use raft::{RaftConfig, RaftNode, RaftRole};
pub use gossip::GossipState;
pub use mesh::MeshTopology;
