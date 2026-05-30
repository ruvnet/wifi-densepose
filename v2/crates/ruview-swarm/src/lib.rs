//! Drone swarm control system — ADR-148.
//!
//! Hierarchical-mesh topology · Raft consensus · MAPPO MARL · CSI sensing integration

pub mod types;
pub mod topology;
pub mod formation;
pub mod planning;
pub mod allocation;
pub mod sensing;
pub mod marl;
pub mod security;
pub mod failsafe;
pub mod config;
pub mod demo;
pub mod evals;
pub mod integration;
pub mod bench_support;
pub mod orchestrator;
pub mod ruflo;

pub use types::{
    ClusterId, CsiDetection, DroneState, FailSafeState, GridCell, NodeId,
    Position3D, SwarmError, SwarmResult, SwarmRole, SwarmTask, TaskId, TaskKind, Velocity3D,
};
pub use config::SwarmConfig;
