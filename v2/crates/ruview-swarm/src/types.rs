//! Core domain types for the swarm control system.

use serde::{Deserialize, Serialize};

/// Unique identifier for a drone node in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u32);

/// Unique identifier for a swarm cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClusterId(pub u32);

/// Unique identifier for a swarm task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub u64);

/// 3-D position in local NED (North-East-Down) frame, metres.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
pub struct Position3D {
    pub x: f64, // north, m
    pub y: f64, // east, m
    pub z: f64, // down, m (negative = above ground)
}

impl Position3D {
    pub fn distance_to(&self, other: &Position3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }
}

/// Velocity in local NED frame, m/s.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Velocity3D {
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

impl Velocity3D {
    pub fn magnitude(&self) -> f64 {
        (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz).sqrt()
    }
}

impl From<(f64, f64, f64)> for Position3D {
    fn from(t: (f64, f64, f64)) -> Self {
        Self { x: t.0, y: t.1, z: t.2 }
    }
}

impl From<Velocity3D> for Position3D {
    fn from(v: Velocity3D) -> Self {
        Self { x: v.vx, y: v.vy, z: v.vz }
    }
}

/// Full kinematic state of a drone node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DroneState {
    pub id: NodeId,
    pub position: Position3D,
    pub velocity: Velocity3D,
    pub heading_rad: f64,
    pub altitude_agl_m: f64,
    pub battery_pct: f32,     // 0.0–100.0
    pub link_quality: f32,    // 0.0–1.0 (RSSI normalised)
    pub timestamp_ms: u64,
}

impl DroneState {
    /// Construct a default state for a node at the origin.
    pub fn default_at_origin(id: NodeId) -> Self {
        Self {
            id,
            position: Position3D::zero(),
            velocity: Velocity3D::default(),
            heading_rad: 0.0,
            altitude_agl_m: 0.0,
            battery_pct: 100.0,
            link_quality: 1.0,
            timestamp_ms: 0,
        }
    }
}

/// CSI detection report from a drone's sensing payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsiDetection {
    pub drone_id: NodeId,
    pub confidence: f32,      // 0.0–1.0
    pub victim_position: Option<Position3D>,
    pub timestamp_ms: u64,
}

/// A cell in the 2-D mission area probability grid.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct GridCell {
    pub x_idx: u32,
    pub y_idx: u32,
    pub victim_probability: f32,   // Bayesian posterior
    pub pheromone: f32,            // stigmergic coverage signal
    pub last_scanned_ms: u64,
}

/// Mission-level task that can be assigned to a drone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmTask {
    pub id: TaskId,
    pub kind: TaskKind,
    pub priority: f32,
    pub target: Position3D,
    pub deadline_ms: Option<u64>,
    pub assigned_to: Option<NodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskKind {
    CoverCell { grid_x: u32, grid_y: u32 },
    InvestigateVictim { estimated_position: Position3D },
    Triangulate { collaborators: Vec<NodeId> },
    ReturnToHome,
    HoverRelay,
    LandEmergency,
}

/// Role of a node within the hierarchical swarm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmRole {
    ClusterHead,
    Worker,
    RelayNode,
    GroundControlStation,
}

/// Failsafe state alias re-exported from failsafe module.
/// Used here to break circular dependency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FailSafeState {
    Nominal,
    AutonomousHold,
    LowBatteryWarn,
    ReturnToHome,
    EmergencyLand,
    EmergencyDiverge,
    ControlledDescent,
}

/// Top-level swarm error type.
#[derive(Debug, thiserror::Error)]
pub enum SwarmError {
    #[error("consensus error: {0}")]
    Consensus(String),
    #[error("communication error: {0}")]
    Communication(String),
    #[error("navigation error: {0}")]
    Navigation(String),
    #[error("security violation: {0}")]
    Security(String),
    #[error("geofence breach at {position:?}")]
    GeofenceBreach { position: Position3D },
    #[error("task allocation failed: {0}")]
    Allocation(String),
    #[error("sensing error: {0}")]
    Sensing(String),
    #[error("config error: {0}")]
    Config(#[from] toml::de::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type SwarmResult<T> = Result<T, SwarmError>;
