//! External system integration: MAVLink v2, PX4 SITL, Gazebo, ROS2 DDS.

pub mod mavlink_messages;
pub mod mission_report;
pub mod swarm_sim;
pub mod telemetry;

pub use mission_report::{MissionReport, SotaComparison, VictimReport};
pub use telemetry::{DroneFrame, TelemetryRecorder};

pub use mavlink_messages::{
    SwarmNodeState, SwarmCsiReport, SwarmClusterHeartbeat, SwarmVictimConfirmed, SwarmMsgId,
};

#[cfg(feature = "itar-unrestricted")]
pub mod flight_controller;

#[cfg(feature = "itar-unrestricted")]
pub use flight_controller::{FlightController, FlightMode, SimulatedFlightController};
