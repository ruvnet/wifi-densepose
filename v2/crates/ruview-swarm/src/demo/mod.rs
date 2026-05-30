//! Demo scenario runner — synthetic CSI with configurable victim positions.
//!
//! Wires together a [`SyntheticCsiGenerator`] and pre-built [`DemoScenario`]
//! definitions for rapid scenario validation without real hardware.

pub mod synthetic_csi;
pub mod scenario;

pub use synthetic_csi::SyntheticCsiGenerator;
pub use scenario::{DemoScenario, ScenarioResult};
