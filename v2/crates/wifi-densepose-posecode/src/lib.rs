//! Multi-actor semantic motion scenes for RuView.
//!
//! This crate uses the PoseCode movement vocabulary and implements the
//! RuView 0.2 multi-actor extension from ADR-266. It deliberately separates
//! the high-rate observed pose stream from the compact semantic scene:
//! [`adapter::RuViewAdapter`] converts persistent RuView tracks into observed
//! frames, [`segmenter::PhaseSegmenter`] reduces those frames to synchronized
//! phases, and [`serialize::to_posecode`] emits deterministic text.

pub mod adapter;
pub mod error;
pub mod model;
pub mod parser;
pub mod segmenter;
pub mod serialize;
pub mod validate;

pub use adapter::{
    AdapterConfig, KeypointObservation, RuViewAdapter, TrackObservation, TrackState,
};
pub use error::{Error, Result};
pub use model::*;
pub use parser::parse_posecode;
pub use segmenter::{PhaseSegmenter, SegmenterConfig};
pub use serialize::to_posecode;
pub use validate::{validate_scene, ValidationConfig, ValidationIssue, ValidationSeverity};

/// Protocol version implemented by this crate.
pub const PROTOCOL_VERSION: &str = "0.2";
