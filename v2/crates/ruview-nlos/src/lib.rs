//! Consumer time-of-flight non-line-of-sight sensing for RuView.
//!
//! This crate preserves the zone-level photon timing histograms needed by
//! motion-induced aperture sampling. It deliberately rejects ordinary depth
//! maps as live NLOS evidence. The deterministic simulator and its benchmarks
//! are evidence level L0 (synthetic); only a captured, witnessed hardware run
//! can satisfy the ADR-331 reproduction gate.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bridge;
pub mod calibration;
pub mod fusion;
pub mod ingest;
pub mod protocol;
pub mod simulator;
pub mod tracker;

#[cfg(feature = "server")]
pub mod server;

pub use bridge::{RuFieldObservation, TemporalFeatureMemory, WorldGraphUpdate};
pub use calibration::{Calibration, CalibrationConfig, PreprocessedFrame};
pub use fusion::{CsiSpatialPrior, FusionScope};
pub use ingest::{StAsciiDecoder, StDecoderConfig};
pub use protocol::{
    EvidenceLevel, FrameSource, NlosTrack, Provenance, SensorPose, TrackEnvelope, TrackState,
    TransientFrame, TransientKind, TransientZone, Vec3,
};
pub use simulator::{BenchmarkReport, SyntheticScene};
pub use tracker::{CanonicalObject, MotionApertureTracker, TrackerConfig};

/// Transient input schema identifier.
pub const TRANSIENT_SCHEMA_V1: &str = "ruview.nlos.transient.v1";
/// Track output schema identifier shared by Rust, Swift, and TypeScript.
pub const TRACK_SCHEMA_V1: &str = "ruview.nlos.track.v1";
/// Maximum JSON frame accepted by network clients.
pub const MAX_WIRE_BYTES: usize = 256 * 1024;
/// Maximum histogram zones accepted from one consumer sensor.
pub const MAX_ZONES: usize = 64;
/// Maximum temporal bins accepted per zone.
pub const MAX_BINS: usize = 128;
/// Maximum simultaneous hidden tracks on the public contract.
pub const MAX_TRACKS: usize = 16;
