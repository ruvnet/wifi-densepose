//! Detection module for vital signs detection from CSI data.
//!
//! This module provides detectors for:
//! - Breathing patterns
//! - Heartbeat signatures
//! - Movement classification
//! - Ensemble classification combining all signals

mod breathing;
mod ensemble;
mod heartbeat;
mod movement;
mod pipeline;

#[cfg(feature = "ruvector")]
pub use breathing::CompressedBreathingBuffer;
pub use breathing::{BreathingDetector, BreathingDetectorConfig};
pub use ensemble::{EnsembleClassifier, EnsembleConfig, EnsembleResult, SignalConfidences};
#[cfg(feature = "ruvector")]
pub use heartbeat::CompressedHeartbeatSpectrogram;
pub use heartbeat::{HeartbeatDetector, HeartbeatDetectorConfig};
pub use movement::{MovementClassifier, MovementClassifierConfig};
pub use pipeline::{CsiDataBuffer, DetectionConfig, DetectionPipeline, VitalSignsDetector};
