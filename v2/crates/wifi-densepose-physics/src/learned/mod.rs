//! Optional Burn residual artifact boundary.
//!
//! Model training and activation remain evidence-gated. This module enforces
//! signed/hash-addressed artifact identity and the deterministic correction cap.

mod artifact;
pub mod features;
pub mod model;
#[cfg(feature = "learned-cpu")]
pub mod runtime;
pub mod training;

pub use artifact::{
    LearnedArtifact, LearnedArtifactError, LearnedArtifactManifest, LearnedResidual,
    SignatureVerification,
};

/// Re-export the selected framework so downstream trainer crates use the exact
/// workspace-pinned Burn version rather than introducing a second runtime.
pub use burn_core as framework;
#[cfg(feature = "learned-cpu")]
pub use burn_ndarray as cpu_backend;
pub use burn_nn as neural_network;
