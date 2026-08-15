//! Native Rust physics-constrained pose assessment (ADR-323).
//!
//! The default feature contains only deterministic kinematic code. Optional
//! dynamics and learned backends cannot override hard validation, correction,
//! provenance, or confidence invariants.

#![forbid(unsafe_code)]
#![allow(missing_docs)]

pub mod config;
pub mod constraints;
#[cfg(feature = "dynamics")]
pub mod dynamics;
pub mod engine;
pub mod error;
#[cfg(feature = "learned")]
pub mod learned;
pub mod metrics;
pub mod projector;
pub mod skeleton;
pub mod uncertainty;

pub use config::PhysicsConfig;
pub use engine::{CorrectionAuthorization, PhysicsEngine, VerifiedFrameContext};
pub use error::PhysicsError;
