//! Localization module for survivor position estimation.
//!
//! This module provides:
//! - Triangulation from multiple access points
//! - Depth estimation through debris
//! - Position fusion combining multiple techniques

mod depth;
mod fusion;
mod triangulation;

pub use depth::{DepthEstimator, DepthEstimatorConfig};
pub use fusion::{LocalizationService, PositionFuser};
#[cfg(feature = "ruvector")]
pub use triangulation::solve_tdoa_triangulation;
pub use triangulation::{TriangulationConfig, Triangulator};
