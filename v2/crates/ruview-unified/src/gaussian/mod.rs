//! RF-aware Gaussian spatial memory (ADR-275).
//!
//! The persistent scene representation: anisotropic 3-D Gaussians that carry
//! *both* geometric/semantic state (position, scale, orientation, occupancy,
//! semantic embedding) *and* RF state (per-band × incident-angle
//! reflectivity, Doppler/motion, extinction), plus the bookkeeping a world
//! model needs (confidence, provenance, timestamp, decay, entity links).
//!
//! Three capabilities distinguish this from a point cloud:
//!
//! 1. **Fusion**: observations of the same region merge by
//!    confidence-weighted precision updates instead of accumulating
//!    duplicates ([`map::GaussianMap::insert`]).
//! 2. **RF queries**: the map predicts complex channel gain between any two
//!    points via Beer–Lambert transmittance through the Gaussians
//!    ([`gain::channel_gain`]) — an empty map degrades to *exact* free-space
//!    Friis, and measured-vs-predicted residuals update the map inversely
//!    ([`gain::observe_link`]).
//! 3. **Task-gated activation**: reasoning code touches a bounded subgraph
//!    ([`graph::SceneGraph::activate`]), never the full memory.

pub mod gain;
pub mod graph;
pub mod map;
pub mod primitive;

pub use gain::{channel_gain, gain_db, observe_link};
pub use graph::{EntityKind, RelationKind, SceneEdge, SceneGraph, SceneNode};
pub use map::GaussianMap;
pub use primitive::{Band, EntityLink, MotionState, Provenance, RfGaussian};
