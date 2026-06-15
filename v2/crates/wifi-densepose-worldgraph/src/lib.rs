//! # WiFi-DensePose WorldGraph (ADR-139)
//!
//! The environmental digital twin for the RuView streaming engine: a typed
//! [`petgraph`] `StableDiGraph` of rooms, zones, walls, doorways, sensors, RF
//! links, person tracks, object anchors, events, and semantic-state beliefs,
//! connected by typed relations (observes / located_in / adjacent_to /
//! supports / contradicts / derived_from / privacy_limited_by).
//!
//! It sits downstream of fusion (ADR-137) — storing fused *beliefs*, not raw
//! frames — and upstream of the semantic/agent layer (ADR-140) and evaluation
//! harness (ADR-145). Every [`model::WorldNode::SemanticState`] carries
//! mandatory [`model::SemanticProvenance`] (signal evidence + model +
//! calibration + privacy decision), honouring the house rule structurally.
//!
//! Persistence is via [`graph::WorldGraph::to_json`] /
//! [`graph::WorldGraph::from_json`] (the RVF payload); the serde-enum node/edge
//! model guarantees a deterministic, schema-versioned wire layout.

#![forbid(unsafe_code)]

pub mod error;
pub mod graph;
pub mod model;

pub use error::WorldGraphError;
pub use graph::{PrivacyRollup, WorldGraph, WorldGraphSnapshot, SCHEMA_VERSION};
pub use model::{
    AnchorKind, EnuPoint, SemanticProvenance, SensorModality, WorldEdge, WorldId, WorldNode,
    ZoneBoundsEnu,
};
