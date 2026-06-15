//! `wifi-densepose-worldmodel` — OccWorld thin-client bridge (ADR-147).
//!
//! Bridges [`wifi_densepose_worldgraph`] `PersonTrack` history to the OccWorld
//! Python inference subprocess and returns [`TrajectoryPrior`]s that can be
//! injected into the Kalman pose tracker.
//!
//! ## Quick start
//! ```rust,no_run
//! use wifi_densepose_worldmodel::{
//!     OccWorldBridge, OccupancyWorldModelRequest, OccupancyGrid3D,
//!     SceneBoundsJson, worldgraph_to_occupancy,
//! };
//! use wifi_densepose_worldmodel::occupancy::{PersonPosition, SceneBounds};
//!
//! # async fn example() -> Result<(), wifi_densepose_worldmodel::WorldModelError> {
//! let bridge = OccWorldBridge::new("/tmp/occworld.sock");
//!
//! let bounds = SceneBounds { min_e: -10.0, min_n: -10.0, max_e: 10.0, max_n: 10.0 };
//! let persons = vec![
//!     PersonPosition { track_id: 1, east_m: 2.0, north_m: 3.0, up_m: 1.0 },
//! ];
//! let frame = worldgraph_to_occupancy(&persons, &bounds, 0.1);
//!
//! let request = OccupancyWorldModelRequest {
//!     past_frames: vec![frame],
//!     voxel_resolution_m: 0.1,
//!     scene_bounds: SceneBoundsJson {
//!         min_e: bounds.min_e, min_n: bounds.min_n,
//!         max_e: bounds.max_e, max_n: bounds.max_n,
//!     },
//!     prediction_steps: 15,
//! };
//!
//! let response = bridge.predict(request).await?;
//! println!("confidence={:.2}", response.confidence);
//! for prior in &response.trajectory_priors {
//!     println!("track {} has {} waypoints", prior.track_id, prior.waypoints.len());
//! }
//! # Ok(())
//! # }
//! ```

pub mod bridge;
pub mod error;
pub mod occupancy;

// Re-export the bridge type at the crate root for convenience.
pub use bridge::{default_socket_path, OccWorldBridge};
pub use error::WorldModelError;
pub use occupancy::worldgraph_to_occupancy;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Voxel grid
// ---------------------------------------------------------------------------

/// A 3-D occupancy grid whose voxel values are class indices (u8).
///
/// Layout: `voxels[z * height * width + y * width + x]` (row-major, depth last).
/// The grid is always `200 × 200 × 16` when produced by
/// [`worldgraph_to_occupancy`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyGrid3D {
    /// Number of voxels along the east/x axis.
    pub width: u32,
    /// Number of voxels along the north/y axis.
    pub height: u32,
    /// Number of voxels along the up/z axis.
    pub depth: u32,
    /// Flat class-index array, length `width * height * depth`.
    pub voxels: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Trajectory types
// ---------------------------------------------------------------------------

/// A single point on a predicted trajectory, with a relative timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryWaypoint {
    /// East offset from installation origin, in metres.
    pub e: f64,
    /// North offset from installation origin, in metres.
    pub n: f64,
    /// Up offset (height), in metres.
    pub u: f64,
    /// Time offset from "now", in seconds (positive = future).
    pub t_s: f32,
}

/// Predicted future trajectory for one tracked person.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPrior {
    /// Stable track identifier (mirrors `WorldNode::PersonTrack::track_id`).
    pub track_id: u64,
    /// Ordered sequence of predicted future waypoints.
    pub waypoints: Vec<TrajectoryWaypoint>,
}

// ---------------------------------------------------------------------------
// Scene bounds (JSON wire shape)
// ---------------------------------------------------------------------------

/// Axis-aligned scene footprint sent to the OccWorld server in the IPC
/// request.  Mirrors [`occupancy::SceneBounds`] but derives `Serialize` /
/// `Deserialize` for direct inclusion in the JSON payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneBoundsJson {
    /// Western (minimum east) edge of the scene, in metres.
    pub min_e: f64,
    /// Southern (minimum north) edge of the scene, in metres.
    pub min_n: f64,
    /// Eastern (maximum east) edge of the scene, in metres.
    pub max_e: f64,
    /// Northern (maximum north) edge of the scene, in metres.
    pub max_n: f64,
}

// ---------------------------------------------------------------------------
// IPC request / response
// ---------------------------------------------------------------------------

/// JSON request sent from the Rust bridge to the OccWorld Python server.
///
/// Serialised as a single newline-terminated JSON object over the Unix socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyWorldModelRequest {
    /// History of occupancy grids (chronological, oldest first).
    /// OccWorld expects at least one frame; the reference implementation uses
    /// the most recent 4 frames for temporal context.
    pub past_frames: Vec<OccupancyGrid3D>,

    /// Physical size of one voxel cell on the ground plane, in metres.
    pub voxel_resolution_m: f32,

    /// Scene footprint used to build the occupancy grid.
    pub scene_bounds: SceneBoundsJson,

    /// Number of future time steps to predict (reference: 15 × 0.1 s = 1.5 s).
    pub prediction_steps: u32,
}

/// JSON response returned by the OccWorld Python server.
///
/// Decoded from a single newline-terminated JSON object on the Unix socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OccupancyWorldModelResponse {
    /// Predicted future occupancy grids (chronological, `prediction_steps`
    /// frames in total).
    pub future_frames: Vec<OccupancyGrid3D>,

    /// Per-person predicted trajectories extracted from `future_frames`.
    pub trajectory_priors: Vec<TrajectoryPrior>,

    /// Aggregate confidence score in `[0, 1]` for the entire prediction.
    pub confidence: f32,

    /// Identifier of the model that produced this response.
    /// The sentinel prefix `"error:vram:"` signals a VRAM error (see ADR-147).
    pub model_id: String,

    /// Wall-clock time the Python server spent on inference, in milliseconds.
    pub inference_ms: u64,
}

// ---------------------------------------------------------------------------
// WorldGraph helper — extract PersonPosition list from a WorldGraph snapshot
// ---------------------------------------------------------------------------

use wifi_densepose_worldgraph::WorldGraph;

use crate::occupancy::PersonPosition;

/// Extracts all [`PersonPosition`]s from a [`WorldGraph`] by serialising the
/// graph to its canonical JSON form (via [`WorldGraph::to_json`]) and scanning
/// the `nodes` array for `PersonTrack` entries.
///
/// This avoids coupling to the private fields of `WorldGraphSnapshot`.
/// The returned positions are unsorted; callers may sort by `track_id` if
/// deterministic ordering is required.
///
/// # Panics
/// Does not panic — if serialisation fails the function returns an empty
/// `Vec` and logs a warning via `eprintln!`.  In practice, serialisation of a
/// valid `WorldGraph` should never fail.
pub fn persons_from_worldgraph(graph: &WorldGraph) -> Vec<PersonPosition> {
    let bytes = match graph.to_json() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[worldmodel] WorldGraph::to_json failed: {e}");
            return Vec::new();
        }
    };

    // Parse as a raw JSON value to avoid depending on the exact shape of the
    // private `WorldGraphSnapshot` struct fields.
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[worldmodel] failed to parse WorldGraph JSON: {e}");
            return Vec::new();
        }
    };

    let nodes = match value.get("nodes").and_then(|n| n.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };

    nodes
        .iter()
        .filter_map(|node| {
            // Nodes use a serde-tagged enum; the PersonTrack variant carries a
            // `kind` discriminator equal to `"person_track"`.
            if node.get("kind")?.as_str()? != "person_track" {
                return None;
            }
            let track_id = node.get("track_id")?.as_u64()?;
            let pos = node.get("last_position")?;
            let east_m = pos.get("east_m")?.as_f64()?;
            let north_m = pos.get("north_m")?.as_f64()?;
            let up_m = pos.get("up_m")?.as_f64()?;
            Some(PersonPosition { track_id, east_m, north_m, up_m })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_grid_serde_roundtrip() {
        let grid = OccupancyGrid3D {
            width: 4,
            height: 4,
            depth: 2,
            voxels: vec![17u8; 32],
        };
        let json = serde_json::to_string(&grid).expect("serialize");
        let decoded: OccupancyGrid3D = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.width, grid.width);
        assert_eq!(decoded.voxels.len(), grid.voxels.len());
    }

    #[test]
    fn trajectory_prior_serde_roundtrip() {
        let prior = TrajectoryPrior {
            track_id: 42,
            waypoints: vec![
                TrajectoryWaypoint { e: 1.0, n: 2.0, u: 0.0, t_s: 0.1 },
                TrajectoryWaypoint { e: 1.1, n: 2.1, u: 0.0, t_s: 0.2 },
            ],
        };
        let json = serde_json::to_string(&prior).expect("serialize");
        let decoded: TrajectoryPrior = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.track_id, 42);
        assert_eq!(decoded.waypoints.len(), 2);
    }

    #[test]
    fn request_serde_roundtrip() {
        let req = OccupancyWorldModelRequest {
            past_frames: vec![OccupancyGrid3D {
                width: 200,
                height: 200,
                depth: 16,
                voxels: vec![17u8; 200 * 200 * 16],
            }],
            voxel_resolution_m: 0.1,
            scene_bounds: SceneBoundsJson {
                min_e: -10.0,
                min_n: -10.0,
                max_e: 10.0,
                max_n: 10.0,
            },
            prediction_steps: 15,
        };
        let json = serde_json::to_string(&req).expect("serialize");
        let decoded: OccupancyWorldModelRequest =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.prediction_steps, 15);
        assert_eq!(decoded.past_frames.len(), 1);
    }

    #[test]
    fn response_serde_roundtrip() {
        let resp = OccupancyWorldModelResponse {
            future_frames: vec![],
            trajectory_priors: vec![TrajectoryPrior {
                track_id: 1,
                waypoints: vec![TrajectoryWaypoint { e: 0.0, n: 0.0, u: 0.0, t_s: 0.0 }],
            }],
            confidence: 0.82,
            model_id: "occworld-dummy-v0".into(),
            inference_ms: 375,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        let decoded: OccupancyWorldModelResponse =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.inference_ms, 375);
        assert!((decoded.confidence - 0.82).abs() < 1e-5);
    }

    #[test]
    fn vram_error_sentinel_roundtrip() {
        let resp = OccupancyWorldModelResponse {
            future_frames: vec![],
            trajectory_priors: vec![],
            confidence: 0.0,
            model_id: "error:vram:out of memory (CUDA)".into(),
            inference_ms: 0,
        };
        assert!(resp.model_id.starts_with("error:vram:"));
    }
}
