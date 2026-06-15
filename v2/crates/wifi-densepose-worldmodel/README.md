# wifi-densepose-worldmodel

**Forward prediction for RF sensing — turn where people *were* into where they'll *be*, as occupancy + trajectory priors.**

[![crates.io](https://img.shields.io/crates/v/wifi-densepose-worldmodel.svg)](https://crates.io/crates/wifi-densepose-worldmodel)
[![docs.rs](https://docs.rs/wifi-densepose-worldmodel/badge.svg)](https://docs.rs/wifi-densepose-worldmodel)

Part of the [RuView / WiFi-DensePose](https://github.com/ruvnet/RuView) project. Implements **ADR-147**.

---

## What it is (plain language)

[`wifi-densepose-worldgraph`](https://crates.io/crates/wifi-densepose-worldgraph) tells you **what the room is
*now*** (who's where, the walls, the doorways). This crate answers the next question: **what happens *next*?**

It's a **thin, async client** to an *occupancy world model* (OccWorld). You give it a short history of where
people have been (their `PersonTrack` positions); it rasterizes that into 3-D occupancy grids, ships them to
an OccWorld inference process, and gets back:

- **predicted future occupancy** (the model rolls the scene forward N steps), and
- **`TrajectoryPrior`s** — per-person predicted waypoints you can feed straight into a Kalman pose tracker to
  stabilize and *anticipate* movement (e.g. someone heading for a doorway).

It is **camera-free and privacy-first**: the world model reasons over **occupancy voxels**, not video — so it
predicts *where*, never *who-looks-like-what*. (This is the deliberate contrast with pixel-space robot world
models like ByteDance's IRASim: same "predict-the-future-conditioned-on-state" idea, kept in occupancy space
for privacy and edge deployment.)

## Where it sits

```
RF frames → fusion → WorldGraph (what is)  ──PersonTrack history──►  wifi-densepose-worldmodel
                          ▲                                                   │
                          │                                          OccWorld inference (Python subprocess)
                          └──────────  TrajectoryPriors (what's next)  ◄──────┘
                                       (injected back into the Kalman tracker)
```

## Symbolic vs predictive — the two halves of the world model

| | `wifi-densepose-worldgraph` | `wifi-densepose-worldmodel` (this crate) |
|---|---|---|
| **Question** | "What is the room *now*?" | "What happens *next*?" |
| **Representation** | typed symbolic graph (rooms, tracks, beliefs) | dense 3-D occupancy voxels + trajectory priors |
| **Nature** | interpretable, evidential, provenance-tracked | predictive, learned (OccWorld) |
| **Compute** | pure Rust, microseconds, edge | Rust client + GPU inference subprocess |
| **Output** | relations & beliefs | future occupancy + per-person waypoints |

Use them together: the graph supplies tracks + privacy decisions; this crate predicts forward and feeds the
priors back.

## Features

- 🔌 **Thin async bridge** — `OccWorldBridge` talks to the OccWorld inference process over a Unix socket (newline-delimited JSON request/response).
- 🧊 **Occupancy rasterization** — `worldgraph_to_occupancy()` turns person positions + scene bounds into a 3-D voxel grid (`200 × 200 × 16` by default; `CLASS_PERSON` / `CLASS_FREE` semantics).
- 🧭 **ENU ↔ voxel mapping** — `SceneBounds::to_voxel_xy()` / `to_voxel_z()` with a configurable resolution (e.g. 0.1 m).
- 🛰️ **Trajectory priors** — predicted per-`track_id` waypoints, ready for Kalman injection.
- 🔁 **Backend-swappable** — the request/response contract (`OccupancyWorldModelRequest` → response with `confidence` + `trajectory_priors`) is model-agnostic (OccWorld today, RoboOccWorld / others later).
- 🔒 **Privacy-gated by design** — meant to be called only when the WorldGraph's privacy mode permits it (ADR-141); reasons over occupancy, never pixels.
- 🚫 **`#![forbid(unsafe_code)]`**, `missing_docs = warn`.

## Install

```toml
[dependencies]
wifi-densepose-worldmodel = "0.3"
```

> The bridge uses Unix domain sockets (`tokio`), so the client targets Unix-like hosts (Linux/macOS — e.g. a Raspberry Pi appliance). The data types (occupancy, bounds, priors) are platform-agnostic.

## Usage

```rust
use wifi_densepose_worldmodel::{
    OccWorldBridge, OccupancyWorldModelRequest, SceneBoundsJson, worldgraph_to_occupancy,
};
use wifi_densepose_worldmodel::occupancy::{PersonPosition, SceneBounds};

# async fn example() -> Result<(), wifi_densepose_worldmodel::WorldModelError> {
let bridge = OccWorldBridge::new("/tmp/occworld.sock");

let bounds = SceneBounds { min_e: -10.0, min_n: -10.0, max_e: 10.0, max_n: 10.0 };
let persons = vec![PersonPosition { track_id: 1, east_m: 2.0, north_m: 3.0, up_m: 1.0 }];

// Rasterize current positions → an occupancy frame (0.1 m voxels).
let frame = worldgraph_to_occupancy(&persons, &bounds, 0.1);

// Ask OccWorld to roll the scene forward 15 steps.
let response = bridge.predict(OccupancyWorldModelRequest {
    past_frames: vec![frame],
    voxel_resolution_m: 0.1,
    scene_bounds: SceneBoundsJson { min_e: bounds.min_e, min_n: bounds.min_n,
                                    max_e: bounds.max_e, max_n: bounds.max_n },
    prediction_steps: 15,
}).await?;

println!("confidence={:.2}", response.confidence);
for prior in &response.trajectory_priors {
    println!("track {} → {} predicted waypoints", prior.track_id, prior.waypoints.len());
}
# Ok(())
# }
```

## Technical details

- **Wire protocol:** newline-delimited JSON over a Unix socket; one request → one response. The Python side
  (OccWorld) loads `PersonTrack` history as a `(B, F, H, W, D)` occupancy tensor and returns predicted voxels
  decoded into `TrajectoryPrior`s.
- **Grid:** `GRID_WIDTH=200 × GRID_HEIGHT=200 × GRID_DEPTH=16` voxels by default; `CLASS_PERSON=10`,
  `CLASS_FREE=17` (RuView indoor class remap from the nuScenes outdoor set).
- **Resolution:** configurable meters-per-voxel; `to_voxel_xy`/`to_voxel_z` handle ENU→index.
- **Backend:** OccWorld (1.65 GB VRAM, ~375 ms/inference on an RTX-class GPU; runs on the Pi+Hailo appliance
  tier). Cosmos is the deferred heavier alternative (ADR-148).
- **Provenance:** predictions carry the originating `calibration_id` + privacy decision so downstream
  consumers can gate on quality and consent (ADR-141).

## Related crates

| Crate | Role |
|---|---|
| [`wifi-densepose-worldgraph`](https://crates.io/crates/wifi-densepose-worldgraph) | The symbolic twin ("what is") that supplies the tracks this crate predicts from |

## License

Licensed as the parent project. See the [repository](https://github.com/ruvnet/RuView).
