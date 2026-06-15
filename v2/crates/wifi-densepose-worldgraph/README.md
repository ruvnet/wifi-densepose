# wifi-densepose-worldgraph

**The environmental digital twin for RF sensing — a typed, evidence-tracked graph of a building and the people in it.**

[![crates.io](https://img.shields.io/crates/v/wifi-densepose-worldgraph.svg)](https://crates.io/crates/wifi-densepose-worldgraph)
[![docs.rs](https://docs.rs/wifi-densepose-worldgraph/badge.svg)](https://docs.rs/wifi-densepose-worldgraph)

Part of the [RuView / WiFi-DensePose](https://github.com/ruvnet/RuView) project. Implements **ADR-139**.

---

## What it is (plain language)

When you sense a space with WiFi/RF (people, motion, vital signs), you get a firehose of *frames*.
What you actually want is a **living map**: which rooms exist, where the walls and doorways are, which
sensors watch which zones, where each person is right now, and *why the system believes that* — with
enough structure to reason over and enough provenance to trust.

`wifi-densepose-worldgraph` is that map. It's a **typed graph** (built on [`petgraph`](https://crates.io/crates/petgraph)):

- **Nodes** are real things — `Room`, `Zone`, `Wall`, `Doorway`, `Sensor`, `RfLink`, `PersonTrack`, `ObjectAnchor`, `Event`, and `SemanticState` (a belief).
- **Edges** are typed relations — `Observes`, `LocatedIn`, `AdjacentTo`, `Supports`, `Contradicts`, `DerivedFrom`, `PrivacyLimitedBy`.

It stores **fused beliefs, not raw frames** — it sits *downstream* of signal fusion and *upstream* of the
semantic/agent layer. Every belief (`SemanticState`) is required to carry **provenance**: the signal
evidence, the model, the calibration id, and the privacy decision that produced it. That's enforced
*structurally*, so "where did this conclusion come from?" always has an answer.

## Why a graph (and not an occupancy grid or an event log)?

| Approach | Good at | Misses |
|---|---|---|
| **Raw event log** | append-only history, audit | no structure; can't ask "who's in the kitchen?" without re-deriving it |
| **Occupancy grid / voxels** | dense geometry, ML input | no identity, no relations, no provenance, no semantics |
| **Scene graph (this crate)** | relations, identity, semantics, provenance, privacy | not a dense field — pair it with a grid for ML (see [`wifi-densepose-worldmodel`](https://crates.io/crates/wifi-densepose-worldmodel)) |

The graph is the **symbolic, interpretable** layer. It answers *relational* questions ("is this person in a
zone observed by sensor X?", "are these two beliefs contradictory?") in O(neighbors), and it keeps the
*why* attached to every *what*.

## Features

- 🧱 **Typed node/edge model** — a closed `enum` schema (serde-tagged) → deterministic, schema-versioned wire format.
- 🧭 **Geometry in ENU meters** — rooms/zones/walls/doorways carry East-North-Up bounds; walls carry `rf_attenuation_db`.
- 🧠 **Beliefs with mandatory provenance** — `SemanticState` → `SemanticProvenance { signal evidence, model, calibration_id, privacy_decision }`.
- 🔀 **Evidence reasoning built in** — `Supports` / `Contradicts` / `DerivedFrom` edges let you score and challenge conclusions, not just store them.
- 🔒 **Privacy as a first-class edge** — `PrivacyLimitedBy` + `apply_privacy_mode()` roll up what a given mode/action is allowed to see.
- 💾 **Deterministic JSON persistence** — `to_json` / `from_json` (the RVF payload), schema-versioned.
- 🚫 **`#![forbid(unsafe_code)]`**, `missing_docs = warn`. Pure Rust, no async, edge-deployable (builds clean on aarch64 — runs on a Raspberry Pi).

## Install

```toml
[dependencies]
wifi-densepose-worldgraph = "0.3"
```

## Usage

```rust
use wifi_densepose_worldgraph::{WorldGraph, WorldNode, WorldEdge, ZoneBoundsEnu};
// (GeoRegistration comes from wifi-densepose-geo — it anchors ENU to a real lat/lon origin)

let mut wg = WorldGraph::new(registration);

// Add a room and a sensor that observes it.
let living_room = wg.upsert_node(WorldNode::Room {
    id: Default::default(),
    area_id: Some("living_room".into()),
    name: "Living Room".into(),
    bounds_enu: ZoneBoundsEnu { /* … */ },
    floor: 0,
});
let sensor = wg.upsert_node(/* WorldNode::Sensor { … } */);
wg.add_edge(sensor, living_room, WorldEdge::Observes { quality: 0.9, last_seen_unix_ms: now });

// Query relations.
let watched = wg.observed_by(sensor);          // what this sensor sees
let room = wg.room_for_area("living_room");    // area_id → room node

// Record a belief WITH provenance, and a contradiction against it.
wg.add_semantic_state(/* state + SemanticProvenance */);
wg.add_contradiction(belief_a, belief_b, /* magnitude */, "two sensors disagree");

// Privacy rollup for a mode/action, then persist.
let rollup = wg.apply_privacy_mode("HOME", "occworld_inference", |node| /* allow? */ true);
let bytes = wg.to_json()?;                      // RVF payload
let restored = WorldGraph::from_json(&bytes)?;
```

## Technical details

- **Backing store:** `petgraph::StableDiGraph` (stable indices across removals) wrapped as `WorldGraph`.
- **Identity:** every node has a `WorldId`; `upsert_node` is idempotent on identity.
- **Snapshots:** `snapshot()` → `WorldGraphSnapshot` (a serializable point-in-time view) with a `PrivacyRollup`.
- **Schema versioning:** `SCHEMA_VERSION` is embedded in the JSON; the closed enum model means readers fail fast on incompatible payloads rather than silently mis-parsing.
- **Coordinates:** ENU (East/North/Up) meters relative to a `GeoRegistration` origin (`wifi-densepose-geo`), so the twin can be georeferenced to a real building.
- **Position in the pipeline:** `fusion (ADR-137) → WorldGraph (ADR-139) → semantic/agent layer (ADR-140) → eval harness (ADR-145)`. For **forward prediction** (where will people be next?), pair it with [`wifi-densepose-worldmodel`](https://crates.io/crates/wifi-densepose-worldmodel), which turns `PersonTrack` history into predicted occupancy + trajectory priors.

## Related crates

| Crate | Role |
|---|---|
| [`wifi-densepose-worldmodel`](https://crates.io/crates/wifi-densepose-worldmodel) | Forward **prediction** — occupancy world model over this graph's tracks |
| [`wifi-densepose-geo`](https://crates.io/crates/wifi-densepose-geo) | Geospatial registration (ENU ↔ lat/lon, DEM, OSM) |

## License

Licensed as the parent project. See the [repository](https://github.com/ruvnet/RuView).
