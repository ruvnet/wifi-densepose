//! # `ruview-ontology` — the canonical spatial ontology (ADR-306, ADR-300 §6)
//!
//! One `Site ▸ Building ▸ Floor ▸ Space ▸ Zone` containment model, plus the
//! leaf entities `Sensor`, `Person`, `Object`, `Observation`, `Track`, and
//! `Event`, that **every** RuView surface reads from and writes to. The same
//! physical fact — "a person is in the kitchen" — is encoded *once* here and
//! every surface (MQTT/Home-Assistant, REST, WebSocket, RuField, Matter, agent
//! queries) is a *projection* of this model rather than an independent schema.
//!
//! This crate is a **pure data / relationship representation**: no I/O, no
//! async, no inference. It says nothing about *how* a `Track` or `Event` is
//! produced (that is owned by ADR-301/ADR-307/ADR-302) and makes no accuracy
//! claim. Identity is caller-supplied and deterministic — ids are never
//! randomly generated here.
//!
//! ## Model at a glance
//!
//! ```text
//! Site ▸ Building ▸ Floor ▸ Space ▸ Zone
//!                                    └▸ { Sensor, Person, Object,
//!                                          Observation, Track, Event }
//! ```
//!
//! - The spine is enforced single-parent by the [`WorldGraph`] registry: a
//!   `Zone` is part of exactly one `Space`, a `Space` on exactly one `Floor`,
//!   and so on. Inserting a child whose parent is absent is rejected with
//!   [`OntologyError::MissingParent`].
//! - Each leaf carries a [`Container`] (a `Space` or `Zone`); the registry
//!   resolves it upward with [`WorldGraph::space_of`] / [`WorldGraph::zone_of`].
//! - Every leaf carries exactly one [`EvidenceLevel`] and a
//!   [`SemanticProvenance`] record, so lineage and evidence level travel *with*
//!   the fact across every projection and cannot be silently dropped.
//!
//! ## Example
//!
//! ```
//! use ruview_ontology::*;
//!
//! let mut g = WorldGraph::new();
//! g.add_site(Site { id: SiteId::new("home")?, name: "Home".into() })?;
//! g.add_building(Building {
//!     id: BuildingId::new("b1")?, parent: SiteId::new("home")?, name: "House".into(),
//! })?;
//! g.add_floor(Floor {
//!     id: FloorId::new("f1")?, parent: BuildingId::new("b1")?, level: 0, name: "Ground".into(),
//! })?;
//! g.add_space(Space {
//!     id: SpaceId::new("kitchen")?, parent: FloorId::new("f1")?,
//!     area_id: Some("area-42".into()), name: "Kitchen".into(),
//! })?;
//!
//! let here = Container::Space { id: SpaceId::new("kitchen")? };
//! g.add_person(Person {
//!     id: PersonId::new("p1")?, located_in: here.clone(),
//!     evidence_level: EvidenceLevel::L2,
//!     provenance: SemanticProvenance::declared("fusion@1"),
//! })?;
//!
//! assert_eq!(g.space_of(&here).unwrap().name, "Kitchen");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Migration path from existing per-surface shapes (docs only)
//!
//! ADR-306 §3 requires a documented, tested bidirectional mapping from each
//! existing per-surface schema onto these canonical types. This crate does not
//! edit those surfaces; the mappings below are the contract each surface's
//! projection implements when it is cut over (one surface at a time). Until a
//! surface is cut over, its mapping layer is authoritative and round-tripped so
//! no fact is lost.
//!
//! | Legacy shape | Source | Canonical target |
//! |---|---|---|
//! | `NodeInference` | ADR-297 MQTT/HA mapper | `Sensor` + an `Observation` whose `sensor` is that node; node-vs-room separation is preserved because the observation is sensor-scoped, not space-scoped. |
//! | `RoomInference` | ADR-297 MQTT/HA mapper | The `Space`-level fused inference: a `Person`/`Track` (or `Event`) whose `located_in` is `Container::Space`. `RoomInference.area_id` ↦ [`Space::area_id`]. |
//! | `WorldNode::Room { area_id, name, floor }` | `worldgraph` | [`Space`] (`area_id`, `name` retained; `floor` index ↦ the parent [`Floor::level`]). |
//! | `WorldNode::Zone { parent_room }` | `worldgraph` | [`Zone`] (`parent_room` ↦ [`Zone::parent`]). |
//! | `WorldNode::Sensor { device_id, modality }` | `worldgraph` | [`Sensor`] (`device_id` retained; placement ↦ its [`Container`]). |
//! | `WorldNode::PersonTrack { track_id }` | `worldgraph` | [`Track`] (`track_id` ↦ [`TrackId`]) optionally resolved to a [`Person`]. |
//! | `WorldNode::Event { event_type, at_unix_ms, located_in }` | `worldgraph` | [`Event`] (fields map 1:1; `located_in` ↦ [`Container`]). |
//! | `SemanticProvenance` | `worldgraph` / RuField `SemanticProvenance` | [`SemanticProvenance`] (`evidence`, `model_version`, `calibration_version`, `privacy_decision` map 1:1). |
//! | MQTT topic `.../<area>/<sensor>` payload | MQTT/HA surface | `area` ↦ [`Space::area_id`], `sensor` ↦ [`Sensor::device_id`]; the payload's belief becomes a `Person`/`Event` under the resolved `Container`. |
//! | REST `GET /spaces/{id}` / `/events` | REST surface | Direct projection of [`Space`] / [`Event`] JSON produced by this crate's canonical serializer. |
//! | RuField observation + `SemanticProvenance` | RuField | [`Observation`] carrying the same [`SemanticProvenance`] and [`EvidenceLevel`]. |
//! | Matter/HomeKit area model | Matter surface | Matter "area" ↦ [`Space`] via the HomeCore `area_id` (ADR-127) join key. |
//!
//! The HomeCore `area_id` linkage (ADR-127) remains the join key between a
//! canonical [`Space`] and external area registries. New surfaces (ROS 2,
//! OpenUSD, OPC UA) plug in as additional projections — the translation matrix
//! stays O(surfaces), not O(surfaces²).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod entity;
mod graph;
mod id;
mod provenance;

pub use entity::{
    Building, Container, Event, Floor, Located, Object, Observation, Person, Sensor, Site, Space,
    Track, Zone,
};
pub use graph::{OntologyError, WorldGraph, SCHEMA_VERSION};
pub use id::{
    BuildingId, EventId, FloorId, IdError, ObjectId, ObservationId, PersonId, SensorId, SiteId,
    SpaceId, TrackId, ZoneId, MAX_ID_LEN,
};
pub use provenance::{EvidenceLevel, SemanticProvenance};

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small but complete two-level hierarchy for reuse in tests.
    fn fixture() -> WorldGraph {
        let mut g = WorldGraph::new();
        g.add_site(Site {
            id: SiteId::new("home").unwrap(),
            name: "Home".into(),
        })
        .unwrap();
        g.add_building(Building {
            id: BuildingId::new("b1").unwrap(),
            parent: SiteId::new("home").unwrap(),
            name: "House".into(),
        })
        .unwrap();
        g.add_floor(Floor {
            id: FloorId::new("f1").unwrap(),
            parent: BuildingId::new("b1").unwrap(),
            level: 0,
            name: "Ground".into(),
        })
        .unwrap();
        g.add_space(Space {
            id: SpaceId::new("kitchen").unwrap(),
            parent: FloorId::new("f1").unwrap(),
            area_id: Some("area-42".into()),
            name: "Kitchen".into(),
        })
        .unwrap();
        g.add_zone(Zone {
            id: ZoneId::new("stove-zone").unwrap(),
            parent: SpaceId::new("kitchen").unwrap(),
            name: "Stove".into(),
        })
        .unwrap();
        g
    }

    fn prov() -> SemanticProvenance {
        SemanticProvenance::declared("fusion@1")
    }

    #[test]
    fn construction_builds_full_spine() {
        let g = fixture();
        assert_eq!(g.sites.len(), 1);
        assert_eq!(g.buildings.len(), 1);
        assert_eq!(g.floors.len(), 1);
        assert_eq!(g.spaces.len(), 1);
        assert_eq!(g.zones.len(), 1);
        assert_eq!(g.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn containment_resolution_walks_zone_to_space_to_floor() {
        let mut g = fixture();
        let in_zone = Container::Zone {
            id: ZoneId::new("stove-zone").unwrap(),
        };
        // A sensor placed in the stove zone resolves up to the kitchen space
        // and the ground floor.
        g.add_sensor(Sensor {
            id: SensorId::new("s1").unwrap(),
            device_id: "dev-aa".into(),
            located_in: in_zone.clone(),
            evidence_level: EvidenceLevel::L3,
            provenance: prov(),
        })
        .unwrap();

        let sensor = g.sensors.get(&SensorId::new("s1").unwrap()).unwrap();
        let container = sensor.located_in.clone();
        assert_eq!(g.zone_of(&container).unwrap().name, "Stove");
        assert_eq!(g.space_of(&container).unwrap().name, "Kitchen");
        assert_eq!(g.space_of(&container).unwrap().area_id.as_deref(), Some("area-42"));
        assert_eq!(g.floor_of(&container).unwrap().level, 0);

        // A person placed directly in the space has no zone but the same space.
        let in_space = Container::Space {
            id: SpaceId::new("kitchen").unwrap(),
        };
        assert!(g.zone_of(&in_space).is_none());
        assert_eq!(g.space_of(&in_space).unwrap().name, "Kitchen");
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let mut g = fixture();
        g.add_person(Person {
            id: PersonId::new("p1").unwrap(),
            located_in: Container::Space {
                id: SpaceId::new("kitchen").unwrap(),
            },
            evidence_level: EvidenceLevel::L2,
            provenance: prov(),
        })
        .unwrap();
        g.add_sensor(Sensor {
            id: SensorId::new("s1").unwrap(),
            device_id: "dev-aa".into(),
            located_in: Container::Zone {
                id: ZoneId::new("stove-zone").unwrap(),
            },
            evidence_level: EvidenceLevel::L4,
            provenance: prov(),
        })
        .unwrap();
        g.add_observation(Observation {
            id: ObservationId::new("o1").unwrap(),
            sensor: SensorId::new("s1").unwrap(),
            located_in: Container::Zone {
                id: ZoneId::new("stove-zone").unwrap(),
            },
            at_unix_ms: 1_700_000_000_000,
            evidence_level: EvidenceLevel::L3,
            provenance: prov(),
        })
        .unwrap();
        g.add_track(Track {
            id: TrackId::new("t1").unwrap(),
            person: Some(PersonId::new("p1").unwrap()),
            located_in: Container::Space {
                id: SpaceId::new("kitchen").unwrap(),
            },
            evidence_level: EvidenceLevel::L3,
            provenance: prov(),
        })
        .unwrap();
        g.add_event(Event {
            id: EventId::new("e1").unwrap(),
            event_type: "entry".into(),
            at_unix_ms: 1_700_000_000_500,
            located_in: Container::Space {
                id: SpaceId::new("kitchen").unwrap(),
            },
            evidence_level: EvidenceLevel::L5,
            provenance: prov(),
        })
        .unwrap();
        g.add_object(Object {
            id: ObjectId::new("obj1").unwrap(),
            located_in: Container::Space {
                id: SpaceId::new("kitchen").unwrap(),
            },
            class: "reflector".into(),
            evidence_level: EvidenceLevel::L1,
            provenance: prov(),
        })
        .unwrap();

        let json = serde_json::to_string_pretty(&g).unwrap();
        let back: WorldGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(g, back);

        // Canonical serialization uses stable string keys (typed ids) and a
        // versioned envelope.
        assert!(json.contains("\"schema_version\": 1"));
        assert!(json.contains("\"container\": \"space\""));
        assert!(json.contains("\"evidence_level\": \"L5\""));
    }

    #[test]
    fn invalid_parent_is_rejected() {
        let mut g = WorldGraph::new();
        // Building without its site.
        let err = g
            .add_building(Building {
                id: BuildingId::new("b1").unwrap(),
                parent: SiteId::new("ghost").unwrap(),
                name: "Orphan".into(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            OntologyError::MissingParent {
                parent_kind: "site",
                ..
            }
        ));

        // Leaf into a non-existent container.
        let mut g = fixture();
        let err = g
            .add_person(Person {
                id: PersonId::new("p1").unwrap(),
                located_in: Container::Zone {
                    id: ZoneId::new("nope").unwrap(),
                },
                evidence_level: EvidenceLevel::L0,
                provenance: prov(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            OntologyError::MissingContainer {
                container_kind: "zone",
                ..
            }
        ));

        // Observation referencing an unknown sensor.
        let err = g
            .add_observation(Observation {
                id: ObservationId::new("o1").unwrap(),
                sensor: SensorId::new("ghost-sensor").unwrap(),
                located_in: Container::Space {
                    id: SpaceId::new("kitchen").unwrap(),
                },
                at_unix_ms: 0,
                evidence_level: EvidenceLevel::L2,
                provenance: prov(),
            })
            .unwrap_err();
        assert!(matches!(
            err,
            OntologyError::MissingParent {
                parent_kind: "sensor",
                ..
            }
        ));
    }

    #[test]
    fn duplicate_id_is_rejected() {
        let mut g = fixture();
        let err = g
            .add_space(Space {
                id: SpaceId::new("kitchen").unwrap(),
                parent: FloorId::new("f1").unwrap(),
                area_id: None,
                name: "Dup".into(),
            })
            .unwrap_err();
        assert!(matches!(err, OntologyError::Duplicate { kind: "space", .. }));
    }
}
