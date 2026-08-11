//! Canonical entity types: the `Site ▸ Building ▸ Floor ▸ Space ▸ Zone`
//! containment spine and the leaf entities located within it (ADR-303 §1).
//!
//! Containment is expressed by a typed `parent` field on each spine node and a
//! [`Container`] reference on each leaf. This is the pure-hierarchy analogue of
//! the `worldgraph` `PartOf`/`LocatedIn` edges: a `Zone` is part of exactly one
//! `Space`, a `Space` on exactly one `Floor`, and so on. The [`WorldGraph`]
//! registry enforces those single-parent invariants.
//!
//! [`WorldGraph`]: crate::WorldGraph

use serde::{Deserialize, Serialize};

use crate::id::{
    BuildingId, EventId, FloorId, ObjectId, ObservationId, PersonId, SensorId, SiteId, SpaceId,
    TrackId, ZoneId,
};
use crate::provenance::{EvidenceLevel, SemanticProvenance};

/// The containment root. A site has no parent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Site {
    /// Stable id.
    pub id: SiteId,
    /// Human-readable name.
    pub name: String,
}

/// A building within a [`Site`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Building {
    /// Stable id.
    pub id: BuildingId,
    /// Containing site.
    pub parent: SiteId,
    /// Human-readable name.
    pub name: String,
}

/// A floor within a [`Building`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Floor {
    /// Stable id.
    pub id: FloorId,
    /// Containing building.
    pub parent: BuildingId,
    /// Storey index (ground = 0, basements negative).
    pub level: i16,
    /// Human-readable name.
    pub name: String,
}

/// A bounded interior space within a [`Floor`] — the ADR-294 "room" and the
/// HomeCore `area_id` join point (ADR-127).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Space {
    /// Stable id.
    pub id: SpaceId,
    /// Containing floor.
    pub parent: FloorId,
    /// HomeCore registry `area_id` — the external entity-linkage join key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area_id: Option<String>,
    /// Human-readable name.
    pub name: String,
}

/// A sub-region of a [`Space`] targeted for sensing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Zone {
    /// Stable id.
    pub id: ZoneId,
    /// Containing space.
    pub parent: SpaceId,
    /// Human-readable name.
    pub name: String,
}

/// Where a leaf entity is located: directly in a [`Space`] or in a [`Zone`].
/// A zone resolves upward to its containing space via the registry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "container", rename_all = "snake_case")]
pub enum Container {
    /// Located directly in a space.
    Space {
        /// The space id.
        id: SpaceId,
    },
    /// Located in a zone (which is itself part of a space).
    Zone {
        /// The zone id.
        id: ZoneId,
    },
}

/// A physical sensing device placement — the entity ADR-302 authenticates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sensor {
    /// Stable id.
    pub id: SensorId,
    /// ADR-302 authenticated device identity (HomeCore `device_id`).
    pub device_id: String,
    /// Where the sensor is placed.
    pub located_in: Container,
    /// Exactly one evidence level travels with this fact.
    pub evidence_level: EvidenceLevel,
    /// Mandatory provenance.
    pub provenance: SemanticProvenance,
}

/// A tracked or known person.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Person {
    /// Stable id.
    pub id: PersonId,
    /// Where the person currently is.
    pub located_in: Container,
    /// Exactly one evidence level travels with this fact.
    pub evidence_level: EvidenceLevel,
    /// Mandatory provenance.
    pub provenance: SemanticProvenance,
}

/// A persistent physical object / static anchor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Object {
    /// Stable id.
    pub id: ObjectId,
    /// Where the object is.
    pub located_in: Container,
    /// Classification tag (e.g. `"furniture"`, `"reflector"`).
    pub class: String,
    /// Exactly one evidence level travels with this fact.
    pub evidence_level: EvidenceLevel,
    /// Mandatory provenance.
    pub provenance: SemanticProvenance,
}

/// A calibrated observation produced from an authenticated frame (ADR-298).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    /// Stable id.
    pub id: ObservationId,
    /// The sensor that produced it.
    pub sensor: SensorId,
    /// Where it was observed.
    pub located_in: Container,
    /// Producer-supplied capture timestamp (Unix ms). Injected, never sampled
    /// from a clock inside this crate.
    pub at_unix_ms: i64,
    /// Exactly one evidence level travels with this fact.
    pub evidence_level: EvidenceLevel,
    /// Mandatory provenance.
    pub provenance: SemanticProvenance,
}

/// A persistent track (ADR-304), optionally resolved to a [`Person`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Track {
    /// Stable id.
    pub id: TrackId,
    /// Resolved person identity, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub person: Option<PersonId>,
    /// Where the track currently is.
    pub located_in: Container,
    /// Exactly one evidence level travels with this fact.
    pub evidence_level: EvidenceLevel,
    /// Mandatory provenance.
    pub provenance: SemanticProvenance,
}

/// A discrete governed event (ADR-315 certified, ADR-316 witnessed).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    /// Stable id.
    pub id: EventId,
    /// Event type tag (e.g. `"fall"`, `"entry"`).
    pub event_type: String,
    /// Producer-supplied event timestamp (Unix ms). Injected.
    pub at_unix_ms: i64,
    /// Where the event occurred.
    pub located_in: Container,
    /// Exactly one evidence level travels with this fact.
    pub evidence_level: EvidenceLevel,
    /// Mandatory provenance.
    pub provenance: SemanticProvenance,
}

/// Shared accessor: the [`Container`] a leaf entity is located in.
pub trait Located {
    /// Borrow this entity's container.
    fn container(&self) -> &Container;
}

macro_rules! impl_located {
    ($($ty:ty),+ $(,)?) => {
        $(impl Located for $ty {
            fn container(&self) -> &Container {
                &self.located_in
            }
        })+
    };
}

impl_located!(Sensor, Person, Object, Observation, Track, Event);
