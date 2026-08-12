//! [`WorldGraph`] — the canonical registry that holds the containment hierarchy
//! and resolves an entity's containing [`Space`]/[`Zone`] (ADR-306 §1).
//!
//! The registry is the sole insertion boundary: every `add_*` method rejects a
//! duplicate id and a dangling parent/container, so the single-parent
//! containment invariants of ADR-306 hold by construction. The graph is a pure
//! data structure — no I/O, no async, deterministic `BTreeMap` ordering for a
//! stable canonical serialization.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::entity::{
    Building, Container, Event, Floor, Object, Observation, Person, Sensor, Site, Space, Track, Zone,
};
use crate::id::{
    BuildingId, EventId, FloorId, IdError, ObjectId, ObservationId, PersonId, SensorId, SiteId,
    SpaceId, TrackId, ZoneId,
};

/// Errors returned when mutating the [`WorldGraph`].
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum OntologyError {
    /// A raw id failed boundary validation.
    #[error("invalid identifier: {0}")]
    Id(#[from] IdError),
    /// An entity with this id already exists.
    #[error("duplicate {kind} id: {id}")]
    Duplicate {
        /// Entity kind tag.
        kind: &'static str,
        /// The conflicting id.
        id: String,
    },
    /// The referenced parent entity does not exist in the registry.
    #[error("missing {parent_kind} parent '{parent_id}' for {child_kind} '{child_id}'")]
    MissingParent {
        /// Kind of the missing parent.
        parent_kind: &'static str,
        /// Id of the missing parent.
        parent_id: String,
        /// Kind of the child being inserted.
        child_kind: &'static str,
        /// Id of the child being inserted.
        child_id: String,
    },
    /// The [`Container`] a leaf references does not exist.
    #[error("missing {container_kind} container '{container_id}' for {child_kind} '{child_id}'")]
    MissingContainer {
        /// `space` or `zone`.
        container_kind: &'static str,
        /// The missing container id.
        container_id: String,
        /// Kind of the leaf being inserted.
        child_kind: &'static str,
        /// Id of the leaf being inserted.
        child_id: String,
    },
}

/// The canonical world registry: the containment spine plus all leaf entities.
/// Serializes to one versioned canonical JSON document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldGraph {
    /// Canonical schema version for the serialized form.
    pub schema_version: u32,
    /// Sites, keyed by id.
    pub sites: BTreeMap<SiteId, Site>,
    /// Buildings, keyed by id.
    pub buildings: BTreeMap<BuildingId, Building>,
    /// Floors, keyed by id.
    pub floors: BTreeMap<FloorId, Floor>,
    /// Spaces, keyed by id.
    pub spaces: BTreeMap<SpaceId, Space>,
    /// Zones, keyed by id.
    pub zones: BTreeMap<ZoneId, Zone>,
    /// Sensors, keyed by id.
    pub sensors: BTreeMap<SensorId, Sensor>,
    /// Persons, keyed by id.
    pub persons: BTreeMap<PersonId, Person>,
    /// Objects, keyed by id.
    pub objects: BTreeMap<ObjectId, Object>,
    /// Observations, keyed by id.
    pub observations: BTreeMap<ObservationId, Observation>,
    /// Tracks, keyed by id.
    pub tracks: BTreeMap<TrackId, Track>,
    /// Events, keyed by id.
    pub events: BTreeMap<EventId, Event>,
}

/// The canonical serialization version this build emits.
pub const SCHEMA_VERSION: u32 = 1;

impl Default for WorldGraph {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            sites: BTreeMap::new(),
            buildings: BTreeMap::new(),
            floors: BTreeMap::new(),
            spaces: BTreeMap::new(),
            zones: BTreeMap::new(),
            sensors: BTreeMap::new(),
            persons: BTreeMap::new(),
            objects: BTreeMap::new(),
            observations: BTreeMap::new(),
            tracks: BTreeMap::new(),
            events: BTreeMap::new(),
        }
    }
}

impl WorldGraph {
    /// A fresh, empty registry at the current [`SCHEMA_VERSION`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ---- containment spine --------------------------------------------------

    /// Insert a site (spine root; no parent to validate).
    pub fn add_site(&mut self, site: Site) -> Result<(), OntologyError> {
        if self.sites.contains_key(&site.id) {
            return Err(OntologyError::Duplicate {
                kind: "site",
                id: site.id.to_string(),
            });
        }
        self.sites.insert(site.id.clone(), site);
        Ok(())
    }

    /// Insert a building; its parent site must already exist.
    pub fn add_building(&mut self, building: Building) -> Result<(), OntologyError> {
        if self.buildings.contains_key(&building.id) {
            return Err(OntologyError::Duplicate {
                kind: "building",
                id: building.id.to_string(),
            });
        }
        if !self.sites.contains_key(&building.parent) {
            return Err(OntologyError::MissingParent {
                parent_kind: "site",
                parent_id: building.parent.to_string(),
                child_kind: "building",
                child_id: building.id.to_string(),
            });
        }
        self.buildings.insert(building.id.clone(), building);
        Ok(())
    }

    /// Insert a floor; its parent building must already exist.
    pub fn add_floor(&mut self, floor: Floor) -> Result<(), OntologyError> {
        if self.floors.contains_key(&floor.id) {
            return Err(OntologyError::Duplicate {
                kind: "floor",
                id: floor.id.to_string(),
            });
        }
        if !self.buildings.contains_key(&floor.parent) {
            return Err(OntologyError::MissingParent {
                parent_kind: "building",
                parent_id: floor.parent.to_string(),
                child_kind: "floor",
                child_id: floor.id.to_string(),
            });
        }
        self.floors.insert(floor.id.clone(), floor);
        Ok(())
    }

    /// Insert a space; its parent floor must already exist.
    pub fn add_space(&mut self, space: Space) -> Result<(), OntologyError> {
        if self.spaces.contains_key(&space.id) {
            return Err(OntologyError::Duplicate {
                kind: "space",
                id: space.id.to_string(),
            });
        }
        if !self.floors.contains_key(&space.parent) {
            return Err(OntologyError::MissingParent {
                parent_kind: "floor",
                parent_id: space.parent.to_string(),
                child_kind: "space",
                child_id: space.id.to_string(),
            });
        }
        self.spaces.insert(space.id.clone(), space);
        Ok(())
    }

    /// Insert a zone; its parent space must already exist.
    pub fn add_zone(&mut self, zone: Zone) -> Result<(), OntologyError> {
        if self.zones.contains_key(&zone.id) {
            return Err(OntologyError::Duplicate {
                kind: "zone",
                id: zone.id.to_string(),
            });
        }
        if !self.spaces.contains_key(&zone.parent) {
            return Err(OntologyError::MissingParent {
                parent_kind: "space",
                parent_id: zone.parent.to_string(),
                child_kind: "zone",
                child_id: zone.id.to_string(),
            });
        }
        self.zones.insert(zone.id.clone(), zone);
        Ok(())
    }

    // ---- leaf entities ------------------------------------------------------

    /// Validate that a [`Container`] resolves to an existing space or zone.
    fn check_container(
        &self,
        container: &Container,
        child_kind: &'static str,
        child_id: String,
    ) -> Result<(), OntologyError> {
        match container {
            Container::Space { id } => {
                if self.spaces.contains_key(id) {
                    Ok(())
                } else {
                    Err(OntologyError::MissingContainer {
                        container_kind: "space",
                        container_id: id.to_string(),
                        child_kind,
                        child_id,
                    })
                }
            }
            Container::Zone { id } => {
                if self.zones.contains_key(id) {
                    Ok(())
                } else {
                    Err(OntologyError::MissingContainer {
                        container_kind: "zone",
                        container_id: id.to_string(),
                        child_kind,
                        child_id,
                    })
                }
            }
        }
    }

    /// Insert a sensor; its container must already exist.
    pub fn add_sensor(&mut self, sensor: Sensor) -> Result<(), OntologyError> {
        if self.sensors.contains_key(&sensor.id) {
            return Err(OntologyError::Duplicate {
                kind: "sensor",
                id: sensor.id.to_string(),
            });
        }
        self.check_container(&sensor.located_in, "sensor", sensor.id.to_string())?;
        self.sensors.insert(sensor.id.clone(), sensor);
        Ok(())
    }

    /// Insert a person; its container must already exist.
    pub fn add_person(&mut self, person: Person) -> Result<(), OntologyError> {
        if self.persons.contains_key(&person.id) {
            return Err(OntologyError::Duplicate {
                kind: "person",
                id: person.id.to_string(),
            });
        }
        self.check_container(&person.located_in, "person", person.id.to_string())?;
        self.persons.insert(person.id.clone(), person);
        Ok(())
    }

    /// Insert an object; its container must already exist.
    pub fn add_object(&mut self, object: Object) -> Result<(), OntologyError> {
        if self.objects.contains_key(&object.id) {
            return Err(OntologyError::Duplicate {
                kind: "object",
                id: object.id.to_string(),
            });
        }
        self.check_container(&object.located_in, "object", object.id.to_string())?;
        self.objects.insert(object.id.clone(), object);
        Ok(())
    }

    /// Insert an observation; its sensor and container must already exist.
    pub fn add_observation(&mut self, obs: Observation) -> Result<(), OntologyError> {
        if self.observations.contains_key(&obs.id) {
            return Err(OntologyError::Duplicate {
                kind: "observation",
                id: obs.id.to_string(),
            });
        }
        if !self.sensors.contains_key(&obs.sensor) {
            return Err(OntologyError::MissingParent {
                parent_kind: "sensor",
                parent_id: obs.sensor.to_string(),
                child_kind: "observation",
                child_id: obs.id.to_string(),
            });
        }
        self.check_container(&obs.located_in, "observation", obs.id.to_string())?;
        self.observations.insert(obs.id.clone(), obs);
        Ok(())
    }

    /// Insert a track; its container (and resolved person, if any) must exist.
    pub fn add_track(&mut self, track: Track) -> Result<(), OntologyError> {
        if self.tracks.contains_key(&track.id) {
            return Err(OntologyError::Duplicate {
                kind: "track",
                id: track.id.to_string(),
            });
        }
        if let Some(person) = &track.person {
            if !self.persons.contains_key(person) {
                return Err(OntologyError::MissingParent {
                    parent_kind: "person",
                    parent_id: person.to_string(),
                    child_kind: "track",
                    child_id: track.id.to_string(),
                });
            }
        }
        self.check_container(&track.located_in, "track", track.id.to_string())?;
        self.tracks.insert(track.id.clone(), track);
        Ok(())
    }

    /// Insert an event; its container must already exist.
    pub fn add_event(&mut self, event: Event) -> Result<(), OntologyError> {
        if self.events.contains_key(&event.id) {
            return Err(OntologyError::Duplicate {
                kind: "event",
                id: event.id.to_string(),
            });
        }
        self.check_container(&event.located_in, "event", event.id.to_string())?;
        self.events.insert(event.id.clone(), event);
        Ok(())
    }

    // ---- containment resolution ---------------------------------------------

    /// Resolve the [`Zone`] a container references, if it is a zone container.
    /// A space container has no zone.
    #[must_use]
    pub fn zone_of(&self, container: &Container) -> Option<&Zone> {
        match container {
            Container::Zone { id } => self.zones.get(id),
            Container::Space { .. } => None,
        }
    }

    /// Resolve the containing [`Space`] for any container, walking a zone up to
    /// its parent space. Returns `None` if the container (or a zone's parent
    /// space) is not registered.
    #[must_use]
    pub fn space_of(&self, container: &Container) -> Option<&Space> {
        match container {
            Container::Space { id } => self.spaces.get(id),
            Container::Zone { id } => {
                let zone = self.zones.get(id)?;
                self.spaces.get(&zone.parent)
            }
        }
    }

    /// Resolve the containing [`Floor`] for any container.
    #[must_use]
    pub fn floor_of(&self, container: &Container) -> Option<&Floor> {
        let space = self.space_of(container)?;
        self.floors.get(&space.parent)
    }
}
