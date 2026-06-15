//! ADR-139 §2.2–2.5 — graph container, provenance, privacy rollup, queries.

use std::collections::HashMap;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use wifi_densepose_geo::types::GeoRegistration;

use crate::error::WorldGraphError;
use crate::model::{SemanticProvenance, WorldEdge, WorldId, WorldNode};

/// Current persisted schema version (ADR-136 §2.1 reserved-flag pattern).
pub const SCHEMA_VERSION: u16 = 1;

/// The typed environmental digital twin (ADR-139). Wraps a petgraph
/// `StableDiGraph` and exposes a domain API; stable `WorldId → NodeIndex`
/// mapping survives node removal.
#[derive(Debug)]
pub struct WorldGraph {
    inner: StableDiGraph<WorldNode, WorldEdge>,
    index: HashMap<WorldId, NodeIndex>,
    registration: GeoRegistration,
    next_id: u64,
    schema_version: u16,
}

/// Serializable snapshot of a [`WorldGraph`] for RVF/JSON persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldGraphSnapshot {
    schema_version: u16,
    registration: GeoRegistration,
    next_id: u64,
    nodes: Vec<WorldNode>,
    /// Edges as (from_id, to_id, edge).
    edges: Vec<(WorldId, WorldId, WorldEdge)>,
}

/// Result of a privacy-impact rollup (ADR-139 §2.4).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PrivacyRollup {
    /// Active mode name.
    pub mode: String,
    /// Nodes that become unobservable under this mode.
    pub suppressed_nodes: Vec<WorldId>,
    /// (sensor, node) pairs newly denied.
    pub denied_pairs: Vec<(WorldId, WorldId)>,
    /// Count of still-allowed (sensor, node) pairs.
    pub allowed_pairs: usize,
}

impl WorldGraph {
    /// Create an empty graph registered to an installation origin.
    #[must_use]
    pub fn new(registration: GeoRegistration) -> Self {
        Self {
            inner: StableDiGraph::new(),
            index: HashMap::new(),
            registration,
            next_id: 1,
            schema_version: SCHEMA_VERSION,
        }
    }

    /// Installation geo-registration (ADR-044).
    #[must_use]
    pub fn registration(&self) -> &GeoRegistration {
        &self.registration
    }

    /// Number of live nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Insert or replace a node, returning its stable `WorldId`. If the node's
    /// embedded id is `UNASSIGNED`, a fresh id is allocated; if it names an
    /// existing id, that node's weight is replaced in place (upsert).
    pub fn upsert_node(&mut self, mut node: WorldNode) -> WorldId {
        let id = if node.id().is_unassigned() {
            let fresh = WorldId(self.next_id);
            self.next_id += 1;
            node.set_id(fresh);
            fresh
        } else {
            self.next_id = self.next_id.max(node.id().0 + 1);
            node.id()
        };

        if let Some(&idx) = self.index.get(&id) {
            self.inner[idx] = node;
        } else {
            let idx = self.inner.add_node(node);
            self.index.insert(id, idx);
        }
        id
    }

    /// Add a typed edge between two known nodes.
    ///
    /// # Errors
    /// [`WorldGraphError::UnknownNode`] if either endpoint is unknown.
    pub fn add_edge(
        &mut self,
        from: WorldId,
        to: WorldId,
        edge: WorldEdge,
    ) -> Result<(), WorldGraphError> {
        let f = *self.index.get(&from).ok_or(WorldGraphError::UnknownNode(from))?;
        let t = *self.index.get(&to).ok_or(WorldGraphError::UnknownNode(to))?;
        self.inner.add_edge(f, t, edge);
        Ok(())
    }

    /// Borrow a node by id.
    #[must_use]
    pub fn node(&self, id: WorldId) -> Option<&WorldNode> {
        self.index.get(&id).map(|&idx| &self.inner[idx])
    }

    /// Remove a node and its incident edges (e.g. a person leaves).
    pub fn remove_node(&mut self, id: WorldId) -> Option<WorldNode> {
        let idx = self.index.remove(&id)?;
        self.inner.remove_node(idx)
    }

    /// Outgoing neighbours of a node with the connecting edge.
    pub fn neighbors(&self, id: WorldId) -> Vec<(WorldId, WorldEdge)> {
        let Some(&idx) = self.index.get(&id) else {
            return Vec::new();
        };
        self.inner
            .edges_directed(idx, Direction::Outgoing)
            .map(|e| (self.inner[e.target()].id(), e.weight().clone()))
            .collect()
    }

    /// Resolve a HomeCore `area_id` to its Room node (entity linkage, ADR-127).
    #[must_use]
    pub fn room_for_area(&self, area_id: &str) -> Option<WorldId> {
        self.inner.node_weights().find_map(|n| match n {
            WorldNode::Room { id, area_id: Some(a), .. } if a == area_id => Some(*id),
            _ => None,
        })
    }

    // ---- ADR-139 §2.5 query API (v1) ----

    /// Observability chain: which nodes a sensor currently `observes`.
    #[must_use]
    pub fn observed_by(&self, sensor: WorldId) -> Vec<WorldId> {
        self.neighbors(sensor)
            .into_iter()
            .filter(|(_, e)| matches!(e, WorldEdge::Observes { .. }))
            .map(|(id, _)| id)
            .collect()
    }

    /// Location query: contents of a room/zone (incoming `located_in` edges).
    #[must_use]
    pub fn contents_of(&self, container: WorldId) -> Vec<WorldId> {
        let Some(&idx) = self.index.get(&container) else {
            return Vec::new();
        };
        self.inner
            .edges_directed(idx, Direction::Incoming)
            .filter(|e| matches!(e.weight(), WorldEdge::LocatedIn { .. }))
            .map(|e| self.inner[e.source()].id())
            .collect()
    }

    /// Append-with-provenance: insert a `SemanticState` and wire `DerivedFrom`
    /// edges to its evidence sources (ADR-139 §2.3). Sources unknown to the
    /// graph are skipped (evidence may be raw frames not modelled as nodes).
    pub fn add_semantic_state(
        &mut self,
        statement: String,
        confidence: f32,
        valid_from_unix_ms: i64,
        provenance: SemanticProvenance,
        evidence_sources: &[WorldId],
    ) -> WorldId {
        let evidence_handles = provenance.evidence.clone();
        let id = self.upsert_node(WorldNode::SemanticState {
            id: WorldId::UNASSIGNED,
            statement,
            confidence,
            provenance,
            valid_from_unix_ms,
        });
        for (src, handle) in evidence_sources.iter().zip(
            evidence_handles
                .iter()
                .cloned()
                .chain(std::iter::repeat(String::new())),
        ) {
            let _ = self.add_edge(id, *src, WorldEdge::DerivedFrom { evidence: handle });
        }
        id
    }

    /// Retention: evict the oldest `SemanticState` nodes (with their incident
    /// edges) until at most `max_states` remain. Returns the evicted ids,
    /// oldest first.
    ///
    /// The live loop appends one belief per cycle (`StreamingEngine::
    /// process_cycle`), which at 20 Hz is ~1.7M nodes/day — unbounded without
    /// this. The WorldGraph holds *current* beliefs; durable history belongs to
    /// the recorder (`homecore-recorder`), so evicting old beliefs loses no
    /// audit data.
    ///
    /// Deterministic: eviction order is ascending `(valid_from_unix_ms, id)`,
    /// so replaying the same cycle sequence prunes identically. Only
    /// `SemanticState` nodes are eligible — rooms, zones, sensors, anchors,
    /// person tracks, and events are never evicted by this method.
    pub fn prune_semantic_states(&mut self, max_states: usize) -> Vec<WorldId> {
        let mut states: Vec<(i64, u64)> = self
            .inner
            .node_weights()
            .filter_map(|n| match n {
                WorldNode::SemanticState { id, valid_from_unix_ms, .. } => {
                    Some((*valid_from_unix_ms, id.0))
                }
                _ => None,
            })
            .collect();
        if states.len() <= max_states {
            return Vec::new();
        }
        states.sort_unstable();
        let n_evict = states.len() - max_states;
        states.truncate(n_evict);
        states
            .into_iter()
            .map(|(_, raw)| {
                let id = WorldId(raw);
                self.remove_node(id);
                id
            })
            .collect()
    }

    /// Record a contradiction between two still-live beliefs (ADR-139 §2.3).
    /// Neither node is deleted — the disagreement stays queryable.
    ///
    /// # Errors
    /// [`WorldGraphError::UnknownNode`] if either node is unknown.
    pub fn add_contradiction(
        &mut self,
        a: WorldId,
        b: WorldId,
        magnitude: f32,
        flag: String,
    ) -> Result<(), WorldGraphError> {
        self.add_edge(a, b, WorldEdge::Contradicts { magnitude, flag })
    }

    /// Recompute `PrivacyLimitedBy` edges for the active mode (ADR-139 §2.4).
    ///
    /// `policy(modality_kind, node_kind) -> allowed` decides, for each existing
    /// `Observes` edge, whether the sensor may still observe the target under
    /// `mode`. A matching `PrivacyLimitedBy` edge is appended recording the
    /// decision; denied pairs are rolled up.
    pub fn apply_privacy_mode<F>(&mut self, mode: &str, action: &str, policy: F) -> PrivacyRollup
    where
        F: Fn(&str, &str) -> bool,
    {
        // Collect (sensor, target, allowed) from current Observes edges.
        let mut decisions: Vec<(WorldId, WorldId, bool)> = Vec::new();
        for e in self.inner.edge_references() {
            if matches!(e.weight(), WorldEdge::Observes { .. }) {
                let sensor = &self.inner[e.source()];
                let target = &self.inner[e.target()];
                let allowed = policy(sensor.kind(), target.kind());
                decisions.push((sensor.id(), target.id(), allowed));
            }
        }

        let mut denied_pairs = Vec::new();
        let mut suppressed = Vec::new();
        let mut allowed_pairs = 0usize;
        for (sensor, target, allowed) in &decisions {
            let _ = self.add_edge(
                *sensor,
                *target,
                WorldEdge::PrivacyLimitedBy {
                    mode: mode.to_string(),
                    action: action.to_string(),
                    allowed: *allowed,
                },
            );
            if *allowed {
                allowed_pairs += 1;
            } else {
                denied_pairs.push((*sensor, *target));
                if !suppressed.contains(target) {
                    suppressed.push(*target);
                }
            }
        }

        PrivacyRollup {
            mode: mode.to_string(),
            suppressed_nodes: suppressed,
            denied_pairs,
            allowed_pairs,
        }
    }

    // ---- Persistence (RVF/JSON) ----

    /// Snapshot the graph for persistence.
    #[must_use]
    pub fn snapshot(&self) -> WorldGraphSnapshot {
        let nodes: Vec<WorldNode> = self.inner.node_weights().cloned().collect();
        let edges: Vec<(WorldId, WorldId, WorldEdge)> = self
            .inner
            .edge_references()
            .map(|e| {
                (
                    self.inner[e.source()].id(),
                    self.inner[e.target()].id(),
                    e.weight().clone(),
                )
            })
            .collect();
        WorldGraphSnapshot {
            schema_version: self.schema_version,
            registration: self.registration.clone(),
            next_id: self.next_id,
            nodes,
            edges,
        }
    }

    /// Serialize to deterministic JSON bytes (RVF payload).
    ///
    /// # Errors
    /// [`WorldGraphError::Serde`] on serialisation failure.
    pub fn to_json(&self) -> Result<Vec<u8>, WorldGraphError> {
        Ok(serde_json::to_vec(&self.snapshot())?)
    }

    /// Reconstruct a graph from a snapshot's JSON bytes.
    ///
    /// # Errors
    /// [`WorldGraphError::Serde`] on parse failure.
    pub fn from_json(bytes: &[u8]) -> Result<Self, WorldGraphError> {
        let snap: WorldGraphSnapshot = serde_json::from_slice(bytes)?;
        let mut g = Self::new(snap.registration);
        g.schema_version = snap.schema_version;
        for node in snap.nodes {
            g.upsert_node(node);
        }
        for (from, to, edge) in snap.edges {
            g.add_edge(from, to, edge)?;
        }
        g.next_id = snap.next_id;
        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{EnuPoint, SensorModality, WorldEdge, ZoneBoundsEnu};

    fn enu(e: f64, n: f64) -> EnuPoint {
        EnuPoint { east_m: e, north_m: n, up_m: 0.0 }
    }

    fn living_room() -> WorldNode {
        WorldNode::Room {
            id: WorldId::UNASSIGNED,
            area_id: Some("living_room".into()),
            name: "Living Room".into(),
            bounds_enu: ZoneBoundsEnu::Rectangle { min_e: 0.0, min_n: 0.0, max_e: 5.0, max_n: 4.0 },
            floor: 0,
        }
    }

    #[test]
    fn upsert_allocates_and_replaces() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let id = g.upsert_node(living_room());
        assert!(!id.is_unassigned());
        assert_eq!(g.node_count(), 1);
        // Upsert same id with new name → replace in place, count unchanged.
        g.upsert_node(WorldNode::Room {
            id,
            area_id: Some("living_room".into()),
            name: "Lounge".into(),
            bounds_enu: ZoneBoundsEnu::Rectangle { min_e: 0.0, min_n: 0.0, max_e: 5.0, max_n: 4.0 },
            floor: 0,
        });
        assert_eq!(g.node_count(), 1);
        assert!(matches!(g.node(id), Some(WorldNode::Room { name, .. }) if name == "Lounge"));
    }

    #[test]
    fn area_linkage_and_observability() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let room = g.upsert_node(living_room());
        let sensor = g.upsert_node(WorldNode::Sensor {
            id: WorldId::UNASSIGNED,
            device_id: "esp32-com9".into(),
            position: enu(1.0, 1.0),
            modality: SensorModality::WifiCsi,
        });
        g.add_edge(sensor, room, WorldEdge::Observes { quality: 0.9, last_seen_unix_ms: 1 })
            .unwrap();

        assert_eq!(g.room_for_area("living_room"), Some(room));
        assert_eq!(g.observed_by(sensor), vec![room]);
    }

    #[test]
    fn add_edge_unknown_endpoint_errors() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let room = g.upsert_node(living_room());
        let err = g.add_edge(room, WorldId(999), WorldEdge::Observes { quality: 1.0, last_seen_unix_ms: 0 });
        assert!(matches!(err, Err(WorldGraphError::UnknownNode(WorldId(999)))));
    }

    #[test]
    fn location_query_contents_of() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let room = g.upsert_node(living_room());
        let person = g.upsert_node(WorldNode::PersonTrack {
            id: WorldId::UNASSIGNED,
            track_id: 7,
            last_position: enu(2.0, 2.0),
            reid_embedding_ref: None,
        });
        g.add_edge(person, room, WorldEdge::LocatedIn { since_unix_ms: 100 }).unwrap();
        assert_eq!(g.contents_of(room), vec![person]);
    }

    #[test]
    fn semantic_state_provenance_and_contradiction() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let event = g.upsert_node(WorldNode::Event {
            id: WorldId::UNASSIGNED,
            event_type: "motion".into(),
            at_unix_ms: 10,
            located_in: None,
        });
        let prov = SemanticProvenance {
            evidence: vec!["ev:abc".into()],
            model_version: "rfenc-1.0".into(),
            calibration_version: "cal:uuid".into(),
            privacy_decision: "PrivateHome/Allow".into(),
        };
        let s1 = g.add_semantic_state("present".into(), 0.9, 11, prov.clone(), &[event]);
        // DerivedFrom edge to the evidence event exists.
        assert!(g.neighbors(s1).iter().any(|(to, e)| *to == event
            && matches!(e, WorldEdge::DerivedFrom { .. })));

        let s2 = g.add_semantic_state("absent".into(), 0.6, 12, prov, &[event]);
        g.add_contradiction(s1, s2, 0.3, "flag:ts".into()).unwrap();
        // Both beliefs retained; contradiction queryable.
        assert!(g.node(s1).is_some() && g.node(s2).is_some());
        assert!(g.neighbors(s1).iter().any(|(_, e)| matches!(e, WorldEdge::Contradicts { .. })));
    }

    #[test]
    fn prune_semantic_states_evicts_oldest_only() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let room = g.upsert_node(living_room());
        let prov = SemanticProvenance {
            evidence: vec!["ev:abc".into()],
            model_version: "rfenc-1.0".into(),
            calibration_version: "cal:uuid".into(),
            privacy_decision: "PrivateHome/Allow".into(),
        };
        let ids: Vec<WorldId> = (0..10)
            .map(|t| g.add_semantic_state(format!("s{t}"), 0.9, t, prov.clone(), &[room]))
            .collect();
        assert_eq!(g.node_count(), 11); // room + 10 beliefs

        let evicted = g.prune_semantic_states(3);
        // Oldest 7 evicted, in ascending timestamp order.
        assert_eq!(evicted, ids[..7].to_vec());
        assert_eq!(g.node_count(), 4); // room + 3 newest beliefs
        for kept in &ids[7..] {
            assert!(g.node(*kept).is_some());
        }
        // The room (structural node) is never eligible for eviction.
        assert!(g.node(room).is_some());
        // Below the cap, pruning is a no-op.
        assert!(g.prune_semantic_states(3).is_empty());
    }

    #[test]
    fn prune_is_deterministic_for_equal_timestamps() {
        let prov = SemanticProvenance {
            evidence: vec![],
            model_version: "m".into(),
            calibration_version: "c".into(),
            privacy_decision: "p".into(),
        };
        let build = || {
            let mut g = WorldGraph::new(GeoRegistration::default());
            let room = g.upsert_node(living_room());
            for _ in 0..6 {
                // Identical timestamps: tie-break must fall back to id order.
                g.add_semantic_state("s".into(), 0.5, 100, prov.clone(), &[room]);
            }
            g
        };
        let mut g1 = build();
        let mut g2 = build();
        assert_eq!(g1.prune_semantic_states(2), g2.prune_semantic_states(2));
    }

    #[test]
    fn privacy_rollup_suppresses_person_tracks() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let room = g.upsert_node(living_room());
        let person = g.upsert_node(WorldNode::PersonTrack {
            id: WorldId::UNASSIGNED,
            track_id: 1,
            last_position: enu(1.0, 1.0),
            reid_embedding_ref: None,
        });
        let sensor = g.upsert_node(WorldNode::Sensor {
            id: WorldId::UNASSIGNED,
            device_id: "s".into(),
            position: enu(0.0, 0.0),
            modality: SensorModality::WifiCsi,
        });
        g.add_edge(sensor, room, WorldEdge::Observes { quality: 1.0, last_seen_unix_ms: 0 }).unwrap();
        g.add_edge(sensor, person, WorldEdge::Observes { quality: 1.0, last_seen_unix_ms: 0 }).unwrap();

        // StrictNoIdentity: rooms observable, person_tracks suppressed.
        let rollup = g.apply_privacy_mode("StrictNoIdentity", "SuppressIdentity", |_modality, node_kind| {
            node_kind != "person_track"
        });
        assert_eq!(rollup.allowed_pairs, 1);
        assert_eq!(rollup.denied_pairs, vec![(sensor, person)]);
        assert_eq!(rollup.suppressed_nodes, vec![person]);
    }

    #[test]
    fn json_roundtrip_preserves_nodes_and_edges() {
        let mut g = WorldGraph::new(GeoRegistration::default());
        let room = g.upsert_node(living_room());
        let sensor = g.upsert_node(WorldNode::Sensor {
            id: WorldId::UNASSIGNED,
            device_id: "s".into(),
            position: enu(0.0, 0.0),
            modality: SensorModality::WifiCsi,
        });
        g.add_edge(sensor, room, WorldEdge::Observes { quality: 0.8, last_seen_unix_ms: 5 }).unwrap();

        let bytes = g.to_json().unwrap();
        let g2 = WorldGraph::from_json(&bytes).unwrap();
        assert_eq!(g2.node_count(), 2);
        assert_eq!(g2.room_for_area("living_room"), Some(room));
        assert_eq!(g2.observed_by(sensor), vec![room]);
        // Deterministic: re-serialising the reconstructed graph matches.
        assert_eq!(g2.to_json().unwrap(), bytes);
    }
}
