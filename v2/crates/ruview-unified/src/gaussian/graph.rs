//! Task-gated scene graph (ADR-275 §5) — sparse symbolic relations over the
//! continuous Gaussian field, activated *only* as far as the current task
//! requires (bounded active memory, the JITOMA lesson: stop trying to
//! remember everything at once).

use std::collections::{BTreeSet, HashMap, VecDeque};

use serde::{Deserialize, Serialize};

/// Kind of entity a scene node represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EntityKind {
    /// A physical object (furniture, appliance).
    Object,
    /// A room / zone.
    Room,
    /// A person *class* (never an identity — identity inference is gated by
    /// the ADR-277 policy engine, not stored in the scene graph).
    PersonClass,
    /// A sensing device.
    Device,
    /// A discrete event (channel anomaly, fall alert, …).
    Event,
}

/// Relation kinds on scene edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationKind {
    /// Spatial containment (room contains object).
    Contains,
    /// Spatial adjacency.
    Near,
    /// Causal attribution (event caused by object/person-class).
    CausedBy,
    /// Sensing coverage (device observes room/object).
    ObservedBy,
}

/// A node in the scene graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneNode {
    /// Stable identifier.
    pub id: String,
    /// Entity kind.
    pub kind: EntityKind,
    /// Indices of the Gaussians in the map that ground this node.
    pub gaussian_refs: Vec<usize>,
}

/// A directed, typed edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Relation.
    pub relation: RelationKind,
}

/// The sparse relational layer over the Gaussian field.
#[derive(Debug, Default, Clone)]
pub struct SceneGraph {
    nodes: HashMap<String, SceneNode>,
    edges: Vec<SceneEdge>,
}

/// A bounded, task-relevant activation of the graph.
#[derive(Debug, Clone)]
pub struct ActiveSubgraph {
    /// Activated node ids in BFS order from the seeds.
    pub nodes: Vec<String>,
    /// Edges with both endpoints activated.
    pub edges: Vec<SceneEdge>,
    /// True when the activation hit `max_nodes` before exhausting reachable
    /// relevant nodes (callers can widen the budget deliberately).
    pub truncated: bool,
}

impl SceneGraph {
    /// Empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Inserts or replaces a node.
    pub fn upsert_node(&mut self, node: SceneNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Adds an edge (endpoints need not exist yet; dangling edges are
    /// simply never activated).
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>, relation: RelationKind) {
        self.edges.push(SceneEdge { from: from.into(), to: to.into(), relation });
    }

    /// Node lookup.
    #[must_use]
    pub fn node(&self, id: &str) -> Option<&SceneNode> {
        self.nodes.get(id)
    }

    /// Task-gated activation: BFS from `seed_ids`, following edges in both
    /// directions, keeping only nodes whose kind is in `relevant_kinds`,
    /// visiting at most `max_nodes` nodes. This is the *only* sanctioned way
    /// for reasoning code to read the graph — full scans are deliberately
    /// not offered on the public surface.
    #[must_use]
    pub fn activate(
        &self,
        relevant_kinds: &[EntityKind],
        seed_ids: &[&str],
        max_nodes: usize,
    ) -> ActiveSubgraph {
        let relevant: BTreeSet<EntityKind> = relevant_kinds.iter().copied().collect();
        let mut visited: BTreeSet<&str> = BTreeSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        let mut activated: Vec<String> = Vec::new();
        let mut truncated = false;

        for &s in seed_ids {
            if let Some(n) = self.nodes.get(s) {
                if relevant.contains(&n.kind) && visited.insert(s) {
                    queue.push_back(s);
                }
            }
        }

        while let Some(id) = queue.pop_front() {
            if activated.len() >= max_nodes {
                truncated = true;
                break;
            }
            activated.push(id.to_string());
            for e in &self.edges {
                let neighbor = if e.from == id {
                    Some(e.to.as_str())
                } else if e.to == id {
                    Some(e.from.as_str())
                } else {
                    None
                };
                let Some(nb) = neighbor else { continue };
                let Some(node) = self.nodes.get(nb) else { continue };
                if relevant.contains(&node.kind) && !visited.contains(nb) {
                    // Re-borrow with the graph's lifetime for the queue.
                    let (key, _) = self.nodes.get_key_value(nb).expect("just found");
                    visited.insert(key);
                    queue.push_back(key);
                }
            }
        }
        if !queue.is_empty() {
            truncated = true;
        }

        let in_set: BTreeSet<&str> = activated.iter().map(String::as_str).collect();
        let edges = self
            .edges
            .iter()
            .filter(|e| in_set.contains(e.from.as_str()) && in_set.contains(e.to.as_str()))
            .cloned()
            .collect();
        ActiveSubgraph { nodes: activated, edges, truncated }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_graph() -> SceneGraph {
        let mut g = SceneGraph::new();
        for (id, kind) in [
            ("kitchen", EntityKind::Room),
            ("living", EntityKind::Room),
            ("fridge", EntityKind::Object),
            ("sofa", EntityKind::Object),
            ("esp32-a", EntityKind::Device),
            ("person-class-1", EntityKind::PersonClass),
            ("anomaly-7", EntityKind::Event),
        ] {
            g.upsert_node(SceneNode { id: id.into(), kind, gaussian_refs: vec![] });
        }
        g.add_edge("kitchen", "fridge", RelationKind::Contains);
        g.add_edge("living", "sofa", RelationKind::Contains);
        g.add_edge("kitchen", "esp32-a", RelationKind::ObservedBy);
        g.add_edge("anomaly-7", "fridge", RelationKind::CausedBy);
        g.add_edge("person-class-1", "living", RelationKind::Near);
        g
    }

    #[test]
    fn activation_is_task_scoped() {
        let g = demo_graph();
        // Task: "which object caused the channel anomaly?" — events, objects,
        // rooms are relevant; devices and person classes are not.
        let active = g.activate(
            &[EntityKind::Event, EntityKind::Object, EntityKind::Room],
            &["anomaly-7"],
            10,
        );
        assert!(active.nodes.contains(&"anomaly-7".to_string()));
        assert!(active.nodes.contains(&"fridge".to_string()));
        assert!(active.nodes.contains(&"kitchen".to_string()));
        assert!(!active.nodes.iter().any(|n| n == "esp32-a"), "devices are gated out");
        assert!(!active.nodes.iter().any(|n| n == "person-class-1"));
        assert!(!active.truncated);
    }

    #[test]
    fn activation_respects_the_node_budget() {
        let g = demo_graph();
        let active = g.activate(
            &[
                EntityKind::Event,
                EntityKind::Object,
                EntityKind::Room,
                EntityKind::Device,
                EntityKind::PersonClass,
            ],
            &["anomaly-7"],
            2,
        );
        assert_eq!(active.nodes.len(), 2, "bounded active memory");
        assert!(active.truncated, "truncation must be reported, not silent");
    }

    #[test]
    fn unreachable_and_irrelevant_seeds_yield_empty() {
        let g = demo_graph();
        let active = g.activate(&[EntityKind::Object], &["nonexistent"], 10);
        assert!(active.nodes.is_empty());
        // Seed exists but its kind is not relevant ⇒ empty activation.
        let active = g.activate(&[EntityKind::Object], &["kitchen"], 10);
        assert!(active.nodes.is_empty());
    }
}
