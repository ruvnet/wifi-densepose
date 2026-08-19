//! Space/zone adjacency that constrains plausible hand-offs (ADR-307 §2).
//!
//! Association across containers is only allowed between the **same** container
//! or two **adjacent** ones (the ADR-306 `AdjacentTo`/`Doorway` analogue): a
//! person can only move between spaces that share a boundary. An empty topology
//! therefore permits continuity only *within* a container — the privacy-safe
//! default for single-room deployments, where cross-room joins never happen by
//! accident.

use std::collections::BTreeSet;

use ruview_ontology::Container;

/// Undirected adjacency between [`Container`]s.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Topology {
    /// Normalized `(low, high)` key pairs of connected containers.
    edges: BTreeSet<(String, String)>,
}

impl Topology {
    /// An empty topology: only same-container continuity is permitted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stable string key for a container, discriminated by kind so a space and a
    /// zone sharing a raw id never collide.
    fn key(c: &Container) -> String {
        match c {
            Container::Space { id } => format!("space:{}", id.as_str()),
            Container::Zone { id } => format!("zone:{}", id.as_str()),
        }
    }

    fn pair(a: &Container, b: &Container) -> (String, String) {
        let (ka, kb) = (Self::key(a), Self::key(b));
        if ka <= kb {
            (ka, kb)
        } else {
            (kb, ka)
        }
    }

    /// Record that two containers are adjacent (idempotent, undirected).
    pub fn connect(&mut self, a: &Container, b: &Container) -> &mut Self {
        if Self::key(a) != Self::key(b) {
            self.edges.insert(Self::pair(a, b));
        }
        self
    }

    /// Whether a hand-off from `from` to `to` is topologically plausible: the
    /// same container, or a recorded adjacency.
    #[must_use]
    pub fn adjacent(&self, from: &Container, to: &Container) -> bool {
        Self::key(from) == Self::key(to) || self.edges.contains(&Self::pair(from, to))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_ontology::SpaceId;

    fn space(id: &str) -> Container {
        Container::Space {
            id: SpaceId::new(id).unwrap(),
        }
    }

    #[test]
    fn same_container_is_always_adjacent() {
        let t = Topology::new();
        assert!(t.adjacent(&space("kitchen"), &space("kitchen")));
    }

    #[test]
    fn empty_topology_blocks_cross_container() {
        let t = Topology::new();
        assert!(!t.adjacent(&space("kitchen"), &space("bedroom")));
    }

    #[test]
    fn connect_is_undirected() {
        let mut t = Topology::new();
        t.connect(&space("kitchen"), &space("hallway"));
        assert!(t.adjacent(&space("kitchen"), &space("hallway")));
        assert!(t.adjacent(&space("hallway"), &space("kitchen")));
        assert!(!t.adjacent(&space("kitchen"), &space("bedroom")));
    }
}
