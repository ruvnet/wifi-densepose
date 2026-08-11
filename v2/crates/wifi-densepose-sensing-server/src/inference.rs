//! ADR-294 — per-node inference vs. fused room inference.
//!
//! The multi-node path used to collapse two distinct concepts into one:
//!
//! 1. A **per-node** classification (what one node sees), and
//! 2. A **room aggregate** (the fused belief across all nodes).
//!
//! Because the top-level room classification was taken from the latest-arriving
//! node while other features were fused, disagreeing nodes made room presence
//! flip at packet frequency (issue #1555); and because the MQTT mapper fell back
//! to the room aggregate when a node lacked its own classification, every node
//! could publish the same value (issues #1540, #1554).
//!
//! This module separates the two types and provides a **pure, deterministic**
//! room fusion ([`fuse_room`]): identical input sets yield identical output,
//! independent of node ordering, and a node that has gone silent longer than
//! the stale window contributes nothing (it goes unavailable/stale rather than
//! holding a frozen value). Freshness is carried as milliseconds so the types
//! serialize cleanly and the logic needs no clock.

use serde::{Deserialize, Serialize};

/// One node's own inference — never the room aggregate. `NodeInfo` carries this
/// so a node reports what *it* sees, with no silent fallback to the room value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInference {
    /// Coarse classification label this node reports (e.g. `"present_moving"`,
    /// `"present_still"`, `"absent"`). Node-local; never overwritten by fusion.
    pub classification: String,
    /// Confidence in `classification`, clamped to `0.0..=1.0`.
    pub confidence: f64,
    /// Age of the backing frame in milliseconds. `None` when the node has never
    /// reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age_ms: Option<u64>,
}

impl NodeInference {
    /// Construct a per-node inference, clamping `confidence` into range.
    pub fn new(classification: impl Into<String>, confidence: f64, age_ms: Option<u64>) -> Self {
        Self {
            classification: classification.into(),
            confidence: clamp01(confidence),
            age_ms,
        }
    }

    /// A node is stale when it has never reported, or its last frame is at least
    /// `stale_after_ms` old. Stale nodes do not vote in [`fuse_room`].
    pub fn is_stale(&self, stale_after_ms: u64) -> bool {
        match self.age_ms {
            None => true,
            Some(a) => a >= stale_after_ms,
        }
    }

    /// Freshness weight in `[0.0, 1.0]`: `1.0` when brand new, decaying linearly
    /// to `0.0` at `stale_after_ms`, and exactly `0.0` once stale. Deterministic
    /// and clock-free.
    pub fn freshness_weight(&self, stale_after_ms: u64) -> f64 {
        if stale_after_ms == 0 {
            return 0.0;
        }
        match self.age_ms {
            None => 0.0,
            Some(a) if a >= stale_after_ms => 0.0,
            Some(a) => {
                let remaining = stale_after_ms.saturating_sub(a) as f64;
                clamp01(remaining / stale_after_ms as f64)
            }
        }
    }
}

/// The fused room aggregate — computed explicitly from the set of per-node
/// inferences, never by overwriting any node's state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoomInference {
    /// Winning classification across the contributing nodes. `"unavailable"`
    /// when no fresh node contributed (never a frozen last value).
    pub classification: String,
    /// Freshness-weighted mean confidence of the nodes that voted for the
    /// winning class, `0.0..=1.0`.
    pub confidence: f64,
    /// How many non-stale nodes contributed to this aggregate.
    pub contributing_nodes: usize,
}

impl RoomInference {
    /// The state a room takes when no node currently backs it. Explicit so
    /// callers never invent an online value out of nothing.
    pub fn unavailable() -> Self {
        Self {
            classification: "unavailable".to_string(),
            confidence: 0.0,
            contributing_nodes: 0,
        }
    }

    /// Migration accessor for consumers of the pre-ADR-294 aggregate-only shape:
    /// with a single node, that node's inference *is* the room aggregate. This
    /// preserves single-node behavior (one node = one inference) exactly.
    pub fn from_single_node(node: &NodeInference, stale_after_ms: u64) -> Self {
        fuse_room(std::iter::once(node), stale_after_ms)
    }
}

/// Fuse per-node inferences into one room aggregate.
///
/// Pure and deterministic (ADR-294): the result depends only on the *set* of
/// inputs, not on their order or on which arrived last. Each non-stale node
/// votes for its classification with its freshness weight; the class with the
/// greatest total freshness weight wins, ties broken by classification string
/// order (via the sorted `BTreeMap`) so the outcome is stable. Room confidence
/// is the freshness-weighted mean confidence of the winning class's voters.
///
/// A node silent longer than `stale_after_ms` contributes nothing; if no node
/// contributes, the room is [`RoomInference::unavailable`] rather than a frozen
/// online value.
pub fn fuse_room<'a, I>(nodes: I, stale_after_ms: u64) -> RoomInference
where
    I: IntoIterator<Item = &'a NodeInference>,
{
    use std::collections::BTreeMap;

    // Per class: (sum of freshness weights, sum of freshness*confidence).
    let mut tally: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    let mut contributing = 0usize;

    for n in nodes {
        let w = n.freshness_weight(stale_after_ms);
        if w <= 0.0 {
            continue;
        }
        contributing += 1;
        let entry = tally.entry(n.classification.as_str()).or_insert((0.0, 0.0));
        entry.0 += w;
        entry.1 += w * n.confidence;
    }

    if contributing == 0 {
        return RoomInference::unavailable();
    }

    // BTreeMap iterates in sorted key order; `>` keeps the first (lowest-order)
    // class on a tie, so the winner is order-independent and deterministic.
    let mut best: Option<(&str, f64, f64)> = None;
    for (&class, &(weight, conf_weight)) in &tally {
        match best {
            Some((_, bw, _)) if weight <= bw => {}
            _ => best = Some((class, weight, conf_weight)),
        }
    }

    let (class, weight, conf_weight) = best.expect("contributing > 0 guarantees a winner");
    let confidence = if weight > 0.0 { clamp01(conf_weight / weight) } else { 0.0 };
    RoomInference {
        classification: class.to_string(),
        confidence,
        contributing_nodes: contributing,
    }
}

fn clamp01(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STALE: u64 = 10_000; // 10 s

    fn node(c: &str, conf: f64, age_ms: u64) -> NodeInference {
        NodeInference::new(c, conf, Some(age_ms))
    }

    // ── NodeInference basics ─────────────────────────────────────────────

    #[test]
    fn new_clamps_confidence() {
        assert_eq!(NodeInference::new("absent", 1.7, Some(0)).confidence, 1.0);
        assert_eq!(NodeInference::new("absent", -0.5, Some(0)).confidence, 0.0);
        assert_eq!(NodeInference::new("absent", f64::NAN, Some(0)).confidence, 0.0);
    }

    #[test]
    fn never_reported_node_is_stale() {
        let n = NodeInference::new("present_moving", 0.9, None);
        assert!(n.is_stale(STALE));
        assert_eq!(n.freshness_weight(STALE), 0.0);
    }

    #[test]
    fn fresh_node_is_not_stale_and_weighted() {
        let n = node("present_moving", 0.9, 0);
        assert!(!n.is_stale(STALE));
        assert_eq!(n.freshness_weight(STALE), 1.0);
        // Half the window ⇒ half weight.
        assert!((node("x", 1.0, STALE / 2).freshness_weight(STALE) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn node_at_window_boundary_is_stale() {
        let n = node("present_still", 0.8, STALE);
        assert!(n.is_stale(STALE));
        assert_eq!(n.freshness_weight(STALE), 0.0);
    }

    // ── Single-node preservation + migration accessor ────────────────────

    #[test]
    fn single_node_room_equals_that_node() {
        let n = node("present_moving", 0.8, 100);
        let room = fuse_room(std::iter::once(&n), STALE);
        assert_eq!(room.classification, "present_moving");
        assert!((room.confidence - 0.8).abs() < 1e-9, "confidence preserved");
        assert_eq!(room.contributing_nodes, 1);
    }

    #[test]
    fn migration_accessor_matches_single_node_fusion() {
        let n = node("present_still", 0.6, 250);
        assert_eq!(RoomInference::from_single_node(&n, STALE), fuse_room([&n], STALE));
    }

    // ── Deterministic, order-independent fusion ──────────────────────────

    #[test]
    fn agreeing_nodes_fuse_to_shared_class() {
        let a = node("present_moving", 0.7, 0);
        let b = node("present_moving", 0.9, 0);
        let room = fuse_room([&a, &b], STALE);
        assert_eq!(room.classification, "present_moving");
        assert_eq!(room.contributing_nodes, 2);
        // Freshness-weighted mean of 0.7 and 0.9 at equal weight = 0.8.
        assert!((room.confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn disagreeing_equal_freshness_is_deterministic_and_order_independent() {
        let a = node("present_moving", 0.9, 0);
        let b = node("absent", 0.9, 0);
        let one = fuse_room([&a, &b], STALE);
        let two = fuse_room([&b, &a], STALE);
        // Same result regardless of input order — no last-writer-wins.
        assert_eq!(one, two);
        // Tie broken by classification string order: "absent" < "present_moving".
        assert_eq!(one.classification, "absent");
    }

    #[test]
    fn fresher_node_outvotes_stale_disagreement() {
        // Fresh "present_moving" vs. an older-but-still-fresh "absent".
        let fresh = node("present_moving", 0.9, 0);
        let older = node("absent", 0.9, STALE - 1); // weight ~0
        let room = fuse_room([&older, &fresh], STALE);
        assert_eq!(room.classification, "present_moving");
    }

    // ── Stale handling: unavailable, never frozen-online ─────────────────

    #[test]
    fn all_stale_nodes_yield_unavailable() {
        let a = node("present_moving", 0.9, STALE + 1);
        let b = NodeInference::new("present_still", 0.8, None);
        let room = fuse_room([&a, &b], STALE);
        assert_eq!(room, RoomInference::unavailable());
        assert_eq!(room.classification, "unavailable");
        assert_eq!(room.contributing_nodes, 0);
        assert_eq!(room.confidence, 0.0);
    }

    #[test]
    fn stale_node_excluded_but_fresh_node_still_counts() {
        let stale_node = node("absent", 0.9, STALE + 5);
        let fresh_node = node("present_moving", 0.7, 10);
        let room = fuse_room([&stale_node, &fresh_node], STALE);
        assert_eq!(room.classification, "present_moving");
        assert_eq!(room.contributing_nodes, 1, "stale node does not vote");
    }

    #[test]
    fn empty_input_is_unavailable() {
        let room = fuse_room(std::iter::empty(), STALE);
        assert_eq!(room, RoomInference::unavailable());
    }

    #[test]
    fn node_inference_round_trips_json() {
        let n = node("present_moving", 0.75, 120);
        let json = serde_json::to_string(&n).unwrap();
        let back: NodeInference = serde_json::from_str(&json).unwrap();
        assert_eq!(n, back);
    }
}
