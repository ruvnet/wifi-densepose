//! Mesh partition guard: dynamic min-cut over the live multistatic node graph.
//!
//! The fusion mesh (nodes = sensing nodes, edge weights = fusion coupling
//! derived from per-node attention weights) changes *incrementally* at cycle
//! rate — one node's coupling drifts, a node joins or drops. This module
//! maintains a [`ruvector_mincut::DynamicMinCut`] over that graph and exposes,
//! per cycle:
//!
//! - the **min-cut value** — the cheapest set of couplings whose loss splits
//!   the mesh in two: a principled, global "how close is the array to
//!   partitioning" number (vs per-node heuristics that miss multi-node
//!   structure);
//! - the **weak side** — which specific nodes are about to partition (feeds
//!   failure/jamming triage, ADR-032 posture);
//! - an **at-risk flag** consumed by the engine: it counts as a structural
//!   event for the drift→recalibration advisor.
//!
//! ## Cost model (the optimization)
//!
//! Weights are quantized (default 1/64; a *nonzero* coupling below one quantum
//! saturates to quantum 1 so a live coupling is never erased — see
//! [`MeshGuard::weight_quantum`]) and updates are **change-gated**: an
//! edge is touched only when its quantized weight actually moves, so the
//! steady-state cycle applies *zero* graph updates and reuses the cached cut —
//! O(active-changes) per cycle, not O(n²) rebuilds. The exact (deterministic)
//! algorithm is used; mesh sizes are ≤ tens of nodes, far inside its budget.

use std::collections::BTreeMap;

use ruvector_mincut::{DynamicMinCut, MinCutBuilder};

/// Per-cycle report from the mesh guard.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshPartitionReport {
    /// Current min-cut value over the coupling graph (higher = more robust).
    pub cut_value: f64,
    /// True when the mesh has ≥ `min_nodes` nodes and the cut value fell to or
    /// below the risk threshold — the array is close to splitting.
    pub at_risk: bool,
    /// The smaller side of the min-cut partition (node ids): the nodes that
    /// would be isolated if the weak couplings failed.
    pub weak_side: Vec<u8>,
    /// Incremental edge updates applied this cycle (0 in steady state).
    pub updates_applied: usize,
}

/// Dynamic min-cut guard over the live mesh.
pub struct MeshGuard {
    mincut: Option<DynamicMinCut>,
    /// Node set the structure was built over (sorted). A change forces rebuild.
    nodes: Vec<u8>,
    /// Quantized edge weights currently installed, keyed `(u, v)` with `u < v`.
    edges: BTreeMap<(u8, u8), i64>,
    /// Weight quantum: weights are snapped to multiples of this before
    /// comparison/installation, gating out sub-quantum jitter.
    ///
    /// Policy: a **nonzero** coupling below one quantum saturates to quantum 1
    /// instead of quantizing to 0 — quantization never erases a live coupling.
    /// (Without the floor, a balanced mesh of ≥ 65 nodes — attention weights
    /// ~1/n ⇒ couplings ~1/n < 1/64 — had every edge erased and was reported
    /// permanently "already partitioned"/at-risk.) Exact zero stays zero: a
    /// truly absent coupling *is* a partition. Relative weakness below one
    /// quantum is not resolved; lower this quantum if that resolution matters.
    pub weight_quantum: f64,
    /// Cut value at or below which the mesh counts as at partition risk.
    pub risk_threshold: f64,
    /// Minimum node count for risk to be meaningful (a 2-node mesh always has
    /// a trivial cut; default 3).
    pub min_nodes: usize,
}

impl Default for MeshGuard {
    fn default() -> Self {
        Self {
            mincut: None,
            nodes: Vec::new(),
            edges: BTreeMap::new(),
            weight_quantum: 1.0 / 64.0,
            risk_threshold: 0.25,
            min_nodes: 3,
        }
    }
}

impl MeshGuard {
    /// Quantize a raw weight to the guard's grid (floor; weights are ≥ 0).
    /// Nonzero sub-quantum weights saturate to quantum 1 — see the
    /// [`Self::weight_quantum`] policy (review finding: sub-quantum couplings
    /// must not produce a false "already partitioned").
    fn quantize(&self, w: f64) -> i64 {
        let w = w.max(0.0);
        let q = (w / self.weight_quantum).floor() as i64;
        if q == 0 && w > 0.0 {
            1
        } else {
            q
        }
    }

    /// Update the guard with this cycle's mesh: `nodes` are the contributing
    /// node ids and `coupling(i, j)` returns the fusion coupling between
    /// `nodes[i]` and `nodes[j]` (symmetric, ≥ 0).
    ///
    /// Returns `None` for meshes of fewer than 2 nodes (no cut exists).
    pub fn update(
        &mut self,
        nodes: &[u8],
        coupling: impl Fn(usize, usize) -> f64,
    ) -> Option<MeshPartitionReport> {
        if nodes.len() < 2 {
            // Mesh degenerated: drop state so a later rebuild starts clean.
            self.mincut = None;
            self.nodes.clear();
            self.edges.clear();
            return None;
        }
        let mut sorted: Vec<u8> = nodes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        // Desired quantized edge set for this cycle.
        let mut desired: BTreeMap<(u8, u8), i64> = BTreeMap::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let (a, b) = if nodes[i] < nodes[j] {
                    (nodes[i], nodes[j])
                } else {
                    (nodes[j], nodes[i])
                };
                if a == b {
                    continue;
                }
                let q = self.quantize(coupling(i, j));
                desired.insert((a, b), q);
            }
        }

        // Change detection: count quantized-weight moves vs the installed set.
        let changed = if self.mincut.is_none() || self.nodes != sorted {
            usize::MAX // node set changed / first cycle: rebuild unconditionally
        } else {
            desired
                .iter()
                .filter(|(k, &q)| self.edges.get(k).copied().unwrap_or(0) != q)
                .count()
        };

        let mut updates = 0usize;
        if changed > 0 {
            // Measured policy (criterion, 12-node mesh): a full exact rebuild
            // is ~170 µs while ONE DynamicMinCut delete+insert is ~240 µs —
            // the incremental machinery's overheads target much larger graphs.
            // At mesh scale the optimum is: change-gate aggressively (the
            // steady state below is ~7 µs and covers almost every cycle) and
            // rebuild whenever anything actually moved.
            let edges: Vec<(u64, u64, f64)> = desired
                .iter()
                .filter(|(_, &q)| q > 0)
                .map(|(&(a, b), &q)| {
                    (u64::from(a), u64::from(b), q as f64 * self.weight_quantum)
                })
                .collect();
            updates = if changed == usize::MAX { edges.len() } else { changed };
            self.mincut = MinCutBuilder::new().exact().with_edges(edges).build().ok();
            self.nodes = sorted;
            self.edges = desired;
        }
        // changed == 0: steady state — zero graph work, cached cut reused.

        // Nodes with no positive coupling never enter the cut structure (zero
        // edges are not installed) — they are already partitioned. Report them
        // as the degenerate cut before consulting the structure.
        let mut isolated: Vec<u8> = self
            .nodes
            .iter()
            .copied()
            .filter(|&v| {
                !self
                    .edges
                    .iter()
                    .any(|(&(a, b), &q)| q > 0 && (a == v || b == v))
            })
            .collect();
        if !isolated.is_empty() {
            isolated.sort_unstable();
            return Some(MeshPartitionReport {
                cut_value: 0.0,
                at_risk: self.nodes.len() >= self.min_nodes,
                weak_side: isolated,
                updates_applied: updates,
            });
        }

        let mc = self.mincut.as_ref()?;
        // A disconnected coupling graph is the degenerate cut: value 0.
        let cut_value = if mc.is_connected() { mc.min_cut_value() } else { 0.0 };
        let (side_a, side_b) = mc.partition();
        let weak_raw = if side_a.len() <= side_b.len() { side_a } else { side_b };
        let mut weak_side: Vec<u8> = weak_raw.into_iter().map(|v| v as u8).collect();
        weak_side.sort_unstable();
        let at_risk = self.nodes.len() >= self.min_nodes && cut_value <= self.risk_threshold;

        Some(MeshPartitionReport { cut_value, at_risk, weak_side, updates_applied: updates })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Triangle with one weakly-attached node: the cut isolates that node and
    /// the cut value equals its total coupling.
    #[test]
    fn weakly_attached_node_is_the_weak_side() {
        let mut g = MeshGuard::default();
        let nodes = [0u8, 1, 2];
        // 0–1 strongly coupled; node 2 hangs on by 0.05 + 0.05.
        let w = |i: usize, j: usize| match (i.min(j), i.max(j)) {
            (0, 1) => 1.0,
            _ => 0.05,
        };
        let r = g.update(&nodes, w).expect("3-node mesh");
        assert!(r.cut_value <= 0.13, "cut {} should be ~0.10", r.cut_value);
        assert_eq!(r.weak_side, vec![2]);
        assert!(r.at_risk, "weak coupling must flag partition risk");
    }

    #[test]
    fn strong_mesh_is_not_at_risk() {
        let mut g = MeshGuard::default();
        let r = g.update(&[0, 1, 2, 3], |_, _| 0.9).expect("mesh");
        assert!(r.cut_value > g.risk_threshold);
        assert!(!r.at_risk);
    }

    #[test]
    fn two_node_mesh_reports_but_never_risks() {
        let mut g = MeshGuard::default();
        let r = g.update(&[0, 1], |_, _| 0.01).expect("2-node mesh");
        // Trivial cut exists but min_nodes=3 keeps the flag off.
        assert!(!r.at_risk);
    }

    #[test]
    fn fewer_than_two_nodes_yields_none() {
        let mut g = MeshGuard::default();
        assert!(g.update(&[7], |_, _| 1.0).is_none());
        assert!(g.update(&[], |_, _| 1.0).is_none());
    }

    /// The optimization contract: identical weights on the next cycle apply
    /// zero updates; a sub-quantum wiggle also applies zero; a real change
    /// applies exactly the changed edges.
    #[test]
    fn steady_state_applies_zero_updates() {
        let mut g = MeshGuard::default();
        let nodes = [0u8, 1, 2, 3];
        let first = g.update(&nodes, |_, _| 0.5).unwrap();
        assert_eq!(first.updates_applied, 6); // cold build installs all edges

        let second = g.update(&nodes, |_, _| 0.5).unwrap();
        assert_eq!(second.updates_applied, 0);

        // Sub-quantum jitter (quantum is 1/64 ≈ 0.0156) is gated out.
        let third = g.update(&nodes, |_, _| 0.5 + 0.004).unwrap();
        assert_eq!(third.updates_applied, 0);

        // One genuinely changed edge touches exactly one edge.
        let fourth = g
            .update(&nodes, |i, j| if (i.min(j), i.max(j)) == (0, 1) { 0.1 } else { 0.5 })
            .unwrap();
        assert_eq!(fourth.updates_applied, 1);
    }

    /// Node set changes force a clean rebuild (drop/join handled correctly).
    #[test]
    fn node_join_and_drop_rebuild() {
        let mut g = MeshGuard::default();
        g.update(&[0, 1, 2], |_, _| 0.8).unwrap();
        // Node 3 joins.
        let joined = g.update(&[0, 1, 2, 3], |_, _| 0.8).unwrap();
        assert_eq!(joined.updates_applied, 6); // rebuild over 4 nodes
        // Node 0 drops.
        let dropped = g.update(&[1, 2, 3], |_, _| 0.8).unwrap();
        assert_eq!(dropped.updates_applied, 3);
        assert!(!dropped.at_risk);
    }

    /// Determinism: same inputs, same report (cut value + weak side).
    #[test]
    fn reports_are_deterministic() {
        let run = || {
            let mut g = MeshGuard::default();
            let w = |i: usize, j: usize| match (i.min(j), i.max(j)) {
                (0, 1) => 0.9,
                (1, 2) => 0.6,
                _ => 0.07,
            };
            g.update(&[0, 1, 2], w).unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.cut_value.to_bits(), b.cut_value.to_bits());
        assert_eq!(a.weak_side, b.weak_side);
    }

    /// Regression (review finding 3): a balanced mesh of ≥ 65 nodes has every
    /// pairwise coupling at ~1/n < quantum (1/64). The old floor-to-zero
    /// quantization erased all edges and reported the mesh permanently
    /// "already partitioned" (cut 0, at_risk). Nonzero sub-quantum couplings
    /// now saturate to one quantum, so the mesh reports a healthy cut.
    #[test]
    fn large_balanced_mesh_is_not_at_risk() {
        let mut g = MeshGuard::default();
        let nodes: Vec<u8> = (0..70u8).collect();
        // Attention-weight product coupling: (1/n)·(1/n)·n = 1/n ≈ 0.0143 < 1/64.
        let n = nodes.len() as f64;
        let r = g.update(&nodes, |_, _| 1.0 / n).expect("70-node mesh");
        assert!(
            r.cut_value > 0.0,
            "live couplings must not quantize to zero"
        );
        // Min cut isolates one node: 69 edges × one quantum (1/64) ≈ 1.08,
        // well above the 0.25 default risk threshold.
        assert!(r.cut_value > g.risk_threshold);
        assert!(
            !r.at_risk,
            "balanced large mesh must not be at partition risk"
        );
        assert!(r.weak_side.len() < nodes.len(), "no false full partition");
    }

    /// Sub-quantum couplings saturate to one quantum but exact zero is still a
    /// real partition (the floor must not invent couplings).
    #[test]
    fn sub_quantum_saturates_but_zero_stays_zero() {
        let mut g = MeshGuard::default();
        // 0.001 < 1/64 everywhere: connected, tiny cut, flagged at risk
        // (cut = 2 × 1/64 ≈ 0.031 ≤ 0.25) — but NOT "already partitioned".
        let r = g.update(&[0, 1, 2], |_, _| 0.001).expect("mesh");
        assert!(r.cut_value > 0.0);
        assert!(r.at_risk);
        // Exact zero to node 2: degenerate cut 0, node 2 isolated.
        let mut g2 = MeshGuard::default();
        let r2 = g2
            .update(&[0, 1, 2], |i, j| if i == 2 || j == 2 { 0.0 } else { 0.5 })
            .expect("mesh");
        assert_eq!(r2.cut_value, 0.0);
        assert_eq!(r2.weak_side, vec![2]);
    }

    /// A fully partitioned mesh (zero coupling to one node) reports cut 0.
    #[test]
    fn disconnected_mesh_is_cut_zero() {
        let mut g = MeshGuard::default();
        let w = |i: usize, j: usize| {
            if i == 2 || j == 2 { 0.0 } else { 0.9 }
        };
        let r = g.update(&[0, 1, 2], w).unwrap();
        assert_eq!(r.cut_value, 0.0);
        assert!(r.at_risk);
        assert_eq!(r.weak_side, vec![2]);
    }
}
