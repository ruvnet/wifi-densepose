//! ADR-138 — `ArrayCoordinator`: a stateless-per-call domain service that gates
//! array nodes on geometry and clock quality and projects *directional evidence*
//! (not pose decisions).
//!
//! # Crate placement (deviation from ADR-138 §2.3, deliberate)
//!
//! ADR-138 placed `ArrayCoordinator` in `wifi-densepose-ruvector`
//! (`viewpoint/fusion.rs`). But `wifi-densepose-signal` already **depends on**
//! `wifi-densepose-ruvector`, and the coordinator must emit the canonical
//! [`ContradictionFlag`](super::fusion_quality::ContradictionFlag) owned by
//! ADR-137 (in this crate). Placing it in ruvector would create a dependency
//! cycle. It therefore lives here in `wifi-densepose-signal`, which can see both
//! ruvector's geometry/coherence types and ADR-137's `ContradictionFlag`. The
//! `ClockQualityGate` (which needs no `ContradictionFlag`) stays in ruvector per
//! the ADR.

use wifi_densepose_ruvector::viewpoint::coherence::{
    ClockGateDecision, ClockQualityGate, ClockQualityScore,
};
use wifi_densepose_ruvector::viewpoint::geometry::{
    CramerRaoBound, GeometricDiversityIndex, NodeId, ViewpointPosition,
};

use super::fusion_quality::ContradictionFlag;
use super::multistatic::node_attention_weights;

/// One node's contribution to the array for a single sensing cycle.
#[derive(Debug, Clone)]
pub struct ArrayNodeInput {
    /// Stable node identifier.
    pub node_id: NodeId,
    /// Node position (x, y) in metres (deployment geometry).
    pub position: (f32, f32),
    /// Azimuth (radians) of the node from the array centroid.
    pub azimuth: f32,
    /// Rolling phasor coherence for this node (`CoherenceState::coherence()`).
    pub coherence: f32,
    /// Clock-sync quality (ADR-110 follower offset stats).
    pub clock: ClockQualityScore,
    /// Optional per-node amplitude vector; when present across nodes, the
    /// directional weights use the real fusion attention (ADR-137
    /// `node_attention_weights`) instead of the clock-only fallback.
    pub amplitude: Option<Vec<f32>>,
}

/// Directional evidence: what the array can resolve right now and how much to
/// trust each direction (ADR-138 §2.3). NOT a pose decision.
#[derive(Debug, Clone)]
pub struct DirectionalEvidence {
    /// Per-admitted-viewpoint attention weight (sums to ~1.0 over admitted).
    pub weights: Vec<(NodeId, f32)>,
    /// Geometric Diversity Index over admitted nodes. `None` when < 2 admitted.
    ///
    /// (ADR-138 §2.3 typed this non-optional; made `Option` here because GDI is
    /// undefined for < 2 viewpoints and a sentinel would be misleading.)
    pub gdi: Option<GeometricDiversityIndex>,
    /// Cramér-Rao RMSE lower bound (m) for a centroid target. `None` when
    /// < 3 admitted viewpoints (under-determined).
    pub credence_rmse_m: Option<f32>,
    /// Per-node gate decisions — the audit trail.
    pub gate_decisions: Vec<(NodeId, ClockGateDecision)>,
    /// Contradiction flags forwarded to the ADR-137 fusion-quality machinery.
    pub contradictions: Vec<ContradictionFlag>,
    /// Viewpoints admitted at full weight.
    pub n_admitted: usize,
    /// Viewpoints admitted MonitorOnly (evidence-only, no environment update).
    pub n_monitoring: usize,
}

/// Configuration for [`ArrayCoordinator`].
#[derive(Debug, Clone)]
pub struct ArrayCoordinatorConfig {
    /// Per-node clock+coherence gate (cloned per node so hysteresis state does
    /// not leak across nodes within a cycle).
    pub gate: ClockQualityGate,
    /// σ multiple defining a cross-sectional coherence-drop contradiction.
    pub contradiction_sigma: f32,
    /// Per-measurement noise std (m) for the Cramér-Rao credence estimate.
    pub crb_noise_std_m: f32,
    /// Attention temperature for the directional weight softmax.
    pub attention_temperature: f32,
}

impl Default for ArrayCoordinatorConfig {
    fn default() -> Self {
        Self {
            gate: ClockQualityGate::default_params(),
            contradiction_sigma: 2.0,
            crb_noise_std_m: 0.1,
            attention_temperature: 1.0,
        }
    }
}

/// Stateless-per-call domain service (ADR-138 §2.3).
#[derive(Debug, Clone)]
pub struct ArrayCoordinator {
    config: ArrayCoordinatorConfig,
}

impl ArrayCoordinator {
    /// Create a coordinator with the given configuration.
    pub fn new(config: ArrayCoordinatorConfig) -> Self {
        Self { config }
    }

    /// Gate the nodes on clock+coherence, then over the admitted set compute
    /// GDI, Cramér-Rao credence, and attention weights, collecting contradiction
    /// flags (cross-sectional coherence drops + geometry insufficiency).
    pub fn coordinate(&self, nodes: &[ArrayNodeInput]) -> DirectionalEvidence {
        // 1. Per-node clock+coherence gate (fresh gate per node).
        let mut gate_decisions = Vec::with_capacity(nodes.len());
        for n in nodes {
            let mut gate = self.config.gate.clone();
            gate_decisions.push((n.node_id, gate.evaluate(n.coherence, &n.clock)));
        }

        // Admitted = full-weight; monitoring = evidence-only.
        let admitted_idx: Vec<usize> = (0..nodes.len())
            .filter(|&i| matches!(gate_decisions[i].1, ClockGateDecision::Admit))
            .collect();
        let monitoring_idx: Vec<usize> = (0..nodes.len())
            .filter(|&i| matches!(gate_decisions[i].1, ClockGateDecision::MonitorOnly { .. }))
            .collect();
        let evidence_idx: Vec<usize> =
            admitted_idx.iter().chain(monitoring_idx.iter()).copied().collect();

        let mut contradictions = Vec::new();

        // 2. Cross-sectional coherence-drop contradictions over the evidence set.
        if evidence_idx.len() >= 3 {
            let cohs: Vec<f32> = evidence_idx.iter().map(|&i| nodes[i].coherence).collect();
            let mean = cohs.iter().sum::<f32>() / cohs.len() as f32;
            let var = cohs.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / cohs.len() as f32;
            let std = var.sqrt();
            if std > 1e-6 {
                for &i in &evidence_idx {
                    let sigma = (mean - nodes[i].coherence) / std;
                    if sigma > self.config.contradiction_sigma {
                        contradictions.push(ContradictionFlag::CoherenceDrop { node_idx: i, sigma });
                    }
                }
            }
        }

        // 3. GDI over admitted nodes.
        let gdi = if admitted_idx.len() >= 2 {
            let azimuths: Vec<f32> = admitted_idx.iter().map(|&i| nodes[i].azimuth).collect();
            let ids: Vec<NodeId> = admitted_idx.iter().map(|&i| nodes[i].node_id).collect();
            GeometricDiversityIndex::compute(&azimuths, &ids)
        } else {
            None
        };
        if let Some(ref g) = gdi {
            if !g.is_sufficient() {
                contradictions.push(ContradictionFlag::GeometryInsufficient { gdi: g.value });
            }
        }

        // 4. Cramér-Rao credence for a centroid target over admitted nodes.
        let credence_rmse_m = if admitted_idx.len() >= 3 {
            let vps: Vec<ViewpointPosition> = admitted_idx
                .iter()
                .map(|&i| ViewpointPosition {
                    x: nodes[i].position.0,
                    y: nodes[i].position.1,
                    noise_std: self.config.crb_noise_std_m,
                })
                .collect();
            let cx = vps.iter().map(|v| v.x).sum::<f32>() / vps.len() as f32;
            let cy = vps.iter().map(|v| v.y).sum::<f32>() / vps.len() as f32;
            CramerRaoBound::estimate((cx, cy), &vps).map(|crb| crb.rmse_lower_bound)
        } else {
            None
        };

        // 5. Attention weights over admitted nodes.
        let weights = self.admitted_weights(nodes, &admitted_idx);

        DirectionalEvidence {
            weights,
            gdi,
            credence_rmse_m,
            gate_decisions,
            contradictions,
            n_admitted: admitted_idx.len(),
            n_monitoring: monitoring_idx.len(),
        }
    }

    /// Directional weights over the admitted set. When every admitted node has
    /// an amplitude vector of equal length, reuse the ADR-137 fusion attention
    /// (`node_attention_weights`); otherwise fall back to a clock-quality
    /// softmax so well-clocked nodes weigh more.
    fn admitted_weights(
        &self,
        nodes: &[ArrayNodeInput],
        admitted_idx: &[usize],
    ) -> Vec<(NodeId, f32)> {
        if admitted_idx.is_empty() {
            return Vec::new();
        }
        // Try the real fusion-attention path when amplitudes are present + uniform.
        let amps: Option<Vec<&[f32]>> = admitted_idx
            .iter()
            .map(|&i| nodes[i].amplitude.as_deref())
            .collect();
        if let Some(amps) = amps {
            let len0 = amps.first().map(|a| a.len()).unwrap_or(0);
            if len0 > 0 && amps.iter().all(|a| a.len() == len0) {
                let w = node_attention_weights(&amps, self.config.attention_temperature);
                return admitted_idx.iter().map(|&i| nodes[i].node_id).zip(w).collect();
            }
        }

        // Clock-quality softmax fallback.
        let max_floor = self.config.gate.max_offset_stdev_us;
        let logits: Vec<f32> = admitted_idx
            .iter()
            .map(|&i| nodes[i].clock.quality(max_floor) / self.config.attention_temperature)
            .collect();
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
        admitted_idx
            .iter()
            .zip(exps)
            .map(|(&i, e)| (nodes[i].node_id, e / sum))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(stdev: f32, age_us: u64) -> ClockQualityScore {
        ClockQualityScore { offset_stdev_us: stdev, age_us, valid: true }
    }

    fn node(id: NodeId, x: f32, y: f32, az: f32, coh: f32, stdev: f32) -> ArrayNodeInput {
        ArrayNodeInput {
            node_id: id,
            position: (x, y),
            azimuth: az,
            coherence: coh,
            clock: clock(stdev, 1000),
            amplitude: None,
        }
    }

    /// 4 well-placed, well-clocked, coherent nodes → all admitted, weights sum
    /// to 1, credence available, no contradictions.
    #[test]
    fn ac_four_good_nodes_all_admitted() {
        use std::f32::consts::PI;
        let coord = ArrayCoordinator::new(ArrayCoordinatorConfig::default());
        let nodes = vec![
            node(0, 1.0, 0.0, 0.0, 0.9, 50.0),
            node(1, 0.0, 1.0, PI / 2.0, 0.9, 50.0),
            node(2, -1.0, 0.0, PI, 0.9, 50.0),
            node(3, 0.0, -1.0, 3.0 * PI / 2.0, 0.9, 50.0),
        ];
        let ev = coord.coordinate(&nodes);
        assert_eq!(ev.n_admitted, 4);
        assert_eq!(ev.n_monitoring, 0);
        assert!((ev.weights.iter().map(|(_, w)| *w).sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(ev.credence_rmse_m.is_some());
        assert!(ev.gdi.is_some() && ev.gdi.as_ref().unwrap().is_sufficient());
        assert!(ev.contradictions.is_empty());
    }

    /// A clock-degraded node (offset ≥ 200 µs floor) is MonitorOnly: evidence
    /// yes, not counted as admitted.
    #[test]
    fn ac_clock_degraded_node_is_monitor_only() {
        use std::f32::consts::PI;
        let coord = ArrayCoordinator::new(ArrayCoordinatorConfig::default());
        let mut nodes = vec![
            node(0, 1.0, 0.0, 0.0, 0.9, 50.0),
            node(1, 0.0, 1.0, PI / 2.0, 0.9, 50.0),
            node(2, -1.0, 0.0, PI, 0.9, 50.0),
        ];
        nodes[2].clock = clock(250.0, 1000); // above 200 µs floor, below 1000 µs hard
        let ev = coord.coordinate(&nodes);
        assert_eq!(ev.n_admitted, 2);
        assert_eq!(ev.n_monitoring, 1);
        assert!(matches!(
            ev.gate_decisions[2].1,
            ClockGateDecision::MonitorOnly { .. }
        ));
    }

    /// A stale node (age > 9 s) is hard-rejected.
    #[test]
    fn ac_stale_node_rejected() {
        let coord = ArrayCoordinator::new(ArrayCoordinatorConfig::default());
        let mut n0 = node(0, 1.0, 0.0, 0.0, 0.9, 50.0);
        n0.clock = clock(50.0, 10_000_000); // 10 s > 9 s ceiling
        let ev = coord.coordinate(&[n0]);
        assert_eq!(ev.n_admitted, 0);
        assert!(matches!(
            ev.gate_decisions[0].1,
            ClockGateDecision::Reject {
                reason: wifi_densepose_ruvector::viewpoint::coherence::ClockRejectReason::ClockStale
            }
        ));
    }

    /// An incoherent node (coherence below the phase gate) is rejected.
    #[test]
    fn ac_incoherent_node_rejected() {
        let coord = ArrayCoordinator::new(ArrayCoordinatorConfig::default());
        let n0 = node(0, 1.0, 0.0, 0.0, 0.2, 50.0); // 0.2 < 0.7 gate
        let ev = coord.coordinate(&[n0]);
        assert_eq!(ev.n_admitted, 0);
    }

    /// A cross-sectional coherence outlier raises a `CoherenceDrop` flag.
    ///
    /// Uses 6 nodes: with a single outlier among N equal values the outlier's
    /// z-score is exactly √(N-1), so N≥6 is required to exceed the default 2σ
    /// threshold (√5≈2.24). This is an inherent property of cross-sectional
    /// outlier detection, not a tuning artefact.
    #[test]
    fn ac_coherence_outlier_flagged() {
        use std::f32::consts::PI;
        let coord = ArrayCoordinator::new(ArrayCoordinatorConfig::default());
        let nodes: Vec<ArrayNodeInput> = (0..6)
            .map(|i| {
                let az = i as f32 * PI / 3.0;
                // Node 5 is the low-coherence outlier (still above the 0.7 gate).
                let coh = if i == 5 { 0.71 } else { 0.95 };
                node(i, az.cos(), az.sin(), az, coh, 50.0)
            })
            .collect();
        let ev = coord.coordinate(&nodes);
        assert!(ev
            .contradictions
            .iter()
            .any(|c| matches!(c, ContradictionFlag::CoherenceDrop { node_idx: 5, .. })));
    }
}
