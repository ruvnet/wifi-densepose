//! ADR-137 — Fusion-engine quality scoring with evidence references and
//! contradiction flags.
//!
//! Every fusion stage emits a [`QualityScore`] alongside its payload. The score
//! names the positive evidence ([`EvidenceRef`]) that justified the fusion and
//! the tolerated-but-recorded disagreements ([`ContradictionFlag`]) that must
//! lower the downstream BFLD privacy class (ADR-141 §2 / ADR-120). It implements
//! the ADR-136 [`QualityScored`](super::QualityScored) trait so the streaming
//! engine can route, gate, and log on quality uniformly.
//!
//! [`ContradictionFlag`] is the **single canonical type** for tolerated fusion
//! disagreements (ADR-137 §2.3); the ADR-138 `ArrayCoordinator` imports it and
//! emits its `CoherenceDrop` / `GeometryInsufficient` variants.

use super::QualityScored;

/// Multiplicative coherence penalty applied per recorded contradiction
/// (ADR-154 §7.4 — de-magicked; EMPIRICAL DEFAULT). `n` contradictions scale
/// coherence by `CONTRADICTION_PENALTY.powi(n)`.
const CONTRADICTION_PENALTY: f32 = 0.8;

/// Confidence-bound half-width added per recorded contradiction (clamped so the
/// interval stays within `[0, 1]`). EMPIRICAL DEFAULT.
const CONTRADICTION_BOUND_HALFWIDTH: f32 = 0.1;

/// Identifies which sensing family produced a fused frame, so one
/// [`QualityScore`] can be correlated across the signal-domain fuser
/// (`multistatic.rs`) and the embedding-domain fuser (`viewpoint/fusion.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FamilyId {
    /// `ruvsense/multistatic.rs` CSI/CIR-domain fusion.
    MultistaticCsi,
    /// `ruvector/viewpoint/fusion.rs` AETHER-embedding fusion.
    ViewpointEmbedding,
}

/// Calibration epoch identifier (ADR-137 §2.1). Derived from the ADR-135
/// `BaselineCalibration` capture time plus device id; stable across reboots,
/// changes only on recalibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CalibrationId(pub u64);

/// A single piece of positive evidence supporting a fusion decision (ADR-137
/// §2.2). Each variant carries the value that crossed a threshold, not just a
/// boolean, so the witness record is reproducible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvidenceRef {
    /// The coherence-gate threshold was met. `coherence` is the value,
    /// `threshold` the configured gate.
    CoherenceGateThreshold { coherence: f32, threshold: f32 },
    /// The ADR-134 CIR dominant-tap ratio contributed to the gate. `blended`
    /// is true when it was folded into `base_coherence` (false on fallback).
    CirDominantTapRatio { ratio: f32, blended: bool },
    /// Attention-weight entropy supported a balanced (multi-node) fusion.
    WeightEntropy { normalized_entropy: f32, n_nodes: usize },
    /// An ADR-135 baseline was applied to every contributing frame at a single
    /// agreed calibration epoch before pooling.
    CalibrationApplied { calibration_id: CalibrationId, n_frames: usize },
}

/// A tolerated disagreement detected during fusion (ADR-137 §2.3). A non-empty
/// set lowers the emitted BFLD privacy class and produces a witness record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContradictionFlag {
    /// Node `capture_ns` values spread within the guard interval but beyond a
    /// stricter "comparable" sub-threshold. Carries the observed spread.
    TimestampMismatch { spread_ns: u64, soft_guard_ns: u64 },
    /// Contributing frames carried different calibration ids. `expected` is the
    /// modal id; `disagreeing` counts the disagreeing frames.
    CalibrationIdMismatch { expected: CalibrationId, disagreeing: usize },
    /// Phase alignment did not converge for at least one node.
    PhaseAlignmentFailed { node_idx: usize },
    /// A node's ADR-135 drift score conflicts with the array consensus.
    DriftProfileConflict { node_idx: usize, drift_score: f32 },
    /// Raised upstream by the ADR-138 `ArrayCoordinator`: a node's coherence
    /// dropped beyond `sigma`σ of its rolling mean.
    CoherenceDrop { node_idx: usize, sigma: f32 },
    /// Raised upstream by the ADR-138 `ArrayCoordinator`: array Geometric
    /// Diversity Index fell below the geometry-sufficiency floor.
    GeometryInsufficient { gdi: f32 },
}

/// Auditable quality record for one fused frame (ADR-137 §2.1).
///
/// Every semantic state downstream of fusion traces back to exactly one
/// `QualityScore`, which names the signal evidence (`evidence_refs`), the
/// calibration epoch (`calibration_id`), and the privacy-relevant disagreements
/// (`contradiction_flags`) that informed it.
#[derive(Debug, Clone)]
pub struct QualityScore {
    /// Which fuser produced this score.
    pub family_id: FamilyId,
    /// Capture-clock timestamp (ns) of the fused cycle (median of contributors).
    pub capture_ns: u64,
    /// The calibration epoch all contributing frames agreed on, or `None` when
    /// they disagreed (see [`ContradictionFlag::CalibrationIdMismatch`]).
    pub calibration_id: Option<CalibrationId>,
    /// Coherence in [0, 1] before any contradiction penalty is applied.
    pub base_coherence: f32,
    /// Per-contributing-node attention weight, node-index aligned. Sums to ~1.0.
    pub per_node_weights: Vec<f32>,
    /// Concrete checks that fired in support of this fusion.
    pub evidence_refs: Vec<EvidenceRef>,
    /// Tolerated-but-recorded disagreements. A non-empty set forces a BFLD
    /// privacy demotion.
    pub contradiction_flags: Vec<ContradictionFlag>,
    /// Monotonic capture-clock time at which this score was computed (ns).
    pub timestamp_computed_ns: u64,
}

impl QualityScore {
    /// True when a non-empty contradiction set must demote the BFLD privacy
    /// class (ADR-137 §2.7 → ADR-141). The fusion stage and the privacy gate
    /// both consult this so the demotion rule lives in one place.
    #[must_use]
    pub fn forces_privacy_demotion(&self) -> bool {
        !self.contradiction_flags.is_empty()
    }

    /// Coherence after the contradiction penalty: each contradiction multiplies
    /// the base coherence by 0.8, clamped to [0, 1]. This is the value the
    /// streaming engine routes/gates on.
    #[must_use]
    pub fn penalized_coherence(&self) -> f32 {
        let penalty = CONTRADICTION_PENALTY.powi(self.contradiction_flags.len() as i32);
        (self.base_coherence * penalty).clamp(0.0, 1.0)
    }
}

impl QualityScored for QualityScore {
    fn quality_score(&self) -> f32 {
        self.penalized_coherence()
    }

    fn confidence_bounds(&self) -> (f32, f32) {
        // Width grows with the number of tolerated contradictions: each adds
        // ±0.1 of uncertainty around the penalized coherence, clamped to [0,1].
        let c = self.penalized_coherence();
        let half =
            (CONTRADICTION_BOUND_HALFWIDTH * self.contradiction_flags.len() as f32).min(c.min(1.0 - c));
        ((c - half).max(0.0), (c + half).min(1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> QualityScore {
        QualityScore {
            family_id: FamilyId::MultistaticCsi,
            capture_ns: 1_000,
            calibration_id: None,
            base_coherence: 0.9,
            per_node_weights: vec![0.5, 0.5],
            evidence_refs: vec![EvidenceRef::WeightEntropy {
                normalized_entropy: 1.0,
                n_nodes: 2,
            }],
            contradiction_flags: vec![],
            timestamp_computed_ns: 1_000,
        }
    }

    #[test]
    fn no_contradiction_no_demotion() {
        let q = base();
        assert!(!q.forces_privacy_demotion());
        assert!((q.penalized_coherence() - 0.9).abs() < 1e-6);
        let (lo, hi) = q.confidence_bounds();
        assert!(lo <= hi && (lo - 0.9).abs() < 1e-6 && (hi - 0.9).abs() < 1e-6);
    }

    #[test]
    fn contradiction_penalizes_and_demotes() {
        let mut q = base();
        q.contradiction_flags.push(ContradictionFlag::TimestampMismatch {
            spread_ns: 2_000,
            soft_guard_ns: 1_000,
        });
        assert!(q.forces_privacy_demotion());
        assert!((q.penalized_coherence() - 0.72).abs() < 1e-5); // 0.9 * 0.8
        let (lo, hi) = q.confidence_bounds();
        assert!(0.0 <= lo && lo <= hi && hi <= 1.0);
    }

    #[test]
    fn quality_scored_trait_bounds_invariant() {
        let mut q = base();
        for _ in 0..5 {
            q.contradiction_flags.push(ContradictionFlag::PhaseAlignmentFailed { node_idx: 0 });
        }
        let s = q.quality_score();
        let (lo, hi) = q.confidence_bounds();
        assert!((0.0..=1.0).contains(&s));
        assert!(0.0 <= lo && lo <= hi && hi <= 1.0);
    }

    // -- ADR-154 §7.4: de-magic-constant + boundary characterization tests.

    /// De-magicked penalty/bound consts must equal the prior literals.
    #[test]
    fn fusion_quality_consts_unchanged_from_literals() {
        assert_eq!(CONTRADICTION_PENALTY, 0.8_f32);
        assert_eq!(CONTRADICTION_BOUND_HALFWIDTH, 0.1_f32);
    }

    /// Zero contradictions: penalty is `0.8^0 = 1.0` (coherence unchanged) and
    /// the confidence bounds collapse to a point. Pins the n=0 boundary.
    #[test]
    fn no_contradiction_is_identity() {
        let q = base();
        assert!(q.contradiction_flags.is_empty());
        assert!((q.penalized_coherence() - q.base_coherence).abs() < 1e-6);
        let (lo, hi) = q.confidence_bounds();
        assert!((hi - lo).abs() < 1e-6); // half-width is 0 with no contradictions
    }
}
