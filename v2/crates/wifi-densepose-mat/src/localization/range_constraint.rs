//! ADR-144 — UWB range-constraint fusion.
//!
//! A [`RangeConstraint`] is one UWB anchor↔tag range measurement. It does NOT
//! replace CSI/CIR localisation — it *constrains* a person-track estimate toward
//! the sphere of points at the measured range from a surveyed anchor, with
//! Mahalanobis gating so an inconsistent (multipath/NLOS) range is rejected
//! rather than corrupting the estimate. Anchors map to ADR-139
//! `WorldNode::ObjectAnchor` (`anchor_kind = UwbBeacon`).
//!
//! Forward-looking: no UWB hardware ships in the current device table, so this
//! module owns the domain model + the constraint-aware refinement; the UART
//! driver/parser (ADR-144 §2) lands when hardware is added.

/// One UWB range measurement from a surveyed anchor to a tag (ADR-144 §2.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeConstraint {
    /// Surveyed anchor identifier (→ ADR-139 ObjectAnchor / WorldId).
    pub anchor_id: u32,
    /// Anchor position (east, north, up) in metres.
    pub anchor_pos: [f64; 3],
    /// Measured range tag↔anchor (m).
    pub measured_range_m: f64,
    /// 1σ range uncertainty (m).
    pub uncertainty_m: f64,
    /// Link quality in [0, 1] (low ⇒ likely NLOS/multipath).
    pub signal_quality: f32,
    /// Capture-clock time (ns).
    pub at_ns: u64,
}

impl RangeConstraint {
    /// Euclidean distance from a candidate position to the anchor.
    #[must_use]
    pub fn predicted_range(&self, p: [f64; 3]) -> f64 {
        (0..3).map(|a| (p[a] - self.anchor_pos[a]).powi(2)).sum::<f64>().sqrt()
    }

    /// Signed range residual `predicted - measured` (m).
    #[must_use]
    pub fn residual(&self, p: [f64; 3]) -> f64 {
        self.predicted_range(p) - self.measured_range_m
    }

    /// Mahalanobis distance `|residual| / uncertainty` (σ units).
    #[must_use]
    pub fn mahalanobis(&self, p: [f64; 3]) -> f64 {
        let u = self.uncertainty_m.max(1e-6);
        self.residual(p).abs() / u
    }

    /// Whether a candidate position is consistent with this constraint within
    /// `gate_sigma` σ.
    #[must_use]
    pub fn is_consistent(&self, p: [f64; 3], gate_sigma: f64) -> bool {
        self.mahalanobis(p) <= gate_sigma
    }
}

/// Outcome of a constraint-aware refinement (ADR-144 §2.3).
#[derive(Debug, Clone)]
pub struct RefineResult {
    /// Refined position estimate (east, north, up) in metres.
    pub position: [f64; 3],
    /// RMS Mahalanobis residual over the *admitted* constraints after refining.
    pub rms_residual_sigma: f64,
    /// Anchor ids gated out as inconsistent at the final estimate.
    pub rejected_anchors: Vec<u32>,
    /// Number of gradient iterations performed.
    pub iterations: usize,
}

/// Constraint-aware position refiner (ADR-144 §2.3).
///
/// Minimises `Σ ((|p - aᵢ| - rᵢ) / σᵢ)²` over admitted constraints by gradient
/// descent from the CSI/CIR prior, gating out constraints beyond `gate_sigma`.
/// One-step weighting by `1/σ²` makes precise ranges dominate.
#[derive(Debug, Clone)]
pub struct RangeConstraintFusion {
    /// Mahalanobis gate (σ) for admitting a constraint.
    pub gate_sigma: f64,
    /// Gradient step size (m per unit gradient).
    pub step: f64,
    /// Maximum iterations.
    pub max_iters: usize,
    /// Convergence threshold on the position update norm (m).
    pub tol_m: f64,
}

impl Default for RangeConstraintFusion {
    fn default() -> Self {
        Self { gate_sigma: 3.0, step: 1.0, max_iters: 200, tol_m: 1e-4 }
    }
}

impl RangeConstraintFusion {
    /// Refine `prior` (the CSI/CIR estimate) against the range constraints.
    /// Constraints inconsistent at the *prior* are gated out up front so a
    /// gross outlier cannot drag the solution.
    #[must_use]
    pub fn refine(&self, prior: [f64; 3], constraints: &[RangeConstraint]) -> RefineResult {
        // Admit constraints consistent at the prior; record the rest.
        let mut admitted: Vec<&RangeConstraint> = Vec::new();
        let mut rejected_anchors = Vec::new();
        for c in constraints {
            if c.is_consistent(prior, self.gate_sigma) {
                admitted.push(c);
            } else {
                rejected_anchors.push(c.anchor_id);
            }
        }

        let mut p = prior;
        let mut iterations = 0;
        if !admitted.is_empty() {
            for _ in 0..self.max_iters {
                iterations += 1;
                // Gradient of Σ w·(d - r)² w.r.t. p, with w = 1/σ². Normalising
                // by the total weight (≈ the Hessian's dominant eigenvalue / 2)
                // turns the descent into a Newton-like step that is invariant to
                // the absolute weight scale — otherwise a small σ (large w) makes
                // a plain gradient step overshoot and diverge.
                let mut grad = [0.0f64; 3];
                let mut sum_w = 0.0f64;
                for c in &admitted {
                    let d = c.predicted_range(p).max(1e-9);
                    let w = 1.0 / (c.uncertainty_m.max(1e-6)).powi(2);
                    sum_w += w;
                    let coeff = 2.0 * w * (d - c.measured_range_m) / d;
                    for a in 0..3 {
                        grad[a] += coeff * (p[a] - c.anchor_pos[a]);
                    }
                }
                let scale = self.step / (2.0 * sum_w.max(1e-12));
                let mut upd_norm = 0.0;
                for a in 0..3 {
                    let delta = scale * grad[a];
                    p[a] -= delta;
                    upd_norm += delta * delta;
                }
                if upd_norm.sqrt() < self.tol_m {
                    break;
                }
            }
        }

        // RMS Mahalanobis residual over admitted constraints at the solution.
        let rms_residual_sigma = if admitted.is_empty() {
            f64::INFINITY
        } else {
            let ss: f64 = admitted.iter().map(|c| c.mahalanobis(p).powi(2)).sum();
            (ss / admitted.len() as f64).sqrt()
        };

        RefineResult { position: p, rms_residual_sigma, rejected_anchors, iterations }
    }

    /// Associate a constraint to the most consistent of several candidate track
    /// positions (ADR-144 §2 — disambiguate which track a range belongs to).
    /// Returns the index of the track with the smallest Mahalanobis distance
    /// that is also within the gate, or `None` if none qualify.
    #[must_use]
    pub fn associate(&self, tracks: &[[f64; 3]], c: &RangeConstraint) -> Option<usize> {
        tracks
            .iter()
            .enumerate()
            .map(|(i, &t)| (i, c.mahalanobis(t)))
            .filter(|(_, m)| *m <= self.gate_sigma)
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rc(id: u32, pos: [f64; 3], range: f64) -> RangeConstraint {
        RangeConstraint {
            anchor_id: id,
            anchor_pos: pos,
            measured_range_m: range,
            uncertainty_m: 0.1,
            signal_quality: 0.9,
            at_ns: 0,
        }
    }

    #[test]
    fn refine_converges_to_true_point() {
        // True tag at (2, 2, 0); 3 anchors with exact ranges.
        let truth: [f64; 3] = [2.0, 2.0, 0.0];
        let anchors: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        // UWB σ = 0.3 m (gate 0.9 m): the CSI prior must already be roughly in
        // the right place — UWB refines it, it does not localise from scratch.
        let constraints: Vec<RangeConstraint> = anchors
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let r = ((truth[0] - a[0]).powi(2) + (truth[1] - a[1]).powi(2) + (truth[2] - a[2]).powi(2)).sqrt();
                RangeConstraint { uncertainty_m: 0.3, ..rc(i as u32, a, r) }
            })
            .collect();

        let fusion = RangeConstraintFusion::default();
        // Biased CSI prior 0.7 m off-truth, within the 0.9 m gate.
        let res = fusion.refine([1.5, 1.5, 0.0], &constraints);
        let err = ((res.position[0] - 2.0).powi(2) + (res.position[1] - 2.0).powi(2)).sqrt();
        assert!(err < 0.05, "refined within 5 cm of truth, got err={err}");
        assert!(res.rejected_anchors.is_empty());
        assert!(res.rms_residual_sigma < 1.0);
    }

    #[test]
    fn inconsistent_constraint_is_gated_out() {
        // Prior near truth (2,2); a bogus 100 m range from anchor 9 is rejected.
        let mut constraints = vec![
            rc(0, [0.0, 0.0, 0.0], 2.83),
            rc(1, [4.0, 0.0, 0.0], 2.83),
        ];
        constraints.push(rc(9, [0.0, 4.0, 0.0], 100.0)); // absurd
        let fusion = RangeConstraintFusion::default();
        let res = fusion.refine([2.0, 2.0, 0.0], &constraints);
        assert!(res.rejected_anchors.contains(&9), "absurd range gated out");
    }

    #[test]
    fn consistency_gate_and_residual() {
        let c = rc(0, [0.0, 0.0, 0.0], 5.0);
        // Point at distance 5.0 → zero residual, consistent.
        assert!(c.residual([5.0, 0.0, 0.0]).abs() < 1e-9);
        assert!(c.is_consistent([5.0, 0.0, 0.0], 3.0));
        // Point at distance 5.5 → 0.5 m / 0.1 = 5σ → inconsistent at 3σ gate.
        assert!(!c.is_consistent([5.5, 0.0, 0.0], 3.0));
        assert!((c.mahalanobis([5.5, 0.0, 0.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn associate_picks_nearest_consistent_track() {
        let c = rc(0, [0.0, 0.0, 0.0], 3.0); // anchor at origin, range 3
        let fusion = RangeConstraintFusion::default();
        // Track A at distance 3 (consistent), B at distance 8 (way off).
        let tracks = [[3.0, 0.0, 0.0], [8.0, 0.0, 0.0]];
        assert_eq!(fusion.associate(&tracks, &c), Some(0));
        // If no track is within gate, None.
        let far = [[20.0, 0.0, 0.0], [25.0, 0.0, 0.0]];
        assert_eq!(fusion.associate(&far, &c), None);
    }
}
