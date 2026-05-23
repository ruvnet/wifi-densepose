//! Multi-node fusion — combine N per-node count distributions into one.
//!
//! v0.1.0 ships **confidence-weighted log-sum** (Bayesian product of expert
//! distributions): the more confident a node, the more its distribution
//! shapes the fused output. With one node the fusion is a no-op; with N
//! nodes uncertainty can only go down (or stay equal), never up.
//!
//! v0.2.0 will add a **Stoer-Wagner min-cut upper bound** on the fused
//! distribution — see ADR-103 §"Multi-node fusion". That requires
//! `ruvector-mincut` as a workspace dep on this crate; it's stubbed below
//! behind `fuse_with_mincut_clip()` so callers can opt in once the dep
//! lands and the min-cut graph builder for our subcarrier feature
//! similarities is ready.

use crate::inference::{CountPrediction, COUNT_CLASSES};

/// Confidence-weighted log-sum of per-node count distributions.
///
/// For each class k, computes `log p_fused(k) = Σ_n c_n · log p_n(k)`,
/// then re-normalises. The fused `confidence` is the **maximum** per-node
/// confidence rather than the average — having at least one confident
/// observation is worth more than many low-confidence ones.
///
/// Edge cases:
/// * Empty input → 1-person, 0-confidence default (matches the stub).
/// * Single input → returned as-is (defined behaviour, no-op).
/// * Zero confidences across all nodes → unweighted log-sum.
pub fn fuse_confidence_weighted(preds: &[CountPrediction]) -> CountPrediction {
    if preds.is_empty() {
        let mut probs = [0.0_f32; COUNT_CLASSES];
        probs[1] = 1.0;
        return CountPrediction {
            probs,
            confidence: 0.0,
        };
    }
    if preds.len() == 1 {
        return preds[0].clone();
    }

    // Compute weights c_n with a small floor so zero-confidence nodes still
    // contribute (log-of-zero would otherwise blow the math up).
    const EPS_CONF: f32 = 1e-3;
    let weights: Vec<f32> = preds.iter().map(|p| p.confidence.max(EPS_CONF)).collect();
    let weight_sum: f32 = weights.iter().sum();

    // Log-sum.
    let mut log_p = [0.0_f32; COUNT_CLASSES];
    for (pred, &w) in preds.iter().zip(weights.iter()) {
        for (lp, &prob) in log_p.iter_mut().zip(pred.probs.iter()).take(COUNT_CLASSES) {
            let p = prob.max(1e-9); // floor to avoid log(0)
            *lp += (w / weight_sum) * p.ln();
        }
    }

    // Subtract max for numerical stability, exponentiate, renormalise.
    let m = log_p.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut p = [0.0_f32; COUNT_CLASSES];
    let mut s = 0.0_f32;
    for (pk, &lp) in p.iter_mut().zip(log_p.iter()) {
        *pk = (lp - m).exp();
        s += *pk;
    }
    if s > 0.0 {
        for pk in p.iter_mut() {
            *pk /= s;
        }
    } else {
        // Pathological — fall back to uniform.
        for pk in p.iter_mut() {
            *pk = 1.0 / COUNT_CLASSES as f32;
        }
    }

    let conf = preds.iter().map(|x| x.confidence).fold(0.0_f32, f32::max);
    CountPrediction {
        probs: p,
        confidence: conf,
    }
}

/// **Stoer-Wagner-clipped fusion** — v0.2.0 hook.
///
/// Takes the same per-node predictions plus a **max-distinct-persons**
/// upper bound derived from the subcarrier-similarity graph's min-cut.
/// Clips the fused distribution to `{0..=max}` and re-normalises.
///
/// Live `ruvector_mincut` integration lands in a follow-up PR; this entry
/// point is here so the runtime can wire to it without an API break.
pub fn fuse_with_mincut_clip(preds: &[CountPrediction], max_distinct: usize) -> CountPrediction {
    let mut fused = fuse_confidence_weighted(preds);
    let max_idx = max_distinct.min(COUNT_CLASSES - 1);
    let mut leak = 0.0_f32;
    for k in (max_idx + 1)..COUNT_CLASSES {
        leak += fused.probs[k];
        fused.probs[k] = 0.0;
    }
    if leak > 0.0 {
        // Re-normalise the surviving prefix.
        let sum: f32 = fused.probs[..=max_idx].iter().sum();
        if sum > 0.0 {
            for k in 0..=max_idx {
                fused.probs[k] /= sum;
            }
        } else {
            // All mass was above the cap — degenerate; place mass at the cap.
            fused.probs[max_idx] = 1.0;
        }
    }
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn pred(probs: [f32; 8], conf: f32) -> CountPrediction {
        CountPrediction {
            probs,
            confidence: conf,
        }
    }

    #[test]
    fn empty_returns_one_person_default() {
        let p = fuse_confidence_weighted(&[]);
        assert_eq!(p.argmax(), 1);
        assert_eq!(p.confidence, 0.0);
    }

    #[test]
    fn single_input_is_passthrough() {
        let probs = [0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0];
        let p = fuse_confidence_weighted(&[pred(probs, 0.8)]);
        assert_eq!(p.argmax(), 2);
        assert_relative_eq!(p.confidence, 0.8, max_relative = 1e-6);
    }

    #[test]
    fn two_agreeing_nodes_sharpen_the_peak() {
        // Both nodes vote 2 with moderate spread. Fusion should sharpen.
        let probs = [0.05, 0.15, 0.60, 0.15, 0.05, 0.0, 0.0, 0.0];
        let fused = fuse_confidence_weighted(&[pred(probs, 0.7), pred(probs, 0.7)]);
        assert_eq!(fused.argmax(), 2);
        assert!(
            fused.probs[2] >= probs[2],
            "expected fusion to sharpen the peak: pre={} post={}",
            probs[2],
            fused.probs[2]
        );
    }

    #[test]
    fn high_confidence_node_overrides_low_confidence_disagreement() {
        let strong = [0.0, 0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]; // says 1
        let weak = [0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]; // weak, says 7
        let fused = fuse_confidence_weighted(&[pred(strong, 0.95), pred(weak, 0.05)]);
        assert_eq!(fused.argmax(), 1, "high-confidence vote should win");
    }

    #[test]
    fn fusion_preserves_normalisation() {
        let a = [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02];
        let b = [0.05, 0.25, 0.35, 0.20, 0.10, 0.03, 0.01, 0.01];
        let fused = fuse_confidence_weighted(&[pred(a, 0.5), pred(b, 0.5)]);
        let s: f32 = fused.probs.iter().sum();
        assert_relative_eq!(s, 1.0, max_relative = 1e-5);
    }

    #[test]
    fn mincut_clip_caps_distribution_at_max_distinct() {
        let probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.2]; // mass on 5,6,7
        let clipped = fuse_with_mincut_clip(&[pred(probs, 0.9)], 4);
        // Anything above 4 must be zero
        for k in 5..8 {
            assert_eq!(clipped.probs[k], 0.0, "class {} should be clipped to 0", k);
        }
        // What's left has to renormalise to sum to 1 — even though pre-clip
        // mass below 4 was zero, the degenerate fallback places mass at the cap.
        let s: f32 = clipped.probs.iter().sum();
        assert_relative_eq!(s, 1.0, max_relative = 1e-5);
        assert_eq!(clipped.argmax(), 4);
    }

    #[test]
    fn p95_range_is_inclusive_and_covers_at_least_95pct() {
        let probs = [0.05, 0.6, 0.25, 0.05, 0.03, 0.01, 0.005, 0.005];
        let p = pred(probs, 0.9);
        let (lo, hi) = p.p95_range();
        assert!(
            lo <= 1 && hi >= 1,
            "mode (1) must be inside [{}, {}]",
            lo,
            hi
        );
        let mass: f32 = probs[lo..=hi].iter().sum();
        assert!(
            mass >= 0.95,
            "[{}, {}] only covers {:.3}, need >= 0.95",
            lo,
            hi,
            mass
        );
    }
}
