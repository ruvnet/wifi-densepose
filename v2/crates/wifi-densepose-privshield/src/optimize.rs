//! Hyper-optimization of the shield's operating point.
//!
//! The reference crate shipped a hand-picked shield config. This module finds
//! the *optimal* one deterministically, and — crucially — proves the optimum is
//! robust rather than tuned to one attacker or one identity count:
//!
//! - [`optimal_feedback_bits`] finds the throughput-maximizing feedback
//!   resolution, exploiting the interior optimum the [`crate::throughput`] model
//!   exposes (residual falls with bits, airtime rises).
//! - [`min_givens_passes`] finds the **smallest** rotation-mixing budget that
//!   still drives re-identification into the chance band — checked against
//!   *every* attacker [`Metric`] and *every* identity count in a robustness set,
//!   so the answer is the minimum that survives the hardest case, not the
//!   easiest.
//! - [`pareto_frontier`] enumerates the non-dominated (privacy, throughput)
//!   points for documentation and inspection.
//! - [`hyper_optimize`] combines the two into a ready-to-ship [`ShieldConfig`]
//!   plus the verifying [`ExperimentReport`].
//!
//! Optimizing over both metrics and multiple `N` is the point: if the collapse
//! held only for Euclidean at N=16, it would be a classifier artifact. It holds
//! across the set because a session-fresh secret rotation removes stable
//! identity information from the *signal*.

use crate::attacker::Metric;
use crate::experiment::{run, ExperimentConfig, ExperimentReport};
use crate::protector::ShieldConfig;

/// Attacker metrics the optimizer must satisfy simultaneously.
pub const ROBUSTNESS_METRICS: [Metric; 2] = [Metric::Euclidean, Metric::Cosine];

/// Identity counts the optimizer must satisfy simultaneously. Larger `N` has a
/// lower chance floor, so it is the harder collapse target.
pub const ROBUSTNESS_IDENTITIES: [usize; 2] = [16, 32];

/// Candidate Givens-pass budgets, ascending. The optimizer returns the first
/// that collapses re-ID across the whole robustness set.
pub const PASS_CANDIDATES: [usize; 12] = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 112];

/// Per-angle feedback resolutions 802.11 compressed beamforming actually uses
/// (ψ/φ are quantized to roughly 5–9 bits). The shipped shield picks the
/// throughput-best value from this *spec-allowed* set, not the unconstrained
/// model optimum, so the config stays standards-faithful.
pub const ALLOWED_FEEDBACK_BITS: [u32; 3] = [5, 7, 9];

/// Safety margin applied to the proven-minimum pass budget. Rotation mixing is
/// keyed (derived from the shared link secret, never signaled), so extra passes
/// cost compute but **no** throughput — we spend a 2× margin on privacy for
/// free.
pub const PRIVACY_MARGIN_FACTOR: usize = 2;

/// Run one experiment variant with the given knobs, holding everything else at
/// `base`.
fn run_variant(
    base: &ExperimentConfig,
    passes: usize,
    bits: u32,
    metric: Metric,
    identities: usize,
) -> ExperimentReport {
    let mut cfg = base.clone();
    cfg.shield = ShieldConfig {
        givens_passes: passes,
        feedback_bits: bits,
        ..base.shield.clone()
    };
    cfg.scene.identities = identities;
    cfg.attacker_metric = metric;
    run(&cfg)
}

/// Throughput of the base link at a given feedback resolution.
fn throughput_at_bits(base: &ExperimentConfig, bits: u32) -> f64 {
    base.link.throughput_ratio(&ShieldConfig {
        feedback_bits: bits,
        ..base.shield.clone()
    })
}

/// Find the throughput-maximizing `feedback_bits` in `1..=max_bits`
/// (unconstrained model optimum). Returns `(bits, throughput_ratio)`.
#[must_use]
pub fn optimal_feedback_bits(base: &ExperimentConfig, max_bits: u32) -> (u32, f64) {
    (1..=max_bits)
        .map(|bits| (bits, throughput_at_bits(base, bits)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((base.shield.feedback_bits, 0.0))
}

/// Find the throughput-maximizing feedback resolution within the spec-allowed
/// set [`ALLOWED_FEEDBACK_BITS`]. This is what the shipped shield uses.
#[must_use]
pub fn spec_optimal_feedback_bits(base: &ExperimentConfig) -> (u32, f64) {
    ALLOWED_FEEDBACK_BITS
        .iter()
        .map(|&bits| (bits, throughput_at_bits(base, bits)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

/// Does `passes` collapse re-ID into the chance band for *every* metric and
/// *every* identity count in the robustness set?
#[must_use]
pub fn passes_collapse_robustly(base: &ExperimentConfig, passes: usize, bits: u32) -> bool {
    for &n in &ROBUSTNESS_IDENTITIES {
        for &m in &ROBUSTNESS_METRICS {
            if !run_variant(base, passes, bits, m, n).drives_to_chance() {
                return false;
            }
        }
    }
    true
}

/// Smallest Givens-pass budget from [`PASS_CANDIDATES`] that collapses re-ID
/// robustly, or `None` if even the largest candidate fails.
#[must_use]
pub fn min_givens_passes(base: &ExperimentConfig, bits: u32) -> Option<usize> {
    PASS_CANDIDATES
        .iter()
        .copied()
        .find(|&p| passes_collapse_robustly(base, p, bits))
}

/// One point on the privacy–throughput tradeoff.
#[derive(Debug, Clone, PartialEq)]
pub struct ParetoPoint {
    /// Givens-pass budget.
    pub givens_passes: usize,
    /// Feedback resolution in bits.
    pub feedback_bits: u32,
    /// Worst-case (highest) re-ID accuracy over the robustness metrics at the
    /// base identity count.
    pub worst_reid: f32,
    /// Modeled throughput ratio.
    pub throughput_ratio: f64,
    /// Whether this point collapses re-ID robustly (all metrics, all N).
    pub robustly_private: bool,
}

/// Enumerate the non-dominated (lower re-ID, higher throughput) points over a
/// grid of pass budgets and feedback resolutions.
#[must_use]
pub fn pareto_frontier(base: &ExperimentConfig, max_bits: u32) -> Vec<ParetoPoint> {
    let mut points: Vec<ParetoPoint> = Vec::new();
    for &passes in &PASS_CANDIDATES {
        for bits in 1..=max_bits {
            // Worst-case re-ID over metrics at the base identity count.
            let worst_reid = ROBUSTNESS_METRICS
                .iter()
                .map(|&m| {
                    run_variant(base, passes, bits, m, base.scene.identities).accuracy_shield_on
                })
                .fold(0.0_f32, f32::max);
            let shield = ShieldConfig {
                givens_passes: passes,
                feedback_bits: bits,
                ..base.shield.clone()
            };
            points.push(ParetoPoint {
                givens_passes: passes,
                feedback_bits: bits,
                worst_reid,
                throughput_ratio: base.link.throughput_ratio(&shield),
                robustly_private: passes_collapse_robustly(base, passes, bits),
            });
        }
    }
    // Keep only non-dominated points: no other point has both lower-or-equal
    // re-ID and higher-or-equal throughput while being strictly better in one.
    points
        .iter()
        .filter(|p| {
            !points.iter().any(|q| {
                let better_or_eq =
                    q.worst_reid <= p.worst_reid && q.throughput_ratio >= p.throughput_ratio;
                let strictly_better =
                    q.worst_reid < p.worst_reid || q.throughput_ratio > p.throughput_ratio;
                better_or_eq && strictly_better
            })
        })
        .cloned()
        .collect()
}

/// The chosen optimum plus the report that verifies it.
#[derive(Debug, Clone)]
pub struct HyperOptimized {
    /// The optimized, ready-to-ship shield configuration.
    pub shield: ShieldConfig,
    /// Minimum Givens passes that collapses re-ID robustly (before the margin).
    pub min_passes: usize,
    /// Shipped Givens passes = `min_passes` grown by [`PRIVACY_MARGIN_FACTOR`].
    pub shipped_passes: usize,
    /// Unconstrained throughput-optimal feedback resolution (a research point).
    pub model_optimal_bits: u32,
    /// Spec-allowed throughput-optimal resolution (what the shield ships with).
    pub spec_optimal_bits: u32,
    /// The verifying experiment at the base identity count.
    pub report: ExperimentReport,
}

/// Smallest pass candidate that is at least `target`.
fn ceil_to_candidate(target: usize) -> usize {
    PASS_CANDIDATES
        .iter()
        .copied()
        .find(|&p| p >= target)
        .unwrap_or_else(|| *PASS_CANDIDATES.last().unwrap())
}

/// Find the optimal shield: the spec-allowed throughput-optimal feedback
/// resolution, and the minimum rotation-mixing budget that collapses re-ID
/// robustly, grown by a free privacy margin. Deterministic and idempotent — the
/// shipped [`ShieldConfig::default`] is exactly this function's output on the
/// default base (asserted in tests).
#[must_use]
pub fn hyper_optimize(base: &ExperimentConfig) -> HyperOptimized {
    let (model_optimal_bits, _) = optimal_feedback_bits(base, 12);
    let (spec_optimal_bits, _) = spec_optimal_feedback_bits(base);

    let min_passes = min_givens_passes(base, spec_optimal_bits)
        .unwrap_or_else(|| *PASS_CANDIDATES.last().unwrap());
    let shipped_passes = ceil_to_candidate(min_passes * PRIVACY_MARGIN_FACTOR);

    let shield = ShieldConfig {
        givens_passes: shipped_passes,
        feedback_bits: spec_optimal_bits,
        ..base.shield.clone()
    };
    let mut cfg = base.clone();
    cfg.shield = shield.clone();
    let report = run(&cfg);

    HyperOptimized {
        shield,
        min_passes,
        shipped_passes,
        model_optimal_bits,
        spec_optimal_bits,
        report,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_optimal_bits_is_interior() {
        let (bits, ratio) = optimal_feedback_bits(&ExperimentConfig::default(), 12);
        assert!(bits > 1 && bits < 12, "optimum at edge: {bits}");
        assert!(ratio > 0.95);
    }

    #[test]
    fn spec_optimal_bits_is_the_low_res_end() {
        // Within {5,7,9}, lower resolution wins because the receiver compensates
        // the keyed rotation, so extra bits mostly buy airtime.
        let (bits, _) = spec_optimal_feedback_bits(&ExperimentConfig::default());
        assert_eq!(bits, 5);
    }

    #[test]
    fn min_passes_is_below_the_original_default() {
        // The original hand-picked default was 112 passes. The optimizer proves
        // far fewer suffice — the "we over-provisioned" finding.
        let (bits, _) = spec_optimal_feedback_bits(&ExperimentConfig::default());
        let p = min_givens_passes(&ExperimentConfig::default(), bits).expect("collapses");
        assert!(p < 112, "min passes {p} should be below the old 112");
        assert!(p >= 2);
    }

    #[test]
    fn shipped_default_equals_optimizer_output() {
        // The crate's default shield IS the optimizer's recommendation — they
        // cannot silently drift apart.
        let opt = hyper_optimize(&ExperimentConfig::default());
        assert_eq!(
            opt.shield.givens_passes,
            ShieldConfig::default().givens_passes
        );
        assert_eq!(
            opt.shield.feedback_bits,
            ShieldConfig::default().feedback_bits
        );
        assert!(opt.report.passed(), "{:#?}", opt.report);
    }

    #[test]
    fn optimum_collapses_under_both_metrics_and_larger_n() {
        let opt = hyper_optimize(&ExperimentConfig::default());
        assert!(passes_collapse_robustly(
            &ExperimentConfig::default(),
            opt.shipped_passes,
            opt.spec_optimal_bits
        ));
    }

    #[test]
    fn frontier_is_non_empty_and_deterministic() {
        // Small grid keeps this fast; the frontier logic is grid-size agnostic.
        let base = ExperimentConfig::default();
        let a = pareto_frontier(&base, 3);
        let b = pareto_frontier(&base, 3);
        assert!(!a.is_empty());
        assert_eq!(a, b);
    }
}
