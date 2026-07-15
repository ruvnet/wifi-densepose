//! ADR-145 — Ablation evaluation harness with privacy-leakage + latency metrics.
//!
//! Runs the sensing pipeline under a matrix of feature combinations
//! (CSI-only / CIR-only / CSI+CIR / +Doppler / +BFLD / +UWB) and binds a metric
//! set — presence accuracy, localisation error, FP/FN, latency p50/p95,
//! privacy-leakage (membership-inference), and cross-room degradation — so every
//! pipeline change is measured, not guessed (ADR-145 §10/§14). The model runs
//! themselves are external; this module owns the deterministic metric
//! computation + the auto-report.

use core::fmt::Write as _;

/// One feature combination in the ablation matrix (ADR-145 §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureSet {
    /// CSI amplitude/phase only.
    CsiOnly,
    /// CIR taps only (ADR-134).
    CirOnly,
    /// CSI + CIR.
    CsiCir,
    /// CSI + CIR + passive Doppler.
    CsiCirDoppler,
    /// CSI + CIR + Doppler + BFLD privacy gate.
    CsiCirDopplerBfld,
    /// Full fusion including UWB range constraints (ADR-144; deferred until hw).
    FullUwb,
}

impl FeatureSet {
    /// The six-variant ablation matrix, runnable order.
    pub const MATRIX: [FeatureSet; 6] = [
        FeatureSet::CsiOnly,
        FeatureSet::CirOnly,
        FeatureSet::CsiCir,
        FeatureSet::CsiCirDoppler,
        FeatureSet::CsiCirDopplerBfld,
        FeatureSet::FullUwb,
    ];

    /// Stable label for reports.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::CsiOnly => "csi_only",
            Self::CirOnly => "cir_only",
            Self::CsiCir => "csi+cir",
            Self::CsiCirDoppler => "csi+cir+doppler",
            Self::CsiCirDopplerBfld => "csi+cir+doppler+bfld",
            Self::FullUwb => "full+uwb",
        }
    }
}

/// `(p50, p95)` percentiles of a latency sample set (ms), nearest-rank.
///
/// Non-finite samples (NaN / ±inf) are discarded before ranking. Sorting uses
/// [`f64::total_cmp`] so a stray NaN can never trigger a `partial_cmp().unwrap()`
/// panic (ADR-155 §Tier-2). If every sample is non-finite (or the slice is
/// empty), returns `(0.0, 0.0)`.
#[must_use]
pub fn latency_percentiles_ms(samples_ms: &[f64]) -> (f64, f64) {
    // Drop non-finite values: a NaN latency is meaningless and must not poison
    // the ranking or panic the sort.
    let mut s: Vec<f64> = samples_ms
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if s.is_empty() {
        return (0.0, 0.0);
    }
    s.sort_by(f64::total_cmp);
    let pick = |q: f64| {
        // Nearest-rank: ceil(q * n) - 1, clamped.
        let rank = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1;
        s[rank]
    };
    (pick(0.50), pick(0.95))
}

/// False-positive and false-negative rates from a confusion count.
#[must_use]
pub fn confusion_rates(tp: u64, fp: u64, tn: u64, fn_: u64) -> (f64, f64) {
    let fp_rate = if fp + tn == 0 {
        0.0
    } else {
        fp as f64 / (fp + tn) as f64
    };
    let fn_rate = if fn_ + tp == 0 {
        0.0
    } else {
        fn_ as f64 / (fn_ + tp) as f64
    };
    (fp_rate, fn_rate)
}

/// Privacy-leakage score in [0, 1] via a membership-inference (MIA) proxy
/// (ADR-145 §2): how separable are confidence scores of training-set *members*
/// from *non-members*? Computed as `|AUC - 0.5| * 2` — 0.0 when the two score
/// distributions are indistinguishable (no leakage), 1.0 when perfectly
/// separable (an attacker can tell who was in the training set).
#[must_use]
pub fn membership_inference_leakage(member_scores: &[f64], nonmember_scores: &[f64]) -> f64 {
    if member_scores.is_empty() || nonmember_scores.is_empty() {
        return 0.0;
    }
    // AUC = P(member_score > nonmember_score) over all pairs (+ 0.5 for ties).
    let mut wins = 0.0;
    let total = (member_scores.len() * nonmember_scores.len()) as f64;
    for &m in member_scores {
        for &n in nonmember_scores {
            if m > n {
                wins += 1.0;
            } else if (m - n).abs() < f64::EPSILON {
                wins += 0.5;
            }
        }
    }
    let auc = wins / total;
    ((auc - 0.5).abs() * 2.0).clamp(0.0, 1.0)
}

/// The metric bundle for one ablation variant (ADR-145 §2).
#[derive(Debug, Clone)]
pub struct AblationMetrics {
    /// Which feature combination was evaluated.
    pub feature_set: FeatureSet,
    /// Presence-detection accuracy in [0, 1].
    pub presence_accuracy: f64,
    /// Mean localisation error (m).
    pub localization_err_m: f64,
    /// False-positive rate.
    pub fp_rate: f64,
    /// False-negative rate.
    pub fn_rate: f64,
    /// Latency 50th percentile (ms).
    pub latency_p50_ms: f64,
    /// Latency 95th percentile (ms).
    pub latency_p95_ms: f64,
    /// Privacy leakage in [0, 1] (MIA proxy; lower is better).
    pub privacy_leakage: f64,
    /// Cross-room accuracy degradation (room_A_acc - room_B_acc), >= 0.
    pub cross_room_degradation: f64,
}

/// Raw per-variant inputs from a pipeline run; metrics are derived
/// deterministically from these.
#[derive(Debug, Clone)]
pub struct VariantRun {
    /// Feature combination evaluated.
    pub feature_set: FeatureSet,
    /// Confusion counts (tp, fp, tn, fn).
    pub confusion: (u64, u64, u64, u64),
    /// Mean localisation error (m).
    pub localization_err_m: f64,
    /// Per-frame latency samples (ms).
    pub latency_samples_ms: Vec<f64>,
    /// Member/non-member confidence scores for the MIA proxy.
    pub member_scores: Vec<f64>,
    /// Non-member confidence scores.
    pub nonmember_scores: Vec<f64>,
    /// Accuracy in the calibration room and a held-out room.
    pub room_a_accuracy: f64,
    /// Held-out room accuracy.
    pub room_b_accuracy: f64,
}

impl AblationMetrics {
    /// Derive the metric bundle from a raw variant run.
    #[must_use]
    pub fn from_run(run: &VariantRun) -> Self {
        let (tp, fp, tn, fn_) = run.confusion;
        let (fp_rate, fn_rate) = confusion_rates(tp, fp, tn, fn_);
        let total = (tp + fp + tn + fn_).max(1);
        let presence_accuracy = (tp + tn) as f64 / total as f64;
        let (p50, p95) = latency_percentiles_ms(&run.latency_samples_ms);
        Self {
            feature_set: run.feature_set,
            presence_accuracy,
            localization_err_m: run.localization_err_m,
            fp_rate,
            fn_rate,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            privacy_leakage: membership_inference_leakage(
                &run.member_scores,
                &run.nonmember_scores,
            ),
            cross_room_degradation: (run.room_a_accuracy - run.room_b_accuracy).max(0.0),
        }
    }
}

/// An ablation report over the variant matrix (ADR-145 auto-report).
#[derive(Debug, Clone, Default)]
pub struct AblationReport {
    /// Per-variant metrics in evaluation order.
    pub rows: Vec<AblationMetrics>,
}

impl AblationReport {
    /// Build from a set of variant runs.
    #[must_use]
    pub fn from_runs(runs: &[VariantRun]) -> Self {
        Self {
            rows: runs.iter().map(AblationMetrics::from_run).collect(),
        }
    }

    /// Look up a variant's metrics.
    #[must_use]
    pub fn get(&self, fs: FeatureSet) -> Option<&AblationMetrics> {
        self.rows.iter().find(|m| m.feature_set == fs)
    }

    /// Acceptance check (ADR-145 / ADR-136 AC): does CSI+CIR beat CSI-only on at
    /// least `min_wins` of {presence accuracy ↑, localisation error ↓, p95 latency ↓}?
    #[must_use]
    pub fn csi_cir_beats_csi_only(&self, min_wins: usize) -> bool {
        let (Some(a), Some(b)) = (self.get(FeatureSet::CsiOnly), self.get(FeatureSet::CsiCir))
        else {
            return false;
        };
        let wins = [
            b.presence_accuracy > a.presence_accuracy,
            b.localization_err_m < a.localization_err_m,
            b.latency_p95_ms <= a.latency_p95_ms,
        ]
        .iter()
        .filter(|w| **w)
        .count();
        wins >= min_wins
    }

    /// Deterministic markdown report (stable column/row order).
    #[must_use]
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "| variant | presence_acc | loc_err_m | fp | fn | p50_ms | p95_ms | privacy_leak | xroom_degr |"
        );
        let _ = writeln!(s, "|---|---|---|---|---|---|---|---|---|");
        for m in &self.rows {
            let _ = writeln!(
                s,
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {:.3} | {:.3} |",
                m.feature_set.label(),
                m.presence_accuracy,
                m.localization_err_m,
                m.fp_rate,
                m.fn_rate,
                m.latency_p50_ms,
                m.latency_p95_ms,
                m.privacy_leakage,
                m.cross_room_degradation,
            );
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_percentiles_nearest_rank() {
        let s: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let (p50, p95) = latency_percentiles_ms(&s);
        assert!((p50 - 50.0).abs() < 1e-9);
        assert!((p95 - 95.0).abs() < 1e-9);
        assert_eq!(latency_percentiles_ms(&[]), (0.0, 0.0));
    }

    // ADR-155 §Tier-2: a NaN in the latency samples must NOT panic the sort
    // (the old `partial_cmp().unwrap()` did) and must yield a sane percentile
    // computed over the finite values only.
    #[test]
    fn latency_percentiles_with_nan_does_not_panic() {
        let s = vec![
            10.0,
            f64::NAN,
            20.0,
            30.0,
            f64::INFINITY,
            40.0,
            f64::NEG_INFINITY,
            50.0,
        ];
        let (p50, p95) = latency_percentiles_ms(&s);
        // Finite set is [10,20,30,40,50]; nearest-rank p50=30, p95=50.
        assert!(p50.is_finite() && p95.is_finite());
        assert!((p50 - 30.0).abs() < 1e-9);
        assert!((p95 - 50.0).abs() < 1e-9);
        // All-NaN input degrades gracefully to (0, 0).
        assert_eq!(latency_percentiles_ms(&[f64::NAN, f64::NAN]), (0.0, 0.0));
    }

    #[test]
    fn confusion_rates_basic() {
        let (fp_rate, fn_rate) = confusion_rates(80, 10, 90, 20);
        assert!((fp_rate - 0.1).abs() < 1e-9); // 10 / (10+90)
        assert!((fn_rate - 0.2).abs() < 1e-9); // 20 / (20+80)
    }

    #[test]
    fn mia_leakage_zero_when_indistinguishable_high_when_separable() {
        // Identical distributions → ~no leakage.
        let same = vec![0.5, 0.6, 0.7];
        assert!(membership_inference_leakage(&same, &same) < 1e-9);
        // Perfectly separable → leakage 1.0.
        let members = vec![0.9, 0.95, 0.99];
        let nonmembers = vec![0.1, 0.2, 0.3];
        assert!((membership_inference_leakage(&members, &nonmembers) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn csi_cir_beats_csi_only_acceptance() {
        let csi_only = VariantRun {
            feature_set: FeatureSet::CsiOnly,
            confusion: (70, 15, 70, 30), // acc 0.756
            localization_err_m: 0.40,
            latency_samples_ms: vec![10.0; 10],
            member_scores: vec![0.5],
            nonmember_scores: vec![0.5],
            room_a_accuracy: 0.8,
            room_b_accuracy: 0.6,
        };
        let csi_cir = VariantRun {
            feature_set: FeatureSet::CsiCir,
            confusion: (88, 6, 90, 12), // acc 0.908
            localization_err_m: 0.22,
            latency_samples_ms: vec![11.0; 10],
            member_scores: vec![0.5],
            nonmember_scores: vec![0.5],
            room_a_accuracy: 0.85,
            room_b_accuracy: 0.80,
        };
        let runs = [csi_only, csi_cir];
        let report = AblationReport::from_runs(&runs);
        // CSI+CIR wins on presence accuracy + localisation error (2 of 3).
        assert!(report.csi_cir_beats_csi_only(2));
        let md = report.to_markdown();
        assert!(md.contains("csi_only") && md.contains("csi+cir"));
        // Deterministic: same input → byte-identical report.
        assert_eq!(md, AblationReport::from_runs(&runs).to_markdown());
    }

    #[test]
    fn matrix_has_six_variants() {
        assert_eq!(FeatureSet::MATRIX.len(), 6);
    }
}
