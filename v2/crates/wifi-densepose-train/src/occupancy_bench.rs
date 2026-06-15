//! Falsifiable occupancy / presence benchmark over labeled CSI sequences.
//!
//! The beyond-SOTA system review found that "beyond SOTA" was *unfalsifiable*:
//! no real-CSI ground-truth benchmark existed, and the eval pyramid (doc 03)
//! lists the field's recurring measurement frauds — subject leakage between
//! train/test, per-environment overfitting, and **mock-mode contamination**
//! (CLAUDE.md: mock missed a real Kconfig bug).
//!
//! This module makes the claim falsifiable. It **grades** predictions against
//! ground truth (it does not run a model — keeping the eval crate light and the
//! scoring model-agnostic), and it enforces, *structurally*, the discipline
//! that prevents overclaiming:
//!
//! 1. **No SOTA claim on non-measured data.** A dataset is tagged
//!    [`DataProvenance`]; only [`DataProvenance::Measured`] can release a claim.
//!    Synthetic/Mock data can still be scored (useful for CI/regression) but the
//!    [`ClaimGate`] returns [`NO_CLAIM`] — you cannot accidentally publish a
//!    "beyond SOTA" number computed on simulated CSI.
//! 2. **No leaky splits.** [`EvalSplit::validate`] refuses a split where any
//!    subject *or* environment id appears in both train and test.
//! 3. **Pre-registered thresholds + bootstrap CI.** The gate compares the
//!    *lower* bound of a deterministic 95% bootstrap CI, not the point estimate,
//!    so a lucky small-sample result cannot pass.
//! 4. **No degenerate test sets.** The test set must contain *both* truth
//!    classes (present-rate ≥ `min_positive_rate`, and at least one absent
//!    sample), with its own failure flag — an all-absent set plus an
//!    always-absent predictor must never release a claim. Vacuous F1 (no
//!    positives anywhere in the confusion) scores **0.0**, never 1.0.
//!
//! The harness is the same shape as the `ruview-gamma` acceptance gate: a single
//! `claim_allowed` invariant, and the claim string is unreadable except through
//! the gate.

use std::collections::BTreeSet;

/// Provenance of the labeled data a benchmark runs on. Gates whether a SOTA
/// claim is releasable at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataProvenance {
    /// Real CSI captured from hardware with independent ground truth. The only
    /// provenance that can release a claim.
    Measured,
    /// Deterministic synthetic CSI (e.g. the proof generator). Scorable for
    /// regression, never claimable.
    Synthetic,
    /// Mock/stub data path. Scorable, never claimable — mock contamination is a
    /// documented failure mode (CLAUDE.md Kconfig-bug lesson).
    Mock,
}

impl DataProvenance {
    /// Whether data of this provenance may ever release a SOTA/accuracy claim.
    pub fn is_claimable(self) -> bool {
        matches!(self, DataProvenance::Measured)
    }

    /// Stable lowercase tag for logs/reports.
    pub fn tag(self) -> &'static str {
        match self {
            DataProvenance::Measured => "measured",
            DataProvenance::Synthetic => "synthetic",
            DataProvenance::Mock => "mock",
        }
    }
}

/// The research-only string returned when a claim is withheld.
pub const NO_CLAIM: &str = "research use only — not claimable (non-measured data, leaky split, or unmet thresholds)";

/// Ground-truth / predicted occupancy for one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Occupancy {
    /// Whether any person is present.
    pub present: bool,
    /// Estimated number of people.
    pub person_count: u32,
}

impl Occupancy {
    /// Construct an occupancy label.
    pub fn new(present: bool, person_count: u32) -> Self {
        Self { present, person_count }
    }
}

/// One labeled, attributed evaluation sample: who/where it came from (for
/// leakage checks) and the ground-truth vs predicted occupancy.
#[derive(Debug, Clone)]
pub struct LabeledSample {
    /// Subject identity (for subject-disjoint split enforcement).
    pub subject_id: String,
    /// Capture environment/room (for environment-disjoint split enforcement).
    pub environment_id: String,
    /// Ground-truth occupancy.
    pub truth: Occupancy,
    /// Model-predicted occupancy.
    pub predicted: Occupancy,
}

/// A train/test split by sample index, with leakage validation.
#[derive(Debug, Clone)]
pub struct EvalSplit {
    /// Indices of training samples.
    pub train_idx: Vec<usize>,
    /// Indices of held-out test samples (graded).
    pub test_idx: Vec<usize>,
}

/// Why a split is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitError {
    /// A subject id appears in both train and test (subject leakage).
    SubjectLeakage(String),
    /// An environment id appears in both (per-environment overfitting risk).
    EnvironmentLeakage(String),
    /// An index is out of range for the sample set.
    IndexOutOfRange(usize),
    /// The test set is empty.
    EmptyTest,
}

impl EvalSplit {
    /// Validate the split against `samples`: every test subject/environment must
    /// be **disjoint** from the training set. This is the single most common
    /// way WiFi-sensing papers overstate accuracy (doc 03).
    pub fn validate(&self, samples: &[LabeledSample]) -> Result<(), SplitError> {
        if self.test_idx.is_empty() {
            return Err(SplitError::EmptyTest);
        }
        for &i in self.train_idx.iter().chain(&self.test_idx) {
            if i >= samples.len() {
                return Err(SplitError::IndexOutOfRange(i));
            }
        }
        let train_subjects: BTreeSet<&str> =
            self.train_idx.iter().map(|&i| samples[i].subject_id.as_str()).collect();
        let train_envs: BTreeSet<&str> =
            self.train_idx.iter().map(|&i| samples[i].environment_id.as_str()).collect();
        for &i in &self.test_idx {
            let s = &samples[i];
            if train_subjects.contains(s.subject_id.as_str()) {
                return Err(SplitError::SubjectLeakage(s.subject_id.clone()));
            }
            if train_envs.contains(s.environment_id.as_str()) {
                return Err(SplitError::EnvironmentLeakage(s.environment_id.clone()));
            }
        }
        Ok(())
    }
}

/// Pre-registered acceptance thresholds (doc 03 acceptance table). Defaults are
/// deliberately conservative; tighten per capability axis.
#[derive(Debug, Clone, Copy)]
pub struct BenchmarkCriteria {
    /// Minimum presence F1 (lower CI bound must clear this).
    pub min_presence_f1: f64,
    /// Maximum person-count mean absolute error.
    pub max_count_mae: f64,
    /// Minimum test samples to grade at all (small-N guard).
    pub min_test_samples: usize,
    /// Minimum fraction of ground-truth **present** samples in the test set
    /// (degenerate-test-set guard, review finding 2): an all-absent (or
    /// nearly all-absent) test set makes presence F1 vacuous — an
    /// always-absent predictor must not be able to release a claim. The gate
    /// additionally requires at least one ground-truth *absent* sample, so
    /// both classes must be represented.
    pub min_positive_rate: f64,
    /// Bootstrap resamples for the CI.
    pub bootstrap_iters: usize,
    /// Deterministic bootstrap seed.
    pub bootstrap_seed: u64,
}

impl Default for BenchmarkCriteria {
    fn default() -> Self {
        Self {
            min_presence_f1: 0.9,
            max_count_mae: 0.5,
            min_test_samples: 30,
            min_positive_rate: 0.1,
            bootstrap_iters: 1000,
            bootstrap_seed: 42,
        }
    }
}

/// The graded result.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkReport {
    /// Data provenance tag (`measured`/`synthetic`/`mock`).
    pub provenance_tag: &'static str,
    /// Number of held-out test samples graded.
    pub n_test: usize,
    /// Presence accuracy (TP+TN)/N.
    pub presence_accuracy: f64,
    /// Presence F1 (point estimate).
    pub presence_f1: f64,
    /// 95% bootstrap CI for presence F1 (lower, upper).
    pub presence_f1_ci: (f64, f64),
    /// Fraction of samples with an exactly correct person count.
    pub count_exact_match: f64,
    /// Person-count mean absolute error.
    pub count_mae: f64,
    /// Data is measured (claimable provenance).
    pub provenance_pass: bool,
    /// Split is leak-free (subject- and environment-disjoint).
    pub split_pass: bool,
    /// Presence F1 CI-lower clears the threshold.
    pub presence_pass: bool,
    /// Count MAE within the threshold.
    pub count_pass: bool,
    /// Test set is large enough to grade.
    pub sample_size_pass: bool,
    /// Test set contains both truth classes with at least `min_positive_rate`
    /// present-true samples (degenerate test set ⇒ fail, own failure reason).
    pub class_balance_pass: bool,
    /// All six criteria pass.
    pub overall_pass: bool,
    /// The released claim string (or [`NO_CLAIM`]).
    pub released_claim: String,
}

impl BenchmarkReport {
    /// The released claim string (program claim on pass, [`NO_CLAIM`] on fail).
    pub fn claim(&self) -> &str {
        &self.released_claim
    }
}

/// **The single claim invariant.** A SOTA/accuracy claim is releasable only when
/// the data is measured, the split is leak-free, the sample is large enough,
/// the test set is non-degenerate (both classes represented), and both the
/// (CI-lower) presence F1 and the count MAE clear their thresholds.
#[inline]
pub fn claim_allowed(
    provenance_pass: bool,
    split_pass: bool,
    sample_size_pass: bool,
    class_balance_pass: bool,
    presence_pass: bool,
    count_pass: bool,
) -> bool {
    provenance_pass
        && split_pass
        && sample_size_pass
        && class_balance_pass
        && presence_pass
        && count_pass
}

/// Grade the test split of `samples` under `criteria`.
///
/// `split` is validated first; on any leakage the report is marked invalid and
/// the claim is withheld (metrics are still computed for visibility).
pub fn evaluate(
    samples: &[LabeledSample],
    provenance: DataProvenance,
    split: &EvalSplit,
    criteria: &BenchmarkCriteria,
) -> BenchmarkReport {
    let split_pass = split.validate(samples).is_ok();
    let test: Vec<&LabeledSample> = split
        .test_idx
        .iter()
        .filter(|&&i| i < samples.len())
        .map(|&i| &samples[i])
        .collect();
    let n_test = test.len();

    // Presence confusion counts.
    let (mut tp, mut fp, mut tn, mut fn_) = (0u64, 0u64, 0u64, 0u64);
    let mut count_abs_err_sum = 0.0;
    let mut count_exact = 0u64;
    let mut truth_present = 0u64;
    for s in &test {
        if s.truth.present {
            truth_present += 1;
        }
        match (s.predicted.present, s.truth.present) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
        count_abs_err_sum +=
            (s.predicted.person_count as f64 - s.truth.person_count as f64).abs();
        if s.predicted.person_count == s.truth.person_count {
            count_exact += 1;
        }
    }
    let presence_accuracy = if n_test > 0 {
        (tp + tn) as f64 / n_test as f64
    } else {
        0.0
    };
    let presence_f1 = f1_from_confusion(tp, fp, fn_);
    let count_mae = if n_test > 0 {
        count_abs_err_sum / n_test as f64
    } else {
        f64::INFINITY
    };
    let count_exact_match = if n_test > 0 {
        count_exact as f64 / n_test as f64
    } else {
        0.0
    };
    let presence_f1_ci = bootstrap_f1_ci(&test, criteria.bootstrap_iters, criteria.bootstrap_seed);

    let provenance_pass = provenance.is_claimable();
    let sample_size_pass = n_test >= criteria.min_test_samples;
    // Degenerate-test-set guard (review finding 2): both truth classes must be
    // represented — at least `min_positive_rate` present samples AND at least
    // one absent sample. Otherwise the F1/accuracy numbers are vacuous (an
    // all-absent set is aced by a predictor that always says "absent").
    let positive_rate = if n_test > 0 {
        truth_present as f64 / n_test as f64
    } else {
        0.0
    };
    let class_balance_pass =
        n_test > 0 && positive_rate >= criteria.min_positive_rate && truth_present < n_test as u64;
    // Gate on the LOWER CI bound, not the point estimate (small-N guard).
    let presence_pass = presence_f1_ci.0 >= criteria.min_presence_f1;
    let count_pass = count_mae <= criteria.max_count_mae;
    let overall_pass = claim_allowed(
        provenance_pass,
        split_pass,
        sample_size_pass,
        class_balance_pass,
        presence_pass,
        count_pass,
    );

    let released_claim = if overall_pass {
        format!(
            "presence F1 {:.3} (95% CI {:.3}-{:.3}), count MAE {:.3} on {} held-out measured samples",
            presence_f1, presence_f1_ci.0, presence_f1_ci.1, count_mae, n_test
        )
    } else {
        NO_CLAIM.to_string()
    };

    BenchmarkReport {
        provenance_tag: provenance.tag(),
        n_test,
        presence_accuracy,
        presence_f1,
        presence_f1_ci,
        count_exact_match,
        count_mae,
        provenance_pass,
        split_pass,
        presence_pass,
        count_pass,
        sample_size_pass,
        class_balance_pass,
        overall_pass,
        released_claim,
    }
}

fn f1_from_confusion(tp: u64, fp: u64, fn_: u64) -> f64 {
    let denom = 2 * tp + fp + fn_;
    if denom == 0 {
        // No positives anywhere (tp = fp = fn = 0): F1 is undefined, and the
        // vacuous case must score 0.0, never 1.0 — an all-absent test set plus
        // an always-absent predictor was previously awarded a perfect F1
        // (review finding 2). The class-balance criterion independently fails
        // such a degenerate set with its own reason.
        return 0.0;
    }
    (2 * tp) as f64 / denom as f64
}

/// Deterministic 95% bootstrap CI for presence F1 (percentile method) using a
/// small splitmix64 PRNG — no external rng, reproducible across machines.
fn bootstrap_f1_ci(test: &[&LabeledSample], iters: usize, seed: u64) -> (f64, f64) {
    let n = test.len();
    if n == 0 || iters == 0 {
        return (0.0, 0.0);
    }
    let mut state = seed;
    let mut next = || {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut f1s = Vec::with_capacity(iters);
    for _ in 0..iters {
        let (mut tp, mut fp, mut fn_) = (0u64, 0u64, 0u64);
        for _ in 0..n {
            let idx = (next() % n as u64) as usize;
            let s = test[idx];
            match (s.predicted.present, s.truth.present) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        f1s.push(f1_from_confusion(tp, fp, fn_));
    }
    f1s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| {
        let rank = ((q * (f1s.len() as f64 - 1.0)).round() as usize).min(f1s.len() - 1);
        f1s[rank]
    };
    (pct(0.025), pct(0.975))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(subj: &str, env: &str, t: (bool, u32), p: (bool, u32)) -> LabeledSample {
        LabeledSample {
            subject_id: subj.into(),
            environment_id: env.into(),
            truth: Occupancy::new(t.0, t.1),
            predicted: Occupancy::new(p.0, p.1),
        }
    }

    /// A perfect predictor on a leak-free MEASURED split releases a claim.
    fn perfect_measured(n: usize) -> (Vec<LabeledSample>, EvalSplit) {
        let mut samples = Vec::new();
        // train subjects s0.., test subjects t0.. (disjoint); envs likewise.
        for i in 0..n {
            samples.push(sample(
                &format!("train-s{i}"),
                &format!("train-e{i}"),
                (i % 2 == 0, (i % 3) as u32),
                (i % 2 == 0, (i % 3) as u32),
            ));
        }
        for i in 0..n {
            samples.push(sample(
                &format!("test-s{i}"),
                &format!("test-e{i}"),
                (i % 2 == 0, (i % 3) as u32),
                (i % 2 == 0, (i % 3) as u32),
            ));
        }
        let split = EvalSplit {
            train_idx: (0..n).collect(),
            test_idx: (n..2 * n).collect(),
        };
        (samples, split)
    }

    #[test]
    fn perfect_measured_releases_claim() {
        let (samples, split) = perfect_measured(40);
        let r = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        assert!(r.overall_pass);
        assert!((r.presence_f1 - 1.0).abs() < 1e-9);
        assert_eq!(r.count_mae, 0.0);
        assert!(r.released_claim.contains("F1"));
        assert!(!r.released_claim.contains("research use only"));
    }

    #[test]
    fn synthetic_data_is_scored_but_never_claimed() {
        let (samples, split) = perfect_measured(40);
        let r = evaluate(&samples, DataProvenance::Synthetic, &split, &BenchmarkCriteria::default());
        // Metrics are still computed...
        assert!((r.presence_f1 - 1.0).abs() < 1e-9);
        // ...but no claim, because the data is not measured.
        assert!(!r.provenance_pass);
        assert!(!r.overall_pass);
        assert_eq!(r.claim(), NO_CLAIM);
    }

    #[test]
    fn mock_data_is_never_claimed() {
        let (samples, split) = perfect_measured(40);
        let r = evaluate(&samples, DataProvenance::Mock, &split, &BenchmarkCriteria::default());
        assert!(!r.provenance_pass);
        assert_eq!(r.claim(), NO_CLAIM);
    }

    #[test]
    fn subject_leakage_is_rejected() {
        // Same subject id in train and test.
        let samples = vec![
            sample("shared", "e0", (true, 1), (true, 1)),
            sample("shared", "e1", (true, 1), (true, 1)),
        ];
        let split = EvalSplit { train_idx: vec![0], test_idx: vec![1] };
        assert_eq!(
            split.validate(&samples),
            Err(SplitError::SubjectLeakage("shared".into()))
        );
        let r = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        assert!(!r.split_pass);
        assert!(!r.overall_pass);
        assert_eq!(r.claim(), NO_CLAIM);
    }

    #[test]
    fn environment_leakage_is_rejected() {
        let samples = vec![
            sample("s0", "shared-room", (true, 1), (true, 1)),
            sample("s1", "shared-room", (true, 1), (true, 1)),
        ];
        let split = EvalSplit { train_idx: vec![0], test_idx: vec![1] };
        assert_eq!(
            split.validate(&samples),
            Err(SplitError::EnvironmentLeakage("shared-room".into()))
        );
    }

    #[test]
    fn small_sample_is_withheld_even_if_perfect() {
        let (samples, split) = perfect_measured(5); // 5 < default min 30
        let r = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        assert!(!r.sample_size_pass);
        assert!(!r.overall_pass);
    }

    /// The probative CI-gate case (review finding 10): a test set whose POINT
    /// F1 clears the 0.9 threshold while the bootstrap CI LOWER bound falls
    /// below it — the claim must be withheld. A point-estimate gate would
    /// (wrongly) release here.
    #[test]
    fn gate_uses_ci_lower_bound_not_point_estimate() {
        let mut samples = Vec::new();
        for i in 0..40 {
            samples.push(sample(
                &format!("train-{i}"),
                &format!("te-{i}"),
                (i % 2 == 0, 1),
                (i % 2 == 0, 1),
            ));
        }
        // Test: 20 truth-present / 20 truth-absent (class-balanced). All
        // absents predicted correctly; 3 of the 20 presents missed (FN).
        // Point F1 = 2·17/(2·17 + 0 + 3) = 34/37 ≈ 0.919 ≥ 0.9, but resamples
        // drawing 4+ of the FNs push F1 below 0.9, so the 2.5th percentile
        // lands under the threshold.
        for i in 0..40 {
            let truth_present = i < 20;
            let predicted_present = truth_present && i >= 3; // i 0..3 → FN
            samples.push(sample(
                &format!("test-{i}"),
                &format!("tn-{i}"),
                (truth_present, u32::from(truth_present)),
                (predicted_present, u32::from(truth_present)),
            ));
        }
        let split = EvalSplit { train_idx: (0..40).collect(), test_idx: (40..80).collect() };
        let criteria = BenchmarkCriteria::default();
        let r = evaluate(&samples, DataProvenance::Measured, &split, &criteria);
        // Construct verified: point estimate above the threshold...
        assert!(
            r.presence_f1 >= criteria.min_presence_f1,
            "fixture must put the point estimate ({:.3}) above the threshold",
            r.presence_f1
        );
        // ...while the CI lower bound is below it...
        assert!(
            r.presence_f1_ci.0 < criteria.min_presence_f1,
            "fixture must put the CI lower bound ({:.3}) below the threshold",
            r.presence_f1_ci.0
        );
        // ...and the claim is therefore withheld.
        assert!(!r.presence_pass);
        assert!(!r.overall_pass);
        assert_eq!(r.claim(), NO_CLAIM);
        // Every other criterion passes, isolating the CI gate as the cause.
        assert!(r.provenance_pass && r.split_pass && r.sample_size_pass);
        assert!(r.class_balance_pass && r.count_pass);
    }

    /// Degenerate test set (review finding 2): all-absent ground truth plus an
    /// always-absent predictor must NOT release a claim — F1 is vacuous (0.0,
    /// not 1.0) and the class-balance criterion fails with its own flag.
    #[test]
    fn all_absent_test_set_is_degenerate_and_withheld() {
        let mut samples = Vec::new();
        for i in 0..40 {
            samples.push(sample(&format!("tr-{i}"), &format!("te-{i}"), (true, 1), (true, 1)));
        }
        for i in 0..40 {
            // Truth all absent; predictor always says absent → tp=fp=fn=0.
            samples.push(sample(&format!("ts-{i}"), &format!("ev-{i}"), (false, 0), (false, 0)));
        }
        let split = EvalSplit { train_idx: (0..40).collect(), test_idx: (40..80).collect() };
        let r = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        // Vacuous F1 scores 0.0 (was 1.0 before the fix).
        assert_eq!(r.presence_f1, 0.0);
        assert_eq!(r.presence_f1_ci, (0.0, 0.0));
        // Degeneracy is named as its own failed criterion.
        assert!(!r.class_balance_pass);
        assert!(!r.overall_pass);
        assert_eq!(r.claim(), NO_CLAIM);
    }

    /// The mirror degeneracy: an all-PRESENT test set (no absent samples) is
    /// also refused — a trivially always-present predictor would ace it.
    #[test]
    fn all_present_test_set_is_degenerate_and_withheld() {
        let mut samples = Vec::new();
        for i in 0..40 {
            samples.push(sample(&format!("tr-{i}"), &format!("te-{i}"), (i % 2 == 0, 1), (i % 2 == 0, 1)));
        }
        for i in 0..40 {
            samples.push(sample(&format!("ts-{i}"), &format!("ev-{i}"), (true, 1), (true, 1)));
        }
        let split = EvalSplit { train_idx: (0..40).collect(), test_idx: (40..80).collect() };
        let r = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        assert!((r.presence_f1 - 1.0).abs() < 1e-9, "metric still computed");
        assert!(!r.class_balance_pass, "single-class test set is degenerate");
        assert!(!r.overall_pass);
        assert_eq!(r.claim(), NO_CLAIM);
    }

    #[test]
    fn bootstrap_ci_is_deterministic() {
        let (samples, split) = perfect_measured(40);
        let a = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        let b = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        assert_eq!(a.presence_f1_ci, b.presence_f1_ci);
    }

    #[test]
    fn count_mae_failure_withholds_claim() {
        let mut samples = Vec::new();
        for i in 0..40 {
            samples.push(sample(&format!("tr-{i}"), &format!("te-{i}"), (true, 1), (true, 1)));
        }
        // Class-balanced test set (so count MAE is the ONLY failing criterion):
        // presence perfect, but the count is always off by 2 -> MAE 2.0 > 0.5.
        for i in 0..40 {
            let present = i % 2 == 0;
            let truth_count = u32::from(present);
            samples.push(sample(
                &format!("ts-{i}"),
                &format!("ev-{i}"),
                (present, truth_count),
                (present, truth_count + 2),
            ));
        }
        let split = EvalSplit { train_idx: (0..40).collect(), test_idx: (40..80).collect() };
        let r = evaluate(&samples, DataProvenance::Measured, &split, &BenchmarkCriteria::default());
        assert!(r.presence_pass);
        assert!(r.class_balance_pass);
        assert!(!r.count_pass);
        assert!(!r.overall_pass);
    }

    #[test]
    fn claim_invariant_requires_all_six() {
        assert!(claim_allowed(true, true, true, true, true, true));
        // Every single-false combination is denied.
        for i in 0..6 {
            let v: Vec<bool> = (0..6).map(|j| j != i).collect();
            assert!(
                !claim_allowed(v[0], v[1], v[2], v[3], v[4], v[5]),
                "criterion {i} false must deny the claim"
            );
        }
    }
}
