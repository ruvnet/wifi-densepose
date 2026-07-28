//! Anti-leakage evaluation protocol (ADR-273 §5).
//!
//! The biggest failure mode of RF sensing results is domain leakage
//! disguised as accuracy: random frame splits let a model recognize the
//! room, session, person, device, or trajectory instead of learning
//! transferable physics. The fix is structural, not statistical:
//!
//! - [`StrictSplit`] holds out **complete values** of a partition dimension
//!   (room / day / person / chipset / firmware / antenna layout) and proves
//!   the train and test sides share none of them.
//! - [`expected_calibration_error`] and [`selective_metrics`] make
//!   confidence quality and abstention first-class metrics: for
//!   safety-relevant uses an uncertain result must become *no decision*.
//! - [`relative_degradation`] tracks known→unknown condition degradation
//!   (the ADR-273 acceptance gate is < 20 %).

use std::collections::BTreeSet;

/// Provenance key of one sample — every dimension that could leak identity
/// or environment structure into a random split.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PartitionKey {
    /// Room / environment identifier.
    pub room: String,
    /// Capture day (coarse session time).
    pub day: String,
    /// Person identifier (or "none").
    pub person: String,
    /// Chipset family.
    pub chipset: String,
    /// Firmware version.
    pub firmware: String,
    /// Antenna layout identifier.
    pub layout: String,
    /// Capture session identifier (packet-session leakage is as real as
    /// room leakage — ADR-279 §4 split manifest).
    pub session: String,
}

/// Which dimension of [`PartitionKey`] a split holds out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionDim {
    /// Hold out complete rooms.
    Room,
    /// Hold out complete days.
    Day,
    /// Hold out complete people.
    Person,
    /// Hold out complete chipsets.
    Chipset,
    /// Hold out complete firmware versions.
    Firmware,
    /// Hold out complete antenna layouts.
    Layout,
    /// Hold out complete capture sessions.
    Session,
}

impl PartitionDim {
    fn value<'a>(&self, k: &'a PartitionKey) -> &'a str {
        match self {
            Self::Room => &k.room,
            Self::Day => &k.day,
            Self::Person => &k.person,
            Self::Chipset => &k.chipset,
            Self::Firmware => &k.firmware,
            Self::Layout => &k.layout,
            Self::Session => &k.session,
        }
    }

    /// All partition dimensions, for exhaustive manifest checks.
    pub const ALL: [Self; 7] = [
        Self::Room,
        Self::Day,
        Self::Person,
        Self::Chipset,
        Self::Firmware,
        Self::Layout,
        Self::Session,
    ];
}

/// The mandatory split manifest (ADR-279 §4): per-dimension disjointness
/// certificates for a train/test split. A result is only reportable as
/// leakage-resistant along the dimensions this manifest certifies.
#[derive(Debug, Clone)]
pub struct SplitManifest {
    /// `(dimension, train∩test == ∅)` for every partition dimension.
    pub disjoint: Vec<(PartitionDim, bool)>,
}

impl SplitManifest {
    /// Builds the manifest for an arbitrary index split.
    #[must_use]
    pub fn build(keys: &[PartitionKey], train: &[usize], test: &[usize]) -> Self {
        let disjoint = PartitionDim::ALL
            .iter()
            .map(|dim| {
                let train_vals: BTreeSet<&str> =
                    train.iter().map(|&i| dim.value(&keys[i])).collect();
                let test_vals: BTreeSet<&str> =
                    test.iter().map(|&i| dim.value(&keys[i])).collect();
                (*dim, train_vals.is_disjoint(&test_vals))
            })
            .collect();
        Self { disjoint }
    }

    /// True iff the given dimension is certified disjoint.
    #[must_use]
    pub fn is_disjoint(&self, dim: PartitionDim) -> bool {
        self.disjoint.iter().any(|(d, ok)| *d == dim && *ok)
    }

    /// True iff every dimension is disjoint (the full anti-leakage bar:
    /// `train_rooms ∩ test_rooms = ∅` … `train_sessions ∩ test_sessions = ∅`).
    #[must_use]
    pub fn fully_disjoint(&self) -> bool {
        self.disjoint.iter().all(|(_, ok)| *ok)
    }
}

/// Mean per-joint position error (metres) between two joint sets.
///
/// # Panics
/// If the slices have different lengths or are empty.
#[must_use]
pub fn mpjpe(pred: &[[f64; 3]], truth: &[[f64; 3]]) -> f64 {
    assert_eq!(pred.len(), truth.len());
    assert!(!pred.is_empty());
    pred.iter()
        .zip(truth)
        .map(|(p, t)| ((p[0] - t[0]).powi(2) + (p[1] - t[1]).powi(2) + (p[2] - t[2]).powi(2)).sqrt())
        .sum::<f64>()
        / pred.len() as f64
}

/// A train/test index split with a proof-of-disjointness certificate.
#[derive(Debug, Clone)]
pub struct StrictSplit {
    /// Training sample indices.
    pub train: Vec<usize>,
    /// Held-out test sample indices.
    pub test: Vec<usize>,
    /// Dimension that was held out.
    pub dim: PartitionDim,
}

impl StrictSplit {
    /// Splits by holding out every sample whose `dim` value is in
    /// `holdout_values`. Guaranteed disjoint by construction; [`Self::verify`]
    /// re-checks it independently (belt and braces for downstream callers
    /// that mutate splits).
    #[must_use]
    pub fn holdout(keys: &[PartitionKey], dim: PartitionDim, holdout_values: &[&str]) -> Self {
        let held: BTreeSet<&str> = holdout_values.iter().copied().collect();
        let mut train = Vec::new();
        let mut test = Vec::new();
        for (i, k) in keys.iter().enumerate() {
            if held.contains(dim.value(k)) {
                test.push(i);
            } else {
                train.push(i);
            }
        }
        Self { train, test, dim }
    }

    /// Independently verifies that no held-out dimension value appears on
    /// the training side (and vice versa). Returns `false` on any leak.
    #[must_use]
    pub fn verify(&self, keys: &[PartitionKey]) -> bool {
        let train_vals: BTreeSet<&str> =
            self.train.iter().map(|&i| self.dim.value(&keys[i])).collect();
        let test_vals: BTreeSet<&str> =
            self.test.iter().map(|&i| self.dim.value(&keys[i])).collect();
        train_vals.is_disjoint(&test_vals)
    }
}

/// Expected Calibration Error over equal-width confidence bins.
///
/// `probs[i]` is the predicted probability of the positive class;
/// `labels[i]` the truth. ECE = Σ (|bin|/N) · |accuracy(bin) − confidence(bin)|.
#[must_use]
pub fn expected_calibration_error(probs: &[f64], labels: &[bool], n_bins: usize) -> f64 {
    assert_eq!(probs.len(), labels.len());
    assert!(n_bins > 0);
    let n = probs.len();
    if n == 0 {
        return 0.0;
    }
    let mut ece = 0.0;
    for b in 0..n_bins {
        let lo = b as f64 / n_bins as f64;
        let hi = (b + 1) as f64 / n_bins as f64;
        let mut count = 0usize;
        let mut conf_sum = 0.0;
        let mut acc_sum = 0.0;
        for (p, &y) in probs.iter().zip(labels) {
            // Confidence of the *predicted* class.
            let (conf, pred) = if *p >= 0.5 { (*p, true) } else { (1.0 - *p, false) };
            let in_bin = if b == n_bins - 1 { conf >= lo && conf <= hi } else { conf >= lo && conf < hi };
            if in_bin {
                count += 1;
                conf_sum += conf;
                acc_sum += f64::from(pred == y);
            }
        }
        if count > 0 {
            let cf = count as f64;
            ece += (cf / n as f64) * ((acc_sum / cf) - (conf_sum / cf)).abs();
        }
    }
    ece
}

/// Coverage/selective-risk pair at a confidence threshold `tau`: predictions
/// with confidence below `tau` abstain (become *no decision*).
#[derive(Debug, Clone, Copy)]
pub struct SelectiveMetrics {
    /// Fraction of samples on which a decision was made.
    pub coverage: f64,
    /// Error rate among decided samples (0 when nothing decided).
    pub selective_risk: f64,
}

/// Computes coverage and selective risk at threshold `tau` ∈ [0.5, 1].
#[must_use]
pub fn selective_metrics(probs: &[f64], labels: &[bool], tau: f64) -> SelectiveMetrics {
    assert_eq!(probs.len(), labels.len());
    let mut decided = 0usize;
    let mut wrong = 0usize;
    for (p, &y) in probs.iter().zip(labels) {
        let (conf, pred) = if *p >= 0.5 { (*p, true) } else { (1.0 - *p, false) };
        if conf >= tau {
            decided += 1;
            if pred != y {
                wrong += 1;
            }
        }
    }
    SelectiveMetrics {
        coverage: decided as f64 / probs.len().max(1) as f64,
        selective_risk: if decided == 0 { 0.0 } else { wrong as f64 / decided as f64 },
    }
}

/// Relative degradation from a known-condition metric to an
/// unknown-condition metric (higher-is-better metrics such as F1).
/// ADR-273's acceptance gate: `< 0.20`.
#[must_use]
pub fn relative_degradation(known: f64, unknown: f64) -> f64 {
    if known <= 0.0 {
        return 0.0;
    }
    (known - unknown) / known
}

/// Binary F1 score.
#[must_use]
pub fn f1_score(predictions: &[bool], labels: &[bool]) -> f64 {
    assert_eq!(predictions.len(), labels.len());
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    for (&p, &y) in predictions.iter().zip(labels) {
        match (p, y) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fn_ += 1.0,
            (false, false) => {}
        }
    }
    if tp == 0.0 {
        return 0.0;
    }
    let precision = tp / (tp + fp);
    let recall = tp / (tp + fn_);
    2.0 * precision * recall / (precision + recall)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> Vec<PartitionKey> {
        let mut out = Vec::new();
        for room in ["room-a", "room-b", "room-c"] {
            for day in ["d1", "d2"] {
                for person in ["p1", "p2"] {
                    out.push(PartitionKey {
                        room: room.into(),
                        day: day.into(),
                        person: person.into(),
                        chipset: "esp32s3".into(),
                        firmware: "v1.2".into(),
                        layout: "L".into(),
                        session: format!("{room}-{day}-{person}"),
                    });
                }
            }
        }
        out
    }

    #[test]
    fn strict_split_holds_out_complete_rooms() {
        let keys = keys();
        let split = StrictSplit::holdout(&keys, PartitionDim::Room, &["room-c"]);
        assert_eq!(split.test.len(), 4);
        assert_eq!(split.train.len(), 8);
        assert!(split.verify(&keys), "disjointness certificate must hold");
        for &i in &split.test {
            assert_eq!(keys[i].room, "room-c");
        }
    }

    #[test]
    fn verify_catches_a_manufactured_leak() {
        let keys = keys();
        let mut split = StrictSplit::holdout(&keys, PartitionDim::Room, &["room-c"]);
        // Manufacture a leak: push a room-c sample into training.
        split.train.push(split.test[0]);
        assert!(!split.verify(&keys), "leak must be detected");
    }

    #[test]
    fn split_manifest_certifies_per_dimension_disjointness() {
        let keys = keys();
        let split = StrictSplit::holdout(&keys, PartitionDim::Room, &["room-c"]);
        let manifest = SplitManifest::build(&keys, &split.train, &split.test);
        // Rooms are disjoint (and sessions, which embed the room)…
        assert!(manifest.is_disjoint(PartitionDim::Room));
        assert!(manifest.is_disjoint(PartitionDim::Session));
        // …but people/days/chipsets are shared, and the manifest says so
        // instead of letting the split masquerade as fully leakage-free.
        assert!(!manifest.is_disjoint(PartitionDim::Person));
        assert!(!manifest.is_disjoint(PartitionDim::Chipset));
        assert!(!manifest.fully_disjoint());
    }

    #[test]
    fn mpjpe_basics() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let b = [[0.0, 0.0, 0.1], [1.0, 0.0, 0.0]];
        assert!((mpjpe(&a, &b) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn ece_is_low_for_calibrated_and_high_for_overconfident() {
        // Calibrated: p = 0.8 predictions that are right 80 % of the time.
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            probs.push(0.8);
            labels.push(i % 10 < 8);
        }
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece < 0.02, "calibrated ECE should be tiny, got {ece}");

        // Overconfident: p = 0.99 but only 60 % correct.
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            probs.push(0.99);
            labels.push(i % 10 < 6);
        }
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece > 0.3, "overconfident ECE should be large, got {ece}");
    }

    #[test]
    fn abstention_trades_coverage_for_risk() {
        // Confident predictions are correct; near-0.5 ones are coin flips.
        let mut probs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..50 {
            probs.push(0.95);
            labels.push(true);
            probs.push(0.55);
            labels.push(i % 2 == 0); // half wrong at low confidence
        }
        let loose = selective_metrics(&probs, &labels, 0.5);
        let strict = selective_metrics(&probs, &labels, 0.9);
        assert!(strict.coverage < loose.coverage);
        assert!(
            strict.selective_risk < loose.selective_risk,
            "raising the threshold must lower risk: {strict:?} vs {loose:?}"
        );
        assert!((strict.selective_risk - 0.0).abs() < 1e-12);
    }

    #[test]
    fn f1_and_degradation_basics() {
        let preds = [true, true, false, true];
        let labels = [true, false, false, true];
        // tp=2 fp=1 fn=0 → precision 2/3, recall 1 → F1 = 0.8.
        assert!((f1_score(&preds, &labels) - 0.8).abs() < 1e-12);
        assert!((relative_degradation(0.95, 0.85) - 0.105_263).abs() < 1e-4);
        assert!((relative_degradation(0.0, 0.5)).abs() < 1e-12);
    }
}
