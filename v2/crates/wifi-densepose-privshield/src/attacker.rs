//! The adversary: a passive re-identification classifier over captured
//! beamforming feedback.
//!
//! The attacker models the BFId/CCS-2025 threat: a sniffer that enrolls a
//! template per candidate from observed reports, then classifies fresh
//! captures. We use a **nearest-centroid** classifier over the full report
//! vector. It is deliberately simple but is the right shape for the effect
//! under test: it succeeds exactly when a *stable* per-identity signature
//! survives across capture sessions, and fails when the signature is rotated
//! unpredictably each session (which is what the protector does).
//!
//! Nearest-centroid is also the honest choice for the collapse claim: a more
//! elaborate classifier cannot recover identity that has been mapped through a
//! fresh secret orthogonal transform each session — the mutual information
//! between a Haar-rotated signature and the identity label, marginalized over
//! unknown rotations, is what the protector drives down. The classifier
//! strength is not the lever; signature stability is.

use crate::identity::BfiSample;
use crate::linalg::{dist_sq, dot, norm};

/// Similarity metric the attacker uses to match a capture to a centroid.
///
/// Sweeping the metric is how [`crate::optimize`] checks that the shield's
/// collapse is a property of the *signal* (a rotated signature carries no
/// stable identity), not an artifact of one classifier's geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Metric {
    /// Euclidean nearest-centroid (default). Sensitive to magnitude.
    #[default]
    Euclidean,
    /// Cosine nearest-centroid. Scale-invariant; a natural stronger attacker
    /// against energy-preserving perturbations, since it ignores magnitude.
    Cosine,
}

/// A nearest-centroid re-identification attacker.
#[derive(Debug, Clone, Default)]
pub struct NearestCentroidAttacker {
    centroids: Vec<Vec<f32>>,
    ids: Vec<usize>,
    metric: Metric,
}

impl NearestCentroidAttacker {
    /// Build an empty attacker using the Euclidean metric.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an empty attacker using the given metric.
    #[must_use]
    pub fn with_metric(metric: Metric) -> Self {
        Self {
            metric,
            ..Self::default()
        }
    }

    /// Enroll from labeled captures: one centroid per identity, the mean of
    /// that identity's observed report vectors.
    pub fn enroll(&mut self, samples: &[(usize, BfiSample)]) {
        // Group by identity, preserving first-seen order.
        let mut ids: Vec<usize> = Vec::new();
        let mut sums: Vec<Vec<f32>> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        for (id, s) in samples {
            let slot = ids.iter().position(|x| x == id).unwrap_or_else(|| {
                ids.push(*id);
                sums.push(vec![0.0; s.values.len()]);
                counts.push(0);
                ids.len() - 1
            });
            for (acc, v) in sums[slot].iter_mut().zip(&s.values) {
                *acc += v;
            }
            counts[slot] += 1;
        }
        for (sum, &c) in sums.iter_mut().zip(&counts) {
            if c > 0 {
                let inv = 1.0 / c as f32;
                for v in sum.iter_mut() {
                    *v *= inv;
                }
            }
        }
        self.ids = ids;
        self.centroids = sums;
    }

    /// Classify a capture to the nearest enrolled centroid. Returns the
    /// predicted identity, or `None` if the attacker has not enrolled.
    #[must_use]
    pub fn classify(&self, sample: &BfiSample) -> Option<usize> {
        // Score is "lower is better" for both metrics: Euclidean uses squared
        // distance; Cosine uses the negated similarity.
        let score = |c: &[f32]| -> f32 {
            match self.metric {
                Metric::Euclidean => dist_sq(c, &sample.values),
                Metric::Cosine => {
                    let denom = norm(c) * norm(&sample.values);
                    if denom > 1e-12 {
                        -dot(c, &sample.values) / denom
                    } else {
                        0.0
                    }
                }
            }
        };
        let mut best: Option<(usize, f32)> = None;
        for (id, c) in self.ids.iter().zip(&self.centroids) {
            let d = score(c);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((*id, d));
            }
        }
        best.map(|(id, _)| id)
    }

    /// Top-1 re-identification accuracy over a labeled test set.
    #[must_use]
    pub fn accuracy(&self, test: &[(usize, BfiSample)]) -> f32 {
        if test.is_empty() {
            return 0.0;
        }
        let correct = test
            .iter()
            .filter(|(id, s)| self.classify(s) == Some(*id))
            .count();
        correct as f32 / test.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{Channel, SceneConfig};

    #[test]
    fn attacker_re_ids_unprotected_traffic() {
        let ch = Channel::new(SceneConfig::default());
        let mut enroll = Vec::new();
        let mut test = Vec::new();
        for id in 0..ch.config().identities {
            for s in 0..12 {
                enroll.push((id, ch.observe(id, b"enroll", s)));
            }
            for s in 0..12 {
                test.push((id, ch.observe(id, b"test", s)));
            }
        }
        let mut atk = NearestCentroidAttacker::new();
        atk.enroll(&enroll);
        // On unprotected traffic the stable signature is trivially recovered.
        assert!(atk.accuracy(&test) > 0.85);
    }
}
