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
use crate::linalg::{dist_sq, dot, norm, set_norm_inplace};

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

/// Which adversary the experiment runs. Added from the 2025–2026 SOTA sweep
/// (ADR-288 §sota) so the collapse is shown to hold against the *strongest*
/// published attacker shapes, not just a plain nearest-centroid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttackerKind {
    /// Nearest-centroid on the full captured report (uses the configured [`Metric`]).
    #[default]
    NearestCentroid,
    /// Models BFI→CSI reconstruction (BFIAttack, arXiv:2604.04179): the adversary
    /// recovers the CSI *consistent with the captured report* and classifies its
    /// direction. Because a keyed secret rotation has no key to invert, what it
    /// reconstructs is the *rotated* CSI — so identity does not survive.
    Reconstruction,
    /// Pools many captures per identity and whitens before matching (the
    /// PrivISAC-style adaptive/retraining adversary). Averaging cannot undo a
    /// fresh secret rotation, so the pooled, whitened template still collapses.
    AdaptivePooling,
}

/// BFI→CSI reconstruction adversary. Classifies the **direction** (L2-normalized
/// fine block) of the reconstructed CSI — the strongest gain-invariant descriptor
/// an attacker can recover from a captured report. Defeated by a secret rotation
/// (it only ever recovers the rotated direction).
#[derive(Debug, Clone, Default)]
pub struct ReconstructionAttacker {
    centroids: Vec<Vec<f32>>,
    ids: Vec<usize>,
}

fn reconstructed_direction(s: &BfiSample) -> Vec<f32> {
    let mut v = s.fine().to_vec();
    set_norm_inplace(&mut v, 1.0);
    v
}

impl ReconstructionAttacker {
    /// Build an empty reconstruction attacker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enroll direction-centroids from reconstructed captures.
    pub fn enroll(&mut self, samples: &[(usize, BfiSample)]) {
        let mut ids: Vec<usize> = Vec::new();
        let mut sums: Vec<Vec<f32>> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        for (id, s) in samples {
            let f = reconstructed_direction(s);
            let slot = ids.iter().position(|x| x == id).unwrap_or_else(|| {
                ids.push(*id);
                sums.push(vec![0.0; f.len()]);
                counts.push(0);
                ids.len() - 1
            });
            for (acc, v) in sums[slot].iter_mut().zip(&f) {
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

    /// Classify a capture by nearest reconstructed direction.
    #[must_use]
    pub fn classify(&self, sample: &BfiSample) -> Option<usize> {
        let f = reconstructed_direction(sample);
        let mut best: Option<(usize, f32)> = None;
        for (id, c) in self.ids.iter().zip(&self.centroids) {
            let d = dist_sq(c, &f);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((*id, d));
            }
        }
        best.map(|(id, _)| id)
    }

    /// Top-1 accuracy over a labeled test set.
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

/// Adaptive pooling adversary: whitens the full report by per-dimension
/// standard deviation (estimated over all captures) before nearest-centroid,
/// modeling an attacker who aggregates many captures and re-fits. Whitening a
/// *fixed* coordinate basis cannot undo a rotation that mixes coordinates
/// afresh each session, so the pooled template still collapses.
#[derive(Debug, Clone, Default)]
pub struct AdaptivePoolingAttacker {
    centroids: Vec<Vec<f32>>,
    ids: Vec<usize>,
    inv_std: Vec<f32>,
}

impl AdaptivePoolingAttacker {
    /// Build an empty adaptive pooling attacker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enroll: estimate global per-dimension inverse std, then pooled per-id
    /// means.
    pub fn enroll(&mut self, samples: &[(usize, BfiSample)]) {
        if samples.is_empty() {
            return;
        }
        let dim = samples[0].1.values.len();
        let n = samples.len() as f32;
        let mut mean = vec![0.0f32; dim];
        for (_, s) in samples {
            for (m, v) in mean.iter_mut().zip(&s.values) {
                *m += v;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut var = vec![0.0f32; dim];
        for (_, s) in samples {
            for ((vv, v), m) in var.iter_mut().zip(&s.values).zip(&mean) {
                let d = v - m;
                *vv += d * d;
            }
        }
        self.inv_std = var
            .iter()
            .map(|v| 1.0 / ((v / n).sqrt().max(1e-6)))
            .collect();

        let mut ids: Vec<usize> = Vec::new();
        let mut sums: Vec<Vec<f32>> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        for (id, s) in samples {
            let slot = ids.iter().position(|x| x == id).unwrap_or_else(|| {
                ids.push(*id);
                sums.push(vec![0.0; dim]);
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

    /// Classify by whitened nearest-centroid.
    #[must_use]
    pub fn classify(&self, sample: &BfiSample) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (id, c) in self.ids.iter().zip(&self.centroids) {
            let mut d = 0.0f32;
            for ((cv, sv), w) in c.iter().zip(&sample.values).zip(&self.inv_std) {
                let diff = (cv - sv) * w;
                d += diff * diff;
            }
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((*id, d));
            }
        }
        best.map(|(id, _)| id)
    }

    /// Top-1 accuracy over a labeled test set.
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
