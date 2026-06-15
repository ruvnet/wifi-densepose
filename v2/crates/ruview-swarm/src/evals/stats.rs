//! Hand-rolled robust statistics for the evaluation harness (Agarwal 2021).
//!
//! Implements the interquartile mean (IQM), a 95% stratified-bootstrap
//! confidence interval of the IQM, and the probability-of-improvement metric —
//! the three statistics recommended by "Deep RL at the Edge of the
//! Statistical Precipice" (Agarwal et al., NeurIPS 2021) for reporting
//! few-seed RL results.
//!
//! All randomness comes from a local linear-congruential generator (LCG) seeded
//! explicitly, so every CI is fully reproducible — no `thread_rng`, no clock.

/// Interquartile mean: mean of the middle 50% of samples (drop the bottom 25%
/// and the top 25%). Robust to outliers in either tail.
///
/// Small-N behaviour: with fewer than 4 samples the trim would empty the set,
/// so it falls back to the plain arithmetic mean. An empty slice returns 0.0.
pub fn iqm(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    if samples.len() < 4 {
        return samples.iter().sum::<f64>() / samples.len() as f64;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo = n / 4; // trim bottom 25%
    let hi = n - lo; // trim top 25% (symmetric)
    let mid = &sorted[lo..hi];
    if mid.is_empty() {
        return sorted.iter().sum::<f64>() / n as f64;
    }
    mid.iter().sum::<f64>() / mid.len() as f64
}

/// A point estimate with its lower / upper 95% confidence bounds.
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    pub point: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Minimal reproducible LCG (Numerical Recipes constants) yielding f64 in [0,1).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid a zero state collapsing the generator.
        Lcg(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform index in [0, n).
    #[inline]
    fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() >> 11) as usize % n
    }
}

/// 95% stratified-bootstrap CI of the IQM.
///
/// `strata` groups samples (one inner `Vec` per stratum, e.g. per task or per
/// seed). Each bootstrap replicate resamples WITH replacement *within* each
/// stratum (preserving the stratum sizes), pools all resampled values, and
/// recomputes the IQM. Repeat `n_boot` times and take the 2.5 / 97.5
/// percentiles for the CI bounds. The `point` estimate is the IQM of the pooled
/// original samples. Deterministic for a fixed `seed`.
pub fn stratified_bootstrap_ci(
    strata: &[Vec<f64>],
    n_boot: usize,
    seed: u64,
) -> ConfidenceInterval {
    let pooled: Vec<f64> = strata.iter().flatten().copied().collect();
    let point = iqm(&pooled);

    if pooled.is_empty() || n_boot == 0 {
        return ConfidenceInterval { point, lo: point, hi: point };
    }

    let mut rng = Lcg::new(seed);
    let mut replicates = Vec::with_capacity(n_boot);
    let mut buf: Vec<f64> = Vec::with_capacity(pooled.len());

    for _ in 0..n_boot {
        buf.clear();
        for stratum in strata {
            let m = stratum.len();
            for _ in 0..m {
                buf.push(stratum[rng.index(m)]);
            }
        }
        replicates.push(iqm(&buf));
    }

    replicates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = percentile(&replicates, 2.5);
    let hi = percentile(&replicates, 97.5);
    ConfidenceInterval { point, lo, hi }
}

/// Linear-interpolated percentile of a pre-sorted slice. `p` in [0, 100].
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Probability of improvement: P(a-sample > b-sample) over all pairs (Agarwal).
///
/// Counts each (a_i, b_j) pair where `a_i > b_j` as 1, a tie as 0.5, and
/// normalizes by the pair count. 1.0 means `a` strictly dominates; ~0.5 means
/// the two are statistically indistinguishable. Returns 0.5 if either is empty.
pub fn probability_of_improvement(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.5;
    }
    let mut wins = 0.0;
    for &ai in a {
        for &bj in b {
            if ai > bj {
                wins += 1.0;
            } else if (ai - bj).abs() < f64::EPSILON {
                wins += 0.5;
            }
        }
    }
    wins / (a.len() as f64 * b.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iqm_trims_outliers() {
        // 0..=100 plus one extreme outlier; IQM should sit near the middle (~50),
        // not be dragged toward 1e9.
        let mut samples: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        samples.push(1e9);
        let v = iqm(&samples);
        assert!(
            (40.0..=60.0).contains(&v),
            "IQM should be near the middle-50% mean (~50), got {v}"
        );
    }

    #[test]
    fn test_iqm_small() {
        // Fewer than 4 samples → plain mean.
        assert_eq!(iqm(&[2.0, 4.0]), 3.0);
        assert_eq!(iqm(&[10.0]), 10.0);
        assert_eq!(iqm(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(iqm(&[]), 0.0);
    }

    #[test]
    fn test_bootstrap_ci_brackets_point() {
        let strata = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
        ];
        let ci = stratified_bootstrap_ci(&strata, 500, 42);
        assert!(ci.lo <= ci.point, "lo ≤ point: {} ≤ {}", ci.lo, ci.point);
        assert!(ci.point <= ci.hi, "point ≤ hi: {} ≤ {}", ci.point, ci.hi);
        // Deterministic: same seed → identical interval.
        let ci2 = stratified_bootstrap_ci(&strata, 500, 42);
        assert_eq!(ci.point, ci2.point);
        assert_eq!(ci.lo, ci2.lo);
        assert_eq!(ci.hi, ci2.hi);
    }

    #[test]
    fn test_prob_improvement_obvious() {
        assert_eq!(
            probability_of_improvement(&[10.0, 10.0, 10.0], &[0.0, 0.0, 0.0]),
            1.0
        );
        // Identical samples → all ties → 0.5.
        let poi = probability_of_improvement(&[5.0, 5.0], &[5.0, 5.0]);
        assert!((poi - 0.5).abs() < 1e-9, "symmetric ties → ~0.5, got {poi}");
    }
}
