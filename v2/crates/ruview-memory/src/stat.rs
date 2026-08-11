//! Online, forgetting running statistics (ADR-312 §2 — continuously learned
//! baseline).
//!
//! **SYNTHETIC / L0.** A [`RunningStat`] is a bounded, deterministic model of a
//! single scalar's *normal* value: an exponentially weighted mean and variance
//! that update online with a documented **forgetting factor**. It is part of a
//! simulation scaffold — it estimates a modelled normal, it never *measures*
//! anything, and it makes no accuracy claim (ADR-282 L0, ADR-300 evidence
//! discipline).
//!
//! ## The forgetting factor
//!
//! The forgetting factor `λ ∈ [0, 1)` is the weight retained on accumulated
//! history at each update; the effective learning rate is `α = 1 − λ`. A value
//! near `1` adapts slowly (long memory, tracks only slow legitimate drift); a
//! value near `0` adapts fast (short memory). Update rule (the standard
//! exponentially weighted moving mean/variance):
//!
//! ```text
//! diff = x − mean
//! incr = α · diff
//! mean ← mean + incr
//! var  ← λ · (var + diff · incr)      // = λ·(var + α·diff²) ≥ 0
//! ```
//!
//! The recursion keeps `var ≥ 0` exactly, so `sqrt` is always defined. There is
//! no wall-clock and no randomness anywhere: the same inputs in the same order
//! always yield the same state (ADR-300 determinism discipline).

use serde::{Deserialize, Serialize};

/// An exponentially weighted running mean/variance with an update count.
///
/// **SYNTHETIC / L0.** A modelled estimate of a scalar's normal value, never a
/// measurement.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunningStat {
    mean: f64,
    var: f64,
    count: u32,
}

impl Default for RunningStat {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStat {
    /// A fresh statistic with no history (`count == 0`).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mean: 0.0,
            var: 0.0,
            count: 0,
        }
    }

    /// A statistic pre-seeded from an external prior — used to anchor a
    /// propagation baseline on the twin's expected distribution (ADR-312 §1:
    /// "RF-propagation baseline … via the twin's expected distributions").
    ///
    /// `count` is the synthetic prior weight; a non-finite mean/variance is
    /// clamped to a valid `(mean = if finite else 0, var = max(0))` so the
    /// seeded stat never carries a poisoned value.
    #[must_use]
    pub fn seeded(mean: f64, var: f64, count: u32) -> Self {
        Self {
            mean: if mean.is_finite() { mean } else { 0.0 },
            var: if var.is_finite() { var.max(0.0) } else { 0.0 },
            count,
        }
    }

    /// Fold one observation into the estimate with the given forgetting factor
    /// `λ ∈ [0, 1)`. Deterministic; total (never panics). The first observation
    /// seeds the mean exactly and leaves the variance at zero.
    pub fn update(&mut self, x: f64, forgetting: f64) {
        if !x.is_finite() {
            return; // malformed value is ignored, never panics or poisons state
        }
        if self.count == 0 {
            self.mean = x;
            self.var = 0.0;
            self.count = 1;
            return;
        }
        let alpha = 1.0 - forgetting;
        let diff = x - self.mean;
        let incr = alpha * diff;
        self.mean += incr;
        // λ·(var + α·diff²): the α·diff² term is non-negative, so var stays ≥ 0.
        self.var = forgetting * (self.var + diff * incr);
        if !self.var.is_finite() {
            self.var = 0.0;
        }
        self.count = self.count.saturating_add(1);
    }

    /// The current modelled mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The current modelled (exponentially weighted) variance, always `≥ 0`.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.var
    }

    /// The number of observations folded in so far (seed weight included).
    #[must_use]
    pub fn count(&self) -> u32 {
        self.count
    }

    /// The standard deviation, floored at `floor` so significance is finite even
    /// for a degenerate zero-variance baseline (a channel that has only ever
    /// held one value). `floor` is expected to be `> 0` (config-validated).
    #[must_use]
    pub fn std_floored(&self, floor: f64) -> f64 {
        self.var.max(0.0).sqrt().max(floor)
    }

    /// How many floored standard deviations `x` sits from the learned mean — the
    /// deviation significance. Non-negative and finite whenever `floor > 0`.
    #[must_use]
    pub fn significance(&self, x: f64, floor: f64) -> f64 {
        let std = self.std_floored(floor);
        if std > 0.0 {
            (x - self.mean).abs() / std
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_update_seeds_mean_and_zero_variance() {
        let mut s = RunningStat::new();
        s.update(10.0, 0.9);
        assert_eq!(s.count(), 1);
        assert_eq!(s.mean(), 10.0);
        assert_eq!(s.variance(), 0.0);
    }

    #[test]
    fn variance_never_negative_and_update_is_deterministic() {
        let run = || {
            let mut s = RunningStat::new();
            for x in [1.0, 2.0, 1.5, 1.7, 1.6, 1.55] {
                s.update(x, 0.8);
            }
            s
        };
        let a = run();
        let b = run();
        assert_eq!(a, b);
        assert!(a.variance() >= 0.0);
    }

    #[test]
    fn non_finite_value_is_ignored_not_panicking() {
        let mut s = RunningStat::new();
        s.update(f64::NAN, 0.9);
        assert_eq!(s.count(), 0);
        s.update(5.0, 0.9);
        s.update(f64::INFINITY, 0.9);
        assert_eq!(s.count(), 1);
        assert_eq!(s.mean(), 5.0);
    }

    #[test]
    fn significance_is_finite_under_zero_variance_floor() {
        let s = RunningStat::seeded(0.0, 0.0, 10);
        // std floored at 0.1 ⇒ significance of 1.0 is 10 sigma, finite.
        assert!((s.significance(1.0, 0.1) - 10.0).abs() < 1e-9);
    }
}
