//! One-euro filter (Casiez, Roussel & Vogel, CHI 2012) for eye-position
//! smoothing.
//!
//! Implemented from the published algorithm: an exponential low-pass whose
//! cutoff adapts to the signal's speed — low cutoff (heavy smoothing) when
//! nearly still, higher cutoff (low lag) when moving fast. This is the
//! standard jitter/lag trade-off filter for interactive tracking.
//!
//! Timestamps are injected by the caller in seconds (monotonic). The crate
//! never reads a clock — repo discipline, and it keeps the wasm build free
//! of `performance.now()` assumptions.

use crate::math::Vec3;
use core::f64::consts::PI;

/// One-euro filter parameters.
///
/// - `min_cutoff` (Hz): smoothing floor. Lower = smoother but laggier at rest.
/// - `beta`: speed coefficient. Higher = less lag during fast motion.
/// - `d_cutoff` (Hz): cutoff for the internal derivative estimate.
///
/// Defaults are the paper's recommended starting point (1.0, 0.0, 1.0);
/// interactive head tracking typically tunes `min_cutoff` down and `beta` up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneEuroConfig {
    /// Smoothing floor in Hz (lower = smoother but laggier at rest).
    pub min_cutoff: f64,
    /// Speed coefficient (higher = less lag during fast motion).
    pub beta: f64,
    /// Cutoff in Hz for the internal derivative estimate.
    pub d_cutoff: f64,
}

impl Default for OneEuroConfig {
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.0,
            d_cutoff: 1.0,
        }
    }
}

/// Smoothing factor for an exponential low-pass at `cutoff` Hz sampled
/// `dt` seconds apart.
#[inline]
fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

/// Scalar one-euro filter.
#[derive(Clone, Copy, Debug, Default)]
pub struct OneEuro {
    cfg: OneEuroConfig,
    /// `(t, x_hat, dx_hat)` from the previous accepted sample.
    state: Option<(f64, f64, f64)>,
}

impl OneEuro {
    /// Create with the given parameters.
    pub fn new(cfg: OneEuroConfig) -> Self {
        Self { cfg, state: None }
    }

    /// Replace the parameters, keeping filter state.
    pub fn set_config(&mut self, cfg: OneEuroConfig) {
        self.cfg = cfg;
    }

    /// Clear state; the next sample passes through unfiltered.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Filter sample `x` taken at time `t_s` (seconds). Non-monotonic or
    /// non-finite input returns the previous estimate unchanged (never NaN).
    pub fn filter(&mut self, x: f64, t_s: f64) -> f64 {
        if !x.is_finite() || !t_s.is_finite() {
            return self.state.map_or(0.0, |(_, xh, _)| xh);
        }
        match self.state {
            None => {
                self.state = Some((t_s, x, 0.0));
                x
            }
            Some((t0, x0, dx0)) => {
                let dt = t_s - t0;
                if dt <= 0.0 {
                    return x0;
                }
                let dx = (x - x0) / dt;
                let a_d = alpha(self.cfg.d_cutoff, dt);
                let dx_hat = a_d * dx + (1.0 - a_d) * dx0;
                let cutoff = self.cfg.min_cutoff + self.cfg.beta * dx_hat.abs();
                let a = alpha(cutoff, dt);
                let x_hat = a * x + (1.0 - a) * x0;
                self.state = Some((t_s, x_hat, dx_hat));
                x_hat
            }
        }
    }
}

/// Component-wise one-euro filter over a [`Vec3`].
#[derive(Clone, Copy, Debug, Default)]
pub struct OneEuro3 {
    x: OneEuro,
    y: OneEuro,
    z: OneEuro,
}

impl OneEuro3 {
    /// Create with the same parameters on all three axes.
    pub fn new(cfg: OneEuroConfig) -> Self {
        Self {
            x: OneEuro::new(cfg),
            y: OneEuro::new(cfg),
            z: OneEuro::new(cfg),
        }
    }

    /// Replace parameters on all axes, keeping state.
    pub fn set_config(&mut self, cfg: OneEuroConfig) {
        self.x.set_config(cfg);
        self.y.set_config(cfg);
        self.z.set_config(cfg);
    }

    /// Clear state on all axes.
    pub fn reset(&mut self) {
        self.x.reset();
        self.y.reset();
        self.z.reset();
    }

    /// Filter a 3-D sample taken at time `t_s` (seconds).
    pub fn filter(&mut self, v: Vec3, t_s: f64) -> Vec3 {
        Vec3::new(
            self.x.filter(v.x, t_s),
            self.y.filter(v.y, t_s),
            self.z.filter(v.z, t_s),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_passes_through() {
        let mut f = OneEuro::new(OneEuroConfig::default());
        assert_eq!(f.filter(3.25, 0.0), 3.25);
    }

    #[test]
    fn constant_input_stays_constant() {
        let mut f = OneEuro::new(OneEuroConfig::default());
        for i in 0..100 {
            let y = f.filter(1.5, i as f64 * 0.016);
            assert!((y - 1.5).abs() < 1e-12);
        }
    }

    #[test]
    fn step_response_converges_monotonically() {
        let mut f = OneEuro::new(OneEuroConfig {
            min_cutoff: 1.0,
            beta: 0.0,
            d_cutoff: 1.0,
        });
        f.filter(0.0, 0.0);
        let mut prev = 0.0;
        for i in 1..200 {
            let y = f.filter(1.0, i as f64 * 0.016);
            assert!(y > prev, "monotone rise");
            assert!(y <= 1.0 + 1e-12, "no overshoot");
            prev = y;
        }
        assert!(prev > 0.99, "converged near the step, got {prev}");
    }

    #[test]
    fn higher_beta_tracks_fast_motion_closer() {
        // A fast ramp: the adaptive filter (beta > 0) must lag less than the
        // pure low-pass (beta = 0).
        let slow_cfg = OneEuroConfig {
            min_cutoff: 0.5,
            beta: 0.0,
            d_cutoff: 1.0,
        };
        let fast_cfg = OneEuroConfig {
            min_cutoff: 0.5,
            beta: 1.0,
            d_cutoff: 1.0,
        };
        let mut slow = OneEuro::new(slow_cfg);
        let mut fast = OneEuro::new(fast_cfg);
        let mut last = (0.0, 0.0, 0.0);
        for i in 0..120 {
            let t = i as f64 * 0.016;
            let x = t * 2.0; // 2 m/s ramp
            last = (x, slow.filter(x, t), fast.filter(x, t));
        }
        let (x, s, f) = last;
        assert!(
            (x - f).abs() < (x - s).abs(),
            "beta reduces lag: |{x}-{f}| < |{x}-{s}|"
        );
    }

    #[test]
    fn rejects_non_monotonic_and_non_finite_samples() {
        let mut f = OneEuro::new(OneEuroConfig::default());
        f.filter(1.0, 1.0);
        let settled = f.filter(1.0, 2.0);
        // Time going backwards: hold the estimate.
        assert_eq!(f.filter(99.0, 0.5), settled);
        // NaN input: hold the estimate.
        assert_eq!(f.filter(f64::NAN, 3.0), settled);
        // NaN never propagates.
        let y = f.filter(1.0, 4.0);
        assert!(y.is_finite());
    }

    #[test]
    fn vec3_filter_is_componentwise() {
        let mut f3 = OneEuro3::new(OneEuroConfig::default());
        let mut fx = OneEuro::new(OneEuroConfig::default());
        for i in 0..10 {
            let t = i as f64 * 0.02;
            let v = Vec3::new(i as f64 * 0.1, -1.0, 2.0);
            let out = f3.filter(v, t);
            assert_eq!(out.x, fx.filter(v.x, t));
            assert_eq!(out.y, -1.0);
            assert_eq!(out.z, 2.0);
        }
    }
}
