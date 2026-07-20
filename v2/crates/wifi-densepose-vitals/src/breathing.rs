//! Respiratory rate extraction from CSI residuals.
//!
//! Uses bandpass filtering (0.1-0.5 Hz) and spectral analysis
//! to extract breathing rate from multi-subcarrier CSI data.
//!
//! The approach follows the same IIR bandpass + zero-crossing pattern
//! used by [`CoarseBreathingExtractor`](wifi_densepose_wifiscan::pipeline::CoarseBreathingExtractor)
//! in the wifiscan crate, adapted for multi-subcarrier f64 processing
//! with weighted subcarrier fusion.

use crate::types::{VitalEstimate, VitalStatus};
use std::collections::VecDeque;

/// IIR bandpass filter state (2nd-order resonator).
#[derive(Clone, Debug)]
struct IirState {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Default for IirState {
    fn default() -> Self {
        Self {
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
}

/// Respiratory rate extractor using bandpass filtering and zero-crossing analysis.
pub struct BreathingExtractor {
    /// Per-sample filtered signal history (sliding window; O(1) push/pop).
    filtered_history: VecDeque<f64>,
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Analysis window in seconds.
    window_secs: f64,
    /// Maximum subcarrier slots.
    n_subcarriers: usize,
    /// Breathing band low cutoff (Hz).
    freq_low: f64,
    /// Breathing band high cutoff (Hz).
    freq_high: f64,
    /// IIR filter state.
    filter_state: IirState,
}

impl BreathingExtractor {
    /// Create a new breathing extractor.
    ///
    /// - `n_subcarriers`: number of subcarrier channels.
    /// - `sample_rate`: input sample rate in Hz.
    /// - `window_secs`: analysis window length in seconds (default: 30).
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(n_subcarriers: usize, sample_rate: f64, window_secs: f64) -> Self {
        let capacity = (sample_rate * window_secs) as usize;
        Self {
            filtered_history: VecDeque::with_capacity(capacity),
            sample_rate,
            window_secs,
            n_subcarriers,
            freq_low: 0.1,
            freq_high: 0.5,
            filter_state: IirState::default(),
        }
    }

    /// Create with ESP32 defaults (56 subcarriers, 100 Hz, 30 s window).
    #[must_use]
    pub fn esp32_default() -> Self {
        Self::new(56, 100.0, 30.0)
    }

    /// Extract respiratory rate from a vector of per-subcarrier residuals.
    ///
    /// - `residuals`: amplitude residuals from the preprocessor.
    /// - `weights`: per-subcarrier attention weights (higher = more
    ///   body-sensitive). If shorter than `residuals`, missing weights
    ///   default to uniform.
    ///
    /// Returns a `VitalEstimate` with the breathing rate in BPM, or
    /// `None` if insufficient history has been accumulated.
    pub fn extract(&mut self, residuals: &[f64], weights: &[f64]) -> Option<VitalEstimate> {
        let n = residuals.len().min(self.n_subcarriers);
        if n == 0 {
            return None;
        }

        // Weighted fusion of subcarrier residuals (normalized — see
        // `fuse_weighted_residuals`).
        let weighted_signal = fuse_weighted_residuals(residuals, weights, n);

        // Apply IIR bandpass filter
        let filtered = self.bandpass_filter(weighted_signal);

        // Defense-in-depth: never let a non-finite filter output (e.g. a
        // diverged resonator pole at a pathological sample rate) enter the
        // history buffer. Mirrors ADR-154 §3 / ADR-157 §A3.
        if !filtered.is_finite() {
            return None;
        }

        // Append to history, enforce window limit. `VecDeque` gives O(1)
        // push_back + pop_front for the sliding window (was a `Vec` with an
        // O(n) `remove(0)` per sample — ADR-157 §A1).
        self.filtered_history.push_back(filtered);
        let max_len = (self.sample_rate * self.window_secs) as usize;
        if self.filtered_history.len() > max_len {
            self.filtered_history.pop_front();
        }

        // Need at least 10 seconds of data
        let min_samples = (self.sample_rate * 10.0) as usize;
        if self.filtered_history.len() < min_samples {
            return None;
        }

        // Zero-crossing rate -> frequency. `make_contiguous` rotates the ring
        // buffer in place once so the slice helpers below can borrow it.
        let history = self.filtered_history.make_contiguous();
        let crossings = count_zero_crossings(history);
        let duration_s = history.len() as f64 / self.sample_rate;
        let frequency_hz = crossings as f64 / (2.0 * duration_s);

        // Validate frequency is within the breathing band
        if frequency_hz < self.freq_low || frequency_hz > self.freq_high {
            return None;
        }

        let bpm = frequency_hz * 60.0;
        let confidence = compute_confidence(history);

        let status = if confidence >= 0.7 {
            VitalStatus::Valid
        } else if confidence >= 0.4 {
            VitalStatus::Degraded
        } else {
            VitalStatus::Unreliable
        };

        Some(VitalEstimate {
            value_bpm: bpm,
            confidence,
            status,
        })
    }

    /// 2nd-order IIR bandpass filter using a resonator topology.
    ///
    /// y[n] = (1-r)*(x[n] - x[n-2]) + 2*r*cos(w0)*y[n-1] - r^2*y[n-2]
    fn bandpass_filter(&mut self, input: f64) -> f64 {
        let state = &mut self.filter_state;

        let omega_low = 2.0 * std::f64::consts::PI * self.freq_low / self.sample_rate;
        let omega_high = 2.0 * std::f64::consts::PI * self.freq_high / self.sample_rate;
        let bw = omega_high - omega_low;
        let center = f64::midpoint(omega_low, omega_high);

        // Clamp the resonator pole radius into a stable range. The pole
        // magnitude is `|r|`; stability needs `|r| < 1`. When `bw` exceeds 4
        // (a very low `fs` relative to the band width) `1 - bw/2` drops below
        // -1, pushing the pole outside the unit circle and diverging the filter
        // exponentially to ±inf. (A merely-negative `r` with `|r| < 1` is still
        // stable.) The clamp keeps the pole inside the unit circle for any
        // sample-rate / band-edge configuration (ADR-157 §A3).
        let r = (1.0 - bw / 2.0).clamp(0.0, 0.9999);
        let cos_w0 = center.cos();

        let output =
            (1.0 - r) * (input - state.x2) + 2.0 * r * cos_w0 * state.y1 - r * r * state.y2;

        // Self-healing non-finite guard (ADR-158 §A1). A single non-finite
        // sample — a NaN/inf residual from a corrupt CSI frame, or a transient
        // overflow — would otherwise be stored into `y1`/`y2` and poison the
        // resonator recurrence *permanently*: every subsequent output stays
        // NaN, the `extract()` finite-check drops it, and the history buffer
        // never refills, so breathing extraction is dead until `reset()`.
        // Resetting the filter state here lets the resonator recover on the next
        // clean frame; the 0.0 we return for this frame is still dropped by the
        // caller's `is_finite()` check, so no spurious sample enters history.
        if !output.is_finite() {
            *state = IirState::default();
            return 0.0;
        }

        state.x2 = state.x1;
        state.x1 = input;
        state.y2 = state.y1;
        state.y1 = output;

        output
    }

    /// Reset all filter state and history.
    pub fn reset(&mut self) {
        self.filtered_history.clear();
        self.filter_state = IirState::default();
    }

    /// Current number of samples in the history buffer.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.filtered_history.len()
    }

    /// Breathing band cutoff frequencies.
    #[must_use]
    pub fn band(&self) -> (f64, f64) {
        (self.freq_low, self.freq_high)
    }
}

/// Fuse the first `n` per-subcarrier residuals into a single scalar using
/// the supplied attention `weights`, normalized by the sum of the
/// **effective** weights actually used.
///
/// Missing weights (when `weights.len() < n`) default to the uniform weight
/// `1/n`. Normalizing by `Σ(effective weights)` is what makes a partial
/// `weights` slice safe: without it, supplied entries (used raw) and the
/// uniform tail are summed at two different scales, silently mis-scaling the
/// breathing signal. Mirrors `heartrate::compute_phase_coherence_signal`
/// (`weighted_sum / weight_total`). (ADR-157 §A2)
fn fuse_weighted_residuals(residuals: &[f64], weights: &[f64], n: usize) -> f64 {
    let uniform_w = 1.0 / n as f64;
    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;
    for (i, &r) in residuals.iter().enumerate().take(n) {
        let w = weights.get(i).copied().unwrap_or(uniform_w);
        weighted_sum += r * w;
        weight_total += w;
    }
    if weight_total.abs() > 1e-15 {
        weighted_sum / weight_total
    } else {
        0.0
    }
}

/// Count zero crossings in a signal.
fn count_zero_crossings(signal: &[f64]) -> usize {
    signal.windows(2).filter(|w| w[0] * w[1] < 0.0).count()
}

/// Compute confidence in the breathing estimate based on signal regularity.
fn compute_confidence(history: &[f64]) -> f64 {
    if history.len() < 4 {
        return 0.0;
    }

    let n = history.len() as f64;
    let mean: f64 = history.iter().sum::<f64>() / n;
    let variance: f64 = history.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;

    if variance < 1e-15 {
        return 0.0;
    }

    let peak = history.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
    let noise = variance.sqrt();

    let snr = if noise > 1e-15 { peak / noise } else { 0.0 };

    // Map SNR to [0, 1] confidence
    (snr / 5.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_data_returns_none() {
        let mut ext = BreathingExtractor::new(4, 10.0, 30.0);
        assert!(ext.extract(&[], &[]).is_none());
    }

    #[test]
    fn insufficient_history_returns_none() {
        let mut ext = BreathingExtractor::new(2, 10.0, 30.0);
        // Just a few frames are not enough
        for _ in 0..5 {
            assert!(ext.extract(&[1.0, 2.0], &[0.5, 0.5]).is_none());
        }
    }

    #[test]
    fn zero_crossings_count() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert_eq!(count_zero_crossings(&signal), 4);
    }

    #[test]
    fn zero_crossings_constant() {
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        assert_eq!(count_zero_crossings(&signal), 0);
    }

    #[test]
    fn sinusoidal_breathing_detected() {
        let sample_rate = 10.0;
        let mut ext = BreathingExtractor::new(1, sample_rate, 60.0);
        let breathing_freq = 0.25; // 15 BPM

        // Generate 60 seconds of sinusoidal breathing signal
        for i in 0..600 {
            let t = i as f64 / sample_rate;
            let signal = (2.0 * std::f64::consts::PI * breathing_freq * t).sin();
            ext.extract(&[signal], &[1.0]);
        }

        let result = ext.extract(&[0.0], &[1.0]);
        if let Some(est) = result {
            // Should be approximately 15 BPM (0.25 Hz * 60)
            assert!(
                est.value_bpm > 5.0 && est.value_bpm < 40.0,
                "estimated BPM should be in breathing range: {}",
                est.value_bpm,
            );
            assert!(est.confidence > 0.0, "confidence should be > 0");
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut ext = BreathingExtractor::new(2, 10.0, 30.0);
        ext.extract(&[1.0, 2.0], &[0.5, 0.5]);
        assert!(ext.history_len() > 0);
        ext.reset();
        assert_eq!(ext.history_len(), 0);
    }

    #[test]
    fn band_returns_correct_values() {
        let ext = BreathingExtractor::new(1, 10.0, 30.0);
        let (low, high) = ext.band();
        assert!((low - 0.1).abs() < f64::EPSILON);
        assert!((high - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_zero_for_flat_signal() {
        let history = vec![0.0; 100];
        let conf = compute_confidence(&history);
        assert!((conf - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn confidence_positive_for_oscillating_signal() {
        let history: Vec<f64> = (0..100).map(|i| (i as f64 * 0.5).sin()).collect();
        let conf = compute_confidence(&history);
        assert!(conf > 0.0);
    }

    #[test]
    fn esp32_default_creates_correctly() {
        let ext = BreathingExtractor::esp32_default();
        assert_eq!(ext.n_subcarriers, 56);
    }

    /// ADR-157 §A2 bug-catching test.
    ///
    /// With `residuals = [1.0; 8]` and `weights = [10.0, 10.0]` (len 2 < n=8),
    /// the supplied weights (10.0) and the uniform-fallback tail (1/8) are at
    /// two different scales. The correct, normalized fusion divides by the sum
    /// of the *effective* weights, so the fused value must equal the
    /// renormalized weighted mean of the residuals = 1.0 (all residuals equal
    /// 1.0). The OLD code returned the un-normalized sum
    /// (`2*10 + 6*0.125 = 20.75`), so this asserts the fix.
    #[test]
    fn partial_weights_are_renormalized_not_scale_mixed() {
        let residuals = [1.0_f64; 8];
        let weights = [10.0_f64, 10.0];
        let fused = fuse_weighted_residuals(&residuals, &weights, 8);

        // Renormalized weighted mean of equal residuals is exactly the residual
        // value, regardless of the weight scale.
        assert!(
            (fused - 1.0).abs() < 1e-12,
            "partial weights must renormalize to the weighted mean (1.0), got {fused}"
        );

        // Explicitly pin that we are NOT returning the old scale-mixed sum.
        let old_scale_mixed_sum: f64 = 2.0 * 10.0 + 6.0 * (1.0 / 8.0);
        assert!(
            (fused - old_scale_mixed_sum).abs() > 1.0,
            "fused value must not equal the old un-normalized sum {old_scale_mixed_sum}"
        );
    }

    /// ADR-157 §A2: with differing residual values, the normalized fusion is a
    /// proper weighted average dominated by the high-weight entries.
    #[test]
    fn partial_weights_fusion_is_weighted_average() {
        // Two heavily-weighted residuals of 2.0, the rest (uniform) of 0.0.
        let residuals = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let weights = [10.0_f64, 10.0];
        let fused = fuse_weighted_residuals(&residuals, &weights, 8);
        // weighted_sum = 2*10*2 ... = 40; weight_total = 20 + 6*0.125 = 20.75
        let expected = (2.0 * 10.0 + 2.0 * 10.0) / (20.0 + 6.0 * 0.125);
        assert!(
            (fused - expected).abs() < 1e-12,
            "expected weighted average {expected}, got {fused}"
        );
        // Must lie within the residual range [0, 2] — a scale-mixed sum would not.
        assert!((0.0..=2.0).contains(&fused), "weighted average must be in-range: {fused}");
    }

    /// ADR-158 §A1 bug-catching test: a single non-finite residual must NOT
    /// permanently poison the IIR filter state.
    ///
    /// The resonator recurrence stores `y[n]` into the filter state. Before the
    /// fix, one NaN/inf residual produced a NaN `output`, the `extract()`
    /// finite-guard dropped that frame from history — but the NaN was already
    /// latched into `state.y1`/`y2`, so every subsequent output stayed NaN, the
    /// finite-guard rejected it too, and the history buffer never refilled.
    /// Breathing extraction was then dead until `reset()`. A control run on the
    /// same clean signal yields 15 BPM (0.25 Hz); after a leading NaN frame the
    /// OLD code returned `None` with `history_len() == 0` forever. This test
    /// asserts recovery (FAILS on the old code, verified by reverting the
    /// `bandpass_filter` self-heal).
    #[test]
    fn nan_frame_does_not_permanently_poison_filter() {
        let sr = 10.0;
        let feed_clean = |ext: &mut BreathingExtractor| {
            let mut last = None;
            for i in 0..600 {
                let t = i as f64 / sr;
                let s = (2.0 * std::f64::consts::PI * 0.25 * t).sin();
                last = ext.extract(&[s], &[1.0]);
            }
            last
        };

        // Control: clean signal accumulates history and detects ~15 BPM.
        let mut control = BreathingExtractor::new(1, sr, 60.0);
        let control_res = feed_clean(&mut control);
        assert!(control.history_len() > 0);
        assert!(control_res.is_some(), "control clean run must produce an estimate");

        // A leading NaN frame must not kill the extractor.
        let mut ext = BreathingExtractor::new(1, sr, 60.0);
        ext.extract(&[f64::NAN], &[1.0]);
        let res = feed_clean(&mut ext);
        assert!(
            ext.history_len() > 0,
            "extractor must recover and refill history after a NaN frame (got {})",
            ext.history_len()
        );
        assert!(res.is_some(), "extractor must recover an estimate after a NaN frame");
    }

    /// ADR-158 §A1: a mid-stream `inf` must not freeze the history buffer.
    #[test]
    fn inf_mid_stream_does_not_freeze_history() {
        let sr = 10.0;
        let mut ext = BreathingExtractor::new(1, sr, 60.0);
        let clean = |ext: &mut BreathingExtractor, count: usize| {
            for i in 0..count {
                let t = i as f64 / sr;
                let s = (2.0 * std::f64::consts::PI * 0.25 * t).sin();
                ext.extract(&[s], &[1.0]);
            }
        };
        clean(&mut ext, 300);
        let before = ext.history_len();
        assert!(before > 0);
        ext.extract(&[f64::INFINITY], &[1.0]); // poison mid-stream
        clean(&mut ext, 600);
        assert!(
            ext.history_len() > before,
            "history must keep growing after an inf frame (before={}, after={})",
            before,
            ext.history_len()
        );
    }

    /// ADR-157 §A3 bug-catching test. Divergence needs the pole magnitude
    /// `|r| >= 1`, i.e. `bw >= 4`. At `fs = 0.5` Hz with the band widened to
    /// 0.1-0.9 Hz, `bw = 2*pi*(0.9-0.1)/0.5 = 10.05`, so the OLD pole radius
    /// `r = 1 - bw/2 = -4.03` has `|r| = 4.03 > 1` and the filter blows up
    /// exponentially, overflowing to ±inf within ~600 unit-step frames. The
    /// clamp + finite-guard keep every accumulated sample finite. This FAILS on
    /// the old code (verified by reverting).
    #[test]
    fn low_sample_rate_filter_stays_finite() {
        let mut ext = BreathingExtractor::new(4, 0.5, 3600.0);
        ext.freq_low = 0.1;
        ext.freq_high = 0.9;
        // Feed a unit step for 600 frames — enough for the un-clamped resonator
        // to overflow to inf.
        for _ in 0..600 {
            ext.extract(&[1.0, 1.0, 1.0, 1.0], &[0.25, 0.25, 0.25, 0.25]);
        }
        assert!(ext.history_len() > 0, "history should accumulate");
        for (i, &v) in ext.filtered_history.iter().enumerate() {
            assert!(v.is_finite(), "filtered_history[{i}] must be finite, got {v}");
        }
    }
}
