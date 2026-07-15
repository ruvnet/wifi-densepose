//! Feature extraction (ADR-151 Stage 3).
//!
//! Turns an anchor capture — a per-frame scalar series derived from the
//! baseline-subtracted CSI (mean amplitude or dominant-subcarrier phase) — into
//! a compact [`Features`] vector the small specialists consume. No giant model:
//! the useful signal (variance, motion, periodicity, dominant rhythm) is cheap
//! to compute and is exactly what breathing/heartbeat/posture/presence need.
//!
//! Heartbeat and breathing are tiny *repeating* disturbances in the RF field, so
//! periodicity is estimated by autocorrelation over the relevant band — the same
//! technique that fixed the firmware HR estimator (#987).

use serde::{Deserialize, Serialize};

use crate::anchor::AnchorLabel;

/// Compact per-capture (or per-window) feature vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Features {
    /// Mean of the scalar series (presence / static load).
    pub mean: f32,
    /// Variance of the series (motion / occupancy energy).
    pub variance: f32,
    /// Mean absolute first difference (instantaneous motion proxy).
    pub motion: f32,
    /// Dominant periodicity score in the breathing band [0, 1].
    pub breathing_score: f32,
    /// Dominant breathing frequency (Hz), 0 if none.
    pub breathing_hz: f32,
    /// Dominant periodicity score in the heart-rate band [0, 1].
    pub heart_score: f32,
    /// Dominant heart-rate frequency (Hz), 0 if none.
    pub heart_hz: f32,
}

/// Minimum periodicity score for a band's frequency to enter the prototype
/// embedding. Below it `autocorr_dominant` still reports its best in-band
/// peak, but for noise windows that peak is a *random* in-band frequency —
/// letting it into the embedding makes posture/anomaly prototype distances
/// noisy (ADR-152 finding, "ungated hz embedding"). The raw `breathing_hz` /
/// `heart_hz` fields stay un-gated: the breathing/heartbeat specialists apply
/// their own (stricter) `min_score` gates.
pub const EMBED_MIN_SCORE: f32 = 0.25;

impl Features {
    /// The all-zero feature vector — the well-defined result of an empty (or
    /// wholly non-finite) capture. Total by construction: downstream
    /// specialists read it as "no signal" rather than panicking or poisoning a
    /// threshold (see [`Features::from_series`]).
    pub const ZERO: Features = Features {
        mean: 0.0,
        variance: 0.0,
        motion: 0.0,
        breathing_score: 0.0,
        breathing_hz: 0.0,
        heart_score: 0.0,
        heart_hz: 0.0,
    };

    /// A fixed-length numeric embedding for nearest-prototype classifiers.
    ///
    /// The hz components are zeroed unless their periodicity score clears
    /// [`EMBED_MIN_SCORE`] — see the constant's docs.
    pub fn embedding(&self) -> [f32; 5] {
        let breathing_hz = if self.breathing_score >= EMBED_MIN_SCORE {
            self.breathing_hz
        } else {
            0.0
        };
        let heart_hz = if self.heart_score >= EMBED_MIN_SCORE {
            self.heart_hz
        } else {
            0.0
        };
        [
            self.mean,
            self.variance,
            self.motion,
            breathing_hz,
            heart_hz,
        ]
    }

    /// Squared Euclidean distance between two embeddings.
    pub fn distance2(&self, other: &Features) -> f32 {
        self.embedding()
            .iter()
            .zip(other.embedding().iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum()
    }

    /// Extract features from a per-frame scalar series sampled at `fs` Hz.
    ///
    /// **Total / fail-closed:** non-finite samples (`NaN`/`±inf`) are dropped
    /// before any statistic is computed, so a single garbage CSI frame cannot
    /// poison `mean`/`variance` into `NaN` and silently disable a persisted
    /// specialist (a `NaN` threshold makes every `>` comparison false). A
    /// series with no finite samples yields [`Features::ZERO`], exactly like
    /// the empty series. Same defensive contract as
    /// [`GeometryEmbedding`](crate::geometry_embedding::GeometryEmbedding):
    /// adversarial input degrades to "no signal", never to `NaN`.
    pub fn from_series(series: &[f32], fs: f32) -> Features {
        // Drop non-finite samples: a corrupt frame counts as no frame, not as
        // a NaN that propagates through every downstream statistic.
        let clean: Vec<f32> = series.iter().copied().filter(|v| v.is_finite()).collect();
        let n = clean.len();
        if n == 0 {
            return Features::ZERO;
        }
        let mean = clean.iter().copied().sum::<f32>() / n as f32;
        let variance = clean.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let motion = if n > 1 {
            clean.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>() / (n - 1) as f32
        } else {
            0.0
        };

        // De-mean before periodicity search.
        let centered: Vec<f32> = clean.iter().map(|v| v - mean).collect();
        let (breathing_hz, breathing_score) = autocorr_dominant(&centered, fs, 0.1, 0.6);
        let (heart_hz, heart_score) = autocorr_dominant(&centered, fs, 0.8, 3.0);

        Features {
            mean,
            variance,
            motion,
            breathing_score,
            breathing_hz,
            heart_score,
            heart_hz,
        }
    }
}

/// A labelled feature record from an enrollment anchor (ADR-151 Stage 3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorFeature {
    /// Room scope.
    pub room_id: String,
    /// Which anchor this came from.
    pub label: AnchorLabel,
    /// The extracted features.
    pub features: Features,
}

impl AnchorFeature {
    /// Build from a per-frame scalar series.
    pub fn from_series(
        room_id: impl Into<String>,
        label: AnchorLabel,
        series: &[f32],
        fs: f32,
    ) -> AnchorFeature {
        AnchorFeature {
            room_id: room_id.into(),
            label,
            features: Features::from_series(series, fs),
        }
    }
}

/// Dominant frequency in `[lo_hz, hi_hz]` via autocorrelation, with a normalized
/// peak score in `[0, 1]`. Returns `(0, 0)` if no confident peak.
///
/// The winning lag must be an **interior local maximum** of the in-band
/// autocorrelation, not a band-edge value (ADR-152 finding, "heart-band
/// leakage"): a strong out-of-band rhythm — breathing bleeding into the HR
/// band — produces a monotonic slope whose largest in-band value sits at the
/// lag floor (pinning `heart_hz` near the band's top frequency with a high
/// score). A genuine in-band periodicity peaks *inside* the band; an edge
/// maximum is leakage and is rejected.
pub fn autocorr_dominant(sig: &[f32], fs: f32, lo_hz: f32, hi_hz: f32) -> (f32, f32) {
    let n = sig.len();
    if n < 16 || fs <= 0.0 || hi_hz <= lo_hz {
        return (0.0, 0.0);
    }
    let lag_min = ((fs / hi_hz).floor() as usize).max(1);
    let lag_max = ((fs / lo_hz).ceil() as usize).min(n - 1);
    if lag_max <= lag_min + 1 {
        return (0.0, 0.0);
    }

    let r0: f32 = sig.iter().map(|v| v * v).sum();
    if r0 <= 1e-6 {
        return (0.0, 0.0);
    }

    // Autocorrelation over the band, extended one lag on each side so the
    // band edges have real neighbors for the local-max test.
    let ext_min = lag_min.saturating_sub(1).max(1);
    let ext_max = (lag_max + 1).min(n - 1);
    let acc: Vec<f32> = (ext_min..=ext_max)
        .map(|lag| (0..(n - lag)).map(|i| sig[i] * sig[i + lag]).sum())
        .collect();

    let mut best = 0.0f32;
    let mut best_lag = 0usize;
    for lag in lag_min..=lag_max {
        let idx = lag - ext_min;
        if idx == 0 || idx + 1 >= acc.len() {
            continue; // no neighbor on one side — cannot prove a local max
        }
        let v = acc[idx];
        // Interior local maximum (ties to the left tolerated for plateaus).
        if v >= acc[idx - 1] && v > acc[idx + 1] && v > best {
            best = v;
            best_lag = lag;
        }
    }
    if best_lag == 0 {
        return (0.0, 0.0);
    }
    let score = (best / r0).clamp(0.0, 1.0);
    (fs / best_lag as f32, score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine(freq_hz: f32, fs: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / fs).sin())
            .collect()
    }

    #[test]
    fn autocorr_finds_breathing_freq() {
        // 0.25 Hz (15 BPM) breathing, sampled at 15 Hz for 20 s.
        let fs = 15.0;
        let s = sine(0.25, fs, (fs * 20.0) as usize);
        let (hz, score) = autocorr_dominant(&s, fs, 0.1, 0.6);
        assert!((hz - 0.25).abs() < 0.05, "got {hz}");
        assert!(score > 0.5, "score {score}");
    }

    #[test]
    fn autocorr_finds_heart_freq() {
        // 1.45 Hz (~87 BPM), sampled at 15 Hz.
        let fs = 15.0;
        let s = sine(1.45, fs, (fs * 20.0) as usize);
        let (hz, _) = autocorr_dominant(&s, fs, 0.8, 3.0);
        assert!((hz * 60.0 - 87.0).abs() < 12.0, "got {} bpm", hz * 60.0);
    }

    #[test]
    fn features_capture_breathing() {
        let fs = 15.0;
        let s = sine(0.3, fs, 300);
        let f = Features::from_series(&s, fs);
        assert!(f.breathing_score > 0.4);
        assert!((f.breathing_hz - 0.3).abs() < 0.06);
    }

    #[test]
    fn motion_distinguishes_still_from_noisy() {
        let still = vec![1.0f32; 200];
        let noisy: Vec<f32> = (0..200)
            .map(|i| if i % 2 == 0 { 0.0 } else { 5.0 })
            .collect();
        assert!(
            Features::from_series(&still, 15.0).motion < Features::from_series(&noisy, 15.0).motion
        );
    }

    #[test]
    fn empty_series_is_safe() {
        let f = Features::from_series(&[], 15.0);
        assert_eq!(f.mean, 0.0);
        assert_eq!(f.breathing_hz, 0.0);
    }

    /// Fail-closed regression: a NaN/inf in the scalar series (corrupt CSI
    /// frame) must NOT poison the features into `NaN`/`inf`. Pre-fix, a single
    /// `NaN` made `mean`/`variance` `NaN`, which — baked into a persisted
    /// `PresenceSpecialist::threshold` — silently disabled presence detection
    /// (every `f.variance > NaN` is false). Non-finite samples are dropped.
    #[test]
    fn non_finite_samples_do_not_poison_features() {
        let f = Features::from_series(&[1.0, 2.0, f32::NAN, 4.0, f32::INFINITY, 6.0], 15.0);
        assert!(f.mean.is_finite(), "mean must stay finite, got {}", f.mean);
        assert!(f.variance.is_finite(), "variance must stay finite, got {}", f.variance);
        assert!(f.motion.is_finite(), "motion must stay finite, got {}", f.motion);
        for x in f.embedding() {
            assert!(x.is_finite(), "embedding slot non-finite: {x}");
        }
        // Mean is over the 4 finite samples {1,2,4,6} only.
        assert!((f.mean - 3.25).abs() < 1e-5, "mean over finite samples, got {}", f.mean);
        // Equivalence: dropping the non-finite samples must equal feeding only
        // the finite ones — proves the filter, not just finiteness.
        let only_finite = Features::from_series(&[1.0, 2.0, 4.0, 6.0], 15.0);
        assert_eq!(f, only_finite);
    }

    /// A series with no finite samples degrades to the all-zero `ZERO`, exactly
    /// like the empty series — never `NaN`.
    #[test]
    fn all_non_finite_series_is_zero() {
        let f = Features::from_series(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 15.0);
        assert_eq!(f, Features::ZERO);
    }

    /// ADR-152 "heart-band leakage" regression: a strong breathing rhythm must
    /// NOT register as a heart-band periodicity — its in-band autocorr maximum
    /// sits at the band edge (monotonic leak), not an interior peak.
    #[test]
    fn heart_band_rejects_breathing_leakage() {
        let fs = 20.0;
        // Pure 0.30 Hz breathing, no heart component at all.
        let s = sine(0.30, fs, (fs * 30.0) as usize);
        let (hz, score) = autocorr_dominant(&s, fs, 0.8, 3.0);
        assert!(
            score < 0.25,
            "breathing-only signal scored {score} in the heart band (hz {hz}) — \
             the lag-floor leak is back"
        );
        // The breathing band itself must still find the true rate.
        let (bhz, bscore) = autocorr_dominant(&s, fs, 0.1, 0.6);
        assert!((bhz - 0.30).abs() < 0.05, "breathing band got {bhz}");
        assert!(bscore > 0.5);
    }

    /// ADR-152 "ungated hz embedding" regression: a low-score in-band peak
    /// (noise) must NOT leak its random frequency into the prototype
    /// embedding, while a confident peak must pass through unchanged.
    #[test]
    fn embedding_gates_hz_on_score() {
        let noisy = Features {
            mean: 1.0,
            variance: 2.0,
            motion: 0.3,
            breathing_score: EMBED_MIN_SCORE - 0.05,
            breathing_hz: 0.42, // random in-band peak from a noise window
            heart_score: EMBED_MIN_SCORE - 0.05,
            heart_hz: 3.3, // breathing leakage pinned at the lag floor
        };
        let e = noisy.embedding();
        assert_eq!(e[3], 0.0, "low-score breathing_hz must be gated out");
        assert_eq!(e[4], 0.0, "low-score heart_hz must be gated out");

        let confident = Features {
            breathing_score: EMBED_MIN_SCORE + 0.3,
            heart_score: EMBED_MIN_SCORE + 0.3,
            ..noisy
        };
        let e = confident.embedding();
        assert_eq!(e[3], 0.42, "confident breathing_hz must pass through");
        assert_eq!(e[4], 3.3, "confident heart_hz must pass through");
    }
}
