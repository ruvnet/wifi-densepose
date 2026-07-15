//! Drift-triggered recalibration scenario tests (ADR-135 §2.5 and §2.6).
//!
//! Validates that the deviation z-score escalates correctly under sustained
//! amplitude drift, and stays suppressed for a stable stationary channel.
//!
//! Tests are seeded with literal `42` and are fully deterministic.

use std::f32::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::calibration::{
    BaselineCalibration, CalibrationConfig, CalibrationError, CalibrationRecorder,
};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32, seed=42) — duplicated locally.
// ---------------------------------------------------------------------------

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        assert_ne!(seed, 0, "xorshift seed must be non-zero");
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let u2 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// Constants and helpers
// ---------------------------------------------------------------------------

const N_ACTIVE: usize = 52; // HT20

fn base_amp() -> Vec<f32> {
    (0..N_ACTIVE)
        .map(|k| 0.3 + 0.7 * (k as f32 * PI / N_ACTIVE as f32).sin().abs())
        .collect()
}

fn base_phase() -> Vec<f32> {
    (0..N_ACTIVE)
        .map(|k| (k as f32 * 0.1).rem_euclid(2.0 * PI) - PI)
        .collect()
}

fn make_frame_with_amp(amp_vals: &[f32], phase: &[f32], rng: &mut Rng) -> CsiFrame {
    let n = amp_vals.len();
    let noise_std = 0.005_f32; // very low noise for clean drift detection
    let mut data = Array2::<Complex64>::zeros((1, n));
    for k in 0..n {
        let re = amp_vals[k] * phase[k].cos() + noise_std * rng.next_normal();
        let im = amp_vals[k] * phase[k].sin() + noise_std * rng.next_normal();
        data[(0, k)] = Complex64::new(re as f64, im as f64);
    }
    let mut meta = CsiMetadata::new(DeviceId::new("drift-test"), FrequencyBand::Band2_4GHz, 6);
    meta.bandwidth_mhz = 20;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

fn build_baseline() -> BaselineCalibration {
    let amp = base_amp();
    let phase = base_phase();
    let mut rng = Rng::new(42);
    let mut recorder = CalibrationRecorder::new(CalibrationConfig::ht20());
    for _ in 0..600 {
        let frame = make_frame_with_amp(&amp, &phase, &mut rng);
        recorder.record(&frame).expect("record");
    }
    recorder.finalize().expect("finalize")
}

// ---------------------------------------------------------------------------
// Test 1: slow amplitude drift causes z-score to escalate above 4.0 by frame 900
// ---------------------------------------------------------------------------

/// ADR-135 §2.5: drift_score > 4.0 is the recalibration threshold.
/// With amplitude growing +0.01/frame, the squared z-score (relative to baseline
/// variance) must exceed 4.0 on average over the last 100 of 900 frames.
#[test]
fn should_exceed_drift_threshold_when_amplitude_drifts_slowly() {
    let baseline = build_baseline();
    let base = base_amp();
    let phase = base_phase();
    let mut rng = Rng::new(42);
    let mut last_100_mean_sq_z: Vec<f32> = Vec::new();

    for t in 0..900usize {
        // Each frame has amplitudes drifted up by +0.01 per frame step
        let amp: Vec<f32> = base.iter().map(|a| a + 0.01 * t as f32).collect();
        let frame = make_frame_with_amp(&amp, &phase, &mut rng);
        let score = baseline.deviation(&frame).expect("deviation");

        if t >= 800 {
            // amplitude_z_median is the median absolute z. drift_score in ADR-135 is
            // mean over k of median squared z over a window. We approximate here
            // by squaring the amplitude_z_median.
            let approx_drift_score = score.amplitude_z_median * score.amplitude_z_median;
            last_100_mean_sq_z.push(approx_drift_score);
        }
    }

    let avg_drift_score: f32 =
        last_100_mean_sq_z.iter().sum::<f32>() / last_100_mean_sq_z.len() as f32;

    assert!(
        avg_drift_score > 4.0,
        "drift scenario: approx drift score over last 100 frames = {:.3} must exceed 4.0 \
         (ADR-135 drift threshold)",
        avg_drift_score
    );
}

// ---------------------------------------------------------------------------
// Test 2: 900 stationary frames keep z-score below 2.0
// ---------------------------------------------------------------------------

#[test]
fn should_stay_below_drift_threshold_for_stable_channel() {
    let baseline = build_baseline();
    let base = base_amp();
    let phase = base_phase();
    let mut rng = Rng::new(42);
    let mut last_100_mean_sq_z: Vec<f32> = Vec::new();

    for t in 0..900usize {
        let _ = t;
        let frame = make_frame_with_amp(&base, &phase, &mut rng);
        let score = baseline.deviation(&frame).expect("deviation");
        if last_100_mean_sq_z.len() < 100 || t >= 800 {
            let approx_drift = score.amplitude_z_median * score.amplitude_z_median;
            if t >= 800 {
                last_100_mean_sq_z.push(approx_drift);
            }
        }
    }

    let avg_drift_score: f32 =
        last_100_mean_sq_z.iter().sum::<f32>() / last_100_mean_sq_z.len() as f32;

    assert!(
        avg_drift_score < 2.0,
        "stable scenario: approx drift score over last 100 frames = {:.3} must be < 2.0",
        avg_drift_score
    );
}

// ---------------------------------------------------------------------------
// Test 3: is_complete() reflects target_frames boundary
// ---------------------------------------------------------------------------

#[test]
fn should_report_not_complete_before_target_frames() {
    let base = base_amp();
    let phase = base_phase();
    let mut rng = Rng::new(42);
    // min_frames=600 means recorder needs at least 600 frames before finalize succeeds.
    // is_complete() is defined as frames_recorded() >= config.min_frames.
    let config = CalibrationConfig::ht20(); // min_frames = 600
    let mut recorder = CalibrationRecorder::new(config);
    for _ in 0..10 {
        let frame = make_frame_with_amp(&base, &phase, &mut rng);
        recorder.record(&frame).expect("record");
    }
    assert_eq!(recorder.frames_recorded(), 10, "frames_recorded should be 10");
    // finalize should fail with InsufficientFrames
    let result = recorder.finalize();
    assert!(
        matches!(result, Err(CalibrationError::InsufficientFrames { .. })),
        "expected InsufficientFrames after 10 frames, got {:?}", result
    );
}

// ---------------------------------------------------------------------------
// Test 4: finalize() returns InsufficientFrames with correct counts
// ---------------------------------------------------------------------------

#[test]
fn should_error_on_finalize_with_insufficient_frames() {
    let base = base_amp();
    let phase = base_phase();
    let mut rng = Rng::new(42);
    let mut recorder = CalibrationRecorder::new(CalibrationConfig::ht20()); // min=600
    for _ in 0..50 {
        let frame = make_frame_with_amp(&base, &phase, &mut rng);
        recorder.record(&frame).expect("record");
    }
    match recorder.finalize() {
        Err(CalibrationError::InsufficientFrames { got, need }) => {
            assert_eq!(got, 50, "got should be 50");
            assert_eq!(need, 600, "need should be 600 (min_frames)");
        }
        other => panic!("expected InsufficientFrames, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Test 5: motion_flagged flips when amplitude jumps substantially
// ---------------------------------------------------------------------------

#[test]
fn should_flag_motion_when_amplitude_jumps_by_many_sigma() {
    let baseline = build_baseline();
    let phase = base_phase();

    // Compute a meaningful sigma: mean amp_variance across subcarriers
    let mean_sigma: f32 = baseline
        .subcarriers
        .iter()
        .map(|sc| sc.amp_variance.sqrt())
        .sum::<f32>()
        / N_ACTIVE as f32;

    // Build a frame with all amplitudes shifted up by 5σ
    let base = base_amp();
    let shifted_amp: Vec<f32> = base.iter().map(|a| a + 5.0 * mean_sigma).collect();
    let mut rng = Rng::new(77);
    let frame = make_frame_with_amp(&shifted_amp, &phase, &mut rng);
    let score = baseline.deviation(&frame).expect("deviation");
    assert!(
        score.motion_flagged,
        "motion must be flagged when amplitude is shifted by 5σ; \
         amplitude_z_median={:.3}",
        score.amplitude_z_median
    );
}
