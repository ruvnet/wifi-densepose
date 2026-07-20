//! Deterministic synthetic channel tests for the empty-room baseline calibration
//! module (ADR-135).
//!
//! Validates Welford online statistics, deviation scoring, and per-PHY-tier
//! subcarrier counts. Tests are seeded with literal `42` via xorshift32 and are
//! fully deterministic.
//!
//! Run (compile-only):
//!   cargo test -p wifi-densepose-signal --no-default-features --tests --no-run

use std::f32::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::calibration::{
    BaselineCalibration, CalibrationConfig, CalibrationRecorder,
};

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32, seed=42) — duplicated locally per ADR-135
// constraint: do not refactor existing test helpers.
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

    /// Sample N(0,1) via Box-Muller (always consumes two draws).
    fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let u2 = (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        r * theta.cos()
    }
}

// ---------------------------------------------------------------------------
// Tier parameters
// ---------------------------------------------------------------------------

struct TierSpec {
    label: &'static str,
    n_active: usize,     // active (non-pilot) subcarriers passed in frame
    bandwidth_mhz: u16,
    config: CalibrationConfig,
}

fn ht20_spec() -> TierSpec {
    TierSpec { label: "HT20", n_active: 52, bandwidth_mhz: 20, config: CalibrationConfig::ht20() }
}
fn ht40_spec() -> TierSpec {
    TierSpec { label: "HT40", n_active: 114, bandwidth_mhz: 40, config: CalibrationConfig::ht40() }
}
fn he20_spec() -> TierSpec {
    // Issue #1009 §1b: real HE20 frames carry all 256 FFT bins (242 data +
    // pilots/guards/DC), and the recorder now records all 256 (he20().num_active
    // == 256). Feed 256-bin frames to match the wire format.
    TierSpec { label: "HE20", n_active: 256, bandwidth_mhz: 20, config: CalibrationConfig::he20() }
}

// ---------------------------------------------------------------------------
// Ground-truth per-subcarrier channel parameters
// ---------------------------------------------------------------------------

fn ground_truth_amp(n: usize) -> Vec<f32> {
    (0..n).map(|k| 0.3 + 0.7 * (k as f32 * PI / n as f32).sin().abs()).collect()
}

fn ground_truth_phase(n: usize) -> Vec<f32> {
    (0..n).map(|k| (k as f32 * 0.1).rem_euclid(2.0 * PI) - PI).collect()
}

// ---------------------------------------------------------------------------
// CSI frame builder helpers
// ---------------------------------------------------------------------------

fn make_stationary_frame(
    bandwidth_mhz: u16,
    n_active: usize,
    amp: &[f32],
    phase: &[f32],
    snr_db: f32,
    rng: &mut Rng,
) -> CsiFrame {
    assert_eq!(amp.len(), n_active);
    let signal_power: f32 = amp.iter().map(|a| a * a).sum::<f32>() / n_active as f32;
    let noise_power = signal_power / 10_f32.powf(snr_db / 10.0);
    let noise_std = (noise_power / 2.0).sqrt();

    let mut data = Array2::<Complex64>::zeros((1, n_active));
    for k in 0..n_active {
        let re = amp[k] * phase[k].cos() + noise_std * rng.next_normal();
        let im = amp[k] * phase[k].sin() + noise_std * rng.next_normal();
        data[(0, k)] = Complex64::new(re as f64, im as f64);
    }
    let mut meta = CsiMetadata::new(DeviceId::new("test"), FrequencyBand::Band2_4GHz, 6);
    meta.bandwidth_mhz = bandwidth_mhz;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

/// Build a frame where subcarrier amplitudes are shifted up by `shift_sigma * sigma`.
fn make_perturbed_frame(
    bandwidth_mhz: u16,
    n_active: usize,
    amp: &[f32],
    phase: &[f32],
    amp_sigma: f32,
    perturb_indices: &[usize],
    shift_sigma: f32,
    rng: &mut Rng,
) -> CsiFrame {
    let noise_std = 0.001_f32;
    let mut data = Array2::<Complex64>::zeros((1, n_active));
    for k in 0..n_active {
        let extra = if perturb_indices.contains(&k) { shift_sigma * amp_sigma } else { 0.0 };
        let a = amp[k] + extra;
        let re = a * phase[k].cos() + noise_std * rng.next_normal();
        let im = a * phase[k].sin() + noise_std * rng.next_normal();
        data[(0, k)] = Complex64::new(re as f64, im as f64);
    }
    let mut meta = CsiMetadata::new(DeviceId::new("test"), FrequencyBand::Band2_4GHz, 6);
    meta.bandwidth_mhz = bandwidth_mhz;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

// ---------------------------------------------------------------------------
// Helper: build a finalised baseline from 600 stationary frames at SNR=30 dB
// ---------------------------------------------------------------------------

fn build_baseline(spec: &TierSpec) -> BaselineCalibration {
    let amp = ground_truth_amp(spec.n_active);
    let phase = ground_truth_phase(spec.n_active);
    let mut rng = Rng::new(42);
    let mut recorder = CalibrationRecorder::new(spec.config.clone());
    for _ in 0..600 {
        let frame = make_stationary_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
        );
        recorder.record(&frame).expect("record should succeed");
    }
    recorder.finalize().expect("finalize should succeed with 600 frames")
}

// ---------------------------------------------------------------------------
// Tests — HT20
// ---------------------------------------------------------------------------

mod ht20 {
    use super::*;

    #[test]
    fn should_record_600_frames_when_600_fed() {
        let spec = ht20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let mut rng = Rng::new(42);
        let mut recorder = CalibrationRecorder::new(spec.config.clone());
        for _ in 0..600 {
            let frame = make_stationary_frame(
                spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
            );
            recorder.record(&frame).expect("record should succeed");
        }
        assert_eq!(
            recorder.frames_recorded(), 600,
            "HT20: frames_recorded() should equal 600"
        );
    }

    #[test]
    fn should_finalize_with_amp_mean_within_tolerance_of_ground_truth() {
        let spec = ht20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let baseline = build_baseline(&spec);
        let tol = 0.05_f32;
        for k in 0..spec.n_active {
            let got = baseline.subcarriers[k].amp_mean;
            let expected = amp[k];
            assert!(
                (got - expected).abs() < tol,
                "HT20 amp_mean[{}]: got={:.4} expected={:.4} tol={:.4}",
                k, got, expected, tol
            );
        }
    }

    #[test]
    fn should_have_positive_amp_variance_after_finalize() {
        let spec = ht20_spec();
        let baseline = build_baseline(&spec);
        for k in 0..spec.n_active {
            assert!(
                baseline.subcarriers[k].amp_variance > 0.0,
                "HT20 amp_variance[{}] must be positive",
                k
            );
        }
    }

    #[test]
    fn should_have_small_amp_variance_for_stationary_channel() {
        let spec = ht20_spec();
        let baseline = build_baseline(&spec);
        for k in 0..spec.n_active {
            assert!(
                baseline.subcarriers[k].amp_variance < 0.1,
                "HT20 amp_variance[{}]={:.6} must be < 0.1",
                k, baseline.subcarriers[k].amp_variance
            );
        }
    }

    #[test]
    fn should_have_tight_phase_dispersion_for_stationary_channel() {
        let spec = ht20_spec();
        let baseline = build_baseline(&spec);
        for k in 0..spec.n_active {
            assert!(
                baseline.subcarriers[k].phase_dispersion < 0.05,
                "HT20 phase_dispersion[{}]={:.6} must be < 0.05",
                k, baseline.subcarriers[k].phase_dispersion
            );
        }
    }

    #[test]
    fn should_not_flag_motion_for_stationary_frame() {
        let spec = ht20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let baseline = build_baseline(&spec);
        let mut rng = Rng::new(999);
        let frame = make_stationary_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
        );
        let score = baseline.deviation(&frame).expect("deviation should succeed");
        assert!(
            score.amplitude_z_median < 1.5,
            "HT20 stationary: amplitude_z_median={:.3} must be < 1.5",
            score.amplitude_z_median
        );
        assert!(
            !score.motion_flagged,
            "HT20 stationary: motion_flagged must be false"
        );
    }

    #[test]
    fn should_flag_motion_for_3sigma_perturbed_frame() {
        let spec = ht20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let baseline = build_baseline(&spec);
        // Use mean amp_variance as the sigma estimate
        let amp_sigma: f32 = baseline
            .subcarriers
            .iter()
            .map(|sc| sc.amp_variance.sqrt())
            .sum::<f32>()
            / spec.n_active as f32;
        let perturb_indices: Vec<usize> = (0..spec.n_active).collect();
        let mut rng = Rng::new(999);
        let frame = make_perturbed_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, amp_sigma,
            &perturb_indices, 3.0, &mut rng,
        );
        let score = baseline.deviation(&frame).expect("deviation should succeed");
        assert!(
            score.amplitude_z_median > 2.5,
            "HT20 perturbed: amplitude_z_median={:.3} must be > 2.5",
            score.amplitude_z_median
        );
        assert!(
            score.motion_flagged,
            "HT20 perturbed: motion_flagged must be true for 3σ perturbation"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — HT40
// ---------------------------------------------------------------------------

mod ht40 {
    use super::*;

    #[test]
    fn should_record_600_frames_when_600_fed() {
        let spec = ht40_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let mut rng = Rng::new(42);
        let mut recorder = CalibrationRecorder::new(spec.config.clone());
        for _ in 0..600 {
            let frame = make_stationary_frame(
                spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
            );
            recorder.record(&frame).expect("record should succeed");
        }
        assert_eq!(recorder.frames_recorded(), 600, "HT40: frames_recorded() should equal 600");
    }

    #[test]
    fn should_finalize_with_amp_mean_within_tolerance() {
        let spec = ht40_spec();
        let amp = ground_truth_amp(spec.n_active);
        let baseline = build_baseline(&spec);
        let tol = 0.05_f32;
        for k in 0..spec.n_active {
            let got = baseline.subcarriers[k].amp_mean;
            let expected = amp[k];
            assert!(
                (got - expected).abs() < tol,
                "HT40 amp_mean[{}]: got={:.4} expected={:.4} tol={:.4}",
                k, got, expected, tol
            );
        }
    }

    #[test]
    fn should_have_tight_phase_dispersion_for_stationary_channel() {
        let spec = ht40_spec();
        let baseline = build_baseline(&spec);
        for k in 0..spec.n_active {
            assert!(
                baseline.subcarriers[k].phase_dispersion < 0.05,
                "HT40 phase_dispersion[{}]={:.6} must be < 0.05",
                k, baseline.subcarriers[k].phase_dispersion
            );
        }
    }

    #[test]
    fn should_not_flag_motion_for_stationary_frame() {
        let spec = ht40_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let baseline = build_baseline(&spec);
        let mut rng = Rng::new(999);
        let frame = make_stationary_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
        );
        let score = baseline.deviation(&frame).expect("deviation should succeed");
        assert!(
            !score.motion_flagged,
            "HT40 stationary: motion_flagged must be false"
        );
    }

    #[test]
    fn should_flag_motion_for_3sigma_perturbed_frame() {
        let spec = ht40_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let baseline = build_baseline(&spec);
        let amp_sigma: f32 = baseline
            .subcarriers
            .iter()
            .map(|sc| sc.amp_variance.sqrt())
            .sum::<f32>()
            / spec.n_active as f32;
        let perturb_indices: Vec<usize> = (0..spec.n_active).collect();
        let mut rng = Rng::new(999);
        let frame = make_perturbed_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, amp_sigma,
            &perturb_indices, 3.0, &mut rng,
        );
        let score = baseline.deviation(&frame).expect("deviation should succeed");
        assert!(
            score.motion_flagged,
            "HT40 perturbed: motion_flagged must be true for 3σ perturbation"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — HE20
// ---------------------------------------------------------------------------

mod he20 {
    use super::*;

    #[test]
    fn should_record_600_frames_when_600_fed() {
        let spec = he20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let mut rng = Rng::new(42);
        let mut recorder = CalibrationRecorder::new(spec.config.clone());
        for _ in 0..600 {
            let frame = make_stationary_frame(
                spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
            );
            recorder.record(&frame).expect("record should succeed");
        }
        assert_eq!(recorder.frames_recorded(), 600, "HE20: frames_recorded() should equal 600");
    }

    #[test]
    fn should_finalize_with_amp_mean_within_tolerance() {
        let spec = he20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let baseline = build_baseline(&spec);
        let tol = 0.05_f32;
        for k in 0..spec.n_active {
            let got = baseline.subcarriers[k].amp_mean;
            let expected = amp[k];
            assert!(
                (got - expected).abs() < tol,
                "HE20 amp_mean[{}]: got={:.4} expected={:.4} tol={:.4}",
                k, got, expected, tol
            );
        }
    }

    #[test]
    fn should_have_tight_phase_dispersion_for_stationary_channel() {
        let spec = he20_spec();
        let baseline = build_baseline(&spec);
        for k in 0..spec.n_active {
            assert!(
                baseline.subcarriers[k].phase_dispersion < 0.05,
                "HE20 phase_dispersion[{}]={:.6} must be < 0.05",
                k, baseline.subcarriers[k].phase_dispersion
            );
        }
    }

    #[test]
    fn should_not_flag_motion_for_stationary_frame() {
        let spec = he20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let baseline = build_baseline(&spec);
        let mut rng = Rng::new(999);
        let frame = make_stationary_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, 30.0, &mut rng,
        );
        let score = baseline.deviation(&frame).expect("deviation should succeed");
        assert!(
            !score.motion_flagged,
            "HE20 stationary: motion_flagged must be false"
        );
    }

    #[test]
    fn should_flag_motion_for_3sigma_perturbed_frame() {
        let spec = he20_spec();
        let amp = ground_truth_amp(spec.n_active);
        let phase = ground_truth_phase(spec.n_active);
        let baseline = build_baseline(&spec);
        let amp_sigma: f32 = baseline
            .subcarriers
            .iter()
            .map(|sc| sc.amp_variance.sqrt())
            .sum::<f32>()
            / spec.n_active as f32;
        let perturb_indices: Vec<usize> = (0..spec.n_active).collect();
        let mut rng = Rng::new(999);
        let frame = make_perturbed_frame(
            spec.bandwidth_mhz, spec.n_active, &amp, &phase, amp_sigma,
            &perturb_indices, 3.0, &mut rng,
        );
        let score = baseline.deviation(&frame).expect("deviation should succeed");
        assert!(
            score.motion_flagged,
            "HE20 perturbed: motion_flagged must be true for 3σ perturbation"
        );
    }
}
