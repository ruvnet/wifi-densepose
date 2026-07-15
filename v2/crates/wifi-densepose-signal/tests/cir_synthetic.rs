//! Deterministic synthetic channel tests for CIR estimation (ADR-134).
//!
//! Validates sparse ISTA recovery against forward-projected multi-tap channels
//! at HT20, HT40, and HE20 hardware tiers.
//!
//! Tests are seeded with literal `42` and must be fully deterministic.
//! JSON fixtures are written to `tests/data/cir_synthetic_*.json` for the
//! witness agent to replay.

#![cfg(feature = "cir")]

use std::f32::consts::PI;

use ndarray::Array2;
use num_complex::Complex64;
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::cir::{CirConfig, CirEstimator};

// ---------------------------------------------------------------------------
// Minimal deterministic PRNG (xorshift32, seeded = 42)
// Avoids pulling in rand/rand_chacha as new dev-dependencies.
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
// Channel parameters shared across tiers
// ---------------------------------------------------------------------------

struct TapSpec {
    delay_s: f64,
    amplitude: f32,
    phase: f32,
}

/// The three ground-truth taps used across all tiers.
fn ground_truth_taps() -> [TapSpec; 3] {
    [
        TapSpec { delay_s: 10e-9, amplitude: 1.0, phase: PI / 4.0 },
        TapSpec { delay_s: 80e-9, amplitude: 0.6, phase: PI },
        TapSpec { delay_s: 180e-9, amplitude: 0.3, phase: -PI / 3.0 },
    ]
}

// ---------------------------------------------------------------------------
// CSI forward-projection helper
//   H[k] = sum_p  a_p * exp(-j * 2*pi * k * delta_f * tau_p)
//
// Parameters:
//   k_active     — number of active (non-pilot) subcarriers
//   delta_f_hz   — subcarrier spacing in Hz
//   taps         — (delay_s, complex_amplitude) pairs
//   snr_db       — additive white Gaussian noise to add after projection
//   rng          — seeded deterministic PRNG
//
// Returns a flat Vec<Complex64> length = k_active.
// ---------------------------------------------------------------------------

fn forward_project(
    k_active: usize,
    delta_f_hz: f64,
    taps: &[(f64, num_complex::Complex<f32>)],
    snr_db: f32,
    rng: &mut Rng,
) -> Vec<Complex64> {
    // Signal power = sum of |a_p|^2
    let signal_power: f32 = taps.iter().map(|(_, a)| a.norm_sqr()).sum();
    let noise_power = signal_power / 10_f32.powf(snr_db / 10.0);
    let noise_std = (noise_power / 2.0).sqrt(); // per I/Q component

    (0..k_active)
        .map(|k| {
            let h_signal: num_complex::Complex<f32> = taps
                .iter()
                .map(|(tau, alpha)| {
                    let angle = -2.0 * PI as f64 * k as f64 * delta_f_hz * tau;
                    let phasor = num_complex::Complex::new(angle.cos() as f32, angle.sin() as f32);
                    alpha * phasor
                })
                .sum();

            // Add AWGN (seeded deterministically)
            let n_i = noise_std * rng.next_normal();
            let n_q = noise_std * rng.next_normal();
            let h_noisy = h_signal + num_complex::Complex::new(n_i, n_q);
            Complex64::new(h_noisy.re as f64, h_noisy.im as f64)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// CsiFrame construction helper
// ---------------------------------------------------------------------------

fn make_frame(bandwidth_mhz: u16, num_subcarriers: usize, csi: Vec<Complex64>) -> CsiFrame {
    assert_eq!(csi.len(), num_subcarriers);
    let mut data = Array2::zeros((1, num_subcarriers));
    for (k, &val) in csi.iter().enumerate() {
        data[(0, k)] = val;
    }
    let mut meta = CsiMetadata::new(
        DeviceId::new("test-device"),
        FrequencyBand::Band2_4GHz,
        6,
    );
    meta.bandwidth_mhz = bandwidth_mhz;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

// ---------------------------------------------------------------------------
// Fixture serialisation helper
// ---------------------------------------------------------------------------

fn save_fixture(path: &str, k_active: usize, csi: &[Complex64], expected_dominant_idx: usize) {
    use std::io::Write as IoWrite;
    let entries: Vec<serde_json::Value> = csi
        .iter()
        .map(|c| serde_json::json!({"re": c.re, "im": c.im}))
        .collect();
    let doc = serde_json::json!({
        "k_active": k_active,
        "expected_dominant_tap_idx": expected_dominant_idx,
        "csi": entries,
    });
    let text = serde_json::to_string_pretty(&doc).expect("serialise fixture");
    let mut f = std::fs::File::create(path).expect("create fixture file");
    f.write_all(text.as_bytes()).expect("write fixture");
}

// ---------------------------------------------------------------------------


// Shared test logic: inject 3-tap channel, run estimator, assert
// ---------------------------------------------------------------------------

fn run_3tap_test(label: &str, cfg: CirConfig, bandwidth_mhz: u16, dominant_ratio_floor: f32, fixture_path: &str) {
    let taps_spec = ground_truth_taps();
    // Per-tier subcarrier spacing: BW / N. HT20/HT40 → 312.5 kHz; HE20 → 78.125 kHz.
    let delta_f_hz = cfg.bandwidth_hz / cfg.num_subcarriers as f64;
    let k_active = cfg.pilot_indices.is_empty().then_some(64).unwrap_or_else(|| {
        // Use the number implied by the config's delay_bins / 3
        cfg.delay_bins / 3
    });
    // Derive k_active from the config: delay_bins = 3 * k_active per ADR-134
    let k_active = cfg.delay_bins / 3;

    let taps: Vec<(f64, num_complex::Complex<f32>)> = taps_spec
        .iter()
        .map(|t| {
            let alpha = num_complex::Complex::new(
                t.amplitude * t.phase.cos(),
                t.amplitude * t.phase.sin(),
            );
            (t.delay_s, alpha)
        })
        .collect();

    let mut rng = Rng::new(42);
    let csi = forward_project(k_active, delta_f_hz, &taps, 20.0, &mut rng);

    // Determine expected dominant delay bin:
    // tau_0 = 10e-9 s;  bin = tau_0 * delay_bins * (k_active * delta_f_hz)
    let delay_resolution_s = 1.0 / (cfg.delay_bins as f64 * delta_f_hz);
    let expected_dominant_bin = (taps_spec[0].delay_s / delay_resolution_s).round() as usize;
    let expected_bin_tau1 = (taps_spec[1].delay_s / delay_resolution_s).round() as usize;
    let expected_bin_tau2 = (taps_spec[2].delay_s / delay_resolution_s).round() as usize;

    // Save fixture (will be created/overwritten)
    save_fixture(fixture_path, k_active, &csi, expected_dominant_bin);

    let num_subcarriers = k_active;
    let frame = make_frame(bandwidth_mhz, num_subcarriers, csi);

    let est = CirEstimator::new(cfg.clone());
    let cir = est.estimate(&frame)
        .unwrap_or_else(|e| panic!("[{}] estimate() failed: {:?}", label, e));

    // 1. dominant_tap_idx corresponds to the direct path (smallest delay) within
    //    ±2 bins. The boundary case τ=10ns at ~20ns/bin lies at bin 0.5 so the
    //    solver may pick bin 0 or bin 1 depending on noise realisation.
    let bin_err = cir.dominant_tap_idx.abs_diff(expected_dominant_bin);
    assert!(
        bin_err <= 2,
        "[{}] dominant_tap_idx={} expected={} (±2 bin tolerance, abs_diff={})",
        label, cir.dominant_tap_idx, expected_dominant_bin, bin_err
    );

    // 2. Taps vector has nonzero magnitude at the 3 ground-truth delay bins (±1 bin)
    let tap_mags: Vec<f32> = cir.taps.iter().map(|c| c.norm()).collect();
    let peak_near = |target_bin: usize| -> bool {
        let lo = target_bin.saturating_sub(1);
        let hi = (target_bin + 1).min(tap_mags.len() - 1);
        (lo..=hi).any(|b| tap_mags[b] > 1e-6)
    };

    assert!(
        peak_near(expected_dominant_bin),
        "[{}] no nonzero tap near bin {} (direct path)",
        label, expected_dominant_bin
    );
    assert!(
        peak_near(expected_bin_tau1),
        "[{}] no nonzero tap near bin {} (reflection 1)",
        label, expected_bin_tau1
    );
    assert!(
        peak_near(expected_bin_tau2),
        "[{}] no nonzero tap near bin {} (reflection 2)",
        label, expected_bin_tau2
    );

    // 3. dominant_tap_ratio meets per-tier floor
    assert!(
        cir.dominant_tap_ratio > dominant_ratio_floor,
        "[{}] dominant_tap_ratio={:.3} < floor={:.3}",
        label, cir.dominant_tap_ratio, dominant_ratio_floor
    );

    // 4. ISTA converged before hitting max_iter
    assert!(
        cir.active_tap_count > 0,
        "[{}] active_tap_count == 0 — solver produced all-zero taps",
        label
    );
}

// ---------------------------------------------------------------------------
// Per-tier tests
// ---------------------------------------------------------------------------

#[test]
fn should_recover_3tap_channel_ht20() {
    // HT20: K_active=52, G=168 (3×), lambda=0.05, max_iter=30
    // ADR-134 Table §2.3: dominant_tap_ratio floor = 0.30 for HT20
    let cfg = CirConfig::for_bandwidth_mhz(20);
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cir_synthetic_ht20.json"
    );
    run_3tap_test("HT20", cfg, 20, 0.30, fixture);
}

#[test]
fn should_recover_3tap_channel_ht40() {
    // HT40: K_active=108, G=342 (3×), lambda=0.03, max_iter=35
    let cfg = CirConfig::for_bandwidth_mhz(40);
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cir_synthetic_ht40.json"
    );
    run_3tap_test("HT40", cfg, 40, 0.35, fixture);
}

#[test]
fn should_recover_3tap_channel_he20() {
    // HE20: K_active=242, G=726 (3×), lambda=0.03, max_iter=32
    // ADR-134: better conditioning → higher dominant_tap_ratio floor
    let cfg = CirConfig::he20();
    let fixture = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cir_synthetic_he20.json"
    );
    run_3tap_test("HE20", cfg, 20, 0.40, fixture);
}

// ---------------------------------------------------------------------------
// dominant_delay_sec / dominant_distance_m accessor tests
// ---------------------------------------------------------------------------

#[test]
fn should_return_none_for_dominant_tof_at_20mhz() {
    // Ranging is disabled at 20 MHz (Tier A / A-HE) per ADR-134 §2.3
    let cfg = CirConfig::for_bandwidth_mhz(20);
    let k_active = cfg.delay_bins / 3;
    let delta_f = 312_500.0_f64;
    let taps = vec![(10e-9_f64, num_complex::Complex::new(1.0_f32, 0.0_f32))];
    let mut rng = Rng::new(42);
    let csi = forward_project(k_active, delta_f, &taps, 30.0, &mut rng);
    let frame = make_frame(20, k_active, csi);
    let est = CirEstimator::new(cfg);
    let cir = est.estimate(&frame).expect("estimate should succeed");
    assert!(
        !cir.ranging_valid,
        "ranging_valid should be false at 20 MHz"
    );
    assert!(
        cir.dominant_tap_tof_s().is_none(),
        "dominant_tap_tof_s() must return None when ranging_valid=false"
    );
}

#[test]
fn should_return_tof_at_40mhz() {
    // Ranging is enabled at 40 MHz (Tier B) per ADR-134 §2.3
    let cfg = CirConfig::for_bandwidth_mhz(40);
    let k_active = cfg.delay_bins / 3;
    let delta_f = 312_500.0_f64;
    let taps = vec![(30e-9_f64, num_complex::Complex::new(1.0_f32, 0.0_f32))];
    let mut rng = Rng::new(42);
    let csi = forward_project(k_active, delta_f, &taps, 30.0, &mut rng);
    let frame = make_frame(40, k_active, csi);
    let est = CirEstimator::new(cfg);
    let cir = est.estimate(&frame).expect("estimate should succeed");
    assert!(
        cir.ranging_valid,
        "ranging_valid should be true at 40 MHz"
    );
    assert!(
        cir.dominant_tap_tof_s().is_some(),
        "dominant_tap_tof_s() must return Some when ranging_valid=true"
    );
}

// ---------------------------------------------------------------------------
// RMS delay spread sanity
// ---------------------------------------------------------------------------

#[test]
fn should_produce_positive_rms_delay_spread() {
    let cfg = CirConfig::for_bandwidth_mhz(20);
    let k_active = cfg.delay_bins / 3;
    let delta_f = 312_500.0_f64;
    let taps: Vec<(f64, num_complex::Complex<f32>)> = ground_truth_taps()
        .iter()
        .map(|t| {
            (t.delay_s, num_complex::Complex::new(
                t.amplitude * t.phase.cos(),
                t.amplitude * t.phase.sin(),
            ))
        })
        .collect();
    let mut rng = Rng::new(42);
    let csi = forward_project(k_active, delta_f, &taps, 20.0, &mut rng);
    let frame = make_frame(20, k_active, csi);
    let est = CirEstimator::new(cfg);
    let cir = est.estimate(&frame).expect("estimate should succeed");
    assert!(
        cir.rms_delay_spread_s > 0.0,
        "rms_delay_spread_s must be positive for a multi-tap channel"
    );
    // 3-tap channel spanning 180 ns → RMS spread must be < 200 ns
    assert!(
        cir.rms_delay_spread_s < 200e-9,
        "rms_delay_spread_s={:.1e} unreasonably large",
        cir.rms_delay_spread_s
    );
}
