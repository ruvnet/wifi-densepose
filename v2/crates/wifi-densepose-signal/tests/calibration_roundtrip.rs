//! Bytes round-trip tests for BaselineCalibration serialisation (ADR-135 §2.4).
//!
//! The implementation uses `to_bytes()` / `from_bytes()` as the binary format.
//! Magic word is 0xCA1B_0001, schema version = 1.
//!
//! Covers:
//!   - Binary round-trip determinism (to_bytes twice → same output)
//!   - deserialise→re-serialise produces identical bytes
//!   - Version mismatch detection
//!   - Truncated buffer detection
//!   - Magic word mismatch detection

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
// Build a deterministic baseline (HT20, 600 frames, seed=42).
// ---------------------------------------------------------------------------

fn build_ht20_baseline() -> BaselineCalibration {
    const N: usize = 52;
    let amp: Vec<f32> = (0..N)
        .map(|k| 0.3 + 0.7 * (k as f32 * PI / N as f32).sin().abs())
        .collect();
    let phase: Vec<f32> = (0..N)
        .map(|k| (k as f32 * 0.1).rem_euclid(2.0 * PI) - PI)
        .collect();

    let mut rng = Rng::new(42);
    let mut recorder = CalibrationRecorder::new(CalibrationConfig::ht20());
    for _ in 0..600 {
        let noise_std = 0.01_f32;
        let mut data = Array2::<Complex64>::zeros((1, N));
        for k in 0..N {
            let re = amp[k] * phase[k].cos() + noise_std * rng.next_normal();
            let im = amp[k] * phase[k].sin() + noise_std * rng.next_normal();
            data[(0, k)] = Complex64::new(re as f64, im as f64);
        }
        let mut meta =
            CsiMetadata::new(DeviceId::new("roundtrip-test"), FrequencyBand::Band2_4GHz, 6);
        meta.bandwidth_mhz = 20;
        meta.antenna_config = AntennaConfig::new(1, 1);
        let frame = CsiFrame::new(meta, data);
        recorder.record(&frame).expect("record");
    }
    recorder.finalize().expect("finalize")
}

// ---------------------------------------------------------------------------
// Binary round-trip determinism
// ---------------------------------------------------------------------------

/// Two calls to `to_bytes()` on the same value must produce identical buffers.
#[test]
fn should_produce_identical_bytes_on_two_calls_to_same_baseline() {
    let baseline = build_ht20_baseline();
    let bytes1 = baseline.to_bytes();
    let bytes2 = baseline.to_bytes();
    assert_eq!(
        bytes1, bytes2,
        "to_bytes must be deterministic across two calls on the same value"
    );
}

/// deserialise → re-serialise must produce identical bytes.
#[test]
fn should_deserialise_and_reserialise_to_identical_bytes() {
    let baseline = build_ht20_baseline();
    let bytes = baseline.to_bytes();
    let recovered = BaselineCalibration::from_bytes(&bytes)
        .expect("from_bytes should succeed on valid bytes");
    let bytes_recovered = recovered.to_bytes();
    assert_eq!(
        bytes, bytes_recovered,
        "round-trip: re-serialised bytes must match original"
    );
}

/// Recovered baseline must have matching field values.
#[test]
fn should_preserve_frame_count_and_subcarrier_count_after_round_trip() {
    let baseline = build_ht20_baseline();
    let bytes = baseline.to_bytes();
    let recovered = BaselineCalibration::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(
        baseline.frame_count, recovered.frame_count,
        "frame_count must survive round-trip"
    );
    assert_eq!(
        baseline.subcarriers.len(),
        recovered.subcarriers.len(),
        "subcarrier count must survive round-trip"
    );
}

/// Per-subcarrier amp_mean values must survive round-trip within f32 precision.
#[test]
fn should_preserve_amp_mean_per_subcarrier_after_round_trip() {
    let baseline = build_ht20_baseline();
    let bytes = baseline.to_bytes();
    let recovered = BaselineCalibration::from_bytes(&bytes).expect("from_bytes");
    for k in 0..baseline.subcarriers.len() {
        assert!(
            (baseline.subcarriers[k].amp_mean - recovered.subcarriers[k].amp_mean).abs() < 1e-6,
            "amp_mean[{}] mismatch: {:.8} vs {:.8}",
            k,
            baseline.subcarriers[k].amp_mean,
            recovered.subcarriers[k].amp_mean
        );
    }
}

/// Magic word 0xCA1B_0001 must appear at offset 0 in serialised bytes.
#[test]
fn should_embed_magic_word_0xca1b0001_at_offset_0() {
    let baseline = build_ht20_baseline();
    let bytes = baseline.to_bytes();
    assert!(bytes.len() >= 4, "serialised bytes must be at least 4 bytes long");
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    assert_eq!(
        magic, 0xCA1B_0001_u32,
        "magic word at offset 0 must be 0xCA1B0001, got 0x{:08X}",
        magic
    );
}

/// Schema version at offset 4 must equal 1.
#[test]
fn should_embed_schema_version_1_at_offset_4() {
    let baseline = build_ht20_baseline();
    let bytes = baseline.to_bytes();
    assert!(bytes.len() >= 6, "bytes too short");
    let version = bytes[4];
    assert_eq!(version, 1, "schema version at offset 4 must be 1, got {}", version);
}

// ---------------------------------------------------------------------------
// Error path: version mismatch
// ---------------------------------------------------------------------------

/// Overwrite version byte with 99 → expect VersionMismatch { got: 99, want: 1 }.
#[test]
fn should_return_version_mismatch_for_version_99() {
    let baseline = build_ht20_baseline();
    let mut bytes = baseline.to_bytes();
    // Version is at offset 4 (u8)
    bytes[4] = 99;

    let result = BaselineCalibration::from_bytes(&bytes);
    match result {
        Err(CalibrationError::VersionMismatch { got, want }) => {
            assert_eq!(got, 99, "VersionMismatch.got should be 99");
            assert_eq!(want, 1, "VersionMismatch.want should be 1");
        }
        other => panic!(
            "expected CalibrationError::VersionMismatch, got {:?}",
            other
        ),
    }
}

// ---------------------------------------------------------------------------
// Error path: truncated buffer
// ---------------------------------------------------------------------------

/// Trim the last 4 bytes → expect TruncatedBuffer.
#[test]
fn should_return_truncated_buffer_error_for_short_input() {
    let baseline = build_ht20_baseline();
    let mut bytes = baseline.to_bytes();
    let new_len = bytes.len().saturating_sub(4);
    bytes.truncate(new_len);

    let result = BaselineCalibration::from_bytes(&bytes);
    assert!(
        matches!(result, Err(CalibrationError::TruncatedBuffer { .. })),
        "expected TruncatedBuffer, got {:?}",
        result
    );
}

/// A completely empty buffer → expect TruncatedBuffer.
#[test]
fn should_return_truncated_buffer_for_empty_input() {
    let result = BaselineCalibration::from_bytes(&[]);
    assert!(
        matches!(result, Err(CalibrationError::TruncatedBuffer { .. })),
        "expected TruncatedBuffer for empty buffer, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Error path: magic word mismatch
// ---------------------------------------------------------------------------

/// Zero out the first 4 bytes (magic word) → expect InvalidMagic error.
#[test]
fn should_return_error_for_zeroed_magic_word() {
    let baseline = build_ht20_baseline();
    let mut bytes = baseline.to_bytes();
    bytes[0] = 0;
    bytes[1] = 0;
    bytes[2] = 0;
    bytes[3] = 0;

    let result = BaselineCalibration::from_bytes(&bytes);
    assert!(
        matches!(result, Err(CalibrationError::InvalidMagic { .. })),
        "expected InvalidMagic when magic word is zeroed, got {:?}",
        result
    );
}
