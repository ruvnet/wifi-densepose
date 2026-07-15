//! Calibration Deterministic Proof Runner (ADR-135)
//!
//! Verifies or generates the canonical SHA-256 hash of the CalibrationRecorder's
//! deterministic output on a synthetic stationary channel (seed=42, HT20, 600 frames).
//!
//! Cross-platform portability lesson (from cir_proof_runner.rs, line 123):
//!   Raw f32 round-trips at high precision (1e-6) and magnitude-sort-then-truncate
//!   both break across libm implementations (glibc / MSVC / Apple) because sin/cos/sqrt
//!   differ by ~1e-7 — enough to flip a rounded integer or re-order near-tied values.
//!   The fix: serialise the full per-subcarrier profile in natural index order at
//!   coarse quantisation (1e-2 / 1e-4 / 1e-3). A 1% drift is invisible to the hash;
//!   a 10× algorithm change moves values by >1e-2 and breaks the hash.
//!   No sort, no truncation, no libm-sensitive comparison.
//!
//! Canonical form (per subcarrier k, 4 × u16 LE):
//!   [0] (amp_mean * 1e2).round() as u16
//!   [1] (amp_variance * 1e4).round() as u16
//!   [2] ((phase_mean + π) * 1e3).round() as u16   ← shifted so always non-negative
//!   [3] (phase_dispersion * 1e3).round() as u16
//!
//! Prefix: tier byte (0 = HT20), frame_count u64 LE.
//! All subcarriers in natural index order; no sort.
//!
//! Usage:
//!   cargo run -p wifi-densepose-signal --bin calibration_proof_runner \
//!     --release --no-default-features -- --generate-hash
//!
//!   cargo run -p wifi-densepose-signal --bin calibration_proof_runner \
//!     --release --no-default-features
//!   (compares against archive/v1/data/proof/expected_calibration_features.sha256)
//!
//! IMPORTANT: This binary cannot compile until CalibrationRecorder is implemented.
//! While the implementation is in progress, a placeholder hash is committed in
//! archive/v1/data/proof/expected_calibration_features.sha256. Regenerate with:
//!
//!   cd v2 && cargo run -p wifi-densepose-signal --bin calibration_proof_runner \
//!     --release --no-default-features -- --generate-hash \
//!     > ../archive/v1/data/proof/expected_calibration_features.sha256

use std::env;
use std::f32::consts::PI;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use ndarray::Array2;
use num_complex::Complex64;
use sha2::{Digest, Sha256};
use wifi_densepose_core::types::{AntennaConfig, CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::calibration::{CalibrationConfig, CalibrationRecorder};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const N_ACTIVE: usize = 52;   // HT20 active subcarriers
const N_FRAMES: usize = 600;  // 30 s × 20 Hz
const TIER_BYTE: u8 = 0;      // 0 = HT20

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
// Synthetic CSI frame generator — stationary channel, seed=42
//
// amp[k]   = 0.3 + 0.7 * |sin(k * π / K)|   (smooth across subcarriers)
// phase[k] = (k * 0.1) mod 2π − π             (slowly rotating)
// AWGN at ~30 dB SNR added via Box-Muller.
// ---------------------------------------------------------------------------

fn make_frame(rng: &mut Rng) -> CsiFrame {
    let n = N_ACTIVE;
    let noise_std = 0.01_f32;

    let mut data = Array2::<Complex64>::zeros((1, n));
    for k in 0..n {
        let amp = 0.3 + 0.7 * (k as f32 * PI / n as f32).sin().abs();
        let phase = (k as f32 * 0.1).rem_euclid(2.0 * PI) - PI;
        let re = amp * phase.cos() + noise_std * rng.next_normal();
        let im = amp * phase.sin() + noise_std * rng.next_normal();
        data[(0, k)] = Complex64::new(re as f64, im as f64);
    }
    let mut meta =
        CsiMetadata::new(DeviceId::new("proof-runner"), FrequencyBand::Band2_4GHz, 6);
    meta.bandwidth_mhz = 20;
    meta.antenna_config = AntennaConfig::new(1, 1);
    CsiFrame::new(meta, data)
}

// ---------------------------------------------------------------------------
// Canonical, cross-platform-deterministic serialisation.
//
// Per ADR-135 proof spec and the cir_proof_runner.rs lesson (line 123):
// coarse u16 quantisation, natural subcarrier order, no sort.
// ---------------------------------------------------------------------------

fn serialise_baseline_canonical(
    subcarriers: &[wifi_densepose_signal::calibration::SubcarrierBaseline],
    frame_count: u64,
) -> Vec<u8> {
    let k = subcarriers.len();
    // Header: tier byte + frame_count as u64 LE
    let mut out = Vec::with_capacity(1 + 8 + k * 8);
    out.push(TIER_BYTE);
    out.extend_from_slice(&frame_count.to_le_bytes());

    for sc in subcarriers {
        // [0] amp_mean at 1e-2 resolution
        let amp_q = (sc.amp_mean * 1e2_f32)
            .round()
            .max(0.0)
            .min(u16::MAX as f32) as u16;
        out.extend_from_slice(&amp_q.to_le_bytes());

        // [1] amp_variance at 1e-4 resolution
        let var_q = (sc.amp_variance * 1e4_f32)
            .round()
            .max(0.0)
            .min(u16::MAX as f32) as u16;
        out.extend_from_slice(&var_q.to_le_bytes());

        // [2] phase_mean shifted by +π so it is non-negative, at 1e-3 resolution
        let phase_q = ((sc.phase_mean + PI) * 1e3_f32)
            .round()
            .max(0.0)
            .min(u16::MAX as f32) as u16;
        out.extend_from_slice(&phase_q.to_le_bytes());

        // [3] phase_dispersion (von Mises 1−R̄, in [0,1]) at 1e-3 resolution
        let disp_q = (sc.phase_dispersion * 1e3_f32)
            .round()
            .max(0.0)
            .min(u16::MAX as f32) as u16;
        out.extend_from_slice(&disp_q.to_le_bytes());
    }

    out
}

// ---------------------------------------------------------------------------
// Repo root discovery
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let candidates = [
        cwd.clone(),
        cwd.join(".."),
        cwd.join("../.."),
        cwd.join("../../.."),
    ];
    for candidate in &candidates {
        if candidate
            .join("archive/v1/data/proof/expected_calibration_features.sha256")
            .exists()
            || candidate.join("archive/v1/data/proof/sample_csi_data.json").exists()
        {
            return candidate.canonicalize().unwrap_or(candidate.clone());
        }
    }
    cwd
}

// ---------------------------------------------------------------------------
// Main hash computation
// ---------------------------------------------------------------------------

fn compute_hash() -> String {
    let config = CalibrationConfig::ht20();
    let mut recorder = CalibrationRecorder::new(config);
    let mut rng = Rng::new(42);

    for _ in 0..N_FRAMES {
        let frame = make_frame(&mut rng);
        recorder
            .record(&frame)
            .expect("record() must succeed for synthetic frames");
    }

    let baseline = recorder
        .finalize()
        .expect("finalize() must succeed after 600 frames");

    let payload = serialise_baseline_canonical(&baseline.subcarriers, baseline.frame_count);

    let mut hasher = Sha256::new();
    hasher.update(&payload);
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    let generate_hash = args.iter().any(|a| a == "--generate-hash");

    let hash = compute_hash();

    if generate_hash {
        println!("{}", hash);
        return;
    }

    // Compare against stored hash
    let root = repo_root();
    let hash_path = root.join("archive/v1/data/proof/expected_calibration_features.sha256");

    if !hash_path.exists() {
        eprintln!(
            "ERROR: expected hash file not found at {}",
            hash_path.display()
        );
        eprintln!("Run with --generate-hash to create it.");
        std::process::exit(1);
    }

    let expected_content = fs::read_to_string(&hash_path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", hash_path.display(), e));

    let expected = expected_content
        .split_whitespace()
        .find(|s| !s.starts_with('#'))
        .unwrap_or("")
        .to_owned();

    if expected.starts_with("PLACEHOLDER") {
        eprintln!("BLOCKED: calibration proof hash is a placeholder.");
        eprintln!(
            "The calibration module (ADR-135) is not yet fully implemented. \
             After the implementation lands, regenerate:"
        );
        eprintln!(
            "  cd v2 && cargo run -p wifi-densepose-signal --bin calibration_proof_runner \
             --release --no-default-features -- --generate-hash \
             > ../archive/v1/data/proof/expected_calibration_features.sha256"
        );
        std::process::exit(2);
    }

    if hash == expected {
        println!("VERDICT: PASS (calibration hash matches)");
        std::process::exit(0);
    } else {
        eprintln!("VERDICT: FAIL");
        eprintln!("expected: {}", expected);
        eprintln!("actual:   {}", hash);
        io::stderr().flush().ok();
        std::process::exit(1);
    }
}
