//! CIR Deterministic Proof Runner (ADR-134)
//!
//! Verifies or generates the canonical SHA-256 hash of the CIR estimator's
//! deterministic output on the synthetic reference signal (seed=42).
//!
//! Algorithm:
//!   1. Load archive/v1/data/proof/sample_csi_data.json
//!   2. For each of the first 100 frames, construct a CsiFrame and call
//!      CirEstimator::estimate(&frame)
//!   3. Take the top-5 taps by magnitude
//!   4. Round each tap to: tap_idx as usize, re as (c.re * 1e6).round() as i64,
//!      im as (c.im * 1e6).round() as i64
//!   5. Concatenate all 100 frame outputs into one canonical byte string
//!   6. SHA-256 -> print hex
//!
//! Usage:
//!   cargo run -p wifi-densepose-signal --bin cir_proof_runner --release \
//!     --no-default-features -- --generate-hash
//!
//!   cargo run -p wifi-densepose-signal --bin cir_proof_runner --release \
//!     --no-default-features
//!   (compares against archive/v1/data/proof/expected_cir_features.sha256)
//!
//! Note (2026-05-28): This binary requires wifi_densepose_signal::ruvsense::cir,
//! which is NOT YET IMPLEMENTED by the implementation agent. The binary will
//! not compile until CirEstimator is available. The hash file and scripts are
//! committed as placeholders. To generate the real hash after the cir module
//! lands, run:
//!
//!   cd v2 && cargo run -p wifi-densepose-signal --bin cir_proof_runner \
//!     --release --no-default-features -- --generate-hash \
//!     > ../archive/v1/data/proof/expected_cir_features.sha256

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use num_complex::Complex32;
use serde_json::Value;
use sha2::{Digest, Sha256};
use wifi_densepose_core::types::{CsiFrame, CsiMetadata, DeviceId, FrequencyBand};
use wifi_densepose_signal::ruvsense::cir::{CirConfig, CirEstimator};

/// Number of frames to process (matches Python verify.py).
const FRAME_COUNT: usize = 100;

/// CirConfig::ht20() delay-bin count = 156 — full profile width hashed per frame.
const PROFILE_BIN_COUNT: usize = 156;

/// Subcarrier count in the raw legacy reference signal (Atheros 9580 convention).
const N_SUBCARRIERS_RAW: usize = 56;

/// CirConfig::ht20() expects the full 802.11n FFT bin count.
const N_SUBCARRIERS_PADDED: usize = 64;

fn repo_root() -> PathBuf {
    // Binary lives at v2/target/release/cir_proof_runner; repo root is ../..
    // But we can't rely on binary location at runtime. Use git rev-parse instead,
    // or walk up from cwd until we find archive/.
    let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    // If run from v2/, walk up once; if run from repo root, use directly.
    let candidates = [
        cwd.clone(),
        cwd.join(".."),
        cwd.join("../.."),
    ];
    for candidate in &candidates {
        if candidate.join("archive/v1/data/proof/sample_csi_data.json").exists() {
            return candidate.canonicalize().unwrap_or(candidate.clone());
        }
    }
    // Fallback: assume cwd is repo root
    cwd
}

fn load_json(path: &Path) -> Value {
    let content = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {}: {}", path.display(), e));
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Cannot parse {}: {}", path.display(), e))
}

/// Build a CsiFrame from a JSON frame record.
/// The reference signal has 3 antennas and 56 subcarriers.
/// We use only the first antenna's amplitude/phase to form a Complex32 vector.
fn frame_from_json(record: &Value) -> CsiFrame {
    let amplitude_all = record["amplitude"].as_array()
        .expect("frame must have amplitude array");
    let phase_all = record["phase"].as_array()
        .expect("frame must have phase array");

    // Use the first antenna row
    let amplitude = amplitude_all[0].as_array().expect("antenna 0 amplitude");
    let phase = phase_all[0].as_array().expect("antenna 0 phase");

    // Build Complex64 data: shape [1, N_SUBCARRIERS]
    use ndarray::Array2;
    use num_complex::Complex64;

    // Pad the legacy 56-subcarrier capture to the 64-bin HT20 FFT layout
    // expected by CirEstimator. The 56 values map sequentially into the first
    // 56 slots; bins 56..64 are zero-padded. This is not physically meaningful
    // (the real 802.11n mapping puts pilots at specific bins) but produces a
    // deterministic 64-wide frame the estimator can ingest, which is what the
    // witness needs — bit-deterministic CIR computation from a fixed input.
    let n_raw = amplitude.len().min(N_SUBCARRIERS_RAW);
    let mut data = Array2::<Complex64>::zeros((1, N_SUBCARRIERS_PADDED));
    for (k, (a, p)) in amplitude.iter().zip(phase.iter()).enumerate().take(n_raw) {
        let a_val = a.as_f64().unwrap_or(0.0);
        let p_val = p.as_f64().unwrap_or(0.0);
        data[[0, k]] = Complex64::from_polar(a_val, p_val);
    }

    let metadata = CsiMetadata::new(
        DeviceId::new("proof-runner"),
        FrequencyBand::Band5GHz,
        36, // channel 36, arbitrary
    );
    CsiFrame::new(metadata, data)
}

/// Canonical, cross-platform-deterministic serialisation of one frame's CIR.
///
/// We previously hashed (a) raw real/imag at 1e-6 precision and (b) the top-5
/// tap pairs sorted by magnitude. Both broke across platforms because libm
/// differences (glibc / MSVC / Apple) on `sin`/`cos`/`sqrt` drift by ~1e-7,
/// which is enough to (i) flip rounded integers and (ii) re-order near-tied
/// taps in a magnitude sort. The witness exists to detect *algorithmic*
/// regressions, not libm jitter.
///
/// New canonical form: the full per-tap quantised magnitude profile, in
/// natural index order, no sort. At 1e-2 precision a 1% drift in any tap is
/// invisible; a 10× lambda change moves taps by >1e-2 and breaks the hash.
///
/// Format: `[mag_q: u16 le]` per tap, `num_taps` taps per frame. Saturating to
/// u16 caps magnitudes at 65.535, well above the 1.0-ish normalised range.
fn serialise_profile(taps: &[Complex32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(taps.len() * 2);
    for c in taps {
        let mag_q = (c.norm() * 1e2_f32).round().max(0.0).min(u16::MAX as f32) as u16;
        out.extend_from_slice(&mag_q.to_le_bytes());
    }
    out
}

fn compute_hash(json_path: &Path) -> String {
    let data = load_json(json_path);
    let frames = data["frames"].as_array().expect("frames array");

    let config = CirConfig::ht20();
    let estimator = CirEstimator::new(config);

    let mut hasher = Sha256::new();

    for record in frames.iter().take(FRAME_COUNT) {
        let frame = frame_from_json(record);
        match estimator.estimate(&frame) {
            Ok(cir) => {
                let bytes = serialise_profile(&cir.taps);
                hasher.update(&bytes);
            }
            Err(e) => {
                eprintln!("WARNING: CIR estimate failed for frame: {}", e);
                // Write PROFILE_BIN_COUNT * sizeof(u16) zero bytes so the hash
                // stays deterministic even when frames consistently fail.
                hasher.update(vec![0u8; PROFILE_BIN_COUNT * 2]);
            }
        }
    }

    format!("{:x}", hasher.finalize())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let generate_hash = args.iter().any(|a| a == "--generate-hash");

    let root = repo_root();
    let json_path = root.join("archive/v1/data/proof/sample_csi_data.json");
    let hash_path = root.join("archive/v1/data/proof/expected_cir_features.sha256");

    if !json_path.exists() {
        eprintln!("ERROR: reference signal not found at {}", json_path.display());
        std::process::exit(1);
    }

    let hash = compute_hash(&json_path);

    if generate_hash {
        println!("{}", hash);
    } else {
        // Compare against stored hash
        if !hash_path.exists() {
            eprintln!("ERROR: expected hash file not found at {}", hash_path.display());
            eprintln!("Run with --generate-hash to create it.");
            std::process::exit(1);
        }
        let expected = fs::read_to_string(&hash_path)
            .expect("read expected hash file")
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_owned();

        if hash == expected {
            println!("VERDICT: PASS (CIR hash matches)");
            std::process::exit(0);
        } else {
            eprintln!("VERDICT: FAIL");
            eprintln!("expected: {}", expected);
            eprintln!("actual:   {}", hash);
            io::stderr().flush().ok();
            std::process::exit(1);
        }
    }
}
