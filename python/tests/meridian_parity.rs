//! ADR-185 §4.1 — MERIDIAN bit-for-bit parity: native-Rust reference half.
//!
//! Calls the canonical `wifi-densepose-signal::hardware_norm` +
//! `wifi-densepose-train::{geometry,rapid_adapt}` code DIRECTLY (no PyO3)
//! on the committed `tests/golden/meridian_input.json` fixture and locks
//! the SHA-256 of the concatenated f32 outputs into
//! `tests/golden/meridian_output.sha256`.
//!
//! Concatenation order (identical in the pytest half, tests/test_meridian.py):
//!   1. esp32 canonical amplitude (56)   2. esp32 canonical phase (56)
//!   3. intel5300 canonical amplitude    4. intel5300 canonical phase
//!   5. geometry.encode(ap_positions)    6. rapid_adapt lora_weights
//!
//! Regenerate (only on an intentional Rust change): delete the .sha256 and
//! re-run `cargo test --features meridian --test meridian_parity`.
#![cfg(feature = "meridian")]

use std::fs;
use std::path::PathBuf;

use serde_json::Value;
use sha2::{Digest, Sha256};
use wifi_densepose_signal::hardware_norm::{HardwareNormalizer, HardwareType};
use wifi_densepose_train::geometry::{GeometryEncoder, MeridianGeometryConfig};
use wifi_densepose_train::rapid_adapt::{AdaptationLoss, RapidAdaptation};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn fixture() -> Value {
    let raw = fs::read_to_string(golden_dir().join("meridian_input.json"))
        .expect("read meridian_input.json fixture");
    serde_json::from_str(&raw).expect("parse meridian_input.json")
}

fn f64_vec(v: &Value, key: &str) -> Vec<f64> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn f32_frames(v: &Value, key: &str) -> Vec<Vec<f32>> {
    v[key]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

/// Compute the full concatenated MERIDIAN output vector, mirroring the
/// Python binding's default construction exactly.
fn meridian_output(fx: &Value) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();

    // 1–4: hardware normalization (default normalizer, canonical 56).
    let norm = HardwareNormalizer::new();
    let esp = norm
        .normalize(
            &f64_vec(fx, "esp32_amplitude"),
            &f64_vec(fx, "esp32_phase"),
            HardwareType::Esp32S3,
        )
        .unwrap();
    out.extend_from_slice(&esp.amplitude);
    out.extend_from_slice(&esp.phase);
    let intel = norm
        .normalize(
            &f64_vec(fx, "intel_amplitude"),
            &f64_vec(fx, "intel_phase"),
            HardwareType::Intel5300,
        )
        .unwrap();
    out.extend_from_slice(&intel.amplitude);
    out.extend_from_slice(&intel.phase);

    // 5: geometry encoding (default config → 64-dim).
    let enc = GeometryEncoder::new(&MeridianGeometryConfig::default());
    let aps: Vec<[f32; 3]> = fx["ap_positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            [
                a[0].as_f64().unwrap() as f32,
                a[1].as_f64().unwrap() as f32,
                a[2].as_f64().unwrap() as f32,
            ]
        })
        .collect();
    out.extend_from_slice(&enc.encode(&aps));

    // 6: rapid adaptation lora weights (Combined, epochs 5, lr 1e-3, λ 0.5).
    let mut ra = RapidAdaptation::new(
        10,
        4,
        AdaptationLoss::Combined {
            epochs: 5,
            lr: 0.001,
            lambda_ent: 0.5,
        },
    );
    for frame in f32_frames(fx, "rapid_frames") {
        ra.push_frame(&frame);
    }
    out.extend_from_slice(&ra.adapt().unwrap().lora_weights);

    out
}

fn sha256_le(vals: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for &x in vals {
        hasher.update(x.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn native_canonical_frames_are_56_wide() {
    let fx = fixture();
    let norm = HardwareNormalizer::new();
    let esp = norm
        .normalize(
            &f64_vec(&fx, "esp32_amplitude"),
            &f64_vec(&fx, "esp32_phase"),
            HardwareType::Esp32S3,
        )
        .unwrap();
    assert_eq!(esp.amplitude.len(), 56);
    assert_eq!(esp.phase.len(), 56);
    let intel = norm
        .normalize(
            &f64_vec(&fx, "intel_amplitude"),
            &f64_vec(&fx, "intel_phase"),
            HardwareType::Intel5300,
        )
        .unwrap();
    assert_eq!(intel.amplitude.len(), 56);
    // 64-dim geometry vector.
    let enc = GeometryEncoder::new(&MeridianGeometryConfig::default());
    assert_eq!(enc.encode(&[[0.25, 0.5, 0.75]]).len(), 64);
}

#[test]
fn native_meridian_matches_committed_golden() {
    let got = sha256_le(&meridian_output(&fixture()));
    let path = golden_dir().join("meridian_output.sha256");
    match fs::read_to_string(&path) {
        Ok(expected) => assert_eq!(
            got,
            expected.trim(),
            "native MERIDIAN hash drifted from committed golden \
             (intentional? delete the .sha256 and regenerate)"
        ),
        Err(_) => {
            fs::write(&path, &got).expect("write golden sha256");
            panic!("no committed golden found; wrote {got}. Re-run to verify parity.");
        }
    }
}
