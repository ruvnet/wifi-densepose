//! ADR-185 §13.a — weight-loading parity: native-Rust reference half.
//!
//! Proves the AETHER `load_weights` path produces a deterministic, non-random
//! embedding, and compares it to the committed golden VECTOR
//! `tests/golden/aether_loaded_embedding.json` within tolerance. The pytest
//! half (`tests/test_aether.py`) writes a byte-identical weight file (same
//! formula + format) through the binding's `load_weights` and compares to the
//! SAME golden within the same tolerance — native≈golden and binding≈golden
//! prove the binding's weight-loading matches native, portably across arch.
//! (See `aether_parity.rs` for why this is a tolerance compare, not a hash.)
//!
//! Weight formula (shared with the pytest half): `w[i] = k/65536 - 0.5` where
//! `k = (i*1103515245 + 12345) mod 65536`. `k/65536` is a multiple of 2⁻¹⁶,
//! exactly representable in both f32 and f64, so both languages produce
//! byte-identical weights.
//!
//! File format: 8-byte magic `AETHERW1`, `u32` little-endian param count, then
//! that many little-endian `f32`.
//!
//! Regenerate (only on an intentional change): delete the .json golden and
//! re-run `cargo test --features aether --test aether_weights_parity`.
#![cfg(feature = "aether")]

use std::fs;
use std::path::PathBuf;

use wifi_densepose_aether::embedding::{EmbeddingConfig, EmbeddingExtractor};
use wifi_densepose_aether::graph_transformer::TransformerConfig;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

fn load_input() -> Vec<Vec<f32>> {
    let raw = fs::read_to_string(golden_dir().join("aether_input.json"))
        .expect("read aether_input.json fixture");
    let rows: Vec<Vec<f64>> = serde_json::from_str(&raw).expect("parse aether_input.json");
    rows.into_iter()
        .map(|row| row.into_iter().map(|x| x as f32).collect())
        .collect()
}

/// Same default construction as `aether_parity.rs` / the Python binding default.
fn new_extractor() -> EmbeddingExtractor {
    let e_config = EmbeddingConfig {
        d_model: 64,
        d_proj: 128,
        temperature: 0.07,
        normalize: true,
    };
    let t_config = TransformerConfig {
        n_subcarriers: 56,
        n_keypoints: 17,
        d_model: 64,
        n_heads: 4,
        n_gnn_layers: 2,
    };
    EmbeddingExtractor::new(t_config, e_config)
}

fn formula_weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let k = (i as u32).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 65_536;
            k as f32 / 65_536.0 - 0.5
        })
        .collect()
}

fn write_weight_file(path: &PathBuf, weights: &[f32]) {
    let mut buf = Vec::with_capacity(12 + weights.len() * 4);
    buf.extend_from_slice(b"AETHERW1");
    buf.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for v in weights {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, buf).unwrap();
}

/// Cross-architecture f32 parity tolerance; see `aether_parity.rs` and the
/// matching constants in `tests/test_aether.py` for why this is a tolerance
/// compare and not a byte hash.
const PARITY_ATOL: f32 = 1e-4;
const PARITY_RTOL: f32 = 1e-4;

fn assert_matches_golden_vector(embedding: &[f32], name: &str) {
    let path = golden_dir().join(name);
    match fs::read_to_string(&path) {
        Ok(raw) => {
            let golden: Vec<f32> = serde_json::from_str(&raw).expect("parse golden vector json");
            assert_eq!(embedding.len(), golden.len(), "{name}: length mismatch");
            for (i, (&got, &want)) in embedding.iter().zip(&golden).enumerate() {
                let tol = PARITY_ATOL + PARITY_RTOL * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "{name}: element {i} diverged beyond tolerance \
                     (got {got}, golden {want}, |Δ|={}) — real regression, not arch drift",
                    (got - want).abs()
                );
            }
        }
        Err(_) => {
            let json = serde_json::to_string(&embedding).expect("serialize golden");
            fs::write(&path, &json).expect("write golden vector");
            panic!("no committed golden {name}; wrote it. Re-run to verify parity.");
        }
    }
}

#[test]
fn native_loaded_embedding_matches_committed_golden() {
    let input = load_input();

    let mut ext = new_extractor();
    let baseline = ext.extract(&input); // random Xavier init

    let weights = formula_weights(ext.param_count());
    let path = std::env::temp_dir().join(format!(
        "aether_weights_parity_{}.bin",
        std::process::id()
    ));
    write_weight_file(&path, &weights);
    ext.load_weights(&path).expect("load_weights");
    fs::remove_file(&path).ok();

    let loaded = ext.extract(&input);
    // Loaded weights must actually take effect.
    assert!(
        baseline.iter().zip(&loaded).any(|(a, b)| (a - b).abs() > 1e-6),
        "load_weights had no effect vs the random-init baseline"
    );
    assert_eq!(loaded.len(), 128);

    assert_matches_golden_vector(&loaded, "aether_loaded_embedding.json");
}
