//! ADR-185 §4.1 — AETHER parity: native-Rust reference half.
//!
//! Produces the golden 128-dim embedding by calling the canonical
//! `wifi-densepose-aether::embedding` code DIRECTLY (no PyO3), for the
//! committed `tests/golden/aether_input.json` fixture, and compares it to the
//! committed golden VECTOR `tests/golden/aether_embedding.json` within a
//! numerical tolerance.
//!
//! Why a vector + tolerance and not a SHA-256 of the f32 bytes: the embedding
//! is pure f32 and uses transcendental ops (ln/sqrt/cos), which are not
//! bit-reproducible across CPU architectures or libm implementations. A byte
//! hash only ever matched the one arch that generated it and failed on every
//! other wheel this project builds (aarch64, macOS-arm). The pytest half
//! (`tests/test_aether.py`) compares the Python binding to the SAME golden
//! within the same tolerance — native≈golden and binding≈golden together prove
//! binding≈native, portably.
//!
//! Regeneration (only when the Rust subsystem intentionally changes): delete
//! `tests/golden/aether_embedding.json` and re-run `cargo test --features aether`.
#![cfg(feature = "aether")]

use std::fs;
use std::path::PathBuf;

use wifi_densepose_aether::embedding::{EmbeddingConfig, EmbeddingExtractor};
use wifi_densepose_aether::graph_transformer::TransformerConfig;

/// Cross-architecture f32 parity tolerance; see the module docs and the
/// matching `PARITY_ATOL`/`PARITY_RTOL` in `tests/test_aether.py`.
const PARITY_ATOL: f32 = 1e-4;
const PARITY_RTOL: f32 = 1e-4;

/// Assert `embedding` matches the committed golden vector `<name>` within
/// tolerance, or (if the golden is absent) write it and fail asking for a re-run.
fn assert_matches_golden_vector(embedding: &[f32], name: &str) {
    let path = golden_dir().join(name);
    match fs::read_to_string(&path) {
        Ok(raw) => {
            let golden: Vec<f32> = serde_json::from_str(&raw)
                .expect("parse golden vector json");
            assert_eq!(embedding.len(), golden.len(), "{name}: length mismatch");
            for (i, (&got, &want)) in embedding.iter().zip(&golden).enumerate() {
                let tol = PARITY_ATOL + PARITY_RTOL * want.abs();
                assert!(
                    (got - want).abs() <= tol,
                    "{name}: element {i} diverged beyond tolerance \
                     (got {got}, golden {want}, |Δ|={}) — a real regression, \
                     not cross-arch f32 drift",
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

/// Build the extractor identically to the Python binding's default
/// construction: `AetherConfig()` + `EmbeddingExtractor(n_subcarriers=56, cfg)`.
fn embed_native(input: &[Vec<f32>]) -> Vec<f32> {
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
    let mut ext = EmbeddingExtractor::new(t_config, e_config);
    ext.extract(input)
}

#[test]
fn native_embedding_is_128_dim_unit_norm() {
    let emb = embed_native(&load_input());
    assert_eq!(emb.len(), 128, "AETHER embedding must be 128-dim");
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "embedding must be L2-normalized, got norm={norm}"
    );
}

#[test]
fn native_embedding_matches_committed_golden() {
    let emb = embed_native(&load_input());
    assert_matches_golden_vector(&emb, "aether_embedding.json");
}
