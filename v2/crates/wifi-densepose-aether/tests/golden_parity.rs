//! ADR-185 §4.1 — NATIVE half of the AETHER parity gate, and the half that runs
//! in CI.
//!
//! The committed golden vectors live under `python/tests/golden/` and are
//! shared with `python/tests/test_aether.py` (the binding half). That pytest
//! runs in python-ci; the native reference tests in `python/tests/*.rs` link
//! against the PyO3 crate and are NOT run by any workflow. This test closes that
//! gap: it recomputes the embedding through THIS std-only crate — no PyO3, no
//! marshalling — and asserts it matches the same golden within tolerance.
//!
//! Together with the pytest half: native≈golden AND binding≈golden ⇒
//! binding≈native, portably. And because this side has no marshalling, a
//! binding-specific defect baked into the golden would surface here as a native
//! mismatch — which is the failure mode a binding-derived golden + pytest-only
//! CI would otherwise hide.
//!
//! Runs under the repo's `cargo test --workspace` (this crate is a member).

use std::path::PathBuf;

use wifi_densepose_aether::embedding::{EmbeddingConfig, EmbeddingExtractor};
use wifi_densepose_aether::graph_transformer::TransformerConfig;

// Same tolerance as the Python and .rs parity tests. f32 + transcendentals are
// not bit-reproducible across arch, so the golden (generated on one machine) is
// compared within a bound, not by hash.
const PARITY_ATOL: f32 = 1e-4;
const PARITY_RTOL: f32 = 1e-4;

fn golden_dir() -> PathBuf {
    // This crate lives at v2/crates/wifi-densepose-aether; the shared golden
    // fixtures are the single source of truth under python/tests/golden.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../python/tests/golden")
}

fn read_vec(name: &str) -> Vec<f32> {
    let raw = std::fs::read_to_string(golden_dir().join(name))
        .unwrap_or_else(|e| panic!("read {name}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

fn load_input() -> Vec<Vec<f32>> {
    let raw = std::fs::read_to_string(golden_dir().join("aether_input.json"))
        .expect("read aether_input.json");
    let rows: Vec<Vec<f64>> = serde_json::from_str(&raw).expect("parse aether_input.json");
    rows.into_iter()
        .map(|r| r.into_iter().map(|x| x as f32).collect())
        .collect()
}

fn extractor() -> EmbeddingExtractor {
    let e = EmbeddingConfig { d_model: 64, d_proj: 128, temperature: 0.07, normalize: true };
    let t = TransformerConfig {
        n_subcarriers: 56,
        n_keypoints: 17,
        d_model: 64,
        n_heads: 4,
        n_gnn_layers: 2,
    };
    EmbeddingExtractor::new(t, e)
}

fn formula_weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u64 * 1103515245 + 12345) % 65536) as f32 / 65536.0 - 0.5)
        .collect()
}

fn assert_matches_golden(embedding: &[f32], name: &str) {
    let golden = read_vec(name);
    assert_eq!(embedding.len(), golden.len(), "{name}: length mismatch");
    for (i, (&a, &b)) in embedding.iter().zip(&golden).enumerate() {
        assert!(a.is_finite(), "{name}: element {i} is not finite ({a})");
        let tol = PARITY_ATOL + PARITY_RTOL * b.abs();
        assert!(
            (a - b).abs() <= tol,
            "{name}: element {i} diverged beyond tolerance \
             (got {a}, golden {b}, |Δ|={}) — real regression, not arch drift",
            (a - b).abs()
        );
    }
}

#[test]
fn native_base_embedding_matches_committed_golden() {
    let emb = extractor().extract(&load_input());
    assert_matches_golden(&emb, "aether_embedding.json");
}

#[test]
fn native_loaded_embedding_matches_committed_golden() {
    let input = load_input();
    let mut ext = extractor();
    let baseline = ext.extract(&input);

    let weights = formula_weights(ext.param_count());
    let mut buf = Vec::new();
    buf.extend_from_slice(b"AETHERW1");
    buf.extend_from_slice(&(weights.len() as u32).to_le_bytes());
    for w in &weights {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    let wpath = std::env::temp_dir().join(format!("aether_golden_parity_{}.bin", std::process::id()));
    std::fs::write(&wpath, &buf).expect("write weights");
    ext.load_weights(&wpath).expect("load_weights");
    let _ = std::fs::remove_file(&wpath);

    let loaded = ext.extract(&input);
    assert!(
        baseline.iter().zip(&loaded).any(|(a, b)| (a - b).abs() > 1e-6),
        "load_weights had no effect vs the random-init baseline"
    );
    assert_matches_golden(&loaded, "aether_loaded_embedding.json");
}
