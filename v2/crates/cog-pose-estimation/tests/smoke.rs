//! Smoke tests for the cog-pose-estimation crate.
//!
//! These are deliberately tight — full inference integration tests
//! depend on a trained safetensors blob that doesn't live in-repo yet.

use cog_pose_estimation::{
    inference::{
        InferenceEngine, SyntheticInput, INPUT_SUBCARRIERS, INPUT_TIMESTEPS, OUTPUT_KEYPOINTS,
    },
    manifest::ManifestSpec,
};

#[test]
fn synthetic_window_has_correct_shape() {
    let syn = SyntheticInput;
    let window = syn.as_window();
    assert_eq!(window.data.len(), INPUT_SUBCARRIERS * INPUT_TIMESTEPS);
}

#[test]
fn engine_produces_finite_output_for_synthetic_input() {
    let engine = InferenceEngine::new().expect("engine init");
    let out = engine.infer(&SyntheticInput.as_window()).expect("infer");
    assert!(
        out.is_finite(),
        "synthetic input must produce finite output"
    );
    assert_eq!(out.keypoints.len(), OUTPUT_KEYPOINTS * 2);
}

#[test]
fn engine_rejects_wrong_shape_input() {
    let engine = InferenceEngine::new().expect("engine init");
    let bad = cog_pose_estimation::inference::CsiWindow {
        data: vec![0.0; 10],
    };
    assert!(engine.infer(&bad).is_err());
}

#[test]
fn real_weights_load_when_available() {
    use cog_pose_estimation::inference::InferenceEngine;
    // Search the cwds a `cargo test` invocation can land in
    // (workspace root, `v2/`, or the crate dir).
    let candidates = [
        "cog/artifacts/pose_v1.safetensors",
        "crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors",
        "v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors",
    ];
    let Some(weights) = candidates
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists())
    else {
        // Skip when running outside the repo (e.g. on a fresh appliance install).
        eprintln!("(skipping — pose_v1.safetensors not present under any known cwd)");
        return;
    };
    let engine = InferenceEngine::with_weights(Some(weights)).expect("load real weights");
    assert!(
        engine.backend().starts_with("candle-"),
        "expected real Candle backend, got {}",
        engine.backend()
    );
    let out = engine.infer(&SyntheticInput.as_window()).expect("infer");
    assert!(out.is_finite());

    // Chunked into 17 (x, y) pairs — every coordinate must be finite and
    // sit inside the sigmoid output range. This is the "real wiring"
    // assertion the README points at.
    let pairs: Vec<[f32; 2]> = out
        .keypoints
        .chunks_exact(2)
        .map(|c| [c[0], c[1]])
        .collect();
    assert_eq!(pairs.len(), OUTPUT_KEYPOINTS);
    for (i, [x, y]) in pairs.iter().enumerate() {
        assert!(
            x.is_finite() && y.is_finite() && !x.is_nan() && !y.is_nan(),
            "keypoint {i} bad: ({x}, {y})"
        );
        assert!(
            (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y),
            "keypoint {i} out of [0, 1]: ({x}, {y})"
        );
    }

    // Real model emits the published validation PCK@50 as its self-reported
    // confidence — stub returns 0.0. This is the key assertion that proves
    // the cog isn't silently falling back to the stub.
    assert!(
        out.confidence > 0.0,
        "real model should emit non-zero confidence"
    );
}

#[test]
fn manifest_roundtrips() {
    let spec = ManifestSpec::embedded("pose-estimation", "0.0.1");
    let s = serde_json::to_string(&spec).unwrap();
    let back: ManifestSpec = serde_json::from_str(&s).unwrap();
    assert_eq!(back.id, "pose-estimation");
    assert_eq!(back.version, "0.0.1");
}
