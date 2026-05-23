//! Smoke tests for cog-person-count.

use cog_person_count::{
    fusion::{fuse_confidence_weighted, fuse_with_mincut_clip},
    inference::{
        CountPrediction, CsiWindow, InferenceEngine, SyntheticInput, COUNT_CLASSES,
        INPUT_SUBCARRIERS, INPUT_TIMESTEPS,
    },
};

#[test]
fn synthetic_window_has_correct_shape() {
    let w = SyntheticInput.as_window();
    assert_eq!(w.data.len(), INPUT_SUBCARRIERS * INPUT_TIMESTEPS);
}

#[test]
fn stub_engine_returns_finite_output() {
    let engine = InferenceEngine::with_weights(None).expect("stub engine");
    let pred = engine.infer(&SyntheticInput.as_window()).expect("infer");
    assert!(pred.is_finite());
    assert_eq!(pred.probs.len(), COUNT_CLASSES);

    let sum: f32 = pred.probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "stub probs must sum to 1, got {}",
        sum
    );
    assert_eq!(pred.argmax(), 1, "stub default is 1-person");
    assert_eq!(pred.confidence, 0.0, "stub confidence is 0");
}

#[test]
fn engine_rejects_wrong_shape_input() {
    let engine = InferenceEngine::with_weights(None).expect("stub engine");
    let bad = CsiWindow {
        data: vec![0.0; 10],
    };
    assert!(engine.infer(&bad).is_err());
}

#[test]
fn stub_backend_string_is_stable() {
    let engine = InferenceEngine::with_weights(None).expect("stub engine");
    assert_eq!(engine.backend(), "stub");
}

#[test]
fn p95_range_includes_mode() {
    // Sharp peak at 2
    let mut probs = [0.0_f32; COUNT_CLASSES];
    probs[2] = 0.85;
    probs[1] = 0.08;
    probs[3] = 0.07;
    let p = CountPrediction {
        probs,
        confidence: 0.9,
    };
    let (lo, hi) = p.p95_range();
    assert!(lo <= 2 && hi >= 2);
}

#[test]
fn fusion_with_no_inputs_is_safe_default() {
    let p = fuse_confidence_weighted(&[]);
    assert_eq!(p.argmax(), 1);
    assert_eq!(p.confidence, 0.0);
}

#[test]
fn fusion_passes_through_single_node() {
    // A single-node ESP32 deployment must produce the same output as the
    // raw inference — fusion is a no-op for N=1.
    let mut probs = [0.0_f32; COUNT_CLASSES];
    probs[3] = 1.0;
    let input = CountPrediction {
        probs,
        confidence: 0.6,
    };
    let out = fuse_confidence_weighted(std::slice::from_ref(&input));
    assert_eq!(out.argmax(), 3);
    assert!((out.confidence - 0.6).abs() < 1e-6);
}

#[test]
fn mincut_clip_with_high_cap_is_noop() {
    let mut probs = [0.0_f32; COUNT_CLASSES];
    probs[2] = 0.5;
    probs[3] = 0.5;
    let input = CountPrediction {
        probs,
        confidence: 0.7,
    };
    let clipped = fuse_with_mincut_clip(&[input], 7);
    // No clip happened (cap == max class)
    assert!((clipped.probs[2] - 0.5).abs() < 1e-6);
    assert!((clipped.probs[3] - 0.5).abs() < 1e-6);
}
