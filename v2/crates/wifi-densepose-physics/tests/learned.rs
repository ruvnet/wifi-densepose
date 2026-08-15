#![allow(missing_docs)]
#![cfg(feature = "learned")]

mod common;

use common::observation;
use wifi_densepose_core::ConstraintResiduals;
use wifi_densepose_physics::{
    constraints::TemporalHistory,
    learned::{
        features::{FeatureHistory, PoseFeatureFrame, FEATURE_WIDTH},
        training::LossComponents,
        LearnedArtifact, LearnedArtifactError, LearnedArtifactManifest, LearnedResidual,
        SignatureVerification,
    },
};

#[test]
fn feature_encoder_is_fixed_width_finite_and_requires_complete_history() {
    let raw = observation(1);
    let frame = PoseFeatureFrame::encode(
        &raw,
        ConstraintResiduals::default(),
        &TemporalHistory::default(),
        None,
    );
    assert_eq!(frame.values().len(), FEATURE_WIDTH);
    assert!(frame.values().iter().all(|value| value.is_finite()));

    let mut history = FeatureHistory::default();
    for _ in 0..19 {
        history.push(frame.clone());
    }
    assert!(!history.is_ready());
    assert!(history.flattened().is_none());
    history.push(frame);
    assert_eq!(history.flattened().unwrap().len(), 20 * FEATURE_WIDTH);
}

#[test]
fn training_objective_has_declared_weights_and_rejects_invalid_components() {
    let components = LossComponents {
        uncertainty_weighted_mpjpe: 1.0,
        bone: 1.0,
        temporal_jerk: 1.0,
        contact: 1.0,
        uncertainty_calibration: 1.0,
        intervention: 1.0,
    };
    assert!((components.total().unwrap() - 1.55).abs() < 1.0e-6);
    assert!(LossComponents {
        bone: f32::NAN,
        ..components
    }
    .total()
    .is_none());
    assert!(LossComponents {
        intervention: -1.0,
        ..components
    }
    .total()
    .is_none());
}

#[test]
fn learned_artifact_requires_hash_manifest_size_and_signature_receipts() {
    let bytes = b"signed-burn-record".to_vec();
    let hash = *blake3::hash(&bytes).as_bytes();
    let manifest = LearnedArtifactManifest::adr323("pose-residual-v1".into());
    let signature = SignatureVerification::accepted("release-key".into(), [7; 32]).unwrap();
    let artifact = LearnedArtifact::verified(
        bytes.clone(),
        hash,
        manifest.clone(),
        signature.clone(),
        bytes.len(),
    )
    .unwrap();
    assert_eq!(artifact.bytes(), bytes);
    assert_eq!(artifact.content_hash, hash);

    assert_eq!(
        LearnedArtifact::verified(
            bytes.clone(),
            [9; 32],
            manifest.clone(),
            signature.clone(),
            64
        )
        .unwrap_err(),
        LearnedArtifactError::HashMismatch
    );
    assert_eq!(
        LearnedArtifact::verified(bytes.clone(), hash, manifest.clone(), signature.clone(), 1)
            .unwrap_err(),
        LearnedArtifactError::ArtifactTooLarge
    );
    let mut incompatible = manifest;
    incompatible.feature_width += 1;
    assert_eq!(
        LearnedArtifact::verified(bytes, hash, incompatible, signature, 64).unwrap_err(),
        LearnedArtifactError::IncompatibleManifest
    );
}

#[test]
fn learned_heads_are_finite_and_hard_bounded() {
    let residual = LearnedResidual {
        joints_m: [[[0.50, -0.50, 0.01]; 17][0]; 17],
        log_variance: [[[-40.0, 40.0, 0.0]; 17][0]; 17],
        contact_probability: [-1.0, 2.0],
        abstention_probability: 1.5,
    }
    .bounded(0.20)
    .unwrap();
    assert!(residual
        .joints_m
        .iter()
        .flatten()
        .all(|coordinate| coordinate.abs() <= 0.20));
    assert!(residual
        .contact_probability
        .iter()
        .zip([0.0, 1.0])
        .all(|(actual, expected)| (actual - expected).abs() <= f32::EPSILON));
    assert!((residual.abstention_probability - 1.0).abs() <= f32::EPSILON);
    assert!(residual.log_variance[0]
        .iter()
        .zip([-20.0, 10.0, 0.0])
        .all(|(actual, expected)| (actual - expected).abs() <= f32::EPSILON));

    let mut invalid = residual;
    invalid.joints_m[0][0] = f32::NAN;
    assert!(invalid.bounded(0.20).is_none());
}

#[cfg(feature = "learned-cpu")]
#[test]
fn burn_gru_executes_and_roundtrips_verified_record_on_cpu() {
    use burn_core::tensor::Tensor;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use wifi_densepose_physics::learned::model::{
        ResidualGru, HIDDEN_WIDTH, HISTORY_FRAMES, JOINT_RESIDUAL_WIDTH,
    };

    type Backend = NdArray<f32>;
    let device = NdArrayDevice::default();
    let model = ResidualGru::<Backend>::init(&device);
    assert!(burn_core::module::Module::num_params(&model) > HIDDEN_WIDTH);
    let bytes = model.into_artifact_bytes().unwrap();
    let hash = *blake3::hash(&bytes).as_bytes();
    let artifact = LearnedArtifact::verified(
        bytes.clone(),
        hash,
        LearnedArtifactManifest::adr323("cpu-roundtrip".into()),
        SignatureVerification::accepted("test-release-key".into(), [5; 32]).unwrap(),
        bytes.len(),
    )
    .unwrap();
    let loaded = ResidualGru::<Backend>::from_verified_artifact(&artifact, &device).unwrap();
    let input = Tensor::<Backend, 3>::zeros([1, HISTORY_FRAMES, FEATURE_WIDTH], &device);
    let output = loaded.forward(input);
    assert_eq!(
        output.joint_residual.shape().dims(),
        [1, JOINT_RESIDUAL_WIDTH]
    );
    assert_eq!(
        output.residual_log_variance.shape().dims(),
        [1, JOINT_RESIDUAL_WIDTH]
    );
    assert_eq!(output.contact_probability.shape().dims(), [1, 2]);
    assert_eq!(output.abstention_probability.shape().dims(), [1, 1]);

    let runtime =
        wifi_densepose_physics::learned::runtime::CpuResidualRuntime::activate(&artifact).unwrap();
    assert_eq!(runtime.artifact_hash(), hash);
    let prediction = runtime
        .predict(&vec![0.0; 20 * FEATURE_WIDTH], 0.20)
        .unwrap();
    assert!(prediction
        .joints_m
        .iter()
        .flatten()
        .all(|value| value.is_finite() && value.abs() <= 0.20));
    assert!(runtime.predict(&[0.0; 3], 0.20).is_none());
}
