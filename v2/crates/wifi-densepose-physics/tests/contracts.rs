#![allow(missing_docs)]
mod common;

use common::observation;
use wifi_densepose_core::{AbstentionReason, PhysicsMode, RefinementDisposition};
use wifi_densepose_physics::{
    CorrectionAuthorization, PhysicsConfig, PhysicsEngine, VerifiedFrameContext,
};

#[test]
fn raw_hash_is_preserved_and_confidence_never_increases() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(result.raw_observation_hash, raw.canonical_hash);
    assert!(result.effective_confidence.get() <= raw.observer_confidence.get());
    assert!(!result.selected);
}

#[test]
fn mode_off_is_identity_and_clears_selection() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::Off,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(result.disposition, RefinementDisposition::Bypassed);
    assert_eq!(result.reason, Some(AbstentionReason::ModeOff));
    assert!(result.refined_joints_m.is_none());
    assert_eq!(result.effective_confidence, raw.observer_confidence);
}

#[test]
fn unauthenticated_source_cannot_select_correction() {
    let mut raw = observation(1);
    raw.source.authenticated = false;
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::OptInCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    engine
        .authorize_correction(
            CorrectionAuthorization::from_verified_evidence(engine.config_hash(), [7; 32]).unwrap(),
        )
        .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(result.reason, Some(AbstentionReason::SourceUnauthenticated));
    assert!(!result.selected);
}

#[test]
fn correction_mode_requires_evidence_authorization() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::OptInCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(
        result.reason,
        Some(AbstentionReason::CorrectionNotAuthorized)
    );
    assert!(!result.selected);
}

#[test]
fn verified_frame_and_release_receipts_allow_bounded_selection() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::OptInCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    engine
        .authorize_correction(
            CorrectionAuthorization::from_verified_evidence(engine.config_hash(), [7; 32]).unwrap(),
        )
        .unwrap();
    let receipt = VerifiedFrameContext::from_verified_envelope(
        raw.canonical_hash,
        raw.sensor_epoch,
        raw.sequence,
        raw.source.sensor_id.clone(),
        raw.calibration_id.0.clone(),
        [8; 32],
    )
    .unwrap();
    let result = engine.process_verified(&raw, raw.timestamp_ns, &receipt);
    assert!(result.selected);
}

#[test]
fn audit_reports_raw_residuals_without_running_projector() {
    let mut raw = observation(1);
    raw.joints[15].position_m[2] = -0.04;
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(result.disposition, RefinementDisposition::Audited);
    assert!(result.residuals.floor_penetration_m >= 0.039);
    assert_eq!(result.intervention.solver_iterations, 0);
    assert!(result.refined_residuals.is_none());
}

#[test]
fn result_hash_is_canonical_and_detects_changes() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(result.compute_canonical_hash(), result.canonical_hash);
    let mut changed = result.clone();
    changed.effective_confidence = wifi_densepose_core::Probability::ZERO;
    assert_ne!(changed.compute_canonical_hash(), result.canonical_hash);
}

#[test]
fn correction_mode_transition_is_atomic_and_evidence_bound() {
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    assert!(engine.set_mode(PhysicsMode::DefaultCorrect).is_err());
    assert_eq!(engine.config().mode, PhysicsMode::Audit);
    let target_hash = engine.config_hash_for_mode(PhysicsMode::OptInCorrect);
    engine
        .authorize_correction(
            CorrectionAuthorization::from_verified_evidence(target_hash, [9; 32]).unwrap(),
        )
        .unwrap();
    engine.set_mode(PhysicsMode::OptInCorrect).unwrap();
    assert_eq!(engine.config().mode, PhysicsMode::OptInCorrect);
}

#[test]
fn duplicate_is_idempotent_but_different_content_is_replay_rejected() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let first = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(first, engine.process(&raw, raw.timestamp_ns));
    let mut tampered = raw.clone();
    tampered.joints[0].position_m[0] += 0.01;
    tampered.seal();
    assert_eq!(
        engine.process(&tampered, tampered.timestamp_ns).reason,
        Some(AbstentionReason::ReplayRejected)
    );
}

#[test]
fn calibration_change_discards_track_temporal_state() {
    let first = observation(1);
    let mut second = observation(2);
    for joint in &mut second.joints {
        joint.position_m[0] += 0.20;
    }
    second.calibration_id.0 = "cal:2".into();
    second.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let _ = engine.process(&first, first.timestamp_ns);
    let result = engine.process(&second, second.timestamp_ns);
    assert!(result.residuals.velocity_mps.abs() <= f32::EPSILON);
    assert!(result.residuals.acceleration_mps2.abs() <= f32::EPSILON);
}
