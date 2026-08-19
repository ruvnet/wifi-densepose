#![allow(missing_docs)]
mod common;

use common::observation;
use wifi_densepose_core::{PhysicsMode, PoseDimensionality, RefinementDisposition};
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

#[test]
fn image_2d_is_audited_but_never_corrected() {
    let mut raw = observation(1);
    raw.dimensionality = PoseDimensionality::Image2d;
    raw.frame.metric = false;
    raw.floor_plane = None;
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::OptInCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert_eq!(result.disposition, RefinementDisposition::Audited2d);
    assert!(!result.selected && result.refined_joints_m.is_none());
}

#[test]
fn image_pixels_use_a_separate_bounded_coordinate_domain() {
    let mut raw = observation(1);
    raw.dimensionality = PoseDimensionality::Image2d;
    raw.frame.metric = false;
    raw.frame.right_handed = false;
    raw.frame.z_up = false;
    raw.floor_plane = None;
    raw.joints[0].position_m = [640.0, 480.0, 0.0];
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    assert_eq!(
        engine.process(&raw, raw.timestamp_ns).disposition,
        RefinementDisposition::Audited2d
    );

    raw.sequence += 1;
    raw.timestamp_ns += 33_000_000;
    raw.joints[0].position_m[0] = 20_000.0;
    raw.seal();
    assert_eq!(
        engine.process(&raw, raw.timestamp_ns).reason,
        Some(wifi_densepose_core::AbstentionReason::InvalidNumber)
    );
}

#[test]
fn prone_pose_is_not_forced_upright() {
    let mut raw = observation(1);
    for joint in &mut raw.joints {
        joint.position_m[2] = 0.05;
    }
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    let refined = result.refined_joints_m.unwrap();
    assert!(refined.iter().all(|joint| joint[2] < 0.20));
}

#[test]
fn image_2d_audit_tracks_temporal_motion() {
    let mut first = observation(1);
    first.dimensionality = PoseDimensionality::Image2d;
    first.frame.metric = false;
    first.floor_plane = None;
    first.seal();
    let mut second = first.clone();
    second.sequence = 2;
    second.timestamp_ns += 33_000_000;
    second.joints[9].position_m[0] += 0.5;
    second.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let _ = engine.process(&first, first.timestamp_ns);
    let result = engine.process(&second, second.timestamp_ns);
    assert!(result.residuals.velocity_mps > 0.0);
}

#[test]
fn image_2d_rejects_non_monotonic_sequence_and_time() {
    let mut first = observation(2);
    first.dimensionality = PoseDimensionality::Image2d;
    first.frame.metric = false;
    first.frame.right_handed = false;
    first.frame.z_up = false;
    first.floor_plane = None;
    first.uncertainty_calibrated = false;
    first.seal();
    let mut older = first.clone();
    older.sequence = 1;
    older.timestamp_ns -= 1;
    older.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    assert_eq!(
        engine.process(&first, first.timestamp_ns).disposition,
        RefinementDisposition::Audited2d
    );
    assert_eq!(
        engine.process(&older, older.timestamp_ns).reason,
        Some(wifi_densepose_core::AbstentionReason::NonMonotonicInput)
    );
}
