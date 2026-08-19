#![allow(missing_docs)]
mod common;

use common::observation;
use wifi_densepose_core::PhysicsMode;
use wifi_densepose_physics::skeleton::SkeletonPosterior;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

#[test]
fn projector_bounds_vector_acceleration() {
    let config = PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        max_velocity_mps: 4.0,
        max_acceleration_mps2: 5.0,
        ..PhysicsConfig::default()
    };
    let mut engine = PhysicsEngine::new(config).unwrap();
    let first = observation(1);
    let _ = engine.process(&first, first.timestamp_ns);

    let mut second = observation(2);
    for joint in &mut second.joints {
        joint.position_m[0] += 0.02;
    }
    second.seal();
    let _ = engine.process(&second, second.timestamp_ns);

    let mut third = observation(3);
    for joint in &mut third.joints {
        joint.position_m[0] -= 0.02;
    }
    third.seal();
    let result = engine.process(&third, third.timestamp_ns);
    let refined = result.refined_residuals.expect("shadow candidate");
    assert!(refined.acceleration_mps2 <= 5.01, "{refined:?}");
}

#[test]
fn projector_reduces_impossible_elbow_angle() {
    let mut raw = observation(1);
    raw.joints[9].position_m = [-0.25, 0.0, 1.30];
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    let refined = result.refined_residuals.expect("shadow candidate");
    assert!(refined.joint_limit_rad < result.residuals.joint_limit_rad);
}

#[test]
fn one_outlier_does_not_redefine_calibrated_bone_length() {
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        ..PhysicsConfig::default()
    })
    .unwrap();
    for sequence in 1..=6 {
        let raw = observation(sequence);
        let _ = engine.process(&raw, raw.timestamp_ns);
    }
    let mut outlier = observation(7);
    outlier.joints[9].position_m[0] -= 0.14;
    outlier.seal();
    let result = engine.process(&outlier, outlier.timestamp_ns);
    assert!(result.residuals.bone_m > 0.001, "{:?}", result.residuals);
    assert!(result.refined_residuals.expect("shadow candidate").bone_m < result.residuals.bone_m);
}

#[test]
fn bone_posterior_rejects_single_calibration_outlier() {
    let baseline = observation(1);
    let joints = baseline.joints.map(|joint| joint.position_m);
    let confidence = [0.9; 17];
    let mut posterior = SkeletonPosterior::default();
    for _ in 0..6 {
        posterior.observe(&joints, &confidence, 0.7);
    }
    let target = posterior.target(7, 0.0);
    let mut outlier = joints;
    outlier[9][0] -= 0.5;
    posterior.observe(&outlier, &confidence, 0.7);
    assert!((posterior.target(7, 0.0) - target).abs() < 0.005);
}
