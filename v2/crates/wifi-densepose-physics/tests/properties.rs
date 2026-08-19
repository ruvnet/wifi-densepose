#![allow(missing_docs)]
mod common;

use proptest::prelude::*;
use wifi_densepose_core::PhysicsMode;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

proptest! {
    #[test]
    fn finite_input_never_increases_confidence(dx in -0.05f32..0.05, dz in -0.05f32..0.05) {
        let mut raw = common::observation(1);
        raw.joints[15].position_m[0] += dx;
        raw.joints[15].position_m[2] += dz;
        raw.seal();
        let mut engine = PhysicsEngine::new(PhysicsConfig { mode: PhysicsMode::ShadowCorrect, ..PhysicsConfig::default() }).unwrap();
        let result = engine.process(&raw, raw.timestamp_ns);
        prop_assert!(result.effective_confidence.get() <= raw.observer_confidence.get());
        if let Some(joints) = result.refined_joints_m {
            prop_assert!(joints.iter().flatten().all(|value| value.is_finite()));
            prop_assert!(result.intervention.max_joint_correction_m <= engine.config().max_joint_correction_m);
        }
    }
}

#[test]
fn translating_pose_and_floor_preserves_audit_residuals() {
    let base = common::observation(1);
    let mut translated = base.clone();
    let delta = [3.0, -2.0, 0.75];
    for joint in &mut translated.joints {
        for (coordinate, offset) in joint.position_m.iter_mut().zip(delta) {
            *coordinate += offset;
        }
    }
    let floor = translated.floor_plane.as_mut().unwrap();
    floor.offset_m -= floor
        .normal
        .iter()
        .zip(delta)
        .map(|(a, b)| a * b)
        .sum::<f32>();
    translated.track_id.0 = "local:translated".into();
    translated.seal();

    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let first = engine.process(&base, base.timestamp_ns);
    let second = engine.process(&translated, translated.timestamp_ns);
    assert!((first.residuals.bone_m - second.residuals.bone_m).abs() < 1.0e-5);
    assert!(
        (first.residuals.floor_penetration_m - second.residuals.floor_penetration_m).abs() < 1.0e-5
    );
}

#[test]
fn mirroring_symmetric_input_preserves_scalar_assessment() {
    let raw = common::observation(1);
    let mut mirrored = raw.clone();
    for joint in &mut mirrored.joints {
        joint.position_m[0] = -joint.position_m[0];
    }
    mirrored.track_id.0 = "local:mirror".into();
    mirrored.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let left = engine.process(&raw, raw.timestamp_ns);
    let right = engine.process(&mirrored, mirrored.timestamp_ns);
    assert!((left.residuals.normalized_total - right.residuals.normalized_total).abs() < 1.0e-5);
}

#[test]
fn independent_track_order_does_not_change_results() {
    let a = common::observation(1);
    let mut b = a.clone();
    b.track_id.0 = "local:8".into();
    b.joints[9].position_m[0] += 0.03;
    b.seal();

    let mut forward = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let forward_a = forward.process(&a, a.timestamp_ns);
    let forward_b = forward.process(&b, b.timestamp_ns);
    let mut reverse = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    let reverse_b = reverse.process(&b, b.timestamp_ns);
    let reverse_a = reverse.process(&a, a.timestamp_ns);
    assert_semantically_equal_ignoring_wall_clock(forward_a, reverse_a);
    assert_semantically_equal_ignoring_wall_clock(forward_b, reverse_b);
}

fn assert_semantically_equal_ignoring_wall_clock(
    mut left: wifi_densepose_core::PoseRefinementV1,
    mut right: wifi_densepose_core::PoseRefinementV1,
) {
    left.intervention.elapsed_us = 0;
    right.intervention.elapsed_us = 0;
    left.seal();
    right.seal();
    assert_eq!(left, right);
}
