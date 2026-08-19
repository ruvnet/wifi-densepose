#![allow(missing_docs)]
#![cfg(feature = "dynamics")]
mod common;

use common::observation;
use wifi_densepose_core::PhysicsMode;
use wifi_densepose_physics::dynamics::DynamicsAuditor;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

#[test]
fn articulated_audit_is_bounded_and_finite() {
    let raw = observation(1);
    let joints = raw.joints.map(|joint| joint.position_m);
    let mut auditor = DynamicsAuditor::new(&joints, raw.floor_plane.unwrap());
    let assessment = auditor.audit(&joints, 1.0 / 30.0, 2);
    assert!(assessment.stable);
    assert_eq!(assessment.segment_count, 14);
    assert!(assessment.joint_count >= 10);
    assert_eq!(assessment.substeps, 2);
    assert!(assessment.max_tracking_error_m.is_finite());
}

#[test]
fn prone_pose_remains_a_valid_dynamics_input() {
    let raw = observation(1);
    let mut joints = raw.joints.map(|joint| joint.position_m);
    for joint in &mut joints {
        joint[2] = 0.05;
    }
    let mut auditor = DynamicsAuditor::new(&joints, raw.floor_plane.unwrap());
    let assessment = auditor.audit(&joints, 1.0 / 30.0, 1);
    assert!(assessment.stable);
}

#[test]
fn dynamics_assessment_is_attached_but_never_selects_shadow_output() {
    let raw = observation(1);
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        deadline_ms: 50,
        dynamics_audit: true,
        ..PhysicsConfig::default()
    })
    .unwrap();
    let result = engine.process(&raw, raw.timestamp_ns);
    assert!(result.dynamics.is_some());
    assert!(!result.selected);
    assert_eq!(result.provenance.engine, "kinematic-pbd+rapier-audit");

    let mut next = observation(2);
    next.joints[9].position_m[0] += 0.01;
    next.seal();
    let next_result = engine.process(&next, next.timestamp_ns);
    assert!(next_result.dynamics.is_some());
    assert!(!next_result.selected);
}
