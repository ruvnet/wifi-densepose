#![allow(missing_docs)]
mod common;

use common::observation;
use wifi_densepose_core::AbstentionReason;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

#[test]
fn rejects_non_psd_covariance() {
    let mut raw = observation(1);
    raw.joints[0].covariance_m2.xy = 1.0;
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    assert_eq!(
        engine.process(&raw, raw.timestamp_ns).reason,
        Some(AbstentionReason::InvalidCovariance)
    );
}

#[test]
fn rejects_nan() {
    let mut raw = observation(1);
    raw.joints[0].position_m[0] = f32::NAN;
    raw.seal();
    let mut engine = PhysicsEngine::new(PhysicsConfig::default()).unwrap();
    assert_eq!(
        engine.process(&raw, raw.timestamp_ns).reason,
        Some(AbstentionReason::InvalidNumber)
    );
}
