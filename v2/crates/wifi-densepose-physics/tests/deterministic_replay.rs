#![allow(missing_docs)]
#![cfg(feature = "deterministic")]
mod common;

use common::observation;
use wifi_densepose_core::PhysicsMode;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

#[test]
fn identical_replay_has_identical_canonical_json() {
    let raw = observation(1);
    let config = PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        ..PhysicsConfig::default()
    };
    let a = PhysicsEngine::new(config.clone())
        .unwrap()
        .process(&raw, raw.timestamp_ns);
    let b = PhysicsEngine::new(config)
        .unwrap()
        .process(&raw, raw.timestamp_ns);
    assert_eq!(
        serde_json::to_vec(&a).unwrap(),
        serde_json::to_vec(&b).unwrap()
    );
}
