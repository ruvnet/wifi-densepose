#![no_main]

use libfuzzer_sys::fuzz_target;
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

fuzz_target!(|bytes: &[u8]| {
    if bytes.len() > 64 * 1024 {
        return;
    }
    if let Ok(raw) = serde_json::from_slice::<wifi_densepose_core::PoseObservationV2>(bytes) {
        if let Ok(mut engine) = PhysicsEngine::new(PhysicsConfig::default()) {
            let _ = engine.process(&raw, raw.timestamp_ns);
        }
    }
});
