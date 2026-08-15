//! Reproducible release-mode latency probe. Results apply only to this host.

use std::{fmt::Write as _, hint::black_box, time::Instant};

use wifi_densepose_core::{
    CalibrationId, Coco17Joint, FloorPlane, JointObservation, JointVisibility, ModelRef,
    PhysicsMode, PoseDimensionality, PoseObservationV2, PoseTrustState, Probability,
    SourceProvenance, SpatialFrameRef, SymmetricCovariance3, TrackId,
};
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = std::env::args()
        .nth(1)
        .map_or(Ok(20_000usize), |value| value.parse())?;
    if samples < 100 {
        return Err("sample count must be at least 100".into());
    }
    let mut engine = PhysicsEngine::new(PhysicsConfig {
        mode: PhysicsMode::ShadowCorrect,
        ..PhysicsConfig::default()
    })?;
    let config_hash = encode_hex(engine.config_hash());
    for sequence in 1..=1_000 {
        let raw = observation(sequence);
        black_box(engine.process(&raw, raw.timestamp_ns));
    }
    let mut nanoseconds = Vec::with_capacity(samples);
    for offset in 0..samples {
        let sequence = 1_001 + offset as u64;
        let raw = observation(sequence);
        let started = Instant::now();
        let result = engine.process(&raw, raw.timestamp_ns);
        nanoseconds.push(started.elapsed().as_nanos() as u64);
        assert!(result.effective_confidence.get() <= raw.observer_confidence.get());
        black_box(result);
    }
    nanoseconds.sort_unstable();
    let percentile =
        |numerator: usize| nanoseconds[((samples - 1) * numerator) / 100] as f64 / 1_000_000.0;
    println!(
        "{{\"evidence\":\"MEASURED\",\"scope\":\"local-host-only\",\"engine\":\"kinematic-pbd\",\"mode\":\"shadow_correct\",\"config_hash\":\"{config_hash}\",\"samples\":{samples},\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"p99_ms\":{:.6},\"max_ms\":{:.6}}}",
        percentile(50),
        percentile(95),
        percentile(99),
        nanoseconds[samples - 1] as f64 / 1_000_000.0,
    );
    Ok(())
}

fn encode_hex(bytes: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

fn observation(sequence: u64) -> PoseObservationV2 {
    let positions = [
        [0.0, 0.0, 1.70],
        [-0.03, 0.0, 1.73],
        [0.03, 0.0, 1.73],
        [-0.08, 0.0, 1.71],
        [0.08, 0.0, 1.71],
        [-0.20, 0.0, 1.45],
        [0.20, 0.0, 1.45],
        [-0.35, 0.0, 1.15],
        [0.35, 0.0, 1.15],
        [-0.45, 0.0, 0.90],
        [0.45, 0.0, 0.90],
        [-0.14, 0.0, 0.90],
        [0.14, 0.0, 0.90],
        [-0.14, 0.0, 0.48],
        [0.14, 0.0, 0.48],
        [-0.14, 0.0, 0.04],
        [0.14, 0.0, 0.04],
    ];
    let mut joints = core::array::from_fn(|index| JointObservation {
        kind: Coco17Joint::ALL[index],
        position_m: positions[index],
        covariance_m2: SymmetricCovariance3 {
            xx: 0.001,
            xy: 0.0,
            xz: 0.0,
            yy: 0.001,
            yz: 0.0,
            zz: 0.001,
        },
        confidence: Probability::new(0.8).expect("fixture confidence is valid"),
        visibility: JointVisibility::Visible,
    });
    joints[15].position_m[2] -= (sequence % 7) as f32 * 0.001;
    let mut raw = PoseObservationV2 {
        schema_version: 2,
        timestamp_ns: 1_000_000_000 + sequence * 33_000_000,
        sensor_epoch: 1,
        sequence,
        track_id: TrackId("latency:1".into()),
        frame: SpatialFrameRef {
            name: "room:latency".into(),
            version: 1,
            metric: true,
            right_handed: true,
            z_up: true,
        },
        calibration_id: CalibrationId("cal:latency".into()),
        floor_plane: Some(FloorPlane {
            normal: [0.0, 0.0, 1.0],
            offset_m: 0.0,
        }),
        model: ModelRef {
            id: "pose:latency".into(),
            artifact_hash: [1; 32],
        },
        source: SourceProvenance {
            sensor_id: "sensor:latency".into(),
            authenticated: false,
            replay_protected: false,
        },
        trust_state: PoseTrustState::Known,
        dimensionality: PoseDimensionality::Metric3d,
        uncertainty_calibrated: true,
        joints,
        observer_confidence: Probability::new(0.78).expect("fixture confidence is valid"),
        canonical_hash: [0; 32],
    };
    raw.seal();
    raw
}
