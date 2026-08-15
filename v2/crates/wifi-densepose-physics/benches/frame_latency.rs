#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wifi_densepose_core::{
    CalibrationId, Coco17Joint, FloorPlane, JointObservation, JointVisibility, ModelRef,
    PhysicsMode, PoseDimensionality, PoseObservationV2, PoseTrustState, Probability,
    SourceProvenance, SpatialFrameRef, SymmetricCovariance3, TrackId,
};
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

fn observation(sequence: u64) -> PoseObservationV2 {
    let joints = Coco17Joint::ALL.map(|kind| JointObservation {
        kind,
        position_m: [f32::from(kind as u8) * 0.01, 0.0, 1.0],
        covariance_m2: SymmetricCovariance3 {
            xx: 0.001,
            xy: 0.0,
            xz: 0.0,
            yy: 0.001,
            yz: 0.0,
            zz: 0.001,
        },
        confidence: Probability::new(0.8).unwrap(),
        visibility: JointVisibility::Visible,
    });
    let mut raw = PoseObservationV2 {
        schema_version: 2,
        timestamp_ns: 1_000_000_000 + sequence * 33_000_000,
        sensor_epoch: 1,
        sequence,
        track_id: TrackId("bench".into()),
        frame: SpatialFrameRef {
            name: "room".into(),
            version: 1,
            metric: true,
            right_handed: true,
            z_up: true,
        },
        calibration_id: CalibrationId("cal".into()),
        floor_plane: Some(FloorPlane {
            normal: [0.0, 0.0, 1.0],
            offset_m: 0.0,
        }),
        model: ModelRef {
            id: "bench".into(),
            artifact_hash: [1; 32],
        },
        source: SourceProvenance {
            sensor_id: "bench".into(),
            authenticated: true,
            replay_protected: true,
        },
        trust_state: PoseTrustState::Known,
        dimensionality: PoseDimensionality::Metric3d,
        uncertainty_calibrated: true,
        joints,
        observer_confidence: Probability::new(0.8).unwrap(),
        canonical_hash: [0; 32],
    };
    raw.seal();
    raw
}

fn bench_frame(c: &mut Criterion) {
    c.bench_function("kinematic_one_track", |b| {
        b.iter_batched(
            || {
                PhysicsEngine::new(PhysicsConfig {
                    mode: PhysicsMode::ShadowCorrect,
                    ..PhysicsConfig::default()
                })
                .unwrap()
            },
            |mut engine| {
                let raw = observation(1);
                black_box(engine.process(&raw, raw.timestamp_ns));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_frame);
criterion_main!(benches);
