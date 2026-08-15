use wifi_densepose_core::{
    CalibrationId, Coco17Joint, FloorPlane, JointObservation, JointVisibility, ModelRef,
    PoseDimensionality, PoseObservationV2, PoseTrustState, Probability, SourceProvenance,
    SpatialFrameRef, SymmetricCovariance3, TrackId,
};

pub fn observation(sequence: u64) -> PoseObservationV2 {
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
    let joints = core::array::from_fn(|index| JointObservation {
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
        confidence: Probability::new(0.8).unwrap(),
        visibility: JointVisibility::Visible,
    });
    let mut raw = PoseObservationV2 {
        schema_version: 2,
        timestamp_ns: 1_000_000_000 + sequence * 33_000_000,
        sensor_epoch: 7,
        sequence,
        track_id: TrackId("local:7".into()),
        frame: SpatialFrameRef {
            name: "room:lab".into(),
            version: 1,
            metric: true,
            right_handed: true,
            z_up: true,
        },
        calibration_id: CalibrationId("cal:1".into()),
        floor_plane: Some(FloorPlane {
            normal: [0.0, 0.0, 1.0],
            offset_m: 0.0,
        }),
        model: ModelRef {
            id: "pose:test".into(),
            artifact_hash: [3; 32],
        },
        source: SourceProvenance {
            sensor_id: "sensor:1".into(),
            authenticated: true,
            replay_protected: true,
        },
        trust_state: PoseTrustState::Known,
        dimensionality: PoseDimensionality::Metric3d,
        uncertainty_calibrated: true,
        joints,
        observer_confidence: Probability::new(0.78).unwrap(),
        canonical_hash: [0; 32],
    };
    raw.seal();
    raw
}
