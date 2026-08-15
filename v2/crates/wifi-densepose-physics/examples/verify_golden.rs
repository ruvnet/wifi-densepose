//! Deterministic L0 replay verifier for ADR-323 golden hashes.

use std::{fmt::Write as _, path::Path};

use wifi_densepose_core::{
    CalibrationId, Coco17Joint, FloorPlane, JointObservation, JointVisibility, ModelRef,
    PoseDimensionality, PoseObservationV2, PoseTrustState, Probability, SourceProvenance,
    SpatialFrameRef, SymmetricCovariance3, TrackId,
};
use wifi_densepose_physics::{PhysicsConfig, PhysicsEngine};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let path = arguments
        .next()
        .ok_or("usage: verify_golden <golden.jsonl> [--emit|--contracts]")?;
    let mode = arguments.next();
    let emit = mode.as_deref() == Some("--emit");
    let contracts = mode.as_deref() == Some("--contracts");
    if mode.is_some() && !emit && !contracts {
        return Err("mode must be --emit or --contracts".into());
    }
    let contents = std::fs::read_to_string(Path::new(&path))?;
    let mut count = 0usize;
    for (line_index, line) in contents
        .lines()
        .filter(|line| !line.trim().is_empty())
        .enumerate()
    {
        let mut value: serde_json::Value = serde_json::from_str(line)?;
        let sequence = value["sequence"].as_u64().ok_or("sequence must be u64")?;
        let penetration = value["ankle_penetration_m"]
            .as_f64()
            .ok_or("ankle_penetration_m must be numeric")? as f32;
        if value["evidence"] != "SYNTHETIC/L0" {
            return Err(format!("line {}: evidence must be SYNTHETIC/L0", line_index + 1).into());
        }
        let mut raw = observation(sequence);
        raw.joints[15].position_m[2] -= penetration;
        raw.seal();
        let result = PhysicsEngine::new(PhysicsConfig::default())?.process(&raw, raw.timestamp_ns);
        let raw_hash = encode_hex(raw.canonical_hash);
        let result_hash = encode_hex(result.canonical_hash);
        if emit {
            value["expected_raw_hash"] = raw_hash.clone().into();
            value["expected_result_hash"] = result_hash.clone().into();
            println!("{}", serde_json::to_string(&value)?);
        } else {
            if value["expected_raw_hash"] != raw_hash
                || value["expected_result_hash"] != result_hash
            {
                return Err(format!(
                    "line {}: hash mismatch (raw={raw_hash}, result={result_hash})",
                    line_index + 1
                )
                .into());
            }
            if contracts {
                println!(
                    "{}",
                    serde_json::to_string(&serde_json::json!({ "raw": raw, "result": result }))?
                );
            }
        }
        count += 1;
    }
    if count == 0 {
        return Err("golden replay is empty".into());
    }
    if !emit && !contracts {
        println!("{{\"verdict\":\"PASS\",\"records\":{count},\"engine\":\"kinematic-pbd\",\"evidence\":\"SYNTHETIC/L0\"}}");
    }
    Ok(())
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
        confidence: Probability::new(0.8).expect("fixture confidence is valid"),
        visibility: JointVisibility::Visible,
    });
    let mut raw = PoseObservationV2 {
        schema_version: 2,
        timestamp_ns: 1_000_000_000 + sequence * 33_000_000,
        sensor_epoch: 7,
        sequence,
        track_id: TrackId("golden:1".into()),
        frame: SpatialFrameRef {
            name: "room:golden".into(),
            version: 1,
            metric: true,
            right_handed: true,
            z_up: true,
        },
        calibration_id: CalibrationId("cal:golden".into()),
        floor_plane: Some(FloorPlane {
            normal: [0.0, 0.0, 1.0],
            offset_m: 0.0,
        }),
        model: ModelRef {
            id: "pose:golden".into(),
            artifact_hash: [3; 32],
        },
        source: SourceProvenance {
            sensor_id: "sensor:golden".into(),
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

fn encode_hex(bytes: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}
