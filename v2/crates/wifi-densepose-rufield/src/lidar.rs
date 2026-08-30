//! Apple ARKit scene-depth wire boundary → HAL → signed RuField event.
//!
//! Raw depth is validated and summarized immediately. Only bounded P1 visible-
//! geometry features enter the RuField ring; neither the base64 depth payload
//! nor an inferred person-presence claim is copied into the event.

use std::collections::BTreeMap;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use rufield_core::{
    FieldAxis, FieldEvent, FieldTensor, Modality, Observation, PrivacyClass, ProvenanceRef,
    SensorDescriptor,
};
use rufield_provenance::{sha256_hex, Signer};
use ruview_hal::{
    AppleLidarDepthAdapter, AppleLidarDepthSample, Confidence, Modality as HalModality,
    NormalizeCtx, SensorHal,
};
use ruview_ontology::{Container, EvidenceLevel, ObservationId, SensorId, SpaceId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MODEL_ID: &str = "apple-arkit-scene-depth@1";
const MAX_PACKET_SAMPLES: usize = 262_144;

/// Exact `ruview.lidar.depth.v1` packet emitted by the iOS native module.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LidarDepthPacket {
    /// Wire discriminator.
    #[serde(rename = "type")]
    pub packet_type: String,
    /// Camera intrinsics.
    pub intrinsics: LidarIntrinsics,
    /// Camera-to-world pose.
    pub pose: LidarPose,
    /// Bounded downsampled depth image.
    pub depth: LidarDepthPayload,
    /// Capture identity and privacy metadata.
    pub provenance: LidarProvenance,
}

/// ARKit camera intrinsics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LidarIntrinsics {
    /// Horizontal focal length.
    pub fx: f32,
    /// Vertical focal length.
    pub fy: f32,
    /// Horizontal principal point.
    pub cx: f32,
    /// Vertical principal point.
    pub cy: f32,
    /// Captured-image width.
    pub image_width: u32,
    /// Captured-image height.
    pub image_height: u32,
}

/// Camera-to-world transform.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LidarPose {
    /// Column-major 4×4 transform.
    pub matrix: Vec<f32>,
}

/// Downsampled depth payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LidarDepthPayload {
    /// Depth width.
    pub width: u32,
    /// Depth height.
    pub height: u32,
    /// Must be `u16le-mm+u8-confidence`.
    pub encoding: String,
    /// Little-endian millimetres.
    pub millimeters_base64: String,
    /// ARKit confidence bytes.
    pub confidence_base64: String,
}

/// Capture provenance emitted by the trusted native boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LidarProvenance {
    /// Sensor implementation identifier.
    pub sensor: String,
    /// Capture session identity.
    pub session_id: String,
    /// ARKit world-coordinate identity.
    pub coordinate_frame_id: String,
    /// Must be `live` for this path.
    pub source: String,
    /// Must be geometry-only.
    pub privacy_class: String,
    /// Monotonic frame sequence within the session.
    pub sequence: u64,
    /// Unix capture time in nanoseconds.
    pub timestamp_ns: u64,
    /// ARKit monotonic capture time in nanoseconds.
    pub capture_time_ns: u64,
    /// Mapping used between monotonic and wall clock domains.
    pub clock_model_id: String,
    /// Spatial calibration handle.
    pub calibration_id: String,
    /// ARKit tracking state associated with the depth frame.
    pub tracking_state: String,
    /// Runtime evidence label for a physical live frame.
    pub evidence: String,
    /// Wire schema identifier.
    pub schema: String,
}

/// Rejection reasons at the untrusted LiDAR wire boundary.
#[derive(Debug, Error)]
pub enum LidarBridgeError {
    /// Packet schema/provenance did not match the supported live producer.
    #[error("invalid LiDAR packet metadata: {0}")]
    InvalidMetadata(&'static str),
    /// Base64 could not be decoded.
    #[error("invalid LiDAR base64 payload")]
    InvalidBase64,
    /// Dimensions and payload lengths did not agree or exceeded bounds.
    #[error("invalid LiDAR depth shape")]
    InvalidShape,
    /// HAL normalization returned UNKNOWN/degraded data.
    #[error("LiDAR HAL rejected the depth sample")]
    HalRejected,
    /// A bounded ontology identifier was invalid.
    #[error("invalid LiDAR identity")]
    InvalidIdentity,
    /// RuField tensor validation failed.
    #[error("invalid RuField LiDAR tensor")]
    InvalidTensor,
}

/// Validate, normalize, summarize, and sign one Apple scene-depth packet.
pub fn lidar_packet_to_field_event(
    packet: &LidarDepthPacket,
    signer: &Signer,
) -> Result<FieldEvent, LidarBridgeError> {
    validate_metadata(packet)?;
    let millimeter_bytes = STANDARD
        .decode(&packet.depth.millimeters_base64)
        .map_err(|_| LidarBridgeError::InvalidBase64)?;
    let confidences = STANDARD
        .decode(&packet.depth.confidence_base64)
        .map_err(|_| LidarBridgeError::InvalidBase64)?;
    let sample_count = usize::try_from(packet.depth.width)
        .ok()
        .and_then(|width| {
            usize::try_from(packet.depth.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .filter(|count| *count > 0 && *count <= MAX_PACKET_SAMPLES)
        .ok_or(LidarBridgeError::InvalidShape)?;
    if millimeter_bytes.len() != sample_count * 2 || confidences.len() != sample_count {
        return Err(LidarBridgeError::InvalidShape);
    }
    let millimeters: Vec<u16> = millimeter_bytes
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect();
    let transform: [f32; 16] = packet
        .pose
        .matrix
        .clone()
        .try_into()
        .map_err(|_| LidarBridgeError::InvalidShape)?;
    let intrinsics = [
        packet.intrinsics.fx,
        0.0,
        0.0,
        0.0,
        packet.intrinsics.fy,
        0.0,
        packet.intrinsics.cx,
        packet.intrinsics.cy,
        1.0,
    ];
    let raw_hash = sha256_hex(
        &[
            millimeter_bytes.as_slice(),
            confidences.as_slice(),
            packet.provenance.session_id.as_bytes(),
            packet.provenance.coordinate_frame_id.as_bytes(),
        ]
        .concat(),
    );
    let sensor_id = SensorId::new(format!("iphone-lidar-{}", packet.provenance.session_id))
        .map_err(|_| LidarBridgeError::InvalidIdentity)?;
    let adapter = AppleLidarDepthAdapter {
        sensor_id: sensor_id.clone(),
        calibration_version: packet.provenance.calibration_id.clone(),
    };
    let context = NormalizeCtx {
        observation_id: ObservationId::new(format!(
            "lidar-{}-{}",
            packet.provenance.session_id, packet.provenance.sequence
        ))
        .map_err(|_| LidarBridgeError::InvalidIdentity)?,
        located_in: Container::Space {
            id: SpaceId::new(packet.provenance.coordinate_frame_id.clone())
                .map_err(|_| LidarBridgeError::InvalidIdentity)?,
        },
        at_unix_ms: i64::try_from(packet.provenance.timestamp_ns / 1_000_000).unwrap_or(i64::MAX),
    };
    let hal = adapter.normalize(
        AppleLidarDepthSample {
            width: packet.depth.width,
            height: packet.depth.height,
            millimeters: millimeters.clone(),
            confidences: confidences.clone(),
            intrinsics,
            camera_transform: transform,
            evidence_hash: raw_hash.clone(),
        },
        &context,
    );
    if hal.modality != HalModality::Lidar
        || hal.evidence_level() != EvidenceLevel::L2
        || hal.uncertainty.degraded
    {
        return Err(LidarBridgeError::HalRejected);
    }

    let valid: Vec<f32> = millimeters
        .iter()
        .zip(&confidences)
        .filter_map(|(&millimeters, &confidence)| {
            (millimeters >= 150 && millimeters <= 12_000 && confidence >= 1)
                .then_some(f32::from(millimeters) / 1000.0)
        })
        .collect();
    if valid.is_empty() {
        return Err(LidarBridgeError::HalRejected);
    }
    let min_range = valid.iter().copied().fold(f32::INFINITY, f32::min);
    let max_range = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean_range = valid.iter().sum::<f32>() / valid.len() as f32;
    let coverage = valid.len() as f32 / sample_count as f32;
    let high_confidence =
        confidences.iter().filter(|&&value| value == 2).count() as f32 / sample_count as f32;
    let confidence = match hal.uncertainty.confidence {
        Confidence::Known(value) => value as f32,
        Confidence::Unknown => return Err(LidarBridgeError::HalRejected),
    };
    let values = vec![min_range, mean_range, max_range, coverage, high_confidence];
    let calibration_id = hal.provenance().calibration_version.clone();
    let tensor = FieldTensor::new(
        packet.provenance.timestamp_ns,
        Modality::LidarPhase,
        vec![FieldAxis::Channel],
        vec![values.len()],
        values,
        confidence,
        1.0 - confidence,
        Some(calibration_id.clone()),
        PrivacyClass::P1,
    )
    .map_err(|_| LidarBridgeError::InvalidTensor)?;
    let mut features = BTreeMap::new();
    features.insert("direct_visible_depth".into(), 1.0);
    features.insert("depth_coverage".into(), coverage);
    features.insert("mean_range_m".into(), mean_range);
    features.insert("high_confidence_fraction".into(), high_confidence);
    let observation = Observation {
        zone_id: Some(packet.provenance.coordinate_frame_id.clone()),
        space_cell: None,
        range_m: Some(mean_range),
        velocity_mps: None,
        motion_vector: None,
        confidence,
        features,
        labels: vec!["direct_visible_depth".into(), "not_nlos".into()],
        privacy_class: PrivacyClass::P1,
    };
    let provenance = ProvenanceRef {
        raw_hash,
        firmware_hash: sha256_hex(MODEL_ID.as_bytes()),
        model_id: MODEL_ID.into(),
        calibration_id,
        synthetic: false,
        signature_hex: None,
        signer_pubkey_hex: None,
    };
    let sensor = SensorDescriptor {
        modality: "lidar_phase".into(),
        vendor: "apple_arkit".into(),
        device_id: sensor_id.to_string(),
        placement: packet.provenance.coordinate_frame_id.clone(),
        clock_domain: "iphone_unix".into(),
    };
    let mut event = FieldEvent::new(
        format!(
            "ruview-lidar-{}-{}",
            packet.provenance.session_id, packet.provenance.sequence
        ),
        packet.provenance.timestamp_ns,
        sensor,
        tensor,
        observation,
        provenance,
    );
    signer
        .sign_event(&mut event)
        .map_err(|_| LidarBridgeError::InvalidTensor)?;
    Ok(event)
}

fn validate_metadata(packet: &LidarDepthPacket) -> Result<(), LidarBridgeError> {
    if packet.packet_type != "ruview.lidar.depth.v1"
        || packet.provenance.schema != packet.packet_type
    {
        return Err(LidarBridgeError::InvalidMetadata("schema"));
    }
    if packet.depth.encoding != "u16le-mm+u8-confidence" {
        return Err(LidarBridgeError::InvalidMetadata("encoding"));
    }
    if packet.provenance.sensor != "apple-arkit-scene-depth"
        || packet.provenance.source != "live"
        || packet.provenance.privacy_class != "geometry-only"
    {
        return Err(LidarBridgeError::InvalidMetadata("provenance"));
    }
    if packet.provenance.session_id.is_empty() || packet.provenance.coordinate_frame_id.is_empty() {
        return Err(LidarBridgeError::InvalidMetadata("identity"));
    }
    if packet.provenance.capture_time_ns == 0
        || packet.provenance.clock_model_id != "arkit-monotonic+session-wall-offset-v1"
        || packet.provenance.calibration_id.is_empty()
        || packet.provenance.tracking_state != "normal"
        || packet.provenance.evidence != "MEASURED"
    {
        return Err(LidarBridgeError::InvalidMetadata("capture context"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packet() -> LidarDepthPacket {
        let millimeters = [1000_u16, 1500, 0, 2000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        LidarDepthPacket {
            packet_type: "ruview.lidar.depth.v1".into(),
            intrinsics: LidarIntrinsics {
                fx: 100.0,
                fy: 100.0,
                cx: 1.0,
                cy: 1.0,
                image_width: 2,
                image_height: 2,
            },
            pose: LidarPose {
                matrix: vec![1.0; 16],
            },
            depth: LidarDepthPayload {
                width: 2,
                height: 2,
                encoding: "u16le-mm+u8-confidence".into(),
                millimeters_base64: STANDARD.encode(millimeters),
                confidence_base64: STANDARD.encode([2, 1, 0, 2]),
            },
            provenance: LidarProvenance {
                sensor: "apple-arkit-scene-depth".into(),
                session_id: "session-1".into(),
                coordinate_frame_id: "room-frame-1".into(),
                source: "live".into(),
                privacy_class: "geometry-only".into(),
                sequence: 1,
                timestamp_ns: 1_700_000_000_000_000_000,
                capture_time_ns: 42_000_000,
                clock_model_id: "arkit-monotonic+session-wall-offset-v1".into(),
                calibration_id: "coordinate-frame:room-frame-1".into(),
                tracking_state: "normal".into(),
                evidence: "MEASURED".into(),
                schema: "ruview.lidar.depth.v1".into(),
            },
        }
    }

    #[test]
    fn lidar_event_is_signed_p1_summary_without_raw_depth() {
        let event = lidar_packet_to_field_event(
            &packet(),
            &Signer::from_seed(b"lidar-rufield-test-seed-32-bytes"),
        )
        .expect("valid packet");
        assert_eq!(event.tensor.modality, Modality::LidarPhase);
        assert_eq!(event.observation.privacy_class, PrivacyClass::P1);
        assert!(rufield_provenance::is_fusable(&event));
        let json = serde_json::to_string(&event).unwrap();
        assert!(!json.contains("millimetersBase64"));
        assert!(!json.contains(&packet().depth.millimeters_base64));
        assert!(!event.observation.features.contains_key("presence"));
    }

    #[test]
    fn malformed_depth_shape_is_rejected() {
        let mut invalid = packet();
        invalid.depth.width = 200;
        assert!(matches!(
            lidar_packet_to_field_event(
                &invalid,
                &Signer::from_seed(b"lidar-rufield-test-seed-32-bytes")
            ),
            Err(LidarBridgeError::InvalidShape)
        ));
    }
}
