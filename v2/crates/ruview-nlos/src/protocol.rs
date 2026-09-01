//! Bounded transient and track wire contracts (ADR-328, ADR-331).

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::{MAX_BINS, MAX_TRACKS, MAX_ZONES, TRACK_SCHEMA_V1, TRANSIENT_SCHEMA_V1};

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_POSITION_M: f32 = 100.0;
const MAX_VELOCITY_MPS: f32 = 20.0;
const MAX_COVARIANCE_M2: f32 = 10.0;
const MAX_EXPIRY_WINDOW_MS: u64 = 5_000;

/// A finite Cartesian vector in metres or metres per second.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vec3 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
    /// Z component.
    pub z: f32,
}

impl Vec3 {
    /// Construct a vector.
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance.
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        let d = self.minus(other);
        (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
    }

    /// Component-wise addition.
    #[must_use]
    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Component-wise subtraction.
    #[must_use]
    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(self, value: f32) -> Self {
        Self::new(self.x * value, self.y * value, self.z * value)
    }

    pub(crate) fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Unit quaternion plus translation describing a sensor pose.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SensorPose {
    /// Sensor origin in world coordinates.
    pub translation_m: Vec3,
    /// Quaternion ordered x, y, z, w.
    pub quaternion_xyzw: [f32; 4],
}

impl Default for SensorPose {
    fn default() -> Self {
        Self {
            translation_m: Vec3::default(),
            quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl SensorPose {
    /// Transform a point from sensor-local to world coordinates.
    #[must_use]
    pub fn transform(self, point: Vec3) -> Vec3 {
        let [qx, qy, qz, qw] = self.quaternion_xyzw;
        let q = Vec3::new(qx, qy, qz);
        let t = cross(q, point).scale(2.0);
        point
            .plus(t.scale(qw))
            .plus(cross(q, t))
            .plus(self.translation_m)
    }

    /// Validate finite translation and an approximately unit quaternion.
    pub fn validate(self) -> Result<(), ContractError> {
        if !self.translation_m.finite()
            || self
                .quaternion_xyzw
                .iter()
                .any(|component| !component.is_finite())
        {
            return Err(ContractError::NonFinite("sensorPose"));
        }
        let norm = self
            .quaternion_xyzw
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        if !(0.99..=1.01).contains(&norm) {
            return Err(ContractError::InvalidPose);
        }
        Ok(())
    }
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Whether a frame came from a live sensor, captured replay, or generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameSource {
    /// Authenticated live sensor input.
    Live,
    /// Immutable captured data replay.
    Replay,
    /// Deterministic generated data; never hardware evidence.
    Synthetic,
}

/// Evidence ladder shared by all NLOS outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLevel {
    /// Generated input only.
    L0Synthetic,
    /// A measured raw sensor frame with unverified calibration.
    L1Measured,
    /// Measured input bound to a valid calibration.
    L2Calibrated,
    /// Independently corroborated modality lineages. Ground-truth maturity is
    /// evaluated separately and is never implied by this wire label.
    L3Corroborated,
}

/// Exact optical signal exposed by the adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransientKind {
    /// Raw photon arrival histogram.
    RawHistogram,
    /// VL53L8CH compact normalized histogram.
    CompactNormalizedHistogram,
    /// Conventional depth map; insufficient for NLOS inversion.
    DepthOnly,
    /// Replayed raw or normalized histogram.
    Replay,
}

/// Transport provenance for a transient or track.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Provenance {
    /// Authenticated or locally bound sensor identifier.
    pub sensor_id: String,
    /// Sensor model, for example `VL53L8CH`.
    pub sensor_model: String,
    /// Sensor firmware revision.
    pub firmware_version: String,
    /// Signal kind retained by the adapter.
    pub transient_kind: TransientKind,
    /// True only when zone timing bins remain available end to end.
    pub histogram_preserved: bool,
    /// `usb_serial`, `ruview_server`, or `replay`.
    pub transport: String,
}

impl Provenance {
    fn validate(&self, source: FrameSource) -> Result<(), ContractError> {
        validate_label("sensorId", &self.sensor_id)?;
        validate_label("sensorModel", &self.sensor_model)?;
        validate_label("firmwareVersion", &self.firmware_version)?;
        if !matches!(
            self.transport.as_str(),
            "usb_serial" | "ruview_server" | "replay"
        ) {
            return Err(ContractError::InvalidValue("provenance.transport"));
        }
        if source == FrameSource::Live
            && (!self.histogram_preserved
                || matches!(
                    self.transient_kind,
                    TransientKind::DepthOnly | TransientKind::Replay
                )
                || self.transport == "replay")
        {
            return Err(ContractError::DepthIsNotNlos);
        }
        if source == FrameSource::Synthetic
            && (self.transport != "replay" || self.transient_kind != TransientKind::Replay)
        {
            return Err(ContractError::EvidenceMismatch);
        }
        if source == FrameSource::Replay
            && (self.transport != "replay"
                || self.transient_kind != TransientKind::Replay
                || !self.histogram_preserved)
        {
            return Err(ContractError::EvidenceMismatch);
        }
        Ok(())
    }
}

/// One SPAD zone with its uncollapsed timing histogram.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TransientZone {
    /// Stable zone index within the frame.
    pub zone_id: u16,
    /// Relay-wall sample in sensor-local coordinates.
    pub wall_point_m: Vec3,
    /// Direct sensor-to-wall distance.
    pub distance_m: f32,
    /// Ambient counts reported by the sensor.
    pub ambient: u32,
    /// Photon counts by arrival-time bin.
    pub histogram: Vec<u16>,
}

/// Raw optical transient frame retained before NLOS processing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TransientFrame {
    /// Must equal [`TRANSIENT_SCHEMA_V1`].
    pub schema: String,
    /// Capture session identity.
    pub session_id: String,
    /// Monotonic sequence within the session.
    pub sequence: u64,
    /// UTC capture time.
    pub captured_at_unix_ms: u64,
    /// Monotonic device time, used for aperture ordering.
    pub monotonic_ns: u64,
    /// Live, replay, or synthetic source.
    pub source: FrameSource,
    /// Evidence level attached at the ingest boundary.
    pub evidence_level: EvidenceLevel,
    /// Timing-bin width in picoseconds.
    pub bin_width_ps: f32,
    /// First physical sensor bin retained by the firmware.
    pub start_bin: u16,
    /// Sensor pose for motion-induced aperture accumulation.
    pub sensor_pose: SensorPose,
    /// Calibration digest, or 64 zeroes before calibration.
    pub calibration_hash: String,
    /// Sensor and transport provenance.
    pub provenance: Provenance,
    /// Zone histograms.
    pub zones: Vec<TransientZone>,
}

impl TransientFrame {
    /// Validate all untrusted dimensions, numbers, labels, and evidence rules.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema != TRANSIENT_SCHEMA_V1 {
            return Err(ContractError::WrongSchema);
        }
        validate_label("sessionId", &self.session_id)?;
        validate_sequence(self.sequence)?;
        validate_timestamp(self.captured_at_unix_ms)?;
        validate_hash(&self.calibration_hash)?;
        self.sensor_pose.validate()?;
        self.provenance.validate(self.source)?;
        if !self.bin_width_ps.is_finite() || !(1.0..=10_000.0).contains(&self.bin_width_ps) {
            return Err(ContractError::InvalidValue("binWidthPs"));
        }
        if self.zones.is_empty() || self.zones.len() > MAX_ZONES {
            return Err(ContractError::Bound("zones"));
        }
        let bins = self.zones[0].histogram.len();
        if !(8..=MAX_BINS).contains(&bins) {
            return Err(ContractError::Bound("histogram"));
        }
        let mut seen = [false; MAX_ZONES];
        for (zone_index, zone) in self.zones.iter().enumerate() {
            let idx = usize::from(zone.zone_id);
            if idx >= self.zones.len() || seen[idx] || idx != zone_index {
                return Err(ContractError::InvalidValue("zoneId"));
            }
            seen[idx] = true;
            if zone.histogram.len() != bins {
                return Err(ContractError::InconsistentBins);
            }
            if !zone.wall_point_m.finite()
                || !zone.distance_m.is_finite()
                || !(0.01..=10.0).contains(&zone.distance_m)
            {
                return Err(ContractError::InvalidValue("zone geometry"));
            }
        }
        match (self.source, self.evidence_level) {
            (FrameSource::Synthetic, EvidenceLevel::L0Synthetic) => {}
            (FrameSource::Synthetic, _) => return Err(ContractError::EvidenceMismatch),
            (FrameSource::Live, EvidenceLevel::L0Synthetic) => {
                return Err(ContractError::EvidenceMismatch)
            }
            _ => {}
        }
        if self.evidence_level == EvidenceLevel::L3Corroborated {
            // v1 has no field that can retain both modality lineages.
            return Err(ContractError::EvidenceMismatch);
        }
        if self.source == FrameSource::Synthetic && self.calibration_hash != "0".repeat(64) {
            return Err(ContractError::EvidenceMismatch);
        }
        if self.evidence_level >= EvidenceLevel::L2Calibrated
            && self.calibration_hash == "0".repeat(64)
        {
            return Err(ContractError::EvidenceMismatch);
        }
        Ok(())
    }
}

/// State of one hidden target hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackState {
    /// Posterior passed all quality gates.
    Tracking,
    /// Some evidence remains, but quality is below the normal threshold.
    Degraded,
    /// No reliable estimate is available.
    Unknown,
}

/// Relative optical and RF contribution to one posterior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModalityContributions {
    /// Optical transient contribution.
    pub lidar: f32,
    /// CSI prior contribution.
    pub csi: f32,
}

/// One hidden target posterior.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NlosTrack {
    /// Privacy-preserving session-local identifier.
    pub track_id: String,
    /// Quality-gated state.
    pub state: TrackState,
    /// Posterior mean in metres.
    pub position_m: Vec3,
    /// Estimated velocity.
    pub velocity_mps: Vec3,
    /// Diagonal covariance in square metres.
    pub covariance_diagonal_m2: Vec3,
    /// Bounded posterior confidence.
    pub confidence: f32,
    /// Shannon entropy of normalized particle weights.
    pub posterior_entropy: f32,
    /// Bounded optical/RF signal quality.
    pub signal_quality: f32,
    /// Relative modality contributions.
    pub modality_contributions: ModalityContributions,
}

/// Public track envelope shared with Swift and TypeScript clients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrackEnvelope {
    /// Must equal [`TRACK_SCHEMA_V1`].
    pub schema: String,
    /// Capture session identity.
    pub session_id: String,
    /// Monotonic sequence within the session.
    pub sequence: u64,
    /// UTC capture time.
    pub captured_at_unix_ms: u64,
    /// Hard expiry; clients fail closed after this time.
    pub expires_at_unix_ms: u64,
    /// Source label.
    pub source: FrameSource,
    /// Evidence level.
    pub evidence_level: EvidenceLevel,
    /// Reproducible algorithm revision.
    pub algorithm_version: String,
    /// Calibration SHA-256 digest.
    pub calibration_hash: String,
    /// Sensor and transport provenance.
    pub provenance: Provenance,
    /// Bounded hidden target hypotheses.
    pub tracks: Vec<NlosTrack>,
}

impl TrackEnvelope {
    /// Validate a track frame received from an untrusted server or replay.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema != TRACK_SCHEMA_V1 {
            return Err(ContractError::WrongSchema);
        }
        validate_label("sessionId", &self.session_id)?;
        validate_label("algorithmVersion", &self.algorithm_version)?;
        validate_sequence(self.sequence)?;
        validate_timestamp(self.captured_at_unix_ms)?;
        validate_timestamp(self.expires_at_unix_ms)?;
        validate_hash(&self.calibration_hash)?;
        self.provenance.validate(self.source)?;
        if self.expires_at_unix_ms <= self.captured_at_unix_ms
            || self.expires_at_unix_ms - self.captured_at_unix_ms > MAX_EXPIRY_WINDOW_MS
        {
            return Err(ContractError::InvalidExpiry);
        }
        if self.tracks.len() > MAX_TRACKS {
            return Err(ContractError::Bound("tracks"));
        }
        if self.source == FrameSource::Synthetic
            && self.evidence_level != EvidenceLevel::L0Synthetic
        {
            return Err(ContractError::EvidenceMismatch);
        }
        if self.source == FrameSource::Synthetic && self.calibration_hash != "0".repeat(64) {
            return Err(ContractError::EvidenceMismatch);
        }
        if self.evidence_level >= EvidenceLevel::L2Calibrated
            && self.calibration_hash == "0".repeat(64)
        {
            return Err(ContractError::EvidenceMismatch);
        }
        if self.source == FrameSource::Live && self.evidence_level == EvidenceLevel::L0Synthetic {
            return Err(ContractError::EvidenceMismatch);
        }
        if self.evidence_level == EvidenceLevel::L3Corroborated {
            return Err(ContractError::EvidenceMismatch);
        }
        let mut track_ids = BTreeSet::new();
        for track in &self.tracks {
            validate_label("trackId", &track.track_id)?;
            if !track_ids.insert(track.track_id.as_str()) {
                return Err(ContractError::InvalidValue("duplicate trackId"));
            }
            validate_bounded_vec(track.position_m, MAX_POSITION_M, "positionM")?;
            validate_bounded_vec(track.velocity_mps, MAX_VELOCITY_MPS, "velocityMps")?;
            validate_nonnegative_vec(
                track.covariance_diagonal_m2,
                MAX_COVARIANCE_M2,
                "covarianceDiagonalM2",
            )?;
            validate_unit(track.confidence, "confidence")?;
            validate_unit(track.signal_quality, "signalQuality")?;
            validate_unit(track.modality_contributions.lidar, "lidar contribution")?;
            validate_unit(track.modality_contributions.csi, "csi contribution")?;
            let contribution_sum =
                track.modality_contributions.lidar + track.modality_contributions.csi;
            if !(0.999..=1.001).contains(&contribution_sum) {
                return Err(ContractError::InvalidValue("modality contributions"));
            }
            if !track.posterior_entropy.is_finite() || track.posterior_entropy < 0.0 {
                return Err(ContractError::NonFinite("posteriorEntropy"));
            }
        }
        Ok(())
    }
}

/// Contract validation failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ContractError {
    /// Schema identifier does not match this implementation.
    #[error("unsupported NLOS schema")]
    WrongSchema,
    /// A collection exceeded its explicit bound.
    #[error("{0} exceeds its allowed bound")]
    Bound(&'static str),
    /// A number was NaN or infinite.
    #[error("{0} must be finite")]
    NonFinite(&'static str),
    /// A field value is invalid.
    #[error("invalid {0}")]
    InvalidValue(&'static str),
    /// A label is empty, too long, or unsafe.
    #[error("invalid bounded label {0}")]
    InvalidLabel(&'static str),
    /// Zone histograms do not share one bin count.
    #[error("all zones must have the same histogram bin count")]
    InconsistentBins,
    /// Ordinary depth is not sufficient live NLOS evidence.
    #[error("depth-only frames cannot be presented as live transient NLOS")]
    DepthIsNotNlos,
    /// Source and evidence labels disagree.
    #[error("source and evidence level disagree")]
    EvidenceMismatch,
    /// A digest was not lowercase SHA-256 hex.
    #[error("calibrationHash must be 64 lowercase hexadecimal characters")]
    InvalidHash,
    /// Sequence is not interoperable with JavaScript clients.
    #[error("sequence exceeds the JavaScript safe integer range")]
    UnsafeSequence,
    /// Timestamp is not interoperable with JavaScript clients.
    #[error("timestamp exceeds the JavaScript safe integer range")]
    UnsafeTimestamp,
    /// Expiry precedes capture or exceeds the five second freshness bound.
    #[error("invalid frame expiry")]
    InvalidExpiry,
    /// Quaternion is not approximately unit length.
    #[error("sensor pose quaternion must be normalized")]
    InvalidPose,
}

fn validate_label(field: &'static str, value: &str) -> Result<(), ContractError> {
    if value.is_empty()
        || value.len() > 64
        || value
            .bytes()
            .any(|b| !(b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':')))
    {
        return Err(ContractError::InvalidLabel(field));
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), ContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(ContractError::InvalidHash);
    }
    Ok(())
}

fn validate_sequence(value: u64) -> Result<(), ContractError> {
    if value > MAX_SAFE_INTEGER {
        return Err(ContractError::UnsafeSequence);
    }
    Ok(())
}

fn validate_timestamp(value: u64) -> Result<(), ContractError> {
    if value > MAX_SAFE_INTEGER {
        return Err(ContractError::UnsafeTimestamp);
    }
    Ok(())
}

fn validate_unit(value: f32, field: &'static str) -> Result<(), ContractError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ContractError::InvalidValue(field));
    }
    Ok(())
}

fn validate_bounded_vec(
    value: Vec3,
    max_abs: f32,
    field: &'static str,
) -> Result<(), ContractError> {
    if !value.finite()
        || [value.x, value.y, value.z]
            .iter()
            .any(|v| v.abs() > max_abs)
    {
        return Err(ContractError::InvalidValue(field));
    }
    Ok(())
}

fn validate_nonnegative_vec(
    value: Vec3,
    max: f32,
    field: &'static str,
) -> Result<(), ContractError> {
    if !value.finite()
        || [value.x, value.y, value.z]
            .iter()
            .any(|v| *v < 0.0 || *v > max)
    {
        return Err(ContractError::InvalidValue(field));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provenance() -> Provenance {
        Provenance {
            sensor_id: "st-01".into(),
            sensor_model: "VL53L8CH".into(),
            firmware_version: "upstream-main".into(),
            transient_kind: TransientKind::CompactNormalizedHistogram,
            histogram_preserved: true,
            transport: "usb_serial".into(),
        }
    }

    #[test]
    fn depth_only_live_is_rejected() {
        let mut p = provenance();
        p.transient_kind = TransientKind::DepthOnly;
        p.histogram_preserved = false;
        assert_eq!(
            p.validate(FrameSource::Live),
            Err(ContractError::DepthIsNotNlos)
        );
        let mut replay_transport = provenance();
        replay_transport.transport = "replay".into();
        assert_eq!(
            replay_transport.validate(FrameSource::Live),
            Err(ContractError::DepthIsNotNlos)
        );
    }

    #[test]
    fn pose_transform_applies_rotation_and_translation() {
        let half = (0.5_f32).sqrt();
        let pose = SensorPose {
            translation_m: Vec3::new(1.0, 0.0, 0.0),
            quaternion_xyzw: [0.0, 0.0, half, half],
        };
        pose.validate().unwrap();
        let out = pose.transform(Vec3::new(1.0, 0.0, 0.0));
        assert!((out.x - 1.0).abs() < 1e-5);
        assert!((out.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn track_contract_rejects_duplicate_ids_and_zero_lifetime() {
        let fixture = include_str!("../tests/fixtures/track_synthetic.json");
        let mut envelope: TrackEnvelope = serde_json::from_str(fixture).unwrap();
        envelope.tracks.push(envelope.tracks[0].clone());
        assert_eq!(
            envelope.validate(),
            Err(ContractError::InvalidValue("duplicate trackId"))
        );

        envelope.tracks.truncate(1);
        envelope.expires_at_unix_ms = envelope.captured_at_unix_ms;
        assert_eq!(envelope.validate(), Err(ContractError::InvalidExpiry));
    }

    #[test]
    fn v1_rejects_l3_without_dual_modality_lineage() {
        let fixture = include_str!("../tests/fixtures/track_synthetic.json");
        let mut frame: TrackEnvelope = serde_json::from_str(fixture).unwrap();
        frame.source = FrameSource::Replay;
        frame.evidence_level = EvidenceLevel::L3Corroborated;
        frame.calibration_hash = "a".repeat(64);
        assert_eq!(frame.validate(), Err(ContractError::EvidenceMismatch));
    }

    #[test]
    fn transient_v1_rejects_l3_at_the_ingest_boundary() {
        let mut scene = crate::simulator::SyntheticScene::default();
        let mut frame = scene.frame(None, 0.0, 1);
        frame.source = FrameSource::Replay;
        frame.evidence_level = EvidenceLevel::L3Corroborated;
        frame.calibration_hash = "a".repeat(64);
        assert_eq!(frame.validate(), Err(ContractError::EvidenceMismatch));
    }

    #[test]
    fn captured_replay_must_preserve_replayed_histogram_provenance() {
        let mut p = provenance();
        p.transport = "replay".into();
        p.transient_kind = TransientKind::Replay;
        assert!(p.validate(FrameSource::Replay).is_ok());
        p.histogram_preserved = false;
        assert_eq!(
            p.validate(FrameSource::Replay),
            Err(ContractError::EvidenceMismatch)
        );
    }
}
