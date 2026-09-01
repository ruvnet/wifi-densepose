//! Empty-room calibration, direct-return removal, and light-cone resampling.

use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::protocol::{EvidenceLevel, FrameSource, Provenance, SensorPose, TransientFrame, Vec3};

/// Preprocessing controls matching the public consumer-NLOS workflow.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationConfig {
    /// Bins masked on either side of the direct wall return.
    pub pulse_half_width: usize,
    /// Earliest extra-path bins discarded after direct-return alignment.
    pub zero_first_bins: usize,
    /// Maximum empty-room frames accepted into one calibration.
    pub max_background_frames: usize,
    /// Maximum sensor translation from the empty-room capture centroid.
    pub max_sensor_translation_m: f32,
    /// Maximum per-zone relay point displacement in the world frame.
    pub max_wall_point_drift_m: f32,
    /// Maximum per-zone reported range drift from calibration.
    pub max_wall_distance_drift_m: f32,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            pulse_half_width: 7,
            zero_first_bins: 15,
            max_background_frames: 256,
            max_sensor_translation_m: 0.5,
            max_wall_point_drift_m: 0.5,
            max_wall_distance_drift_m: 0.25,
        }
    }
}

/// Immutable empty-room calibration bound by a deterministic SHA-256 digest.
#[derive(Clone, Debug)]
pub struct Calibration {
    /// Configuration used to build the calibration.
    config: CalibrationConfig,
    /// Session in which the empty room was captured.
    session_id: String,
    /// Homogeneous source classification of the calibration capture.
    source: FrameSource,
    /// Evidence ceiling of the calibration inputs.
    input_evidence_level: EvidenceLevel,
    /// Full sensor/signal/transport provenance bound into the calibration.
    provenance: Provenance,
    /// Number of zones.
    zone_count: usize,
    /// Number of native timing bins.
    bin_count: usize,
    /// Timing width in picoseconds.
    bin_width_ps: f32,
    /// First firmware bin.
    start_bin: u16,
    /// Sensor identity bound into the calibration digest.
    sensor_id: String,
    /// Sensor model bound into the calibration digest.
    sensor_model: String,
    /// Firmware revision bound into the calibration digest.
    firmware_version: String,
    /// Number of background samples committed to the digest.
    sample_count: usize,
    /// Mean sensor translation during empty-room capture.
    reference_sensor_translation_m: Vec3,
    /// Average relay-wall points in sensor-local coordinates.
    wall_points_m: Vec<Vec3>,
    /// Average relay-wall points transformed into the world coordinate frame.
    wall_points_world_m: Vec<Vec3>,
    /// Average per-zone range reported during calibration.
    wall_distances_m: Vec<f32>,
    /// Peak direct-return index for each zone.
    direct_peak_bins: Vec<usize>,
    /// Empty-room mean counts, row-major by zone and bin.
    background: Vec<f32>,
    /// SHA-256 digest over all fields above.
    hash: String,
}

impl Calibration {
    /// Build a deterministic empty-room calibration from two or more frames.
    pub fn from_background(
        frames: &[TransientFrame],
        config: CalibrationConfig,
    ) -> Result<Self, CalibrationError> {
        if frames.len() < 2 || frames.len() > config.max_background_frames {
            return Err(CalibrationError::FrameCount);
        }
        if !config.max_sensor_translation_m.is_finite()
            || !(0.01..=5.0).contains(&config.max_sensor_translation_m)
            || !config.max_wall_point_drift_m.is_finite()
            || !(0.01..=5.0).contains(&config.max_wall_point_drift_m)
            || !config.max_wall_distance_drift_m.is_finite()
            || !(0.01..=5.0).contains(&config.max_wall_distance_drift_m)
        {
            return Err(CalibrationError::InvalidConfig);
        }
        for frame in frames {
            frame.validate()?;
        }
        let first = &frames[0];
        let zone_count = first.zones.len();
        let bin_count = first.zones[0].histogram.len();
        if config.pulse_half_width >= bin_count || config.zero_first_bins >= bin_count {
            return Err(CalibrationError::InvalidConfig);
        }
        if frames.windows(2).any(|pair| {
            pair[1].sequence <= pair[0].sequence
                || pair[1].monotonic_ns <= pair[0].monotonic_ns
                || pair[1].captured_at_unix_ms < pair[0].captured_at_unix_ms
        }) {
            return Err(CalibrationError::OutOfOrderFrames);
        }
        if frames.iter().any(|frame| {
            frame.session_id != first.session_id
                || frame.source != first.source
                || frame.evidence_level != first.evidence_level
                || frame.provenance != first.provenance
                || frame.calibration_hash != first.calibration_hash
                || frame.zones.len() != zone_count
                || frame.zones[0].histogram.len() != bin_count
                || frame.bin_width_ps != first.bin_width_ps
                || frame.start_bin != first.start_bin
                || frame.provenance.sensor_id != first.provenance.sensor_id
                || frame.provenance.sensor_model != first.provenance.sensor_model
                || frame.provenance.firmware_version != first.provenance.firmware_version
                || frame
                    .zones
                    .iter()
                    .any(|zone| zone.histogram.len() != bin_count)
        }) {
            return Err(CalibrationError::InconsistentFrames);
        }

        let mut background = vec![0.0_f32; zone_count * bin_count];
        let mut wall_points_m = vec![Vec3::default(); zone_count];
        let mut wall_points_world_m = vec![Vec3::default(); zone_count];
        let mut wall_distances_m = vec![0.0_f32; zone_count];
        let mut reference_sensor_translation_m = Vec3::default();
        for frame in frames {
            reference_sensor_translation_m =
                reference_sensor_translation_m.plus(frame.sensor_pose.translation_m);
            for (zone_index, zone) in frame.zones.iter().enumerate() {
                wall_points_m[zone_index] = wall_points_m[zone_index].plus(zone.wall_point_m);
                wall_points_world_m[zone_index] = wall_points_world_m[zone_index]
                    .plus(frame.sensor_pose.transform(zone.wall_point_m));
                wall_distances_m[zone_index] += zone.distance_m;
                for (bin, count) in zone.histogram.iter().enumerate() {
                    background[zone_index * bin_count + bin] += f32::from(*count);
                }
            }
        }
        let inv = 1.0 / frames.len() as f32;
        for value in &mut background {
            *value *= inv;
        }
        for point in &mut wall_points_m {
            *point = point.scale(inv);
        }
        for point in &mut wall_points_world_m {
            *point = point.scale(inv);
        }
        for distance in &mut wall_distances_m {
            *distance *= inv;
        }
        reference_sensor_translation_m = reference_sensor_translation_m.scale(inv);
        let direct_peak_bins = (0..zone_count)
            .map(|zone| {
                background[zone * bin_count..(zone + 1) * bin_count]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map_or(0, |(index, _)| index)
            })
            .collect::<Vec<_>>();

        let mut calibration = Self {
            config,
            session_id: first.session_id.clone(),
            source: first.source,
            input_evidence_level: first.evidence_level,
            provenance: first.provenance.clone(),
            zone_count,
            bin_count,
            bin_width_ps: first.bin_width_ps,
            start_bin: first.start_bin,
            sensor_id: first.provenance.sensor_id.clone(),
            sensor_model: first.provenance.sensor_model.clone(),
            firmware_version: first.provenance.firmware_version.clone(),
            sample_count: frames.len(),
            reference_sensor_translation_m,
            wall_points_m,
            wall_points_world_m,
            wall_distances_m,
            direct_peak_bins,
            background,
            hash: String::new(),
        };
        calibration.hash = calibration.compute_hash();
        Ok(calibration)
    }

    /// Stable calibration digest used by measured track envelopes.
    #[must_use]
    pub fn hash(&self) -> &str {
        &self.hash
    }

    /// Capture session bound into this calibration.
    #[must_use]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Number of calibrated relay zones.
    #[must_use]
    pub fn zone_count(&self) -> usize {
        self.zone_count
    }

    /// Background-subtract, mask the direct peak, align extra path time, and
    /// resample from time to uniform squared-distance (light-cone) bins.
    pub fn preprocess(
        &self,
        frame: &TransientFrame,
    ) -> Result<PreprocessedFrame, CalibrationError> {
        if self.compute_hash() != self.hash {
            return Err(CalibrationError::IntegrityMismatch);
        }
        frame.validate()?;
        if frame.session_id != self.session_id
            || frame.source != self.source
            || frame.provenance != self.provenance
            || frame.zones.len() != self.zone_count
            || frame.zones[0].histogram.len() != self.bin_count
            || frame.bin_width_ps != self.bin_width_ps
            || frame.start_bin != self.start_bin
            || frame.provenance.sensor_id != self.sensor_id
            || frame.provenance.sensor_model != self.sensor_model
            || frame.provenance.firmware_version != self.firmware_version
        {
            return Err(CalibrationError::InconsistentFrames);
        }
        if frame.source != FrameSource::Synthetic
            && frame.calibration_hash != self.hash
            && frame.calibration_hash != "0".repeat(64)
        {
            return Err(CalibrationError::WrongCalibration);
        }

        if frame
            .sensor_pose
            .translation_m
            .distance(self.reference_sensor_translation_m)
            > self.config.max_sensor_translation_m
            || frame.zones.iter().enumerate().any(|(zone_index, zone)| {
                frame
                    .sensor_pose
                    .transform(zone.wall_point_m)
                    .distance(self.wall_points_world_m[zone_index])
                    > self.config.max_wall_point_drift_m
                    || (zone.distance_m - self.wall_distances_m[zone_index]).abs()
                        > self.config.max_wall_distance_drift_m
            })
        {
            return Err(CalibrationError::GeometryDrift);
        }

        let mut light_cone_histograms = vec![0.0_f32; self.zone_count * self.bin_count];
        let mut foreground_sum = 0.0_f32;
        let mut noise_floor = 0.0_f32;
        let denominator = (self.bin_count - 1).max(1);
        for (zone_index, zone) in frame.zones.iter().enumerate() {
            let peak = self.direct_peak_bins[zone_index];
            let direct_end = peak.saturating_add(self.config.pulse_half_width);
            for native_bin in 0..self.bin_count {
                let baseline = self.background[zone_index * self.bin_count + native_bin];
                let residual = (f32::from(zone.histogram[native_bin]) - baseline).max(0.0);
                noise_floor += baseline.sqrt().max(1.0);
                if native_bin <= direct_end {
                    continue;
                }
                let extra_bin = native_bin - peak;
                if extra_bin < self.config.zero_first_bins {
                    continue;
                }
                let v_bin = ((extra_bin * extra_bin) / denominator).min(self.bin_count - 1);
                light_cone_histograms[zone_index * self.bin_count + v_bin] += residual;
                foreground_sum += residual;
            }
        }
        let signal_quality = foreground_sum / (foreground_sum + noise_floor.max(1.0));
        let wall_points_world_m = frame
            .zones
            .iter()
            .map(|zone| frame.sensor_pose.transform(zone.wall_point_m))
            .collect();
        Ok(PreprocessedFrame {
            sequence: frame.sequence,
            captured_at_unix_ms: frame.captured_at_unix_ms,
            monotonic_ns: frame.monotonic_ns,
            source: frame.source,
            evidence_level: frame.evidence_level,
            sensor_pose: frame.sensor_pose,
            wall_points_world_m,
            light_cone_histograms,
            zone_count: self.zone_count,
            bin_count: self.bin_count,
            bin_width_ps: self.bin_width_ps,
            signal_quality: signal_quality.clamp(0.0, 1.0),
        })
    }

    fn compute_hash(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"ruview.nlos.calibration.v1\0");
        update_string(&mut digest, &self.session_id);
        digest.update([frame_source_code(self.source)]);
        digest.update([evidence_level_code(self.input_evidence_level)]);
        update_string(&mut digest, &self.provenance.sensor_id);
        update_string(&mut digest, &self.provenance.sensor_model);
        update_string(&mut digest, &self.provenance.firmware_version);
        digest.update([transient_kind_code(self.provenance.transient_kind)]);
        digest.update([u8::from(self.provenance.histogram_preserved)]);
        update_string(&mut digest, &self.provenance.transport);
        digest.update((self.zone_count as u64).to_le_bytes());
        digest.update((self.bin_count as u64).to_le_bytes());
        digest.update(self.bin_width_ps.to_le_bytes());
        digest.update(self.start_bin.to_le_bytes());
        digest.update((self.config.pulse_half_width as u64).to_le_bytes());
        digest.update((self.config.zero_first_bins as u64).to_le_bytes());
        digest.update((self.config.max_background_frames as u64).to_le_bytes());
        digest.update(self.config.max_sensor_translation_m.to_le_bytes());
        digest.update(self.config.max_wall_point_drift_m.to_le_bytes());
        digest.update(self.config.max_wall_distance_drift_m.to_le_bytes());
        digest.update((self.sample_count as u64).to_le_bytes());
        digest.update(self.reference_sensor_translation_m.x.to_le_bytes());
        digest.update(self.reference_sensor_translation_m.y.to_le_bytes());
        digest.update(self.reference_sensor_translation_m.z.to_le_bytes());
        for point in &self.wall_points_m {
            digest.update(point.x.to_le_bytes());
            digest.update(point.y.to_le_bytes());
            digest.update(point.z.to_le_bytes());
        }
        for point in &self.wall_points_world_m {
            digest.update(point.x.to_le_bytes());
            digest.update(point.y.to_le_bytes());
            digest.update(point.z.to_le_bytes());
        }
        for distance in &self.wall_distances_m {
            digest.update(distance.to_le_bytes());
        }
        for peak in &self.direct_peak_bins {
            digest.update((*peak as u64).to_le_bytes());
        }
        for value in &self.background {
            digest.update(value.to_le_bytes());
        }
        format!("{:x}", digest.finalize())
    }
}

fn update_string(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u32).to_le_bytes());
    digest.update(value.as_bytes());
}

fn frame_source_code(value: FrameSource) -> u8 {
    match value {
        FrameSource::Live => 1,
        FrameSource::Replay => 2,
        FrameSource::Synthetic => 3,
    }
}

fn evidence_level_code(value: EvidenceLevel) -> u8 {
    match value {
        EvidenceLevel::L0Synthetic => 0,
        EvidenceLevel::L1Measured => 1,
        EvidenceLevel::L2Calibrated => 2,
        EvidenceLevel::L3Corroborated => 3,
    }
}

fn transient_kind_code(value: crate::protocol::TransientKind) -> u8 {
    match value {
        crate::protocol::TransientKind::RawHistogram => 1,
        crate::protocol::TransientKind::CompactNormalizedHistogram => 2,
        crate::protocol::TransientKind::DepthOnly => 3,
        crate::protocol::TransientKind::Replay => 4,
    }
}

/// A frame ready for motion-induced aperture likelihood evaluation.
#[derive(Clone, Debug)]
pub struct PreprocessedFrame {
    /// Input sequence.
    pub sequence: u64,
    /// UTC capture time.
    pub captured_at_unix_ms: u64,
    /// Monotonic capture time used for motion compensation.
    pub monotonic_ns: u64,
    /// Source label.
    pub source: FrameSource,
    /// Evidence level.
    pub evidence_level: crate::protocol::EvidenceLevel,
    /// Sensor pose used for this aperture sample.
    pub sensor_pose: SensorPose,
    /// Relay wall samples transformed into world coordinates.
    pub wall_points_world_m: Vec<Vec3>,
    /// Uniform squared-distance histogram, row-major by zone and bin.
    pub light_cone_histograms: Vec<f32>,
    /// Number of zones.
    pub zone_count: usize,
    /// Number of squared-distance bins.
    pub bin_count: usize,
    /// Native timing width in picoseconds.
    pub bin_width_ps: f32,
    /// Foreground-to-noise quality estimate.
    pub signal_quality: f32,
}

/// Calibration or preprocessing failure.
#[derive(Debug, Error)]
pub enum CalibrationError {
    /// Too few or too many background frames.
    #[error("background calibration requires 2..=max_background_frames frames")]
    FrameCount,
    /// Invalid pulse or early-bin mask configuration.
    #[error("invalid calibration configuration")]
    InvalidConfig,
    /// Frames differ in session, shape, timing, or calibration.
    #[error("inconsistent transient frames")]
    InconsistentFrames,
    /// Calibration frames repeated or moved backwards.
    #[error("calibration frames must have strictly increasing sequence and monotonic time")]
    OutOfOrderFrames,
    /// A measured frame names a different calibration digest.
    #[error("transient frame is bound to a different calibration")]
    WrongCalibration,
    /// The supposedly immutable calibration payload no longer matches its digest.
    #[error("calibration integrity digest mismatch")]
    IntegrityMismatch,
    /// Sensor pose or relay geometry moved outside the calibrated operating volume.
    #[error("transient geometry is outside the calibrated operating volume")]
    GeometryDrift,
    /// Raw frame contract violation.
    #[error(transparent)]
    Contract(#[from] crate::protocol::ContractError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator::SyntheticScene;

    #[test]
    fn calibration_is_deterministic_and_foreground_survives() {
        let mut scene = SyntheticScene::default();
        let background = scene.background_frames(10);
        let a = Calibration::from_background(&background, CalibrationConfig::default()).unwrap();
        let b = Calibration::from_background(&background, CalibrationConfig::default()).unwrap();
        assert_eq!(a.hash, b.hash);
        let frame = scene.frame(Some(Vec3::new(0.2, 0.1, 1.0)), 1.0, 20);
        let processed = a.preprocess(&frame).unwrap();
        assert!(processed.signal_quality > 0.0);
        assert_eq!(
            processed.light_cone_histograms.len(),
            a.zone_count * a.bin_count
        );
    }

    #[test]
    fn calibration_rejects_duplicate_and_mixed_provenance_frames() {
        let mut scene = SyntheticScene::default();
        let background = scene.background_frames(3);
        let duplicate = vec![background[0].clone(), background[0].clone()];
        assert!(matches!(
            Calibration::from_background(&duplicate, CalibrationConfig::default()),
            Err(CalibrationError::OutOfOrderFrames)
        ));

        let mut mixed = background[..2].to_vec();
        mixed[1].provenance.sensor_id = "another-sensor".into();
        assert!(matches!(
            Calibration::from_background(&mixed, CalibrationConfig::default()),
            Err(CalibrationError::InconsistentFrames)
        ));
    }

    #[test]
    fn calibration_hash_uses_unambiguous_length_prefixed_identity() {
        let mut scene = SyntheticScene::default();
        let mut left = scene.background_frames(2);
        for frame in &mut left {
            frame.provenance.sensor_id = "ab".into();
            frame.provenance.sensor_model = "c".into();
        }
        let mut right = left.clone();
        for frame in &mut right {
            frame.provenance.sensor_id = "a".into();
            frame.provenance.sensor_model = "bc".into();
        }
        let left = Calibration::from_background(&left, CalibrationConfig::default()).unwrap();
        let right = Calibration::from_background(&right, CalibrationConfig::default()).unwrap();
        assert_ne!(left.hash, right.hash);
    }

    #[test]
    fn preprocessing_rejects_tampered_calibration_and_geometry_drift() {
        let mut scene = SyntheticScene::default();
        let background = scene.background_frames(3);
        let calibration =
            Calibration::from_background(&background, CalibrationConfig::default()).unwrap();
        let frame = scene.frame(Some(Vec3::new(0.0, 0.0, 1.0)), 1.0, 10);

        let mut tampered = calibration.clone();
        tampered.background[0] += 1.0;
        assert!(matches!(
            tampered.preprocess(&frame),
            Err(CalibrationError::IntegrityMismatch)
        ));

        let mut drifted = frame;
        drifted.sensor_pose.translation_m.x += 1.0;
        assert!(matches!(
            calibration.preprocess(&drifted),
            Err(CalibrationError::GeometryDrift)
        ));
    }
}
