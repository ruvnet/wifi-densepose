//! Motion-induced aperture particle tracking with optional CSI prior.

use std::collections::VecDeque;

use thiserror::Error;

use crate::calibration::{Calibration, CalibrationError, PreprocessedFrame};
use crate::fusion::{CsiSpatialPrior, FusionError, FusionScope};
use crate::protocol::{
    FrameSource, ModalityContributions, NlosTrack, TrackEnvelope, TrackState, TransientFrame, Vec3,
};
use crate::TRACK_SCHEMA_V1;

const SPEED_OF_LIGHT_MPS: f32 = 299_792_458.0;
const ALGORITHM_VERSION: &str = "motion-aperture-canonical-v1";
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// A known rigid object represented as weighted points around its origin.
#[derive(Clone, Debug)]
pub struct CanonicalObject {
    /// Canonical local-space points.
    pub points_m: Vec<Vec3>,
    /// Relative non-negative return weight for each point.
    pub weights: Vec<f32>,
}

impl CanonicalObject {
    /// A single point reflector used for the first reproduction milestone.
    #[must_use]
    pub fn point() -> Self {
        Self {
            points_m: vec![Vec3::default()],
            weights: vec![1.0],
        }
    }

    fn validate(&self) -> Result<(), TrackerError> {
        if self.points_m.is_empty()
            || self.points_m.len() > 4_096
            || self.points_m.len() != self.weights.len()
            || self.points_m.iter().any(|point| !point.finite())
            || self
                .weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            || self.weights.iter().all(|weight| *weight == 0.0)
        {
            return Err(TrackerError::InvalidConfig);
        }
        Ok(())
    }
}

/// Search volume and quality gates.
#[derive(Clone, Debug)]
pub struct TrackerConfig {
    /// Particle count; 1,000 matches the public reproduction default.
    pub particle_count: usize,
    /// Inclusive search minimum.
    pub search_min_m: Vec3,
    /// Inclusive search maximum.
    pub search_max_m: Vec3,
    /// Gaussian random-walk standard deviation per frame.
    pub motion_std_m: f32,
    /// Likelihood sharpening exponent.
    pub eta: f32,
    /// Maximum aperture samples retained.
    pub aperture_frames: usize,
    /// Minimum combined signal quality for `tracking`.
    pub min_signal_quality: f32,
    /// Minimum posterior confidence for `tracking`.
    pub min_confidence: f32,
    /// Maximum CSI-to-optical time difference.
    pub max_csi_age_ms: u64,
    /// Output freshness window.
    pub output_ttl_ms: u64,
    /// Deterministic particle RNG seed.
    pub seed: u64,
    /// Exact policy/coordinate scope for synthetic fusion tests. Measured
    /// fusion remains blocked by the v1 lineage boundary.
    pub fusion_scope: Option<FusionScope>,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            particle_count: 1_000,
            search_min_m: Vec3::new(-1.0, -0.8, 0.1),
            search_max_m: Vec3::new(1.0, 0.8, 1.8),
            motion_std_m: 0.05,
            eta: 3.0,
            aperture_frames: 8,
            min_signal_quality: 0.25,
            min_confidence: 0.08,
            max_csi_age_ms: 100,
            output_ttl_ms: 250,
            seed: 42,
            fusion_scope: None,
        }
    }
}

impl TrackerConfig {
    fn validate(&self) -> Result<(), TrackerError> {
        if !(64..=20_000).contains(&self.particle_count)
            || !self.search_min_m.finite()
            || !self.search_max_m.finite()
            || self.search_min_m.x >= self.search_max_m.x
            || self.search_min_m.y >= self.search_max_m.y
            || self.search_min_m.z >= self.search_max_m.z
            || [
                self.search_min_m.x,
                self.search_min_m.y,
                self.search_min_m.z,
                self.search_max_m.x,
                self.search_max_m.y,
                self.search_max_m.z,
            ]
            .iter()
            .any(|value| value.abs() > 100.0)
            || self.search_max_m.x - self.search_min_m.x > 6.0
            || self.search_max_m.y - self.search_min_m.y > 6.0
            || self.search_max_m.z - self.search_min_m.z > 6.0
            || !self.motion_std_m.is_finite()
            || !(0.001..=0.5).contains(&self.motion_std_m)
            || !self.eta.is_finite()
            || !(0.1..=16.0).contains(&self.eta)
            || !(1..=32).contains(&self.aperture_frames)
            || !(0.0..=1.0).contains(&self.min_signal_quality)
            || !(0.0..=1.0).contains(&self.min_confidence)
            || self.max_csi_age_ms > 5_000
            || !(1..=5_000).contains(&self.output_ttl_ms)
            || self
                .fusion_scope
                .as_ref()
                .is_some_and(|scope| scope.validate().is_err())
        {
            return Err(TrackerError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct Particle {
    position: Vec3,
    velocity: Vec3,
    weight: f32,
}

/// Stateful single-target motion-induced aperture tracker.
pub struct MotionApertureTracker {
    calibration: Calibration,
    canonical: CanonicalObject,
    config: TrackerConfig,
    particles: Vec<Particle>,
    aperture: VecDeque<PreprocessedFrame>,
    last_sequence: Option<u64>,
    last_monotonic_ns: Option<u64>,
    last_csi_sequence: Option<u64>,
    rng: DeterministicRng,
}

impl MotionApertureTracker {
    /// Construct a tracker with a deterministic uniform prior.
    pub fn new(
        calibration: Calibration,
        canonical: CanonicalObject,
        config: TrackerConfig,
    ) -> Result<Self, TrackerError> {
        config.validate()?;
        canonical.validate()?;
        if config
            .fusion_scope
            .as_ref()
            .is_some_and(|scope| scope.session_id != calibration.session_id())
        {
            return Err(TrackerError::InvalidConfig);
        }
        let work = config
            .particle_count
            .checked_mul(canonical.points_m.len())
            .and_then(|value| value.checked_mul(calibration.zone_count()))
            .and_then(|value| value.checked_mul(config.aperture_frames))
            .ok_or(TrackerError::InvalidConfig)?;
        if work > 100_000_000 {
            return Err(TrackerError::InvalidConfig);
        }
        let mut rng = DeterministicRng::new(config.seed);
        let uniform = 1.0 / config.particle_count as f32;
        let particles = (0..config.particle_count)
            .map(|_| Particle {
                position: random_point(&mut rng, config.search_min_m, config.search_max_m),
                velocity: Vec3::default(),
                weight: uniform,
            })
            .collect();
        Ok(Self {
            calibration,
            canonical,
            config,
            particles,
            aperture: VecDeque::new(),
            last_sequence: None,
            last_monotonic_ns: None,
            last_csi_sequence: None,
            rng,
        })
    }

    /// Process one ordered transient frame and optional calibrated CSI prior.
    pub fn update(
        &mut self,
        frame: &TransientFrame,
        csi_prior: Option<&CsiSpatialPrior>,
    ) -> Result<TrackEnvelope, TrackerError> {
        let expires_at_unix_ms = frame
            .captured_at_unix_ms
            .checked_add(self.config.output_ttl_ms)
            .filter(|value| *value <= MAX_SAFE_INTEGER)
            .ok_or(TrackerError::TimestampOverflow)?;
        if self
            .last_sequence
            .is_some_and(|last| frame.sequence <= last)
            || self
                .last_monotonic_ns
                .is_some_and(|last| frame.monotonic_ns <= last)
        {
            return Err(TrackerError::ReplayOrOutOfOrder);
        }
        let delta_seconds = self.last_monotonic_ns.map_or(1.0 / 30.0, |last| {
            ((frame.monotonic_ns - last) as f32 * 1e-9).clamp(1.0 / 240.0, 0.5)
        });
        let processed = self.calibration.preprocess(frame)?;
        let csi = if let Some(prior) = csi_prior {
            prior.validate()?;
            let expected_scope = self
                .config
                .fusion_scope
                .as_ref()
                .ok_or(FusionError::BindingMismatch)?;
            if &prior.scope != expected_scope
                || frame.session_id != expected_scope.session_id
                || prior.source != frame.source
            {
                return Err(FusionError::BindingMismatch.into());
            }
            if frame.source != FrameSource::Synthetic {
                return Err(FusionError::MeasuredFusionUnavailable.into());
            }
            if self
                .last_csi_sequence
                .is_some_and(|last| prior.sequence <= last)
            {
                return Err(FusionError::ReplayOrOutOfOrder.into());
            }
            let age = frame
                .captured_at_unix_ms
                .abs_diff(prior.captured_at_unix_ms);
            if age > self.config.max_csi_age_ms {
                return Err(TrackerError::Fusion(FusionError::Stale));
            }
            Some(prior)
        } else {
            None
        };
        self.last_sequence = Some(frame.sequence);
        self.last_monotonic_ns = Some(frame.monotonic_ns);
        if let Some(prior) = csi {
            self.last_csi_sequence = Some(prior.sequence);
        }
        self.aperture.push_back(processed);
        while self.aperture.len() > self.config.aperture_frames {
            self.aperture.pop_front();
        }

        self.propagate(delta_seconds);
        let mut log_weights = Vec::with_capacity(self.particles.len());
        let mut max_log = f32::NEG_INFINITY;
        for particle in &self.particles {
            let lidar_score = self.score_particle(particle, frame.monotonic_ns).max(1e-9);
            let mut log_weight = self.config.eta * lidar_score.ln();
            if let Some(prior) = csi {
                log_weight += prior.log_likelihood(particle.position);
            }
            max_log = max_log.max(log_weight);
            log_weights.push(log_weight);
        }
        let mut sum = 0.0_f32;
        for (particle, log_weight) in self.particles.iter_mut().zip(log_weights) {
            particle.weight = (log_weight - max_log).exp();
            sum += particle.weight;
        }
        if !sum.is_finite() || sum <= f32::EPSILON {
            for particle in &mut self.particles {
                particle.weight = 1.0 / self.config.particle_count as f32;
            }
        } else {
            for particle in &mut self.particles {
                particle.weight /= sum;
            }
        }

        let envelope = self.posterior(frame, csi, expires_at_unix_ms);
        self.systematic_resample();
        Ok(envelope)
    }

    fn propagate(&mut self, delta_seconds: f32) {
        let min = self.config.search_min_m;
        let max = self.config.search_max_m;
        for particle in &mut self.particles {
            let innovation = Vec3::new(
                self.rng.gaussian() * self.config.motion_std_m,
                self.rng.gaussian() * self.config.motion_std_m,
                self.rng.gaussian() * self.config.motion_std_m,
            );
            let previous_position = particle.position;
            let proposed = particle
                .position
                .plus(particle.velocity.scale(delta_seconds))
                .plus(innovation);
            particle.position = clamp_vec(proposed, min, max);
            let observed_velocity = particle
                .position
                .minus(previous_position)
                .scale(1.0 / delta_seconds);
            particle.velocity = clamp_vec(
                particle
                    .velocity
                    .scale(0.65)
                    .plus(observed_velocity.scale(0.35)),
                Vec3::new(-20.0, -20.0, -20.0),
                Vec3::new(20.0, 20.0, 20.0),
            );
        }
    }

    fn score_particle(&self, particle: &Particle, current_monotonic_ns: u64) -> f32 {
        let mut aperture_score = 0.0_f32;
        let mut valid_frames = 0_usize;
        for frame in &self.aperture {
            let aperture_age_seconds =
                current_monotonic_ns.saturating_sub(frame.monotonic_ns) as f32 * 1e-9;
            let historical_origin = particle
                .position
                .minus(particle.velocity.scale(aperture_age_seconds));
            let observed_norm = frame
                .light_cone_histograms
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            if observed_norm <= f32::EPSILON {
                continue;
            }
            let native_denominator = (frame.bin_count - 1).max(1) as f32;
            let mut dot = 0.0_f32;
            let mut predicted_norm_sq = 0.0_f32;
            for (zone_index, wall) in frame.wall_points_world_m.iter().enumerate() {
                for (point, weight) in self
                    .canonical
                    .points_m
                    .iter()
                    .zip(self.canonical.weights.iter())
                {
                    let target = historical_origin.plus(*point);
                    let distance = wall.distance(target).max(0.01);
                    let extra_bin = (2.0 * distance
                        / (SPEED_OF_LIGHT_MPS * frame.bin_width_ps * 1e-12))
                        .round();
                    let v_bin = ((extra_bin * extra_bin / native_denominator).floor() as usize)
                        .min(frame.bin_count - 1);
                    let amplitude = *weight / distance.powi(4).max(1e-4);
                    for (offset, kernel) in [(-1_i32, 0.5_f32), (0, 1.0), (1, 0.5)] {
                        let index = v_bin as i32 + offset;
                        if !(0..frame.bin_count as i32).contains(&index) {
                            continue;
                        }
                        let predicted = amplitude * kernel;
                        dot += frame.light_cone_histograms
                            [zone_index * frame.bin_count + index as usize]
                            * predicted;
                        predicted_norm_sq += predicted * predicted;
                    }
                }
            }
            let denom = observed_norm * predicted_norm_sq.sqrt();
            if denom > f32::EPSILON {
                aperture_score += (dot / denom).clamp(0.0, 1.0);
                valid_frames += 1;
            }
        }
        if valid_frames == 0 {
            0.0
        } else {
            aperture_score / valid_frames as f32
        }
    }

    fn posterior(
        &self,
        frame: &TransientFrame,
        csi: Option<&CsiSpatialPrior>,
        expires_at_unix_ms: u64,
    ) -> TrackEnvelope {
        let mut position = Vec3::default();
        let mut velocity = Vec3::default();
        let mut entropy = 0.0_f32;
        for particle in &self.particles {
            position = position.plus(particle.position.scale(particle.weight));
            velocity = velocity.plus(particle.velocity.scale(particle.weight));
            if particle.weight > 0.0 {
                entropy -= particle.weight * particle.weight.ln();
            }
        }
        let mut covariance = Vec3::default();
        for particle in &self.particles {
            let delta = particle.position.minus(position);
            covariance.x += particle.weight * delta.x * delta.x;
            covariance.y += particle.weight * delta.y * delta.y;
            covariance.z += particle.weight * delta.z * delta.z;
        }
        let normalized_entropy = entropy / (self.particles.len() as f32).ln().max(1.0);
        let posterior_focus = (1.0 - normalized_entropy).clamp(0.0, 1.0);
        let lidar_quality = self
            .aperture
            .iter()
            .map(|sample| sample.signal_quality)
            .sum::<f32>()
            / self.aperture.len().max(1) as f32;
        let csi_quality = csi.map_or(0.0, |prior| prior.confidence);
        let signal_quality = 1.0 - (1.0 - lidar_quality) * (1.0 - 0.6 * csi_quality);
        let confidence = (posterior_focus * 1.4 + signal_quality * 0.45).clamp(0.0, 1.0);
        let optical_present = lidar_quality > 1e-6;
        let state = if optical_present
            && signal_quality >= self.config.min_signal_quality
            && confidence >= self.config.min_confidence
        {
            TrackState::Tracking
        } else if optical_present && signal_quality >= self.config.min_signal_quality * 0.5 {
            TrackState::Degraded
        } else {
            TrackState::Unknown
        };
        let lidar_contribution = if csi.is_some() {
            lidar_quality / (lidar_quality + csi_quality + f32::EPSILON)
        } else {
            1.0
        };
        let csi_contribution = if csi.is_some() {
            1.0 - lidar_contribution
        } else {
            0.0
        };
        // v1 cannot retain both modality lineages and therefore never promotes
        // measured output to L3. The only accepted CSI path is synthetic L0.
        let evidence_level = frame.evidence_level;
        let mut provenance = frame.provenance.clone();
        provenance.transport = if frame.source == FrameSource::Live {
            "ruview_server".into()
        } else {
            "replay".into()
        };
        let output_calibration_hash = if frame.source == FrameSource::Synthetic {
            "0".repeat(64)
        } else {
            self.calibration.hash().to_owned()
        };
        let envelope = TrackEnvelope {
            schema: TRACK_SCHEMA_V1.into(),
            session_id: frame.session_id.clone(),
            sequence: frame.sequence,
            captured_at_unix_ms: frame.captured_at_unix_ms,
            expires_at_unix_ms,
            source: frame.source,
            evidence_level,
            algorithm_version: ALGORITHM_VERSION.into(),
            calibration_hash: output_calibration_hash,
            provenance,
            tracks: vec![NlosTrack {
                track_id: "hidden-target-0".into(),
                state,
                position_m: position,
                velocity_mps: velocity,
                covariance_diagonal_m2: covariance,
                confidence,
                posterior_entropy: entropy,
                signal_quality,
                modality_contributions: ModalityContributions {
                    lidar: lidar_contribution.clamp(0.0, 1.0),
                    csi: csi_contribution.clamp(0.0, 1.0),
                },
            }],
        };
        debug_assert!(envelope.validate().is_ok());
        envelope
    }

    fn systematic_resample(&mut self) {
        let count = self.particles.len();
        let step = 1.0 / count as f32;
        let start = self.rng.uniform() * step;
        let mut cumulative = self.particles[0].weight;
        let mut index = 0_usize;
        let mut next = Vec::with_capacity(count);
        for sample in 0..count {
            let threshold = start + sample as f32 * step;
            while threshold > cumulative && index + 1 < count {
                index += 1;
                cumulative += self.particles[index].weight;
            }
            let mut particle = self.particles[index];
            particle.weight = step;
            next.push(particle);
        }
        self.particles = next;
    }
}

fn random_point(rng: &mut DeterministicRng, min: Vec3, max: Vec3) -> Vec3 {
    Vec3::new(
        min.x + rng.uniform() * (max.x - min.x),
        min.y + rng.uniform() * (max.y - min.y),
        min.z + rng.uniform() * (max.z - min.z),
    )
}

fn clamp_vec(value: Vec3, min: Vec3, max: Vec3) -> Vec3 {
    Vec3::new(
        value.x.clamp(min.x, max.x),
        value.y.clamp(min.y, max.y),
        value.z.clamp(min.z, max.z),
    )
}

#[derive(Clone, Debug)]
struct DeterministicRng {
    state: u64,
    spare_gaussian: Option<f32>,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_gaussian: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 + 0.5) / 16_777_216.0
    }

    fn gaussian(&mut self) -> f32 {
        if let Some(value) = self.spare_gaussian.take() {
            return value;
        }
        let u1 = self.uniform().max(f32::EPSILON);
        let u2 = self.uniform();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        self.spare_gaussian = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

/// Tracking failure at a fail-closed boundary.
#[derive(Debug, Error)]
pub enum TrackerError {
    /// Tracker or canonical-object bounds are invalid.
    #[error("invalid tracker configuration")]
    InvalidConfig,
    /// Sequence repeated or moved backwards.
    #[error("replayed or out-of-order transient frame")]
    ReplayOrOutOfOrder,
    /// Capture time plus output TTL cannot be represented by the v1 JSON-safe
    /// integer contract.
    #[error("track expiry exceeds the v1 timestamp range")]
    TimestampOverflow,
    /// Calibration/preprocessing failure.
    #[error(transparent)]
    Calibration(#[from] CalibrationError),
    /// CSI validation or freshness failure.
    #[error(transparent)]
    Fusion(#[from] FusionError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::{Calibration, CalibrationConfig};
    use crate::simulator::SyntheticScene;

    #[test]
    fn replayed_sequence_is_rejected() {
        let mut scene = SyntheticScene::default();
        let calibration = Calibration::from_background(
            &scene.background_frames(20),
            CalibrationConfig::default(),
        )
        .unwrap();
        let mut tracker = MotionApertureTracker::new(
            calibration,
            CanonicalObject::point(),
            TrackerConfig::default(),
        )
        .unwrap();
        let frame = scene.frame(Some(Vec3::new(0.0, 0.0, 1.0)), 1.0, 100);
        tracker.update(&frame, None).unwrap();
        assert!(matches!(
            tracker.update(&frame, None),
            Err(TrackerError::ReplayOrOutOfOrder)
        ));

        let mut advanced_sequence = frame;
        advanced_sequence.sequence += 1;
        assert!(matches!(
            tracker.update(&advanced_sequence, None),
            Err(TrackerError::ReplayOrOutOfOrder)
        ));
    }

    #[test]
    fn max_safe_capture_time_fails_closed_before_expiry_overflow() {
        let mut scene = SyntheticScene::default();
        let calibration = Calibration::from_background(
            &scene.background_frames(20),
            CalibrationConfig::default(),
        )
        .unwrap();
        let mut tracker = MotionApertureTracker::new(
            calibration,
            CanonicalObject::point(),
            TrackerConfig::default(),
        )
        .unwrap();
        let mut frame = scene.frame(Some(Vec3::new(0.0, 0.0, 1.0)), 1.0, 100);
        frame.captured_at_unix_ms = MAX_SAFE_INTEGER;
        assert!(matches!(
            tracker.update(&frame, None),
            Err(TrackerError::TimestampOverflow)
        ));
    }
}
