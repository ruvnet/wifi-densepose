//! Deterministic L0 simulator and comparative fusion benchmark.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::calibration::{Calibration, CalibrationConfig};
use crate::fusion::{CsiSpatialPrior, FusionScope};
use crate::protocol::{
    EvidenceLevel, FrameSource, Provenance, SensorPose, TrackState, TransientFrame, TransientKind,
    TransientZone, Vec3,
};
use crate::tracker::{CanonicalObject, MotionApertureTracker, TrackerConfig};
use crate::TRANSIENT_SCHEMA_V1;

const C: f32 = 299_792_458.0;

/// Controlled photon-histogram generator. It is intentionally labelled L0 and
/// cannot substitute for a VL53L8CH hardware capture.
pub struct SyntheticScene {
    /// Timing width.
    pub bin_width_ps: f32,
    /// Timing bins.
    pub bin_count: usize,
    /// Relay-wall samples.
    pub wall_points_m: Vec<Vec3>,
    /// Direct return peak index.
    pub direct_peak_bin: usize,
    /// Direct peak photon count.
    pub direct_amplitude: u16,
    /// Third-bounce target count scale.
    pub target_amplitude: f32,
    rng: SimRng,
}

impl Default for SyntheticScene {
    fn default() -> Self {
        let mut wall_points_m = Vec::new();
        for row in 0..4 {
            for col in 0..4 {
                wall_points_m.push(Vec3::new(
                    -0.45 + col as f32 * 0.3,
                    -0.45 + row as f32 * 0.3,
                    0.0,
                ));
            }
        }
        Self {
            bin_width_ps: 250.0,
            bin_count: 48,
            wall_points_m,
            direct_peak_bin: 3,
            direct_amplitude: 600,
            target_amplitude: 900.0,
            rng: SimRng::new(0x4e4c_4f53),
        }
    }
}

impl SyntheticScene {
    /// Generate an empty-room calibration sequence.
    pub fn background_frames(&mut self, count: usize) -> Vec<TransientFrame> {
        (0..count)
            .map(|sequence| self.frame(None, 0.0, sequence as u64))
            .collect()
    }

    /// Generate one raw transient frame. `visibility` scales only the weak
    /// third-bounce return and can model optical dropout.
    pub fn frame(
        &mut self,
        target: Option<Vec3>,
        visibility: f32,
        sequence: u64,
    ) -> TransientFrame {
        let aperture_shift = 0.018 * (sequence as f32 * 0.31).sin();
        let pose = SensorPose {
            translation_m: Vec3::new(aperture_shift, 0.0, 0.0),
            quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
        };
        let zones = self
            .wall_points_m
            .clone()
            .into_iter()
            .enumerate()
            .map(|(zone_id, wall)| {
                let mut histogram = vec![0_u16; self.bin_count];
                for value in &mut histogram {
                    *value = 8 + (self.rng.next_u32() % 5) as u16;
                }
                for (offset, scale) in [(-1_i32, 0.25_f32), (0, 1.0), (1, 0.25)] {
                    let index = self.direct_peak_bin as i32 + offset;
                    if (0..self.bin_count as i32).contains(&index) {
                        histogram[index as usize] = histogram[index as usize]
                            .saturating_add((f32::from(self.direct_amplitude) * scale) as u16);
                    }
                }
                if let Some(target) = target {
                    let world_wall = pose.transform(wall);
                    let distance = world_wall.distance(target).max(0.05);
                    let extra_bin =
                        (2.0 * distance / (C * self.bin_width_ps * 1e-12)).round() as usize;
                    let target_bin = self.direct_peak_bin + extra_bin;
                    if target_bin < self.bin_count {
                        let amplitude = (self.target_amplitude * visibility / distance.powi(4))
                            .clamp(0.0, 20_000.0);
                        for (offset, scale) in [(-1_i32, 0.4_f32), (0, 1.0), (1, 0.4)] {
                            let index = target_bin as i32 + offset;
                            if (0..self.bin_count as i32).contains(&index) {
                                histogram[index as usize] = histogram[index as usize]
                                    .saturating_add((amplitude * scale) as u16);
                            }
                        }
                    }
                }
                TransientZone {
                    zone_id: zone_id as u16,
                    wall_point_m: wall,
                    distance_m: 0.8,
                    ambient: 8,
                    histogram,
                }
            })
            .collect();
        let frame = TransientFrame {
            schema: TRANSIENT_SCHEMA_V1.into(),
            session_id: "synthetic-nlos-1".into(),
            sequence,
            captured_at_unix_ms: 1_800_000_000_000 + sequence * 33,
            monotonic_ns: sequence * 33_333_333,
            source: FrameSource::Synthetic,
            evidence_level: EvidenceLevel::L0Synthetic,
            bin_width_ps: self.bin_width_ps,
            start_bin: 30,
            sensor_pose: pose,
            calibration_hash: "0".repeat(64),
            provenance: Provenance {
                sensor_id: "sim-vl53l8ch".into(),
                sensor_model: "VL53L8CH-simulator".into(),
                firmware_version: "sim-v1".into(),
                transient_kind: TransientKind::Replay,
                histogram_preserved: true,
                transport: "replay".into(),
            },
            zones,
        };
        debug_assert!(frame.validate().is_ok());
        frame
    }

    /// Run a deterministic LiDAR-only versus LiDAR-plus-CSI comparison.
    pub fn benchmark(frames: usize, particles: usize) -> BenchmarkReport {
        let mut scene = Self::default();
        let fusion_scope = FusionScope {
            tenant_id: "synthetic-tenant".into(),
            workspace_id: "synthetic-workspace".into(),
            site_id: "synthetic-site".into(),
            world_frame_id: "synthetic-world".into(),
            session_id: "synthetic-nlos-1".into(),
            coordinate_transform_hash: "f".repeat(64),
        };
        let calibration = Calibration::from_background(
            &scene.background_frames(60),
            CalibrationConfig::default(),
        )
        .expect("synthetic calibration is valid");
        let config = TrackerConfig {
            particle_count: particles,
            search_min_m: Vec3::new(-0.6, -0.5, 0.35),
            search_max_m: Vec3::new(0.6, 0.5, 1.5),
            fusion_scope: Some(fusion_scope.clone()),
            ..TrackerConfig::default()
        };
        let mut lidar = MotionApertureTracker::new(
            calibration.clone(),
            CanonicalObject::point(),
            config.clone(),
        )
        .expect("benchmark config is valid");
        let mut fused = MotionApertureTracker::new(calibration, CanonicalObject::point(), config)
            .expect("benchmark config is valid");
        let mut lidar_errors = Vec::new();
        let mut fused_errors = Vec::new();
        let mut lidar_lost = 0_usize;
        let mut fused_lost = 0_usize;
        let started = Instant::now();
        for index in 0..frames {
            let t = index as f32 / frames.max(1) as f32;
            let truth = Vec3::new(
                -0.32 + 0.64 * t,
                0.12 * (t * std::f32::consts::TAU).sin(),
                0.92 + 0.08 * (t * std::f32::consts::TAU * 0.5).cos(),
            );
            // Twelve-frame optical dropouts outlast the eight-frame aperture,
            // making lost-track recovery measurable rather than cosmetic.
            let dropout = index % 24 < 12;
            let visibility = if dropout { 0.005 } else { 1.0 };
            let frame = scene.frame(Some(truth), visibility, 100 + index as u64);
            let csi = CsiSpatialPrior {
                source: FrameSource::Synthetic,
                sequence: index as u64,
                captured_at_unix_ms: frame.captured_at_unix_ms,
                scope: fusion_scope.clone(),
                mean_m: Vec3::new(
                    truth.x + 0.025 * (index as f32 * 0.7).sin(),
                    truth.y + 0.03 * (index as f32 * 0.4).cos(),
                    truth.z + 0.04 * (index as f32 * 0.2).sin(),
                ),
                covariance_diagonal_m2: Vec3::new(0.04, 0.04, 0.09),
                confidence: 0.72,
                evidence_level: EvidenceLevel::L0Synthetic,
                sensor_id: "sim-csi-1".into(),
                calibration_hash: "0".repeat(64),
            };
            let lidar_output = lidar.update(&frame, None).expect("ordered frame");
            let fused_output = fused.update(&frame, Some(&csi)).expect("valid prior");
            collect_metric(&lidar_output, truth, &mut lidar_errors, &mut lidar_lost);
            collect_metric(&fused_output, truth, &mut fused_errors, &mut fused_lost);
        }
        let elapsed = started.elapsed();
        let lidar_lost_rate = lidar_lost as f32 / frames.max(1) as f32;
        let fused_lost_rate = fused_lost as f32 / frames.max(1) as f32;
        let lost_track_reduction_percent = if lidar_lost_rate > 0.0 {
            100.0 * (lidar_lost_rate - fused_lost_rate) / lidar_lost_rate
        } else {
            0.0
        };
        BenchmarkReport {
            evidence: "SYNTHETIC_L0".into(),
            frames,
            particles,
            throughput_fps: frames as f64 / elapsed.as_secs_f64().max(1e-9),
            lidar_only_mean_error_m: mean(&lidar_errors),
            fused_mean_error_m: mean(&fused_errors),
            lidar_only_p95_error_m: percentile95(&mut lidar_errors),
            fused_p95_error_m: percentile95(&mut fused_errors),
            lidar_only_lost_track_rate: lidar_lost_rate,
            fused_lost_track_rate: fused_lost_rate,
            lost_track_reduction_percent,
            hardware_reproduction_gate_passed: false,
        }
    }
}

fn collect_metric(
    envelope: &crate::protocol::TrackEnvelope,
    truth: Vec3,
    errors: &mut Vec<f32>,
    lost: &mut usize,
) {
    let track = &envelope.tracks[0];
    if track.state == TrackState::Tracking {
        errors.push(track.position_m.distance(truth));
    } else {
        *lost += 1;
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        f32::INFINITY
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

fn percentile95(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return f32::INFINITY;
    }
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * 0.95).round() as usize]
}

/// Reproducible benchmark result. The hardware gate always remains false for
/// this simulator regardless of performance.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkReport {
    /// Explicit evidence watermark.
    pub evidence: String,
    /// Frames evaluated.
    pub frames: usize,
    /// Particles per tracker.
    pub particles: usize,
    /// Combined LiDAR-only and fused updates per wall-clock second.
    pub throughput_fps: f64,
    /// LiDAR-only mean error over non-lost frames.
    pub lidar_only_mean_error_m: f32,
    /// Fused mean error over non-lost frames.
    pub fused_mean_error_m: f32,
    /// LiDAR-only p95 error.
    pub lidar_only_p95_error_m: f32,
    /// Fused p95 error.
    pub fused_p95_error_m: f32,
    /// LiDAR-only lost-track fraction.
    pub lidar_only_lost_track_rate: f32,
    /// Fused lost-track fraction.
    pub fused_lost_track_rate: f32,
    /// Relative lost-track reduction.
    pub lost_track_reduction_percent: f32,
    /// Always false for generated input.
    pub hardware_reproduction_gate_passed: bool,
}

#[derive(Clone, Debug)]
struct SimRng(u64);

impl SimRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_frames_are_always_l0_and_preserve_histograms() {
        let mut scene = SyntheticScene::default();
        let frame = scene.frame(Some(Vec3::new(0.0, 0.0, 1.0)), 1.0, 1);
        assert_eq!(frame.source, FrameSource::Synthetic);
        assert_eq!(frame.evidence_level, EvidenceLevel::L0Synthetic);
        assert!(frame.provenance.histogram_preserved);
        frame.validate().unwrap();
    }

    #[test]
    fn benchmark_never_promotes_synthetic_to_hardware_evidence() {
        let report = SyntheticScene::benchmark(40, 128);
        assert_eq!(report.evidence, "SYNTHETIC_L0");
        assert!(!report.hardware_reproduction_gate_passed);
    }
}
