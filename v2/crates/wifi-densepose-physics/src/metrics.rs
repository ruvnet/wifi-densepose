//! Privacy-safe metric labels and bounded in-process aggregation.

use std::{collections::BTreeMap, fmt::Write as _, sync::Mutex};

use wifi_densepose_core::{PoseRefinementV1, RefinementDisposition};

/// Fixed allowlisted scalars. It deliberately contains no joint, room, or ID data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricSnapshot {
    pub latency_seconds: f64,
    pub solver_iterations: u8,
    pub max_correction_meters: f32,
    pub bone_residual: f32,
    pub floor_penetration_meters: f32,
    pub temporal_jerk: f32,
    pub confidence_delta: f32,
}

#[derive(Default)]
struct Aggregate {
    frames: BTreeMap<(String, String, String), u64>,
    total_latency_seconds: f64,
    observations: u64,
    deadline_exceeded: u64,
    invalid_inputs: u64,
    track_resets: BTreeMap<String, u64>,
    dropped: u64,
    latest: Option<MetricSnapshot>,
}

/// Thread-safe, bounded metrics registry with no person, joint, or room labels.
#[derive(Default)]
pub struct PhysicsMetrics {
    aggregate: Mutex<Aggregate>,
}

impl PhysicsMetrics {
    /// Record one returned result. The label cardinality is bounded by enums.
    pub fn observe(&self, result: &PoseRefinementV1, observer_confidence: f32) {
        let snapshot = MetricSnapshot::from_result(result, observer_confidence);
        let mut aggregate = self
            .aggregate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let key = (
            enum_label(result.mode),
            enum_label(result.disposition),
            result.reason.map_or_else(|| "none".into(), enum_label),
        );
        *aggregate.frames.entry(key).or_default() += 1;
        aggregate.total_latency_seconds += snapshot.latency_seconds;
        aggregate.observations += 1;
        if result.reason == Some(wifi_densepose_core::AbstentionReason::DeadlineExceeded) {
            aggregate.deadline_exceeded += 1;
        }
        if result.disposition == RefinementDisposition::Rejected {
            aggregate.invalid_inputs += 1;
        }
        aggregate.latest = Some(snapshot);
    }

    /// Record a bounded internal track reset reason.
    pub fn track_reset(&self, reason: &'static str) {
        let mut aggregate = self
            .aggregate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *aggregate.track_resets.entry(reason.into()).or_default() += 1;
    }

    /// Record discarded intermediate refinement work under backpressure.
    pub fn dropped_intermediate(&self) {
        let mut aggregate = self
            .aggregate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        aggregate.dropped += 1;
    }

    /// Encode the fixed metric allowlist in Prometheus text format.
    #[must_use]
    pub fn encode_prometheus(&self) -> String {
        let aggregate = self
            .aggregate
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut output = String::with_capacity(2048);
        for ((mode, disposition, reason), count) in &aggregate.frames {
            writeln!(
                output,
                "ruview_pose_physics_frames_total{{mode=\"{mode}\",disposition=\"{disposition}\",reason=\"{reason}\"}} {count}"
            )
            .expect("writing to String cannot fail");
        }
        let average_latency = if aggregate.observations == 0 {
            0.0
        } else {
            aggregate.total_latency_seconds / aggregate.observations as f64
        };
        writeln!(
            output,
            "ruview_pose_physics_latency_seconds{{stage=\"total\",target=\"local\"}} {average_latency}"
        )
        .expect("writing to String cannot fail");
        if let Some(latest) = aggregate.latest {
            write!(
                output,
                "ruview_pose_physics_solver_iterations{{engine=\"kinematic-pbd\"}} {}\nruview_pose_physics_max_correction_meters {}\nruview_pose_physics_bone_residual {}\nruview_pose_physics_floor_penetration_meters {}\nruview_pose_physics_temporal_jerk {}\nruview_pose_physics_confidence_delta {}\nruview_pose_physics_raw_refined_divergence_meters {}\n",
                latest.solver_iterations,
                latest.max_correction_meters,
                latest.bone_residual,
                latest.floor_penetration_meters,
                latest.temporal_jerk,
                latest.confidence_delta,
                latest.max_correction_meters,
            )
            .expect("writing to String cannot fail");
        }
        for (reason, count) in &aggregate.track_resets {
            writeln!(
                output,
                "ruview_pose_physics_track_resets_total{{reason=\"{reason}\"}} {count}"
            )
            .expect("writing to String cannot fail");
        }
        write!(
            output,
            "ruview_pose_physics_invalid_input_total{{reason=\"contract\"}} {}\nruview_pose_physics_deadline_exceeded_total{{stage=\"total\"}} {}\nruview_pose_physics_dropped_intermediate_total {}\n",
            aggregate.invalid_inputs, aggregate.deadline_exceeded, aggregate.dropped,
        )
        .expect("writing to String cannot fail");
        output
    }
}

fn enum_label(value: impl core::fmt::Debug) -> String {
    format!("{value:?}").to_ascii_lowercase()
}

impl MetricSnapshot {
    /// Extract only approved aggregate scalars.
    #[must_use]
    pub fn from_result(result: &PoseRefinementV1, observer_confidence: f32) -> Self {
        Self {
            latency_seconds: result.intervention.elapsed_us as f64 / 1_000_000.0,
            solver_iterations: result.intervention.solver_iterations,
            max_correction_meters: result.intervention.max_joint_correction_m,
            bone_residual: result.residuals.bone_m,
            floor_penetration_meters: result.residuals.floor_penetration_m,
            temporal_jerk: result.residuals.temporal_jerk,
            confidence_delta: result.effective_confidence.get() - observer_confidence,
        }
    }
}
