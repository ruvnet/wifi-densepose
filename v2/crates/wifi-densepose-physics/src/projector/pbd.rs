//! Covariance-weighted deterministic position-based dynamics projector.

use std::time::{Duration, Instant};

use nalgebra::Vector3;
use wifi_densepose_core::{FloorPlane, InterventionSummary};

use crate::{
    config::PhysicsConfig,
    constraints::{bone_residual, project_floor, project_joint_limits, TemporalHistory},
    skeleton::{
        distance, virtual_joints, SkeletonPosterior, OBSERVED_EDGES, VIRTUAL_NECK_EDGE,
        VIRTUAL_TRUNK_EDGE,
    },
};

/// Successful bounded candidate.
#[derive(Clone, Debug)]
pub struct Projection {
    pub joints: [[f32; 3]; 17],
    pub intervention: InterventionSummary,
}

/// Project distances, temporal motion, and floor penetration within a deadline.
#[allow(clippy::too_many_arguments)]
pub fn project(
    observed: &[[f32; 3]; 17],
    weights: &[f32; 17],
    posterior: &SkeletonPosterior,
    history: &TemporalHistory,
    timestamp_ns: u64,
    floor: FloorPlane,
    config: &PhysicsConfig,
    started: Instant,
) -> Result<Projection, wifi_densepose_core::AbstentionReason> {
    let mut candidate = *observed;
    let deadline = Duration::from_millis(config.deadline_ms);
    let mut iterations = 0u8;
    let mut previous_residual = f32::INFINITY;

    for iteration in 0..config.max_iterations {
        if started.elapsed() >= deadline {
            return Err(wifi_densepose_core::AbstentionReason::DeadlineExceeded);
        }
        for (edge, (parent, child)) in OBSERVED_EDGES.iter().copied().enumerate() {
            let delta = Vector3::from(candidate[child]) - Vector3::from(candidate[parent]);
            let length = delta.norm();
            if length <= 1.0e-6 {
                continue;
            }
            let error = length - posterior.target(edge, length);
            let direction = delta / length;
            let parent_weight = weights[parent];
            let child_weight = weights[child];
            let total_weight = (parent_weight + child_weight).max(1.0e-6);
            let parent_move = direction * error * parent_weight / total_weight;
            let child_move = direction * error * child_weight / total_weight;
            candidate[parent] = (Vector3::from(candidate[parent]) + parent_move).into();
            candidate[child] = (Vector3::from(candidate[child]) - child_move).into();
        }
        project_virtual_edges(&mut candidate, weights, posterior);
        project_joint_limits(&mut candidate, weights);
        project_temporal(&mut candidate, history, timestamp_ns, config);
        project_floor(&mut candidate, floor, weights);
        iterations = iteration + 1;
        let residual = bone_residual(&candidate, posterior);
        if (previous_residual - residual).abs() < config.solver_epsilon_m {
            break;
        }
        previous_residual = residual;
    }

    let mut max_correction = 0.0f32;
    let mut corrected = 0u8;
    for (before, after) in observed.iter().zip(candidate.iter()) {
        let correction = distance(*before, *after);
        max_correction = max_correction.max(correction);
        corrected += u8::from(correction > config.solver_epsilon_m);
    }
    let root_before = midpoint(observed[11], observed[12]);
    let root_after = midpoint(candidate[11], candidate[12]);
    let root_correction = distance(root_before, root_after);
    if max_correction > config.max_joint_correction_m
        || root_correction > config.max_root_correction_m
    {
        return Err(wifi_densepose_core::AbstentionReason::CorrectionTooLarge);
    }
    Ok(Projection {
        joints: candidate,
        intervention: InterventionSummary {
            max_joint_correction_m: max_correction,
            root_correction_m: root_correction,
            corrected_joint_count: corrected,
            solver_iterations: iterations,
            elapsed_us: elapsed_us(started),
        },
    })
}

fn project_virtual_edges(
    joints: &mut [[f32; 3]; 17],
    weights: &[f32; 17],
    posterior: &SkeletonPosterior,
) {
    let (pelvis, thorax) = virtual_joints(joints);
    project_group_edge(
        joints,
        weights,
        &[11, 12],
        &[5, 6],
        pelvis,
        thorax,
        posterior.target(VIRTUAL_TRUNK_EDGE, distance(pelvis, thorax)),
    );
    let (_, thorax) = virtual_joints(joints);
    let nose = joints[0];
    project_group_edge(
        joints,
        weights,
        &[5, 6],
        &[0],
        thorax,
        nose,
        posterior.target(VIRTUAL_NECK_EDGE, distance(thorax, nose)),
    );
}

#[allow(clippy::too_many_arguments)]
fn project_group_edge(
    joints: &mut [[f32; 3]; 17],
    weights: &[f32; 17],
    parent_indices: &[usize],
    child_indices: &[usize],
    parent_point: [f32; 3],
    child_point: [f32; 3],
    target: f32,
) {
    let delta = Vector3::from(child_point) - Vector3::from(parent_point);
    let length = delta.norm();
    if length <= 1.0e-6 {
        return;
    }
    let error = length - target;
    let direction = delta / length;
    let parent_weight =
        parent_indices.iter().map(|&i| weights[i]).sum::<f32>() / parent_indices.len() as f32;
    let child_weight =
        child_indices.iter().map(|&i| weights[i]).sum::<f32>() / child_indices.len() as f32;
    let total = (parent_weight + child_weight).max(1.0e-6);
    let parent_move = direction * error * parent_weight / total;
    let child_move = direction * error * child_weight / total;
    for &index in parent_indices {
        joints[index] = (Vector3::from(joints[index]) + parent_move).into();
    }
    for &index in child_indices {
        joints[index] = (Vector3::from(joints[index]) - child_move).into();
    }
}

fn project_temporal(
    candidate: &mut [[f32; 3]; 17],
    history: &TemporalHistory,
    timestamp_ns: u64,
    config: &PhysicsConfig,
) {
    let Some((previous, previous_ns)) = &history.previous else {
        return;
    };
    let dt = timestamp_ns.saturating_sub(*previous_ns) as f32 / 1_000_000_000.0;
    if dt <= 0.0 || dt * 1000.0 > config.max_frame_gap_ms as f32 {
        return;
    }
    for (index, (joint, prior)) in candidate.iter_mut().zip(previous).enumerate() {
        let mut velocity = (Vector3::from(*joint) - Vector3::from(*prior)) / dt;
        let speed = velocity.norm();
        if speed > config.max_velocity_mps {
            velocity *= config.max_velocity_mps / speed;
        }
        if let Some((older, older_ns)) = &history.previous_previous {
            let previous_dt = previous_ns.saturating_sub(*older_ns) as f32 / 1_000_000_000.0;
            if previous_dt > 0.0 {
                let previous_velocity =
                    (Vector3::from(previous[index]) - Vector3::from(older[index])) / previous_dt;
                let delta_velocity = velocity - previous_velocity;
                let maximum_delta = config.max_acceleration_mps2 * dt;
                if delta_velocity.norm() > maximum_delta {
                    velocity = previous_velocity + delta_velocity.normalize() * maximum_delta;
                }
            }
        }
        *joint = (Vector3::from(*prior) + velocity * dt).into();
    }
}

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}

fn elapsed_us(started: Instant) -> u64 {
    #[cfg(feature = "deterministic")]
    {
        let _ = started;
        0
    }
    #[cfg(not(feature = "deterministic"))]
    {
        started.elapsed().as_micros().try_into().unwrap_or(u64::MAX)
    }
}
