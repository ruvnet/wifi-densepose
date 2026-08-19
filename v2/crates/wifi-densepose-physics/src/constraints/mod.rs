//! Deterministic kinematic residuals.

mod bone;
mod contact;
mod floor;
mod joint_limit;
mod temporal;

pub use bone::bone_residual;
pub use contact::contact_slide_residual;
pub use floor::{floor_penetration, project_floor};
pub use joint_limit::{joint_limit_residual, project_joint_limits};
pub use temporal::{temporal_residuals, TemporalHistory};

use crate::skeleton::{distance, virtual_joints};
use wifi_densepose_core::ConstraintResiduals;

/// Evaluate residuals without mutating the candidate.
#[must_use]
pub fn audit(
    joints: &[[f32; 3]; 17],
    posterior: &crate::skeleton::SkeletonPosterior,
    floor: wifi_densepose_core::FloorPlane,
    history: &TemporalHistory,
    timestamp_ns: u64,
) -> ConstraintResiduals {
    let bone_m = bone_residual(joints, posterior);
    let floor_penetration_m = floor_penetration(joints, floor);
    let (velocity_mps, acceleration_mps2, temporal_jerk) =
        temporal_residuals(joints, history, timestamp_ns);
    let contact_m = contact_slide_residual(joints, floor, history, timestamp_ns);
    let (pelvis, thorax) = virtual_joints(joints);
    let trunk = distance(pelvis, thorax);
    let trunk_violation = if (0.05..=1.2).contains(&trunk) {
        0.0
    } else {
        1.0
    };
    let joint_limit_rad = joint_limit_residual(joints).max(trunk_violation);
    let normalized_total = (bone_m / 0.05
        + floor_penetration_m / 0.05
        + velocity_mps / 8.0
        + acceleration_mps2 / 40.0
        + joint_limit_rad
        + contact_m / 0.25)
        / 6.0;
    ConstraintResiduals {
        bone_m,
        joint_limit_rad,
        velocity_mps,
        acceleration_mps2,
        temporal_jerk,
        floor_penetration_m,
        contact_m,
        collision_m: 0.0,
        normalized_total,
    }
}
