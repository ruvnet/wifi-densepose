//! Bone-length consistency.

use crate::skeleton::{
    distance, virtual_joints, SkeletonPosterior, CONSTRAINED_EDGE_COUNT, OBSERVED_EDGES,
    VIRTUAL_NECK_EDGE, VIRTUAL_TRUNK_EDGE,
};

/// Mean absolute edge-length residual.
#[must_use]
pub fn bone_residual(joints: &[[f32; 3]; 17], posterior: &SkeletonPosterior) -> f32 {
    let observed_total = OBSERVED_EDGES
        .iter()
        .enumerate()
        .map(|(index, (a, b))| {
            let current = distance(joints[*a], joints[*b]);
            (current - posterior.target(index, current)).abs()
        })
        .sum::<f32>();
    let (pelvis, thorax) = virtual_joints(joints);
    let trunk = distance(pelvis, thorax);
    let neck = distance(thorax, joints[0]);
    let total = observed_total
        + (trunk - posterior.target(VIRTUAL_TRUNK_EDGE, trunk)).abs()
        + (neck - posterior.target(VIRTUAL_NECK_EDGE, neck)).abs();
    total / CONSTRAINED_EDGE_COUNT as f32
}
