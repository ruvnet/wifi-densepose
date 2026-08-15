//! Broad hinge-angle constraints that reject only anatomically impossible poses.

use nalgebra::Vector3;

/// `(parent, hinge, distal, minimum angle, maximum angle)`.
const HINGES: [(usize, usize, usize, f32, f32); 4] = [
    (5, 7, 9, 0.087_266_46, 3.124_139_3),
    (6, 8, 10, 0.087_266_46, 3.124_139_3),
    (11, 13, 15, 0.087_266_46, 3.124_139_3),
    (12, 14, 16, 0.087_266_46, 3.124_139_3),
];

/// Maximum angular violation in radians.
#[must_use]
pub fn joint_limit_residual(joints: &[[f32; 3]; 17]) -> f32 {
    HINGES
        .iter()
        .map(|&(parent, hinge, distal, minimum, maximum)| {
            let angle = angle_at(joints[parent], joints[hinge], joints[distal]);
            if angle < minimum {
                minimum - angle
            } else if angle > maximum {
                angle - maximum
            } else {
                0.0
            }
        })
        .fold(0.0, f32::max)
}

/// Move only the distal point toward a legal hinge angle. The movement weight
/// makes uncertain joints converge faster while the outer correction cap
/// remains authoritative.
pub fn project_joint_limits(joints: &mut [[f32; 3]; 17], weights: &[f32; 17]) {
    for &(parent, hinge, distal, minimum, maximum) in &HINGES {
        let parent_direction = Vector3::from(joints[parent]) - Vector3::from(joints[hinge]);
        let distal_vector = Vector3::from(joints[distal]) - Vector3::from(joints[hinge]);
        let parent_length = parent_direction.norm();
        let distal_length = distal_vector.norm();
        if parent_length <= 1.0e-6 || distal_length <= 1.0e-6 {
            continue;
        }
        let parent_unit = parent_direction / parent_length;
        let distal_unit = distal_vector / distal_length;
        let angle = parent_unit.dot(&distal_unit).clamp(-1.0, 1.0).acos();
        let target = angle.clamp(minimum, maximum);
        if (target - angle).abs() <= f32::EPSILON {
            continue;
        }
        let mut perpendicular = distal_unit - parent_unit * parent_unit.dot(&distal_unit);
        if perpendicular.norm_squared() <= 1.0e-8 {
            let basis = if parent_unit.x.abs() < 0.8 {
                Vector3::x()
            } else {
                Vector3::y()
            };
            perpendicular = parent_unit.cross(&basis);
        }
        let perpendicular = perpendicular.normalize();
        let target_direction = parent_unit * target.cos() + perpendicular * target.sin();
        let target_point = Vector3::from(joints[hinge]) + target_direction * distal_length;
        let alpha = weights[distal].clamp(0.05, 1.0);
        joints[distal] =
            (Vector3::from(joints[distal]) * (1.0 - alpha) + target_point * alpha).into();
    }
}

fn angle_at(parent: [f32; 3], hinge: [f32; 3], distal: [f32; 3]) -> f32 {
    let a = Vector3::from(parent) - Vector3::from(hinge);
    let b = Vector3::from(distal) - Vector3::from(hinge);
    let denominator = a.norm() * b.norm();
    if denominator <= 1.0e-8 {
        return 0.0;
    }
    (a.dot(&b) / denominator).clamp(-1.0, 1.0).acos()
}
