//! Calibrated floor-plane constraint. It prevents penetration only.

use wifi_densepose_core::FloorPlane;

/// Maximum penetration among observed joints.
#[must_use]
pub fn floor_penetration(joints: &[[f32; 3]; 17], floor: FloorPlane) -> f32 {
    joints
        .iter()
        .map(|point| (-floor.signed_distance(*point)).max(0.0))
        .fold(0.0, f32::max)
}

/// Move penetrating joints to the plane. No upright or contact prior is used.
pub fn project_floor(joints: &mut [[f32; 3]; 17], floor: FloorPlane, _weights: &[f32; 17]) {
    for joint in joints.iter_mut() {
        let penetration = (-floor.signed_distance(*joint)).max(0.0);
        if penetration <= 0.0 {
            continue;
        }
        for (coordinate, normal) in joint.iter_mut().zip(floor.normal) {
            *coordinate += normal * penetration;
        }
    }
}
