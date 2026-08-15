//! Foot-contact hypotheses derived from floor proximity and tangential motion.

use nalgebra::Vector3;
use wifi_densepose_core::FloorPlane;

use super::TemporalHistory;

/// Maximum tangential foot displacement while an ankle is close enough to the floor
/// to be a contact hypothesis. This is never labeled as measured contact.
#[must_use]
pub fn contact_slide_residual(
    joints: &[[f32; 3]; 17],
    floor: FloorPlane,
    history: &TemporalHistory,
    timestamp_ns: u64,
) -> f32 {
    let Some((previous, previous_ns)) = &history.previous else {
        return 0.0;
    };
    if timestamp_ns <= *previous_ns {
        return 0.0;
    }
    let normal = Vector3::from(floor.normal);
    [15usize, 16usize]
        .into_iter()
        .filter(|&index| floor.signed_distance(joints[index]).abs() <= 0.05)
        .map(|index| {
            let displacement = Vector3::from(joints[index]) - Vector3::from(previous[index]);
            (displacement - normal * displacement.dot(&normal)).norm()
        })
        .fold(0.0, f32::max)
}
