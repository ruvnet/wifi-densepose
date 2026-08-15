//! Bounded temporal state and residuals.

use nalgebra::Vector3;

use crate::skeleton::distance;

/// At most two previous frames, held only for a local track.
#[derive(Clone, Debug, Default)]
pub struct TemporalHistory {
    pub previous: Option<([[f32; 3]; 17], u64)>,
    pub previous_previous: Option<([[f32; 3]; 17], u64)>,
}

impl TemporalHistory {
    /// Commit a validated observation/candidate.
    pub fn push(&mut self, joints: [[f32; 3]; 17], timestamp_ns: u64) {
        self.previous_previous = self.previous.take();
        self.previous = Some((joints, timestamp_ns));
    }

    /// Drop derivatives after a discontinuity.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Maximum velocity, acceleration, and normalized jerk.
#[must_use]
pub fn temporal_residuals(
    joints: &[[f32; 3]; 17],
    history: &TemporalHistory,
    timestamp_ns: u64,
) -> (f32, f32, f32) {
    let Some((previous, previous_ns)) = &history.previous else {
        return (0.0, 0.0, 0.0);
    };
    let dt = seconds(timestamp_ns.saturating_sub(*previous_ns));
    if dt <= 0.0 {
        return (f32::INFINITY, f32::INFINITY, f32::INFINITY);
    }
    let velocity = joints
        .iter()
        .zip(previous)
        .map(|(a, b)| distance(*a, *b) / dt)
        .fold(0.0, f32::max);
    let Some((older, older_ns)) = &history.previous_previous else {
        return (velocity, 0.0, 0.0);
    };
    let previous_dt = seconds(previous_ns.saturating_sub(*older_ns));
    if previous_dt <= 0.0 {
        return (velocity, f32::INFINITY, f32::INFINITY);
    }
    let mut acceleration = 0.0f32;
    for index in 0..17 {
        let v_now = (Vector3::from(joints[index]) - Vector3::from(previous[index])) / dt;
        let v_previous =
            (Vector3::from(previous[index]) - Vector3::from(older[index])) / previous_dt;
        acceleration = acceleration.max((v_now - v_previous).norm() / dt);
    }
    (velocity, acceleration, acceleration * dt)
}

fn seconds(nanoseconds: u64) -> f32 {
    nanoseconds as f32 / 1_000_000_000.0
}
