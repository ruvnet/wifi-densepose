//! COCO-17 skeleton topology and anonymous track-scoped bone posteriors.

use nalgebra::Vector3;
use smallvec::SmallVec;

/// Observed parent-child constraints. Virtual pelvis/thorax constraints are
/// evaluated separately so virtual joints can never be mislabeled observed.
pub const OBSERVED_EDGES: [(usize, usize); 14] = [
    (11, 12),
    (5, 6),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),
];

/// Fourteen observed edges plus pelvis↔thorax and thorax↔nose virtual edges.
pub const CONSTRAINED_EDGE_COUNT: usize = OBSERVED_EDGES.len() + 2;
/// Posterior index for the derived pelvis↔thorax edge.
pub const VIRTUAL_TRUNK_EDGE: usize = OBSERVED_EDGES.len();
/// Posterior index for the derived thorax↔nose edge.
pub const VIRTUAL_NECK_EDGE: usize = OBSERVED_EDGES.len() + 1;

/// Anonymous, memory-only per-track length estimate.
#[derive(Clone, Debug)]
pub struct SkeletonPosterior {
    samples_m: [SmallVec<[f32; 9]>; CONSTRAINED_EDGE_COUNT],
}

impl Default for SkeletonPosterior {
    fn default() -> Self {
        Self {
            samples_m: core::array::from_fn(|_| SmallVec::new()),
        }
    }
}

impl SkeletonPosterior {
    /// Update only edges whose endpoints pass the high-confidence gate.
    pub fn observe(&mut self, joints: &[[f32; 3]; 17], confidence: &[f32; 17], threshold: f32) {
        for (index, (a, b)) in OBSERVED_EDGES.iter().copied().enumerate() {
            if confidence[a] < threshold || confidence[b] < threshold {
                continue;
            }
            let length = distance(joints[a], joints[b]);
            if !length.is_finite() || !(0.01..=2.5).contains(&length) {
                continue;
            }
            self.observe_sample(index, length);
        }
        let (pelvis, thorax) = virtual_joints(joints);
        let trunk_confidence = confidence[11]
            .min(confidence[12])
            .min(confidence[5])
            .min(confidence[6]);
        if trunk_confidence >= threshold {
            self.observe_sample(VIRTUAL_TRUNK_EDGE, distance(pelvis, thorax));
        }
        let neck_confidence = confidence[5].min(confidence[6]).min(confidence[0]);
        if neck_confidence >= threshold {
            self.observe_sample(VIRTUAL_NECK_EDGE, distance(thorax, joints[0]));
        }
    }

    /// Return the posterior target, or the current length before calibration.
    #[must_use]
    pub fn target(&self, edge: usize, current: f32) -> f32 {
        let samples = &self.samples_m[edge];
        if samples.len() < 3 {
            current
        } else {
            median(samples)
        }
    }

    fn observe_sample(&mut self, edge: usize, length: f32) {
        if !length.is_finite() || !(0.01..=2.5).contains(&length) {
            return;
        }
        let samples = &mut self.samples_m[edge];
        if samples.len() >= 3 {
            let center = median(samples);
            let mut deviations = SmallVec::<[f32; 9]>::new();
            deviations.extend(samples.iter().map(|sample| (sample - center).abs()));
            let mad = median(&deviations);
            let tolerance = (center * 0.15).max(mad * 4.0).max(0.02);
            if (length - center).abs() > tolerance {
                return;
            }
        }
        if samples.len() == 9 {
            samples.remove(0);
        }
        samples.push(length);
    }
}

fn median(samples: &[f32]) -> f32 {
    let mut sorted = SmallVec::<[f32; 9]>::from_slice(samples);
    sorted.sort_unstable_by(f32::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) * 0.5
    } else {
        sorted[middle]
    }
}

/// Euclidean distance between points.
#[must_use]
pub fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    (Vector3::from(b) - Vector3::from(a)).norm()
}

/// Derived pelvis and thorax, never exposed as observations.
#[must_use]
pub fn virtual_joints(joints: &[[f32; 3]; 17]) -> ([f32; 3], [f32; 3]) {
    (
        midpoint(joints[11], joints[12]),
        midpoint(joints[5], joints[6]),
    )
}

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ]
}
