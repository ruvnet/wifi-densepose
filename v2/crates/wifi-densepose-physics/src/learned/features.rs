//! Fixed-width, bounded learned-residual feature encoding.

use std::collections::VecDeque;

use wifi_densepose_core::{ConstraintResiduals, PoseObservationV2};

use crate::constraints::TemporalHistory;

pub const RF_EMBEDDING_WIDTH: usize = 128;
pub const BASE_FEATURE_WIDTH: usize = 319;
pub const FEATURE_WIDTH: usize = BASE_FEATURE_WIDTH + RF_EMBEDDING_WIDTH;

/// Exactly one encoded model timestep.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseFeatureFrame {
    values: Vec<f32>,
}

impl PoseFeatureFrame {
    /// Encode observer, uncertainty, deterministic residual, temporal, floor,
    /// OOD, and optional RF embedding features without allocating unbounded data.
    #[must_use]
    pub fn encode(
        raw: &PoseObservationV2,
        residuals: ConstraintResiduals,
        history: &TemporalHistory,
        rf_embedding: Option<&[f32; RF_EMBEDDING_WIDTH]>,
    ) -> Self {
        let mut values = Vec::with_capacity(FEATURE_WIDTH);
        for joint in &raw.joints {
            values.extend(joint.position_m);
        }
        for joint in &raw.joints {
            values.extend([
                joint.covariance_m2.xx,
                joint.covariance_m2.xy,
                joint.covariance_m2.xz,
                joint.covariance_m2.yy,
                joint.covariance_m2.yz,
                joint.covariance_m2.zz,
            ]);
        }
        values.extend(raw.joints.iter().map(|joint| joint.confidence.get()));
        values.extend(
            raw.joints
                .iter()
                .map(|joint| f32::from(joint.visibility as u8)),
        );
        values.extend([
            residuals.bone_m,
            residuals.joint_limit_rad,
            residuals.velocity_mps,
            residuals.acceleration_mps2,
            residuals.temporal_jerk,
            residuals.floor_penetration_m,
            residuals.contact_m,
            residuals.collision_m,
            residuals.normalized_total,
        ]);
        let current = raw.joints.map(|joint| joint.position_m);
        let velocity = history
            .previous
            .as_ref()
            .map_or([[0.0; 3]; 17], |(previous, ns)| {
                derivative(&current, previous, raw.timestamp_ns.saturating_sub(*ns))
            });
        for joint in velocity {
            values.extend(joint);
        }
        let acceleration = match (&history.previous, &history.previous_previous) {
            (Some((previous, previous_ns)), Some((older, older_ns))) => {
                let previous_velocity =
                    derivative(previous, older, previous_ns.saturating_sub(*older_ns));
                derivative(
                    &velocity,
                    &previous_velocity,
                    raw.timestamp_ns.saturating_sub(*previous_ns),
                )
            }
            _ => [[0.0; 3]; 17],
        };
        for joint in acceleration {
            values.extend(joint);
        }
        values.extend(raw.joints.iter().map(|joint| {
            raw.floor_plane
                .map_or(0.0, |floor| floor.signed_distance(joint.position_m))
        }));
        values.extend(match raw.trust_state {
            wifi_densepose_core::PoseTrustState::Known => [1.0, 0.0, 0.0],
            wifi_densepose_core::PoseTrustState::Degraded => [0.0, 1.0, 0.0],
            wifi_densepose_core::PoseTrustState::Unknown => [0.0, 0.0, 1.0],
        });
        values.push(raw.observer_confidence.get());
        values.extend(rf_embedding.copied().unwrap_or([0.0; RF_EMBEDDING_WIDTH]));
        debug_assert_eq!(values.len(), FEATURE_WIDTH);
        Self { values }
    }

    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Bounded twenty-frame history used by both trainer and inference.
#[derive(Clone, Debug, Default)]
pub struct FeatureHistory {
    frames: VecDeque<PoseFeatureFrame>,
}

impl FeatureHistory {
    pub fn push(&mut self, frame: PoseFeatureFrame) {
        if self.frames.len() == super::model::HISTORY_FRAMES {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.frames.len() == super::model::HISTORY_FRAMES
    }

    /// Flatten only a complete history; partial windows must abstain.
    #[must_use]
    pub fn flattened(&self) -> Option<Vec<f32>> {
        self.is_ready().then(|| {
            self.frames
                .iter()
                .flat_map(|frame| frame.values.iter().copied())
                .collect()
        })
    }
}

fn derivative(current: &[[f32; 3]; 17], previous: &[[f32; 3]; 17], dt_ns: u64) -> [[f32; 3]; 17] {
    let dt = dt_ns as f32 / 1_000_000_000.0;
    if dt <= 0.0 {
        return [[0.0; 3]; 17];
    }
    core::array::from_fn(|joint| {
        core::array::from_fn(|axis| (current[joint][axis] - previous[joint][axis]) / dt)
    })
}
