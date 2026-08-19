//! Auditable supervised residual objective shared by trainers and reports.

/// Unweighted components reported for every training/evaluation batch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LossComponents {
    pub uncertainty_weighted_mpjpe: f32,
    pub bone: f32,
    pub temporal_jerk: f32,
    pub contact: f32,
    pub uncertainty_calibration: f32,
    pub intervention: f32,
}

impl LossComponents {
    /// ADR-323 supervised objective. Non-finite components are rejected.
    #[must_use]
    pub fn total(self) -> Option<f32> {
        let values = [
            self.uncertainty_weighted_mpjpe,
            self.bone,
            self.temporal_jerk,
            self.contact,
            self.uncertainty_calibration,
            self.intervention,
        ];
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            .then_some({
                self.uncertainty_weighted_mpjpe
                    + 0.20 * self.bone
                    + 0.10 * self.temporal_jerk
                    + 0.10 * self.contact
                    + 0.10 * self.uncertainty_calibration
                    + 0.05 * self.intervention
            })
    }
}
