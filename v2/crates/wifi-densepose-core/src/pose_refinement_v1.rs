//! Additive physics assessment and correction contract (ADR-323).

#![allow(missing_docs)]

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::string::String;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Probability;

/// Runtime rollout mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum PhysicsMode {
    Off,
    Audit,
    ShadowCorrect,
    OptInCorrect,
    DefaultCorrect,
}

impl PhysicsMode {
    /// Whether a validated candidate may be selected for downstream use.
    #[must_use]
    pub const fn selects_correction(self) -> bool {
        matches!(self, Self::OptInCorrect | Self::DefaultCorrect)
    }
}

/// Outcome of processing exactly one accepted frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum RefinementDisposition {
    Bypassed,
    Audited2d,
    Audited,
    Shadowed,
    Corrected,
    Abstained,
    Rejected,
}

/// Typed fail-to-raw reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AbstentionReason {
    ModeOff,
    UnsupportedSchema,
    HashMismatch,
    InvalidNumber,
    InvalidCovariance,
    StaleInput,
    CoordinateFrameMismatch,
    CalibrationUnavailable,
    UncertaintyUncalibrated,
    OodUnknown,
    SourceUnauthenticated,
    ReplayProtectionUnavailable,
    CorrectionNotAuthorized,
    ReplayRejected,
    NonMonotonicInput,
    TooFewKnownJoints,
    CorrectionTooLarge,
    DeadlineExceeded,
    TrackCapacity,
    InternalError,
}

/// Constraint residuals. Values are metric or normalized as named.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConstraintResiduals {
    pub bone_m: f32,
    pub joint_limit_rad: f32,
    pub velocity_mps: f32,
    pub acceleration_mps2: f32,
    pub temporal_jerk: f32,
    pub floor_penetration_m: f32,
    pub contact_m: f32,
    pub collision_m: f32,
    pub normalized_total: f32,
}

/// Summary of how much the candidate differs from the observation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct InterventionSummary {
    pub max_joint_correction_m: f32,
    pub root_correction_m: f32,
    pub corrected_joint_count: u8,
    pub solver_iterations: u8,
    pub elapsed_us: u64,
}

/// Optional articulated-body audit scalars.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DynamicsResiduals {
    pub stable: bool,
    pub segment_count: u8,
    pub joint_count: u8,
    pub contact_count: u16,
    pub substeps: u8,
    pub tracking_error_m: f32,
    pub joint_anchor_error_m: f32,
    pub floor_penetration_m: f32,
    pub control_effort: f32,
}

/// Contact state is never promoted to measured without sensor provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum ContactState {
    Hypothesis,
    Measured,
    Unknown,
}

/// One foot-contact assessment.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContactHypothesis {
    pub state: ContactState,
    pub probability: Probability,
}

impl Default for ContactHypothesis {
    fn default() -> Self {
        Self {
            state: ContactState::Unknown,
            probability: Probability::ZERO,
        }
    }
}

/// Version and artifact lineage for an assessment.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PhysicsProvenance {
    pub engine: String,
    pub engine_version: String,
    pub config_hash: [u8; 32],
    pub rf_model_hash: [u8; 32],
    pub calibration_id: String,
    pub learned_artifact_hash: Option<[u8; 32]>,
}

/// Physics result returned for every accepted input frame.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PoseRefinementV1 {
    pub schema_version: u16,
    pub raw_observation_hash: [u8; 32],
    pub mode: PhysicsMode,
    pub disposition: RefinementDisposition,
    pub selected: bool,
    pub refined_joints_m: Option<[[f32; 3]; 17]>,
    pub physics_confidence: Probability,
    pub effective_confidence: Probability,
    pub intervention: InterventionSummary,
    /// Residuals of the immutable observer output.
    pub residuals: ConstraintResiduals,
    /// Residuals of the bounded candidate, when a candidate was computed.
    pub refined_residuals: Option<ConstraintResiduals>,
    /// Present only when the separately gated Rapier auditor ran.
    pub dynamics: Option<DynamicsResiduals>,
    pub contact_hypotheses: [ContactHypothesis; 2],
    pub provenance: PhysicsProvenance,
    pub reason: Option<AbstentionReason>,
    /// Canonical BLAKE3 hash of every preceding result field.
    pub canonical_hash: [u8; 32],
}

impl PoseRefinementV1 {
    /// Compute a deterministic content hash excluding `canonical_hash` itself.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn compute_canonical_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"ruview.pose-refinement-v1\0");
        h.update(&self.schema_version.to_le_bytes());
        h.update(&self.raw_observation_hash);
        h.update(&[
            self.mode as u8,
            self.disposition as u8,
            self.selected.into(),
        ]);
        match self.refined_joints_m {
            Some(joints) => {
                h.update(&[1]);
                for joint in joints {
                    hash_f32s(&mut h, &joint);
                }
            }
            None => {
                h.update(&[0]);
            }
        }
        hash_f32s(
            &mut h,
            &[
                self.physics_confidence.get(),
                self.effective_confidence.get(),
            ],
        );
        hash_f32s(
            &mut h,
            &[
                self.intervention.max_joint_correction_m,
                self.intervention.root_correction_m,
            ],
        );
        h.update(&[
            self.intervention.corrected_joint_count,
            self.intervention.solver_iterations,
        ]);
        h.update(&self.intervention.elapsed_us.to_le_bytes());
        hash_residuals(&mut h, self.residuals);
        match self.refined_residuals {
            Some(residuals) => {
                h.update(&[1]);
                hash_residuals(&mut h, residuals);
            }
            None => {
                h.update(&[0]);
            }
        }
        match self.dynamics {
            Some(dynamics) => {
                h.update(&[1]);
                h.update(&[
                    dynamics.stable.into(),
                    dynamics.segment_count,
                    dynamics.joint_count,
                    dynamics.substeps,
                ]);
                h.update(&dynamics.contact_count.to_le_bytes());
                hash_f32s(
                    &mut h,
                    &[
                        dynamics.tracking_error_m,
                        dynamics.joint_anchor_error_m,
                        dynamics.floor_penetration_m,
                        dynamics.control_effort,
                    ],
                );
            }
            None => {
                h.update(&[0]);
            }
        }
        for contact in self.contact_hypotheses {
            h.update(&[contact.state as u8]);
            hash_f32s(&mut h, &[contact.probability.get()]);
        }
        hash_str(&mut h, &self.provenance.engine);
        hash_str(&mut h, &self.provenance.engine_version);
        h.update(&self.provenance.config_hash);
        h.update(&self.provenance.rf_model_hash);
        hash_str(&mut h, &self.provenance.calibration_id);
        match self.provenance.learned_artifact_hash {
            Some(hash) => {
                h.update(&[1]);
                h.update(&hash);
            }
            None => {
                h.update(&[0]);
            }
        }
        match self.reason {
            Some(reason) => h.update(&[1, reason as u8]),
            None => h.update(&[0]),
        };
        *h.finalize().as_bytes()
    }

    /// Seal the result after all fields have been populated.
    pub fn seal(&mut self) {
        self.canonical_hash = self.compute_canonical_hash();
    }
}

fn hash_residuals(hasher: &mut blake3::Hasher, residuals: ConstraintResiduals) {
    hash_f32s(
        hasher,
        &[
            residuals.bone_m,
            residuals.joint_limit_rad,
            residuals.velocity_mps,
            residuals.acceleration_mps2,
            residuals.temporal_jerk,
            residuals.floor_penetration_m,
            residuals.contact_m,
            residuals.collision_m,
            residuals.normalized_total,
        ],
    );
}

fn hash_f32s(hasher: &mut blake3::Hasher, values: &[f32]) {
    for value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
