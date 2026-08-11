//! Mandatory session scope (ADR-303 §3, mirroring ADR-293).
//!
//! An agreement report without scope cannot be constructed: WiFi-sensing
//! numbers without stated scope (subject count, motion, line-of-sight,
//! distance) are systematically misleading (ADR-293 Context). [`SessionScope`]
//! is a required argument to [`crate::AgreementReport::build`], so the type
//! system enforces the rule.

use serde::{Deserialize, Serialize};

use crate::error::GroundTruthError;

/// The largest subject count accepted, bounding untrusted input.
pub const MAX_SUBJECTS: u16 = 4096;

/// Whether subjects were static or moving during the session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionState {
    /// Subject(s) static / at rest.
    Static,
    /// Subject(s) moving.
    Moving,
    /// A mix of static and moving intervals.
    Mixed,
}

/// The propagation condition between sensor and subject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LineOfSight {
    /// Line-of-sight.
    Los,
    /// Non-line-of-sight (obstructed, same room).
    Nlos,
    /// Through-wall.
    ThroughWall,
}

/// A coarse distance band between sensor and subject.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceBand {
    /// Near (roughly < 2 m).
    Near,
    /// Mid (roughly 2–5 m).
    Mid,
    /// Far (roughly > 5 m).
    Far,
}

/// Mandatory metadata attached to every [`crate::AgreementReport`]. A report
/// cannot exist without it, so an agreement number always states the conditions
/// it was measured under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionScope {
    /// Number of subjects present (0 is valid for an empty-room session).
    pub subject_count: u16,
    /// Motion state during the session.
    pub motion: MotionState,
    /// Line-of-sight condition.
    pub line_of_sight: LineOfSight,
    /// Distance band.
    pub distance: DistanceBand,
}

impl SessionScope {
    /// Construct a validated session scope. `subject_count` is bounded to
    /// [`MAX_SUBJECTS`] so untrusted metadata cannot claim an absurd count.
    ///
    /// # Errors
    /// [`GroundTruthError::SubjectCountTooLarge`] if `subject_count` exceeds
    /// [`MAX_SUBJECTS`].
    pub fn new(
        subject_count: u16,
        motion: MotionState,
        line_of_sight: LineOfSight,
        distance: DistanceBand,
    ) -> Result<Self, GroundTruthError> {
        if subject_count > MAX_SUBJECTS {
            return Err(GroundTruthError::SubjectCountTooLarge {
                count: u32::from(subject_count),
                max: u32::from(MAX_SUBJECTS),
            });
        }
        Ok(Self {
            subject_count,
            motion,
            line_of_sight,
            distance,
        })
    }
}
