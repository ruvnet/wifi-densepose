//! Modality-agnostic measurands and readings (ADR-303 §1).
//!
//! ADR-293 built ground truth for a single measurand family (heart rate,
//! breathing rate). This module generalizes the *value* being compared to any
//! phenomenon RuView senses — continuous scalars (vitals, count, range) and
//! categorical labels (presence, activity, posture) — so the same alignment
//! and agreement machinery applies to every modality.

use serde::{Deserialize, Serialize};

/// Whether a measurand is compared as a continuous scalar or a discrete label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingKind {
    /// A continuous numeric value (heart rate, range, count).
    Scalar,
    /// A discrete class label (presence, activity, posture).
    Label,
}

/// A phenomenon compared against an independent reference. This is the
/// modality-agnostic generalization of ADR-293's per-device measurand: the set
/// is deliberately small and closed so the agreement math per family stays
/// honest (pose keypoint PCK, which needs the ADR-291 mean-pose baseline and a
/// leakage-free split, is intentionally out of scope for this crate).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Measurand {
    /// Someone present in the space (categorical: e.g. `"present"`/`"absent"`).
    Presence,
    /// The activity a subject is performing (categorical label).
    Activity,
    /// A subject's posture (categorical label).
    Posture,
    /// Heart rate, beats per minute (continuous).
    HeartRateBpm,
    /// Breathing rate, breaths per minute (continuous).
    BreathingRateBrpm,
    /// The number of people present (continuous count).
    PersonCount,
    /// Range / localization distance, metres (continuous).
    RangeMeters,
}

impl Measurand {
    /// The reading family this measurand is compared in.
    #[must_use]
    pub const fn kind(self) -> ReadingKind {
        match self {
            Measurand::Presence | Measurand::Activity | Measurand::Posture => ReadingKind::Label,
            Measurand::HeartRateBpm
            | Measurand::BreathingRateBrpm
            | Measurand::PersonCount
            | Measurand::RangeMeters => ReadingKind::Scalar,
        }
    }

    /// Whether this measurand is compared as a continuous scalar.
    #[must_use]
    pub const fn is_continuous(self) -> bool {
        matches!(self.kind(), ReadingKind::Scalar)
    }

    /// A stable, human-readable tag used in error messages.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Measurand::Presence => "presence",
            Measurand::Activity => "activity",
            Measurand::Posture => "posture",
            Measurand::HeartRateBpm => "heart_rate_bpm",
            Measurand::BreathingRateBrpm => "breathing_rate_brpm",
            Measurand::PersonCount => "person_count",
            Measurand::RangeMeters => "range_meters",
        }
    }
}

/// A single reading value: either a continuous scalar or a discrete label.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reading {
    /// A continuous numeric value.
    Scalar(f64),
    /// A discrete class label.
    Label(String),
}

impl Reading {
    /// The family of this reading.
    #[must_use]
    pub const fn kind(&self) -> ReadingKind {
        match self {
            Reading::Scalar(_) => ReadingKind::Scalar,
            Reading::Label(_) => ReadingKind::Label,
        }
    }

    /// Borrow the scalar value, if this is a scalar reading.
    #[must_use]
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Reading::Scalar(v) => Some(*v),
            Reading::Label(_) => None,
        }
    }

    /// Borrow the label, if this is a label reading.
    #[must_use]
    pub fn as_label(&self) -> Option<&str> {
        match self {
            Reading::Label(s) => Some(s.as_str()),
            Reading::Scalar(_) => None,
        }
    }
}

/// Whether the compared data is real inference/measurement or a generated
/// (SYNTHETIC/L0) fixture. This is what forces an [`crate::EvidenceGrade`] to
/// `Synthetic`; it is never inferred, it is declared by the producer (mirrors
/// ADR-304's synthetic-is-L0-by-construction rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataProvenance {
    /// Real inference / real measurement.
    Real,
    /// Generated / simulated input — grades as SYNTHETIC (L0).
    Synthetic,
}
