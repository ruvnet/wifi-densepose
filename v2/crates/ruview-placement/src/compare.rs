//! Post-install loop: predicted vs. measured observability → adjustments
//! (ADR-305 §3).
//!
//! **This crate never measures.** The `measured` observability values are
//! supplied by the caller — the ADR-299 runtime observability signal from freshly
//! enrolled, calibrated sensors — and this module only *compares* them against the
//! optimizer's own SYNTHETIC/L0 prediction. The predicted side stays labelled
//! `L0`; a `MEASURED` statement, if any, belongs to the caller's measured input
//! together with its reproducer (CLAUDE.md hardware rule). Where measurement
//! disagrees with prediction, the module recommends an adjustment and a coarse
//! twin-parameter residual to feed back into the ADR-312 twin. Following ADR-297
//! rule 1, a target with no measured value yields a first-class UNKNOWN verdict,
//! never an error.

use serde::{Deserialize, Serialize};

use ruview_ontology::{Container, EvidenceLevel};

use crate::coverage::{Observability, PlacementScore};

/// Default tolerance (in observability units) inside which predicted and measured
/// are treated as matching.
pub const DEFAULT_COMPARE_TOLERANCE: f64 = 0.15;

/// Coarse dB of implied effective attenuation per unit of observability shortfall,
/// used only to suggest a twin-parameter residual to feed back into ADR-312. A
/// rough SYNTHETIC heuristic, not a calibrated figure.
const RESIDUAL_DB_PER_UNIT: f64 = 30.0;

/// A single caller-supplied measured observability for a target.
///
/// The value's evidence level is the *caller's* to assert (with a reproducer, per
/// the hardware rule); this crate carries it through unchanged.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasuredTarget {
    /// The target region measured.
    pub target: Container,
    /// The caller-supplied observed observability score, `[0, 1]`.
    pub observed_score: f64,
}

/// A set of caller-supplied measured observability values.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct MeasuredObservability {
    /// Per-target measurements.
    pub per_target: Vec<MeasuredTarget>,
}

impl MeasuredObservability {
    /// An empty measurement set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one measured target and return `self` for chaining.
    #[must_use]
    pub fn with(mut self, target: Container, observed_score: f64) -> Self {
        self.per_target.push(MeasuredTarget { target, observed_score });
        self
    }

    /// The measured score for a target, if present.
    #[must_use]
    fn score_for(&self, target: &Container) -> Option<f64> {
        self.per_target
            .iter()
            .find(|m| &m.target == target)
            .map(|m| m.observed_score)
    }
}

/// The verdict comparing predicted and measured observability for one target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareVerdict {
    /// Measured is within tolerance of predicted.
    Match,
    /// Measured is materially below predicted (reality is worse than the model).
    Underperforming,
    /// Measured is materially above predicted (the model was pessimistic).
    Overperforming,
    /// Cannot compare (no measured value, or prediction was UNKNOWN). First-class
    /// UNKNOWN (ADR-297 rule 1).
    Unknown,
}

/// The recommended action for a target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdjustmentAction {
    /// Nothing to change; prediction and measurement agree.
    NoActionNeeded,
    /// Re-aim / reposition an existing node (cheap first move — favoured when the
    /// prediction itself was uncertain).
    ReAim,
    /// Move a node materially, or accept a larger geometry change.
    MoveNode,
    /// Add another node to recover the objective.
    AddNode,
    /// Not enough information to recommend anything (measurement missing/UNKNOWN).
    InsufficientData,
}

/// Direction of a suggested twin-parameter residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    /// Reality attenuates more than the twin modelled (measured < predicted).
    EffectiveAttenuationHigher,
    /// Reality attenuates less than the twin modelled (measured > predicted).
    EffectiveAttenuationLower,
}

/// A coarse twin-parameter residual to feed back into the ADR-312 twin.
///
/// **SYNTHETIC / L0.** A rough model-improvement hint, not a calibrated value.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwinResidual {
    /// Which way the twin's effective attenuation should move.
    pub kind: ResidualKind,
    /// Rough magnitude of the suggested effective-attenuation change, dB.
    pub magnitude_db: f64,
}

/// A recommended adjustment for one target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Adjustment {
    /// The target this adjustment concerns.
    pub target: Container,
    /// The recommended action.
    pub action: AdjustmentAction,
    /// Human-readable rationale.
    pub rationale: String,
    /// Coarse twin-parameter residual to feed back into ADR-312, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub twin_residual: Option<TwinResidual>,
}

/// The predicted-vs-measured comparison for one target.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetComparison {
    /// The target region.
    pub target: Container,
    /// The optimizer's predicted observability (SYNTHETIC/L0).
    pub predicted: Observability,
    /// The caller-supplied measured score, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<f64>,
    /// `measured - predicted_score`, when both are available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<f64>,
    /// The verdict.
    pub verdict: CompareVerdict,
}

/// The full post-install adjustment report.
///
/// The predicted side is `L0` (SYNTHETIC); the measured side is caller-supplied.
/// This report is a set of recommendations, never a sensing claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdjustmentReport {
    /// Per-target comparisons.
    pub comparisons: Vec<TargetComparison>,
    /// Recommended adjustments (targets needing action).
    pub adjustments: Vec<Adjustment>,
    /// Evidence level of the *predicted* side. Always `L0` (SYNTHETIC).
    pub predicted_evidence_level: EvidenceLevel,
}

impl AdjustmentReport {
    /// True when at least one target needs a corrective action.
    #[must_use]
    pub fn needs_adjustment(&self) -> bool {
        self.adjustments.iter().any(|a| {
            !matches!(
                a.action,
                AdjustmentAction::NoActionNeeded | AdjustmentAction::InsufficientData
            )
        })
    }
}

/// Compare predicted against measured observability using the default tolerance.
#[must_use]
pub fn compare_post_install(
    predicted: &PlacementScore,
    measured: &MeasuredObservability,
) -> AdjustmentReport {
    compare_post_install_with_tolerance(predicted, measured, DEFAULT_COMPARE_TOLERANCE)
}

/// Compare predicted against measured observability with an explicit tolerance.
///
/// Deterministic and never panics. A target whose prediction is UNKNOWN, or that
/// has no measured value, yields a [`CompareVerdict::Unknown`] and an
/// [`AdjustmentAction::InsufficientData`] recommendation (ADR-297 rule 1).
#[must_use]
pub fn compare_post_install_with_tolerance(
    predicted: &PlacementScore,
    measured: &MeasuredObservability,
    tolerance: f64,
) -> AdjustmentReport {
    let tol = if tolerance.is_finite() && tolerance >= 0.0 {
        tolerance
    } else {
        DEFAULT_COMPARE_TOLERANCE
    };

    let mut comparisons = Vec::with_capacity(predicted.per_target.len());
    let mut adjustments = Vec::new();

    for cov in &predicted.per_target {
        let target = cov.target.clone();
        let measured_score = measured.score_for(&target).filter(|v| v.is_finite());

        let (predicted_score, predicted_uncertainty) = match cov.observability.known() {
            Some((s, u)) => (Some(s), u),
            None => (None, 1.0),
        };

        match (predicted_score, measured_score) {
            (Some(pred), Some(meas)) => {
                let delta = meas - pred;
                let (verdict, action, residual) = classify(delta, tol, predicted_uncertainty);
                let rationale = rationale_for(action, delta, pred, meas);
                comparisons.push(TargetComparison {
                    target: target.clone(),
                    predicted: cov.observability,
                    measured: Some(meas),
                    delta: Some(delta),
                    verdict,
                });
                adjustments.push(Adjustment { target, action, rationale, twin_residual: residual });
            }
            _ => {
                // Missing measurement or UNKNOWN prediction: first-class UNKNOWN.
                comparisons.push(TargetComparison {
                    target: target.clone(),
                    predicted: cov.observability,
                    measured: measured_score,
                    delta: None,
                    verdict: CompareVerdict::Unknown,
                });
                adjustments.push(Adjustment {
                    target,
                    action: AdjustmentAction::InsufficientData,
                    rationale: "no measured observability to compare against prediction".to_string(),
                    twin_residual: None,
                });
            }
        }
    }

    AdjustmentReport {
        comparisons,
        adjustments,
        predicted_evidence_level: EvidenceLevel::L0,
    }
}

/// Classify a predicted-vs-measured delta into a verdict, action, and residual.
fn classify(
    delta: f64,
    tolerance: f64,
    predicted_uncertainty: f64,
) -> (CompareVerdict, AdjustmentAction, Option<TwinResidual>) {
    if delta < -tolerance {
        // Reality worse than modelled: recommend a corrective move.
        // Favour the cheap re-aim when the prediction itself was uncertain.
        let action = if predicted_uncertainty > 0.5 {
            AdjustmentAction::ReAim
        } else if delta < -2.0 * tolerance {
            AdjustmentAction::AddNode
        } else {
            AdjustmentAction::MoveNode
        };
        let residual = TwinResidual {
            kind: ResidualKind::EffectiveAttenuationHigher,
            magnitude_db: (delta.abs() * RESIDUAL_DB_PER_UNIT).min(120.0),
        };
        (CompareVerdict::Underperforming, action, Some(residual))
    } else if delta > tolerance {
        // Reality better than modelled: no action, but the twin was pessimistic.
        let residual = TwinResidual {
            kind: ResidualKind::EffectiveAttenuationLower,
            magnitude_db: (delta.abs() * RESIDUAL_DB_PER_UNIT).min(120.0),
        };
        (CompareVerdict::Overperforming, AdjustmentAction::NoActionNeeded, Some(residual))
    } else {
        (CompareVerdict::Match, AdjustmentAction::NoActionNeeded, None)
    }
}

/// Build a human-readable rationale string.
fn rationale_for(action: AdjustmentAction, delta: f64, predicted: f64, measured: f64) -> String {
    match action {
        AdjustmentAction::NoActionNeeded => format!(
            "measured {measured:.2} matches predicted {predicted:.2} within tolerance"
        ),
        AdjustmentAction::ReAim => format!(
            "measured {measured:.2} below predicted {predicted:.2} (Δ {delta:.2}); prediction was \
             uncertain, so re-aim an existing node first"
        ),
        AdjustmentAction::MoveNode => format!(
            "measured {measured:.2} below predicted {predicted:.2} (Δ {delta:.2}); move a node to \
             recover coverage"
        ),
        AdjustmentAction::AddNode => format!(
            "measured {measured:.2} far below predicted {predicted:.2} (Δ {delta:.2}); add a node \
             to recover the objective"
        ),
        AdjustmentAction::InsufficientData => {
            "no measured observability to compare against prediction".to_string()
        }
    }
}
