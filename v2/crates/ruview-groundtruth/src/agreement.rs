//! Agreement as validation, not fusion (ADR-303 §3–§4).
//!
//! An [`AgreementReport`] compares an RF [`EstimateSeries`] against an
//! independent [`ReferenceSeries`] after time alignment, computes
//! modality-appropriate agreement metrics (MAE/RMSE/bias/within-tolerance for
//! continuous measurands; label-agreement for categorical ones), grades the
//! result on the ADR-293/301 evidence ladder, and feeds a per-context record
//! into the [`ruview_evidence`] ledger. Reference sensors are strictly a
//! validation plane here — this crate never returns a reference reading to an
//! estimator.

use ruview_evidence::{AccuracyMetrics, EvidenceContext, EvidenceRecord};
use ruview_ontology::EvidenceLevel as OntEvidenceLevel;
use serde::{Deserialize, Serialize};

use crate::align::{estimate_alignment, paired_at, Alignment, AlignmentConfig};
use crate::error::{check_bound, GroundTruthError};
use crate::model::{DataProvenance, Measurand, Reading};
use crate::scope::SessionScope;
use crate::series::{EstimateSeries, ReferenceSeries};
use crate::source::ReferenceSource;

/// Map the ontology's canonical evidence ladder onto the evidence ledger's
/// (structurally identical) ladder, so the report speaks the ADR-306 vocabulary
/// while still writing an ADR-304 record.
fn to_ledger_level(level: OntEvidenceLevel) -> ruview_evidence::EvidenceLevel {
    match level {
        OntEvidenceLevel::L0 => ruview_evidence::EvidenceLevel::L0,
        OntEvidenceLevel::L1 => ruview_evidence::EvidenceLevel::L1,
        OntEvidenceLevel::L2 => ruview_evidence::EvidenceLevel::L2,
        OntEvidenceLevel::L3 => ruview_evidence::EvidenceLevel::L3,
        OntEvidenceLevel::L4 => ruview_evidence::EvidenceLevel::L4,
        OntEvidenceLevel::L5 => ruview_evidence::EvidenceLevel::L5,
    }
}

/// The honesty grade of an agreement report (mirrors ADR-293/301). Fixed by the
/// data provenance, the reference, coverage, paired samples, and a reproducer —
/// never aliasable upward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceGrade {
    /// Generated input — L0 by construction.
    Synthetic,
    /// Real data, but not backed by a reference + coverage + reproducer.
    Claimed,
    /// Backed by an independent reference, sufficient coverage, paired samples,
    /// and a reproducer handle.
    Measured,
}

/// Modality-appropriate agreement metrics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "family")]
pub enum AgreementMetrics {
    /// Continuous measurand agreement (ADR-293 statistics).
    Continuous {
        /// Mean absolute error.
        mae: f64,
        /// Root-mean-square error.
        rmse: f64,
        /// Mean error (estimate − reference), i.e. bias.
        bias: f64,
        /// Fraction of pairs within the configured tolerance, `[0, 1]`.
        within_tolerance: f64,
    },
    /// Categorical / detection agreement.
    Categorical {
        /// Fraction of pairs whose labels matched, `[0, 1]`.
        agreement: f64,
        /// Number of matching pairs.
        n_agree: usize,
    },
}

/// The policy that decides an agreement report's grade and stamped level.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GradingPolicy {
    /// Minimum coverage fraction required for a MEASURED grade, `[0, 1]`.
    pub min_coverage: f64,
    /// The evidence level stamped on a `Claimed`/`Measured` record. Must not be
    /// `L0` (which is reserved for synthetic input).
    pub level: OntEvidenceLevel,
    /// The reproducer command handle. Required (non-empty) for a MEASURED
    /// grade; ignored otherwise.
    pub reproducer: Option<String>,
}

impl GradingPolicy {
    /// Construct and validate a grading policy.
    ///
    /// # Errors
    /// [`GroundTruthError::InvalidCoverage`] if `min_coverage` is outside
    /// `[0, 1]`, [`GroundTruthError::GradeLevelConflict`] if `level` is `L0`,
    /// or [`GroundTruthError::TooLong`] for an over-length reproducer.
    pub fn new(
        min_coverage: f64,
        level: OntEvidenceLevel,
        reproducer: Option<String>,
    ) -> Result<Self, GroundTruthError> {
        if !min_coverage.is_finite() || !(0.0..=1.0).contains(&min_coverage) {
            return Err(GroundTruthError::InvalidCoverage {
                value: min_coverage,
            });
        }
        if level == OntEvidenceLevel::L0 {
            return Err(GroundTruthError::GradeLevelConflict {
                reason: "L0 is reserved for synthetic input; use L1+ for a graded record",
            });
        }
        if let Some(r) = &reproducer {
            check_bound("reproducer", r)?;
        }
        Ok(Self {
            min_coverage,
            level,
            reproducer,
        })
    }
}

/// A validation-plane agreement report: how RF inference compared against an
/// independent reference, under a mandatory session scope, graded on the
/// evidence ladder and ready to feed the evidence ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgreementReport {
    /// The independent reference source.
    pub source: ReferenceSource,
    /// The measurand compared.
    pub measurand: Measurand,
    /// The estimating model version.
    pub model_version: String,
    /// Mandatory session scope — a report cannot exist without it.
    pub scope: SessionScope,
    /// The recovered time alignment.
    pub alignment: Alignment,
    /// Number of aligned pairs the metrics summarize.
    pub n_pairs: usize,
    /// Coverage: paired points over total overlap grid points, `[0, 1]`.
    pub coverage: f64,
    /// The agreement metrics.
    pub metrics: AgreementMetrics,
    /// The honesty grade.
    pub grade: EvidenceGrade,
    /// The evidence level (canonical ADR-306 ladder) stamped on emission.
    pub evidence_level: OntEvidenceLevel,
    /// The reproducer handle, when the report is MEASURED.
    pub reproducer: Option<String>,
    /// Whether the estimate data was real or synthetic.
    pub data_provenance: DataProvenance,
}

impl AgreementReport {
    /// Build an agreement report. `scope` is a required argument, so a report
    /// can never be constructed without it (ADR-303 §3).
    ///
    /// The estimate and reference must describe the same measurand. `tolerance`
    /// is the within-tolerance band for continuous measurands (ignored for
    /// categorical). Insufficient overlap is **not** an error: it yields a
    /// report with zero pairs and a non-MEASURED grade — a first-class UNKNOWN.
    ///
    /// # Errors
    /// [`GroundTruthError::MeasurandMismatch`],
    /// [`GroundTruthError::InvalidTolerance`], a configuration error from
    /// alignment, or [`GroundTruthError::GradeLevelConflict`] if the policy
    /// level is inconsistent with a non-synthetic grade.
    pub fn build(
        estimate: &EstimateSeries,
        reference: &ReferenceSeries,
        scope: SessionScope,
        align_cfg: &AlignmentConfig,
        tolerance: f64,
        policy: &GradingPolicy,
    ) -> Result<Self, GroundTruthError> {
        if estimate.measurand != reference.measurand {
            return Err(GroundTruthError::MeasurandMismatch {
                estimate: estimate.measurand.label(),
                reference: reference.measurand.label(),
            });
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(GroundTruthError::InvalidTolerance { value: tolerance });
        }

        let alignment = estimate_alignment(estimate, reference, align_cfg)?;
        let (total, pairs) = paired_at(estimate, reference, alignment.offset_ms, align_cfg);
        let n_pairs = pairs.len();
        let coverage = if total == 0 {
            0.0
        } else {
            n_pairs as f64 / total as f64
        };
        let metrics = compute_metrics(estimate.measurand, &pairs, tolerance);

        // Grade: synthetic input is always Synthetic; otherwise MEASURED only
        // with an independent reference, coverage, paired samples, and a
        // reproducer — else Claimed.
        let grade = match estimate.provenance {
            DataProvenance::Synthetic => EvidenceGrade::Synthetic,
            DataProvenance::Real => {
                let reproducer_ok = policy
                    .reproducer
                    .as_deref()
                    .is_some_and(|r| !r.is_empty());
                if reference.source.modality.is_independent_reference()
                    && n_pairs > 0
                    && coverage >= policy.min_coverage
                    && reproducer_ok
                {
                    EvidenceGrade::Measured
                } else {
                    EvidenceGrade::Claimed
                }
            }
        };

        let (evidence_level, reproducer) = match grade {
            EvidenceGrade::Synthetic => (OntEvidenceLevel::L0, None),
            EvidenceGrade::Claimed => (policy.level, None),
            EvidenceGrade::Measured => (policy.level, policy.reproducer.clone()),
        };

        Ok(Self {
            source: reference.source.clone(),
            measurand: estimate.measurand,
            model_version: estimate.model_version.clone(),
            scope,
            alignment,
            n_pairs,
            coverage,
            metrics,
            grade,
            evidence_level,
            reproducer,
            data_provenance: estimate.provenance,
        })
    }

    /// Emit this report as an evidence-ledger record, keyed by `context`, with
    /// caller-supplied per-context [`AccuracyMetrics`]. The record's provenance
    /// class and level follow the report's grade: `Synthetic → L0 synthetic`,
    /// `Claimed → claimed`, `Measured → measured` (with the reproducer). The
    /// evidence crate enforces the honesty invariants; failures surface as
    /// [`GroundTruthError::Evidence`].
    ///
    /// The agreement statistics (MAE/RMSE/coverage/label-agreement) live on the
    /// report for the benchmark (ADR-317); the ledger record carries the
    /// per-context accuracy metrics with the correct, non-upgradable grade.
    ///
    /// # Errors
    /// [`GroundTruthError::GradeLevelConflict`] if a MEASURED report lacks its
    /// reproducer, or [`GroundTruthError::Evidence`] from the ledger boundary.
    pub fn to_evidence_record(
        &self,
        context: EvidenceContext,
        metrics: AccuracyMetrics,
        timestamp_ns: u64,
    ) -> Result<EvidenceRecord, GroundTruthError> {
        let level = to_ledger_level(self.evidence_level);
        let record = match self.grade {
            EvidenceGrade::Synthetic => {
                EvidenceRecord::synthetic(context, metrics, timestamp_ns)?
            }
            EvidenceGrade::Claimed => {
                EvidenceRecord::claimed(context, metrics, level, timestamp_ns)?
            }
            EvidenceGrade::Measured => {
                let reproducer = self.reproducer.as_deref().ok_or(
                    GroundTruthError::GradeLevelConflict {
                        reason: "measured report is missing its reproducer handle",
                    },
                )?;
                EvidenceRecord::measured(context, metrics, level, reproducer, timestamp_ns)?
            }
        };
        Ok(record)
    }
}

/// Compute agreement metrics for the measurand's family from aligned pairs.
fn compute_metrics(
    measurand: Measurand,
    pairs: &[(Reading, Reading)],
    tolerance: f64,
) -> AgreementMetrics {
    if measurand.is_continuous() {
        let n = pairs.len();
        if n == 0 {
            return AgreementMetrics::Continuous {
                mae: 0.0,
                rmse: 0.0,
                bias: 0.0,
                within_tolerance: 0.0,
            };
        }
        let mut sum_abs = 0.0;
        let mut sum_sq = 0.0;
        let mut sum_err = 0.0;
        let mut within = 0usize;
        for (e, r) in pairs {
            // Both are scalars for a continuous measurand (validated at ingest).
            let ev = e.as_scalar().unwrap_or(0.0);
            let rv = r.as_scalar().unwrap_or(0.0);
            let err = ev - rv;
            sum_abs += err.abs();
            sum_sq += err * err;
            sum_err += err;
            if err.abs() <= tolerance {
                within += 1;
            }
        }
        let nf = n as f64;
        AgreementMetrics::Continuous {
            mae: sum_abs / nf,
            rmse: (sum_sq / nf).sqrt(),
            bias: sum_err / nf,
            within_tolerance: within as f64 / nf,
        }
    } else {
        let n = pairs.len();
        let n_agree = pairs
            .iter()
            .filter(|(e, r)| e.as_label() == r.as_label())
            .count();
        let agreement = if n == 0 {
            0.0
        } else {
            n_agree as f64 / n as f64
        };
        AgreementMetrics::Categorical { agreement, n_agree }
    }
}
