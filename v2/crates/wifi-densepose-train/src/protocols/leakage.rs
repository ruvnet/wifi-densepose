//! Structural leakage guards, mean-pose baseline, and evidence-graded
//! evaluation reports (ADR-291 §3).
//!
//! Three enforcement points, all `Err`-on-failure (never a warning):
//!
//! 1. [`LeakageAudit`] verifies a proposed train/test split structurally:
//!    subject-disjointness and environment/orientation-disjointness **where
//!    the protocol claims them**, and — unconditionally — that no two windows
//!    cut from the same continuous recording straddle the boundary.
//! 2. [`MeanPoseBaseline`] is fitted from the *training* split only; PCK /
//!    MPJPE numbers are meaningless without it (CLAUDE.md: pose PCK requires
//!    the mean-pose baseline).
//! 3. [`EvaluationReport`] pairs the model metric with the baseline metric
//!    and carries an [`EvidenceGrade`]; `MEASURED` cannot be constructed
//!    without an embedded reproducer command string.

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::error::ProtocolError;
use crate::protocols::{SampleMeta, SplitProtocol};

// ---------------------------------------------------------------------------
// LeakageAudit
// ---------------------------------------------------------------------------

/// Disjointness properties a protocol claims; the audit verifies each claimed
/// one. Obtained from [`SplitProtocol::claims`], or constructed directly for
/// custom protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakageClaims {
    /// Train and test must share no subject.
    pub subject_disjoint: bool,
    /// Train and test must share no environment/room.
    pub environment_disjoint: bool,
    /// Train and test must share no orientation.
    pub orientation_disjoint: bool,
}

/// Summary returned by a **passing** audit — counts for reporting, no claim
/// stronger than what was structurally checked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakageAuditPass {
    /// Claims that were verified.
    pub claims: LeakageClaims,
    /// Number of training windows.
    pub train_windows: usize,
    /// Number of test windows.
    pub test_windows: usize,
    /// Distinct subjects in train.
    pub train_subjects: usize,
    /// Distinct subjects in test.
    pub test_subjects: usize,
    /// Distinct continuous recordings in train.
    pub train_recordings: usize,
    /// Distinct continuous recordings in test.
    pub test_recordings: usize,
}

/// Structural train/test-split auditor (ADR-291 §3).
///
/// A failed audit is an [`Err`], not a warning: leaky splits must be unusable
/// for reporting, not merely frowned upon.
#[derive(Debug, Clone, Copy)]
pub struct LeakageAudit {
    claims: LeakageClaims,
}

impl LeakageAudit {
    /// Auditor for an explicit set of claims.
    #[must_use]
    pub fn new(claims: LeakageClaims) -> Self {
        LeakageAudit { claims }
    }

    /// Auditor for the claims a standard protocol makes.
    #[must_use]
    pub fn for_protocol(protocol: SplitProtocol) -> Self {
        LeakageAudit {
            claims: protocol.claims(),
        }
    }

    /// Verify a proposed split.
    ///
    /// Checks, in order:
    /// 1. both partitions are non-empty;
    /// 2. no continuous recording has windows on both sides (unconditional —
    ///    overlapping windows of one recording are near-duplicates);
    /// 3. subject-disjointness, when claimed;
    /// 4. environment-disjointness, when claimed;
    /// 5. orientation-disjointness, when claimed.
    ///
    /// # Errors
    ///
    /// The [`ProtocolError`] variant describing the **first** violation found.
    pub fn audit(
        &self,
        train: &[SampleMeta],
        test: &[SampleMeta],
    ) -> Result<LeakageAuditPass, ProtocolError> {
        if train.is_empty() {
            return Err(ProtocolError::EmptyPartition { side: "train" });
        }
        if test.is_empty() {
            return Err(ProtocolError::EmptyPartition { side: "test" });
        }

        // (2) Recording windows must never cross the boundary.
        let train_recordings: BTreeSet<u64> = train.iter().map(|m| m.recording_id).collect();
        let test_recordings: BTreeSet<u64> = test.iter().map(|m| m.recording_id).collect();
        if let Some(&recording_id) = train_recordings.intersection(&test_recordings).next() {
            return Err(ProtocolError::RecordingCrossesSplit { recording_id });
        }

        // (3–5) Claimed disjointness.
        let train_subjects: BTreeSet<u32> = train.iter().map(|m| m.subject_id).collect();
        let test_subjects: BTreeSet<u32> = test.iter().map(|m| m.subject_id).collect();
        if self.claims.subject_disjoint {
            if let Some(&subject_id) = train_subjects.intersection(&test_subjects).next() {
                return Err(ProtocolError::SubjectOverlap { subject_id });
            }
        }
        if self.claims.environment_disjoint {
            let train_envs: BTreeSet<u32> = train.iter().map(|m| m.environment_id).collect();
            let test_envs: BTreeSet<u32> = test.iter().map(|m| m.environment_id).collect();
            if let Some(&environment_id) = train_envs.intersection(&test_envs).next() {
                return Err(ProtocolError::EnvironmentOverlap { environment_id });
            }
        }
        if self.claims.orientation_disjoint {
            let train_orients: BTreeSet<u32> = train.iter().map(|m| m.orientation_id).collect();
            let test_orients: BTreeSet<u32> = test.iter().map(|m| m.orientation_id).collect();
            if let Some(&orientation_id) = train_orients.intersection(&test_orients).next() {
                return Err(ProtocolError::OrientationOverlap { orientation_id });
            }
        }

        Ok(LeakageAuditPass {
            claims: self.claims,
            train_windows: train.len(),
            test_windows: test.len(),
            train_subjects: train_subjects.len(),
            test_subjects: test_subjects.len(),
            train_recordings: train_recordings.len(),
            test_recordings: test_recordings.len(),
        })
    }
}

// ---------------------------------------------------------------------------
// MeanPoseBaseline
// ---------------------------------------------------------------------------

/// The mean-pose baseline: predicts the per-joint mean of the **training**
/// poses for every test sample (CLAUDE.md: pose PCK requires this baseline —
/// a model must beat "always predict the average pose" before any number
/// means anything).
#[derive(Debug, Clone, PartialEq)]
pub struct MeanPoseBaseline {
    mean_pose: Array2<f32>,
    num_train_poses: usize,
}

impl MeanPoseBaseline {
    /// Fit the baseline from training-split poses only. Each pose is
    /// `[num_joints, 2]` (normalised x, y); all poses must share one shape.
    ///
    /// # Errors
    ///
    /// - [`ProtocolError::EmptyTrainingPoses`] when `train_poses` is empty.
    /// - [`ProtocolError::PoseShapeMismatch`] when poses disagree in shape.
    pub fn fit(train_poses: &[Array2<f32>]) -> Result<Self, ProtocolError> {
        let first = train_poses.first().ok_or(ProtocolError::EmptyTrainingPoses)?;
        let shape = first.dim();

        let mut mean_pose = Array2::<f32>::zeros(shape);
        for pose in train_poses {
            if pose.dim() != shape {
                return Err(ProtocolError::PoseShapeMismatch {
                    expected: vec![shape.0, shape.1],
                    actual: pose.shape().to_vec(),
                });
            }
            mean_pose += pose;
        }
        mean_pose /= train_poses.len() as f32;

        Ok(MeanPoseBaseline {
            mean_pose,
            num_train_poses: train_poses.len(),
        })
    }

    /// The fitted mean pose, `[num_joints, 2]`.
    #[must_use]
    pub fn mean_pose(&self) -> &Array2<f32> {
        &self.mean_pose
    }

    /// Number of training poses the baseline was fitted on.
    #[must_use]
    pub fn num_train_poses(&self) -> usize {
        self.num_train_poses
    }

    /// Mean per-joint position error (MPJPE) of the baseline over test-split
    /// poses: the mean Euclidean distance between each test joint and the
    /// corresponding mean-pose joint.
    ///
    /// # Errors
    ///
    /// - [`ProtocolError::EmptyTrainingPoses`] when `test_poses` is empty
    ///   (nothing to evaluate).
    /// - [`ProtocolError::PoseShapeMismatch`] when a test pose does not match
    ///   the fitted shape.
    pub fn mpjpe(&self, test_poses: &[Array2<f32>]) -> Result<f64, ProtocolError> {
        if test_poses.is_empty() {
            return Err(ProtocolError::EmptyTrainingPoses);
        }
        let shape = self.mean_pose.dim();
        let mut total = 0.0f64;
        let mut joints = 0usize;
        for pose in test_poses {
            if pose.dim() != shape {
                return Err(ProtocolError::PoseShapeMismatch {
                    expected: vec![shape.0, shape.1],
                    actual: pose.shape().to_vec(),
                });
            }
            for j in 0..shape.0 {
                let dx = (pose[[j, 0]] - self.mean_pose[[j, 0]]) as f64;
                let dy = (pose[[j, 1]] - self.mean_pose[[j, 1]]) as f64;
                total += (dx * dx + dy * dy).sqrt();
                joints += 1;
            }
        }
        Ok(total / joints as f64)
    }

    /// Baseline PCK@`threshold` over test poses: fraction of joints whose
    /// distance to the mean-pose joint is `< threshold` (same units as the
    /// pose coordinates).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::mpjpe`].
    pub fn pck_at(
        &self,
        test_poses: &[Array2<f32>],
        threshold: f32,
    ) -> Result<f64, ProtocolError> {
        if test_poses.is_empty() {
            return Err(ProtocolError::EmptyTrainingPoses);
        }
        let shape = self.mean_pose.dim();
        let mut correct = 0usize;
        let mut joints = 0usize;
        for pose in test_poses {
            if pose.dim() != shape {
                return Err(ProtocolError::PoseShapeMismatch {
                    expected: vec![shape.0, shape.1],
                    actual: pose.shape().to_vec(),
                });
            }
            for j in 0..shape.0 {
                let dx = pose[[j, 0]] - self.mean_pose[[j, 0]];
                let dy = pose[[j, 1]] - self.mean_pose[[j, 1]];
                if (dx * dx + dy * dy).sqrt() < threshold {
                    correct += 1;
                }
                joints += 1;
            }
        }
        Ok(correct as f64 / joints as f64)
    }
}

// ---------------------------------------------------------------------------
// EvaluationReport
// ---------------------------------------------------------------------------

/// Evidence grade of a reported number (CLAUDE.md tagging rules).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceGrade {
    /// Measured on real data with a leak-free split; carries the exact
    /// command that reproduces the number. Constructible only through
    /// [`EvaluationReport::measured`], which rejects an empty reproducer.
    Measured {
        /// Command line that reproduces this result.
        reproducer: String,
    },
    /// Computed on synthetic/generated data.
    Synthetic,
    /// Quoted from elsewhere; not reproduced in this repository.
    Claimed,
}

impl EvidenceGrade {
    /// Stable uppercase tag (`MEASURED` / `SYNTHETIC` / `CLAIMED`).
    #[must_use]
    pub fn tag(&self) -> &'static str {
        match self {
            EvidenceGrade::Measured { .. } => "MEASURED",
            EvidenceGrade::Synthetic => "SYNTHETIC",
            EvidenceGrade::Claimed => "CLAIMED",
        }
    }
}

/// An evaluation result that structurally pairs the model metric with the
/// mean-pose (or other) baseline metric and an [`EvidenceGrade`] — a model
/// number can never be reported without its baseline (ADR-291 §3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Protocol the split followed.
    pub protocol: SplitProtocol,
    /// Metric name, e.g. `"pck@0.2"` or `"mpjpe"`.
    pub metric_name: String,
    /// The model's metric on the audited test split.
    pub model_metric: f64,
    /// The baseline's metric on the same split (e.g.
    /// [`MeanPoseBaseline::mpjpe`]).
    pub baseline_metric: f64,
    /// Evidence grade; `MEASURED` embeds its reproducer.
    pub evidence: EvidenceGrade,
}

impl EvaluationReport {
    /// Build a `MEASURED` report. The reproducer command is mandatory and
    /// must be non-blank — a measured number without a reproducer is not
    /// measured (CLAUDE.md).
    ///
    /// # Errors
    ///
    /// - [`ProtocolError::MissingReproducer`] when `reproducer` is blank.
    /// - [`ProtocolError::NonFiniteMetric`] when either metric is NaN/±inf.
    pub fn measured(
        protocol: SplitProtocol,
        metric_name: impl Into<String>,
        model_metric: f64,
        baseline_metric: f64,
        reproducer: impl Into<String>,
    ) -> Result<Self, ProtocolError> {
        let reproducer = reproducer.into();
        if reproducer.trim().is_empty() {
            return Err(ProtocolError::MissingReproducer);
        }
        Self::build(
            protocol,
            metric_name.into(),
            model_metric,
            baseline_metric,
            EvidenceGrade::Measured { reproducer },
        )
    }

    /// Build a `SYNTHETIC` report (synthetic/generated data).
    ///
    /// # Errors
    ///
    /// [`ProtocolError::NonFiniteMetric`] when either metric is NaN/±inf.
    pub fn synthetic(
        protocol: SplitProtocol,
        metric_name: impl Into<String>,
        model_metric: f64,
        baseline_metric: f64,
    ) -> Result<Self, ProtocolError> {
        Self::build(
            protocol,
            metric_name.into(),
            model_metric,
            baseline_metric,
            EvidenceGrade::Synthetic,
        )
    }

    /// Build a `CLAIMED` report (quoted, not reproduced here).
    ///
    /// # Errors
    ///
    /// [`ProtocolError::NonFiniteMetric`] when either metric is NaN/±inf.
    pub fn claimed(
        protocol: SplitProtocol,
        metric_name: impl Into<String>,
        model_metric: f64,
        baseline_metric: f64,
    ) -> Result<Self, ProtocolError> {
        Self::build(
            protocol,
            metric_name.into(),
            model_metric,
            baseline_metric,
            EvidenceGrade::Claimed,
        )
    }

    fn build(
        protocol: SplitProtocol,
        metric_name: String,
        model_metric: f64,
        baseline_metric: f64,
        evidence: EvidenceGrade,
    ) -> Result<Self, ProtocolError> {
        if !model_metric.is_finite() {
            return Err(ProtocolError::NonFiniteMetric {
                name: format!("{metric_name} (model)"),
                value: model_metric,
            });
        }
        if !baseline_metric.is_finite() {
            return Err(ProtocolError::NonFiniteMetric {
                name: format!("{metric_name} (baseline)"),
                value: baseline_metric,
            });
        }
        Ok(EvaluationReport {
            protocol,
            metric_name,
            model_metric,
            baseline_metric,
            evidence,
        })
    }

    /// `model − baseline` (positive is better for higher-is-better metrics
    /// such as PCK; interpret per metric).
    #[must_use]
    pub fn margin_over_baseline(&self) -> f64 {
        self.model_metric - self.baseline_metric
    }

    /// One-line evidence-tagged summary, e.g.
    /// `"[MEASURED] cross-subject pck@0.2: model 0.6100 vs mean-pose baseline 0.4100"`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "[{}] {} {}: model {:.4} vs mean-pose baseline {:.4}",
            self.evidence.tag(),
            self.protocol.tag(),
            self.metric_name,
            self.model_metric,
            self.baseline_metric
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn meta(
        subject: u32,
        environment: u32,
        orientation: u32,
        recording: u64,
        window: u64,
    ) -> SampleMeta {
        SampleMeta {
            subject_id: subject,
            environment_id: environment,
            orientation_id: orientation,
            gesture_id: 1,
            recording_id: recording,
            window_index: window,
        }
    }

    // ----- LeakageAudit -----------------------------------------------------

    #[test]
    fn audit_passes_clean_cross_subject_split() {
        let train = vec![meta(1, 1, 1, 0, 0), meta(1, 1, 1, 0, 1), meta(2, 1, 2, 1, 0)];
        let test = vec![meta(3, 1, 1, 2, 0), meta(3, 1, 1, 2, 1)];
        let pass = LeakageAudit::for_protocol(SplitProtocol::CrossSubject)
            .audit(&train, &test)
            .expect("clean split must pass");
        assert_eq!(pass.train_windows, 3);
        assert_eq!(pass.test_windows, 2);
        assert_eq!(pass.train_subjects, 2);
        assert_eq!(pass.test_subjects, 1);
        assert_eq!(pass.train_recordings, 2);
        assert_eq!(pass.test_recordings, 1);
    }

    #[test]
    fn audit_rejects_subject_overlap() {
        let train = vec![meta(1, 1, 1, 0, 0), meta(2, 1, 1, 1, 0)];
        let test = vec![meta(2, 2, 2, 2, 0)]; // subject 2 on both sides
        let err = LeakageAudit::for_protocol(SplitProtocol::CrossSubject)
            .audit(&train, &test)
            .unwrap_err();
        assert!(matches!(err, ProtocolError::SubjectOverlap { subject_id: 2 }));
    }

    #[test]
    fn audit_rejects_environment_overlap_when_claimed() {
        let train = vec![meta(1, 1, 1, 0, 0)];
        let test = vec![meta(2, 1, 2, 1, 0)]; // environment 1 on both sides
        let err = LeakageAudit::for_protocol(SplitProtocol::CrossEnvironment)
            .audit(&train, &test)
            .unwrap_err();
        assert!(matches!(
            err,
            ProtocolError::EnvironmentOverlap { environment_id: 1 }
        ));
        // The same split passes a protocol that does not claim env-disjointness.
        assert!(LeakageAudit::for_protocol(SplitProtocol::CrossSubject)
            .audit(&train, &test)
            .is_ok());
    }

    #[test]
    fn audit_rejects_orientation_overlap_when_claimed() {
        let train = vec![meta(1, 1, 3, 0, 0)];
        let test = vec![meta(2, 2, 3, 1, 0)];
        let err = LeakageAudit::for_protocol(SplitProtocol::CrossOrientation)
            .audit(&train, &test)
            .unwrap_err();
        assert!(matches!(
            err,
            ProtocolError::OrientationOverlap { orientation_id: 3 }
        ));
    }

    #[test]
    fn audit_always_rejects_recording_crossing() {
        // Even a protocol claiming nothing (RandomBaseline) must fail when a
        // continuous recording straddles the boundary.
        let train = vec![meta(1, 1, 1, 5, 0)];
        let test = vec![meta(2, 2, 2, 5, 1)]; // same recording 5
        let err = LeakageAudit::for_protocol(SplitProtocol::RandomBaseline)
            .audit(&train, &test)
            .unwrap_err();
        assert!(matches!(
            err,
            ProtocolError::RecordingCrossesSplit { recording_id: 5 }
        ));
    }

    #[test]
    fn audit_rejects_empty_partitions() {
        let some = vec![meta(1, 1, 1, 0, 0)];
        let audit = LeakageAudit::for_protocol(SplitProtocol::CrossSubject);
        assert!(matches!(
            audit.audit(&[], &some),
            Err(ProtocolError::EmptyPartition { side: "train" })
        ));
        assert!(matches!(
            audit.audit(&some, &[]),
            Err(ProtocolError::EmptyPartition { side: "test" })
        ));
    }

    #[test]
    fn random_baseline_partition_fails_audit_end_to_end() {
        // Wire a real RandomBaseline SplitPlan into the audit: the leaky
        // window-level split must be rejected, which is exactly its purpose.
        use crate::protocols::SplitPlan;
        let mut metas = Vec::new();
        for recording in 0..8u64 {
            for window in 0..6u64 {
                metas.push(meta(1 + (recording % 3) as u32, 1, 1, recording, window));
            }
        }
        let plan = SplitPlan::new(SplitProtocol::RandomBaseline, 9, 0.5).unwrap();
        let (train_idx, test_idx) = plan.partition(&metas);
        let train: Vec<SampleMeta> = train_idx.iter().map(|&i| metas[i]).collect();
        let test: Vec<SampleMeta> = test_idx.iter().map(|&i| metas[i]).collect();
        assert!(matches!(
            LeakageAudit::for_protocol(SplitProtocol::RandomBaseline).audit(&train, &test),
            Err(ProtocolError::RecordingCrossesSplit { .. })
        ));
    }

    // ----- MeanPoseBaseline -------------------------------------------------

    #[test]
    fn mean_pose_is_elementwise_mean_of_training_poses() {
        let train = vec![
            array![[0.0f32, 0.0], [1.0, 1.0]],
            array![[0.2f32, 0.4], [0.6, 0.0]],
        ];
        let baseline = MeanPoseBaseline::fit(&train).unwrap();
        assert_eq!(baseline.num_train_poses(), 2);
        let mean = baseline.mean_pose();
        assert_abs_diff_eq!(mean[[0, 0]], 0.1, epsilon = 1e-6);
        assert_abs_diff_eq!(mean[[0, 1]], 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(mean[[1, 0]], 0.8, epsilon = 1e-6);
        assert_abs_diff_eq!(mean[[1, 1]], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn mean_pose_mpjpe_math() {
        // Baseline fitted on a single pose ⇒ mean equals it exactly.
        let train = vec![array![[0.0f32, 0.0], [1.0, 0.0]]];
        let baseline = MeanPoseBaseline::fit(&train).unwrap();

        // Test pose offset by (0.3, 0.4) on both joints ⇒ distance 0.5 each.
        let test = vec![array![[0.3f32, 0.4], [1.3, 0.4]]];
        let mpjpe = baseline.mpjpe(&test).unwrap();
        assert_abs_diff_eq!(mpjpe, 0.5, epsilon = 1e-6);

        // Distances are ~0.5 up to f32 rounding, so probe strictly either
        // side: PCK@0.6 counts both joints, PCK@0.49 counts neither.
        assert_abs_diff_eq!(baseline.pck_at(&test, 0.6).unwrap(), 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(baseline.pck_at(&test, 0.49).unwrap(), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn mean_pose_rejects_empty_and_mismatched() {
        assert!(matches!(
            MeanPoseBaseline::fit(&[]),
            Err(ProtocolError::EmptyTrainingPoses)
        ));
        let train = vec![
            array![[0.0f32, 0.0], [1.0, 1.0]],
            array![[0.0f32, 0.0]], // 1 joint vs 2
        ];
        assert!(matches!(
            MeanPoseBaseline::fit(&train),
            Err(ProtocolError::PoseShapeMismatch { .. })
        ));

        let baseline = MeanPoseBaseline::fit(&[array![[0.0f32, 0.0]]]).unwrap();
        assert!(baseline.mpjpe(&[]).is_err());
        assert!(baseline
            .mpjpe(&[array![[0.0f32, 0.0], [1.0, 1.0]]])
            .is_err());
    }

    // ----- EvaluationReport -------------------------------------------------

    #[test]
    fn measured_requires_reproducer() {
        let err = EvaluationReport::measured(
            SplitProtocol::CrossSubject,
            "pck@0.2",
            0.61,
            0.41,
            "   ",
        )
        .unwrap_err();
        assert!(matches!(err, ProtocolError::MissingReproducer));

        let report = EvaluationReport::measured(
            SplitProtocol::CrossSubject,
            "pck@0.2",
            0.61,
            0.41,
            "cargo run -p wifi-densepose-train --bin train -- eval --protocol cross-subject --seed 42",
        )
        .unwrap();
        assert_eq!(report.evidence.tag(), "MEASURED");
        assert_abs_diff_eq!(report.margin_over_baseline(), 0.2, epsilon = 1e-9);
        assert!(report.summary().starts_with("[MEASURED] cross-subject pck@0.2"));
    }

    #[test]
    fn synthetic_and_claimed_tags() {
        let s =
            EvaluationReport::synthetic(SplitProtocol::CrossOrientation, "mpjpe", 0.1, 0.3)
                .unwrap();
        assert_eq!(s.evidence.tag(), "SYNTHETIC");
        let c = EvaluationReport::claimed(SplitProtocol::CrossSubject, "pck@0.5", 0.9, 0.5)
            .unwrap();
        assert_eq!(c.evidence.tag(), "CLAIMED");
    }

    #[test]
    fn report_rejects_non_finite_metrics() {
        assert!(matches!(
            EvaluationReport::synthetic(SplitProtocol::CrossSubject, "pck", f64::NAN, 0.5),
            Err(ProtocolError::NonFiniteMetric { .. })
        ));
        assert!(matches!(
            EvaluationReport::synthetic(SplitProtocol::CrossSubject, "pck", 0.5, f64::INFINITY),
            Err(ProtocolError::NonFiniteMetric { .. })
        ));
    }

    #[test]
    fn report_serializes_roundtrip() {
        let report = EvaluationReport::measured(
            SplitProtocol::CrossEnvironment,
            "mpjpe",
            0.07,
            0.19,
            "cargo test -p wifi-densepose-train",
        )
        .unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let back: EvaluationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }
}
