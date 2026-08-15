//! Fail-closed sensing-server integration for ADR-323.
//!
//! This adapter deliberately accepts only the canonical contract. Existing
//! renderer poses are image/pixel-space and must not be silently promoted to
//! metric 3D observations.

use std::collections::BTreeMap;

use wifi_densepose_core::{
    PhysicsMode, PoseObservationV2, PoseRefinementV1, RefinementDisposition,
};
use wifi_densepose_physics::{
    metrics::PhysicsMetrics, CorrectionAuthorization, PhysicsConfig, PhysicsEngine, PhysicsError,
    VerifiedFrameContext,
};

/// Operator-requested API representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoseView {
    /// Immutable observer output only (migration default).
    #[default]
    Raw,
    /// Observer output plus additive physics assessment.
    Both,
    /// A selected corrected pose; unavailable is a typed conflict.
    Refined,
}

/// Borrowed publication selected without relabeling raw data.
#[derive(Debug)]
pub enum SelectedPose<'a> {
    Raw(&'a PoseObservationV2),
    Both {
        raw: &'a PoseObservationV2,
        physics: &'a PoseRefinementV1,
    },
    Refined(&'a [[f32; 3]; 17]),
}

/// Typed body for HTTP 409 when `view=refined` is unavailable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct RefinedUnavailable {
    pub code: &'static str,
    pub disposition: RefinementDisposition,
}

/// Result of scheduling refinement work while raw publication continues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// First pending frame for this track.
    Queued,
    /// A pending frame was discarded and the newest frame was retained.
    ReplacedIntermediate,
    /// A new track could not be scheduled within the configured cardinality cap.
    RejectedCapacity,
}

/// Process-local engine wrapper. Control-plane authentication is enforced by
/// the caller before [`Self::set_mode`] is invoked.
pub struct PosePhysicsRuntime {
    engine: PhysicsEngine,
    metrics: PhysicsMetrics,
    latest: Option<(PoseObservationV2, PoseRefinementV1)>,
    pending: BTreeMap<(u64, String), PoseObservationV2>,
    max_pending_tracks: usize,
}

impl PosePhysicsRuntime {
    /// Create the engine with validated, operator-owned limits.
    pub fn new(config: PhysicsConfig) -> Result<Self, PhysicsError> {
        let max_pending_tracks = config.max_tracks;
        Ok(Self {
            engine: PhysicsEngine::new(config)?,
            metrics: PhysicsMetrics::default(),
            latest: None,
            pending: BTreeMap::new(),
            max_pending_tracks,
        })
    }

    /// Retain only the newest pending frame per track. This queue covers
    /// refinement work only; callers must publish every immutable raw frame
    /// independently before scheduling it here.
    pub fn enqueue_latest(&mut self, raw: PoseObservationV2) -> EnqueueOutcome {
        let key = (raw.sensor_epoch, raw.track_id.0.clone());
        if let Some(current) = self.pending.get_mut(&key) {
            if (raw.timestamp_ns, raw.sequence) > (current.timestamp_ns, current.sequence) {
                *current = raw;
            }
            self.metrics.dropped_intermediate();
            return EnqueueOutcome::ReplacedIntermediate;
        }
        if self.pending.len() >= self.max_pending_tracks {
            return EnqueueOutcome::RejectedCapacity;
        }
        self.pending.insert(key, raw);
        EnqueueOutcome::Queued
    }

    /// Drain the bounded queue in deterministic track order.
    #[must_use]
    pub fn process_pending(&mut self, now_ns: u64) -> Vec<PoseRefinementV1> {
        let pending = core::mem::take(&mut self.pending);
        pending
            .into_values()
            .map(|raw| self.process(&raw, now_ns))
            .collect()
    }

    /// Assess exactly one canonical observation.
    #[must_use]
    pub fn process(&mut self, raw: &PoseObservationV2, now_ns: u64) -> PoseRefinementV1 {
        let result = self.engine.process(raw, now_ns);
        self.record(raw, &result);
        result
    }

    /// Process a frame whose signed envelope and replay window were verified by
    /// the serving data plane. Corrected selection is impossible without this receipt.
    #[must_use]
    pub fn process_verified(
        &mut self,
        raw: &PoseObservationV2,
        now_ns: u64,
        verification: &VerifiedFrameContext,
    ) -> PoseRefinementV1 {
        let result = self.engine.process_verified(raw, now_ns, verification);
        self.record(raw, &result);
        result
    }

    /// Activate release-gate evidence previously verified by the signed local
    /// control plane.
    pub fn authorize_correction(
        &mut self,
        authorization: CorrectionAuthorization,
    ) -> Result<(), PhysicsError> {
        self.engine.authorize_correction(authorization)
    }

    /// Active configuration hash used to bind signed evidence receipts.
    #[must_use]
    pub const fn config_hash(&self) -> [u8; 32] {
        self.engine.config_hash()
    }

    /// Apply a previously authenticated and witnessed mode transition.
    pub fn set_mode(&mut self, mode: PhysicsMode) -> Result<(), PhysicsError> {
        self.engine.set_mode(mode)?;
        if mode == PhysicsMode::Off {
            self.latest = None;
            self.pending.clear();
        }
        Ok(())
    }

    /// Most recent immutable raw/result pair, if a canonical frame was processed.
    #[must_use]
    pub const fn latest(&self) -> Option<&(PoseObservationV2, PoseRefinementV1)> {
        self.latest.as_ref()
    }

    /// Privacy-safe Prometheus exposition with fixed-cardinality labels only.
    #[must_use]
    pub fn metrics_text(&self) -> String {
        self.metrics.encode_prometheus()
    }

    fn record(&mut self, raw: &PoseObservationV2, result: &PoseRefinementV1) {
        self.metrics.observe(result, raw.observer_confidence.get());
        self.latest = Some((raw.clone(), result.clone()));
    }

    /// Select an additive API view. Refined never silently falls back to raw.
    pub fn select<'a>(
        view: PoseView,
        raw: &'a PoseObservationV2,
        physics: &'a PoseRefinementV1,
    ) -> Result<SelectedPose<'a>, RefinedUnavailable> {
        match view {
            PoseView::Raw => Ok(SelectedPose::Raw(raw)),
            PoseView::Both => Ok(SelectedPose::Both { raw, physics }),
            PoseView::Refined => physics
                .refined_joints_m
                .as_ref()
                .filter(|_| physics.selected)
                .map(SelectedPose::Refined)
                .ok_or(RefinedUnavailable {
                    code: "pose_refined_unavailable",
                    disposition: physics.disposition,
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wifi_densepose_core::{
        AbstentionReason, CalibrationId, Coco17Joint, ConstraintResiduals, ContactHypothesis,
        InterventionSummary, JointObservation, JointVisibility, ModelRef, PhysicsProvenance,
        PoseDimensionality, PoseTrustState, Probability, SourceProvenance, SpatialFrameRef,
        SymmetricCovariance3, TrackId,
    };

    fn pair() -> (PoseObservationV2, PoseRefinementV1) {
        let joints = Coco17Joint::ALL.map(|kind| JointObservation {
            kind,
            position_m: [0.0; 3],
            covariance_m2: SymmetricCovariance3 {
                xx: 0.0,
                xy: 0.0,
                xz: 0.0,
                yy: 0.0,
                yz: 0.0,
                zz: 0.0,
            },
            confidence: Probability::ZERO,
            visibility: JointVisibility::Unknown,
        });
        let mut raw = PoseObservationV2 {
            schema_version: 2,
            timestamp_ns: 1,
            sensor_epoch: 1,
            sequence: 1,
            track_id: TrackId("local:1".into()),
            frame: SpatialFrameRef {
                name: "image".into(),
                version: 1,
                metric: false,
                right_handed: false,
                z_up: false,
            },
            calibration_id: CalibrationId("none".into()),
            floor_plane: None,
            model: ModelRef {
                id: "test".into(),
                artifact_hash: [0; 32],
            },
            source: SourceProvenance {
                sensor_id: "test".into(),
                authenticated: false,
                replay_protected: false,
            },
            trust_state: PoseTrustState::Degraded,
            dimensionality: PoseDimensionality::Image2d,
            uncertainty_calibrated: false,
            joints,
            observer_confidence: Probability::ZERO,
            canonical_hash: [0; 32],
        };
        raw.seal();
        let physics = PoseRefinementV1 {
            schema_version: 1,
            raw_observation_hash: raw.canonical_hash,
            mode: PhysicsMode::Audit,
            disposition: RefinementDisposition::Audited2d,
            selected: false,
            refined_joints_m: None,
            physics_confidence: Probability::ONE,
            effective_confidence: Probability::ZERO,
            intervention: InterventionSummary::default(),
            residuals: ConstraintResiduals::default(),
            refined_residuals: None,
            dynamics: None,
            contact_hypotheses: [ContactHypothesis::default(); 2],
            provenance: PhysicsProvenance {
                engine: "test".into(),
                engine_version: "1".into(),
                config_hash: [0; 32],
                rf_model_hash: [0; 32],
                calibration_id: "none".into(),
                learned_artifact_hash: None,
            },
            reason: Some(AbstentionReason::UncertaintyUncalibrated),
            canonical_hash: [0; 32],
        };
        (raw, physics)
    }

    #[test]
    fn refined_view_never_silently_falls_back() {
        let (raw, physics) = pair();
        let error = PosePhysicsRuntime::select(PoseView::Refined, &raw, &physics).unwrap_err();
        assert_eq!(error.code, "pose_refined_unavailable");
    }

    #[test]
    fn processing_publishes_latest_pair_and_privacy_safe_metrics() {
        let (raw, _) = pair();
        let mut runtime = PosePhysicsRuntime::new(PhysicsConfig::default()).unwrap();
        let result = runtime.process(&raw, raw.timestamp_ns);
        let (published_raw, published_result) = runtime.latest().unwrap();
        assert_eq!(published_raw.canonical_hash, raw.canonical_hash);
        assert_eq!(published_result.canonical_hash, result.canonical_hash);
        let metrics = runtime.metrics_text();
        assert!(metrics.contains("ruview_pose_physics_frames_total"));
        assert!(metrics.contains("ruview_pose_physics_confidence_delta"));
        assert!(!metrics.contains(&raw.track_id.0));
        assert!(!metrics.contains(&raw.source.sensor_id));
    }

    #[test]
    fn backpressure_retains_newest_frame_per_track_with_a_hard_cap() {
        let (raw, _) = pair();
        let mut config = PhysicsConfig::default();
        config.max_tracks = 1;
        let mut runtime = PosePhysicsRuntime::new(config).unwrap();

        assert_eq!(runtime.enqueue_latest(raw.clone()), EnqueueOutcome::Queued);
        let mut newer = raw.clone();
        newer.sequence = 2;
        newer.timestamp_ns = 2;
        newer.seal();
        assert_eq!(
            runtime.enqueue_latest(newer.clone()),
            EnqueueOutcome::ReplacedIntermediate
        );

        let mut second_track = raw;
        second_track.track_id = TrackId("local:2".into());
        second_track.seal();
        assert_eq!(
            runtime.enqueue_latest(second_track),
            EnqueueOutcome::RejectedCapacity
        );

        let results = runtime.process_pending(newer.timestamp_ns);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].raw_observation_hash, newer.canonical_hash);
        assert!(runtime
            .metrics_text()
            .contains("ruview_pose_physics_dropped_intermediate_total 1"));
    }
}
