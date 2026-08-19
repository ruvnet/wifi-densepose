//! Stateful, bounded top-level control flow.

use std::{collections::HashMap, time::Instant};

use wifi_densepose_core::{
    AbstentionReason, ConstraintResiduals, ContactHypothesis, ContactState, PhysicsMode,
    PhysicsProvenance, PoseDimensionality, PoseObservationV2, PoseRefinementV1, PoseTrustState,
    Probability, RefinementDisposition, POSE_OBSERVATION_SCHEMA_VERSION,
};

use crate::{
    config::PhysicsConfig,
    constraints::{self, TemporalHistory},
    error::PhysicsError,
    projector,
    skeleton::SkeletonPosterior,
    uncertainty::movement_weight,
};

#[derive(Default)]
struct TrackState {
    last_sequence: Option<u64>,
    last_timestamp_ns: Option<u64>,
    last_hash: Option<[u8; 32]>,
    last_config_hash: Option<[u8; 32]>,
    cached: Option<PoseRefinementV1>,
    skeleton: SkeletonPosterior,
    temporal: TemporalHistory,
    last_calibration_id: Option<String>,
    #[cfg(feature = "dynamics")]
    dynamics: Option<crate::dynamics::DynamicsAuditor>,
}

/// Evidence receipt produced only after the serving boundary verifies the
/// signed configuration, release gate, sensor authentication, and replay
/// policy. The physics crate binds it to one exact configuration hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorrectionAuthorization {
    config_hash: [u8; 32],
    evidence_hash: [u8; 32],
}

impl CorrectionAuthorization {
    /// Construct a receipt from already verified, hash-addressed evidence.
    pub fn from_verified_evidence(
        config_hash: [u8; 32],
        evidence_hash: [u8; 32],
    ) -> Result<Self, PhysicsError> {
        if config_hash == [0; 32] || evidence_hash == [0; 32] {
            return Err(PhysicsError::InvalidCorrectionAuthorization(
                "configuration and evidence hashes must be non-zero",
            ));
        }
        Ok(Self {
            config_hash,
            evidence_hash,
        })
    }
}

/// Per-frame receipt returned by the ADR-305 verifier after signature,
/// freshness, sequence, replay-window, calibration, and trust-state checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedFrameContext {
    raw_hash: [u8; 32],
    sensor_epoch: u64,
    sequence: u64,
    sensor_id: String,
    calibration_id: String,
    receipt_hash: [u8; 32],
}

impl VerifiedFrameContext {
    /// Bind process-owned verification evidence to one canonical frame.
    pub fn from_verified_envelope(
        raw_hash: [u8; 32],
        sensor_epoch: u64,
        sequence: u64,
        sensor_id: String,
        calibration_id: String,
        receipt_hash: [u8; 32],
    ) -> Result<Self, PhysicsError> {
        if raw_hash == [0; 32]
            || receipt_hash == [0; 32]
            || sensor_id.is_empty()
            || calibration_id.is_empty()
        {
            return Err(PhysicsError::InvalidCorrectionAuthorization(
                "verified frame receipt is incomplete",
            ));
        }
        Ok(Self {
            raw_hash,
            sensor_epoch,
            sequence,
            sensor_id,
            calibration_id,
            receipt_hash,
        })
    }

    fn matches(&self, raw: &PoseObservationV2) -> bool {
        self.raw_hash == raw.canonical_hash
            && self.sensor_epoch == raw.sensor_epoch
            && self.sequence == raw.sequence
            && self.sensor_id == raw.source.sensor_id
            && self.calibration_id == raw.calibration_id.0
            && self.receipt_hash != [0; 32]
    }
}

/// In-memory physics engine. Track state is anonymous and discarded on expiry.
pub struct PhysicsEngine {
    config: PhysicsConfig,
    config_hash: [u8; 32],
    tracks: HashMap<(wifi_densepose_core::TrackId, u64), TrackState>,
    correction_authorization: Option<CorrectionAuthorization>,
}

impl PhysicsEngine {
    /// Activate a validated trusted configuration.
    pub fn new(config: PhysicsConfig) -> Result<Self, PhysicsError> {
        config
            .validate()
            .map_err(PhysicsError::InvalidConfiguration)?;
        let config_hash = config.canonical_hash();
        Ok(Self {
            config,
            config_hash,
            tracks: HashMap::new(),
            correction_authorization: None,
        })
    }

    /// Current immutable configuration.
    #[must_use]
    pub const fn config(&self) -> &PhysicsConfig {
        &self.config
    }

    /// Hash of the exact active configuration.
    #[must_use]
    pub const fn config_hash(&self) -> [u8; 32] {
        self.config_hash
    }

    /// Hash a prospective mode without activating it, so a signed evidence
    /// receipt can be bound before the atomic transition.
    #[must_use]
    pub fn config_hash_for_mode(&self, mode: PhysicsMode) -> [u8; 32] {
        let mut prospective = self.config.clone();
        prospective.mode = mode;
        prospective.canonical_hash()
    }

    /// Activate a correction receipt after the serving boundary has verified it.
    pub fn authorize_correction(
        &mut self,
        authorization: CorrectionAuthorization,
    ) -> Result<(), PhysicsError> {
        self.correction_authorization = Some(authorization);
        Ok(())
    }

    /// Authenticated control planes may change rollout mode without restart.
    pub fn set_mode(&mut self, mode: PhysicsMode) -> Result<(), PhysicsError> {
        let mut prospective = self.config.clone();
        prospective.mode = mode;
        let prospective_hash = prospective.canonical_hash();
        if mode.selects_correction()
            && self
                .correction_authorization
                .is_none_or(|receipt| receipt.config_hash != prospective_hash)
        {
            return Err(PhysicsError::InvalidCorrectionAuthorization(
                "correction mode requires evidence bound to the target configuration",
            ));
        }
        self.config.mode = mode;
        self.config_hash = prospective_hash;
        if matches!(mode, PhysicsMode::Off) {
            self.tracks.clear();
        }
        Ok(())
    }

    /// Process one frame and always return a typed assessment.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn process(&mut self, raw: &PoseObservationV2, now_ns: u64) -> PoseRefinementV1 {
        self.process_with_context(raw, now_ns, None)
    }

    /// Process a frame with process-owned source verification evidence.
    #[must_use]
    pub fn process_verified(
        &mut self,
        raw: &PoseObservationV2,
        now_ns: u64,
        verification: &VerifiedFrameContext,
    ) -> PoseRefinementV1 {
        self.process_with_context(raw, now_ns, Some(verification))
    }

    #[allow(clippy::too_many_lines)]
    fn process_with_context(
        &mut self,
        raw: &PoseObservationV2,
        now_ns: u64,
        verification: Option<&VerifiedFrameContext>,
    ) -> PoseRefinementV1 {
        let started = Instant::now();
        if let Some(reason) = self.validate_contract(raw, now_ns) {
            return self.raw_result(raw, RefinementDisposition::Rejected, Some(reason));
        }
        if self.config.mode == PhysicsMode::Off {
            return self.raw_result(
                raw,
                RefinementDisposition::Bypassed,
                Some(AbstentionReason::ModeOff),
            );
        }
        if raw.trust_state == PoseTrustState::Unknown {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::OodUnknown),
            );
        }
        if raw.dimensionality == PoseDimensionality::Image2d {
            return self.audit_2d(raw, started);
        }
        if self.config.mode.selects_correction()
            && self
                .correction_authorization
                .is_none_or(|receipt| receipt.config_hash != self.config_hash)
        {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::CorrectionNotAuthorized),
            );
        }
        if self.config.mode.selects_correction()
            && verification.is_none_or(|receipt| !receipt.matches(raw))
        {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::SourceUnauthenticated),
            );
        }
        if !raw.uncertainty_calibrated {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::UncertaintyUncalibrated),
            );
        }
        let Some(floor) = raw.floor_plane else {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::CalibrationUnavailable),
            );
        };
        if !floor.is_valid() {
            return self.raw_result(
                raw,
                RefinementDisposition::Rejected,
                Some(AbstentionReason::CoordinateFrameMismatch),
            );
        }

        self.expire_tracks(now_ns);
        let key = (raw.track_id.clone(), raw.sensor_epoch);
        if !self.tracks.contains_key(&key) && self.tracks.len() >= self.config.max_tracks {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::TrackCapacity),
            );
        }
        let mut state = self.tracks.remove(&key).unwrap_or_default();
        if state.last_sequence == Some(raw.sequence) {
            if state.last_hash == Some(raw.canonical_hash)
                && state.last_config_hash == Some(self.config_hash)
            {
                if let Some(cached) = state.cached.clone() {
                    self.tracks.insert(key, state);
                    return cached;
                }
            }
            self.tracks.insert(key, state);
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::ReplayRejected),
            );
        }
        if state
            .last_sequence
            .is_some_and(|sequence| raw.sequence < sequence)
            || state
                .last_timestamp_ns
                .is_some_and(|timestamp| raw.timestamp_ns <= timestamp)
        {
            state.temporal.reset();
            self.tracks.insert(key, state);
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::NonMonotonicInput),
            );
        }
        if state.last_timestamp_ns.is_some_and(|timestamp| {
            raw.timestamp_ns.saturating_sub(timestamp) > self.config.track_reset_gap_ms * 1_000_000
        }) {
            state = TrackState::default();
        } else if state.last_timestamp_ns.is_some_and(|timestamp| {
            raw.timestamp_ns.saturating_sub(timestamp) > self.config.max_frame_gap_ms * 1_000_000
        }) {
            state.temporal.reset();
        }
        if state
            .last_calibration_id
            .as_ref()
            .is_some_and(|calibration| calibration != &raw.calibration_id.0)
        {
            state = TrackState::default();
        }

        let known = raw
            .joints
            .iter()
            .filter(|joint| {
                joint.visibility != wifi_densepose_core::JointVisibility::Unknown
                    && joint.confidence.get() >= self.config.minimum_joint_confidence
            })
            .count();
        if known < self.config.minimum_known_joints {
            self.tracks.insert(key, state);
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::TooFewKnownJoints),
            );
        }

        let observed = raw.joints.map(|joint| joint.position_m);
        let confidence = raw.joints.map(|joint| joint.confidence.get());
        let weights = raw.joints.map(|joint| movement_weight(joint.covariance_m2));
        let audit = constraints::audit(
            &observed,
            &state.skeleton,
            floor,
            &state.temporal,
            raw.timestamp_ns,
        );
        state.skeleton.observe(
            &observed,
            &confidence,
            self.config.calibration_joint_confidence,
        );

        if self.config.mode == PhysicsMode::Audit {
            let result = self.build_result(
                raw,
                RefinementDisposition::Audited,
                false,
                None,
                wifi_densepose_core::InterventionSummary {
                    elapsed_us: elapsed_us(started),
                    ..wifi_densepose_core::InterventionSummary::default()
                },
                audit,
                None,
                None,
            );
            state.temporal.push(observed, raw.timestamp_ns);
            state.last_sequence = Some(raw.sequence);
            state.last_timestamp_ns = Some(raw.timestamp_ns);
            state.last_hash = Some(raw.canonical_hash);
            state.last_config_hash = Some(self.config_hash);
            state.last_calibration_id = Some(raw.calibration_id.0.clone());
            state.cached = Some(result.clone());
            self.tracks.insert(key, state);
            return result;
        }

        let projection = projector::project(
            &observed,
            &weights,
            &state.skeleton,
            &state.temporal,
            raw.timestamp_ns,
            floor,
            &self.config,
            started,
        );
        let result = match projection {
            Err(reason) => self.raw_result(raw, RefinementDisposition::Abstained, Some(reason)),
            Ok(candidate) => {
                #[allow(unused_mut)]
                let mut refined_residuals = constraints::audit(
                    &candidate.joints,
                    &state.skeleton,
                    floor,
                    &state.temporal,
                    raw.timestamp_ns,
                );
                #[cfg(feature = "dynamics")]
                let dynamics = if self.config.dynamics_audit {
                    let auditor = state.dynamics.get_or_insert_with(|| {
                        crate::dynamics::DynamicsAuditor::new(&candidate.joints, floor)
                    });
                    let assessment = auditor.audit(
                        &candidate.joints,
                        self.config.dynamics_dt_seconds,
                        self.config.dynamics_substeps,
                    );
                    refined_residuals.collision_m = assessment.max_floor_penetration_m;
                    refined_residuals.normalized_total += (assessment.max_tracking_error_m / 0.05
                        + assessment.max_joint_anchor_error_m / 0.02
                        + assessment.max_floor_penetration_m / 0.05)
                        / 3.0;
                    Some(wifi_densepose_core::DynamicsResiduals {
                        stable: assessment.stable,
                        segment_count: assessment.segment_count,
                        joint_count: assessment.joint_count,
                        contact_count: assessment.contact_count,
                        substeps: assessment.substeps,
                        tracking_error_m: assessment.max_tracking_error_m,
                        joint_anchor_error_m: assessment.max_joint_anchor_error_m,
                        floor_penetration_m: assessment.max_floor_penetration_m,
                        control_effort: assessment.max_control_effort,
                    })
                } else {
                    None
                };
                #[cfg(not(feature = "dynamics"))]
                let dynamics: Option<wifi_densepose_core::DynamicsResiduals> = None;
                if started.elapsed() >= std::time::Duration::from_millis(self.config.deadline_ms) {
                    self.raw_result(
                        raw,
                        RefinementDisposition::Abstained,
                        Some(AbstentionReason::DeadlineExceeded),
                    )
                } else {
                    let (disposition, selected, refined) = match self.config.mode {
                        PhysicsMode::Audit => unreachable!("audit returned before projection"),
                        PhysicsMode::ShadowCorrect => (
                            RefinementDisposition::Shadowed,
                            false,
                            Some(candidate.joints),
                        ),
                        PhysicsMode::OptInCorrect | PhysicsMode::DefaultCorrect => (
                            RefinementDisposition::Corrected,
                            true,
                            Some(candidate.joints),
                        ),
                        PhysicsMode::Off => unreachable!("off returned before projection"),
                    };
                    let mut result = self.build_result(
                        raw,
                        disposition,
                        selected,
                        refined,
                        candidate.intervention,
                        audit,
                        Some(refined_residuals),
                        None,
                    );
                    result.dynamics = dynamics;
                    if result.dynamics.is_some() {
                        result.provenance.engine = "kinematic-pbd+rapier-audit".into();
                        result.seal();
                    }
                    result
                }
            }
        };

        let committed_joints = if result.selected {
            result.refined_joints_m.unwrap_or(observed)
        } else {
            observed
        };
        state.temporal.push(committed_joints, raw.timestamp_ns);
        state.last_sequence = Some(raw.sequence);
        state.last_timestamp_ns = Some(raw.timestamp_ns);
        state.last_hash = Some(raw.canonical_hash);
        state.last_config_hash = Some(self.config_hash);
        state.last_calibration_id = Some(raw.calibration_id.0.clone());
        state.cached = Some(result.clone());
        self.tracks.insert(key, state);
        debug_assert!(result.effective_confidence.get() <= raw.observer_confidence.get());
        debug_assert_eq!(result.raw_observation_hash, raw.canonical_hash);
        result
    }

    fn validate_contract(&self, raw: &PoseObservationV2, now_ns: u64) -> Option<AbstentionReason> {
        if raw.schema_version != POSE_OBSERVATION_SCHEMA_VERSION {
            return Some(AbstentionReason::UnsupportedSchema);
        }
        if raw.compute_canonical_hash() != raw.canonical_hash {
            return Some(AbstentionReason::HashMismatch);
        }
        if raw.track_id.0.is_empty()
            || raw.track_id.0.len() > 128
            || raw.frame.name.is_empty()
            || raw.frame.name.len() > 128
            || raw.calibration_id.0.is_empty()
            || raw.calibration_id.0.len() > 128
            || raw.model.id.is_empty()
            || raw.model.id.len() > 128
            || raw.source.sensor_id.is_empty()
            || raw.source.sensor_id.len() > 128
        {
            return Some(AbstentionReason::InvalidNumber);
        }
        if now_ns.saturating_sub(raw.timestamp_ns) > self.config.stale_after_ms * 1_000_000
            || raw.timestamp_ns > now_ns.saturating_add(50_000_000)
        {
            return Some(AbstentionReason::StaleInput);
        }
        let coordinate_bound = match raw.dimensionality {
            PoseDimensionality::Metric3d => self.config.room_bound_m,
            PoseDimensionality::Image2d => self.config.image_coordinate_bound,
        };
        for (index, joint) in raw.joints.iter().enumerate() {
            if joint.kind != wifi_densepose_core::Coco17Joint::ALL[index]
                || !joint.position_m.iter().all(|v| v.is_finite())
                || joint.position_m.iter().any(|v| v.abs() > coordinate_bound)
            {
                return Some(AbstentionReason::InvalidNumber);
            }
            if !joint.covariance_m2.is_positive_semidefinite() {
                return Some(AbstentionReason::InvalidCovariance);
            }
        }
        match raw.dimensionality {
            PoseDimensionality::Metric3d
                if !(raw.frame.metric && raw.frame.right_handed && raw.frame.z_up) =>
            {
                Some(AbstentionReason::CoordinateFrameMismatch)
            }
            PoseDimensionality::Image2d if raw.frame.metric => {
                Some(AbstentionReason::CoordinateFrameMismatch)
            }
            _ => None,
        }
    }

    fn audit_2d(&mut self, raw: &PoseObservationV2, started: Instant) -> PoseRefinementV1 {
        self.expire_tracks(raw.timestamp_ns);
        let key = (raw.track_id.clone(), raw.sensor_epoch);
        if !self.tracks.contains_key(&key) && self.tracks.len() >= self.config.max_tracks {
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::TrackCapacity),
            );
        }
        let mut state = self.tracks.remove(&key).unwrap_or_default();
        if state.last_sequence == Some(raw.sequence) {
            if state.last_hash == Some(raw.canonical_hash)
                && state.last_config_hash == Some(self.config_hash)
            {
                if let Some(cached) = state.cached.clone() {
                    self.tracks.insert(key, state);
                    return cached;
                }
            }
            self.tracks.insert(key, state);
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::ReplayRejected),
            );
        }
        if state
            .last_sequence
            .is_some_and(|sequence| raw.sequence < sequence)
            || state
                .last_timestamp_ns
                .is_some_and(|timestamp| raw.timestamp_ns <= timestamp)
        {
            state.temporal.reset();
            self.tracks.insert(key, state);
            return self.raw_result(
                raw,
                RefinementDisposition::Abstained,
                Some(AbstentionReason::NonMonotonicInput),
            );
        }
        if state.last_timestamp_ns.is_some_and(|timestamp| {
            raw.timestamp_ns.saturating_sub(timestamp) > self.config.track_reset_gap_ms * 1_000_000
        }) {
            state = TrackState::default();
        } else if state.last_timestamp_ns.is_some_and(|timestamp| {
            raw.timestamp_ns.saturating_sub(timestamp) > self.config.max_frame_gap_ms * 1_000_000
        }) {
            state.temporal.reset();
        }
        if state
            .last_calibration_id
            .as_ref()
            .is_some_and(|calibration| calibration != &raw.calibration_id.0)
        {
            state = TrackState::default();
        }
        let joints = raw.joints.map(|joint| joint.position_m);
        let residuals = image_residuals(&joints, &state.temporal, raw.timestamp_ns);
        let intervention = wifi_densepose_core::InterventionSummary {
            elapsed_us: elapsed_us(started),
            ..wifi_densepose_core::InterventionSummary::default()
        };
        let result = self.build_result(
            raw,
            RefinementDisposition::Audited2d,
            false,
            None,
            intervention,
            residuals,
            None,
            None,
        );
        state.temporal.push(joints, raw.timestamp_ns);
        state.last_sequence = Some(raw.sequence);
        state.last_timestamp_ns = Some(raw.timestamp_ns);
        state.last_hash = Some(raw.canonical_hash);
        state.last_config_hash = Some(self.config_hash);
        state.last_calibration_id = Some(raw.calibration_id.0.clone());
        state.cached = Some(result.clone());
        self.tracks.insert(key, state);
        result
    }

    fn raw_result(
        &self,
        raw: &PoseObservationV2,
        disposition: RefinementDisposition,
        reason: Option<AbstentionReason>,
    ) -> PoseRefinementV1 {
        self.build_result(
            raw,
            disposition,
            false,
            None,
            wifi_densepose_core::InterventionSummary::default(),
            ConstraintResiduals::default(),
            None,
            reason,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_result(
        &self,
        raw: &PoseObservationV2,
        disposition: RefinementDisposition,
        selected: bool,
        refined: Option<[[f32; 3]; 17]>,
        intervention: wifi_densepose_core::InterventionSummary,
        residuals: ConstraintResiduals,
        refined_residuals: Option<ConstraintResiduals>,
        reason: Option<AbstentionReason>,
    ) -> PoseRefinementV1 {
        let penalty = self.config.beta_residual.mul_add(
            residuals.normalized_total,
            self.config.beta_intervention * intervention.max_joint_correction_m
                / self.config.max_joint_correction_m,
        );
        let physics_value = if reason == Some(AbstentionReason::ModeOff) {
            1.0
        } else if reason.is_some() {
            0.0
        } else {
            (-penalty.max(0.0)).exp().clamp(0.0, 1.0)
        };
        let physics_confidence = Probability::new(physics_value).unwrap_or(Probability::ZERO);
        let effective_value = raw
            .observer_confidence
            .get()
            .min(raw.observer_confidence.get() * physics_value);
        let effective_confidence = Probability::new(effective_value).unwrap_or(Probability::ZERO);
        let contact_hypotheses = if raw.dimensionality == PoseDimensionality::Metric3d {
            [15usize, 16usize].map(|index| {
                let probability = raw.floor_plane.map_or(0.0, |floor| {
                    (1.0 - floor.signed_distance(raw.joints[index].position_m).abs() / 0.05)
                        .clamp(0.0, 1.0)
                });
                ContactHypothesis {
                    state: ContactState::Hypothesis,
                    probability: Probability::new(probability).unwrap_or(Probability::ZERO),
                }
            })
        } else {
            [ContactHypothesis::default(); 2]
        };
        let mut result = PoseRefinementV1 {
            schema_version: 1,
            raw_observation_hash: raw.canonical_hash,
            mode: self.config.mode,
            disposition,
            selected,
            refined_joints_m: refined,
            physics_confidence,
            effective_confidence,
            intervention,
            residuals,
            refined_residuals,
            dynamics: None,
            contact_hypotheses,
            provenance: PhysicsProvenance {
                engine: "kinematic-pbd".into(),
                engine_version: env!("CARGO_PKG_VERSION").into(),
                config_hash: self.config_hash,
                rf_model_hash: raw.model.artifact_hash,
                calibration_id: raw.calibration_id.0.clone(),
                learned_artifact_hash: None,
            },
            reason,
            canonical_hash: [0; 32],
        };
        result.seal();
        result
    }

    fn expire_tracks(&mut self, now_ns: u64) {
        let expiry = self.config.track_reset_gap_ms * 1_000_000;
        self.tracks.retain(|_, state| {
            state
                .last_timestamp_ns
                .is_none_or(|timestamp| now_ns.saturating_sub(timestamp) <= expiry)
        });
    }
}

fn image_residuals(
    joints: &[[f32; 3]; 17],
    temporal: &crate::constraints::TemporalHistory,
    timestamp_ns: u64,
) -> ConstraintResiduals {
    let lengths = crate::skeleton::OBSERVED_EDGES
        .map(|(a, b)| crate::skeleton::distance(joints[a], joints[b]));
    let mean = lengths.iter().sum::<f32>() / lengths.len() as f32;
    let ratio_variance = if mean > 1.0e-6 {
        lengths
            .iter()
            .map(|length| ((length - mean) / mean).powi(2))
            .sum::<f32>()
            / lengths.len() as f32
    } else {
        0.0
    };
    let (velocity, acceleration, jerk) =
        crate::constraints::temporal_residuals(joints, temporal, timestamp_ns);
    ConstraintResiduals {
        normalized_total: ratio_variance.sqrt().min(1.0),
        velocity_mps: velocity,
        acceleration_mps2: acceleration,
        temporal_jerk: jerk,
        ..ConstraintResiduals::default()
    }
}

fn elapsed_us(started: Instant) -> u64 {
    #[cfg(feature = "deterministic")]
    {
        let _ = started;
        0
    }
    #[cfg(not(feature = "deterministic"))]
    {
        started.elapsed().as_micros().try_into().unwrap_or(u64::MAX)
    }
}
