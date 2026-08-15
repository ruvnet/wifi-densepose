//! Bounded runtime configuration.

use wifi_densepose_core::PhysicsMode;

/// Runtime limits. Untrusted frames cannot modify these values.
#[derive(Clone, Debug)]
pub struct PhysicsConfig {
    pub mode: PhysicsMode,
    pub max_tracks: usize,
    pub deadline_ms: u64,
    pub max_iterations: u8,
    pub max_joint_correction_m: f32,
    pub max_root_correction_m: f32,
    pub max_frame_gap_ms: u64,
    pub track_reset_gap_ms: u64,
    pub stale_after_ms: u64,
    pub minimum_joint_confidence: f32,
    pub minimum_known_joints: usize,
    pub calibration_joint_confidence: f32,
    pub solver_epsilon_m: f32,
    pub max_velocity_mps: f32,
    pub max_acceleration_mps2: f32,
    pub room_bound_m: f32,
    pub image_coordinate_bound: f32,
    pub beta_residual: f32,
    pub beta_intervention: f32,
    pub dynamics_audit: bool,
    pub dynamics_substeps: u8,
    pub dynamics_dt_seconds: f32,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            mode: PhysicsMode::Audit,
            max_tracks: 4,
            deadline_ms: 5,
            max_iterations: 4,
            max_joint_correction_m: 0.20,
            max_root_correction_m: 0.10,
            max_frame_gap_ms: 250,
            track_reset_gap_ms: 500,
            stale_after_ms: 500,
            minimum_joint_confidence: 0.10,
            minimum_known_joints: 10,
            calibration_joint_confidence: 0.70,
            solver_epsilon_m: 0.001,
            max_velocity_mps: 8.0,
            max_acceleration_mps2: 40.0,
            room_bound_m: 100.0,
            image_coordinate_bound: 16_384.0,
            beta_residual: 1.0,
            beta_intervention: 2.0,
            dynamics_audit: false,
            dynamics_substeps: 2,
            dynamics_dt_seconds: 1.0 / 30.0,
        }
    }
}

impl PhysicsConfig {
    /// Validate operator-controlled limits before activation.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.max_tracks == 0 || self.max_tracks > 256 {
            return Err("max_tracks must be in 1..=256");
        }
        if self.max_iterations == 0 || self.max_iterations > 8 {
            return Err("max_iterations must be in 1..=8");
        }
        if self.deadline_ms == 0 || self.deadline_ms > 1000 {
            return Err("deadline_ms must be in 1..=1000");
        }
        let finite_positive = [
            self.max_joint_correction_m,
            self.max_root_correction_m,
            self.solver_epsilon_m,
            self.max_velocity_mps,
            self.max_acceleration_mps2,
            self.room_bound_m,
            self.image_coordinate_bound,
            self.beta_residual,
            self.beta_intervention,
            self.dynamics_dt_seconds,
        ];
        if finite_positive.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("numeric limits must be finite and positive");
        }
        if self.image_coordinate_bound > 1_000_000.0 {
            return Err("image coordinate bound is excessive");
        }
        if !(0.0..=1.0).contains(&self.minimum_joint_confidence)
            || !(0.0..=1.0).contains(&self.calibration_joint_confidence)
        {
            return Err("confidence limits must be in [0, 1]");
        }
        if !(1..=17).contains(&self.minimum_known_joints) {
            return Err("minimum_known_joints must be in 1..=17");
        }
        if self.max_frame_gap_ms == 0
            || self.track_reset_gap_ms < self.max_frame_gap_ms
            || self.stale_after_ms < self.max_frame_gap_ms
        {
            return Err("frame/reset/stale gaps must be ordered and non-zero");
        }
        if self.max_root_correction_m > self.max_joint_correction_m
            || self.solver_epsilon_m >= self.max_joint_correction_m
        {
            return Err("correction limits are inconsistent");
        }
        if self.calibration_joint_confidence < self.minimum_joint_confidence {
            return Err("calibration confidence must meet the joint-use threshold");
        }
        if self.dynamics_audit && !cfg!(feature = "dynamics") {
            return Err("dynamics_audit requires the dynamics Cargo feature");
        }
        if !(1..=8).contains(&self.dynamics_substeps)
            || !self.dynamics_dt_seconds.is_finite()
            || !(1.0 / 240.0..=0.05).contains(&self.dynamics_dt_seconds)
        {
            return Err("dynamics limits are invalid");
        }
        Ok(())
    }

    /// Stable hash used in idempotency and provenance.
    #[must_use]
    pub fn canonical_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"ruview.pose-physics-config-v1\0");
        h.update(&[
            self.mode as u8,
            self.max_iterations,
            self.dynamics_audit.into(),
            self.dynamics_substeps,
        ]);
        for value in [
            self.max_tracks as u64,
            self.deadline_ms,
            self.max_frame_gap_ms,
            self.track_reset_gap_ms,
            self.stale_after_ms,
            self.minimum_known_joints as u64,
        ] {
            h.update(&value.to_le_bytes());
        }
        for value in [
            self.max_joint_correction_m,
            self.max_root_correction_m,
            self.minimum_joint_confidence,
            self.calibration_joint_confidence,
            self.solver_epsilon_m,
            self.max_velocity_mps,
            self.max_acceleration_mps2,
            self.room_bound_m,
            self.image_coordinate_bound,
            self.beta_residual,
            self.beta_intervention,
            self.dynamics_dt_seconds,
        ] {
            h.update(&value.to_bits().to_le_bytes());
        }
        *h.finalize().as_bytes()
    }
}
