//! Calibrated CSI spatial prior for optical NLOS particle fusion.

use thiserror::Error;

use crate::protocol::{EvidenceLevel, FrameSource, Vec3};

const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

/// Exact policy and coordinate scope required before two modalities may be
/// joined. The v1 track envelope cannot retain this complete lineage, so the
/// current tracker uses the binding only for L0 synthetic architecture tests
/// and rejects measured fusion until a lineage-preserving wire revision lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusionScope {
    /// Owning tenant.
    pub tenant_id: String,
    /// Owning workspace.
    pub workspace_id: String,
    /// Physical deployment site.
    pub site_id: String,
    /// Calibrated common coordinate frame.
    pub world_frame_id: String,
    /// Capture session shared by both modalities.
    pub session_id: String,
    /// Digest of the accepted CSI-to-world coordinate transform.
    pub coordinate_transform_hash: String,
}

impl FusionScope {
    /// Validate bounded identifiers and the non-zero transform digest.
    pub fn validate(&self) -> Result<(), FusionError> {
        for value in [
            &self.tenant_id,
            &self.workspace_id,
            &self.site_id,
            &self.world_frame_id,
            &self.session_id,
        ] {
            if !valid_label(value) {
                return Err(FusionError::InvalidBinding);
            }
        }
        if !valid_hash(&self.coordinate_transform_hash)
            || self.coordinate_transform_hash == "0".repeat(64)
        {
            return Err(FusionError::InvalidBinding);
        }
        Ok(())
    }
}

/// A coarse RF spatial prior. CSI is never treated as centimetre-scale ground
/// truth; its covariance and confidence explicitly bound its influence.
#[derive(Clone, Debug)]
pub struct CsiSpatialPrior {
    /// CSI source classification.
    pub source: FrameSource,
    /// Monotonic sequence within the CSI capture session.
    pub sequence: u64,
    /// UTC time of the RF observation.
    pub captured_at_unix_ms: u64,
    /// Tenant/session/world-frame binding for the temporal-spatial join.
    pub scope: FusionScope,
    /// Coarse region mean in the NLOS world frame.
    pub mean_m: Vec3,
    /// Diagonal covariance in square metres.
    pub covariance_diagonal_m2: Vec3,
    /// Bounded RF confidence.
    pub confidence: f32,
    /// Evidence level of the RF observation.
    pub evidence_level: EvidenceLevel,
    /// Authenticated RF sensor identifier.
    pub sensor_id: String,
    /// RF calibration digest.
    pub calibration_hash: String,
}

impl CsiSpatialPrior {
    /// Validate all values before the prior can affect optical particles.
    pub fn validate(&self) -> Result<(), FusionError> {
        self.scope.validate()?;
        if self.sequence > MAX_SAFE_INTEGER
            || self.captured_at_unix_ms > MAX_SAFE_INTEGER
            || !self.mean_m.finite()
            || [self.mean_m.x, self.mean_m.y, self.mean_m.z]
                .iter()
                .any(|value| value.abs() > 100.0)
            || !self.covariance_diagonal_m2.finite()
            || [
                self.covariance_diagonal_m2.x,
                self.covariance_diagonal_m2.y,
                self.covariance_diagonal_m2.z,
            ]
            .iter()
            .any(|value| !(0.0025..=25.0).contains(value))
            || !self.confidence.is_finite()
            || !(0.0..=1.0).contains(&self.confidence)
        {
            return Err(FusionError::InvalidPrior);
        }
        if !valid_label(&self.sensor_id) || !valid_hash(&self.calibration_hash) {
            return Err(FusionError::InvalidPrior);
        }
        if (self.source == FrameSource::Synthetic)
            != (self.evidence_level == EvidenceLevel::L0Synthetic)
            || (self.source == FrameSource::Synthetic && self.calibration_hash != "0".repeat(64))
        {
            return Err(FusionError::EvidenceMismatch);
        }
        if self.source != FrameSource::Synthetic
            && self.evidence_level == EvidenceLevel::L0Synthetic
        {
            return Err(FusionError::EvidenceMismatch);
        }
        if self.evidence_level >= EvidenceLevel::L2Calibrated
            && self.calibration_hash == "0".repeat(64)
        {
            return Err(FusionError::EvidenceMismatch);
        }
        Ok(())
    }

    pub(crate) fn log_likelihood(&self, point: Vec3) -> f32 {
        let d = point.minus(self.mean_m);
        let mahalanobis = d.x * d.x / self.covariance_diagonal_m2.x
            + d.y * d.y / self.covariance_diagonal_m2.y
            + d.z * d.z / self.covariance_diagonal_m2.z;
        -0.5 * mahalanobis * self.confidence
    }
}

fn valid_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':'))
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// CSI fusion boundary failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FusionError {
    /// A field was non-finite, out of bounds, or malformed.
    #[error("invalid CSI spatial prior")]
    InvalidPrior,
    /// Synthetic evidence carried a non-synthetic calibration identity.
    #[error("CSI source and evidence labels disagree")]
    EvidenceMismatch,
    /// The two modalities do not share the exact authorized join scope.
    #[error("CSI and optical fusion bindings do not match")]
    BindingMismatch,
    /// Measured fusion is blocked until the output contract retains both
    /// modality lineages and the deployment supplies authenticated bindings.
    #[error("measured CSI fusion is unavailable in the v1 evidence contract")]
    MeasuredFusionUnavailable,
    /// A fusion binding is malformed or missing required identity.
    #[error("invalid fusion scope binding")]
    InvalidBinding,
    /// A CSI observation was repeated or moved backwards.
    #[error("replayed or out-of-order CSI prior")]
    ReplayOrOutOfOrder,
    /// Prior and optical frame are too far apart in time.
    #[error("CSI prior is stale")]
    Stale,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibrated_prior_prefers_nearby_particle() {
        let p = CsiSpatialPrior {
            source: FrameSource::Live,
            sequence: 1,
            captured_at_unix_ms: 1,
            scope: FusionScope {
                tenant_id: "tenant-1".into(),
                workspace_id: "workspace-1".into(),
                site_id: "site-1".into(),
                world_frame_id: "world-1".into(),
                session_id: "session-1".into(),
                coordinate_transform_hash: "b".repeat(64),
            },
            mean_m: Vec3::new(0.0, 0.0, 1.0),
            covariance_diagonal_m2: Vec3::new(0.04, 0.04, 0.09),
            confidence: 0.8,
            evidence_level: EvidenceLevel::L2Calibrated,
            sensor_id: "csi-1".into(),
            calibration_hash: "a".repeat(64),
        };
        p.validate().unwrap();
        assert!(
            p.log_likelihood(Vec3::new(0.01, 0.0, 1.0))
                > p.log_likelihood(Vec3::new(1.0, 0.0, 1.0))
        );
    }

    #[test]
    fn prior_rejects_synthetic_evidence_on_a_live_source() {
        let mut p = CsiSpatialPrior {
            source: FrameSource::Live,
            sequence: 1,
            captured_at_unix_ms: 1,
            scope: FusionScope {
                tenant_id: "tenant-1".into(),
                workspace_id: "workspace-1".into(),
                site_id: "site-1".into(),
                world_frame_id: "world-1".into(),
                session_id: "session-1".into(),
                coordinate_transform_hash: "b".repeat(64),
            },
            mean_m: Vec3::new(0.0, 0.0, 1.0),
            covariance_diagonal_m2: Vec3::new(0.04, 0.04, 0.09),
            confidence: 0.8,
            evidence_level: EvidenceLevel::L0Synthetic,
            sensor_id: "csi-1".into(),
            calibration_hash: "0".repeat(64),
        };
        assert_eq!(p.validate(), Err(FusionError::EvidenceMismatch));
        p.source = FrameSource::Synthetic;
        assert!(p.validate().is_ok());
    }
}
