//! Canonical metric pose observation contract for physics assessment (ADR-323).

#![allow(missing_docs)]

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(feature = "std")]
use std::string::String;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Schema version accepted by [`PoseObservationV2`].
pub const POSE_OBSERVATION_SCHEMA_VERSION: u16 = 2;

/// A validated probability in the inclusive range `[0, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "f32", into = "f32"))]
pub struct Probability(f32);

impl Probability {
    /// Construct a finite probability.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite values or values outside `[0, 1]`.
    pub fn new(value: f32) -> Result<Self, &'static str> {
        if value.is_finite() && (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err("probability must be finite and in [0, 1]")
        }
    }

    /// Return the underlying value.
    #[must_use]
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Zero probability.
    pub const ZERO: Self = Self(0.0);
    /// Unit probability.
    pub const ONE: Self = Self(1.0);
}

impl TryFrom<f32> for Probability {
    type Error = &'static str;
    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Probability> for f32 {
    fn from(value: Probability) -> Self {
        value.get()
    }
}

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        pub struct $name(pub String);
    };
}

string_id!(TrackId);
string_id!(CalibrationId);

/// Versioned coordinate frame. Metric correction requires every boolean here.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpatialFrameRef {
    /// Stable frame name.
    pub name: String,
    /// Frame definition version.
    pub version: u32,
    /// Coordinates are metres rather than image-normalized units.
    pub metric: bool,
    /// The axes form a right-handed system.
    pub right_handed: bool,
    /// Positive Z is vertical/up.
    pub z_up: bool,
}

/// Signed model identity.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ModelRef {
    /// Human-readable model/version identifier.
    pub id: String,
    /// Verified model artifact digest.
    pub artifact_hash: [u8; 32],
}

/// Sensor provenance relevant to correction selection.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SourceProvenance {
    /// Stable sensor identity.
    pub sensor_id: String,
    /// True only after message/source authentication succeeds.
    pub authenticated: bool,
    /// True only when sequence binding and replay-window checks are active.
    pub replay_protected: bool,
}

/// RF distribution trust state from ADR-302.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "SCREAMING_SNAKE_CASE"))]
pub enum PoseTrustState {
    /// Calibration/evidence says the sample is in distribution.
    Known,
    /// Input is usable for audit but confidence should be reduced.
    Degraded,
    /// Input is out-of-distribution or cannot be classified.
    Unknown,
}

/// Whether the observation is image-plane or physical 3D.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum PoseDimensionality {
    /// Normalized or pixel image coordinates; audit-only.
    Image2d,
    /// Metric room-frame X/Y/Z coordinates.
    Metric3d,
}

/// Normalized plane `normal dot point + offset_m = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FloorPlane {
    /// Upward-pointing unit normal.
    pub normal: [f32; 3],
    /// Plane offset in metres.
    pub offset_m: f32,
}

impl FloorPlane {
    /// Whether all values are finite and the normal is approximately unit length.
    #[must_use]
    pub fn is_valid(self) -> bool {
        if !self.normal.iter().all(|v| v.is_finite()) || !self.offset_m.is_finite() {
            return false;
        }
        let n2 = self.normal.iter().map(|v| v * v).sum::<f32>();
        (n2 - 1.0).abs() <= 1.0e-3 && self.normal[2] > 0.0
    }

    /// Signed distance to the plane in metres.
    #[must_use]
    pub fn signed_distance(self, point: [f32; 3]) -> f32 {
        self.normal[0].mul_add(
            point[0],
            self.normal[1].mul_add(point[1], self.normal[2].mul_add(point[2], self.offset_m)),
        )
    }
}

/// The six unique entries of a symmetric 3x3 covariance matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SymmetricCovariance3 {
    pub xx: f32,
    pub xy: f32,
    pub xz: f32,
    pub yy: f32,
    pub yz: f32,
    pub zz: f32,
}

impl SymmetricCovariance3 {
    /// Conservative positive-semidefinite validation using principal minors.
    #[must_use]
    #[allow(
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::suspicious_operation_groupings
    )]
    pub fn is_positive_semidefinite(self) -> bool {
        const EPS: f32 = 1.0e-8;
        let values = [self.xx, self.xy, self.xz, self.yy, self.yz, self.zz];
        if !values.iter().all(|v| v.is_finite()) {
            return false;
        }
        if self.xx < -EPS || self.yy < -EPS || self.zz < -EPS {
            return false;
        }
        let xy_minor = self.xx * self.yy - self.xy * self.xy;
        let xz_minor = self.xx * self.zz - self.xz * self.xz;
        let yz_minor = self.yy * self.zz - self.yz * self.yz;
        let det = self.xx * (self.yy * self.zz - self.yz * self.yz)
            - self.xy * (self.xy * self.zz - self.yz * self.xz)
            + self.xz * (self.xy * self.yz - self.yy * self.xz);
        xy_minor >= -EPS && xz_minor >= -EPS && yz_minor >= -EPS && det >= -EPS
    }

    /// Trace, used as a bounded uncertainty weight.
    #[must_use]
    pub fn trace(self) -> f32 {
        self.xx + self.yy + self.zz
    }
}

/// COCO-17 joint identity in canonical order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[repr(u8)]
pub enum Coco17Joint {
    Nose,
    LeftEye,
    RightEye,
    LeftEar,
    RightEar,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
}

impl Coco17Joint {
    /// Canonical COCO-17 order.
    pub const ALL: [Self; 17] = [
        Self::Nose,
        Self::LeftEye,
        Self::RightEye,
        Self::LeftEar,
        Self::RightEar,
        Self::LeftShoulder,
        Self::RightShoulder,
        Self::LeftElbow,
        Self::RightElbow,
        Self::LeftWrist,
        Self::RightWrist,
        Self::LeftHip,
        Self::RightHip,
        Self::LeftKnee,
        Self::RightKnee,
        Self::LeftAnkle,
        Self::RightAnkle,
    ];
}

/// Whether a joint is usable by the observer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum JointVisibility {
    Visible,
    Occluded,
    Unknown,
}

/// One observed COCO joint and calibrated uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct JointObservation {
    pub kind: Coco17Joint,
    pub position_m: [f32; 3],
    pub covariance_m2: SymmetricCovariance3,
    pub confidence: Probability,
    pub visibility: JointVisibility,
}

/// Immutable pose observation presented to the physics boundary.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PoseObservationV2 {
    pub schema_version: u16,
    pub timestamp_ns: u64,
    pub sensor_epoch: u64,
    pub sequence: u64,
    pub track_id: TrackId,
    pub frame: SpatialFrameRef,
    pub calibration_id: CalibrationId,
    pub floor_plane: Option<FloorPlane>,
    pub model: ModelRef,
    pub source: SourceProvenance,
    pub trust_state: PoseTrustState,
    pub dimensionality: PoseDimensionality,
    pub uncertainty_calibrated: bool,
    pub joints: [JointObservation; 17],
    pub observer_confidence: Probability,
    pub canonical_hash: [u8; 32],
}

impl PoseObservationV2 {
    /// Compute a deterministic content hash excluding `canonical_hash` itself.
    #[must_use]
    pub fn compute_canonical_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"ruview.pose-observation-v2\0");
        h.update(&self.schema_version.to_le_bytes());
        h.update(&self.timestamp_ns.to_le_bytes());
        h.update(&self.sensor_epoch.to_le_bytes());
        h.update(&self.sequence.to_le_bytes());
        hash_str(&mut h, &self.track_id.0);
        hash_str(&mut h, &self.frame.name);
        h.update(&self.frame.version.to_le_bytes());
        h.update(&[
            self.frame.metric.into(),
            self.frame.right_handed.into(),
            self.frame.z_up.into(),
        ]);
        hash_str(&mut h, &self.calibration_id.0);
        match self.floor_plane {
            Some(plane) => {
                h.update(&[1]);
                for value in plane.normal {
                    h.update(&value.to_bits().to_le_bytes());
                }
                h.update(&plane.offset_m.to_bits().to_le_bytes());
            }
            None => {
                h.update(&[0]);
            }
        }
        hash_str(&mut h, &self.model.id);
        h.update(&self.model.artifact_hash);
        hash_str(&mut h, &self.source.sensor_id);
        h.update(&[
            self.source.authenticated.into(),
            self.source.replay_protected.into(),
        ]);
        h.update(&[
            self.trust_state as u8,
            self.dimensionality as u8,
            self.uncertainty_calibrated.into(),
        ]);
        for joint in &self.joints {
            h.update(&[joint.kind as u8, joint.visibility as u8]);
            for value in joint.position_m {
                h.update(&value.to_bits().to_le_bytes());
            }
            for value in [
                joint.covariance_m2.xx,
                joint.covariance_m2.xy,
                joint.covariance_m2.xz,
                joint.covariance_m2.yy,
                joint.covariance_m2.yz,
                joint.covariance_m2.zz,
            ] {
                h.update(&value.to_bits().to_le_bytes());
            }
            h.update(&joint.confidence.get().to_bits().to_le_bytes());
        }
        h.update(&self.observer_confidence.get().to_bits().to_le_bytes());
        *h.finalize().as_bytes()
    }

    /// Set `canonical_hash` to the deterministic content hash.
    pub fn seal(&mut self) {
        self.canonical_hash = self.compute_canonical_hash();
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_rejects_non_finite_and_out_of_range() {
        assert!(Probability::new(f32::NAN).is_err());
        assert!(Probability::new(-0.1).is_err());
        assert!(Probability::new(1.1).is_err());
        assert!((Probability::new(0.5).unwrap().get() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn covariance_rejects_negative_principal_minor() {
        let covariance = SymmetricCovariance3 {
            xx: 0.1,
            xy: 1.0,
            xz: 0.0,
            yy: 0.1,
            yz: 0.0,
            zz: 0.1,
        };
        assert!(!covariance.is_positive_semidefinite());
    }
}
