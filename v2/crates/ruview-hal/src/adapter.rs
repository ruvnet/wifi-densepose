//! The [`SensorHal`] trait (ADR-320 §1) and two deterministic reference
//! adapters.
//!
//! The trait is the extension point: every sensing modality lands as one
//! `SensorHal` implementation instead of a bespoke ingest pipeline. It has
//! exactly two responsibilities — [`describe`](SensorHal::describe) the device
//! in canonical terms, and [`normalize`](SensorHal::normalize) one native raw
//! sample into a [`HalObservation`]. `normalize` is the hardware/FFI boundary
//! where untrusted input is validated (CLAUDE.md); it is **infallible** by
//! design — malformed or out-of-bounds input yields an UNKNOWN-flagged
//! observation, never a panic or an error (ADR-300 rule 1).
//!
//! Two reference adapters ship here, one RF (CSI) and one non-RF (IMU), per the
//! ADR-320 validation requirement of at least two modalities. Both are labelled
//! SYNTHETIC / L0: they prove the abstraction, not a fielded device, and make
//! no MEASURED claim (CLAUDE.md; ADR-320 "Category and honesty discipline").

use ruview_ontology::{
    Container, EvidenceLevel, Observation, ObservationId, SemanticProvenance, SensorId,
};

use crate::descriptor::{SamplingSpec, SensorDescriptor};
use crate::label::CapabilityTag;
use crate::modality::Modality;
use crate::observation::{HalObservation, Uncertainty};

/// Maximum number of depth pixels accepted from one Apple scene-depth packet.
/// The mobile producer currently sends 128×96; this larger bound leaves room
/// for future devices while keeping hostile allocation and compute bounded.
pub const MAX_LIDAR_DEPTH_SAMPLES: usize = 262_144;

/// One native ARKit scene-depth sample. Values remain in the device's native
/// millimetre/confidence representation until this HAL boundary validates them.
#[derive(Clone, Debug, PartialEq)]
pub struct AppleLidarDepthSample {
    /// Downsampled depth image width.
    pub width: u32,
    /// Downsampled depth image height.
    pub height: u32,
    /// Little-endian values decoded by the wire boundary, in millimetres.
    pub millimeters: Vec<u16>,
    /// ARKit confidence values (`0`, `1`, or `2`), one per depth value.
    pub confidences: Vec<u8>,
    /// Camera intrinsics in column-major order.
    pub intrinsics: [f32; 9],
    /// Camera-to-world transform in column-major order.
    pub camera_transform: [f32; 16],
    /// Content-addressed evidence handle over the received packet bytes.
    pub evidence_hash: String,
}

/// Normalizes an authenticated Apple ARKit scene-depth stream. A valid frame
/// is L2 single-surface evidence: it measures directly visible surfaces only
/// and does not establish person presence or any NLOS fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AppleLidarDepthAdapter {
    /// Authenticated ontology sensor identity.
    pub sensor_id: SensorId,
    /// Spatial calibration artifact/digest active for this capture.
    pub calibration_version: String,
}

impl SensorHal for AppleLidarDepthAdapter {
    type Raw = AppleLidarDepthSample;

    fn describe(&self) -> SensorDescriptor {
        SensorDescriptor {
            sensor_id: self.sensor_id.clone(),
            modality: Modality::Lidar,
            capabilities: vec![
                CapabilityTag::new("direct-depth").expect("static tag is valid"),
                CapabilityTag::new("confidence").expect("static tag is valid"),
                CapabilityTag::new("world-pose").expect("static tag is valid"),
            ],
            sampling: SamplingSpec {
                sample_rate_hz: None,
                unit: "millimeters".to_string(),
                dimensions: 0,
            },
        }
    }

    fn normalize(&self, raw: Self::Raw, ctx: &NormalizeCtx) -> HalObservation {
        let expected = usize::try_from(raw.width).ok().and_then(|w| {
            usize::try_from(raw.height)
                .ok()
                .and_then(|h| w.checked_mul(h))
        });
        let finite_geometry = raw
            .intrinsics
            .iter()
            .chain(raw.camera_transform.iter())
            .all(|v| v.is_finite());
        let malformed = expected.is_none()
            || expected == Some(0)
            || expected.is_some_and(|count| count > MAX_LIDAR_DEPTH_SAMPLES)
            || expected != Some(raw.millimeters.len())
            || raw.confidences.len() != raw.millimeters.len()
            || raw.confidences.iter().any(|&value| value > 2)
            || !finite_geometry
            || raw.evidence_hash.is_empty()
            || self.calibration_version.is_empty();

        let valid_count = raw
            .millimeters
            .iter()
            .zip(&raw.confidences)
            .filter(|(millimeters, confidence)| {
                **millimeters >= 150 && **millimeters <= 12_000 && **confidence >= 1
            })
            .count();
        let confidence = expected
            .filter(|&count| count > 0)
            .map(|count| valid_count as f64 / count as f64)
            .unwrap_or(0.0);

        let observation = Observation {
            id: ctx.observation_id.clone(),
            sensor: self.sensor_id.clone(),
            located_in: ctx.located_in.clone(),
            at_unix_ms: ctx.at_unix_ms,
            evidence_level: if malformed {
                EvidenceLevel::L0
            } else {
                EvidenceLevel::L2
            },
            provenance: SemanticProvenance {
                evidence: if malformed {
                    Vec::new()
                } else {
                    vec![raw.evidence_hash]
                },
                model_version: "apple-arkit-scene-depth@1".to_string(),
                calibration_version: self.calibration_version.clone(),
                privacy_decision: "p1-derived-visible-geometry".to_string(),
            },
        };

        HalObservation {
            modality: Modality::Lidar,
            uncertainty: if malformed || valid_count == 0 {
                Uncertainty::degraded()
            } else {
                Uncertainty::known(confidence)
            },
            observation,
        }
    }
}

/// Injected context a HAL adapter needs to build a canonical observation.
///
/// Identity, placement, and time are supplied by the caller — the HAL never
/// mints ids or reads a wall clock (deterministic; time is injected, mirroring
/// the ontology's `at_unix_ms` contract).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormalizeCtx {
    /// Caller-supplied stable id for the observation to be produced.
    pub observation_id: ObservationId,
    /// Where the observation is located (resolved against the ontology graph).
    pub located_in: Container,
    /// Injected capture timestamp (Unix ms). Never sampled from a clock here.
    pub at_unix_ms: i64,
}

/// The hardware abstraction: map any sensing modality to one canonical
/// observation.
///
/// Implementations wrap existing producers — CSI (ESP32/Nexmon/FeitCSI via the
/// ADR-279 `RfFrameV2` path), 802.11bf (ADR-310), BLE, UWB, mmWave (ADR-063),
/// acoustic, camera, lidar, IMU, and `custom` — behind this single trait, so
/// the world model and fusion (ADR-311) see only [`HalObservation`]s.
pub trait SensorHal {
    /// The native, modality-specific raw sample type this adapter consumes.
    /// Kept native (not canonicalized) per the ADR-279 shared-latent lesson.
    type Raw;

    /// Describe this device in canonical terms.
    fn describe(&self) -> SensorDescriptor;

    /// Normalize one native raw sample into a canonical [`HalObservation`].
    ///
    /// Infallible: malformed / out-of-bounds input produces an UNKNOWN-flagged,
    /// `degraded` observation rather than panicking or erroring.
    fn normalize(&self, raw: Self::Raw, ctx: &NormalizeCtx) -> HalObservation;
}

/// Build the canonical ontology observation shared by every reference adapter.
///
/// Reference adapters are synthetic, so the evidence level is pinned to
/// [`EvidenceLevel::L0`] and the provenance carries the synthetic calibration
/// handle — the fact can never alias to a measured/calibrated observation.
fn synthetic_observation(sensor: SensorId, ctx: &NormalizeCtx, model_version: &str) -> Observation {
    Observation {
        id: ctx.observation_id.clone(),
        sensor,
        located_in: ctx.located_in.clone(),
        at_unix_ms: ctx.at_unix_ms,
        evidence_level: EvidenceLevel::L0,
        provenance: synthetic_provenance(model_version),
    }
}

/// A provenance record stamped SYNTHETIC via its calibration handle, so
/// [`HalObservation::is_synthetic`] is true and the fact cannot look calibrated.
#[must_use]
pub fn synthetic_provenance(model_version: impl Into<String>) -> SemanticProvenance {
    SemanticProvenance {
        evidence: Vec::new(),
        model_version: model_version.into(),
        calibration_version: crate::SYNTHETIC_CALIBRATION.to_string(),
        privacy_decision: "synthetic".to_string(),
    }
}

/// Maximum CSI taps a reference adapter will read, bounding allocation/compute
/// on untrusted input.
pub const MAX_CSI_TAPS: usize = 4096;

/// A native CSI raw sample: per-subcarrier amplitude and phase.
///
/// This is the *native* frame the adapter keeps — the pipeline never sees it,
/// only the [`HalObservation`] it is lifted into.
#[derive(Clone, Debug, PartialEq)]
pub struct CsiSample {
    /// Per-subcarrier amplitudes (linear).
    pub amplitudes: Vec<f32>,
    /// Per-subcarrier phases (radians).
    pub phases: Vec<f32>,
}

/// A deterministic, synthetic CSI reference adapter (SYNTHETIC / L0).
///
/// Mirrors the ADR-279 per-device latent adapters in shape without claiming any
/// real device: it demonstrates that a CSI producer lifts into the canonical
/// observation. It makes no MEASURED claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntheticCsiAdapter {
    /// The ontology sensor identity this adapter is authenticated as.
    pub sensor_id: SensorId,
    /// Declared native subcarrier count.
    pub subcarriers: u32,
}

impl SensorHal for SyntheticCsiAdapter {
    type Raw = CsiSample;

    fn describe(&self) -> SensorDescriptor {
        SensorDescriptor {
            sensor_id: self.sensor_id.clone(),
            modality: Modality::Csi,
            capabilities: vec![
                CapabilityTag::new("amplitude").expect("static tag is valid"),
                CapabilityTag::new("phase").expect("static tag is valid"),
            ],
            sampling: SamplingSpec {
                sample_rate_hz: Some(100.0),
                unit: "csi-complex".to_string(),
                dimensions: self.subcarriers,
            },
        }
    }

    fn normalize(&self, raw: Self::Raw, ctx: &NormalizeCtx) -> HalObservation {
        let observation =
            synthetic_observation(self.sensor_id.clone(), ctx, "synthetic-csi-adapter@0");

        // Boundary validation: empty, mismatched, over-bounded, or non-finite
        // input degrades to UNKNOWN rather than panicking or fabricating a
        // confident value.
        let malformed = raw.amplitudes.is_empty()
            || raw.amplitudes.len() != raw.phases.len()
            || raw.amplitudes.len() > MAX_CSI_TAPS
            || raw.amplitudes.iter().any(|v| !v.is_finite())
            || raw.phases.iter().any(|v| !v.is_finite());

        let uncertainty = if malformed {
            Uncertainty::degraded()
        } else {
            // Deterministic confidence from the mean amplitude, bounded to
            // [0, 1) by a saturating map. No randomness, no clock.
            let sum: f64 = raw.amplitudes.iter().map(|&v| f64::from(v).abs()).sum();
            let mean = sum / raw.amplitudes.len() as f64;
            Uncertainty::known(mean / (mean + 1.0))
        };

        HalObservation {
            modality: Modality::Csi,
            uncertainty,
            observation,
        }
    }
}

/// A native IMU raw sample: 3-axis acceleration and angular rate.
#[derive(Clone, Debug, PartialEq)]
pub struct ImuSample {
    /// Acceleration `[x, y, z]` in m/s².
    pub accel: [f32; 3],
    /// Angular rate `[x, y, z]` in rad/s.
    pub gyro: [f32; 3],
}

/// A deterministic, synthetic IMU reference adapter (SYNTHETIC / L0).
///
/// The required non-RF second modality (ADR-320 validation). Demonstrates that
/// a wholly different phenomenon class lifts into the *same* canonical
/// observation with its own honest evidence level — it is never lifted to
/// camera- or RF-grade.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntheticImuAdapter {
    /// The ontology sensor identity this adapter is authenticated as.
    pub sensor_id: SensorId,
}

impl SensorHal for SyntheticImuAdapter {
    type Raw = ImuSample;

    fn describe(&self) -> SensorDescriptor {
        SensorDescriptor {
            sensor_id: self.sensor_id.clone(),
            modality: Modality::Imu,
            capabilities: vec![
                CapabilityTag::new("accel").expect("static tag is valid"),
                CapabilityTag::new("gyro").expect("static tag is valid"),
            ],
            sampling: SamplingSpec {
                sample_rate_hz: Some(200.0),
                unit: "m/s^2|rad/s".to_string(),
                dimensions: 6,
            },
        }
    }

    fn normalize(&self, raw: Self::Raw, ctx: &NormalizeCtx) -> HalObservation {
        let observation =
            synthetic_observation(self.sensor_id.clone(), ctx, "synthetic-imu-adapter@0");

        let finite = raw
            .accel
            .iter()
            .chain(raw.gyro.iter())
            .all(|v| v.is_finite());

        let uncertainty = if !finite {
            Uncertainty::degraded()
        } else {
            // Deterministic confidence: how close the acceleration magnitude is
            // to 1 g (a stationary device). Bounded to [0, 1].
            let g: f64 = raw
                .accel
                .iter()
                .map(|&v| f64::from(v) * f64::from(v))
                .sum::<f64>()
                .sqrt();
            let closeness = 1.0 - ((g - 9.81).abs() / 9.81);
            Uncertainty::known(closeness)
        };

        HalObservation {
            modality: Modality::Imu,
            uncertainty,
            observation,
        }
    }
}
