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

use ruview_ontology::{Container, EvidenceLevel, Observation, ObservationId, SemanticProvenance, SensorId};

use crate::descriptor::{SamplingSpec, SensorDescriptor};
use crate::label::CapabilityTag;
use crate::modality::Modality;
use crate::observation::{HalObservation, Uncertainty};

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
        let observation = synthetic_observation(self.sensor_id.clone(), ctx, "synthetic-csi-adapter@0");

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
        let observation = synthetic_observation(self.sensor_id.clone(), ctx, "synthetic-imu-adapter@0");

        let finite = raw.accel.iter().chain(raw.gyro.iter()).all(|v| v.is_finite());

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
