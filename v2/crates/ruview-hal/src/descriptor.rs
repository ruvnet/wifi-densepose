//! The sensor descriptor (ADR-320 §1): what a device is, in canonical terms.
//!
//! A [`SensorDescriptor`] binds a HAL implementation to its ontology
//! [`Sensor`](ruview_ontology::Sensor) identity, its [`Modality`], the
//! capability tags it advertises, and the native sampling/units metadata of its
//! raw frame. The native frame is described, not canonicalized: per the ADR-279
//! shared-latent lesson, premature canonicalization discards information
//! (bandwidth, antenna structure, phase), so the descriptor records the native
//! shape and the adapter lifts it into an [`Observation`](ruview_ontology::Observation)
//! only at [`normalize`](crate::SensorHal::normalize) time.

use serde::{Deserialize, Serialize};

use ruview_ontology::SensorId;

use crate::label::CapabilityTag;
use crate::modality::Modality;

/// Native sampling and unit metadata for a sensor's raw frame.
///
/// This is descriptive, not prescriptive: it records how the device natively
/// produces samples so downstream stages can interpret provenance, without the
/// pipeline ever having to understand the raw frame itself.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamplingSpec {
    /// Native sampling rate in Hz when fixed/known. `None` is a first-class
    /// UNKNOWN — an event-driven or unspecified source is not an error
    /// (ADR-300 rule 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<f64>,
    /// Native unit label for one raw sample (e.g. `"csi-complex"`, `"m/s^2"`,
    /// `"dBm"`). Descriptive free-form metadata, not a parsed quantity.
    pub unit: String,
    /// Native dimensionality of one raw frame (e.g. subcarriers × antennas, or
    /// IMU axes). `0` means unknown.
    pub dimensions: u32,
}

/// A canonical description of one sensing device.
///
/// Round-trips losslessly through serde so a fleet controller (ADR-316) can
/// enumerate heterogeneous hardware uniformly. The `sensor_id` is the ontology
/// identity the device is authenticated as (ADR-305) before its observations
/// are trusted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SensorDescriptor {
    /// The ontology sensor identity this device is authenticated as (ADR-305).
    pub sensor_id: SensorId,
    /// What phenomenon class the device senses.
    pub modality: Modality,
    /// Capability tags — the phenomena the device advertises it can observe.
    #[serde(default)]
    pub capabilities: Vec<CapabilityTag>,
    /// Native sampling / units metadata for the raw frame.
    pub sampling: SamplingSpec,
}

impl SensorDescriptor {
    /// Re-validate a descriptor received from an untrusted source. Checks the
    /// modality label bounds; ids and capability tags are validated when
    /// constructed. Returns UNKNOWN-friendly `Ok(())` for any well-formed
    /// descriptor.
    pub fn validate(&self) -> Result<(), crate::label::LabelError> {
        self.modality.validate()
    }
}
