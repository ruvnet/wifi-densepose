//! Named reference sources on the validation plane (ADR-300 §1).
//!
//! A reference source is an *independent observer* used only to check RF
//! inference — never an inference input (ADR-300 Decision, option 1 rejected).
//! It carries the modality, a named source, device metadata, and the recorded
//! measurement principle so a MEASURED claim states what it was measured
//! against.

use serde::{Deserialize, Serialize};

use crate::error::{check_bound, check_nonempty, GroundTruthError};

/// The modality of an independent reference. Camera/mmWave references arrive as
/// exported label/keypoint streams, not live model feeds (ADR-300 §1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceModality {
    /// Optical camera (exported labels/keypoints).
    Camera,
    /// mmWave radar (exported detections/point cloud).
    MmWave,
    /// Pressure mat / floor sensor.
    Pressure,
    /// Body-worn wearable (e.g. chest strap, IMU).
    Wearable,
    /// Pulse oximeter.
    PulseOximeter,
    /// Microphone (acoustic reference).
    Microphone,
    /// A human-provided manual label.
    ManualLabel,
}

impl ReferenceModality {
    /// A stable, human-readable tag.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            ReferenceModality::Camera => "camera",
            ReferenceModality::MmWave => "mmwave",
            ReferenceModality::Pressure => "pressure",
            ReferenceModality::Wearable => "wearable",
            ReferenceModality::PulseOximeter => "pulse_oximeter",
            ReferenceModality::Microphone => "microphone",
            ReferenceModality::ManualLabel => "manual_label",
        }
    }

    /// Whether this modality constitutes an *independent* ground-truth
    /// reference. Every modality here is independent of the RF estimator — that
    /// independence is exactly what makes a MEASURED grade admissible. Kept as
    /// a method so the grading rule reads intentionally rather than assuming.
    #[must_use]
    pub const fn is_independent_reference(self) -> bool {
        true
    }
}

/// A named reference source: modality plus device/source metadata and the
/// measurement principle. Validated at construction so untrusted metadata is
/// bounded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceSource {
    /// The reference modality.
    pub modality: ReferenceModality,
    /// A named source (e.g. `"ceiling-cam-1"`, `"chest-strap-A"`).
    pub name: String,
    /// Device make/model.
    pub device: String,
    /// The recorded measurement principle (e.g. `"ppg"`, `"tof-depth"`); may be
    /// empty when not applicable, but is length-bounded.
    pub principle: String,
}

impl ReferenceSource {
    /// Construct a reference source, validating metadata at the boundary.
    /// `name` and `device` must be non-empty; all fields are length-bounded.
    ///
    /// # Errors
    /// [`GroundTruthError::EmptyField`] for a missing `name`/`device`;
    /// [`GroundTruthError::TooLong`] for any over-length field.
    pub fn new(
        modality: ReferenceModality,
        name: impl Into<String>,
        device: impl Into<String>,
        principle: impl Into<String>,
    ) -> Result<Self, GroundTruthError> {
        let name = name.into();
        let device = device.into();
        let principle = principle.into();

        check_bound("name", &name)?;
        check_bound("device", &device)?;
        check_bound("principle", &principle)?;
        check_nonempty("name", &name)?;
        check_nonempty("device", &device)?;

        Ok(Self {
            modality,
            name,
            device,
            principle,
        })
    }
}
