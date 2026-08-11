//! The sensing modality tag (ADR-317 §1).
//!
//! [`Modality`] enumerates the phenomenon class a sensor measures. It is the
//! only place the pipeline distinguishes "how the world was sensed"; every
//! modality flows through the same [`SensorHal`](crate::SensorHal) trait into
//! the same canonical [`Observation`](ruview_ontology::Observation), so the
//! world model never branches on a modality-specific frame shape (ADR-297 rule
//! 3: one canonical semantics downstream).

use serde::{Deserialize, Serialize};

use crate::label::{validate_label, LabelError};

/// The class of physical phenomenon a sensor observes.
///
/// The closed variants cover the modalities named in ADR-317; [`Modality::Custom`]
/// is the open extension point for a modality not yet enumerated, carrying a
/// validated free-form label. `Custom` is validated with [`Modality::custom`]
/// (or [`Modality::validate`]) at the boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// WiFi channel-state information (ESP32/Nexmon/FeitCSI via ADR-279).
    Csi,
    /// IEEE 802.11bf native sensing (ADR-307, phase 2).
    Ieee80211bf,
    /// Bluetooth Low Energy ranging / RSSI.
    Ble,
    /// Ultra-wideband ranging.
    Uwb,
    /// Millimetre-wave radar (ADR-063).
    Mmwave,
    /// Acoustic / ultrasonic sensing.
    Acoustic,
    /// Optical camera.
    Camera,
    /// Lidar point cloud.
    Lidar,
    /// Inertial measurement unit (accelerometer + gyroscope).
    Imu,
    /// An open-ended modality carrying a validated label.
    Custom(String),
}

impl Modality {
    /// Construct a validated [`Modality::Custom`], rejecting empty, over-long,
    /// or control-character labels at the boundary.
    pub fn custom(raw: impl Into<String>) -> Result<Self, LabelError> {
        let s = raw.into();
        validate_label(&s)?;
        Ok(Self::Custom(s))
    }

    /// Re-validate a modality received from an untrusted source (e.g. after
    /// deserialization). Closed variants are always valid; a `Custom` payload
    /// must satisfy the label bounds.
    pub fn validate(&self) -> Result<(), LabelError> {
        match self {
            Self::Custom(s) => validate_label(s),
            _ => Ok(()),
        }
    }

    /// A stable lowercase label for this modality, matching its serialized tag.
    /// For [`Modality::Custom`] this is the inner label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Csi => "csi",
            Self::Ieee80211bf => "ieee80211bf",
            Self::Ble => "ble",
            Self::Uwb => "uwb",
            Self::Mmwave => "mmwave",
            Self::Acoustic => "acoustic",
            Self::Camera => "camera",
            Self::Lidar => "lidar",
            Self::Imu => "imu",
            Self::Custom(s) => s,
        }
    }

    /// True for radio-frequency modalities, which reuse the ADR-279 native RF
    /// frame / shared-latent adapters wholesale.
    #[must_use]
    pub fn is_rf(&self) -> bool {
        matches!(
            self,
            Self::Csi | Self::Ieee80211bf | Self::Ble | Self::Uwb | Self::Mmwave
        )
    }
}
