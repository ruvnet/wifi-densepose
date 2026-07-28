//! Native RF frame contract — `RfFrameV2` (ADR-279).
//!
//! The ADR-273 P1 canonical tensor (`RfTensor`, 56 bins × 8 snapshots) is
//! useful for compatibility, but it must **not** be the authoritative data
//! format: resampling every device into one fixed tensor discards
//! bandwidth, antenna, phase-state, and hardware-specific information.
//! `RfFrameV2` preserves the **native complex tensor** with explicit
//! validity masks, phase state, geometry, calibration, quality, and
//! provenance; the canonical tensor is demoted to a *derived view*
//! ([`RfFrameV2::to_canonical`]) computed on demand and never written back.
//!
//! Required invariants (ADR-279 §2, each enforced by a constructor check or
//! a test):
//! 1. Native complex samples are never overwritten by normalized samples
//!    (`to_canonical` takes `&self`; test proves byte-stability).
//! 2. Subcarrier/antenna masks are explicit (`valid_mask`).
//! 3. Phase declares its state: raw, sanitized, calibrated, or unavailable.
//! 4. TX/RX geometry uses one building coordinate frame (`Pose3`).
//! 5. Results retain source frame ids + model version (via `receipt_id`).
//! 6. Synthetic and measured frames can never share a provenance class,
//!    and a synthetic frame can never claim evidence above L0.
//! 7. Sample age is carried through the whole inference path.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::tensor::{LinkGeometry, RfModality, RfTensor};
use crate::{Result, UnifiedError};

/// Current schema version of [`RfFrameV2`].
pub const SCHEMA_VERSION: u16 = 2;

/// A pose in the building coordinate frame (metres, unit quaternion).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pose3 {
    /// Position `[x, y, z]`, metres.
    pub position_m: [f64; 3],
    /// Orientation quaternion `[w, x, y, z]`.
    pub orientation: [f64; 4],
}

/// One antenna element of an array.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AntennaElement {
    /// Element position relative to the device pose, metres.
    pub position_m: [f64; 3],
    /// Element gain, dBi.
    pub gain_dbi: f64,
}

/// Declared state of the phase axis — consumers must branch on this
/// instead of guessing whether detrending already happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseState {
    /// As captured; CFO/STO artifacts present.
    Raw,
    /// Linear ramp + constant offset removed (ADR-274 stage 3).
    Sanitized,
    /// Hardware/baseline calibrated upstream.
    Calibrated,
    /// Magnitude-only capture (e.g. some vendor RSSI/BF reports).
    Unavailable,
}

/// Calibration state carried by every native frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationState {
    /// Phase axis state.
    pub phase_state: PhaseState,
    /// Whether amplitude gain has been calibrated.
    pub gain_calibrated: bool,
    /// Oscillator drift, ppm.
    pub clock_ppm: f64,
    /// Empty-room baseline applied, if any (ADR-135).
    pub baseline_id: Option<String>,
    /// Calibration confidence in `[0, 1]`.
    pub confidence: f64,
}

/// Front-end quality indicators.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SignalQuality {
    /// Received signal strength, dBm.
    pub rssi_dbm: f64,
    /// Noise floor, dBm.
    pub noise_floor_dbm: f64,
    /// Fraction of expected packets lost in the capture window `[0, 1]`.
    pub packet_loss: f64,
    /// Interference score `[0, 1]` (0 = clean).
    pub interference: f64,
}

/// Whether the evidence is measured or synthetic. The two classes can
/// never alias: there is no third variant and no default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceClass {
    /// Captured from real hardware.
    Measured,
    /// Produced by a simulator/generator (ADR-276).
    Synthetic,
}

/// The public evidence ladder (ADR-282 §4): every capability and every
/// dataset carries exactly one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// Level 0 — simulation only.
    L0Simulation,
    /// Level 1 — captured replay of real signals.
    L1CapturedReplay,
    /// Level 2 — controlled laboratory.
    L2Lab,
    /// Level 3 — held-out room and subject validation.
    L3HeldOutValidation,
    /// Level 4 — multi-site field pilot.
    L4MultisiteField,
    /// Level 5 — production operational evidence.
    L5Production,
}

/// Frame provenance: class + evidence level + source identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameProvenance {
    /// Measured vs synthetic (invariant 6).
    pub class: ProvenanceClass,
    /// Evidence-ladder level.
    pub evidence: EvidenceLevel,
    /// Capturing device identifier.
    pub device_id: String,
    /// Firmware version string.
    pub firmware: String,
    /// Receipt id linking results back to this frame (invariant 5).
    pub receipt_id: u128,
}

/// Native axes a frame's tensor may be laid out over (delay-Doppler-native
/// modalities such as OTFS ISAC must not be collapsed into scalar motion
/// energy before storage — ADR-281 §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldAxis {
    /// Sample time.
    Time,
    /// Subcarrier / frequency.
    Frequency,
    /// Delay (multipath arrival).
    Delay,
    /// Doppler shift.
    Doppler,
    /// Radar range bin.
    Range,
    /// Azimuth angle.
    Azimuth,
    /// Elevation angle.
    Elevation,
    /// Antenna element.
    Antenna,
    /// Polarization.
    Polarization,
}

/// The authoritative native RF frame (ADR-279 §2).
#[derive(Debug, Clone)]
pub struct RfFrameV2 {
    /// Schema version ([`SCHEMA_VERSION`]).
    pub schema_version: u16,
    /// Unique frame id.
    pub frame_id: u128,
    /// Capture timestamp, ns since epoch.
    pub timestamp_ns: u64,
    /// Modality.
    pub modality: RfModality,
    /// Native axis semantics, one entry per dimension of `native_shape`.
    pub axes: Vec<FieldAxis>,
    /// Carrier centre frequency, Hz.
    pub centre_frequency_hz: f64,
    /// Occupied bandwidth, Hz.
    pub bandwidth_hz: f64,
    /// Native sample rate along the time-like axis, Hz.
    pub sample_rate_hz: f64,
    /// Native tensor shape (arbitrary rank), row-major over `native_iq`.
    pub native_shape: Vec<usize>,
    /// Native complex samples — **never overwritten** (invariant 1).
    pub native_iq: Vec<Complex64>,
    /// Per-sample validity mask (invariant 2), same length as `native_iq`.
    pub valid_mask: Vec<bool>,
    /// Transmitter pose in the building frame, when known.
    pub transmitter_pose: Option<Pose3>,
    /// Receiver pose in the building frame, when known.
    pub receiver_pose: Option<Pose3>,
    /// Antenna elements of the capturing array.
    pub antenna_geometry: Vec<AntennaElement>,
    /// Age of the capture at hand-off, ns (invariant 7).
    pub sample_age_ns: u64,
    /// Calibration state (invariant 3).
    pub calibration: CalibrationState,
    /// Signal quality.
    pub quality: SignalQuality,
    /// Provenance (invariants 5–6).
    pub provenance: FrameProvenance,
}

impl RfFrameV2 {
    /// Validated constructor — the only way to build a native frame.
    ///
    /// Enforced here: shape/product/mask arity, finite samples on valid
    /// positions, positive frequencies, axes rank match, and the
    /// provenance-class ⇄ evidence-level consistency rule:
    /// `Synthetic ⇒ exactly L0Simulation`, `Measured ⇒ at least
    /// L1CapturedReplay` — so synthetic evidence can never masquerade as
    /// field evidence, and vice versa (invariant 6).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame_id: u128,
        timestamp_ns: u64,
        modality: RfModality,
        axes: Vec<FieldAxis>,
        centre_frequency_hz: f64,
        bandwidth_hz: f64,
        sample_rate_hz: f64,
        native_shape: Vec<usize>,
        native_iq: Vec<Complex64>,
        valid_mask: Vec<bool>,
        transmitter_pose: Option<Pose3>,
        receiver_pose: Option<Pose3>,
        antenna_geometry: Vec<AntennaElement>,
        sample_age_ns: u64,
        calibration: CalibrationState,
        quality: SignalQuality,
        provenance: FrameProvenance,
    ) -> Result<Self> {
        let expected: usize = native_shape.iter().product();
        if native_shape.is_empty() || expected == 0 {
            return Err(UnifiedError::ShapeMismatch("empty native shape".into()));
        }
        if native_iq.len() != expected {
            return Err(UnifiedError::ShapeMismatch(format!(
                "native_iq has {} samples, shape {:?} implies {expected}",
                native_iq.len(),
                native_shape
            )));
        }
        if valid_mask.len() != expected {
            return Err(UnifiedError::ShapeMismatch(format!(
                "valid_mask has {} entries, expected {expected}",
                valid_mask.len()
            )));
        }
        if axes.len() != native_shape.len() {
            return Err(UnifiedError::ShapeMismatch(format!(
                "{} axes declared for rank-{} tensor",
                axes.len(),
                native_shape.len()
            )));
        }
        if !(centre_frequency_hz.is_finite()
            && centre_frequency_hz > 0.0
            && bandwidth_hz.is_finite()
            && bandwidth_hz > 0.0
            && sample_rate_hz.is_finite()
            && sample_rate_hz > 0.0)
        {
            return Err(UnifiedError::InvalidInput(
                "frequencies and sample rate must be finite and positive".into(),
            ));
        }
        for (z, ok) in native_iq.iter().zip(&valid_mask) {
            if *ok && (!z.re.is_finite() || !z.im.is_finite()) {
                return Err(UnifiedError::InvalidInput(
                    "non-finite sample marked valid".into(),
                ));
            }
        }
        if !(0.0..=1.0).contains(&calibration.confidence) {
            return Err(UnifiedError::InvalidInput(
                "calibration confidence must be in [0,1]".into(),
            ));
        }
        match (provenance.class, provenance.evidence) {
            (ProvenanceClass::Synthetic, EvidenceLevel::L0Simulation) => {}
            (ProvenanceClass::Synthetic, level) => {
                return Err(UnifiedError::InvalidInput(format!(
                    "synthetic frames are L0Simulation by definition, got {level:?}"
                )));
            }
            (ProvenanceClass::Measured, EvidenceLevel::L0Simulation) => {
                return Err(UnifiedError::InvalidInput(
                    "measured frames cannot claim L0Simulation".into(),
                ));
            }
            (ProvenanceClass::Measured, _) => {}
        }
        Ok(Self {
            schema_version: SCHEMA_VERSION,
            frame_id,
            timestamp_ns,
            modality,
            axes,
            centre_frequency_hz,
            bandwidth_hz,
            sample_rate_hz,
            native_shape,
            native_iq,
            valid_mask,
            transmitter_pose,
            receiver_pose,
            antenna_geometry,
            sample_age_ns,
            calibration,
            quality,
            provenance,
        })
    }

    /// Fraction of valid samples.
    #[must_use]
    pub fn valid_fraction(&self) -> f64 {
        self.valid_mask.iter().filter(|v| **v).count() as f64 / self.valid_mask.len() as f64
    }

    /// Derived compatibility view (ADR-279 §3): projects a rank-3
    /// `(links, bins, snapshots)` native frame into the ADR-274 canonical
    /// tensor. Invalid samples are filled by linear interpolation from the
    /// nearest valid bins on the same `(link, snapshot)` column before
    /// resampling. The native frame is untouched (`&self`).
    pub fn to_canonical(&self, links: Vec<LinkGeometry>) -> Result<RfTensor> {
        if self.native_shape.len() != 3 {
            return Err(UnifiedError::ShapeMismatch(format!(
                "canonical view needs a rank-3 (links, bins, snapshots) frame, got rank {}",
                self.native_shape.len()
            )));
        }
        let (n_links, n_bins, n_snaps) =
            (self.native_shape[0], self.native_shape[1], self.native_shape[2]);
        if links.len() != n_links {
            return Err(UnifiedError::ShapeMismatch(format!(
                "geometry for {} links, frame has {n_links}",
                links.len()
            )));
        }
        // Gap-fill invalid bins per (link, snapshot) column, then hand a
        // dense grid to the shared normalization used by every adapter.
        let mut grid = ndarray::Array3::zeros((n_links, n_bins, n_snaps));
        for l in 0..n_links {
            for s in 0..n_snaps {
                let at = |b: usize| l * n_bins * n_snaps + b * n_snaps + s;
                let valid: Vec<usize> = (0..n_bins).filter(|b| self.valid_mask[at(*b)]).collect();
                if valid.is_empty() {
                    return Err(UnifiedError::InvalidInput(format!(
                        "link {l} snapshot {s} has no valid bins"
                    )));
                }
                for b in 0..n_bins {
                    let v = if self.valid_mask[at(b)] {
                        self.native_iq[at(b)]
                    } else {
                        // Nearest valid neighbors, linear on the complex plane.
                        let before = valid.iter().rev().find(|x| **x < b);
                        let after = valid.iter().find(|x| **x > b);
                        match (before, after) {
                            (Some(&lo), Some(&hi)) => {
                                let t = (b - lo) as f64 / (hi - lo) as f64;
                                self.native_iq[at(lo)] * (1.0 - t) + self.native_iq[at(hi)] * t
                            }
                            (Some(&lo), None) => self.native_iq[at(lo)],
                            (None, Some(&hi)) => self.native_iq[at(hi)],
                            (None, None) => unreachable!("valid is non-empty"),
                        }
                    };
                    grid[[l, b, s]] = v;
                }
            }
        }
        let uncertainty = {
            let snr = self.quality.rssi_dbm - self.quality.noise_floor_dbm;
            (1.0 - snr / 40.0).clamp(0.0, 1.0)
        };
        crate::adapters::normalize_grid(
            self.modality,
            grid,
            links,
            self.centre_frequency_hz,
            self.bandwidth_hz,
            self.sample_age_ns as f64 / 1e9,
            self.timestamp_ns,
            self.provenance.device_id.clone(),
            (1.0 - self.calibration.clock_ppm / 40.0).clamp(0.0, 1.0),
            uncertainty,
            matches!(self.calibration.phase_state, PhaseState::Sanitized | PhaseState::Calibrated),
        )
    }
}

/// IEEE P3162-profile synthetic-aperture channel-sounding import
/// (ADR-281 §5): the calibration bridge between measured environments,
/// simulators, and learned RF scene models.
#[derive(Debug, Clone)]
pub struct SyntheticApertureSoundingDataset {
    /// Sounded frequency range `[low, high]`, Hz.
    pub frequency_range_hz: [f64; 2],
    /// Aperture element poses (building frame).
    pub aperture_geometry: Vec<Pose3>,
    /// Directional power-delay profile, `(direction, delay)` row-major.
    pub directional_pdp: Vec<f64>,
    /// PDP shape.
    pub pdp_shape: [usize; 2],
    /// Coordinate system identifier (P3162 vocabulary).
    pub coordinate_system: String,
    /// Hash of the processing manifest that produced the dataset.
    pub processing_manifest_hash: u64,
}

impl SyntheticApertureSoundingDataset {
    /// Validated constructor.
    pub fn new(
        frequency_range_hz: [f64; 2],
        aperture_geometry: Vec<Pose3>,
        directional_pdp: Vec<f64>,
        pdp_shape: [usize; 2],
        coordinate_system: impl Into<String>,
        processing_manifest_hash: u64,
    ) -> Result<Self> {
        if !(frequency_range_hz[0] > 0.0 && frequency_range_hz[1] > frequency_range_hz[0]) {
            return Err(UnifiedError::InvalidInput(format!(
                "frequency range must be ordered and positive, got {frequency_range_hz:?}"
            )));
        }
        if aperture_geometry.is_empty() {
            return Err(UnifiedError::InvalidInput("empty aperture geometry".into()));
        }
        if directional_pdp.len() != pdp_shape[0] * pdp_shape[1] {
            return Err(UnifiedError::ShapeMismatch(format!(
                "PDP has {} entries, shape {pdp_shape:?} implies {}",
                directional_pdp.len(),
                pdp_shape[0] * pdp_shape[1]
            )));
        }
        if directional_pdp.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(UnifiedError::InvalidInput("PDP entries must be finite power".into()));
        }
        Ok(Self {
            frequency_range_hz,
            aperture_geometry,
            directional_pdp,
            pdp_shape,
            coordinate_system: coordinate_system.into(),
            processing_manifest_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{CANONICAL_BINS, CANONICAL_SNAPSHOTS};

    fn quality() -> SignalQuality {
        SignalQuality { rssi_dbm: -45.0, noise_floor_dbm: -92.0, packet_loss: 0.02, interference: 0.05 }
    }

    fn calibration(phase: PhaseState) -> CalibrationState {
        CalibrationState {
            phase_state: phase,
            gain_calibrated: false,
            clock_ppm: 12.0,
            baseline_id: None,
            confidence: 0.8,
        }
    }

    fn provenance(class: ProvenanceClass, evidence: EvidenceLevel) -> FrameProvenance {
        FrameProvenance {
            class,
            evidence,
            device_id: "esp32s3-a1".into(),
            firmware: "fw-2.1".into(),
            receipt_id: 42,
        }
    }

    fn frame(shape: Vec<usize>, mask_off: &[usize]) -> RfFrameV2 {
        let n: usize = shape.iter().product();
        let iq: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(1.0 + 0.01 * (i % 13) as f64, 0.002 * (i % 7) as f64))
            .collect();
        let mut mask = vec![true; n];
        for &i in mask_off {
            mask[i] = false;
        }
        RfFrameV2::new(
            7,
            1_000,
            RfModality::WifiCsi,
            vec![FieldAxis::Antenna, FieldAxis::Frequency, FieldAxis::Time],
            2.437e9,
            20e6,
            100.0,
            shape,
            iq,
            mask,
            Some(Pose3 { position_m: [0.0, 0.0, 2.0], orientation: [1.0, 0.0, 0.0, 0.0] }),
            Some(Pose3 { position_m: [4.0, 0.0, 2.0], orientation: [1.0, 0.0, 0.0, 0.0] }),
            vec![AntennaElement { position_m: [0.0; 3], gain_dbi: 2.0 }],
            5_000_000,
            calibration(PhaseState::Raw),
            quality(),
            provenance(ProvenanceClass::Measured, EvidenceLevel::L2Lab),
        )
        .expect("valid frame")
    }

    #[test]
    fn synthetic_and_measured_provenance_can_never_alias() {
        let build = |class, evidence| {
            RfFrameV2::new(
                1,
                0,
                RfModality::Synthetic,
                vec![FieldAxis::Frequency],
                2.4e9,
                20e6,
                100.0,
                vec![4],
                vec![Complex64::new(1.0, 0.0); 4],
                vec![true; 4],
                None,
                None,
                vec![],
                0,
                calibration(PhaseState::Sanitized),
                quality(),
                provenance(class, evidence),
            )
        };
        // Synthetic above L0 is refused.
        assert!(build(ProvenanceClass::Synthetic, EvidenceLevel::L3HeldOutValidation).is_err());
        // Measured claiming L0 is refused.
        assert!(build(ProvenanceClass::Measured, EvidenceLevel::L0Simulation).is_err());
        // The two legal pairings work.
        assert!(build(ProvenanceClass::Synthetic, EvidenceLevel::L0Simulation).is_ok());
        assert!(build(ProvenanceClass::Measured, EvidenceLevel::L1CapturedReplay).is_ok());
    }

    #[test]
    fn constructor_enforces_shape_mask_and_axes_arity() {
        let n = 2 * 10 * 4;
        let iq = vec![Complex64::new(1.0, 0.0); n];
        let bad_mask = RfFrameV2::new(
            1,
            0,
            RfModality::WifiCsi,
            vec![FieldAxis::Antenna, FieldAxis::Frequency, FieldAxis::Time],
            2.4e9,
            20e6,
            100.0,
            vec![2, 10, 4],
            iq.clone(),
            vec![true; n - 1],
            None,
            None,
            vec![],
            0,
            calibration(PhaseState::Raw),
            quality(),
            provenance(ProvenanceClass::Measured, EvidenceLevel::L2Lab),
        );
        assert!(matches!(bad_mask, Err(UnifiedError::ShapeMismatch(_))));

        let bad_axes = RfFrameV2::new(
            1,
            0,
            RfModality::WifiCsi,
            vec![FieldAxis::Frequency],
            2.4e9,
            20e6,
            100.0,
            vec![2, 10, 4],
            iq,
            vec![true; n],
            None,
            None,
            vec![],
            0,
            calibration(PhaseState::Raw),
            quality(),
            provenance(ProvenanceClass::Measured, EvidenceLevel::L2Lab),
        );
        assert!(matches!(bad_axes, Err(UnifiedError::ShapeMismatch(_))));
    }

    #[test]
    fn canonical_view_is_derived_and_native_is_untouched() {
        // 114-subcarrier native with two masked-out bins.
        let f = frame(vec![1, 114, 12], &[5 * 12, 60 * 12 + 3]);
        let native_before = f.native_iq.clone();
        let mask_before = f.valid_mask.clone();

        let t = f
            .to_canonical(vec![LinkGeometry { tx_pos: [0.0, 0.0, 2.0], rx_pos: [4.0, 0.0, 2.0] }])
            .expect("derived view");
        assert_eq!(t.dims(), (1, CANONICAL_BINS, CANONICAL_SNAPSHOTS));
        assert!(t.data.iter().all(|z| z.re.is_finite() && z.im.is_finite()));

        // Invariant 1: the native samples and mask are byte-identical after
        // deriving the view — normalization never writes back.
        assert_eq!(f.native_iq, native_before);
        assert_eq!(f.valid_mask, mask_before);
        assert!((f.valid_fraction() - (114.0 * 12.0 - 2.0) / (114.0 * 12.0)).abs() < 1e-12);
    }

    #[test]
    fn canonical_view_rejects_wrong_rank_or_geometry() {
        let f = frame(vec![2, 10, 4], &[]);
        assert!(f.to_canonical(vec![]).is_err());
        // Rank-1 frame has no canonical projection.
        let flat = RfFrameV2::new(
            9,
            0,
            RfModality::WifiCsi,
            vec![FieldAxis::Frequency],
            2.4e9,
            20e6,
            100.0,
            vec![80],
            vec![Complex64::new(1.0, 0.0); 80],
            vec![true; 80],
            None,
            None,
            vec![],
            0,
            calibration(PhaseState::Raw),
            quality(),
            provenance(ProvenanceClass::Measured, EvidenceLevel::L2Lab),
        )
        .expect("rank-1 frame is a valid native frame");
        assert!(flat
            .to_canonical(vec![LinkGeometry { tx_pos: [0.0; 3], rx_pos: [1.0, 0.0, 0.0] }])
            .is_err());
    }

    #[test]
    fn synthetic_aperture_profile_validates() {
        let ok = SyntheticApertureSoundingDataset::new(
            [3.0e9, 10.0e9],
            vec![Pose3 { position_m: [0.0; 3], orientation: [1.0, 0.0, 0.0, 0.0] }],
            vec![0.5; 8 * 16],
            [8, 16],
            "P3162-spherical",
            0xABCD,
        );
        assert!(ok.is_ok());
        assert!(SyntheticApertureSoundingDataset::new(
            [10.0e9, 3.0e9], // unordered
            vec![Pose3 { position_m: [0.0; 3], orientation: [1.0, 0.0, 0.0, 0.0] }],
            vec![0.5; 4],
            [2, 2],
            "x",
            0
        )
        .is_err());
    }
}
