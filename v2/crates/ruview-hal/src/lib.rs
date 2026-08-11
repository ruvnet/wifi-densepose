//! # `ruview-hal` — the RuView sensor HAL (ADR-317, ADR-297 primitive 20)
//!
//! One hardware abstraction that maps **any** sensing modality — {CSI, 802.11bf,
//! BLE, UWB, mmWave, acoustic, camera, lidar, IMU, custom} — onto one canonical
//! [`Observation`](ruview_ontology::Observation) feeding one world model. This
//! is the boundary that turns RuView from a WiFi-CSI pipeline into an open
//! spatial-intelligence ingest layer: the world model never sees a
//! modality-specific frame, only a provenance-bearing, evidence-labelled
//! observation.
//!
//! This crate consumes the canonical ontology (ADR-303) — its output is an
//! ontology `Observation` bound to a `Sensor` — and its observations feed real
//! sensor fusion (ADR-308). It is a **pure abstraction**: no I/O, no async, no
//! inference, no accuracy claim. A passing trait test proves the abstraction,
//! not a fielded device; hardware support for any modality stays CLAIMED until
//! demonstrated on real silicon with captured evidence (CLAUDE.md).
//!
//! ## The four ADR-297 non-negotiable rules, as they bind this crate
//!
//! 1. **UNKNOWN is first-class, never an error.** [`SensorHal::normalize`] is
//!    infallible: malformed / out-of-bounds raw input yields an UNKNOWN-flagged
//!    ([`Uncertainty::degraded`]) observation, never a panic or `Err`.
//! 2. **Certificates bind cryptographically.** Out of scope for the HAL, but a
//!    device is authenticated as an ADR-302 `Sensor` (the descriptor's
//!    `sensor_id`) before its observations are trusted.
//! 3. **One canonical semantics downstream.** The HAL reuses the ontology's
//!    `Observation`, `Sensor`, `EvidenceLevel`, and `SemanticProvenance` rather
//!    than reinventing per-crate shapes; [`HalObservation`] *wraps* the
//!    canonical observation and delegates its evidence/provenance accessors.
//! 4. **Honest evidence.** A camera-derived and a CSI-derived observation are
//!    the same type with different provenance; neither is lifted to the other's
//!    grade. The reference adapters are SYNTHETIC / [`EvidenceLevel::L0`] and
//!    cannot alias to a measured level.
//!
//! ## Core shapes
//!
//! - [`Modality`] — the phenomenon class (closed variants + `Custom`).
//! - [`SensorDescriptor`] / [`SamplingSpec`] — canonical device description with
//!   ontology `SensorId`, capability tags, and native sampling/units metadata.
//! - [`HalObservation`] — wraps [`Observation`](ruview_ontology::Observation)
//!   with a [`Modality`] and per-observation [`Uncertainty`]; delegates
//!   `EvidenceLevel` / `SemanticProvenance`.
//! - [`SensorHal`] — the extension-point trait: `describe` + `normalize`.
//! - [`SyntheticCsiAdapter`], [`SyntheticImuAdapter`] — deterministic reference
//!   adapters (one RF, one non-RF), labelled SYNTHETIC / L0.
//!
//! ## Mapping existing adapters onto the trait (docs only)
//!
//! This crate does not rewrite the existing producers; it is the trait they are
//! re-expressed as. RF modalities reuse the ADR-279 per-device latent adapters
//! wholesale — the HAL adds the non-RF and ranging modalities under the same
//! trait. Each row is the `SensorHal` an existing producer implements when it
//! is brought under the abstraction:
//!
//! | Existing producer | Source ADR | `Modality` | `SensorHal::Raw` (native frame) | Notes |
//! |---|---|---|---|---|
//! | ESP32-S3/C6 CSI node | ADR-279 / firmware | [`Modality::Csi`] | `RfFrameV2` (subcarrier complex) | Reuses the ADR-279 native-frame → shared-latent adapter; the HAL only lifts the latent into an `Observation`. |
//! | Nexmon CSI | ADR-279 | [`Modality::Csi`] | `RfFrameV2` | Per-device adapter into the shared latent; same trait, different native layout. |
//! | FeitCSI / Intel / Atheros / Realtek | ADR-279 | [`Modality::Csi`] | `RfFrameV2` | Same shared-latent path; bandwidth/antenna structure kept native, not canonicalized. |
//! | 802.11bf sensing | ADR-307 (phase 2) | [`Modality::Ieee80211bf`] | native 11bf measurement frame | Enters under the same trait as it lands. |
//! | mmWave radar | ADR-063 | [`Modality::Mmwave`] | range-doppler / point frame | The ADR-063 fusion producer becomes a `SensorHal` implementation. |
//! | Multistatic WiFi | ADR-029 | [`Modality::Csi`] | multi-link `RfFrameV2` set | Multiple links, one authenticated `Sensor`, one `Observation`. |
//!
//! Non-RF modalities (camera, lidar, acoustic) enter the same governed plane
//! with the same provenance and privacy discipline — a camera is not a
//! privacy-free shortcut; it inherits ADR-277 governance and carries its own
//! honest evidence level. The two synthetic reference adapters in this crate
//! ([`SyntheticCsiAdapter`], [`SyntheticImuAdapter`]) are the executable
//! template such implementations follow.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod adapter;
mod descriptor;
mod label;
mod modality;
mod observation;

/// The provenance calibration handle that marks an observation SYNTHETIC. A
/// synthetic observation stamped with this handle can never present as
/// measured/calibrated (ADR-279 invariant 6; CLAUDE.md honesty discipline).
pub const SYNTHETIC_CALIBRATION: &str = "synthetic";

pub use adapter::{
    synthetic_provenance, CsiSample, ImuSample, NormalizeCtx, SensorHal, SyntheticCsiAdapter,
    SyntheticImuAdapter, MAX_CSI_TAPS,
};
pub use descriptor::{SamplingSpec, SensorDescriptor};
pub use label::{CapabilityTag, LabelError, MAX_LABEL_LEN};
pub use modality::Modality;
pub use observation::{Confidence, HalObservation, Uncertainty};

#[cfg(test)]
mod tests {
    use super::*;
    use ruview_ontology::{Container, EvidenceLevel, ObservationId, SensorId, SpaceId};

    fn ctx() -> NormalizeCtx {
        NormalizeCtx {
            observation_id: ObservationId::new("obs-1").unwrap(),
            located_in: Container::Space {
                id: SpaceId::new("kitchen").unwrap(),
            },
            at_unix_ms: 1_700_000_000_000,
        }
    }

    fn csi_adapter() -> SyntheticCsiAdapter {
        SyntheticCsiAdapter {
            sensor_id: SensorId::new("csi-1").unwrap(),
            subcarriers: 52,
        }
    }

    fn imu_adapter() -> SyntheticImuAdapter {
        SyntheticImuAdapter {
            sensor_id: SensorId::new("imu-1").unwrap(),
        }
    }

    fn good_csi() -> CsiSample {
        CsiSample {
            amplitudes: vec![1.0, 2.0, 3.0, 4.0],
            phases: vec![0.1, 0.2, 0.3, 0.4],
        }
    }

    // ADR-317 validation: descriptor round-trips losslessly through serde.
    #[test]
    fn descriptor_round_trip() {
        for descriptor in [csi_adapter().describe(), imu_adapter().describe()] {
            let json = serde_json::to_string(&descriptor).unwrap();
            let back: SensorDescriptor = serde_json::from_str(&json).unwrap();
            assert_eq!(descriptor, back);
            assert!(descriptor.validate().is_ok());
        }

        // A custom modality descriptor also round-trips and re-validates.
        let custom = SensorDescriptor {
            sensor_id: SensorId::new("x-1").unwrap(),
            modality: Modality::custom("thermal-array").unwrap(),
            capabilities: vec![CapabilityTag::new("temperature").unwrap()],
            sampling: SamplingSpec {
                sample_rate_hz: None,
                unit: "celsius".into(),
                dimensions: 64,
            },
        };
        let back: SensorDescriptor =
            serde_json::from_str(&serde_json::to_string(&custom).unwrap()).unwrap();
        assert_eq!(custom, back);
        assert!(back.validate().is_ok());
    }

    // ADR-317 validation: a reference adapter normalizes a synthetic sample to a
    // uniform HalObservation carrying sensor id, container, time, exactly one
    // evidence level, and provenance.
    #[test]
    fn reference_adapter_normalizes_synthetic_sample() {
        let a = csi_adapter();
        let obs = a.normalize(good_csi(), &ctx());

        assert_eq!(obs.modality, Modality::Csi);
        assert_eq!(obs.sensor().as_str(), "csi-1");
        assert_eq!(obs.observation.located_in, ctx().located_in);
        assert_eq!(obs.observation.at_unix_ms, 1_700_000_000_000);
        assert_eq!(obs.evidence_level(), EvidenceLevel::L0);
        assert!(obs.is_synthetic());
        assert!(!obs.is_unknown());
        assert!(!obs.uncertainty.degraded);

        // The non-RF adapter produces the *same* type with its own provenance.
        let imu = imu_adapter().normalize(
            ImuSample {
                accel: [0.0, 0.0, 9.81],
                gyro: [0.0, 0.0, 0.0],
            },
            &ctx(),
        );
        assert_eq!(imu.modality, Modality::Imu);
        assert_eq!(imu.evidence_level(), EvidenceLevel::L0);
        assert!(imu.is_synthetic());
        assert!(!imu.is_unknown());
        // Honest evidence: synthetic never reaches a measured/corroborated level.
        assert!(imu.evidence_level() < EvidenceLevel::L2);
        assert_eq!(imu.provenance().model_version, "synthetic-imu-adapter@0");
    }

    // ADR-317 validation: unknown / degraded input yields an UNKNOWN-flagged
    // observation, never a panic.
    #[test]
    fn degraded_input_yields_unknown_not_panic() {
        let a = csi_adapter();

        // Empty frame.
        let empty = a.normalize(
            CsiSample {
                amplitudes: vec![],
                phases: vec![],
            },
            &ctx(),
        );
        assert!(empty.is_unknown());
        assert!(empty.uncertainty.degraded);
        assert_eq!(empty.uncertainty.confidence, Confidence::Unknown);
        // Still a well-formed canonical observation.
        assert_eq!(empty.sensor().as_str(), "csi-1");
        assert_eq!(empty.evidence_level(), EvidenceLevel::L0);
        // Cannot alias to measured.
        assert!(empty.evidence_level() < EvidenceLevel::L2);

        // Length mismatch.
        let mismatch = a.normalize(
            CsiSample {
                amplitudes: vec![1.0, 2.0],
                phases: vec![0.1],
            },
            &ctx(),
        );
        assert!(mismatch.is_unknown());

        // Non-finite (NaN) input.
        let nan = a.normalize(
            CsiSample {
                amplitudes: vec![f32::NAN, 1.0, 2.0, 3.0],
                phases: vec![0.0, 0.0, 0.0, 0.0],
            },
            &ctx(),
        );
        assert!(nan.is_unknown());

        // Over-bounded input is rejected as degraded, bounding compute.
        let huge = a.normalize(
            CsiSample {
                amplitudes: vec![1.0; MAX_CSI_TAPS + 1],
                phases: vec![0.0; MAX_CSI_TAPS + 1],
            },
            &ctx(),
        );
        assert!(huge.is_unknown());

        // IMU with an infinite gyro component.
        let imu = imu_adapter().normalize(
            ImuSample {
                accel: [0.0, 0.0, 9.81],
                gyro: [f32::INFINITY, 0.0, 0.0],
            },
            &ctx(),
        );
        assert!(imu.is_unknown());
        assert!(imu.uncertainty.degraded);
    }

    // Malformed labels are rejected at the boundary, not panicked on.
    #[test]
    fn label_validation_at_boundary() {
        assert_eq!(CapabilityTag::new(""), Err(LabelError::Empty));
        assert!(matches!(
            CapabilityTag::new("a\nb"),
            Err(LabelError::ControlChar { pos: 1 })
        ));
        let long = "x".repeat(MAX_LABEL_LEN + 1);
        assert!(matches!(
            Modality::custom(long),
            Err(LabelError::TooLong { .. })
        ));
        // Closed variants always validate; a well-formed custom validates.
        assert!(Modality::Camera.validate().is_ok());
        assert!(Modality::custom("thermal").unwrap().validate().is_ok());
        assert!(Modality::Csi.is_rf());
        assert!(!Modality::Imu.is_rf());
        assert_eq!(Modality::Ieee80211bf.label(), "ieee80211bf");
    }

    // Serde round-trips a HalObservation (both known and unknown) losslessly.
    #[test]
    fn hal_observation_serde_round_trip() {
        let known = csi_adapter().normalize(good_csi(), &ctx());
        let back: HalObservation =
            serde_json::from_str(&serde_json::to_string(&known).unwrap()).unwrap();
        assert_eq!(known, back);

        let unknown = csi_adapter().normalize(
            CsiSample {
                amplitudes: vec![],
                phases: vec![],
            },
            &ctx(),
        );
        let back: HalObservation =
            serde_json::from_str(&serde_json::to_string(&unknown).unwrap()).unwrap();
        assert_eq!(unknown, back);

        // Modality serializes to its canonical tag; UNKNOWN confidence to a
        // stable string.
        let json = serde_json::to_string(&unknown).unwrap();
        assert!(json.contains("\"csi\""));
        assert!(json.contains("\"unknown\""));
        assert!(json.contains("\"L0\""));
    }

    // Normalization is deterministic: identical input + ctx → identical output.
    #[test]
    fn normalization_is_deterministic() {
        let a = csi_adapter();
        let c = ctx();
        assert_eq!(a.normalize(good_csi(), &c), a.normalize(good_csi(), &c));

        let imu = imu_adapter();
        let s = ImuSample {
            accel: [1.0, 2.0, 9.0],
            gyro: [0.01, 0.02, 0.03],
        };
        assert_eq!(imu.normalize(s.clone(), &c), imu.normalize(s, &c));
    }

    // A synthetic observation cannot be constructed as measured/calibrated: the
    // synthetic calibration handle and L0 evidence pin it below corroboration.
    #[test]
    fn synthetic_cannot_alias_to_measured() {
        let obs = csi_adapter().normalize(good_csi(), &ctx());
        assert!(obs.is_synthetic());
        assert_eq!(
            obs.provenance().calibration_version,
            SYNTHETIC_CALIBRATION
        );
        assert!(obs.evidence_level() < EvidenceLevel::L2);
        assert_ne!(obs.evidence_level(), EvidenceLevel::L4);
        assert_ne!(obs.evidence_level(), EvidenceLevel::L5);
    }

    // Confidence clamps and collapses non-finite values rather than poisoning.
    #[test]
    fn confidence_is_bounded() {
        assert_eq!(Confidence::known(2.0), Confidence::Known(1.0));
        assert_eq!(Confidence::known(-1.0), Confidence::Known(0.0));
        assert_eq!(Confidence::known(f64::NAN), Confidence::Unknown);
        assert!(Uncertainty::unknown().is_unknown());
        assert!(!Uncertainty::unknown().degraded);
        assert!(Uncertainty::degraded().degraded);
    }
}
