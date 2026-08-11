//! # ruview-ood — out-of-distribution detection (ADR-299)
//!
//! Primitive 2 of the ADR-297 perception substrate: the gate that attaches a
//! [`DomainState`] — `KNOWN` / `DEGRADED` / `UNKNOWN` — to **every** inference,
//! so RuView can say *"I do not recognize this situation"* instead of returning
//! a confidently-wrong label when it leaves its calibrated domain.
//!
//! It fuses four measured inputs (ADR-299 §1) against the ADR-298
//! [`CalibrationCertificate`](wifi_densepose_calibration::certificate::CalibrationCertificate):
//!
//! 1. **domain distance** — [`domain_distance`] over live vs certified
//!    fingerprints (reusing ADR-298's [`FingerprintDistance`]);
//! 2. **signal quality** — [`SignalQuality`] (ADR-137);
//! 3. **calibration compatibility** — [`CalibrationCompat`], derived from a
//!    certificate via [`assess_certificate`] / [`no_certificate`];
//! 4. **uncertainty** — the model head's own predictive uncertainty, attached
//!    at the [`InferenceGate`].
//!
//! ## The four non-negotiable rules (ADR-297)
//!
//! - **UNKNOWN is a first-class value, never an error.** [`DomainState::Unknown`]
//!   is returned, not thrown; the gate suppresses the confident class rather
//!   than defaulting to one or silently holding a stale value.
//! - **Staleness guard `VALID → DEGRADED → UNKNOWN`.** [`DomainThresholds::classify`]
//!   escalates monotonically: crossing the envelope's inner threshold →
//!   DEGRADED, the outer threshold (or a missing/stale/mismatched certificate)
//!   → UNKNOWN. DEGRADED/UNKNOWN both raise a [`RecalibrationRequest`] — a
//!   *signal*, not an action.
//! - **Honesty.** No accuracy is claimed here; this crate ships the gating
//!   machinery only. Synthetic test fixtures are labelled as such; no MEASURED
//!   or hardware claim is made.
//!
//! All logic is pure and deterministic: time is injected, there is no
//! randomness, allocation is bounded, and malformed runtime input yields
//! UNKNOWN rather than a panic.

#![forbid(unsafe_code)]

pub mod certificate;
pub mod domain;
pub mod error;
pub mod gate;

pub use certificate::{assess_certificate, no_certificate, ExpectedIdentity};
pub use domain::{
    domain_distance, CalibrationCompat, DomainCause, DomainState, DomainThresholds, SignalQuality,
};
pub use error::{OodError, Result};
pub use gate::{
    GatedInference, Inference, InferenceGate, RecalibrationRequest, RecalibrationUrgency,
};

// Re-export the calibration primitives this crate gates against, so consumers
// have one import surface.
pub use wifi_densepose_calibration::certificate::{
    CompatibilityEnvelope, FingerprintDistance, RoomFingerprint,
};

#[cfg(test)]
mod tests {
    use super::*;
    use wifi_densepose_calibration::certificate::{
        CalibrationCertificate, CalibrationTier, CharacterizationSource, CompatibilityEnvelope,
        EvidenceLevel, FingerprintDistance, KeyedHashSigner, MintParams, RoomFingerprint,
    };
    use wifi_densepose_calibration::{
        anchor::AnchorLabel,
        bank::SpecialistBank,
        extract::{AnchorFeature, Features},
    };

    // --- synthetic fixtures (SYNTHETIC / L0) -------------------------------

    /// A synthetic fingerprint with a tunable empty-baseline mean, so drift is
    /// deterministic and monotone. SYNTHETIC — no measured/hardware claim.
    fn fingerprint(empty_mean: f32) -> RoomFingerprint {
        RoomFingerprint {
            schema_version: 1,
            empty_mean,
            empty_variance: 1.0,
            occupied_variance: 10.0,
            presence_threshold: 5.0,
            occupancy_mean_shift: 2.0,
            geometry: Default::default(),
        }
    }

    fn envelope() -> CompatibilityEnvelope {
        // outer = 0.15; with default inner_drift_fraction 0.6, inner = 0.09.
        CompatibilityEnvelope::default()
    }

    fn good_quality() -> SignalQuality {
        SignalQuality::new(0.9, false, true).unwrap()
    }

    /// Distance producing exactly `total` (bypassing fingerprint math when a
    /// precise drift value is needed for a boundary test). Fields are public in
    /// the calibration crate, so this is a legitimate synthetic construction.
    fn dist(total: f32) -> FingerprintDistance {
        FingerprintDistance {
            baseline_drift: total,
            occupancy_drift: 0.0,
            total,
        }
    }

    // --- (1) domain distance ----------------------------------------------

    #[test]
    fn domain_distance_reuses_fingerprint_metric() {
        let certified = fingerprint(1.0);
        let identical = fingerprint(1.0);
        let drifted = fingerprint(50.0);

        let d0 = domain_distance(&certified, &identical);
        assert_eq!(d0.total, 0.0, "identical fingerprints have zero drift");

        let d1 = domain_distance(&certified, &drifted);
        assert!(d1.total > 0.0, "a moved empty-baseline registers drift");
        // Matches the calibration crate's own metric (no second definition).
        assert_eq!(d1, certified.distance(&drifted));
    }

    // --- (2) classify: KNOWN / DEGRADED / UNKNOWN --------------------------

    #[test]
    fn known_within_envelope() {
        let state = DomainState::classify(dist(0.02), envelope(), good_quality(), CalibrationCompat::Valid);
        assert_eq!(state, DomainState::Known);
        assert!(state.is_known());
        assert_eq!(state.cause(), None);
    }

    #[test]
    fn degraded_at_inner_threshold_crossing() {
        // inner = 0.15 * 0.6 = 0.09; just above it, still within the outer 0.15.
        let state = DomainState::classify(dist(0.10), envelope(), good_quality(), CalibrationCompat::Valid);
        assert_eq!(state, DomainState::Degraded(DomainCause::ModerateDrift));
        assert!(state.is_degraded());
    }

    #[test]
    fn unknown_past_outer_threshold() {
        let state = DomainState::classify(dist(0.20), envelope(), good_quality(), CalibrationCompat::Valid);
        assert_eq!(state, DomainState::Unknown(DomainCause::DriftBeyondEnvelope));
        assert!(state.is_unknown());
    }

    #[test]
    fn unknown_missing_certificate_defaults_unknown() {
        // Absent certificate → UNKNOWN even with zero drift and perfect quality.
        let state = DomainState::classify(dist(0.0), envelope(), good_quality(), no_certificate());
        assert_eq!(state, DomainState::Unknown(DomainCause::NoCertificate));
    }

    #[test]
    fn unknown_stale_and_mismatched_certificates() {
        for (compat, cause) in [
            (CalibrationCompat::Expired, DomainCause::CertificateExpired),
            (CalibrationCompat::Tampered, DomainCause::CertificateTampered),
            (CalibrationCompat::DeviceMismatch, DomainCause::DeviceMismatch),
            (CalibrationCompat::SpaceMismatch, DomainCause::SpaceMismatch),
            (CalibrationCompat::DriftedBeyondEnvelope, DomainCause::DriftBeyondEnvelope),
        ] {
            let state = DomainState::classify(dist(0.0), envelope(), good_quality(), compat);
            assert_eq!(state, DomainState::Unknown(cause), "compat {compat:?} → UNKNOWN");
        }
    }

    #[test]
    fn certificate_check_precedes_drift_in_staleness_guard() {
        // Absent certificate wins over an otherwise-in-envelope distance.
        let state = DomainState::classify(dist(0.01), envelope(), good_quality(), CalibrationCompat::Absent);
        assert_eq!(state, DomainState::Unknown(DomainCause::NoCertificate));
    }

    #[test]
    fn degraded_on_contradiction_and_low_quality() {
        let contra = SignalQuality::new(0.9, true, true).unwrap();
        assert_eq!(
            DomainState::classify(dist(0.0), envelope(), contra, CalibrationCompat::Valid),
            DomainState::Degraded(DomainCause::Contradiction)
        );

        let lowish = SignalQuality::new(0.45, false, true).unwrap(); // floor 0.3 < 0.45 < 0.6
        assert_eq!(
            DomainState::classify(dist(0.0), envelope(), lowish, CalibrationCompat::Valid),
            DomainState::Degraded(DomainCause::LowSignalQuality)
        );
    }

    #[test]
    fn unknown_on_unusable_signal() {
        let below_floor = SignalQuality::new(0.1, false, true).unwrap();
        assert_eq!(
            DomainState::classify(dist(0.0), envelope(), below_floor, CalibrationCompat::Valid),
            DomainState::Unknown(DomainCause::SignalUnusable)
        );
        let invalid = SignalQuality::new(0.9, false, false).unwrap();
        assert_eq!(
            DomainState::classify(dist(0.0), envelope(), invalid, CalibrationCompat::Valid),
            DomainState::Unknown(DomainCause::SignalUnusable)
        );
    }

    #[test]
    fn hysteresis_inner_below_outer() {
        // The inner (DEGRADED) line is strictly below the outer (UNKNOWN) line,
        // so drift straddling one boundary cannot flap KNOWN⇄UNKNOWN directly.
        let t = DomainThresholds::default();
        let outer = envelope().max_total_drift;
        let inner = outer * t.inner_drift_fraction;
        assert!(inner < outer);
        // A value between the two lines is DEGRADED, not KNOWN and not UNKNOWN.
        let mid = 0.5 * (inner + outer);
        assert_eq!(
            DomainState::classify(dist(mid), envelope(), good_quality(), CalibrationCompat::Valid),
            DomainState::Degraded(DomainCause::ModerateDrift)
        );
    }

    // --- (3) gate suppresses confident class under DEGRADED / UNKNOWN ------

    #[test]
    fn gate_returns_confident_class_when_known() {
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("standing", 0.95, 0.1),
            dist(0.02),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert_eq!(out.state, DomainState::Known);
        assert_eq!(out.class, Some("standing"));
        assert_eq!(out.confidence, Some(0.95));
        assert!(out.recalibration.is_none());
        assert!(out.is_confident());
    }

    #[test]
    fn gate_flags_but_keeps_class_when_degraded() {
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("sitting", 0.9, 0.1),
            dist(0.10), // inner-crossing drift
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert!(out.state.is_degraded());
        // DEGRADED still returns the class, but flagged + recalibration recommended.
        assert_eq!(out.class, Some("sitting"));
        assert!(!out.is_confident(), "a degraded class is not a confident class");
        let rec = out.recalibration.expect("degraded requests recalibration");
        assert_eq!(rec.urgency, RecalibrationUrgency::Recommended);
        assert_eq!(rec.reason, DomainCause::ModerateDrift);
    }

    #[test]
    fn gate_suppresses_class_when_unknown() {
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("lying_down", 0.99, 0.05), // model is very "confident"
            dist(0.30), // past the outer envelope
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert!(out.state.is_unknown());
        // ADR-297 rule 1: no confident class survives an UNKNOWN domain.
        assert_eq!(out.class, None);
        assert_eq!(out.confidence, None);
        assert!(!out.is_confident());
        let rec = out.recalibration.expect("unknown requires recalibration");
        assert_eq!(rec.urgency, RecalibrationUrgency::Required);
    }

    #[test]
    fn gate_suppresses_class_when_certificate_absent() {
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("standing", 0.99, 0.01),
            dist(0.0),
            envelope(),
            good_quality(),
            no_certificate(),
        );
        assert_eq!(out.state, DomainState::Unknown(DomainCause::NoCertificate));
        assert_eq!(out.class, None);
    }

    #[test]
    fn gate_escalates_known_to_degraded_on_uncertainty() {
        let gate = InferenceGate::default();
        // In-envelope + good quality would be KNOWN, but high uncertainty (>0.5).
        let out = gate.evaluate(
            Inference::new("standing", 0.8, 0.9),
            dist(0.02),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert_eq!(out.state, DomainState::Degraded(DomainCause::ElevatedUncertainty));
        assert_eq!(out.class, Some("standing")); // degraded keeps the flagged class
        assert!(out.recalibration.is_some());
    }

    #[test]
    fn uncertainty_never_upgrades_a_worse_state() {
        // Even zero uncertainty cannot rescue an UNKNOWN domain.
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("x", 1.0, 0.0),
            dist(0.5),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert!(out.state.is_unknown());
        assert_eq!(out.class, None);
    }

    // --- (4) recalibration signalled on DEGRADED and UNKNOWN --------------

    #[test]
    fn recalibration_signalled_only_when_not_known() {
        let gate = InferenceGate::default();

        let known = gate.evaluate(
            Inference::new(1u8, 0.9, 0.1),
            dist(0.0),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert!(known.recalibration.is_none());

        let degraded = gate.evaluate(
            Inference::new(1u8, 0.9, 0.1),
            dist(0.10),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert!(degraded.recalibration.is_some());

        let unknown = gate.evaluate(
            Inference::new(1u8, 0.9, 0.1),
            dist(0.0),
            envelope(),
            good_quality(),
            no_certificate(),
        );
        assert!(unknown.recalibration.is_some());
    }

    // --- determinism -------------------------------------------------------

    #[test]
    fn classification_is_deterministic() {
        let inputs = (dist(0.10), envelope(), good_quality(), CalibrationCompat::Valid);
        let first = DomainState::classify(inputs.0, inputs.1, inputs.2, inputs.3);
        for _ in 0..1000 {
            assert_eq!(DomainState::classify(inputs.0, inputs.1, inputs.2, inputs.3), first);
        }
    }

    #[test]
    fn gated_inference_serializes_stably() {
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("standing".to_string(), 0.9, 0.1),
            dist(0.10),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        let a = serde_json::to_string(&out).unwrap();
        let b = serde_json::to_string(&out).unwrap();
        assert_eq!(a, b, "serialization is deterministic");
        assert!(a.contains("Degraded"), "state is present on the record");
    }

    // --- boundary validation ----------------------------------------------

    #[test]
    fn malformed_config_is_rejected_not_panicked() {
        assert!(SignalQuality::new(f32::NAN, false, true).is_err());
        assert!(SignalQuality::new(1.5, false, true).is_err());
        assert!(SignalQuality::new(-0.1, false, true).is_err());

        assert!(DomainThresholds::new(1.0, 0.6, 0.3).is_err()); // fraction not < 1
        assert!(DomainThresholds::new(f32::INFINITY, 0.6, 0.3).is_err());
        assert!(DomainThresholds::new(0.6, 0.3, 0.6).is_err()); // floor > known_min
        assert!(DomainThresholds::new(0.6, 0.6, 0.3).is_ok());

        assert!(InferenceGate::new(DomainThresholds::default(), 2.0).is_err());
        assert!(InferenceGate::new(DomainThresholds::default(), 0.5).is_ok());
    }

    #[test]
    fn malformed_runtime_input_yields_unknown_not_panic() {
        // A non-finite live distance is treated as maximal drift → UNKNOWN.
        let state = DomainState::classify(dist(f32::NAN), envelope(), good_quality(), CalibrationCompat::Valid);
        assert_eq!(state, DomainState::Unknown(DomainCause::DriftBeyondEnvelope));

        // A hostile model uncertainty (NaN) is sanitized (→ worst), never panics.
        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("x", f32::NAN, f32::NAN),
            dist(0.02),
            envelope(),
            good_quality(),
            CalibrationCompat::Valid,
        );
        assert!(out.uncertainty.is_finite());
        // NaN uncertainty clamps to 0.0 here (worst-for-unit maps low); the
        // point is no panic and a finite, bounded value.
        assert!((0.0..=1.0).contains(&out.uncertainty));
    }

    #[test]
    fn signal_quality_from_signals_is_bounded_under_hostile_input() {
        let q = SignalQuality::from_signals(f32::NAN, f32::INFINITY, false, true);
        assert!((0.0..=1.0).contains(&q.score));
        let q2 = SignalQuality::from_signals(2.0, 100.0, false, true); // out-of-range clamps
        assert!((0.0..=1.0).contains(&q2.score));
    }

    // --- cross-ADR: consume a real ADR-298 certificate --------------------

    fn af(label: AnchorLabel, mean: f32, variance: f32, motion: f32) -> AnchorFeature {
        AnchorFeature {
            room_id: "living-room".into(),
            label,
            features: Features {
                mean,
                variance,
                motion,
                breathing_score: 0.0,
                breathing_hz: 0.0,
                heart_score: 0.0,
                heart_hz: 0.0,
            },
        }
    }

    fn synthetic_bank() -> SpecialistBank {
        let anchors = vec![
            af(AnchorLabel::Empty, 1.0, 1.0, 0.1),
            af(AnchorLabel::StandStill, 3.0, 10.0, 0.2),
            af(AnchorLabel::Sit, 1.0, 6.0, 0.2),
            af(AnchorLabel::LieDown, 1.0, 3.0, 0.2),
        ];
        SpecialistBank::train("living-room", "base-1", &anchors, 1000).unwrap()
    }

    /// Mint a SYNTHETIC / L0 certificate — honest labelling (CLAUDE.md).
    fn synthetic_certificate() -> (CalibrationCertificate, KeyedHashSigner) {
        let signer = KeyedHashSigner::new("sensor-42", b"secret".to_vec());
        let params = MintParams {
            space_id: "home/living-room".into(),
            sensor_id: "sensor-42".into(),
            captured_at_unix_s: 1_000_000,
            validity_secs: 3600,
            version: 1,
            tier: CalibrationTier::Auto,
            evidence: EvidenceLevel::L0Synthetic,
            source: CharacterizationSource::Synthetic,
            envelope: CompatibilityEnvelope::default(),
        };
        let cert = CalibrationCertificate::mint(params, &synthetic_bank(), &signer).unwrap();
        (cert, signer)
    }

    #[test]
    fn cross_adr_valid_certificate_drives_known() {
        let (cert, signer) = synthetic_certificate();
        let live = cert.fingerprint.clone(); // no drift
        let now = cert.captured_at_unix_s + 10;
        let expected = ExpectedIdentity {
            space_id: "home/living-room",
            device_id: "sensor-42",
        };
        let (distance, compat) = assess_certificate(&cert, &live, expected, now, &signer);
        assert_eq!(compat, CalibrationCompat::Valid);

        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("standing", 0.9, 0.1),
            distance,
            cert.envelope,
            good_quality(),
            compat,
        );
        assert_eq!(out.state, DomainState::Known);
        assert_eq!(out.class, Some("standing"));
    }

    #[test]
    fn cross_adr_expired_certificate_drives_unknown() {
        let (cert, signer) = synthetic_certificate();
        let live = cert.fingerprint.clone();
        let now = cert.expires_at_unix_s + 1; // stale
        let expected = ExpectedIdentity {
            space_id: "home/living-room",
            device_id: "sensor-42",
        };
        let (distance, compat) = assess_certificate(&cert, &live, expected, now, &signer);
        assert_eq!(compat, CalibrationCompat::Expired);

        let gate = InferenceGate::default();
        let out = gate.evaluate(
            Inference::new("standing", 0.99, 0.01),
            distance,
            cert.envelope,
            good_quality(),
            compat,
        );
        assert_eq!(out.state, DomainState::Unknown(DomainCause::CertificateExpired));
        assert_eq!(out.class, None, "no confident class from a stale certificate");
    }

    #[test]
    fn cross_adr_device_and_space_mismatch_drive_unknown() {
        let (cert, signer) = synthetic_certificate();
        let live = cert.fingerprint.clone();
        let now = cert.captured_at_unix_s + 10;

        let wrong_device = ExpectedIdentity {
            space_id: "home/living-room",
            device_id: "sensor-99",
        };
        let (_d, compat) = assess_certificate(&cert, &live, wrong_device, now, &signer);
        assert_eq!(compat, CalibrationCompat::DeviceMismatch);

        let wrong_space = ExpectedIdentity {
            space_id: "office/lab",
            device_id: "sensor-42",
        };
        let (_d2, compat2) = assess_certificate(&cert, &live, wrong_space, now, &signer);
        assert_eq!(compat2, CalibrationCompat::SpaceMismatch);
    }
}
