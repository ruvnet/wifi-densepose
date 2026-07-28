//! Security property tests (ADR-273 pre-merge item 12): the crate's
//! system boundaries must never panic on hostile input and must stay
//! fail-closed under arbitrary authorization states.
//!
//! Strategy: proptest drives the validated constructors and policy gates
//! with arbitrary values (including NaN/±inf smuggled through
//! `f64::from_bits`) and asserts the *contract*, not specific values:
//! every input either yields a valid object or a typed error — never a
//! panic, and never a permissive default.

use proptest::prelude::*;

use ruview_unified::adapters::{ble_cs_range, BleCsFrame};
use ruview_unified::control::{
    admit_task, validate_representation, CoherentSensorGroup, MemberSyncState, PrivacyClass,
    SensingTask, SpatialZone, TaskSufficientRepresentation,
};
use ruview_unified::gaussian::primitive::{Provenance, RfGaussian};
use ruview_unified::policy::{
    BoundedEvent, EventValue, PolicyEngine, PrivacyZone, SensingPurpose,
};
use ruview_unified::tensor::{CalibrationMeta, LinkGeometry, RfModality, RfTensor};

/// Arbitrary f64 including NaN, ±inf, subnormals — the values an attacker
/// or a broken driver would deliver.
fn any_f64() -> impl Strategy<Value = f64> {
    any::<u64>().prop_map(f64::from_bits)
}

fn any_purpose() -> impl Strategy<Value = SensingPurpose> {
    prop_oneof![
        Just(SensingPurpose::Presence),
        Just(SensingPurpose::Activity),
        Just(SensingPurpose::Vitals),
        Just(SensingPurpose::Localization),
        Just(SensingPurpose::PoseTracking),
        Just(SensingPurpose::IdentityRecognition),
        Just(SensingPurpose::ChannelDiagnostics),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// RfTensor::new never panics; invalid numeric fields are typed errors.
    #[test]
    fn rf_tensor_constructor_never_panics(
        re in any_f64(),
        im in any_f64(),
        freq in any_f64(),
        bw in any_f64(),
        age in any_f64(),
        clock in any_f64(),
        unc in any_f64(),
        tx in prop::array::uniform3(any_f64()),
    ) {
        let data = ndarray::Array3::from_elem((1, 4, 2), num_complex::Complex64::new(re, im));
        let links = vec![LinkGeometry { tx_pos: tx, rx_pos: [1.0, 0.0, 1.0] }];
        let result = RfTensor::new(
            RfModality::WifiCsi, freq, bw, data, links, age, 0, "prop".into(),
            clock, unc, CalibrationMeta::default(),
        );
        // Contract: Ok only when every validated field is actually valid.
        if let Ok(t) = result {
            prop_assert!(t.center_freq_hz.is_finite() && t.center_freq_hz > 0.0);
            prop_assert!((0.0..=1.0).contains(&t.clock_quality));
            prop_assert!((0.0..=1.0).contains(&t.uncertainty));
            prop_assert!(t.sample_age_s.is_finite() && t.sample_age_s >= 0.0);
            prop_assert!(t.data.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
        }
    }

    /// RfGaussian::new never panics; accepted Gaussians have a normalized
    /// quaternion and in-range trust fields.
    #[test]
    fn rf_gaussian_constructor_never_panics(
        pos in prop::array::uniform3(any_f64()),
        scale in prop::array::uniform3(any_f64()),
        quat in prop::array::uniform4(any_f64()),
        occ in any_f64(),
        conf in any_f64(),
        tau in any_f64(),
    ) {
        let result = RfGaussian::new(
            pos, scale, quat, occ, conf, 0, tau,
            Provenance { device_id: "prop".into(), model_version: 1, synthetic: true },
        );
        if let Ok(g) = result {
            let qn: f64 = g.orientation.iter().map(|q| q * q).sum::<f64>().sqrt();
            prop_assert!((qn - 1.0).abs() < 1e-9, "quaternion must be normalized");
            prop_assert!(g.scale.iter().all(|s| *s > 0.0));
            prop_assert!(g.occupancy >= 0.0);
            prop_assert!((0.0..=1.0).contains(&g.confidence));
            // Density at the centre of a valid Gaussian is exactly 1.
            prop_assert!((g.density_at(g.position) - 1.0).abs() < 1e-9);
        }
    }

    /// BoundedEvent::new never panics; exported accountability fields are
    /// always in range.
    #[test]
    fn bounded_event_constructor_never_panics(
        uncertainty in any_f64(),
        model_version in any::<u32>(),
        value in any_f64(),
    ) {
        let result = BoundedEvent::new(
            SensingPurpose::Presence,
            EventValue::RespirationBpm(value),
            uncertainty,
            Provenance { device_id: "prop".into(), model_version: 1, synthetic: false },
            model_version,
            0,
            "zone",
        );
        if let Ok(e) = result {
            prop_assert!((0.0..=1.0).contains(&e.uncertainty));
            prop_assert!(e.model_version != 0, "unassigned model version must never export");
        }
    }

    /// ble_cs_range never panics on arbitrary phases/frequencies/RTT, and
    /// any Ok evidence has a finite, non-negative distance.
    #[test]
    fn ble_cs_range_never_panics(
        phases in prop::collection::vec(any_f64(), 0..24),
        f0 in any_f64(),
        df in any_f64(),
        rtt in prop::option::of(any_f64()),
    ) {
        let n = phases.len();
        let frame = BleCsFrame {
            frequency_steps_hz: (0..n).map(|k| f0 + df * k as f64).collect(),
            phase_samples_rad: phases,
            round_trip_time_ns: rtt,
        };
        if let Ok(ev) = ble_cs_range(&frame) {
            // Non-finite inputs are rejected at the boundary (the unwrap
            // loop would otherwise spin forever on +inf — the DoS this
            // suite originally caught), so Ok evidence is fully finite.
            prop_assert!(ev.phase_distance_m.is_finite() && ev.phase_distance_m >= 0.0);
            prop_assert!((0.0..=1.0).contains(&ev.confidence));
            if let Some(d) = ev.rtt_distance_m {
                prop_assert!(d.is_finite());
            }
        }
    }

    /// The policy engine is fail-closed for every purpose against every
    /// zone configuration that does not explicitly grant it.
    #[test]
    fn policy_engine_is_fail_closed_under_arbitrary_grants(
        purpose in any_purpose(),
        granted in prop::collection::btree_set(any_purpose(), 0..7),
        identity_enabled in any::<bool>(),
        query_unknown_zone in any::<bool>(),
    ) {
        let mut engine = PolicyEngine::new();
        engine.upsert_zone(PrivacyZone {
            id: "z".into(),
            allowed_purposes: granted.clone(),
            retention_s: 60,
            identity_explicitly_enabled: identity_enabled,
        });
        let zone_id = if query_unknown_zone { "nope" } else { "z" };
        let verdict = engine.authorize(zone_id, purpose);
        if query_unknown_zone {
            prop_assert!(verdict.is_err(), "unknown zone must always deny");
        } else if !granted.contains(&purpose) {
            prop_assert!(verdict.is_err(), "ungranted purpose must deny");
        } else if purpose == SensingPurpose::IdentityRecognition && !identity_enabled {
            prop_assert!(verdict.is_err(), "identity single-gate must deny");
        } else {
            prop_assert!(verdict.is_ok());
        }
    }

    /// Task admission can never approve raw export, whatever else is true.
    #[test]
    fn raw_export_is_unreachable(
        purpose in any_purpose(),
        raw in any::<bool>(),
        confidence in any_f64(),
    ) {
        let mut engine = PolicyEngine::new();
        engine.upsert_zone(PrivacyZone {
            id: "z".into(),
            allowed_purposes: [purpose].into_iter().collect(),
            retention_s: 60,
            identity_explicitly_enabled: true,
        });
        let task = SensingTask {
            task_id: 1,
            purpose,
            target_area: SpatialZone { id: "z".into(), min_m: [0.0; 3], max_m: [1.0; 3] },
            modalities: vec![RfModality::WifiCsi],
            requested_resolution_m: 0.5,
            maximum_latency_ms: 100,
            minimum_confidence: confidence,
            raw_retention_seconds: 60,
            result_retention_seconds: 60,
            authorized_consumers: vec![],
            consent_reference: Some("consent-1".into()),
            raw_export_allowed: raw,
        };
        let verdict = admit_task(&engine, &task);
        if raw {
            prop_assert!(verdict.is_err(), "raw export must be structurally unreachable");
        }
    }

    /// Coherent fusion denies whenever any reported error is non-finite or
    /// out of bounds — NaN cannot sneak past the gate.
    #[test]
    fn coherent_fusion_rejects_non_finite_sync_state(
        time_err in any_f64(),
        phase_err in any_f64(),
        hash in any::<u64>(),
    ) {
        let group = CoherentSensorGroup {
            group_id: "g".into(),
            members: vec!["m".into()],
            maximum_time_error_ns: 50.0,
            maximum_phase_error_rad: 0.2,
            baseline_geometry_hash: 7,
        };
        let verdict = group.can_fuse(&[MemberSyncState {
            member_id: "m".into(),
            time_error_ns: time_err,
            phase_error_rad: phase_err,
            geometry_hash: hash,
        }]);
        let in_bounds = time_err.is_finite()
            && time_err.abs() <= 50.0
            && phase_err.is_finite()
            && phase_err.abs() <= 0.2
            && hash == 7;
        prop_assert_eq!(verdict.is_ok(), in_bounds, "NaN/inf must deny, bounds must bind");
    }

    /// Representation validation never approves an identity-retaining
    /// occupancy representation, whatever the other fields say — checked
    /// across *every* `SensingPurpose` branch (each has a distinct ceiling
    /// and exclusion set in `validate_representation`; testing only
    /// `Presence`, as this property previously did, leaves the other three
    /// branches — covering P3–P5, the higher-risk representations —
    /// completely unverified).
    #[test]
    fn occupancy_representations_must_exclude_identity(
        purpose in any_purpose(),
        class in prop_oneof![
            Just(PrivacyClass::P0), Just(PrivacyClass::P1), Just(PrivacyClass::P2),
            Just(PrivacyClass::P3), Just(PrivacyClass::P4), Just(PrivacyClass::P5)
        ],
        excluded in prop::collection::vec("[a-z]{1,10}", 0..4),
        bits in any_f64(),
    ) {
        let rep = TaskSufficientRepresentation {
            task_id: 1,
            source_receipts: vec![1],
            semantic_state: vec![0.5],
            information_bound_bits: bits,
            excluded_information: excluded.clone(),
            privacy_class: class,
        };
        let verdict = validate_representation(&rep, purpose);
        // Oracle mirrors the (ceiling, must_exclude) table in
        // `control::validate_representation` so this property checks the
        // *contract* per purpose group, not just the Presence case.
        let (ceiling, must_exclude): (PrivacyClass, &[&str]) = match purpose {
            SensingPurpose::Presence | SensingPurpose::ChannelDiagnostics => {
                (PrivacyClass::P2, &["identity", "vitals"])
            }
            SensingPurpose::Activity | SensingPurpose::Localization => {
                (PrivacyClass::P3, &["identity"])
            }
            SensingPurpose::Vitals | SensingPurpose::PoseTracking => {
                (PrivacyClass::P4, &["identity"])
            }
            SensingPurpose::IdentityRecognition => (PrivacyClass::P5, &[]),
        };
        let excludes_required =
            must_exclude.iter().all(|c| excluded.iter().any(|e| e == c));
        if verdict.is_ok() {
            prop_assert!(
                excludes_required,
                "approved rep for {:?} must exclude {:?}", purpose, must_exclude
            );
            prop_assert!(
                class <= ceiling,
                "approved rep for {:?} must respect ceiling {:?}", purpose, ceiling
            );
        }
    }
}
