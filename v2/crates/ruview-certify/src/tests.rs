//! Deterministic tests (ADR-315 validation matrix): mint+verify, no-evidence =>
//! no certificate, evidence-level floor enforced, expiry, unsigned invalid,
//! not-KNOWN domain invalidates, canonical-bytes determinism, serde round-trip,
//! calibration-linked expiry ceiling, and context binding. No wall clock, no
//! randomness; every fixture is synthetic (L0) and built in code.

use super::*;

use ruview_attest::{Blake3MacSigner, DeviceId};
use ruview_evidence::{
    AccuracyMetrics, EvidenceContext, EvidenceLedger, EvidenceLevel as EvLevel, EvidenceRecord,
};
use ruview_ontology::SpaceId;
use wifi_densepose_calibration::{
    CalibrationCertificate, CalibrationTier, CharacterizationSource, CompatibilityEnvelope,
    EvidenceLevel as CalibLevel, KeyedHashSigner, MintParams, SpecialistBank,
};
use wifi_densepose_calibration::extract::AnchorFeature;
use wifi_densepose_calibration::AnchorLabel;

const ROOM: &str = "kitchen";
const DEVICE: &str = "dev-1";
const MODEL: &str = "m-1";

fn cert_signer() -> Blake3MacSigner {
    Blake3MacSigner::new([7u8; 32])
}

/// A synthetic calibration certificate (L0) for `ROOM`, captured at
/// `captured_at`, valid for `validity` seconds.
fn calibration(captured_at: i64, validity: i64) -> CalibrationCertificate {
    let anchors = vec![AnchorFeature::from_series(
        ROOM,
        AnchorLabel::Empty,
        &[0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
        20.0,
    )];
    let bank = SpecialistBank::train(ROOM, "base-1", &anchors, captured_at).unwrap();
    let signer = KeyedHashSigner::new("sensor-1", b"secret".to_vec());
    let params = MintParams {
        space_id: ROOM.into(),
        sensor_id: "sensor-1".into(),
        captured_at_unix_s: captured_at,
        validity_secs: validity,
        version: 1,
        tier: CalibrationTier::Auto,
        evidence: CalibLevel::L0Synthetic,
        source: CharacterizationSource::Synthetic,
        envelope: CompatibilityEnvelope::default(),
    };
    CalibrationCertificate::mint(params, &bank, &signer).unwrap()
}

fn context() -> EvidenceContext {
    EvidenceContext::new(ROOM, DEVICE, "moving", MODEL).unwrap()
}

fn metrics() -> AccuracyMetrics {
    AccuracyMetrics {
        moving_recall: 0.8,
        stationary_recall: 0.4,
        false_positive_rate: 0.02,
        drift: 0.1,
        uncertainty: 0.05,
        calibration_age_secs: 100,
        sample_count: 10,
    }
}

/// A ledger holding one synthetic (L0) record for `context()`.
fn synthetic_ledger() -> EvidenceLedger {
    let mut ledger = EvidenceLedger::new();
    ledger
        .append(EvidenceRecord::synthetic(context(), metrics(), 1).unwrap())
        .unwrap();
    ledger
}

fn base_request<'c>(calibration: &'c CalibrationCertificate) -> MintRequest<'c> {
    MintRequest {
        capability: Capability::Presence,
        room: SpaceId::new(ROOM).unwrap(),
        hardware: DeviceId::new(DEVICE).unwrap(),
        model_version: MODEL.into(),
        calibration,
        valid_until_unix_s: 5_000,
        evidence_level: EvLevel::L0,
    }
}

#[test]
fn mint_then_verify_round_trips_and_rejects_tampering() {
    let calibration = calibration(1_000, 5_000); // expires at 6_000
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();

    let cert = mint(&signer, base_request(&calibration), &slice).unwrap();

    // Frozen from the ledger slice, not a global average.
    assert_eq!(cert.content.metrics.moving_recall, 0.8);
    assert_eq!(cert.content.metrics.stationary_recall, 0.4);
    assert_eq!(cert.content.metrics.false_presence_per_24h, 0.02);
    // Synthetic ledger => L0 by construction (never upgraded).
    assert_eq!(cert.content.evidence_level, EvLevel::L0);
    // Calibration binding carried through.
    assert_eq!(cert.content.calibration_version, 1);
    assert_eq!(cert.content.calibration_expires_at_unix_s, 6_000);
    assert_eq!(cert.content.calibrated_date_unix_s, 1_000);

    assert!(cert.verify(&signer), "freshly minted certificate verifies");

    // Tamper with a signed field: the signature no longer verifies.
    let mut tampered = cert.clone();
    tampered.content.metrics.moving_recall = 0.99;
    assert!(!tampered.verify(&signer), "tampered metric is rejected");

    let mut tampered2 = cert.clone();
    tampered2.content.valid_until_unix_s += 1;
    assert!(!tampered2.verify(&signer), "tampered expiry is rejected");
}

#[test]
fn no_evidence_context_yields_no_certificate() {
    let calibration = calibration(1_000, 5_000);
    let ledger = EvidenceLedger::new(); // empty
    let slice = ledger.query(&context());
    let signer = cert_signer();

    let err = mint(&signer, base_request(&calibration), &slice).unwrap_err();
    assert_eq!(err, CertifyError::NoEvidence);
}

#[test]
fn evidence_level_cannot_exceed_the_slice_floor() {
    let calibration = calibration(1_000, 5_000);
    // One measured L3 record => floor L3.
    let mut ledger = EvidenceLedger::new();
    ledger
        .append(EvidenceRecord::measured(context(), metrics(), EvLevel::L3, "repro-1", 1).unwrap())
        .unwrap();
    let slice = ledger.query(&context());
    let signer = cert_signer();

    // Requesting L4 over an L3 floor is an upgrade — refused.
    let mut req = base_request(&calibration);
    req.evidence_level = EvLevel::L4;
    let err = mint(&signer, req, &slice).unwrap_err();
    assert_eq!(
        err,
        CertifyError::EvidenceLevelUpgrade {
            requested: EvLevel::L4,
            floor: EvLevel::L3,
        }
    );

    // Requesting at or below the floor is honest and permitted.
    let mut req_ok = base_request(&calibration);
    req_ok.evidence_level = EvLevel::L2;
    let cert = mint(&signer, req_ok, &slice).unwrap();
    assert_eq!(cert.content.evidence_level, EvLevel::L2);
}

#[test]
fn valid_until_cannot_outlive_calibration() {
    let calibration = calibration(1_000, 5_000); // expires 6_000
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();

    let mut req = base_request(&calibration);
    req.valid_until_unix_s = 7_000; // past calibration expiry
    let err = mint(&signer, req, &slice).unwrap_err();
    assert_eq!(
        err,
        CertifyError::OutlivesCalibration {
            valid_until_unix_s: 7_000,
            calibration_expires_at_unix_s: 6_000,
        }
    );
}

#[test]
fn is_valid_enforces_expiry() {
    let calibration = calibration(1_000, 5_000);
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();
    let cert = mint(&signer, base_request(&calibration), &slice).unwrap();
    // valid_until = 5_000.

    assert!(cert.is_valid(4_999, DomainState::Known), "before expiry");
    assert!(!cert.is_valid(5_000, DomainState::Known), "at expiry");
    assert!(!cert.is_valid(6_000, DomainState::Known), "after expiry");
}

#[test]
fn unsigned_certificate_is_never_valid() {
    let calibration = calibration(1_000, 5_000);
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();
    let cert = mint(&signer, base_request(&calibration), &slice).unwrap();

    let unsigned = CapabilityCertificate::unsigned(cert.content.clone());
    assert!(!unsigned.verify(&signer), "unsigned does not verify");
    assert!(
        !unsigned.is_valid(0, DomainState::Known),
        "unsigned is never valid even fresh and KNOWN"
    );
}

#[test]
fn non_known_domain_invalidates() {
    let calibration = calibration(1_000, 5_000);
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();
    let cert = mint(&signer, base_request(&calibration), &slice).unwrap();

    // Same instant, only the live domain signature differs.
    assert!(cert.is_valid(4_000, DomainState::Known));
    assert!(!cert.is_valid(4_000, DomainState::Degraded));
    assert!(!cert.is_valid(4_000, DomainState::Unknown));
}

#[test]
fn context_mismatch_refuses_to_bind_metrics() {
    let calibration = calibration(1_000, 5_000);
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();

    let mut req = base_request(&calibration);
    req.hardware = DeviceId::new("other-device").unwrap();
    let err = mint(&signer, req, &slice).unwrap_err();
    assert_eq!(err, CertifyError::ContextMismatch { field: "device" });
}

#[test]
fn canonical_bytes_are_deterministic() {
    let calibration = calibration(1_000, 5_000);
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();

    let a = mint(&signer, base_request(&calibration), &slice).unwrap();
    let b = mint(&signer, base_request(&calibration), &slice).unwrap();
    assert_eq!(
        a.content.canonical_bytes(),
        b.content.canonical_bytes(),
        "identical content => identical bytes"
    );
    assert_eq!(a, b, "mint is a pure function of its inputs");
    assert_eq!(a.signature, b.signature);
}

#[test]
fn serde_round_trips() {
    let calibration = calibration(1_000, 5_000);
    let ledger = synthetic_ledger();
    let slice = ledger.query(&context());
    let signer = cert_signer();
    let cert = mint(&signer, base_request(&calibration), &slice).unwrap();

    let json = serde_json::to_string(&cert).unwrap();
    let back: CapabilityCertificate = serde_json::from_str(&json).unwrap();
    assert_eq!(cert, back);
    // The deserialized certificate still verifies against the same key.
    assert!(back.verify(&signer));
}
