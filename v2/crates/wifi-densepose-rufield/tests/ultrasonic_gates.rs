//! ADR-262 P4 gates for the second modality.
//!
//! Same shape as `p1_gates.rs`: round-trip, fusability, privacy-safety,
//! determinism. Not accuracy — nothing here claims an ultrasonic scan is
//! *right*, only that it is well-formed, signed, correctly classified, and that
//! the things it must refuse to do, it refuses.
//!
//! The fixture is not hand-written. `batvu_living_room.ultrasonic.jsonl` is
//! produced by BatVu's own `npm run artifacts` — a 72-ping simulated sweep,
//! written by its TypeScript emitter — and copied here verbatim. The same file
//! is a test fixture in `ruvnet/rufield`, so a schema drift between BatVu's
//! emitter and RuField's parser fails a build in one of three repositories
//! rather than an ingest in a deployment.

use rufield_core::{FieldEvent, FusionEngine, InferenceQuery, Modality, PrivacyClass};
use rufield_fusion::RuFieldFusion;
use rufield_provenance::{is_fusable, verify_event};
use wifi_densepose_rufield::{
    network_egress_allowed, ScanSource, UltrasonicScan, ULTRASONIC_EGRESS_CLASS,
};

const SCAN: &str = include_str!("fixtures/batvu_living_room.ultrasonic.jsonl");

fn scan() -> UltrasonicScan {
    UltrasonicScan::load(SCAN, "living_room", ScanSource::Simulated).expect("fixture loads")
}

// ── round-trip ───────────────────────────────────────────────────────────────

#[test]
fn gate_round_trip_every_event_is_well_formed_and_serializes() {
    let mut s = scan();
    assert_eq!(s.ping_count(), 72);
    assert_eq!(s.device_id(), "batvu-reference-01");
    assert!(!s.calibration_id().is_empty());

    let events = s.events().expect("events");
    assert_eq!(events.len(), 72);

    let mut previous = 0u64;
    for event in &events {
        event
            .validate_evidence_at(event.timestamp_ns)
            .expect("structural evidence invariants hold");

        assert_eq!(event.tensor.modality, Modality::Ultrasonic);
        assert_eq!(event.sensor.vendor, "batvu");
        assert_eq!(event.observation.zone_id.as_deref(), Some("living_room"));

        // Every profile value finite and non-negative. `FieldTensor::validate`
        // checks only shape and axis rank, so a NaN would serialize to JSON
        // `null` and then fail to deserialize as an f32 on the far side of the
        // wire — the worst place to find it. The adapter rejects it at parse.
        assert!(
            event
                .tensor
                .values
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0),
            "profile values are finite and non-negative"
        );

        // Strictly increasing, which RuField's replay watermark requires in
        // every trust mode and silently drops what does not satisfy.
        assert!(event.timestamp_ns > previous);
        previous = event.timestamp_ns;

        // serde round-trip, byte-stable.
        let json = serde_json::to_string(event).expect("serializes");
        let back: FieldEvent = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(*event, back);
    }

    let mut ids: Vec<&str> = events.iter().map(|e| e.event_id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), events.len(), "event ids are unique");
}

// ── fusability ───────────────────────────────────────────────────────────────

#[test]
fn gate_every_event_carries_a_signature_that_verifies() {
    let mut s = scan();
    let events = s.events().expect("events");
    for event in &events {
        verify_event(event).expect("ed25519 signature verifies");
        assert!(is_fusable(event));
    }

    // And tampering breaks it, so the signature is load-bearing rather than
    // decorative.
    let mut tampered = events[0].clone();
    tampered.tensor.values[0] = 999.0;
    assert!(verify_event(&tampered).is_err());
}

#[test]
fn gate_fusion_ingests_every_event() {
    let mut s = scan();
    let mut engine = RuFieldFusion::new();
    for event in s.events().expect("events") {
        engine.ingest(event).expect("fusion accepts the event");
    }
}

// ── privacy safety — the correctness item ────────────────────────────────────

#[test]
fn gate_coarse_scan_passes_the_egress_gate_intact() {
    // The claim the module makes: configuring the adapter for the coarse output
    // means the egress gate has nothing left to catch. If a submodule bump ever
    // changes the adapter's default class, this is where it surfaces.
    let mut s = scan();
    let all = s.events().expect("events").len();

    let mut s = scan();
    let egress = s.egress_events().expect("egress events").len();

    assert_eq!(all, 72);
    assert_eq!(
        egress, all,
        "no event is dropped at the gate in coarse mode"
    );
}

#[test]
fn gate_every_event_is_p1_on_both_tensor_and_observation() {
    let mut s = scan();
    for event in s.events().expect("events") {
        assert_eq!(event.tensor.privacy_class, ULTRASONIC_EGRESS_CLASS);
        assert_eq!(event.observation.privacy_class, ULTRASONIC_EGRESS_CLASS);
        // Both, because the guard is conjunctive over the pair — a P0 tensor
        // under a P1 observation is exactly the composite leak the default
        // guard exists to close.
        assert!(network_egress_allowed(event.tensor.privacy_class, false));
        assert!(network_egress_allowed(
            event.observation.privacy_class,
            false
        ));
    }
    assert_eq!(ULTRASONIC_EGRESS_CLASS, PrivacyClass::P1);
}

#[test]
fn gate_ultrasonic_can_never_reach_the_identity_tiers() {
    // P5 is only reachable through `identity_evidence`, which
    // `validate_evidence_at` restricts to BLE advertisement RSSI — so an
    // ultrasonic event carrying it is a hard validation failure rather than a
    // policy question. The ceiling is structural.
    let mut s = scan();
    for event in s.events().expect("events") {
        assert!(event.tensor.privacy_class < PrivacyClass::P4);
        assert!(event.observation.privacy_class < PrivacyClass::P4);
        assert!(event.observation.identity_evidence.is_none());
        assert!(event.observation.channel_sounding_provenance.is_none());
    }
}

#[test]
fn gate_no_event_claims_a_person() {
    // The tempting mistake, refused deliberately. `presence` is one of exactly
    // six feature keys the fusion window reads, and populating it would light
    // up the shipped `person_present` rule. One transducer pair cannot
    // distinguish a person from a coat over the back of a chair.
    let mut s = scan();
    for event in s.events().expect("events") {
        for forbidden in ["presence", "breathing_band", "posture_height", "transient"] {
            assert!(
                !event.observation.features.contains_key(forbidden),
                "an ultrasonic event must not claim `{forbidden}`"
            );
        }
        for label in &event.observation.labels {
            assert!(
                !label.contains("person") && !label.contains("presence"),
                "unexpected personhood label: {label}"
            );
        }
    }
}

/// The honest negative result, asserted rather than avoided.
///
/// Because nothing claims presence, a BatVu scan produces no fused inferences
/// under the shipped rules. Two independent reasons, and pinning both matters
/// because fixing only the first would look like progress and change nothing:
/// no rule lists `"ultrasonic"` among its inputs, and the engine's feature
/// vocabulary is entirely statements about a body, so `range_m` has nothing to
/// drive. RuField v0.1 has no predicate for static geometry.
#[test]
fn gate_no_inferences_and_that_is_the_correct_outcome() {
    let mut s = scan();
    let mut engine = RuFieldFusion::new();
    for event in s.events().expect("events") {
        engine.ingest(event).expect("ingest");
    }
    let inferences = engine
        .infer(&InferenceQuery {
            zone_id: Some("living_room".into()),
            labels: vec![],
            track_id: None,
            as_of_ns: None,
        })
        .expect("infer");
    assert!(
        inferences.is_empty(),
        "no shipped rule can fire on a range-only sensor: {inferences:?}"
    );
}

// ── trust tier ───────────────────────────────────────────────────────────────

#[test]
fn gate_a_recording_cannot_relabel_itself_into_a_higher_trust_tier() {
    // The fixture declares `simulated`. An operator pointing a capture-trusting
    // deployment at it gets a hard refusal rather than a silently upgraded
    // tier — and the reverse is refused too, so neither direction is a quiet
    // reinterpretation.
    let err = UltrasonicScan::load(SCAN, "living_room", ScanSource::DeviceCapture)
        .expect_err("must refuse");
    assert!(
        err.to_string().contains("device_capture"),
        "the refusal names the mismatch: {err}"
    );
}

#[test]
fn gate_simulated_recordings_are_marked_synthetic() {
    // Not cosmetic: `synthetic: true` is what keeps simulator output out of
    // captured-replay and production trust, which reject it before any key
    // lookup. Marking it otherwise to get it accepted is the §11 invariant
    // violation ADR-262 forbids.
    let mut s = scan();
    for event in s.events().expect("events") {
        assert!(event.provenance.synthetic);
    }
}

// ── determinism ──────────────────────────────────────────────────────────────

#[test]
fn gate_same_recording_yields_a_byte_identical_event_stream() {
    let mut a = scan();
    let mut b = scan();
    let left = serde_json::to_string(&a.events().expect("events")).expect("serializes");
    let right = serde_json::to_string(&b.events().expect("events")).expect("serializes");
    assert_eq!(left, right);
}

#[test]
fn gate_a_malformed_recording_is_refused_whole() {
    // No partial ingest. A stream that half-replays and then dies is the worst
    // outcome for a consumer, because it has already acted on the good prefix.
    let mut bad = SCAN.lines().take(4).collect::<Vec<_>>().join("\n");
    bad.push_str("\n{\"timestamp\":1756162800,\"source\":\"simulated\",\"device_id\":\"x\"}\n");
    let err =
        UltrasonicScan::load(&bad, "living_room", ScanSource::Simulated).expect_err("must refuse");
    assert!(err.to_string().contains("rejected"));
}
