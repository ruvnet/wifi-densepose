//! ADR-153 acceptance tests — types (serde round trips, boundary
//! validation), the SimTransport double, and the ESP32 CSI bridge.
//! FSM/timeout/threshold/SBP coverage lives in [`super::tests_fsm`].
//! All tests are hardware-free (simulation only).

use super::messages::*;
use super::testutil::{csi_frame, params, payload, setup_request};
use super::transport::{
    OpportunisticCsiBridge, SensingFrame, SensingTransport, SimTransport, TransportError,
};
use super::types::*;

// ---------- serde round trips ----------

#[test]
fn serde_round_trips_setup_instance_report_sbp_termination() {
    let req = setup_request(7);
    let json = serde_json::to_string(&req).unwrap();
    assert_eq!(
        serde_json::from_str::<SensingMeasurementSetupRequest>(&json).unwrap(),
        req
    );

    let resp = SensingMeasurementSetupResponse {
        setup_id: req.setup_id,
        status: SetupStatus::Accepted,
    };
    let json = serde_json::to_string(&resp).unwrap();
    assert_eq!(
        serde_json::from_str::<SensingMeasurementSetupResponse>(&json).unwrap(),
        resp
    );

    let instance = SensingMeasurementInstance {
        setup_id: req.setup_id,
        instance_id: MeasurementInstanceId::new(3),
        timestamp_us: 300_000,
    };
    let json = serde_json::to_string(&instance).unwrap();
    assert_eq!(
        serde_json::from_str::<SensingMeasurementInstance>(&json).unwrap(),
        instance
    );

    let report = SensingMeasurementReport {
        setup_id: req.setup_id,
        instance_id: MeasurementInstanceId::new(3),
        payload: payload(42.0),
    };
    let json = serde_json::to_string(&report).unwrap();
    assert_eq!(
        serde_json::from_str::<SensingMeasurementReport>(&json).unwrap(),
        report
    );

    let sbp = SbpRequest {
        profile: SpecProfile::VendorExtension("acme-presensing".into()),
        proxy_setup_id: req.setup_id,
        params: params(),
    };
    let json = serde_json::to_string(&sbp).unwrap();
    assert_eq!(serde_json::from_str::<SbpRequest>(&json).unwrap(), sbp);

    let sbp_resp = SbpResponse {
        proxy_setup_id: req.setup_id,
        status: SbpStatus::Accepted,
    };
    let json = serde_json::to_string(&sbp_resp).unwrap();
    assert_eq!(
        serde_json::from_str::<SbpResponse>(&json).unwrap(),
        sbp_resp
    );

    let term = SensingSessionTermination {
        setup_id: req.setup_id,
        reason: TerminationReason::InitiatorRequested,
    };
    let json = serde_json::to_string(&term).unwrap();
    assert_eq!(
        serde_json::from_str::<SensingSessionTermination>(&json).unwrap(),
        term
    );
}

#[test]
fn serde_rejects_out_of_range_setup_id() {
    assert!(serde_json::from_str::<MeasurementSetupId>("200").is_err());
    assert!(serde_json::from_str::<MeasurementSetupId>("127").is_ok());
}

#[test]
fn serde_rejects_out_of_range_threshold_params() {
    assert!(serde_json::from_str::<ThresholdParams>(r#"{"delta_percent":255}"#).is_err());
    let ok = serde_json::from_str::<ThresholdParams>(r#"{"delta_percent":100}"#).unwrap();
    assert_eq!(ok.delta_percent(), 100);
}

// ---------- validation, no panics ----------

#[test]
fn setup_id_construction_never_panics_and_bounds_hold() {
    for v in 0u8..=255 {
        let result = MeasurementSetupId::new(v);
        assert_eq!(result.is_ok(), v <= MAX_SETUP_ID);
    }
}

#[test]
fn params_validation_rejects_malformed() {
    let mut p = params();
    p.period_ms = MIN_PERIOD_MS - 1;
    assert!(matches!(p.validate(), Err(BfError::InvalidPeriod { .. })));
    p = params();
    p.period_ms = MAX_PERIOD_MS + 1;
    assert!(matches!(p.validate(), Err(BfError::InvalidPeriod { .. })));
    p = params();
    p.burst_instances = 0;
    assert!(matches!(
        p.validate(),
        Err(BfError::InvalidBurstInstances { .. })
    ));
    p = params();
    p.burst_instances = MAX_BURST_INSTANCES + 1;
    assert!(matches!(
        p.validate(),
        Err(BfError::InvalidBurstInstances { .. })
    ));
    p = params();
    p.initiator_role = TransceiverRole::Receiver; // no transmitter anywhere
    assert!(matches!(
        p.validate(),
        Err(BfError::InvalidTransceiverRoles)
    ));
    p = params();
    p.consent = ConsentMode::Disabled;
    assert!(matches!(
        p.validate(),
        Err(BfError::SensingDisabledByPolicy)
    ));
    assert!(ThresholdParams::new(101).is_err());
    assert!(ThresholdParams::new(100).is_ok());
}

#[test]
fn payload_validation_rejects_adversarial_values_without_panic() {
    let adversarial = [
        CsiReportPayload {
            n_subcarriers: 0,
            amplitudes: vec![],
            phases: vec![],
        },
        CsiReportPayload {
            n_subcarriers: u16::MAX,
            amplitudes: vec![1.0; 4],
            phases: vec![0.0; 4],
        },
        CsiReportPayload {
            n_subcarriers: 4,
            amplitudes: vec![1.0; 3],
            phases: vec![0.0; 4],
        },
        CsiReportPayload {
            n_subcarriers: 2,
            amplitudes: vec![f32::NAN, 1.0],
            phases: vec![0.0; 2],
        },
        CsiReportPayload {
            n_subcarriers: 2,
            amplitudes: vec![1.0, f32::INFINITY],
            phases: vec![0.0; 2],
        },
        CsiReportPayload {
            n_subcarriers: 2,
            amplitudes: vec![-1.0, 1.0],
            phases: vec![0.0; 2],
        },
        CsiReportPayload {
            n_subcarriers: 2,
            amplitudes: vec![1.0; 2],
            phases: vec![f32::NEG_INFINITY, 0.0],
        },
    ];
    for p in adversarial {
        assert!(p.validate().is_err());
    }
    assert!(payload(5.0).validate().is_ok());
}

#[test]
fn spec_profile_compatibility() {
    let published = SpecProfile::Ieee80211Bf2025;
    assert!(published.accepts(&SpecProfile::DraftCompatible));
    assert!(published.accepts(&SpecProfile::Ieee80211Bf2025));
    assert!(!published.accepts(&SpecProfile::VendorExtension("x".into())));
    let vendor = SpecProfile::VendorExtension("x".into());
    assert!(vendor.accepts(&SpecProfile::VendorExtension("x".into())));
    assert!(!vendor.accepts(&SpecProfile::VendorExtension("y".into())));
}

// ---------- bridge: ESP32 CSI → standardized report ----------

#[test]
fn bridge_maps_csi_batches_to_measurement_reports() {
    let setup_id = MeasurementSetupId::new(1).unwrap();
    let mut bridge = OpportunisticCsiBridge::new(setup_id, 4).unwrap();
    assert!(OpportunisticCsiBridge::new(setup_id, 0).is_err());

    // 3 frames: no report yet. 4th completes the instance batch.
    for _ in 0..3 {
        assert!(bridge.ingest(&csi_frame(8, 30, 40)).is_none());
    }
    let report = bridge
        .ingest(&csi_frame(8, 30, 40))
        .expect("batch complete");
    assert_eq!(report.setup_id, setup_id);
    assert_eq!(report.instance_id.value(), 0);
    assert_eq!(report.payload.n_subcarriers, 8);
    assert!(report.payload.validate().is_ok());
    // |30 + 40i| = 50 on every subcarrier of every frame.
    assert!(report
        .payload
        .amplitudes
        .iter()
        .all(|a| (a - 50.0).abs() < 1e-3));

    // Invalid (all-zero) frames are skipped and do not advance the batch.
    for _ in 0..10 {
        assert!(bridge.ingest(&csi_frame(8, 0, 0)).is_none());
    }
    // A mid-batch subcarrier-shape change restarts the batch on the new shape.
    assert!(bridge.ingest(&csi_frame(8, 10, 0)).is_none());
    assert!(bridge.ingest(&csi_frame(4, 10, 0)).is_none()); // restart at n=4
    for _ in 0..2 {
        assert!(bridge.ingest(&csi_frame(4, 10, 0)).is_none());
    }
    let report = bridge.ingest(&csi_frame(4, 10, 0)).expect("second batch");
    assert_eq!(report.instance_id.value(), 1); // instance counter advanced
    assert_eq!(report.payload.n_subcarriers, 4);
}

// ---------- transport ----------

#[test]
fn sim_transport_scripted_responses_and_failures() {
    let mut t = SimTransport::new();
    let resp = SensingMeasurementSetupResponse {
        setup_id: MeasurementSetupId::new(7).unwrap(),
        status: SetupStatus::Accepted,
    };
    t.script_response(SensingFrame::SetupResponse(resp));
    assert!(t.poll_frame().is_none());
    t.send_setup_request(setup_request(7)).unwrap();
    assert_eq!(t.poll_frame(), Some(SensingFrame::SetupResponse(resp)));
    assert_eq!(t.sent().len(), 1);

    let mut tiny = SimTransport::with_capacity(1);
    tiny.send_setup_request(setup_request(1)).unwrap();
    assert_eq!(
        tiny.send_setup_request(setup_request(2)),
        Err(TransportError::QueueFull { capacity: 1 })
    );

    let mut down = SimTransport::new();
    down.set_link_down(true);
    assert_eq!(
        down.send_setup_request(setup_request(1)),
        Err(TransportError::LinkDown)
    );
}
