//! ADR-153 acceptance tests — session FSM full cycle, rejection paths,
//! timeout handling, threshold-based reporting, single-role enforcement,
//! and adversarial no-panic coverage. SBP flows live in [`super::tests_sbp`];
//! type/serde/transport/bridge tests in [`super::tests`]. All tests are
//! hardware-free (simulation only).

use super::messages::*;
use super::session::{
    Action, CloseReason, SensingSession, SessionConfig, SessionEvent, SessionState,
};
use super::table::SessionTable;
use super::testutil::{dispatch, ferry, params, payload, pump, setup_request};
use super::transport::{SensingFrame, SimTransport};
use super::types::*;
use crate::csi_frame::Bandwidth;

// ---------- FSM: full cycle ----------

#[test]
fn fsm_full_cycle_setup_measure_report_terminate() {
    let cfg = SessionConfig::default();
    let mut initiator = SensingSession::new_initiator(cfg.clone());
    let mut responder = SensingSession::new_responder(cfg);
    let mut wire_i = SimTransport::new();
    let mut wire_r = SimTransport::new();

    // Idle → SetupNegotiating
    dispatch(
        &mut initiator,
        SessionEvent::StartSetup(setup_request(7)),
        &mut wire_i,
    );
    assert_eq!(initiator.state(), SessionState::SetupNegotiating);

    // Responder accepts → Active
    ferry(&mut wire_i, &mut wire_r);
    pump(&mut responder, &mut wire_r);
    assert_eq!(responder.state(), SessionState::Active);

    // Initiator sees Accepted → Active + first instance trigger on the wire
    ferry(&mut wire_r, &mut wire_i);
    pump(&mut initiator, &mut wire_i);
    assert_eq!(initiator.state(), SessionState::Active);
    assert!(wire_i
        .sent()
        .iter()
        .any(|f| matches!(f, SensingFrame::InstanceTrigger(i) if i.setup_id.value() == 7)));

    // Responder captures a measurement → report on the wire
    wire_i.drain_sent();
    let actions = dispatch(
        &mut responder,
        SessionEvent::MeasurementCaptured {
            instance_id: MeasurementInstanceId::new(0),
            payload: payload(10.0),
        },
        &mut wire_r,
    );
    assert!(actions.iter().any(|a| matches!(a, Action::SendReport(_))));

    // Initiator delivers the report to its consumer
    ferry(&mut wire_r, &mut wire_i);
    let actions = pump(&mut initiator, &mut wire_i);
    assert!(actions
        .iter()
        .any(|a| matches!(a, Action::DeliverReport(_))));

    // Active → Terminating → Idle (peer notified, quiescence completes)
    wire_i.drain_sent();
    dispatch(
        &mut initiator,
        SessionEvent::Terminate(TerminationReason::InitiatorRequested),
        &mut wire_i,
    );
    assert_eq!(initiator.state(), SessionState::Terminating);
    ferry(&mut wire_i, &mut wire_r);
    let actions = pump(&mut responder, &mut wire_r);
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::SessionClosed(CloseReason::Terminated(
            TerminationReason::InitiatorRequested
        ))
    )));
    assert_eq!(responder.state(), SessionState::Idle);
    let actions = initiator.handle(SessionEvent::Timeout).unwrap();
    assert!(actions
        .iter()
        .any(|a| matches!(a, Action::SessionClosed(CloseReason::Completed))));
    assert_eq!(initiator.state(), SessionState::Idle);
}

// ---------- FSM: rejection paths ----------

#[test]
fn responder_rejects_unsupported_bandwidth_and_initiator_resets() {
    let mut cfg = SessionConfig::default();
    cfg.capabilities = SensingCapabilities::esp32_opportunistic(); // max 40 MHz
    let mut responder = SensingSession::new_responder(cfg);
    let mut initiator = SensingSession::new_initiator(SessionConfig::default());

    let mut req = setup_request(3);
    req.params.bandwidth = Bandwidth::Bw80;
    initiator
        .handle(SessionEvent::StartSetup(req.clone()))
        .unwrap();

    let actions = responder
        .handle(SessionEvent::SetupRequestReceived(req))
        .unwrap();
    let resp = match &actions[..] {
        [Action::SendSetupResponse(r)] => *r,
        other => panic!("expected single rejection response, got {other:?}"),
    };
    assert_eq!(resp.status, SetupStatus::RejectedUnsupportedParams);
    assert_eq!(responder.state(), SessionState::Idle);

    let actions = initiator
        .handle(SessionEvent::SetupResponseReceived(resp))
        .unwrap();
    assert!(actions.iter().any(|a| matches!(
        a,
        Action::SessionClosed(CloseReason::SetupRejected(
            SetupStatus::RejectedUnsupportedParams
        ))
    )));
    assert_eq!(initiator.state(), SessionState::Idle);
}

#[test]
fn invalid_period_rejected_on_both_sides() {
    let mut req = setup_request(4);
    req.params.period_ms = 1; // below MIN_PERIOD_MS
    let mut initiator = SensingSession::new_initiator(SessionConfig::default());
    assert!(matches!(
        initiator.handle(SessionEvent::StartSetup(req.clone())),
        Err(BfError::InvalidPeriod { period_ms: 1 })
    ));
    assert_eq!(initiator.state(), SessionState::Idle);

    let mut responder = SensingSession::new_responder(SessionConfig::default());
    let actions = responder
        .handle(SessionEvent::SetupRequestReceived(req))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::RejectedUnsupportedParams,
            ..
        })]
    ));
}

#[test]
fn duplicate_setup_id_rejected_by_session_table() {
    let mut table = SessionTable::new(SessionConfig::default());
    let actions = table.handle_setup_request(setup_request(9)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::Accepted,
            ..
        })]
    ));
    let actions = table.handle_setup_request(setup_request(9)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::RejectedSetupIdCollision,
            ..
        })]
    ));
    assert_eq!(table.active_setups(), 1);
}

#[test]
fn capacity_and_policy_and_profile_rejections() {
    // Capacity
    let mut cfg = SessionConfig::default();
    cfg.capabilities.max_active_setups = 1;
    let mut table = SessionTable::new(cfg);
    table.handle_setup_request(setup_request(1)).unwrap();
    let actions = table.handle_setup_request(setup_request(2)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::RejectedCapacity,
            ..
        })]
    ));

    // Consent policy
    let mut responder = SensingSession::new_responder(SessionConfig::default());
    let mut req = setup_request(5);
    req.params.consent = ConsentMode::Disabled;
    let actions = responder
        .handle(SessionEvent::SetupRequestReceived(req))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::RejectedByPolicy,
            ..
        })]
    ));

    // Incompatible profile
    let mut cfg = SessionConfig::default();
    cfg.profile = SpecProfile::VendorExtension("acme".into());
    let mut responder = SensingSession::new_responder(cfg);
    let actions = responder
        .handle(SessionEvent::SetupRequestReceived(setup_request(6)))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::RejectedIncompatibleProfile,
            ..
        })]
    ));
}

// ---------- FSM: timeouts ----------

#[test]
fn negotiation_timeout_returns_typed_error_and_resets_to_idle() {
    let mut initiator = SensingSession::new_initiator(SessionConfig::default()); // 3 timeouts
    initiator
        .handle(SessionEvent::StartSetup(setup_request(7)))
        .unwrap();

    // First two timeouts re-send the pending request.
    for _ in 0..2 {
        let actions = initiator.handle(SessionEvent::Timeout).unwrap();
        assert!(matches!(actions[..], [Action::SendSetupRequest(_)]));
        assert_eq!(initiator.state(), SessionState::SetupNegotiating);
    }
    // Third gives up: typed error + Idle.
    assert_eq!(
        initiator.handle(SessionEvent::Timeout),
        Err(BfError::NegotiationTimeout {
            setup_id: 7,
            attempts: 3
        })
    );
    assert_eq!(initiator.state(), SessionState::Idle);
}

#[test]
fn active_missed_instance_timeouts_terminate_session() {
    let mut responder = SensingSession::new_responder(SessionConfig::default()); // 5 missed max
    responder
        .handle(SessionEvent::SetupRequestReceived(setup_request(2)))
        .unwrap();
    assert_eq!(responder.state(), SessionState::Active);
    for _ in 0..4 {
        assert!(responder.handle(SessionEvent::Timeout).unwrap().is_empty());
    }
    let actions = responder.handle(SessionEvent::Timeout).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendTermination(SensingSessionTermination {
            reason: TerminationReason::Timeout,
            ..
        })]
    ));
    assert_eq!(responder.state(), SessionState::Terminating);
    let actions = responder.handle(SessionEvent::Timeout).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SessionClosed(CloseReason::Completed)]
    ));
    assert_eq!(responder.state(), SessionState::Idle);
}

// ---------- threshold-based reporting ----------

#[test]
fn threshold_report_emitted_only_when_threshold_crossed() {
    let mut responder = SensingSession::new_responder(SessionConfig::default());
    let mut req = setup_request(8);
    req.params.reporting = ReportingConfig::ThresholdBased(ThresholdParams::new(20).unwrap());
    responder
        .handle(SessionEvent::SetupRequestReceived(req))
        .unwrap();

    let capture = |mean: f32| SessionEvent::MeasurementCaptured {
        instance_id: MeasurementInstanceId::new(0),
        payload: payload(mean),
    };
    // First measurement always reported (establishes the baseline).
    let actions = responder.handle(capture(100.0)).unwrap();
    assert!(matches!(actions[..], [Action::SendReport(_)]));
    // +10% — below threshold, suppressed; baseline stays at 100.
    assert!(responder.handle(capture(110.0)).unwrap().is_empty());
    // +19% vs the *reported* baseline — still suppressed.
    assert!(responder.handle(capture(119.0)).unwrap().is_empty());
    // +50% — crossed, reported, baseline moves to 150.
    let actions = responder.handle(capture(150.0)).unwrap();
    assert!(matches!(actions[..], [Action::SendReport(_)]));
    // 150 → 125 is ~16.7% — suppressed against the new baseline.
    assert!(responder.handle(capture(125.0)).unwrap().is_empty());
}

// ---------- consecutive missed-instance semantics ----------

#[test]
fn missed_instance_budget_is_consecutive_not_cumulative() {
    // Review finding 2: a successful measurement must reset the
    // missed-instance counter — `max_missed_instances` bounds *consecutive*
    // misses (as documented on SessionConfig), not cumulative ones.
    let mut responder = SensingSession::new_responder(SessionConfig::default()); // 5 missed max
    responder
        .handle(SessionEvent::SetupRequestReceived(setup_request(2)))
        .unwrap();
    assert_eq!(responder.state(), SessionState::Active);
    let capture = || SessionEvent::MeasurementCaptured {
        instance_id: MeasurementInstanceId::new(0),
        payload: payload(10.0),
    };

    // Miss 4, then succeed once...
    for _ in 0..4 {
        assert!(responder.handle(SessionEvent::Timeout).unwrap().is_empty());
    }
    let actions = responder.handle(capture()).unwrap();
    assert!(matches!(actions[..], [Action::SendReport(_)]));

    // ...so 4 more misses still leave the session alive.
    for _ in 0..4 {
        assert!(responder.handle(SessionEvent::Timeout).unwrap().is_empty());
        assert_eq!(responder.state(), SessionState::Active);
    }
    // The 5th consecutive miss terminates.
    let actions = responder.handle(SessionEvent::Timeout).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendTermination(SensingSessionTermination {
            reason: TerminationReason::Timeout,
            ..
        })]
    ));
    assert_eq!(responder.state(), SessionState::Terminating);
}

// ---------- single-role enforcement & out-of-state commands ----------

#[test]
fn initiator_role_session_rejects_inbound_setup_and_sbp_requests() {
    // Review finding 4a: single-role design — a peer must not be able to
    // hijack an initiator-role session into the responder path.
    let mut initiator = SensingSession::new_initiator(SessionConfig::default());
    let actions = initiator
        .handle(SessionEvent::SetupRequestReceived(setup_request(3)))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSetupResponse(SensingMeasurementSetupResponse {
            status: SetupStatus::RejectedNotSupported,
            ..
        })]
    ));
    assert_eq!(initiator.state(), SessionState::Idle);

    let sbp = SbpRequest {
        profile: SpecProfile::Ieee80211Bf2025,
        proxy_setup_id: MeasurementSetupId::new(4).unwrap(),
        params: params(),
    };
    let actions = initiator
        .handle(SessionEvent::SbpRequestReceived(sbp))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedNotSupported,
            ..
        })]
    ));
    assert_eq!(initiator.state(), SessionState::Idle);
    assert!(!initiator.is_sbp_proxy());
}

#[test]
fn local_start_commands_error_outside_idle() {
    // Review finding 4b: StartSetup/StartSbp outside Idle are caller bugs
    // and must surface as typed errors, not silent no-ops.
    let sbp = SbpRequest {
        profile: SpecProfile::Ieee80211Bf2025,
        proxy_setup_id: MeasurementSetupId::new(13).unwrap(),
        params: params(),
    };
    let start_err = |s: &mut SensingSession, expected: SessionState| {
        assert!(matches!(
            s.handle(SessionEvent::StartSetup(setup_request(8))),
            Err(BfError::InvalidStateForCommand { .. })
        ));
        assert!(matches!(
            s.handle(SessionEvent::StartSbp(sbp.clone())),
            Err(BfError::InvalidStateForCommand { .. })
        ));
        // The rejected commands must not disturb the session.
        assert_eq!(s.state(), expected);
    };

    let mut s = SensingSession::new_initiator(SessionConfig::default());
    s.handle(SessionEvent::StartSetup(setup_request(7)))
        .unwrap();
    start_err(&mut s, SessionState::SetupNegotiating);

    s.handle(SessionEvent::SetupResponseReceived(
        SensingMeasurementSetupResponse {
            setup_id: MeasurementSetupId::new(7).unwrap(),
            status: SetupStatus::Accepted,
        },
    ))
    .unwrap();
    start_err(&mut s, SessionState::Active);
    // Genuinely ignorable stray frames remain no-ops in Active.
    assert!(s
        .handle(SessionEvent::SbpResponseReceived(SbpResponse {
            proxy_setup_id: MeasurementSetupId::new(7).unwrap(),
            status: SbpStatus::Accepted,
        }))
        .unwrap()
        .is_empty());

    s.handle(SessionEvent::Terminate(
        TerminationReason::InitiatorRequested,
    ))
    .unwrap();
    start_err(&mut s, SessionState::Terminating);
}

// ---------- adversarial: no panics anywhere ----------

#[test]
fn malformed_and_out_of_state_events_never_panic() {
    let junk_payload = CsiReportPayload {
        n_subcarriers: 3,
        amplitudes: vec![f32::NAN, -5.0, f32::INFINITY],
        phases: vec![f32::NAN],
    };
    let bad_report = SensingMeasurementReport {
        setup_id: MeasurementSetupId::new(99).unwrap(),
        instance_id: MeasurementInstanceId::new(255),
        payload: junk_payload.clone(),
    };
    let events: Vec<SessionEvent> = vec![
        SessionEvent::StartSetup(setup_request(0)),
        SessionEvent::StartSbp(SbpRequest {
            profile: SpecProfile::DraftCompatible,
            proxy_setup_id: MeasurementSetupId::new(0).unwrap(),
            params: params(),
        }),
        SessionEvent::SetupRequestReceived(setup_request(127)),
        SessionEvent::SetupResponseReceived(SensingMeasurementSetupResponse {
            setup_id: MeasurementSetupId::new(50).unwrap(),
            status: SetupStatus::RejectedCapacity,
        }),
        SessionEvent::SbpResponseReceived(SbpResponse {
            proxy_setup_id: MeasurementSetupId::new(50).unwrap(),
            status: SbpStatus::RejectedByPolicy,
        }),
        SessionEvent::InstanceElapsed,
        SessionEvent::MeasurementCaptured {
            instance_id: MeasurementInstanceId::new(0),
            payload: junk_payload,
        },
        SessionEvent::ReportReceived(bad_report),
        SessionEvent::Timeout,
        SessionEvent::Terminate(TerminationReason::PolicyChange),
        SessionEvent::TerminationReceived(SensingSessionTermination {
            setup_id: MeasurementSetupId::new(1).unwrap(),
            reason: TerminationReason::Timeout,
        }),
    ];
    // Drive both roles through every event repeatedly from whatever state
    // each lands in; typed errors are fine, panics are not.
    for session in [
        &mut SensingSession::new_initiator(SessionConfig::default()),
        &mut SensingSession::new_responder(SessionConfig::default()),
    ] {
        for _ in 0..4 {
            for event in &events {
                let _ = session.handle(event.clone());
            }
        }
    }
}
