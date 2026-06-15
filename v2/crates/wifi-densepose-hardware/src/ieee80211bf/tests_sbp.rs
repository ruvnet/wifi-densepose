//! ADR-153 sensing-by-proxy (SBP) acceptance tests — proxy lifecycle
//! (re-triggering + report relay), client flow, table-driven AP entry
//! point, and the single-validation-path status mapping. Other FSM tests
//! live in [`super::tests_fsm`]; type/serde/transport/bridge tests in
//! [`super::tests`]. All tests are hardware-free (simulation only).

use super::messages::*;
use super::session::{
    Action, CloseReason, SensingSession, SessionConfig, SessionEvent, SessionState,
};
use super::table::SessionTable;
use super::testutil::{params, payload};
use super::transport::{action_to_frame, frame_to_event, SensingFrame};
use super::types::*;
use crate::csi_frame::Bandwidth;

fn sbp_request(id: u8) -> SbpRequest {
    SbpRequest {
        profile: SpecProfile::Ieee80211Bf2025,
        proxy_setup_id: MeasurementSetupId::new(id).unwrap(),
        params: params(),
    }
}

#[test]
fn sbp_proxy_request_maps_to_standard_responder_path() {
    // Proxy AP: accepts the SBP request and initiates an ordinary setup
    // toward the sensing responder — no direct sensor coupling.
    let mut proxy = SensingSession::new_responder(SessionConfig::default());
    let actions = proxy
        .handle(SessionEvent::SbpRequestReceived(sbp_request(11)))
        .unwrap();
    let forwarded = match &actions[..] {
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::Accepted,
            ..
        }), Action::SendSetupRequest(req)] => req.clone(),
        other => panic!("expected SBP accept + setup request, got {other:?}"),
    };
    assert_eq!(proxy.state(), SessionState::SetupNegotiating);
    assert_eq!(forwarded.setup_id.value(), 11);

    // The forwarded request drives a *normal* responder session.
    let mut responder = SensingSession::new_responder(SessionConfig::default());
    let actions = responder
        .handle(SessionEvent::SetupRequestReceived(forwarded))
        .unwrap();
    let resp = match &actions[..] {
        [Action::SendSetupResponse(r)] => *r,
        other => panic!("expected accept, got {other:?}"),
    };
    assert_eq!(resp.status, SetupStatus::Accepted);
    proxy
        .handle(SessionEvent::SetupResponseReceived(resp))
        .unwrap();
    assert_eq!(proxy.state(), SessionState::Active);
}

#[test]
fn sbp_client_flow_and_rejections() {
    let mut client = SensingSession::new_initiator(SessionConfig::default());
    let sbp = sbp_request(12);
    let actions = client.handle(SessionEvent::StartSbp(sbp.clone())).unwrap();
    assert!(matches!(actions[..], [Action::SendSbpRequest(_)]));
    let accept = SbpResponse {
        proxy_setup_id: sbp.proxy_setup_id,
        status: SbpStatus::Accepted,
    };
    client
        .handle(SessionEvent::SbpResponseReceived(accept))
        .unwrap();
    assert_eq!(client.state(), SessionState::Active);
    // Proxied report is delivered to the local consumer.
    let report = SensingMeasurementReport {
        setup_id: sbp.proxy_setup_id,
        instance_id: MeasurementInstanceId::new(0),
        payload: payload(1.0),
    };
    let actions = client.handle(SessionEvent::ReportReceived(report)).unwrap();
    assert!(matches!(actions[..], [Action::DeliverReport(_)]));

    // A proxy without SBP capability rejects.
    let mut cfg = SessionConfig::default();
    cfg.capabilities.sensing_by_proxy = false;
    let mut no_sbp = SensingSession::new_responder(cfg);
    let actions = no_sbp
        .handle(SessionEvent::SbpRequestReceived(sbp))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedNotSupported,
            ..
        })]
    ));
    assert_eq!(no_sbp.state(), SessionState::Idle);
}

#[test]
fn sbp_proxy_full_lifecycle_retriggers_and_relays() {
    // Review finding 1: the SBP proxy is a first-class mode — after the
    // proxied setup is accepted it keeps driving measurement instances on
    // InstanceElapsed (like an initiator) and relays every received report
    // to the SBP client in addition to local delivery.
    let mut proxy = SensingSession::new_responder(SessionConfig::default());

    // Accept: SBP response to the client + proxied setup to the responder.
    let actions = proxy
        .handle(SessionEvent::SbpRequestReceived(sbp_request(21)))
        .unwrap();
    let forwarded = match &actions[..] {
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::Accepted,
            ..
        }), Action::SendSetupRequest(req)] => req.clone(),
        other => panic!("expected SBP accept + setup request, got {other:?}"),
    };
    assert!(proxy.is_sbp_proxy());

    // Responder accepts → proxy Active, instance 0 triggered.
    let actions = proxy
        .handle(SessionEvent::SetupResponseReceived(
            SensingMeasurementSetupResponse {
                setup_id: forwarded.setup_id,
                status: SetupStatus::Accepted,
            },
        ))
        .unwrap();
    assert_eq!(proxy.state(), SessionState::Active);
    match &actions[..] {
        [Action::TriggerInstance(i)] => assert_eq!(i.instance_id.value(), 0),
        other => panic!("expected instance 0 trigger, got {other:?}"),
    }

    // InstanceElapsed re-triggers instance 1+ (proxy drives the schedule).
    let actions = proxy.handle(SessionEvent::InstanceElapsed).unwrap();
    match &actions[..] {
        [Action::TriggerInstance(i)] => assert_eq!(i.instance_id.value(), 1),
        other => panic!("expected instance 1 trigger, got {other:?}"),
    }

    // A report from the sensing responder is delivered locally AND relayed.
    let report = SensingMeasurementReport {
        setup_id: forwarded.setup_id,
        instance_id: MeasurementInstanceId::new(1),
        payload: payload(5.0),
    };
    let actions = proxy
        .handle(SessionEvent::ReportReceived(report.clone()))
        .unwrap();
    assert_eq!(
        actions,
        vec![
            Action::DeliverReport(report.clone()),
            Action::RelaySbpReport(report.clone()),
        ]
    );
    // The relay action maps to a frame toward the SBP client, which
    // consumes it through the standard report path.
    let frame = action_to_frame(&Action::RelaySbpReport(report.clone())).unwrap();
    assert_eq!(frame, SensingFrame::SbpReport(report.clone()));
    assert_eq!(
        frame_to_event(frame),
        Some(SessionEvent::ReportReceived(report))
    );

    // Terminate cleanly: notify the responder, quiesce back to Idle.
    let actions = proxy
        .handle(SessionEvent::Terminate(
            TerminationReason::InitiatorRequested,
        ))
        .unwrap();
    assert!(matches!(actions[..], [Action::SendTermination(_)]));
    assert_eq!(proxy.state(), SessionState::Terminating);
    let actions = proxy.handle(SessionEvent::Timeout).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SessionClosed(CloseReason::Completed)]
    ));
    assert_eq!(proxy.state(), SessionState::Idle);
    assert!(!proxy.is_sbp_proxy());
}

#[test]
fn session_table_routes_sbp_end_to_end() {
    // Review finding 3: the table has a first-class SBP entry point with
    // the same collision/capacity guards as direct setups — a table-driven
    // AP accepts SBP instead of silently dropping it.
    let mut table = SessionTable::new(SessionConfig::default());
    let actions = table.handle_sbp_request(sbp_request(31)).unwrap();
    let forwarded = match &actions[..] {
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::Accepted,
            ..
        }), Action::SendSetupRequest(req)] => req.clone(),
        other => panic!("expected SBP accept + setup request, got {other:?}"),
    };
    let setup_id = forwarded.setup_id;
    assert_eq!(table.active_setups(), 1);
    assert!(table.session(setup_id).unwrap().is_sbp_proxy());

    // Proxy-setup-ID collision while the first proxy is live.
    let actions = table.handle_sbp_request(sbp_request(31)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedSetupIdCollision,
            ..
        })]
    ));

    // Drive the proxied negotiation to Active through the table.
    let actions = table
        .handle_for(
            setup_id,
            SessionEvent::SetupResponseReceived(SensingMeasurementSetupResponse {
                setup_id,
                status: SetupStatus::Accepted,
            }),
        )
        .unwrap();
    assert!(matches!(actions[..], [Action::TriggerInstance(_)]));
    assert_eq!(
        table.session(setup_id).unwrap().state(),
        SessionState::Active
    );

    // Reports relay to the SBP client through the table-owned proxy.
    let report = SensingMeasurementReport {
        setup_id,
        instance_id: MeasurementInstanceId::new(0),
        payload: payload(2.0),
    };
    let actions = table
        .handle_for(setup_id, SessionEvent::ReportReceived(report.clone()))
        .unwrap();
    assert!(actions.contains(&Action::RelaySbpReport(report)));

    // Capacity guard mirrors the direct-setup path.
    let mut cfg = SessionConfig::default();
    cfg.capabilities.max_active_setups = 1;
    let mut small = SessionTable::new(cfg);
    small.handle_sbp_request(sbp_request(1)).unwrap();
    let actions = small.handle_sbp_request(sbp_request(2)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedCapacity,
            ..
        })]
    ));

    // Unknown-setup drops are observable, not silent (finding 3).
    assert_eq!(table.unknown_setup_drops(), 0);
    let actions = table
        .handle_for(MeasurementSetupId::new(99).unwrap(), SessionEvent::Timeout)
        .unwrap();
    assert!(actions.is_empty());
    assert_eq!(table.unknown_setup_drops(), 1);
}

#[test]
fn sbp_validation_shares_setup_chain_with_one_to_one_status_mapping() {
    // Review finding 5: SBP requests are validated by building the proxied
    // setup request first and running it through the single evaluate_setup
    // chain — statuses map 1:1, so no rejection class is folded away and no
    // setup policy can be bypassed via SBP.

    // Incompatible profile now surfaces as its own status (the old
    // duplicated SBP chain folded it into RejectedUnsupportedParams).
    let mut cfg = SessionConfig::default();
    cfg.profile = SpecProfile::VendorExtension("acme".into());
    let mut proxy = SensingSession::new_responder(cfg);
    let actions = proxy
        .handle(SessionEvent::SbpRequestReceived(sbp_request(41)))
        .unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedIncompatibleProfile,
            ..
        })]
    ));

    // Consent policy rejection passes through unchanged.
    let mut proxy = SensingSession::new_responder(SessionConfig::default());
    let mut sbp = sbp_request(42);
    sbp.params.consent = ConsentMode::Disabled;
    let actions = proxy.handle(SessionEvent::SbpRequestReceived(sbp)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedByPolicy,
            ..
        })]
    ));

    // Capability rejection (bandwidth beyond the advertised maximum).
    let mut cfg = SessionConfig::default();
    cfg.capabilities.max_bandwidth_mhz = 40;
    let mut proxy = SensingSession::new_responder(cfg);
    let mut sbp = sbp_request(43);
    sbp.params.bandwidth = Bandwidth::Bw80;
    let actions = proxy.handle(SessionEvent::SbpRequestReceived(sbp)).unwrap();
    assert!(matches!(
        actions[..],
        [Action::SendSbpResponse(SbpResponse {
            status: SbpStatus::RejectedUnsupportedParams,
            ..
        })]
    ));

    // The status translation itself is exhaustive and 1:1.
    let pairs = [
        (SetupStatus::Accepted, SbpStatus::Accepted),
        (
            SetupStatus::RejectedNotSupported,
            SbpStatus::RejectedNotSupported,
        ),
        (
            SetupStatus::RejectedUnsupportedParams,
            SbpStatus::RejectedUnsupportedParams,
        ),
        (
            SetupStatus::RejectedSetupIdCollision,
            SbpStatus::RejectedSetupIdCollision,
        ),
        (
            SetupStatus::RejectedIncompatibleProfile,
            SbpStatus::RejectedIncompatibleProfile,
        ),
        (SetupStatus::RejectedByPolicy, SbpStatus::RejectedByPolicy),
        (SetupStatus::RejectedCapacity, SbpStatus::RejectedCapacity),
    ];
    for (setup, sbp) in pairs {
        assert_eq!(SbpStatus::from(setup), sbp);
    }
}
