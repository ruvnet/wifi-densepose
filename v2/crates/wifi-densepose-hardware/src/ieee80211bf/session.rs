//! Sensing session state machine for the 802.11bf forward-compatibility model.
//!
//! Deterministic, event-driven, no async, no clocks: callers inject
//! [`SessionEvent`]s (including `Timeout` ticks) and act on the returned
//! [`Action`]s. State flow (ADR-153):
//!
//! ```text
//! Idle → SetupNegotiating → Active → Terminating → Idle
//! ```
//!
//! Rejection paths: unsupported parameters / incompatible profile / policy
//! (responder responds with a rejected setup status), setup-ID collision
//! ([`super::table::SessionTable`]), and negotiation timeout (typed
//! [`BfError::NegotiationTimeout`] + reset to Idle).
//!
//! **Single-role design:** a session is constructed as initiator or responder
//! and keeps that role for its whole lifetime. An initiator-role session
//! receiving a peer's setup or SBP request answers `RejectedNotSupported`
//! instead of accepting — a peer must never be able to hijack a session out
//! of its configured role. Endpoints that play both roles run one session per
//! role (or a [`super::table::SessionTable`] for the responder side).
//!
//! **SBP proxy mode:** a responder session that accepts an SBP request
//! becomes a first-class proxy ([`SensingSession::is_sbp_proxy`]): it drives
//! the standard initiator path toward the actual sensing responder —
//! including re-triggering measurement instances on
//! [`SessionEvent::InstanceElapsed`] — and relays every received report to
//! the SBP client via [`Action::RelaySbpReport`], in addition to local
//! [`Action::DeliverReport`] delivery.
//!
//! Local `Start*` commands issued outside Idle are caller bugs and surface
//! as typed [`BfError::InvalidStateForCommand`]; genuinely ignorable stray
//! frames/ticks remain silent no-ops. The FSM I/O types live in
//! [`super::events`] and are re-exported here.

use super::messages::{
    SbpRequest, SbpResponse, SbpStatus, SensingMeasurementInstance, SensingMeasurementReport,
    SensingMeasurementSetupRequest, SensingMeasurementSetupResponse, SensingSessionTermination,
    TerminationReason,
};
use super::types::{
    BfError, MeasurementInstanceId, MeasurementSetupId, MeasurementSetupParams, ReportingConfig,
    SensingRole, SetupStatus,
};

pub use super::events::{Action, CloseReason, SessionConfig, SessionEvent, SessionState};

/// One sensing session (one measurement setup) on one endpoint.
#[derive(Debug, Clone)]
pub struct SensingSession {
    role: SensingRole,
    state: SessionState,
    config: SessionConfig,
    /// Last setup request we sent (for negotiation re-sends).
    pending_request: Option<SensingMeasurementSetupRequest>,
    /// Negotiated (or in-negotiation) setup.
    setup: Option<(MeasurementSetupId, MeasurementSetupParams)>,
    /// True when this session awaits proxied sensing (SBP client).
    sbp_client: bool,
    /// True when this responder-role session proxies sensing for an SBP
    /// client: it drives the initiator path toward the sensing responder
    /// and relays received reports back to the client.
    sbp_proxy: bool,
    setup_timeouts: u8,
    missed_instances: u8,
    instance_counter: u32,
    /// Mean amplitude of the last *reported* measurement (threshold trigger).
    last_reported_mean: Option<f64>,
}

impl SensingSession {
    pub fn new_initiator(config: SessionConfig) -> Self {
        Self::new(SensingRole::Initiator, config)
    }

    pub fn new_responder(config: SessionConfig) -> Self {
        Self::new(SensingRole::Responder, config)
    }

    fn new(role: SensingRole, config: SessionConfig) -> Self {
        Self {
            role,
            state: SessionState::Idle,
            config,
            pending_request: None,
            setup: None,
            sbp_client: false,
            sbp_proxy: false,
            setup_timeouts: 0,
            missed_instances: 0,
            instance_counter: 0,
            last_reported_mean: None,
        }
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    pub fn role(&self) -> SensingRole {
        self.role
    }

    /// True when this session is acting as an SBP proxy (accepted via
    /// [`SessionEvent::SbpRequestReceived`]); cleared on reset to Idle.
    pub fn is_sbp_proxy(&self) -> bool {
        self.sbp_proxy
    }

    pub fn setup_id(&self) -> Option<MeasurementSetupId> {
        self.setup.as_ref().map(|(id, _)| *id)
    }

    /// Drive the FSM with one event. Protocol-level rejections surface as
    /// `Ok` actions (responses to the peer); malformed/adversarial input,
    /// out-of-state local commands, and negotiation timeout surface as typed
    /// `Err` (never a panic).
    pub fn handle(&mut self, event: SessionEvent) -> Result<Vec<Action>, BfError> {
        match self.state {
            SessionState::Idle => self.handle_idle(event),
            SessionState::SetupNegotiating => self.handle_negotiating(event),
            SessionState::Active => self.handle_active(event),
            SessionState::Terminating => self.handle_terminating(event),
        }
    }

    fn handle_idle(&mut self, event: SessionEvent) -> Result<Vec<Action>, BfError> {
        match event {
            SessionEvent::StartSetup(req) => {
                if self.role != SensingRole::Initiator {
                    return Err(BfError::InvalidStateForCommand {
                        state: "Idle (responder cannot StartSetup)",
                    });
                }
                req.validate()?;
                self.setup = Some((req.setup_id, req.params.clone()));
                self.pending_request = Some(req.clone());
                self.setup_timeouts = 0;
                self.state = SessionState::SetupNegotiating;
                Ok(vec![Action::SendSetupRequest(req)])
            }
            SessionEvent::StartSbp(sbp) => {
                if self.role != SensingRole::Initiator {
                    return Err(BfError::InvalidStateForCommand {
                        state: "Idle (responder cannot StartSbp)",
                    });
                }
                sbp.validate()?;
                self.setup = Some((sbp.proxy_setup_id, sbp.params.clone()));
                self.sbp_client = true;
                self.setup_timeouts = 0;
                self.state = SessionState::SetupNegotiating;
                Ok(vec![Action::SendSbpRequest(sbp)])
            }
            SessionEvent::SetupRequestReceived(req) => {
                let response = |status| {
                    Action::SendSetupResponse(SensingMeasurementSetupResponse {
                        setup_id: req.setup_id,
                        status,
                    })
                };
                // Single-role design (module docs): an initiator-role
                // session never accepts a peer's setup request — accepting
                // here would let a peer hijack the session into the
                // responder path.
                if self.role != SensingRole::Responder {
                    return Ok(vec![response(SetupStatus::RejectedNotSupported)]);
                }
                match self.evaluate_setup(&req) {
                    SetupStatus::Accepted => {
                        self.setup = Some((req.setup_id, req.params.clone()));
                        self.missed_instances = 0;
                        self.last_reported_mean = None;
                        self.state = SessionState::Active;
                        Ok(vec![response(SetupStatus::Accepted)])
                    }
                    status => Ok(vec![response(status)]),
                }
            }
            SessionEvent::SbpRequestReceived(sbp) => {
                // Single-role design: only responder-role sessions proxy.
                if self.role != SensingRole::Responder {
                    return Ok(vec![Action::SendSbpResponse(SbpResponse {
                        proxy_setup_id: sbp.proxy_setup_id,
                        status: SbpStatus::RejectedNotSupported,
                    })]);
                }
                Ok(self.handle_sbp_request(sbp))
            }
            // Stray frames/ticks in Idle are ignored, not errors.
            _ => Ok(vec![]),
        }
    }

    /// SBP proxy path: accept the request, then run the *standard initiator
    /// path* toward the actual sensing responder. No direct sensor coupling —
    /// the proxied setup is an ordinary `SendSetupRequest` on the transport.
    ///
    /// Validation is the single [`Self::evaluate_setup`] chain: the proxied
    /// setup request is built first and evaluated exactly as a direct setup
    /// would be, with the resulting [`SetupStatus`] mapped 1:1 onto
    /// [`SbpStatus`] — no SBP-only re-implementation that could drift from
    /// (or bypass) the setup policy.
    fn handle_sbp_request(&mut self, sbp: SbpRequest) -> Vec<Action> {
        let respond = |status| {
            Action::SendSbpResponse(SbpResponse {
                proxy_setup_id: sbp.proxy_setup_id,
                status,
            })
        };
        // SBP-specific capability gate; everything else is the setup chain.
        if !self.config.capabilities.sensing_by_proxy {
            return vec![respond(SbpStatus::RejectedNotSupported)];
        }
        let req = SensingMeasurementSetupRequest {
            profile: sbp.profile.clone(),
            setup_id: sbp.proxy_setup_id,
            params: sbp.params.clone(),
        };
        match self.evaluate_setup(&req) {
            SetupStatus::Accepted => {}
            status => return vec![respond(SbpStatus::from(status))],
        }
        self.setup = Some((req.setup_id, req.params.clone()));
        self.pending_request = Some(req.clone());
        self.sbp_proxy = true;
        self.setup_timeouts = 0;
        self.state = SessionState::SetupNegotiating;
        vec![respond(SbpStatus::Accepted), Action::SendSetupRequest(req)]
    }

    fn evaluate_setup(&self, req: &SensingMeasurementSetupRequest) -> SetupStatus {
        if !self.config.profile.accepts(&req.profile) {
            return SetupStatus::RejectedIncompatibleProfile;
        }
        match req.validate() {
            Err(BfError::SensingDisabledByPolicy) => return SetupStatus::RejectedByPolicy,
            Err(_) => return SetupStatus::RejectedUnsupportedParams,
            Ok(()) => {}
        }
        match self.config.capabilities.evaluate(&req.params) {
            Err(status) => status,
            Ok(()) => SetupStatus::Accepted,
        }
    }

    fn handle_negotiating(&mut self, event: SessionEvent) -> Result<Vec<Action>, BfError> {
        match event {
            SessionEvent::SetupResponseReceived(resp) => {
                let expected = match self.setup_id() {
                    Some(id) => id,
                    None => return Ok(vec![]),
                };
                if resp.setup_id != expected {
                    return Err(BfError::SetupIdMismatch {
                        expected: expected.value(),
                        got: resp.setup_id.value(),
                    });
                }
                match resp.status {
                    SetupStatus::Accepted => {
                        self.setup_timeouts = 0;
                        self.missed_instances = 0;
                        self.state = SessionState::Active;
                        match self.next_instance_record() {
                            Some(instance) => Ok(vec![Action::TriggerInstance(instance)]),
                            None => Ok(vec![]),
                        }
                    }
                    status => {
                        self.reset();
                        Ok(vec![Action::SessionClosed(CloseReason::SetupRejected(
                            status,
                        ))])
                    }
                }
            }
            SessionEvent::SbpResponseReceived(resp) if self.sbp_client => {
                let expected = match self.setup_id() {
                    Some(id) => id,
                    None => return Ok(vec![]),
                };
                if resp.proxy_setup_id != expected {
                    return Err(BfError::SetupIdMismatch {
                        expected: expected.value(),
                        got: resp.proxy_setup_id.value(),
                    });
                }
                match resp.status {
                    SbpStatus::Accepted => {
                        // Proxied reports will arrive via ReportReceived.
                        self.setup_timeouts = 0;
                        self.state = SessionState::Active;
                        Ok(vec![])
                    }
                    status => {
                        self.reset();
                        Ok(vec![Action::SessionClosed(CloseReason::SbpRejected(
                            status,
                        ))])
                    }
                }
            }
            SessionEvent::Timeout => {
                self.setup_timeouts = self.setup_timeouts.saturating_add(1);
                if self.setup_timeouts >= self.config.max_setup_timeouts {
                    let setup_id = self.setup_id().map(|id| id.value()).unwrap_or(0);
                    let attempts = self.setup_timeouts;
                    self.reset();
                    Err(BfError::NegotiationTimeout { setup_id, attempts })
                } else if let Some(req) = &self.pending_request {
                    Ok(vec![Action::SendSetupRequest(req.clone())])
                } else {
                    Ok(vec![])
                }
            }
            SessionEvent::Terminate(reason) => {
                self.reset();
                Ok(vec![Action::SessionClosed(CloseReason::Terminated(reason))])
            }
            SessionEvent::TerminationReceived(term) => {
                self.reset();
                Ok(vec![Action::SessionClosed(CloseReason::Terminated(
                    term.reason,
                ))])
            }
            // Local Start* outside Idle is a caller bug — typed error.
            SessionEvent::StartSetup(_) | SessionEvent::StartSbp(_) => {
                Err(BfError::InvalidStateForCommand {
                    state: "SetupNegotiating",
                })
            }
            // Genuinely ignorable stray frames/ticks are no-ops.
            _ => Ok(vec![]),
        }
    }

    fn handle_active(&mut self, event: SessionEvent) -> Result<Vec<Action>, BfError> {
        match event {
            SessionEvent::InstanceElapsed => {
                // The measurement-driving endpoint re-triggers here: the
                // initiator, or an SBP proxy running the initiator path
                // toward the sensing responder. SBP *clients* only consume
                // proxied reports and never trigger instances.
                let drives_instances =
                    (self.role == SensingRole::Initiator || self.sbp_proxy) && !self.sbp_client;
                if drives_instances {
                    match self.next_instance_record() {
                        Some(instance) => Ok(vec![Action::TriggerInstance(instance)]),
                        None => Ok(vec![]),
                    }
                } else {
                    Ok(vec![])
                }
            }
            SessionEvent::MeasurementCaptured {
                instance_id,
                payload,
            } => {
                payload.validate()?;
                let (setup_id, params) = match &self.setup {
                    Some((id, p)) => (*id, p.clone()),
                    None => return Ok(vec![]),
                };
                // A successful capture means this instance was not missed —
                // the missed-instance budget counts *consecutive* misses,
                // so it resets here even when threshold-based reporting
                // suppresses the report below.
                self.missed_instances = 0;
                let mean = payload.mean_amplitude();
                let should_report = match params.reporting {
                    ReportingConfig::EveryInstance => true,
                    ReportingConfig::ThresholdBased(threshold) => match self.last_reported_mean {
                        None => true,
                        Some(previous) => threshold.exceeds(previous, mean),
                    },
                };
                if !should_report {
                    return Ok(vec![]);
                }
                self.last_reported_mean = Some(mean);
                Ok(vec![Action::SendReport(SensingMeasurementReport {
                    setup_id,
                    instance_id,
                    payload,
                })])
            }
            SessionEvent::ReportReceived(report) => {
                report.validate()?;
                let expected = match self.setup_id() {
                    Some(id) => id,
                    None => return Ok(vec![]),
                };
                if report.setup_id != expected {
                    return Err(BfError::SetupIdMismatch {
                        expected: expected.value(),
                        got: report.setup_id.value(),
                    });
                }
                self.missed_instances = 0;
                if self.sbp_proxy {
                    // Proxy mode: deliver to the local consumer *and* relay
                    // toward the SBP client on the transport.
                    Ok(vec![
                        Action::DeliverReport(report.clone()),
                        Action::RelaySbpReport(report),
                    ])
                } else {
                    Ok(vec![Action::DeliverReport(report)])
                }
            }
            SessionEvent::Timeout => {
                self.missed_instances = self.missed_instances.saturating_add(1);
                if self.missed_instances >= self.config.max_missed_instances {
                    self.state = SessionState::Terminating;
                    Ok(self.termination_actions(TerminationReason::Timeout))
                } else {
                    Ok(vec![])
                }
            }
            SessionEvent::Terminate(reason) => {
                self.state = SessionState::Terminating;
                Ok(self.termination_actions(reason))
            }
            SessionEvent::TerminationReceived(term) => {
                self.reset();
                Ok(vec![Action::SessionClosed(CloseReason::Terminated(
                    term.reason,
                ))])
            }
            // Local Start* outside Idle is a caller bug — typed error.
            SessionEvent::StartSetup(_) | SessionEvent::StartSbp(_) => {
                Err(BfError::InvalidStateForCommand { state: "Active" })
            }
            // Genuinely ignorable stray frames (duplicate setup/SBP traffic)
            // are no-ops.
            _ => Ok(vec![]),
        }
    }

    fn handle_terminating(&mut self, event: SessionEvent) -> Result<Vec<Action>, BfError> {
        match event {
            SessionEvent::TerminationReceived(term) => {
                self.reset();
                Ok(vec![Action::SessionClosed(CloseReason::Terminated(
                    term.reason,
                ))])
            }
            // No peer echo is required: a quiescence tick completes teardown.
            SessionEvent::Timeout => {
                self.reset();
                Ok(vec![Action::SessionClosed(CloseReason::Completed)])
            }
            // Local Start* outside Idle is a caller bug — typed error.
            SessionEvent::StartSetup(_) | SessionEvent::StartSbp(_) => {
                Err(BfError::InvalidStateForCommand {
                    state: "Terminating",
                })
            }
            _ => Ok(vec![]),
        }
    }

    fn termination_actions(&self, reason: TerminationReason) -> Vec<Action> {
        match self.setup_id() {
            Some(setup_id) => vec![Action::SendTermination(SensingSessionTermination {
                setup_id,
                reason,
            })],
            None => vec![],
        }
    }

    fn next_instance_record(&mut self) -> Option<SensingMeasurementInstance> {
        let (setup_id, params) = match &self.setup {
            Some((id, p)) => (*id, p.clone()),
            None => return None,
        };
        let n = self.instance_counter;
        self.instance_counter = self.instance_counter.wrapping_add(1);
        Some(SensingMeasurementInstance {
            setup_id,
            instance_id: MeasurementInstanceId::new((n % 256) as u8),
            timestamp_us: u64::from(n) * u64::from(params.period_ms) * 1_000,
        })
    }

    fn reset(&mut self) {
        self.state = SessionState::Idle;
        self.pending_request = None;
        self.setup = None;
        self.sbp_client = false;
        self.sbp_proxy = false;
        self.setup_timeouts = 0;
        self.missed_instances = 0;
        self.instance_counter = 0;
        self.last_reported_mean = None;
    }
}
