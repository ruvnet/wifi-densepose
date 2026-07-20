//! Session FSM I/O types for the 802.11bf sensing model: events in
//! ([`SessionEvent`]), actions out ([`Action`]), close reasons, static
//! configuration, and the state enum.
//!
//! Split from [`super::session`] to keep each file under the ADR-153
//! 500-line maintainability cap; the canonical public path re-exports
//! these from [`super::session`].

use super::messages::{
    CsiReportPayload, SbpRequest, SbpResponse, SbpStatus, SensingMeasurementInstance,
    SensingMeasurementReport, SensingMeasurementSetupRequest, SensingMeasurementSetupResponse,
    SensingSessionTermination, TerminationReason,
};
use super::types::{MeasurementInstanceId, SensingCapabilities, SetupStatus, SpecProfile};

/// Session FSM states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    SetupNegotiating,
    Active,
    Terminating,
}

/// Inputs to the session FSM. `Start*` are local commands; `*Received` are
/// frames from the peer; `Timeout`/`InstanceElapsed` are scheduler ticks.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    /// Local command (initiator): begin setup negotiation.
    StartSetup(SensingMeasurementSetupRequest),
    /// Local command (initiator): request sensing-by-proxy from an AP.
    StartSbp(SbpRequest),
    SetupRequestReceived(SensingMeasurementSetupRequest),
    SetupResponseReceived(SensingMeasurementSetupResponse),
    SbpRequestReceived(SbpRequest),
    SbpResponseReceived(SbpResponse),
    /// Scheduler tick: the negotiated periodicity elapsed (the
    /// measurement-driving endpoint — initiator or SBP proxy — emits the
    /// next measurement-instance trigger).
    InstanceElapsed,
    /// A sensing receiver captured a measurement for an instance (payload is
    /// fed by the transport/bridge — see `OpportunisticCsiBridge`).
    MeasurementCaptured {
        instance_id: MeasurementInstanceId,
        payload: CsiReportPayload,
    },
    ReportReceived(SensingMeasurementReport),
    /// Generic timeout tick for the current state.
    Timeout,
    /// Local command: terminate the session.
    Terminate(TerminationReason),
    TerminationReceived(SensingSessionTermination),
}

/// Outputs of the session FSM. `Send*`/`TriggerInstance`/`RelaySbpReport`
/// go to the transport; `DeliverReport`/`SessionClosed` go to the local
/// consumer.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    SendSetupRequest(SensingMeasurementSetupRequest),
    SendSetupResponse(SensingMeasurementSetupResponse),
    SendSbpRequest(SbpRequest),
    SendSbpResponse(SbpResponse),
    TriggerInstance(SensingMeasurementInstance),
    SendReport(SensingMeasurementReport),
    DeliverReport(SensingMeasurementReport),
    /// SBP proxy mode: forward a report received from the sensing responder
    /// to the SBP client. The transport maps this to a frame toward the
    /// client (`SensingFrame::SbpReport`), distinct from `SendReport`,
    /// which travels toward the sensing initiator.
    RelaySbpReport(SensingMeasurementReport),
    SendTermination(SensingSessionTermination),
    SessionClosed(CloseReason),
}

/// Why a session returned to Idle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloseReason {
    SetupRejected(SetupStatus),
    SbpRejected(SbpStatus),
    Terminated(TerminationReason),
    /// Terminating-state quiescence completed (no peer echo required).
    Completed,
}

/// Static configuration for a sensing session.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionConfig {
    /// Spec profile this endpoint advertises/accepts.
    pub profile: SpecProfile,
    /// Capability set used to evaluate inbound setups.
    pub capabilities: SensingCapabilities,
    /// Consecutive negotiation timeouts before aborting to Idle.
    pub max_setup_timeouts: u8,
    /// Consecutive missed instances (Active timeouts) before terminating.
    pub max_missed_instances: u8,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            profile: SpecProfile::Ieee80211Bf2025,
            capabilities: SensingCapabilities::sim_full(),
            max_setup_timeouts: 3,
            max_missed_instances: 5,
        }
    }
}
