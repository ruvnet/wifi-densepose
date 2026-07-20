//! Transport abstraction for the 802.11bf forward-compatibility model.
//!
//! [`SensingTransport`] is the seam where a real chipset binding will land
//! when commodity silicon implements IEEE 802.11bf-2025 (none does today —
//! ADR-152 F4, ADR-153). Until then:
//!
//! - [`SimTransport`] is a scriptable in-memory test double for protocol
//!   tests in CI (no hardware).
//! - [`OpportunisticCsiBridge`] maps today's opportunistic ESP32 CSI
//!   extraction (ADR-018 frames parsed by [`crate::Esp32CsiParser`] and
//!   delivered by [`crate::aggregator::Esp32Aggregator`]) onto the
//!   standardized report path: one measurement instance ≈ one batch of
//!   [`CsiFrame`]s.
//!
//! **Replaceability benchmark (ADR-153):** consumers must depend only on
//! `SensingTransport` plus the report types in [`super::types`] — a future
//! chipset adapter replaces `OpportunisticCsiBridge` without touching them.

use std::collections::VecDeque;

use thiserror::Error;

use super::messages::{
    CsiReportPayload, SbpRequest, SbpResponse, SensingMeasurementInstance,
    SensingMeasurementReport, SensingMeasurementSetupRequest, SensingMeasurementSetupResponse,
    SensingSessionTermination,
};
use super::session::Action;
use super::types::{BfError, MeasurementInstanceId, MeasurementSetupId, MAX_REPORT_SUBCARRIERS};
use crate::csi_frame::CsiFrame;

/// Frames exchanged between sensing endpoints. This is a *logical* frame
/// set — no OTA encoding is defined until silicon exists to bind to.
#[derive(Debug, Clone, PartialEq)]
pub enum SensingFrame {
    SetupRequest(SensingMeasurementSetupRequest),
    SetupResponse(SensingMeasurementSetupResponse),
    InstanceTrigger(SensingMeasurementInstance),
    Report(SensingMeasurementReport),
    SbpRequest(SbpRequest),
    SbpResponse(SbpResponse),
    /// Proxied measurement report forwarded by an SBP proxy toward its SBP
    /// client ([`Action::RelaySbpReport`]) — distinct from [`Self::Report`],
    /// which travels toward the sensing initiator.
    SbpReport(SensingMeasurementReport),
    Termination(SensingSessionTermination),
}

/// Errors surfaced by a sensing transport.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TransportError {
    #[error("transport link down")]
    LinkDown,
    #[error("transport queue full (capacity {capacity})")]
    QueueFull { capacity: usize },
}

/// Frame-exchange abstraction for sensing endpoints.
///
/// The required surface is deliberately tiny (`send_frame`/`poll_frame`);
/// the named helpers are convenience wrappers so call sites read like the
/// standard's procedures.
pub trait SensingTransport {
    /// Queue one logical frame toward the peer.
    fn send_frame(&mut self, frame: SensingFrame) -> Result<(), TransportError>;

    /// Pop the next inbound frame, if any.
    fn poll_frame(&mut self) -> Option<SensingFrame>;

    fn send_setup_request(
        &mut self,
        req: SensingMeasurementSetupRequest,
    ) -> Result<(), TransportError> {
        self.send_frame(SensingFrame::SetupRequest(req))
    }

    fn send_setup_response(
        &mut self,
        resp: SensingMeasurementSetupResponse,
    ) -> Result<(), TransportError> {
        self.send_frame(SensingFrame::SetupResponse(resp))
    }

    fn trigger_measurement_instance(
        &mut self,
        instance: SensingMeasurementInstance,
    ) -> Result<(), TransportError> {
        self.send_frame(SensingFrame::InstanceTrigger(instance))
    }

    fn send_report(&mut self, report: SensingMeasurementReport) -> Result<(), TransportError> {
        self.send_frame(SensingFrame::Report(report))
    }

    fn send_termination(
        &mut self,
        termination: SensingSessionTermination,
    ) -> Result<(), TransportError> {
        self.send_frame(SensingFrame::Termination(termination))
    }
}

/// Map a session [`Action`] to the frame it puts on the wire, if any.
/// `DeliverReport`/`SessionClosed` are local-consumer actions and map to `None`.
pub fn action_to_frame(action: &Action) -> Option<SensingFrame> {
    match action {
        Action::SendSetupRequest(req) => Some(SensingFrame::SetupRequest(req.clone())),
        Action::SendSetupResponse(resp) => Some(SensingFrame::SetupResponse(*resp)),
        Action::SendSbpRequest(req) => Some(SensingFrame::SbpRequest(req.clone())),
        Action::SendSbpResponse(resp) => Some(SensingFrame::SbpResponse(*resp)),
        Action::TriggerInstance(instance) => Some(SensingFrame::InstanceTrigger(*instance)),
        Action::SendReport(report) => Some(SensingFrame::Report(report.clone())),
        Action::RelaySbpReport(report) => Some(SensingFrame::SbpReport(report.clone())),
        Action::SendTermination(term) => Some(SensingFrame::Termination(*term)),
        Action::DeliverReport(_) | Action::SessionClosed(_) => None,
    }
}

/// Map an inbound frame to the session event it raises on the receiver.
///
/// `InstanceTrigger` maps to `None`: a sensing receiver pairs the trigger
/// with locally captured CSI and raises `MeasurementCaptured` itself (see
/// [`OpportunisticCsiBridge`]).
pub fn frame_to_event(frame: SensingFrame) -> Option<super::session::SessionEvent> {
    use super::session::SessionEvent as E;
    match frame {
        SensingFrame::SetupRequest(req) => Some(E::SetupRequestReceived(req)),
        SensingFrame::SetupResponse(resp) => Some(E::SetupResponseReceived(resp)),
        SensingFrame::Report(report) => Some(E::ReportReceived(report)),
        // The SBP client consumes proxied reports through the standard
        // report path (its session is in sbp_client mode).
        SensingFrame::SbpReport(report) => Some(E::ReportReceived(report)),
        SensingFrame::SbpRequest(req) => Some(E::SbpRequestReceived(req)),
        SensingFrame::SbpResponse(resp) => Some(E::SbpResponseReceived(resp)),
        SensingFrame::Termination(term) => Some(E::TerminationReceived(term)),
        SensingFrame::InstanceTrigger(_) => None,
    }
}

/// In-memory scriptable transport test double.
///
/// Every successful `send_frame` is recorded in [`SimTransport::sent`]; if a
/// scripted response is queued, it is moved to the inbound queue so the next
/// `poll_frame` returns it — letting tests script a peer without one.
#[derive(Debug, Default)]
pub struct SimTransport {
    sent: Vec<SensingFrame>,
    inbound: VecDeque<SensingFrame>,
    scripted: VecDeque<SensingFrame>,
    link_down: bool,
    capacity: usize,
}

impl SimTransport {
    pub fn new() -> Self {
        Self {
            capacity: 1024,
            ..Default::default()
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            ..Default::default()
        }
    }

    /// Frames sent so far, in order.
    pub fn sent(&self) -> &[SensingFrame] {
        &self.sent
    }

    /// Drain the sent log (useful when ferrying frames between two doubles).
    pub fn drain_sent(&mut self) -> Vec<SensingFrame> {
        std::mem::take(&mut self.sent)
    }

    /// Queue a frame as if the peer transmitted it.
    pub fn push_inbound(&mut self, frame: SensingFrame) {
        self.inbound.push_back(frame);
    }

    /// Script a response: the next successful send moves it to the inbound
    /// queue (one scripted frame consumed per send).
    pub fn script_response(&mut self, frame: SensingFrame) {
        self.scripted.push_back(frame);
    }

    pub fn set_link_down(&mut self, down: bool) {
        self.link_down = down;
    }
}

impl SensingTransport for SimTransport {
    fn send_frame(&mut self, frame: SensingFrame) -> Result<(), TransportError> {
        if self.link_down {
            return Err(TransportError::LinkDown);
        }
        if self.sent.len() >= self.capacity {
            return Err(TransportError::QueueFull {
                capacity: self.capacity,
            });
        }
        self.sent.push(frame);
        if let Some(response) = self.scripted.pop_front() {
            self.inbound.push_back(response);
        }
        Ok(())
    }

    fn poll_frame(&mut self) -> Option<SensingFrame> {
        self.inbound.pop_front()
    }
}

/// Adapter mapping today's opportunistic ESP32 CSI extraction onto the
/// standardized sensing report path.
///
/// A "measurement instance" is approximated by one batch of `batch_size`
/// ADR-018 [`CsiFrame`]s from a node (as produced by
/// [`crate::aggregator::Esp32Aggregator`]'s mpsc channel). Amplitudes are
/// averaged arithmetically; phases via the circular mean (consistent with
/// the RuvSense `phase_align` treatment of LO phase). Invalid frames
/// ([`CsiFrame::is_valid`] false) are skipped; a mid-batch subcarrier-shape
/// change (node reconfiguration) restarts the batch on the new shape.
///
/// This is the *interim backend*: when 802.11bf silicon exists, a chipset
/// adapter producing the same [`SensingMeasurementReport`]s replaces this
/// bridge with no change to consumers (ADR-153 replaceability benchmark).
#[derive(Debug)]
pub struct OpportunisticCsiBridge {
    setup_id: MeasurementSetupId,
    batch_size: usize,
    instance_counter: u32,
    amp_accum: Vec<f64>,
    phase_cos_accum: Vec<f64>,
    phase_sin_accum: Vec<f64>,
    frames_in_batch: usize,
}

impl OpportunisticCsiBridge {
    pub fn new(setup_id: MeasurementSetupId, batch_size: usize) -> Result<Self, BfError> {
        if batch_size == 0 {
            return Err(BfError::InvalidBatchSize { got: 0 });
        }
        Ok(Self {
            setup_id,
            batch_size,
            instance_counter: 0,
            amp_accum: Vec::new(),
            phase_cos_accum: Vec::new(),
            phase_sin_accum: Vec::new(),
            frames_in_batch: 0,
        })
    }

    pub fn setup_id(&self) -> MeasurementSetupId {
        self.setup_id
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Feed one parsed CSI frame; returns a standardized measurement report
    /// when a batch completes. Never panics on malformed frames.
    pub fn ingest(&mut self, frame: &CsiFrame) -> Option<SensingMeasurementReport> {
        if !frame.is_valid() || frame.subcarrier_count() > MAX_REPORT_SUBCARRIERS as usize {
            return None;
        }
        let (amplitudes, phases) = frame.to_amplitude_phase();
        if self.frames_in_batch == 0 || amplitudes.len() != self.amp_accum.len() {
            // Fresh batch (or node reconfigured mid-batch — restart on the
            // new subcarrier shape, dropping the partial batch).
            self.amp_accum = vec![0.0; amplitudes.len()];
            self.phase_cos_accum = vec![0.0; amplitudes.len()];
            self.phase_sin_accum = vec![0.0; amplitudes.len()];
            self.frames_in_batch = 0;
        }
        for (i, (a, p)) in amplitudes.iter().zip(phases.iter()).enumerate() {
            self.amp_accum[i] += a;
            self.phase_cos_accum[i] += p.cos();
            self.phase_sin_accum[i] += p.sin();
        }
        self.frames_in_batch += 1;
        if self.frames_in_batch < self.batch_size {
            return None;
        }

        let scale = self.frames_in_batch as f64;
        // Drop-instead-of-truncate: `as u16` would silently wrap a subcarrier
        // count above 65_535. That count is already gated to
        // `<= MAX_REPORT_SUBCARRIERS` (484) at `ingest`'s entry, so this branch
        // is unreachable in practice — but `try_from().ok()?` makes the
        // construction correct-by-construction rather than relying on the
        // upstream gate, and drops the batch cleanly if the invariant ever
        // changes (ADR-157 §B1, defense-in-depth — not a live bug).
        let n_subcarriers = u16::try_from(self.amp_accum.len()).ok()?;
        let payload = CsiReportPayload {
            n_subcarriers,
            amplitudes: self.amp_accum.iter().map(|a| (a / scale) as f32).collect(),
            phases: self
                .phase_sin_accum
                .iter()
                .zip(self.phase_cos_accum.iter())
                .map(|(s, c)| s.atan2(*c) as f32)
                .collect(),
        };
        self.amp_accum.clear();
        self.phase_cos_accum.clear();
        self.phase_sin_accum.clear();
        self.frames_in_batch = 0;

        let n = self.instance_counter;
        self.instance_counter = self.instance_counter.wrapping_add(1);
        let report = SensingMeasurementReport {
            setup_id: self.setup_id,
            instance_id: MeasurementInstanceId::new((n % 256) as u8),
            payload,
        };
        // Boundary check before handing to consumers; drop instead of panic.
        report.validate().ok()?;
        Some(report)
    }
}
