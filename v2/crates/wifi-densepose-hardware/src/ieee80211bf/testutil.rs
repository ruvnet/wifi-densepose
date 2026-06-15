//! Shared helpers for the ADR-153 acceptance tests (hardware-free).

use chrono::Utc;

use super::messages::{CsiReportPayload, SensingMeasurementSetupRequest};
use super::session::{Action, SensingSession, SessionEvent};
use super::transport::{action_to_frame, frame_to_event, SensingTransport, SimTransport};
use super::types::{
    ConsentMode, MeasurementSetupId, MeasurementSetupParams, ReportingConfig, SpecProfile,
    TransceiverRole,
};
use crate::csi_frame::{
    Adr018Flags, AntennaConfig, Bandwidth, CsiFrame, CsiMetadata, PpduType, SubcarrierData,
};

pub(super) fn params() -> MeasurementSetupParams {
    MeasurementSetupParams {
        bandwidth: Bandwidth::Bw20,
        period_ms: 100,
        burst_instances: 4,
        reporting: ReportingConfig::EveryInstance,
        initiator_role: TransceiverRole::Transmitter,
        responder_role: TransceiverRole::Receiver,
        consent: ConsentMode::ExplicitConsent,
    }
}

pub(super) fn setup_request(id: u8) -> SensingMeasurementSetupRequest {
    SensingMeasurementSetupRequest {
        profile: SpecProfile::Ieee80211Bf2025,
        setup_id: MeasurementSetupId::new(id).unwrap(),
        params: params(),
    }
}

pub(super) fn payload(mean: f32) -> CsiReportPayload {
    CsiReportPayload {
        n_subcarriers: 4,
        amplitudes: vec![mean; 4],
        phases: vec![0.25; 4],
    }
}

pub(super) fn csi_frame(n: usize, i: i16, q: i16) -> CsiFrame {
    CsiFrame {
        metadata: CsiMetadata {
            timestamp: Utc::now(),
            node_id: 1,
            n_antennas: 1,
            n_subcarriers: n as u16,
            channel_freq_mhz: 2437,
            rssi_dbm: -50,
            noise_floor_dbm: -95,
            bandwidth: Bandwidth::Bw20,
            antenna_config: AntennaConfig::default(),
            sequence: 0,
            ppdu_type: PpduType::HtLegacy,
            adr018_flags: Adr018Flags::default(),
        },
        subcarriers: (0..n)
            .map(|k| SubcarrierData {
                i,
                q,
                index: k as i16,
            })
            .collect(),
    }
}

/// Drive a session, forwarding wire-bound actions onto a transport.
pub(super) fn dispatch(
    s: &mut SensingSession,
    event: SessionEvent,
    out: &mut SimTransport,
) -> Vec<Action> {
    let actions = s.handle(event).expect("handle must not error");
    for a in &actions {
        if let Some(f) = action_to_frame(a) {
            out.send_frame(f).expect("send must not error");
        }
    }
    actions
}

pub(super) fn ferry(from: &mut SimTransport, to: &mut SimTransport) {
    for f in from.drain_sent() {
        to.push_inbound(f);
    }
}

/// Consume inbound frames on `wire`, sending any resulting outbound frames
/// back onto the same transport's sent log.
pub(super) fn pump(s: &mut SensingSession, wire: &mut SimTransport) -> Vec<Action> {
    let mut all = Vec::new();
    while let Some(frame) = wire.poll_frame() {
        if let Some(event) = frame_to_event(frame) {
            all.extend(dispatch(s, event, wire));
        }
    }
    all
}
