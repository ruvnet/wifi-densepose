//! Procedure message types for the 802.11bf sensing model: measurement
//! setup request/response, measurement instance, CSI-variant measurement
//! report, sensing-by-proxy (SBP) exchange, session termination, and the
//! minimal DMG (>45 GHz) stubs. Negotiation-core types (identifiers,
//! parameters, capabilities, statuses) live in [`super::types`].

use serde::{Deserialize, Serialize};

use super::types::{
    BfError, MeasurementInstanceId, MeasurementSetupId, MeasurementSetupParams, SetupStatus,
    SpecProfile, MAX_REPORT_SUBCARRIERS,
};

/// Sensing measurement setup request (initiator → responder).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensingMeasurementSetupRequest {
    /// Version gate for the negotiated surface.
    pub profile: SpecProfile,
    pub setup_id: MeasurementSetupId,
    pub params: MeasurementSetupParams,
}

impl SensingMeasurementSetupRequest {
    pub fn validate(&self) -> Result<(), BfError> {
        self.params.validate()
    }
}

/// Sensing measurement setup response (responder → initiator).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensingMeasurementSetupResponse {
    pub setup_id: MeasurementSetupId,
    pub status: SetupStatus,
}

/// One scheduled sensing measurement instance within an active setup.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensingMeasurementInstance {
    pub setup_id: MeasurementSetupId,
    pub instance_id: MeasurementInstanceId,
    /// Deterministic schedule offset of this instance (µs since setup
    /// activation; synthesized from the negotiated periodicity).
    pub timestamp_us: u64,
}

/// CSI-variant sensing measurement report payload (amplitude/phase per
/// usable subcarrier, averaged over the measurement instance).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsiReportPayload {
    pub n_subcarriers: u16,
    pub amplitudes: Vec<f32>,
    pub phases: Vec<f32>,
}

impl CsiReportPayload {
    /// Boundary validation: shape coherence and value sanity. Rejects NaN,
    /// infinities, and negative amplitudes from adversarial peers.
    pub fn validate(&self) -> Result<(), BfError> {
        if self.n_subcarriers == 0 {
            return Err(BfError::EmptyPayload);
        }
        if self.n_subcarriers > MAX_REPORT_SUBCARRIERS {
            return Err(BfError::PayloadTooLarge {
                count: self.n_subcarriers,
            });
        }
        let declared = self.n_subcarriers as usize;
        if self.amplitudes.len() != declared || self.phases.len() != declared {
            return Err(BfError::PayloadLengthMismatch {
                declared,
                amplitudes: self.amplitudes.len(),
                phases: self.phases.len(),
            });
        }
        for (index, a) in self.amplitudes.iter().enumerate() {
            if !a.is_finite() || *a < 0.0 {
                return Err(BfError::PayloadValueInvalid { index });
            }
        }
        for (index, p) in self.phases.iter().enumerate() {
            if !p.is_finite() {
                return Err(BfError::PayloadValueInvalid { index });
            }
        }
        Ok(())
    }

    /// Mean amplitude across subcarriers (threshold-trigger metric).
    pub fn mean_amplitude(&self) -> f64 {
        if self.amplitudes.is_empty() {
            return 0.0;
        }
        self.amplitudes.iter().map(|a| *a as f64).sum::<f64>() / self.amplitudes.len() as f64
    }
}

/// Sensing measurement report (sensing receiver → initiator).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensingMeasurementReport {
    pub setup_id: MeasurementSetupId,
    pub instance_id: MeasurementInstanceId,
    pub payload: CsiReportPayload,
}

impl SensingMeasurementReport {
    pub fn validate(&self) -> Result<(), BfError> {
        self.payload.validate()
    }
}

/// Sensing-by-Proxy (SBP) request: a non-AP STA asks an AP to act as sensing
/// initiator on its behalf and forward the resulting reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SbpRequest {
    pub profile: SpecProfile,
    /// Setup ID the proxy uses for the sensing it conducts on our behalf.
    pub proxy_setup_id: MeasurementSetupId,
    pub params: MeasurementSetupParams,
}

impl SbpRequest {
    pub fn validate(&self) -> Result<(), BfError> {
        self.params.validate()
    }
}

/// Status carried by an SBP response.
///
/// Mirrors [`SetupStatus`] 1:1 (see the `From<SetupStatus>` impl): an SBP
/// request is validated through the same chain as a direct setup, so every
/// rejection class must survive the proxy translation.
/// `RejectedNotSupported` additionally covers a proxy that lacks the SBP
/// capability itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SbpStatus {
    Accepted,
    RejectedNotSupported,
    RejectedUnsupportedParams,
    RejectedSetupIdCollision,
    RejectedIncompatibleProfile,
    RejectedByPolicy,
    RejectedCapacity,
}

impl From<SetupStatus> for SbpStatus {
    /// 1:1 mapping from the direct-setup status space, keeping the SBP path
    /// on the single `evaluate_setup` validation chain (no SBP-only policy
    /// drift or bypass).
    fn from(status: SetupStatus) -> Self {
        match status {
            SetupStatus::Accepted => SbpStatus::Accepted,
            SetupStatus::RejectedNotSupported => SbpStatus::RejectedNotSupported,
            SetupStatus::RejectedUnsupportedParams => SbpStatus::RejectedUnsupportedParams,
            SetupStatus::RejectedSetupIdCollision => SbpStatus::RejectedSetupIdCollision,
            SetupStatus::RejectedIncompatibleProfile => SbpStatus::RejectedIncompatibleProfile,
            SetupStatus::RejectedByPolicy => SbpStatus::RejectedByPolicy,
            SetupStatus::RejectedCapacity => SbpStatus::RejectedCapacity,
        }
    }
}

/// Sensing-by-Proxy (SBP) response (proxy AP → requesting STA).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SbpResponse {
    pub proxy_setup_id: MeasurementSetupId,
    pub status: SbpStatus,
}

/// Reason carried by a sensing session termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminationReason {
    InitiatorRequested,
    ResponderRequested,
    Timeout,
    PolicyChange,
}

/// Sensing measurement setup termination (either side may send).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SensingSessionTermination {
    pub setup_id: MeasurementSetupId,
    pub reason: TerminationReason,
}

/// Minimal stub for DMG/EDMG (>45 GHz) sensing types. The standard also
/// covers directional multi-gigabit sensing; this model does not elaborate
/// it beyond a typed placeholder (ADR-153 scope: sub-7 GHz focus).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DmgSensingType {
    Monostatic,
    Bistatic,
    Multistatic,
}

/// Placeholder for a future DMG sensing setup surface.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DmgSensingSetupStub {
    pub setup_id: MeasurementSetupId,
    pub sensing_type: DmgSensingType,
}
