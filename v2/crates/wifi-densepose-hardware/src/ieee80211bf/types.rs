//! Typed structures for IEEE 802.11bf-2025 WLAN sensing procedures.
//!
//! Sub-7 GHz focus; DMG (>45 GHz) types are stubbed minimally. Concept names
//! follow the standard's procedure vocabulary descriptively — "Sensing
//! Measurement Setup", "Sensing Measurement Instance", "Sensing Measurement
//! Report", "Sensing by Proxy (SBP)", session termination — without claiming
//! clause-level conformance. See [`crate::ieee80211bf`] module docs and
//! ADR-153 for framing; ADR-152 §1.1 F4 for the standards-body evidence.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::csi_frame::Bandwidth;

/// Largest measurement setup identifier accepted by this model (7-bit space;
/// chosen conservatively — the standard encodes the Measurement Setup ID in a
/// compact identifier field).
pub const MAX_SETUP_ID: u8 = 127;
/// Minimum measurement-instance periodicity accepted by this model.
pub const MIN_PERIOD_MS: u32 = 10;
/// Maximum measurement-instance periodicity accepted by this model (1 hour).
pub const MAX_PERIOD_MS: u32 = 3_600_000;
/// Maximum measurement instances per burst accepted by this model.
pub const MAX_BURST_INSTANCES: u8 = 64;
/// Maximum subcarriers in a CSI-variant report payload (matches the 160 MHz
/// usable-subcarrier count, [`Bandwidth::Bw160`]).
pub const MAX_REPORT_SUBCARRIERS: u16 = 484;

/// Errors produced by validation at the protocol-model boundary.
///
/// Adversarial or malformed input must surface as one of these — never a
/// panic (crate rule: input validation at system boundaries).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BfError {
    /// Measurement setup ID outside the accepted identifier space.
    #[error("invalid measurement setup ID {value} (valid 0..={MAX_SETUP_ID})")]
    InvalidSetupId { value: u8 },
    /// Measurement periodicity outside the accepted range.
    #[error("measurement period {period_ms} ms out of range ({MIN_PERIOD_MS}..={MAX_PERIOD_MS})")]
    InvalidPeriod { period_ms: u32 },
    /// Instances-per-burst outside the accepted range.
    #[error("burst instance count {count} out of range (1..={MAX_BURST_INSTANCES})")]
    InvalidBurstInstances { count: u8 },
    /// Threshold-based reporting parameter outside 0..=100 percent.
    #[error("reporting threshold {value}% out of range (0..=100)")]
    InvalidThreshold { value: u8 },
    /// The initiator/responder transceiver roles leave the measurement with
    /// no sensing transmitter or no sensing receiver.
    #[error("transceiver roles leave no sensing transmitter/receiver pair")]
    InvalidTransceiverRoles,
    /// Setup carries [`ConsentMode::Disabled`] — sensing must not start.
    #[error("sensing disabled by consent policy")]
    SensingDisabledByPolicy,
    /// Report payload declares zero subcarriers.
    #[error("report payload empty")]
    EmptyPayload,
    /// Report payload claims more subcarriers than this model supports.
    #[error("report payload claims {count} subcarriers (max {MAX_REPORT_SUBCARRIERS})")]
    PayloadTooLarge { count: u16 },
    /// Declared subcarrier count and vector lengths disagree.
    #[error(
        "report payload length mismatch: declared {declared}, amplitudes {amplitudes}, phases {phases}"
    )]
    PayloadLengthMismatch {
        declared: usize,
        amplitudes: usize,
        phases: usize,
    },
    /// A payload value is NaN/infinite, or an amplitude is negative.
    #[error("report payload value at index {index} is not finite (or negative amplitude)")]
    PayloadValueInvalid { index: usize },
    /// A frame referenced a setup ID that does not match the session.
    #[error("setup ID mismatch: session {expected}, frame {got}")]
    SetupIdMismatch { expected: u8, got: u8 },
    /// Sensing measurement setup negotiation timed out (session resets to Idle).
    #[error("negotiation timed out for setup {setup_id} after {attempts} attempts")]
    NegotiationTimeout { setup_id: u8, attempts: u8 },
    /// A local command (`StartSetup`/`StartSbp`) was issued in a state or
    /// role that cannot accept it.
    #[error("command not valid in state {state}")]
    InvalidStateForCommand { state: &'static str },
    /// CSI bridge batch size must be at least one frame.
    #[error("invalid CSI batch size {got} (must be >= 1)")]
    InvalidBatchSize { got: usize },
}

/// Version gate for every negotiated surface (ADR-153).
///
/// Vendors will expose partial or renamed capabilities before full
/// IEEE 802.11bf-2025 conformance; tagging setups and capability
/// advertisements with a profile keeps that drift explicit.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecProfile {
    /// Pre-publication draft semantics (D-series compatible behavior).
    DraftCompatible,
    /// Published standard semantics (IEEE 802.11bf-2025, published 2025-09-26).
    Ieee80211Bf2025,
    /// Vendor-specific extension or renamed capability set.
    VendorExtension(String),
}

impl SpecProfile {
    /// Whether a peer advertising `self` accepts a setup tagged `requested`.
    ///
    /// Published-standard peers accept draft-compatible requests; vendor
    /// extensions must match exactly.
    pub fn accepts(&self, requested: &SpecProfile) -> bool {
        self == requested
            || matches!(
                (self, requested),
                (SpecProfile::Ieee80211Bf2025, SpecProfile::DraftCompatible)
            )
    }
}

/// Consent/governance mode carried by every sensing measurement setup
/// (ADR-153: sensing is presence inference, not just radio telemetry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConsentMode {
    /// Lab/bench use only; not a deployment consent basis.
    LabOnly,
    /// Sensed persons gave explicit consent.
    ExplicitConsent,
    /// Enterprise-managed policy authorizes sensing.
    ManagedEnterprisePolicy,
    /// Sensing administratively disabled — setups must be rejected.
    Disabled,
}

/// WLAN sensing procedure role: sensing initiator or sensing responder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensingRole {
    Initiator,
    Responder,
}

/// Per-measurement-instance role: sensing transmitter, sensing receiver,
/// or both (a STA may act as either within a measurement instance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransceiverRole {
    Transmitter,
    Receiver,
    TransmitterReceiver,
}

impl TransceiverRole {
    pub fn is_transmitter(self) -> bool {
        matches!(self, Self::Transmitter | Self::TransmitterReceiver)
    }
    pub fn is_receiver(self) -> bool {
        matches!(self, Self::Receiver | Self::TransmitterReceiver)
    }
}

/// Identifier of a sensing measurement setup ("Measurement Setup ID").
///
/// Validated newtype: construction and deserialization both reject values
/// above [`MAX_SETUP_ID`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u8", into = "u8")]
pub struct MeasurementSetupId(u8);

impl MeasurementSetupId {
    pub fn new(value: u8) -> Result<Self, BfError> {
        if value > MAX_SETUP_ID {
            Err(BfError::InvalidSetupId { value })
        } else {
            Ok(Self(value))
        }
    }
    pub fn value(self) -> u8 {
        self.0
    }
}

impl TryFrom<u8> for MeasurementSetupId {
    type Error = BfError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<MeasurementSetupId> for u8 {
    fn from(id: MeasurementSetupId) -> u8 {
        id.0
    }
}

/// Identifier of a sensing measurement instance within a setup
/// ("Measurement Instance ID"). Wraps modulo 256.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MeasurementInstanceId(u8);

impl MeasurementInstanceId {
    pub fn new(value: u8) -> Self {
        Self(value)
    }
    pub fn value(self) -> u8 {
        self.0
    }
    pub fn wrapping_next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

/// Channel width of a bandwidth variant in MHz (capability comparisons).
pub fn bandwidth_mhz(bw: Bandwidth) -> u16 {
    match bw {
        Bandwidth::Bw20 => 20,
        Bandwidth::Bw40 => 40,
        Bandwidth::Bw80 => 80,
        Bandwidth::Bw160 => 160,
    }
}

/// Threshold-based reporting parameters: a report is generated only when the
/// measurement changes by at least `delta_percent` relative to the last
/// reported measurement (normalized-change trigger).
///
/// Deserialization validates through [`ThresholdParams::new`] so the
/// `delta_percent <= 100` invariant holds on every construction path,
/// including untrusted wire/persisted payloads (same convention as
/// [`MeasurementSetupId`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawThresholdParams")]
pub struct ThresholdParams {
    delta_percent: u8,
}

#[derive(Deserialize)]
struct RawThresholdParams {
    delta_percent: u8,
}

impl TryFrom<RawThresholdParams> for ThresholdParams {
    type Error = BfError;

    fn try_from(raw: RawThresholdParams) -> Result<Self, Self::Error> {
        Self::new(raw.delta_percent)
    }
}

impl ThresholdParams {
    pub fn new(delta_percent: u8) -> Result<Self, BfError> {
        if delta_percent > 100 {
            Err(BfError::InvalidThreshold {
                value: delta_percent,
            })
        } else {
            Ok(Self { delta_percent })
        }
    }
    pub fn delta_percent(self) -> u8 {
        self.delta_percent
    }
    /// Whether the change from `previous` to `current` crosses the threshold.
    pub fn exceeds(self, previous: f64, current: f64) -> bool {
        let denom = previous.abs().max(f64::EPSILON);
        ((current - previous).abs() / denom) * 100.0 >= self.delta_percent as f64
    }
}

/// Reporting discipline negotiated in the sensing measurement setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportingConfig {
    /// Report every measurement instance.
    EveryInstance,
    /// Threshold-based reporting (report only on significant change).
    ThresholdBased(ThresholdParams),
}

/// Parameters of a sensing measurement setup ("Sensing Measurement Setup
/// element" parameters, sub-7 GHz). Consent metadata is **required**.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasurementSetupParams {
    /// Sounding bandwidth.
    pub bandwidth: Bandwidth,
    /// Periodicity of measurement instances, in milliseconds.
    pub period_ms: u32,
    /// Measurement instances per burst.
    pub burst_instances: u8,
    /// Reporting discipline (per-instance or threshold-based).
    pub reporting: ReportingConfig,
    /// Transceiver role the initiator takes during measurement instances.
    pub initiator_role: TransceiverRole,
    /// Transceiver role the responder takes during measurement instances.
    pub responder_role: TransceiverRole,
    /// Required governance metadata (ADR-153 privacy requirement).
    pub consent: ConsentMode,
}

impl MeasurementSetupParams {
    /// Boundary validation: range checks plus role/consent coherence.
    pub fn validate(&self) -> Result<(), BfError> {
        if self.period_ms < MIN_PERIOD_MS || self.period_ms > MAX_PERIOD_MS {
            return Err(BfError::InvalidPeriod {
                period_ms: self.period_ms,
            });
        }
        if self.burst_instances == 0 || self.burst_instances > MAX_BURST_INSTANCES {
            return Err(BfError::InvalidBurstInstances {
                count: self.burst_instances,
            });
        }
        let has_tx = self.initiator_role.is_transmitter() || self.responder_role.is_transmitter();
        let has_rx = self.initiator_role.is_receiver() || self.responder_role.is_receiver();
        if !has_tx || !has_rx {
            return Err(BfError::InvalidTransceiverRoles);
        }
        if self.consent == ConsentMode::Disabled {
            return Err(BfError::SensingDisabledByPolicy);
        }
        Ok(())
    }
}

/// Capability advertisement for capability negotiation (ADR-153): no
/// hardcoded ESP32 assumptions in the future-silicon path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensingCapabilities {
    pub sub_7_ghz: bool,
    pub dmg: bool,
    pub edmg: bool,
    pub csi_report: bool,
    pub threshold_reporting: bool,
    pub sensing_by_proxy: bool,
    pub max_bandwidth_mhz: u16,
    pub max_period_ms: u32,
    pub max_active_setups: u16,
}

impl SensingCapabilities {
    /// Permissive capability set for simulation and tests.
    pub fn sim_full() -> Self {
        Self {
            sub_7_ghz: true,
            dmg: false,
            edmg: false,
            csi_report: true,
            threshold_reporting: true,
            sensing_by_proxy: true,
            max_bandwidth_mhz: 160,
            max_period_ms: MAX_PERIOD_MS,
            max_active_setups: 8,
        }
    }

    /// What today's opportunistic ESP32 CSI extraction (ADR-018/ADR-028) can
    /// honor when mapped through [`crate::ieee80211bf::transport::OpportunisticCsiBridge`].
    pub fn esp32_opportunistic() -> Self {
        Self {
            sub_7_ghz: true,
            dmg: false,
            edmg: false,
            csi_report: true,
            threshold_reporting: true,
            sensing_by_proxy: false,
            max_bandwidth_mhz: 40,
            max_period_ms: 60_000,
            max_active_setups: 4,
        }
    }

    /// Evaluate setup parameters against this capability set; `Err` carries
    /// the protocol-level rejection status to return to the peer.
    pub fn evaluate(&self, params: &MeasurementSetupParams) -> Result<(), SetupStatus> {
        if !self.sub_7_ghz || !self.csi_report {
            return Err(SetupStatus::RejectedUnsupportedParams);
        }
        if bandwidth_mhz(params.bandwidth) > self.max_bandwidth_mhz {
            return Err(SetupStatus::RejectedUnsupportedParams);
        }
        if params.period_ms > self.max_period_ms {
            return Err(SetupStatus::RejectedUnsupportedParams);
        }
        if matches!(params.reporting, ReportingConfig::ThresholdBased(_))
            && !self.threshold_reporting
        {
            return Err(SetupStatus::RejectedUnsupportedParams);
        }
        Ok(())
    }
}

/// Status carried by a sensing measurement setup response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetupStatus {
    Accepted,
    /// The receiving endpoint does not act as a sensing responder for this
    /// request — e.g. an initiator-role session received a setup request
    /// (single-role design, see [`crate::ieee80211bf::session`]).
    RejectedNotSupported,
    RejectedUnsupportedParams,
    RejectedSetupIdCollision,
    RejectedIncompatibleProfile,
    RejectedByPolicy,
    RejectedCapacity,
}
