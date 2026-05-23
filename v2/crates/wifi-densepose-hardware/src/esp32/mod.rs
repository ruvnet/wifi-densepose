//! ESP32 hardware protocol modules.
//!
//! Implements sensing-first RF protocols for ESP32-S3 mesh nodes,
//! including TDM (Time-Division Multiplexed) sensing schedules
//! per ADR-029 (RuvSense) and ADR-031 (RuView).
//!
//! ## Security (ADR-032 / ADR-032a)
//!
//! - `quic_transport` -- QUIC-based authenticated transport for aggregator nodes
//! - `secure_tdm` -- Secured TDM protocol with dual-mode (QUIC / manual crypto)

pub mod quic_transport;
pub mod secure_tdm;
pub mod tdm;

pub use tdm::{SyncBeacon, TdmCoordinator, TdmError, TdmSchedule, TdmSlot, TdmSlotCompleted};

pub use quic_transport::{
    ConnectionState, FramedMessage, MessageType, QuicTransportConfig, QuicTransportError,
    QuicTransportHandle, SecurityMode, TransportStats, STREAM_BEACON, STREAM_CONTROL, STREAM_CSI,
};

pub use secure_tdm::{
    AuthenticatedBeacon, ReplayWindow, SecLevel, SecureCycleOutput, SecureTdmConfig,
    SecureTdmCoordinator, SecureTdmError, AUTHENTICATED_BEACON_SIZE,
};
