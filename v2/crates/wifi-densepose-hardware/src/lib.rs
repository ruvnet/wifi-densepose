//! WiFi-DensePose hardware interface abstractions.
//!
//! This crate provides platform-agnostic types and parsers for WiFi CSI data
//! from various hardware sources:
//!
//! - **ESP32/ESP32-S3**: Parses ADR-018 binary CSI frames streamed over UDP
//! - **UDP Aggregator**: Receives frames from multiple ESP32 nodes (ADR-018 Layer 2)
//! - **Bridge**: Converts CsiFrame → CsiData for the detection pipeline (ADR-018 Layer 3)
//!
//! # Design Principles
//!
//! 1. **No mock data**: All parsers either parse real bytes or return explicit errors
//! 2. **No hardware dependency at compile time**: Parsing is done on byte buffers,
//!    not through FFI to ESP-IDF or kernel modules
//! 3. **Deterministic**: Same bytes in → same parsed output, always
//!
//! # Example
//!
//! ```rust
//! use wifi_densepose_hardware::{CsiFrame, Esp32CsiParser, ParseError};
//!
//! // Parse ESP32 CSI data from UDP bytes
//! let raw_bytes: &[u8] = &[/* ADR-018 binary frame */];
//! match Esp32CsiParser::parse_frame(raw_bytes) {
//!     Ok((frame, consumed)) => {
//!         println!("Parsed {} subcarriers ({} bytes)", frame.subcarrier_count(), consumed);
//!         let (amplitudes, phases) = frame.to_amplitude_phase();
//!         // Feed into detection pipeline...
//!     }
//!     Err(ParseError::InsufficientData { needed, got }) => {
//!         eprintln!("Need {} bytes, got {}", needed, got);
//!     }
//!     Err(e) => eprintln!("Parse error: {}", e),
//! }
//! ```

pub mod aggregator;
mod bridge;
mod csi_frame;
mod error;
pub mod esp32;
mod esp32_parser;
// ADR-153: IEEE 802.11bf-2025 forward-compatibility protocol model
// (sensing setup / measurement instance / report / SBP / termination).
// Simulation-tested; no commodity silicon implements the standard yet —
// the OpportunisticCsiBridge maps today's ESP32 CSI extraction onto the
// standardized report path until an OTA binding exists.
pub mod ieee80211bf;
pub mod sync_packet;

// ADR-081: Rust mirror of the firmware radio abstraction layer (L1) and
// mesh sensing plane (L3). Lets host tests, simulators, and future
// coordinator-node Rust code drive the controller stack without
// touching any downstream signal/ruvector/train/mat crate.
pub mod radio_ops;

pub use bridge::CsiData;
pub use csi_frame::{
    Adr018Flags, AntennaConfig, Bandwidth, CsiFrame, CsiMetadata, PpduType, SubcarrierData,
};
pub use error::ParseError;
pub use esp32_parser::{
    ruview_sibling_packet_name, Esp32CsiParser, ESP32_CSI_MAGIC, RUVIEW_COMPRESSED_CSI_MAGIC,
    RUVIEW_FEATURE_MAGIC, RUVIEW_FEATURE_STATE_MAGIC, RUVIEW_FUSED_VITALS_MAGIC,
    RUVIEW_TEMPORAL_MAGIC, RUVIEW_VITALS_MAGIC,
};
pub use sync_packet::{
    SyncPacket, SyncPacketFlags, SYNC_PACKET_MAGIC, SYNC_PACKET_SIZE, SYNC_PACKET_PROTO_VER,
};
pub use radio_ops::{
    crc32_ieee, decode_anomaly_alert, decode_mesh, decode_node_status, encode_health, AnomalyAlert,
    AuthClass, CaptureProfile, MeshError, MeshHeader, MeshMsgType, MeshRole, MockRadio, NodeStatus,
    RadioError, RadioHealth, RadioMode, RadioOps, MESH_HEADER_SIZE, MESH_MAGIC, MESH_MAX_PAYLOAD,
    MESH_VERSION,
};
