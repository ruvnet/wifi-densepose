//! Error types for hardware parsing.

use thiserror::Error;

/// Errors that can occur when parsing CSI data from hardware.
#[derive(Debug, Error)]
pub enum ParseError {
    /// Not enough bytes in the buffer to parse a complete frame.
    #[error("Insufficient data: need {needed} bytes, got {got}")]
    InsufficientData { needed: usize, got: usize },

    /// The frame header magic bytes don't match expected values.
    #[error("Invalid magic: expected {expected:#06x}, got {got:#06x}")]
    InvalidMagic { expected: u32, got: u32 },

    /// A recognized RuView wire packet was received that is *not* an
    /// ADR-018 raw CSI frame (e.g. ADR-039 vitals, ADR-081 feature state,
    /// ADR-095 temporal classification). The firmware multiplexes several
    /// packet types onto the same UDP port, so a CSI parser will see these
    /// interleaved with CSI frames — that is expected, not a corruption.
    /// Consumers should route the packet to the matching decoder or skip it.
    #[error("Non-CSI RuView packet on CSI socket: {kind} (magic {magic:#010x})")]
    NonCsiPacket { magic: u32, kind: &'static str },

    /// The frame indicates more subcarriers than physically possible.
    #[error("Invalid subcarrier count: {count} (max {max})")]
    InvalidSubcarrierCount { count: usize, max: usize },

    /// The I/Q data buffer length doesn't match expected size.
    #[error("I/Q data length mismatch: expected {expected}, got {got}")]
    IqLengthMismatch { expected: usize, got: usize },

    /// RSSI value is outside the valid range.
    #[error("Invalid RSSI value: {value} dBm (expected -100..0)")]
    InvalidRssi { value: i32 },

    /// Invalid antenna count (must be 1-4 for ESP32).
    #[error("Invalid antenna count: {count} (expected 1-4)")]
    InvalidAntennaCount { count: u8 },

    /// Generic byte-level parse error.
    #[error("Parse error at offset {offset}: {message}")]
    ByteError { offset: usize, message: String },
}
