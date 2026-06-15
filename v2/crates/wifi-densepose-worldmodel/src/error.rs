//! Error types for the OccWorld world-model bridge (ADR-147).

use thiserror::Error;

/// All errors that can be returned by the OccWorld bridge.
#[derive(Debug, Error)]
pub enum WorldModelError {
    /// Could not connect to the Unix-domain socket served by the Python
    /// OccWorld inference process.
    #[error("could not connect to OccWorld socket at `{path}`: {source}")]
    SocketConnect {
        /// The socket path that was attempted.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },

    /// A request or response exceeded the 30-second wall-clock deadline.
    #[error("OccWorld inference timed out after {timeout_s}s")]
    Timeout {
        /// The configured timeout in seconds.
        timeout_s: u64,
    },

    /// The JSON payload received from the server could not be decoded, or the
    /// payload we tried to send could not be encoded.
    #[error("JSON (de)serialisation error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    /// The server sent a response that violates the newline-delimited JSON
    /// protocol (e.g. an unexpected EOF before the newline delimiter, or an
    /// oversized frame that exceeded the read buffer limit).
    #[error("protocol error: {0}")]
    Protocol(String),

    /// The OccWorld inference server reported that GPU VRAM is unavailable
    /// (out-of-memory condition on the device side).
    #[error("OccWorld server reports VRAM unavailable: {0}")]
    VramUnavailable(String),
}
