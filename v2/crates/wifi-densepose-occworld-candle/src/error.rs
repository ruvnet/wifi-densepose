//! Error types for `wifi-densepose-occworld-candle`.

/// All errors that can occur during OccWorld inference.
#[derive(Debug, thiserror::Error)]
pub enum OccWorldError {
    /// A Candle operation failed.
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    /// Input or output tensor has an unexpected shape.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// The checkpoint file could not be found or opened.
    #[error("checkpoint not found: {0}")]
    CheckpointNotFound(String),

    /// The checkpoint file exists but could not be parsed.
    #[error("checkpoint parse error: {0}")]
    CheckpointParse(String),

    /// A required tensor key is missing from the checkpoint.
    #[error("missing weight key '{0}' in checkpoint")]
    MissingKey(String),

    /// I/O error reading the checkpoint file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
