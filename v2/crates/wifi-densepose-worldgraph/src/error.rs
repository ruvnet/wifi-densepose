//! WorldGraph error type.

use crate::model::WorldId;

/// Errors from WorldGraph operations.
#[derive(Debug, thiserror::Error)]
pub enum WorldGraphError {
    /// An edge endpoint referenced an unknown node.
    #[error("unknown node {0:?}")]
    UnknownNode(WorldId),

    /// (De)serialisation of the persisted graph failed.
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
