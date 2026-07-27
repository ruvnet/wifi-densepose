//! homecore-recorder — SQLite state history + semantic search.
//!
//! Implements ADR-132: dual-write architecture. P1 ships SQLite structural
//! persistence with an HA-compatible schema (mirrors HA recorder schema v48).
//! P2 (feature `ruvector`) adds a `SemanticIndex` backed by ruvector
//! embeddings for natural-language state queries.
//!
//! ## P1 architecture
//!
//! ```text
//!   StateMachine ──broadcast──► RecorderListener ──► Recorder
//!                                                       │
//!                                               ┌───────┴──────────┐
//!                                           states            state_attributes
//!                                           events            recorder_runs
//! ```
//!
//! ## P2 hand-off (ruvector feature)
//!
//! When the `ruvector` feature is enabled, the `Recorder` additionally
//! calls a `SemanticIndex` implementation that embeds state attributes and
//! stores vectors in ruvector for k-NN semantic search. See [`semantic`].

pub mod db;
pub mod dedup;
pub mod listener;
pub mod schema;

#[cfg(feature = "ruvector")]
pub mod semantic;

// Re-export the primary public API surface.
pub use db::{
    LatestStates, PurgeStats, Recorder, RecorderError, RestoreReport, RestoreWarning,
    SemanticIndex, StateRow, MAX_HISTORY_ROWS, MAX_RESTORE_STATES,
};
pub use listener::RecorderListener;

/// Null semantic index used when the `ruvector` feature is off.
/// Satisfies the [`db::SemanticIndex`] trait bound without any allocation.
pub use db::NullSemanticIndex;

#[cfg(feature = "ruvector")]
pub use semantic::RuvectorSemanticIndex;
