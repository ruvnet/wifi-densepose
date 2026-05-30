//! Ruflo AI-agent capabilities integration.
//!
//! Integrates the claude-flow daemon's AgentDB, AIDefence, and SONA intelligence
//! hooks into the ruview-swarm orchestrator via a trait-based backend.
//!
//! Feature gate: `ruflo`. The `RufloBackend` trait and `MockRufloBackend` are always
//! compiled so tests can use them without enabling the `ruflo` feature. Only
//! `HttpRufloBackend` (which requires `reqwest` + `serde_json`) is gated.

pub mod backend;
pub mod mock_backend;
pub mod mission_summary;

#[cfg(feature = "ruflo")]
pub mod http_backend;

pub use backend::{RufloBackend, RufloError, MissionMemoryEntry, PatternEntry, MavlinkScanResult};
pub use mock_backend::MockRufloBackend;
pub use mission_summary::MissionSummary;

#[cfg(feature = "ruflo")]
pub use http_backend::HttpRufloBackend;
