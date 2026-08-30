//! ADR-115 §3.11 — Matter Bridge (HA-FABRIC) scaffolding.
//!
//! This module owns the **Matter device-type and cluster mappings**
//! independent of any specific Matter SDK. Pure types + lookup tables
//! land here in v0.7.0; the actual SDK wiring (rs-matter or chip-tool
//! FFI per §9.10) lands in P7 → P8 in v0.7.1 once the SDK choice is
//! validated by a pairing spike against Apple Home / Google Home / HA.
//!
//! ## Why scaffolding-first
//!
//! 1. **Decision principle** (maintainer ACK §9): preserve clean
//!    protocols, avoid fake semantics, ship MQTT first, validate Matter
//!    second. This module defines what Matter *would* expose without
//!    committing to an SDK.
//! 2. **Reusability**. The mapping table is the same regardless of SDK
//!    choice — rs-matter and chip-tool both speak in cluster IDs +
//!    attribute IDs. Defining it here means the SDK swap (if needed
//!    at P7) is local.
//! 3. **Testability**. Cluster / attribute / event IDs are well-known
//!    integers in the Matter spec; we can validate the mapping against
//!    the spec without a live controller.
//!
//! ## Spec versions tracked
//!
//! - **Matter Core Spec 1.3** (CSA, 2024) — the surface this module
//!   targets. ID values below match §1.3 §A.1 Reserved Cluster IDs.
//!
//! Future Matter spec revisions that add biometric clusters (HR / BR)
//! would expand `EntityKind::matter_mapping` to cover them. Today HR /
//! BR have no Matter cluster and stay MQTT-only.

mod bridge;
mod clusters;
mod commissioning;

pub use bridge::{build_bridge_tree, BridgeTree, Endpoint, EndpointRef, NodeBranch};
pub use clusters::{matter_mapping, ClusterId, EndpointTypeId, MatterClusterMapping};
pub use commissioning::{DecodedManualCode, ManualPairingCode, SetupCodeInput};
