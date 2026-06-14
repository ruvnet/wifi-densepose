//! # wifi-densepose-rufield
//!
//! ADR-262 **anti-corruption bridge**: converts RuView's live WiFi-CSI sensing
//! output into signed RuField [`FieldEvent`](rufield_core::FieldEvent)s.
//!
//! This crate is the **single coupling point** (ADR-262 §5.4) between RuView and
//! the standalone RuField MFS spec (`vendor/rufield`, ADR-260). It depends on
//! the four pure-Rust rufield crates **via path** — `rufield-core`,
//! `-provenance`, `-privacy`, `-fusion` — and on **no** RuView internal crate.
//! Inputs are owned primitives ([`SensingSnapshot`]) that mirror what RuView's
//! sensing cycle produces, so the bridge never imports `SensingUpdate` /
//! `TrustedOutput` directly.
//!
//! ## What P1 ships (honesty — ADR-262 §0 / §6)
//!
//! This is **P1 plumbing**: a tested `SensingSnapshot → FieldEvent` conversion
//! plus the **fail-closed privacy mapping** that is the §3.3 correctness item.
//! It is **not** wired into the live server (that is P3) and makes **no accuracy
//! claim** — RuField v0.1 is synthetic end-to-end and RuView's single-link CSI
//! carries its own caveats. The gates here are round-trip / fusability /
//! privacy-safety / determinism, not validated F1.
//!
//! ## The critical correctness item: the privacy mapping (§3.3)
//!
//! RuView's `Derived` class has byte value `1` (below `Anonymous = 2`) yet
//! carries an identity embedding. The bridge maps it to **P4/P5 by information
//! content, never P1** — see [`map_privacy`]. Mapping off the byte would leak
//! identity as low-privacy; [`map_privacy`] (and its dedicated test
//! `derived_identity_never_maps_to_low_privacy`) exist specifically to prevent
//! that.
//!
//! ## Example
//!
//! ```
//! use wifi_densepose_rufield::{
//!     snapshot_to_field_event, SensingSnapshot, SensingFeatures, SensingClass,
//!     RuViewPrivacyClass,
//! };
//! use rufield_provenance::{Signer, is_fusable};
//!
//! let snap = SensingSnapshot {
//!     timestamp_ns: 1_791_986_400_000_000_000,
//!     features: SensingFeatures {
//!         mean_rssi: -55.0,
//!         variance: 0.4,
//!         motion_band_power: 2.0,
//!         breathing_band_power: 0.3,
//!         dominant_freq_hz: 0.25,
//!         change_points: 1,
//!         spectral_power: 3.0,
//!     },
//!     classification: SensingClass {
//!         motion_level: "low".into(),
//!         presence: true,
//!         confidence: 0.82,
//!     },
//!     signal_field: None,
//!     trust_class: RuViewPrivacyClass::Anonymous,
//!     demoted: false,
//!     identity_bound: false,
//!     node_id: "esp32_room_01".into(),
//! };
//!
//! let signer = Signer::from_seed(b"adr-262-bridge-seed-32-bytes-ok!");
//! let event = snapshot_to_field_event(&snap, &signer);
//! assert!(is_fusable(&event)); // ed25519-signed, non-synthetic ⇒ fusable
//! ```

#![forbid(unsafe_code)]

pub mod bridge;
pub mod privacy;
pub mod snapshot;

pub use bridge::{snapshot_egress_class, snapshot_to_field_event};
pub use privacy::{apply_demotion_floor, egress_class, map_privacy};
pub use snapshot::{
    RuViewPrivacyClass, SensingClass, SensingFeatures, SensingSnapshot, SignalField,
};

// Re-export the rufield surface a bridge consumer needs, so callers depend on
// one crate.
pub use rufield_core::{FieldEvent, Modality, PrivacyClass};
pub use rufield_fusion::RuFieldFusion;
pub use rufield_provenance::{is_fusable, verify_event, Signer};
