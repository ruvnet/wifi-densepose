//! `homecore-hap` — Apple Home HomeKit Accessory Protocol bridge (ADR-125).
//!
//! # P1 scope
//!
//! Ships the trait surface and type definitions needed to map HOMECORE entity
//! states onto HAP accessory / characteristic values. The actual HAP-1.1 TLS
//! server and real mDNS advertisement are gated behind the `hap-server`
//! feature (P2). P1 ships `NullAdvertiser` (no-op) so the bridge compiles and
//! all tests pass with `--no-default-features`.
//!
//! # Module layout
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`accessory`] | HAP service / characteristic enum catalogue |
//! | [`mapping`] | `EntityToAccessoryMapper` — HOMECORE entity → HAP |
//! | [`bridge`] | `HapBridge` — owns exposed accessories |
//! | [`mdns`] | `MdnsAdvertiser` trait + `NullAdvertiser` stub |
//! | [`ruview`] | `RuViewToHapMapper` — sensing primitives → HAP |
//! | [`error`] | Unified `HapError` type |

pub mod accessory;
pub mod bridge;
pub mod error;
pub mod mapping;
pub mod mdns;
pub mod ruview;

pub use accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};
pub use bridge::{ExposedAccessory, HapBridge};
pub use error::HapError;
pub use mapping::EntityToAccessoryMapper;
pub use mdns::{MdnsAdvertiser, NullAdvertiser};
pub use ruview::RuViewToHapMapper;
