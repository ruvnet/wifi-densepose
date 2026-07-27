//! `homecore-hap` — Apple Home HomeKit Accessory Protocol bridge (ADR-125).
//!
//! # Network foundation scope
//!
//! The crate provides persisted accessory/controller identity, SRP-6a
//! Pair-Setup, X25519/Ed25519 Pair-Verify, encrypted HAP IP framing, bounded
//! TLV8/HTTP parsing, characteristic event flow, and (with `hap-server`) a
//! bounded TCP listener plus real mDNS.
//!
//! # Module layout
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`accessory`] | HAP service / characteristic enum catalogue |
//! | [`mapping`] | `EntityToAccessoryMapper` — HOMECORE entity → HAP |
//! | [`bridge`] | `HapBridge` — owns exposed accessories |
//! | [`mdns`] | `MdnsAdvertiser` trait + `NullAdvertiser` stub |
//! | [`pairing`] | Atomic accessory identity, setup, and pairing persistence |
//! | [`protocol`] | Bounded TLV8 protocol primitives |
//! | [`ruview`] | `RuViewToHapMapper` — sensing primitives → HAP |
//! | [`session`] | Authenticated request-gating state machine |
//! | `server` | Feature-gated bounded TCP/HTTP lifecycle |
//! | [`error`] | Unified `HapError` type |

pub mod accessory;
pub mod bridge;
mod crypto;
pub mod error;
pub mod mapping;
pub mod mdns;
mod pair_setup;
mod pair_verify;
pub mod pairing;
pub mod protocol;
pub mod ruview;
#[cfg(feature = "hap-server")]
pub mod server;
pub mod session;

pub use accessory::{HapAccessoryType, HapCharacteristic, HapCharacteristicValue};
pub use bridge::{CharacteristicEvent, ExposedAccessory, HapBridge};
pub use error::HapError;
pub use mapping::EntityToAccessoryMapper;
#[cfg(feature = "hap-server")]
pub use mdns::MdnsSdAdvertiser;
pub use mdns::{HapServiceRecord, MdnsAdvertiser, NullAdvertiser};
pub use pairing::{ControllerPairing, PairingStore, PairingStoreProvisioning, SetupCode};
pub use ruview::RuViewToHapMapper;
#[cfg(feature = "hap-server")]
pub use server::{start_server, HapServerConfig, HapServerHandle};
pub use session::{Session, SessionState};
