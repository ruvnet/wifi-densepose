//! `homecore-hap` — Apple Home HomeKit Accessory Protocol bridge (ADR-125).
//!
//! # Network foundation scope
//!
//! The crate provides persisted controller records, a fail-closed session
//! state machine, bounded TLV8 parsing, characteristic event flow, and (with
//! `hap-server`) a bounded TCP/HTTP listener plus real mDNS. It does **not**
//! yet implement SRP Pair-Setup, X25519/HKDF/ChaCha20-Poly1305 Pair-Verify, or
//! encrypted HAP framing, so Apple Home pairing is deliberately unavailable.
//!
//! # Module layout
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`accessory`] | HAP service / characteristic enum catalogue |
//! | [`mapping`] | `EntityToAccessoryMapper` — HOMECORE entity → HAP |
//! | [`bridge`] | `HapBridge` — owns exposed accessories |
//! | [`mdns`] | `MdnsAdvertiser` trait + `NullAdvertiser` stub |
//! | [`pairing`] | Atomic controller pairing persistence |
//! | [`protocol`] | Bounded TLV8 protocol primitives |
//! | [`ruview`] | `RuViewToHapMapper` — sensing primitives → HAP |
//! | [`session`] | Authenticated request-gating state machine |
//! | `server` | Feature-gated bounded TCP/HTTP lifecycle |
//! | [`error`] | Unified `HapError` type |

pub mod accessory;
pub mod bridge;
pub mod error;
pub mod mapping;
pub mod mdns;
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
pub use pairing::{ControllerPairing, PairingStore};
pub use ruview::RuViewToHapMapper;
#[cfg(feature = "hap-server")]
pub use server::{start_server, HapServerConfig, HapServerHandle};
pub use session::{Session, SessionState};
