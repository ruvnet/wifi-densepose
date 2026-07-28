//! HOMECORE — Rust port of `homeassistant/core.py`.
//!
//! Implements [ADR-127](../../docs/adr/ADR-127-homecore-state-machine-rust.md):
//! the state machine, event bus, service registry, and entity registry that
//! every other HOMECORE module depends on.
//!
//! ## Layout (P1 scaffold)
//!
//! - [`entity`] — `EntityId` newtype + validation; `State` snapshot type
//! - [`event`] — typed `SystemEvent` + untyped `DomainEvent` + `Context`
//! - [`state`] — `StateMachine`: DashMap-backed concurrent state store
//! - [`bus`] — `EventBus`: tokio broadcast wiring for system + domain events
//! - [`service`] — `ServiceRegistry` (stub; full mpsc dispatch lands in P2)
//! - [`registry`] — in-memory entity and device registries, restored by the server
//! - [`homecore`] — `HomeCore` runtime coordinator: holds bus + states + services
//!
//! ## Threading model
//!
//! HOMECORE is multi-threaded — concurrent reads from any number of tasks
//! return zero-copy `Arc<State>` clones. Writes are serialised per-entity
//! by the DashMap shard lock but the global state machine itself is never
//! locked. See ADR-127 §2.1.
//!
//! ## What's NOT here yet (deferred to P2+)
//!
//! - Automatic persistence of registry mutations (startup restoration exists)
//! - Schema validation (`schemas` module from §3 stub)
//! - Service handler mpsc dispatch (`service::ServiceRegistry::call`)
//! - Witness chain integration (ADR-028)
//!
//! Each is marked `// TODO P2:` at the relevant call site.

pub mod bus;
pub mod entity;
pub mod event;
pub mod registry;
pub mod service;
pub mod state;

mod homecore;

pub use homecore::HomeCore;

pub use bus::EventBus;
pub use entity::{EntityId, EntityIdError, State};
pub use event::{Context, DomainEvent, EventType, StateChangedEvent, SystemEvent};
pub use registry::{DeviceEntry, DeviceRegistry, EntityCategory, EntityEntry, EntityRegistry};
pub use service::{ServiceCall, ServiceError, ServiceName, ServiceRegistry};
pub use state::StateMachine;

/// HOMECORE protocol/data-model version. Bumped when the public surface
/// or on-disk persistence schema changes in a backwards-incompatible way.
/// Mirrors HA's `core.entity_registry` schema version (currently 13).
pub const HOMECORE_VERSION: u32 = 1;

/// Compile-time identifier for the HOMECORE build. Wired in by `vergen`
/// or git SHA in a later phase; constant for now.
pub const HOMECORE_BUILD_TAG: &str = env!("CARGO_PKG_VERSION");
