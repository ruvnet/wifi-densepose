# ADR-127: HOMECORE-CORE — Rust state machine, entity registry, event bus, service registry

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-CORE** |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master), [ADR-028](ADR-028-esp32-capability-audit.md) (witness chain), [ADR-124](ADR-124-rvagent-mcp-ruvector-npm-integration.md) (RUVIEW-POLICY) |
| **Tracking issue** | TBD |

---

## 1. Context

`homeassistant/core.py` is the 3,200-line heart of Python Home Assistant. It defines five objects that every other HA component depends on:

1. **`HomeAssistant`** — the runtime coordinator, event loop holder, and service locator. Contains `bus` (EventBus), `states` (StateMachine), `services` (ServiceRegistry), `config` (Config), `components` (loaded component set).
2. **`EventBus`** — publish/subscribe event dispatch. `async_fire(event_type, event_data)` dispatches to all registered listeners. Listener registration is `async_listen(event_type, callback)`. Wildcard listener is `MATCH_ALL`. Event data is a plain Python dict.
3. **`StateMachine`** — an in-memory dictionary from `entity_id` (str) to `State`. `async_set(entity_id, new_state, attributes)` writes and fires `state_changed`. `get(entity_id)` reads. `async_remove(entity_id)` fires `state_removed`. States are immutable snapshots with `last_changed`, `last_updated`, `context`.
4. **`ServiceRegistry`** — maps `(domain, service_name)` → async handler function. `async_call(domain, service, data)` fires a `call_service` event, waits for the registered handler. `async_register(domain, service, handler, schema)` registers a handler with optional voluptuous schema validation.
5. **`EntityRegistry`** (`homeassistant/helpers/entity_registry.py`) — persists metadata (enabled/disabled, name override, area assignment, device ID, unique ID, entity category) across restarts. Stored in `.storage/core.entity_registry`. Loaded at startup; written on every change.

The **DeviceRegistry** (`homeassistant/helpers/device_registry.py`, stored in `.storage/core.device_registry`) tracks physical devices that entities belong to. Entities link to devices via `device_id`; devices link to config entries via `config_entry_id`.

### 1.1 Why these specific files matter

Python HA's `core.py` is a single-process Python 3.12 module that:
- Holds the asyncio event loop directly
- Serialises all state-changed writes through `asyncio.Lock`
- Fires event listeners in the same event loop iteration that fired the event (listeners cannot block)
- Is single-threaded by design — concurrent writes to the state machine are impossible without explicit async primitives

For HOMECORE the same semantic requirements apply, but the implementation must support:
- **Concurrent reads** from dozens of integration WASM sandboxes polling current state
- **High-frequency writes** from the RuView sensing stack (CSI at 100 Hz; state updates at up to 20 Hz per entity)
- **Ordered delivery** of state_changed events to automation triggers (ADR-129) and recorder (ADR-132) subscribers
- **Zero-copy reads** where possible for the REST API (ADR-130) path

---

## 2. Decision

Implement the `homecore` Rust crate at `v2/crates/homecore/` with the following design.

### 2.1 State machine: `DashMap` + Tokio broadcast

The primary state store is a `DashMap<EntityId, Arc<State>>` where:
- `EntityId` is a validated newtype around `String` (validated format: `domain.name`)
- `State` is a frozen struct: `entity_id`, `state` (String), `attributes` (serde_json::Value), `last_changed` (DateTime<Utc>), `last_updated` (DateTime<Utc>), `context` (Context)
- `Arc<State>` allows zero-copy cloning for readers while the writer atomically replaces the map entry

State changes are published to a `tokio::sync::broadcast::Sender<StateChangedEvent>` channel (capacity: 4,096 events). Any number of receivers subscribe — the recorder, automation engine, WebSocket subscriber handler, and ruvector dual-write task all hold independent receivers. Slow receivers that fall behind by 4,096 events receive a `RecvError::Lagged` and must re-sync from the current state map.

### 2.2 Event bus: typed + untyped channels

HOMECORE distinguishes two event categories:

1. **System events** (typed): `StateChanged`, `ServiceCall`, `ComponentLoaded`, `PlatformDiscovered`, `HomeAssistantStart`, `HomeAssistantStop`. These use Tokio typed broadcast channels with zero allocation on the read path.
2. **Integration events** (untyped): integrations fire arbitrary event types (`event_type: String`, `event_data: serde_json::Value`). These use a single `broadcast::Sender<DomainEvent>` where `DomainEvent` carries the type string and data blob. This mirrors HA's `EventBus.async_fire()`.

### 2.3 Service registry: `HashMap` + mpsc dispatch

Services are registered as `(Domain, ServiceName) → ServiceHandler` where `ServiceHandler` is a `Box<dyn Fn(ServiceCall) -> BoxFuture<ServiceResponse> + Send + Sync>`. The registry lives in a `tokio::sync::RwLock<HashMap<(Domain, ServiceName), ServiceHandler>>`. Service calls go through the event bus (fire `call_service`) and are dispatched to the handler by an internal router task. This matches HA's indirection: `hass.services.async_call(domain, service, data)` does not call the handler directly; it fires an event.

### 2.4 Entity registry: persisted metadata sidecar

The entity registry is a `RwLock<HashMap<EntityId, EntityEntry>>` backed by an async JSON writer that flushes to `.homecore/storage/core.entity_registry` on every write. The schema matches HA's `core.entity_registry` schema (version 13 as of HA 2025.1) so ADR-134 migration can read both formats interchangeably.

`EntityEntry` fields mirrored from HA:
- `entity_id: EntityId`
- `unique_id: Option<String>`
- `platform: String`
- `name: Option<String>` (user override)
- `disabled_by: Option<DisabledBy>` (user, integration, config_entry)
- `area_id: Option<AreaId>`
- `device_id: Option<DeviceId>`
- `entity_category: Option<EntityCategory>` (config, diagnostic)
- `config_entry_id: Option<ConfigEntryId>`

### 2.5 Device registry: parallel sidecar

`DeviceRegistry` mirrors HA's `core.device_registry` schema (version 13). Devices are identified by a set of `(id_type, id_value)` tuples (the `identifiers` field), which matches HA's pattern of accepting multiple identifier types per device (MAC address, serial number, integration-specific ID).

---

## 3. HA-side reference table

| HA module / file | What it does | HOMECORE preserves | Changes | Drops |
|---|---|---|---|---|
| `homeassistant/core.py` `StateMachine` | In-memory state store, fire `state_changed` | Same semantics: immutable snapshots, `last_changed`, `last_updated`, `context` | `DashMap` instead of asyncio-locked `dict`; `broadcast::Sender` instead of asyncio callbacks | Python asyncio coupling |
| `homeassistant/core.py` `EventBus` | Pub/sub event dispatch | `MATCH_ALL` listener; per-type listener; event data dict | Typed system events + untyped domain events; no Python dict — use `serde_json::Value` | `@callback` decorator, HassJob abstraction |
| `homeassistant/core.py` `ServiceRegistry` | Register/call services | Same `(domain, service)` key structure; schema validation | Schema validation via `serde` `Deserialize` trait instead of voluptuous | voluptuous, Python type coercions |
| `homeassistant/core.py` `HomeAssistant` | Runtime coordinator / service locator | State machine + event bus + services accessible on one struct | Struct with `Arc<HomeCoreInner>` for cheap cloning across tasks | asyncio event loop holder, Python executor |
| `homeassistant/helpers/entity_registry.py` | Persist entity metadata | All fields listed in §2.4; file format compatible | Async tokio I/O; no Python pickle | Python-specific persistence helpers |
| `homeassistant/helpers/device_registry.py` | Persist device metadata | `identifiers`, `connections`, `manufacturer`, `model`, `name`, `via_device_id` | Async tokio I/O | — |
| `homeassistant/helpers/entity.py` | Entity base class | `entity_id`, `state`, `attributes`, `unique_id`, `device_info`, async_write_ha_state semantics | Trait `HomeCoreEntity` instead of class | Python MRO, `@property` decorators |
| `homeassistant/helpers/event.py` | Convenience event helpers | `async_track_state_change`, `async_track_time_interval` (as Rust timer tasks) | Rust closures / async tasks | Python asyncio task wrappers |

---

## 4. Public API parity table

| HA Python surface | HOMECORE Rust equivalent |
|---|---|
| `hass.states.get(entity_id)` | `hass.states.get(&entity_id) -> Option<Arc<State>>` |
| `hass.states.async_set(entity_id, state, attributes)` | `hass.states.set(entity_id, state, attributes).await` |
| `hass.states.async_remove(entity_id)` | `hass.states.remove(&entity_id).await` |
| `hass.states.async_all(domain_filter)` | `hass.states.all(domain_filter) -> Vec<Arc<State>>` |
| `hass.bus.async_fire(event_type, data)` | `hass.bus.fire(event_type, data).await` |
| `hass.bus.async_listen(event_type, callback)` | `hass.bus.subscribe(event_type) -> broadcast::Receiver<DomainEvent>` |
| `hass.services.async_call(domain, service, data)` | `hass.services.call(domain, service, data).await -> ServiceResponse` |
| `hass.services.async_register(domain, service, handler, schema)` | `hass.services.register(domain, service, handler)` |
| `hass.services.has_service(domain, service)` | `hass.services.has(domain, service) -> bool` |
| `entity_registry.async_get(entity_id)` | `entity_registry.get(&entity_id) -> Option<&EntityEntry>` |
| `entity_registry.async_update_entity(entity_id, **kwargs)` | `entity_registry.update(entity_id, patch).await` |
| `device_registry.async_get_device(identifiers)` | `device_registry.get_by_identifiers(identifiers) -> Option<&DeviceEntry>` |
| `Context(user_id, parent_id)` | `Context { id: Uuid, parent_id: Option<Uuid>, user_id: Option<UserId> }` |

---

## 5. Phased implementation plan

### P1 — Skeleton (2 weeks)

- [ ] Create `v2/crates/homecore/` workspace member with `Cargo.toml`.
- [ ] Define `State`, `EntityId`, `Domain`, `ServiceName`, `Context`, `DomainEvent` types.
- [ ] `StateMachine`: `DashMap` + broadcast channel; `set()`, `get()`, `remove()`, `all()`.
- [ ] `EventBus`: typed broadcast for system events + untyped broadcast for domain events.
- [ ] Unit tests: 50 state writes/reads with concurrent readers; verify broadcast delivery.

### P2 — Service registry + entity registry (2 weeks)

- [ ] `ServiceRegistry`: `RwLock<HashMap>` + mpsc dispatch task.
- [ ] `EntityRegistry`: in-memory + JSON async writer to `.homecore/storage/core.entity_registry`.
- [ ] `DeviceRegistry`: in-memory + JSON async writer to `.homecore/storage/core.device_registry`.
- [ ] Serialization: `serde` with `#[serde(rename_all = "snake_case")]`; schema version 13 header written to match HA format.
- [ ] Unit tests: register service, call service, verify handler invoked; persist and reload entity registry.

### P3 — Trait surface for integrations (1 week)

- [ ] `HomeCoreEntity` trait: `entity_id()`, `unique_id()`, `name()`, `device_info()`, `state()`, `attributes()`, `async_write_ha_state(&hass)`.
- [ ] `Platform` trait: `async_setup_entry(hass, config_entry) -> Result<()>`.
- [ ] `ConfigEntry` struct mirroring HA's `ConfigEntry` fields.
- [ ] Integration test: a minimal test integration registers an entity, writes a state, reads it back from the state machine.

### P4 — Performance validation (1 week)

- [ ] Benchmark: 1,000 state writes/s on Pi 5; measure latency at p50/p95/p99.
- [ ] Benchmark: 100 concurrent WS subscribers each receiving all state_changed events; measure delivery lag.
- [ ] Benchmark: broadcast channel saturation test at 4,096 capacity; verify `RecvError::Lagged` handling.
- [ ] Acceptance criterion: p99 state write latency < 1 ms on Pi 5 (8 GB, 4 cores).

---

## 6. Risks

| Risk | Likelihood | Severity | Mitigation | Cross-ADR impact |
|---|---|---|---|---|
| **Broadcast channel lag** — a slow subscriber (e.g. ruvector recorder write) lags behind and drops events | Medium | High | Give recorder its own channel separate from WS subscribers; recorder is the hot path, give it highest priority | ADR-132: recorder write path must be designed to keep up with 100 Hz state writes |
| **DashMap contention** — shard count default (16) may be too low for 100 Hz writes on a single entity | Low | Medium | Increase DashMap shard count to 64; benchmark before ADR-130 integration | ADR-130: REST API reads state directly from DashMap — must be lock-free |
| **Entity registry format drift** — HA updates `.storage/core.entity_registry` schema; HOMECORE falls behind | Medium | Medium | Pin to schema version 13; version-check on load; fail loudly on unknown version | ADR-134: migration tool reads HA entity registry — must support the same schema version |
| **Context propagation** — HA's `Context` is used for audit trails (which automation triggered which service call). HOMECORE must propagate it correctly or automation audits break | High | Low | Derive `Context` from source event at every service call; thread through `ServiceCall.context` field | ADR-129: automation engine must supply context when calling services |

---

## 7. Open questions

**Q1**: Should `EntityId` validation be strict (reject anything that doesn't match `[a-z0-9_]+\.[a-z0-9_]+`) or lenient (accept any UTF-8 string)? HA itself accepts unicode entity IDs since 2024.3. Strict validation simplifies routing; lenient matches HA's actual behaviour.

**Q2**: The `broadcast::Sender` capacity of 4,096 is chosen based on a worst-case of 100 state writes/s × 40 s of acceptable lag before a slow receiver is declared dead. Is 40 s the right threshold, or should it be configurable per receiver?

**Q3**: Should the `HomeCoreEntity` trait be object-safe (enabling `Vec<Box<dyn HomeCoreEntity>>`) or use associated types (enabling monomorphisation)? Object safety is required for the WASM plugin boundary (ADR-128); monomorphisation is faster for built-in integrations.

**Q4**: HA's `State.context` carries a `user_id` that traces which user or automation initiated a state change. HOMECORE uses `UserId` from the auth layer (ADR-130). Is the auth layer a dependency of the core state machine, or should `user_id` be an optional opaque string to avoid circular deps?

---

## 8. References

### HA upstream

- `homeassistant/core.py` — `HomeAssistant`, `StateMachine` (lines 1–800), `EventBus` (lines 800–1100), `ServiceRegistry` (lines 1100–1500), `Config` (lines 1500–2000)
- `homeassistant/helpers/entity_registry.py` — `EntityRegistry`, `RegistryEntry` (all ~1,900 lines); schema version constant `STORAGE_VERSION`
- `homeassistant/helpers/device_registry.py` — `DeviceRegistry`, `DeviceEntry`; schema version
- `homeassistant/helpers/entity.py` — `Entity` base class; `async_write_ha_state`; entity lifecycle hooks
- `homeassistant/helpers/event.py` — `async_track_state_change`, `async_track_time_interval`

### This repo

- `v2/crates/wifi-densepose-sensing-server/src/main.rs` — Axum + Tokio architecture pattern used throughout the existing server stack
- `docs/adr/ADR-126-ruview-native-ha-port-master.md` — HOMECORE master; §5.5 crate naming; §6 compatibility contract; §5.1 RUVIEW-POLICY

---

## 9. Security & concurrency review (P1 core, beyond-SOTA sweep)

Foundational review of the `homecore` crate — the state store + event bus +
service/entity registries every other HOMECORE module trusts. Same rigor as
the ADR-129/130/132/133/161 sibling reviews. **Three real fixes (one
concurrency, two hardening), each pinned by a fails-on-old test; the bus-lag
and lock-discipline dimensions confirmed clean with evidence.**

- **HC-RACE-01 (state-set TOCTOU — lost / reordered `state_changed`, the
  crux). FIXED.** `StateMachine::set` did `get()` (releasing the DashMap
  shard lock) → compute the next snapshot + the no-op / `last_changed`
  decision → `insert()` (re-acquiring the lock) → `send()`. The
  read-modify-write was **not atomic** w.r.t. a concurrent writer on the
  same entity, contradicting §2.1's promise that "the writer atomically
  replaces the map entry." A writer that read a stale `old` could
  mis-classify a genuine transition as a no-op and **drop its
  `state_changed` event** (a missed automation trigger) or fire an event
  whose `new_state` duplicated the previously delivered one (a spurious
  trigger for any automation keyed on `old_state != new_state`). **Fix:**
  hold the shard write-lock across the entire read→decide→insert→fire
  sequence via `entry()`/`insert_entry()`; `tx.send` is non-blocking,
  non-async, and never re-enters the map, so firing under the shard lock
  cannot deadlock and keeps global event order in lock-step with global
  commit order. Pinned by `concurrent_set_fires_no_duplicate_adjacent_events`
  (4 writers toggling one entity A/B; asserts no two consecutive fired
  events carry an identical `new_state` — impossible under correct
  serialisation; a probe observed ~93k such duplicate-adjacent events across
  200 trials on the racy code, zero on the fix).
- **HC-EID-LEN-01 (unbounded `entity_id` — memory-DoS at the REST boundary).
  FIXED.** `homecore-api/src/rest.rs` parses untrusted path segments
  straight through `EntityId::parse`; with no length cap, an
  otherwise-valid id (`a.` + many MB of `[a-z0-9_]`) was accepted and a
  `POST /api/states/<giant>` would persist it into the DashMap state store
  (permanent growth across distinct ids). **Fix:** reject ids longer than
  `MAX_ENTITY_ID_LEN` (255, HA-compatible) up front in `parse()`, before any
  per-char scan, with a new `EntityIdError::TooLong`; fail-closed at the
  boundary type protects every caller. Pinned by `entity_id_length_boundary`
  (exactly-MAX accepted, MAX+1 and a 4 MiB id rejected — fails on old code).
- **HC-SVC-PANIC-01 (service-handler panic not isolated). HARDENED.**
  `ServiceRegistry::call` already ran handlers outside the registry lock (no
  `RwLock` poisoning, no blocking of other callers — clean), but a
  panicking handler unwound through `call()` into the caller's task. **Fix:**
  wrap the handler future in `AssertUnwindSafe` + `catch_unwind`, converting
  a panic to `ServiceError::HandlerPanicked`; the registry stays fully
  usable. Pinned by `panicking_handler_is_isolated_and_registry_survives`.

**Dimensions confirmed clean (with evidence):**

- **Event-bus bounds / lag (same class as the homecore-api WS lag-DoS).**
  Both `StateMachine` and `EventBus` use bounded `tokio::sync::broadcast`
  (capacity 4,096). A slow subscriber gets a recoverable `Lagged(n)`
  (drop-oldest + re-sync); `fire_*` is non-blocking and **never waits on
  slow receivers**, so a lagging subscriber cannot block the publisher, grow
  the channel without bound, or take down a fast subscriber. Evidenced by
  `slow_subscriber_does_not_block_publisher_or_kill_the_bus` (fire 3×
  capacity at an idle subscriber; publisher unblocked, bus stays live).
- **Lock ordering / lock-across-await (deadlock).** No code path holds two
  of `{state DashMap, registry RwLock, service RwLock}` simultaneously, so
  no inconsistent-ordering deadlock can exist. Every `tokio::sync::RwLock`
  guard in `registry.rs`/`service.rs` is used in a single synchronous
  statement and dropped before any `.await`; `call` explicitly scopes the
  read guard out before awaiting the handler. The only guard held across a
  send is the DashMap shard lock in `set`, across a synchronous
  (non-await) broadcast send — safe.
- **Panic-on-input.** No reachable `unwrap`/`expect`/index in non-test code
  beyond the safe `send().unwrap_or(0)` and the dead-but-harmless
  `split_once(...).unwrap_or(...)` fallbacks on already-validated ids.

`cargo test -p homecore --no-default-features`: **20 → 24 passed, 0 failed**
(+4 pins). Workspace green; Python deterministic proof unchanged
(`f8e76f21…46f7a`, bit-exact — `homecore` is off the signal proof path).
- `docs/adr/ADR-028-esp32-capability-audit.md` — witness chain pattern (Ed25519 per state transition)
