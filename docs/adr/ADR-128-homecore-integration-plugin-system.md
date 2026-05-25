# ADR-128: HOMECORE-PLUGINS — WASM integration plugin system

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-PLUGINS** |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE), [ADR-102](ADR-102-edge-module-registry.md) (cog registry), [ADR-100](ADR-100-cog-packaging-specification.md) (cog packaging spec) |
| **Tracking issue** | TBD |

---

## 1. Context

Home Assistant ships approximately 2,000 integrations, each a Python module in `homeassistant/components/<domain>/`. Each integration:

1. Declares a **manifest** (`manifest.json`) with `domain`, `name`, `version`, `requirements` (pip packages), `dependencies` (other HA integrations), `codeowners`, `iot_class`, `config_flow` (bool), and `quality_scale`.
2. Provides **`async_setup`** (global domain setup, called once at HA startup) and/or **`async_setup_entry`** (per-config-entry setup, called when a user adds an integration via the UI).
3. Imports Python packages from `requirements` at load time — these are installed into HA's Python environment by the loader at first run.
4. Communicates with the HA core exclusively through the `hass` object (the `HomeAssistant` instance) — setting states, calling services, registering services, subscribing to events.

In Python HA, integrations run **in-process** with the hub. A buggy integration can crash the event loop, read arbitrary HA memory, or import packages that conflict with other integrations. HA mitigates this via code review and quality scale requirements, but there is no runtime isolation boundary.

### 1.1 The Cognitum Seed cog system

The project already has a cog system (ADR-102, ADR-100) for the Cognitum Seed appliance. A **cog** is a signed, sandboxed module that installs from the Seed app registry. ADR-101 (`cog-pose-estimation`) shipped signed aarch64/x86_64 binaries with a model weight blob. ADR-116 (`cog-ha-matter`) shipped HA+Matter integration as a cog.

The cog system uses a different packaging model from HA integrations (binary artifacts vs Python packages), but the same conceptual pattern: a manifest, a lifecycle hook, and communication through a defined interface.

HOMECORE-PLUGINS unifies these two patterns: every HOMECORE integration is a **WASM module** that speaks the cog ABI, can be hot-loaded without restarting the hub, and is sandboxed by the WASM runtime.

---

## 2. Decision

HOMECORE integrations are **WASM modules** loaded by a Rust host runtime (`homecore-plugins` crate). Each plugin:

1. Compiles to a `.wasm` binary (from Rust, AssemblyScript, Go, or any WASM-targeting language).
2. Declares a `manifest.json` (superset of HA's manifest schema — see §3).
3. Exports exactly three WASM functions: `setup_entry(config_entry_ptr, config_entry_len) → i32`, `call_service(call_ptr, call_len) → i32`, and `receive_event(event_ptr, event_len) → i32`.
4. Imports a set of **host functions** from the HOMECORE host runtime: `hc_state_get`, `hc_state_set`, `hc_event_fire`, `hc_service_call`, `hc_log`, `hc_entity_register`.
5. Communicates with the host exclusively through those imports — no direct memory access outside its own linear memory.

The WASM runtime is **Wasmtime** (Cranelift JIT on Pi 5 and x86_64; interpretation mode available for low-memory targets via `--features wasm3`).

### 2.1 Why WASM over Python-in-process

| Criterion | Python in-process (HA today) | WASM sandbox (HOMECORE) |
|---|---|---|
| Memory isolation | None — any integration can read any HA object | WASM linear memory; host allocates shared buffer only for ABI calls |
| Crash isolation | Integration panic = HA event loop crash | WASM trap = plugin terminated, hub continues |
| Language support | Python only | Any WASM-targeting language: Rust, Go, AssemblyScript, C, Zig |
| Hot-load without restart | No — requires `asyncio.run_coroutine_threadsafe` patching | Yes — Wasmtime `Engine` + `Module::deserialize` from compiled `.cwasm` cache |
| Dependency conflicts | pip requirements collide across integrations | Each WASM module carries its own static dependencies (no runtime pip) |
| Startup cost per integration | Python import + pip install | Wasmtime JIT compile (~5 ms for a typical 200 kB WASM module); cached to `.cwasm` |

### 2.2 Cog system as the plugin substrate

The existing cog system (ADR-102) is the distribution and lifecycle layer. HOMECORE-PLUGINS extends it:

- **Distribution**: cogs are fetched from the Seed app registry (`app-registry.json`) or from a HOMECORE plugin registry (superset of the cog registry, same JSON schema + a `wasm_module` field).
- **Lifecycle**: `cognitum-agent` (ADR-116) already handles OTA update, signature verification, and sandboxed execution. HOMECORE-PLUGINS reuses this lifecycle by treating each HOMECORE integration as a cog with a WASM payload.
- **Ed25519 signatures**: every plugin `.wasm` is signed with the publisher's Ed25519 key. The HOMECORE host verifies the signature before compiling the module (same pattern as ADR-028 witness chain).

---

## 3. Manifest schema

HOMECORE's manifest is a superset of HA's `manifest.json`. Fields not present in HA are marked **[HOMECORE]**.

```json
{
  "domain": "mqtt",
  "name": "MQTT",
  "version": "2025.1.0",
  "documentation": "https://www.home-assistant.io/integrations/mqtt/",
  "iot_class": "local_push",
  "config_flow": true,
  "dependencies": [],
  "quality_scale": "platinum",
  "wasm_module": "mqtt.wasm",
  "wasm_module_hash": "sha256:abcdef...",
  "wasm_module_sig": "ed25519:<base64>",
  "publisher_key": "<base64 Ed25519 public key>",
  "min_homecore_version": "0.1.0",
  "host_imports_required": ["hc_state_get", "hc_state_set", "hc_event_fire", "hc_service_call"],
  "homecore_permissions": ["state:write:sensor.*", "state:read:*", "service:call:homeassistant.*"],
  "cog_id": "homecore-mqtt-2025.1.0"
}
```

**[HOMECORE]** fields:
- `wasm_module` — relative path to the `.wasm` binary
- `wasm_module_hash` — SHA-256 of the wasm binary; verified before execution
- `wasm_module_sig` — Ed25519 signature of the wasm binary hash
- `publisher_key` — Ed25519 public key of the publisher
- `min_homecore_version` — minimum HOMECORE version required
- `host_imports_required` — subset of host functions the module needs (security auditable)
- `homecore_permissions` — coarse-grained permission claims (glob patterns); future: enforcement via RUVIEW-POLICY layer (ADR-124 §4.1a)
- `cog_id` — Seed app registry ID for the cog distribution

---

## 4. HA-side reference table

| HA module / file | What it does | HOMECORE preserves | Changes | Drops |
|---|---|---|---|---|
| `homeassistant/components/<domain>/manifest.json` | Integration metadata | `domain`, `name`, `version`, `iot_class`, `config_flow`, `dependencies`, `quality_scale`, `documentation` | Add WASM fields; remove `requirements` (no pip) | `requirements` (pip packages) |
| `homeassistant/loader.py` | Loads Python modules; installs pip requirements | Manifest parsing; dependency resolution between cogs | WASM module loading via Wasmtime; no pip | Python `importlib`, pip subprocess |
| `homeassistant/components/<domain>/__init__.py` | `async_setup` + `async_setup_entry` | `setup_entry` hook (per config entry) | WASM export function instead of Python async function | Python module structure |
| `homeassistant/config_entries.py` | Config entry lifecycle management | `ConfigEntry` struct: `entry_id`, `domain`, `title`, `data`, `options`, `state`, `version` | Rust struct; async state machine | Python class hierarchy; `FlowManager` |
| `homeassistant/components/<domain>/config_flow.py` | UI configuration flow | Config flow metadata (steps, schemas) | JSON-schema-based flow descriptor shipped in manifest | `voluptuous`, Python UI flow runtime |

---

## 5. WASM ABI specification

### 5.1 Host functions imported by plugins

```
hc_state_get(key_ptr: i32, key_len: i32, out_ptr: i32, out_cap: i32) → i32
  // Returns JSON-encoded State into out_ptr buffer; returns bytes written or -1 if not found.

hc_state_set(entity_ptr: i32, entity_len: i32, state_ptr: i32, state_len: i32,
             attrs_ptr: i32, attrs_len: i32) → i32
  // Sets state for entity_id; returns 0 on success, negative on error.

hc_event_fire(event_type_ptr: i32, event_type_len: i32,
              event_data_ptr: i32, event_data_len: i32) → i32
  // Fires a domain event.

hc_service_call(domain_ptr: i32, domain_len: i32,
                service_ptr: i32, service_len: i32,
                data_ptr: i32, data_len: i32) → i32
  // Calls a service synchronously from the plugin's perspective (async on the host).

hc_entity_register(entry_ptr: i32, entry_len: i32) → i32
  // Registers an entity with the entity registry; entry is JSON-encoded EntityEntry.

hc_log(level: i32, msg_ptr: i32, msg_len: i32) → void
  // Structured log output; level: 0=debug, 1=info, 2=warn, 3=error.
```

### 5.2 WASM exports required by host

```
setup_entry(config_entry_ptr: i32, config_entry_len: i32) → i32
  // Called when a config entry is set up. config_entry is JSON-encoded ConfigEntry.
  // Returns 0 on success, negative error code on failure.

call_service_handler(domain_ptr: i32, domain_len: i32,
                     service_ptr: i32, service_len: i32,
                     data_ptr: i32, data_len: i32) → i32
  // Called when a service registered by this plugin is invoked.

receive_event(event_type_ptr: i32, event_type_len: i32,
              event_data_ptr: i32, event_data_len: i32) → i32
  // Called when an event type the plugin subscribed to fires.
  // Subscription is declared in manifest `subscribed_events` array.

alloc(size: i32) → i32
  // Host calls this to allocate a buffer inside the WASM linear memory
  // before writing data for a callback. Required for ABI memory passing.

dealloc(ptr: i32, size: i32) → void
  // Host calls this to free a previously allocated buffer.
```

### 5.3 Execution model

Each WASM module instance runs in its own Wasmtime `Store`. The host calls WASM exports from a dedicated Tokio task per plugin. Incoming events are queued in an `mpsc::Sender<PluginEvent>` per plugin; the plugin task drains the queue and calls `receive_event`. This isolates plugin execution from the hot state-machine path.

---

## 6. Public API parity table

| HA integration pattern | HOMECORE WASM equivalent |
|---|---|
| `async_setup_entry(hass, entry)` Python async function | `setup_entry(config_entry_json)` WASM export |
| `hass.states.async_set(entity_id, state, attrs)` | `hc_state_set(...)` host import |
| `hass.states.get(entity_id)` | `hc_state_get(...)` host import |
| `hass.bus.async_fire(event_type, data)` | `hc_event_fire(...)` host import |
| `hass.services.async_call(domain, service, data)` | `hc_service_call(...)` host import |
| `hass.services.async_register(domain, service, handler)` | Declared in manifest `registered_services`; `call_service_handler` WASM export handles all |
| `async_track_state_change(hass, entity_ids, callback)` | Declared in manifest `subscribed_state_entities`; `receive_event` called with `state_changed` events |
| Config flow `FlowManager.async_init()` | Config flow metadata in manifest; UI calls HOMECORE-API `/config/config_entries/flow` |
| `ConfigEntry.entry_id`, `.domain`, `.data`, `.options` | Same fields in `ConfigEntry` JSON passed to `setup_entry` |

---

## 7. Phased implementation plan

### P1 — WASM host skeleton (2 weeks)

- [ ] Create `v2/crates/homecore-plugins/` workspace member.
- [ ] Wasmtime dependency; compile a trivial WASM module that calls `hc_log` and verify it runs.
- [ ] Define the host function ABI in a `host_api.rs` module; write the Wasmtime `Linker` registration for all 6 host functions.
- [ ] Manifest schema: `serde`-deserialised `Manifest` struct; validate required fields.
- [ ] Hash + Ed25519 signature verification of `.wasm` bytes before compilation.

### P2 — State machine bridge (2 weeks)

- [ ] Wire `hc_state_get` and `hc_state_set` to the `homecore` state machine (ADR-127).
- [ ] Wire `hc_event_fire` to the event bus.
- [ ] Wire `hc_service_call` to the service registry.
- [ ] Wire `hc_entity_register` to the entity registry.
- [ ] Write a test plugin in Rust compiled to WASM: registers one entity, writes its state via host imports, verifies the state machine sees the update.

### P3 — Config entry lifecycle + hot-load (2 weeks)

- [ ] `ConfigEntryManager` — tracks loaded plugins, calls `setup_entry` on new config entries, handles teardown.
- [ ] Hot-load: watch a directory for new `.wasm` + `manifest.json` pairs; load without hub restart.
- [ ] Wasmtime compiled module cache: serialize to `.cwasm` after first JIT compile; deserialize on subsequent loads (sub-1 ms plugin restart).
- [ ] Integration test: MQTT plugin loaded at runtime, registers `sensor.test` entity, state readable via HOMECORE-API.

### P4 — Cog registry integration (1 week)

- [ ] Fetch plugin from Seed app registry `app-registry.json`; verify Ed25519 signature against publisher key.
- [ ] Expose `/api/homecore/plugins` REST endpoint (HOMECORE-API ADR-130 extension): list loaded plugins, load new plugin by URL, unload plugin.
- [ ] First-party plugin: ship an MQTT plugin WASM module that provides the same function as HA's `homeassistant/components/mqtt/`.

### P5 — Permission enforcement (1 week)

- [ ] Enforce `homecore_permissions` claims: reject `hc_state_set` calls that write to entities outside the plugin's declared `state:write:*` pattern.
- [ ] Log all permission denials to the Ed25519 witness chain.
- [ ] Expose permission audit via `/api/homecore/plugins/<domain>/audit`.

---

## 8. Risks

| Risk | Likelihood | Severity | Mitigation | Cross-ADR impact |
|---|---|---|---|---|
| **ADR-127 state machine not stable** — plugin ABI calls into the state machine; if the API changes, all plugins break | High (early phase) | High | Freeze the `hc_state_get`/`hc_state_set` ABI in P1; never change pointer/length convention; version the host ABI in the manifest `min_homecore_version` | ADR-127 must freeze public API before ADR-128 P2 begins |
| **Wasmtime binary size** — adding Wasmtime to HOMECORE adds ~15 MB to the binary on Pi 5 | Medium | Medium | Use Cranelift JIT only; skip LLVM optimizer. Alternative: `wasm3` feature flag (~50 kB) for constrained hardware | ADR-126: binary size target < 50 MB idle RAM; Wasmtime itself uses ~5 MB RAM at runtime |
| **ABI memory overhead** — every state read/write from a plugin must JSON-encode/decode through shared memory | Medium | Medium | Cap state value size at 64 kB; use a pool allocator for ABI buffers; profile on Pi 5 at 10 state writes/s per plugin | ADR-130: REST API reads state from DashMap directly, bypassing plugin ABI — no overhead there |
| **Community plugin trust** — WASM sandbox prevents crashes but cannot prevent malicious plugins from calling `hc_service_call` to turn off all lights | Medium | High | `homecore_permissions` permission claims (P5); future: RUVIEW-POLICY enforcement (ADR-124 §4.1a) for biometric data access | ADR-124 RUVIEW-POLICY must be made aware of HOMECORE as a policy principal |

---

## 9. Open questions

**Q1**: Should the WASM module ABI use JSON-over-shared-memory (current proposal) or a more compact binary encoding (MessagePack, FlatBuffers)? JSON is simpler to debug and matches HA's existing JSON-everywhere convention; MessagePack cuts ABI overhead by ~4×. Decide before P2 implementation.

**Q2**: HA's `config_flow.py` is a multi-step UI wizard with voluptuous schema validation. HOMECORE's config flow is described in the manifest JSON. Is a JSON-schema-based config flow sufficient for the 100 most popular integrations, or do some require imperative step logic that can't be expressed declaratively?

**Q3**: Should existing Python HA community integrations be automatically compilable to WASM via a transpilation layer (e.g. CPython compiled to WASM via Pyodide), or should HOMECORE accept only natively compiled WASM modules? Pyodide+WASM would make migration easier but adds ~25 MB per plugin and loses the performance argument.

**Q4**: The `host_imports_required` manifest field lists which host functions the plugin needs. Should this be verified at load time (reject plugin that imports undeclared functions) or only advisory? Strict enforcement prevents surprises; advisory aids migration.

---

## 10. References

### HA upstream

- `homeassistant/loader.py` — integration loader; pip requirement installation; `async_setup_entry` invocation
- `homeassistant/config_entries.py` — `ConfigEntry`, `ConfigEntryState`, `ConfigEntriesError`, `FlowManager`
- `homeassistant/components/mqtt/manifest.json` — canonical example of HA manifest structure
- `homeassistant/components/mqtt/__init__.py` — `async_setup_entry` pattern for a complex integration with services
- `homeassistant/components/mqtt/config_flow.py` — multi-step config flow example

### This repo

- `docs/adr/ADR-102-edge-module-registry.md` — cog registry architecture; `app-registry.json` schema
- `docs/adr/ADR-100-cog-packaging-specification.md` — cog packaging spec; Ed25519 signing
- `docs/adr/ADR-101-pose-estimation-cog.md` — cog lifecycle precedent
- `docs/adr/ADR-127-homecore-state-machine-rust.md` — state machine ABI that plugins call
- `docs/adr/ADR-126-ruview-native-ha-port-master.md` — §5.7 "do not port" list (legacy Python integrations)
