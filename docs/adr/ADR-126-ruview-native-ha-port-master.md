# ADR-126: HOMECORE — Native Rust + WASM + TypeScript port of Home Assistant

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE** — native hub, RuView-first, WASM-safe, semantically aware |
| **Relates to** | [ADR-115](ADR-115-home-assistant-integration.md) (HA-DISCO), [ADR-116](ADR-116-cog-ha-matter-seed.md) (HA-COG), [ADR-117](ADR-117-pip-wifi-densepose-modernization.md) (PIP-PHOENIX), [ADR-118](ADR-118-bfld-beamforming-feedback-layer-for-detection.md) (BFLD), [ADR-124](ADR-124-rvagent-mcp-ruvector-npm-integration.md) (SENSE-BRIDGE), [ADR-125](ADR-125-ruview-apple-home-native-hap-bridge.md) (APPLE-FABRIC) |
| **Tracking issue** | TBD |
| **Sub-ADRs** | ADR-127 through ADR-134 |

---

## 1. Context

### 1.1 Strategic position in 2026

Home Assistant (HA) is the dominant open-source home automation hub with more than 500,000 active installs (ADR-115 §1.2 competitive scan). Every prior RuView integration decision has been made with HA as a given constraint: ADR-115 built an MQTT auto-discovery publisher to fit inside HA, ADR-116 packaged it as a Cognitum Seed cog, ADR-122 extended it with BFLD presence events, and ADR-125 layered a native HAP bridge on top of the same stack.

This approach yields functioning integrations, but it positions RuView permanently as a **guest in someone else's hub**. The architectural limits of Python HA are not just cosmetic:

| Limit | Impact on RuView's roadmap |
|---|---|
| **Single-process Python GIL** | CSI DSP pipeline, BFLD analysis, and ruvector semantic search cannot run concurrently inside the HA process; they must run as external services connected over MQTT or WebSocket, introducing a round-trip on every sensor update |
| **Startup time (15–30 s on a Pi 5)** | The Cognitum Seed appliance restarts firmware-update-by-firmware-update; a 30 s hub startup on every OTA cycle is user-visible latency |
| **Memory footprint (300 MB+ idle)** | On a Pi 5 with 8 GB this is tolerable; on a Pi Zero 2 W or an embedded board with 512 MB it precludes co-location with the sensing stack |
| **No WASM safety boundary for integrations** | HA's 2,000+ community integrations are Python modules loaded directly into the HA process — one buggy integration can crash the hub or read arbitrary memory |
| **Recorder is structural only** | SQLite + InfluxDB store state history as rows; there is no semantic search. "Show me when the porch light correlated with the bedroom CSI anomaly last week" requires manual SQL |
| **Voice assistant is additive** | Assist (`homeassistant/components/assist_pipeline/`) was added in 2022–2023 and is well-designed, but intent matching is keyword-based, not embedding-based; ruflo LLM pipelines cannot natively plug in |
| **Frontend is a 5 MB Lit-element bundle** | The dashboard compiles to ~5 MB of JavaScript; on low-bandwidth appliance UIs or Progressive-Web-App installs, this is perceptible load time |

These are not HA's failures — they are Python architectural realities. For a generic home automation hub they are acceptable. For a hub where the core value proposition is **real-time RF sensing, AI-augmented automation, and edge-native deployment on constrained hardware**, they are ceilings.

### 1.2 The opportunity

Three recent ADR shipments create the inflection point:

1. **ADR-117 (PIP-PHOENIX)** — `wifi-densepose==2.0.0a1` + `ruview==2.0.0a1` on PyPI as PyO3/maturin wheels, providing a Python developer surface over the Rust sensing core.
2. **ADR-118 (BFLD)** — a complete beamforming feedback capture and privacy-risk scoring layer, proving that RuView's sensing stack can be a compliance instrument, not just a sensor.
3. **ADR-124 (SENSE-BRIDGE)** — `@ruvnet/rvagent` on npm as a dual-transport MCP server, proving that the sensing stack can be expressed as a first-class AI-agent tool surface.

The gap that remains: there is no hub that treats all of these as **native first-class features** rather than bolt-on integrations. HOMECORE fills that gap by porting the HA data model and API surface to Rust, replacing HA's Python internals with the RuView Rust crates, and wrapping community integrations in WASM sandboxes.

### 1.3 What this ADR is *not*

- Not a fork of the Python HA codebase. HOMECORE is a **clean-room Rust implementation** of HA's public API contracts and data model, not a line-by-line port.
- Not a replacement of the existing sensing stack. `v2/crates/wifi-densepose-*` remain authoritative.
- Not a deprecation of ADR-115/116/117/124/125. Those integrations continue to work with Python HA installs. HOMECORE is an additional deployment target, not a replacement mandate.
- Not a Matter SDK full-implementation. ADR-125 handles Matter; HOMECORE consumes the Matter bridge via the existing `cog-ha-matter` surface.
- Not a target for this quarter's sprint. HOMECORE is a multi-quarter initiative. This master ADR and its sub-ADRs define the architecture; implementation begins in P1.

---

## 2. Decision

Build **HOMECORE**: a native Rust + WASM + TypeScript implementation of the Home Assistant hub contract, integrated with the RuView sensing platform, the ruflo agent toolchain, and the ruvector vector layer.

HOMECORE is wire-compatible with HA's REST and WebSocket APIs so that existing HA-native clients (the iOS/Android Home Assistant companion apps, HACS, Nabu Casa Cloud, and the HA voice satellite stack) operate without modification against a HOMECORE instance.

HOMECORE is NOT a drop-in replacement on day one. The compatibility contract is phased (§6). The architecture is designed so that clients that work with HA today work with HOMECORE P3+.

### 2.1 Codename rationale

**HOMECORE** — the `core` of HA reimplemented at native speed, with the sensing stack at the center rather than at the periphery.

---

## 3. Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│                     HOMECORE process                          │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  homecore   │  │  homecore-   │  │  homecore-        │  │
│  │  state      │  │  automation  │  │  recorder         │  │
│  │  machine    │  │  engine      │  │  (SQLite +        │  │
│  │  (ADR-127)  │  │  (ADR-129)   │  │  ruvector)        │  │
│  └──────┬──────┘  └──────┬───────┘  │  (ADR-132)        │  │
│         │                │          └───────────────────┘  │
│  ┌──────▼──────────────────────────────────┐               │
│  │              Event Bus (Tokio broadcast) │               │
│  └──────┬──────────────────────────────────┘               │
│         │                                                    │
│  ┌──────▼──────────────────────────────────┐               │
│  │     homecore-rest-websocket-api (ADR-130)│               │
│  │     Axum server — HA wire-compat API     │               │
│  └──────────────────────────────────────────┘               │
│                                                               │
│  ┌──────────────┐  ┌──────────────────────────────────────┐ │
│  │ Integration  │  │  homecore-assist-ruflo (ADR-133)      │ │
│  │ Plugin System│  │  ruflo agent orchestration            │ │
│  │ (ADR-128)    │  │  ruvector intent embeddings           │ │
│  │ WASM sandbox │  │  Wyoming protocol edge               │ │
│  └──────────────┘  └──────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  RuView sensing core (wifi-densepose-sensing-server) │   │
│  │  CSI → presence / vitals / pose / BFLD / semantic    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
          │ HA-compatible REST + WebSocket
          ▼
┌──────────────────────────┐
│ homecore-frontend-ts-wasm │  (ADR-131)
│ TypeScript + Rust→WASM   │
│ SharedWorker state sync  │
└──────────────────────────┘
```

The HOMECORE process is a single Tokio-based async Rust binary. The state machine and event bus are the authoritative core (ADR-127). Integrations run in WASM sandboxes that communicate with the core via a defined ABI (ADR-128). The automation engine runs Rust-native trigger evaluation with a WASM expression evaluator for templates (ADR-129). The REST/WebSocket API layer is Axum-based and wire-compatible with HA (ADR-130). The frontend is TypeScript with the state machine compiled to WASM running in a SharedWorker (ADR-131). Historical state is stored in SQLite with ruvector for semantic search (ADR-132). Voice/text assistance uses ruflo agent orchestration (ADR-133).

---

## 4. Series map

| ADR | Codename | Scope | Critical path? | Estimated P5-completion |
|---|---|---|---|---|
| **ADR-127** | HOMECORE-CORE | Rust state machine, entity registry, event bus, service registry (`homecore` crate) | **Yes — all others depend on it** | Q3 2026 |
| **ADR-128** | HOMECORE-PLUGINS | WASM integration plugin system, cog substrate, manifest schema, hot-load | **Yes — needed before any integration can run** | Q3 2026 |
| **ADR-129** | HOMECORE-AUTO | Automation engine, YAML parser, Jinja2-equivalent WASM evaluator, blueprints | Yes (automation is core to HA UX) | Q4 2026 |
| **ADR-130** | HOMECORE-API | REST + WebSocket wire-compat API, Axum server, HA companion app support | **Yes — needed for client compat** | Q3 2026 |
| **ADR-131** | HOMECORE-UI | TS + Rust→WASM frontend, SharedWorker state sync, Material 3 design lang | No (can run alongside Python HA UI initially) | Q1 2027 |
| **ADR-132** | HOMECORE-RECORDER | SQLite recorder + ruvector semantic history, schema migration | No (structural recorder ships before ruvector layer) | Q4 2026 |
| **ADR-133** | HOMECORE-ASSIST | ruflo agent voice assistant, ruvector intent matching, Wyoming edge path | No | Q4 2026 |
| **ADR-134** | HOMECORE-MIGRATE | Migration tooling from Python HA, config-entry parser, side-by-side mode | No (needed for user adoption) | Q1 2027 |

**Critical path**: ADR-127 → ADR-128 → ADR-130 must land in that order. ADR-129, ADR-132, ADR-133, ADR-131, ADR-134 can proceed in parallel once the core triad is stable.

---

## 5. Cross-cutting decisions

The following decisions govern all 8 sub-ADRs and are not repeated in each.

### 5.1 Governance via RUVIEW-POLICY (ADR-124 §4.1a)

Every HOMECORE component that returns biometric data (presence, HR/BR, pose keypoints, BFLD identity-risk) MUST route through the RUVIEW-POLICY layer defined in ADR-124 §4.1a. The policy store is the same `~/.config/rvagent/policy.json` used by `@ruvnet/rvagent`. HOMECORE is a first-class policy principal — its agent ID in the policy store is `homecore`.

### 5.2 Semantic memory via ruvector

Historical state is not only stored in SQLite rows (structural). Every state-changed event is also embedded via ruvector (using the same napi-rs bindings as ADR-124) and indexed in an HNSW store for semantic search. The `homecore-recorder` crate (ADR-132) owns this dual-write. Queries like "when did the living room motion last exceed baseline?" become vector-nearest-neighbour searches, not SQL BETWEEN clauses.

### 5.3 Agent orchestration via ruflo

The automation engine (ADR-129) and the assist pipeline (ADR-133) both have an optional ruflo-agent mode where complex conditions or voice intents are routed to a ruflo agent (using the `mcp__claude-flow__*` tool namespace) for LLM-backed resolution. This is gated by RUVIEW-POLICY: a policy grant is required before HOMECORE sends any state-history context to a ruflo agent.

### 5.4 Witness and audit via Ed25519 chain (ADR-028 pattern)

Every state transition that crosses a privacy boundary (e.g. BFLD identity-risk score elevated, a biometric entity state published) is logged to an Ed25519 witness chain using the same structure as ADR-028 §3. The witness bundle is exportable for regulated deployments (care homes, hotels, shared offices).

### 5.5 Crate naming and workspace placement

All HOMECORE crates live in `v2/crates/homecore-*/`:

| Crate | ADR |
|---|---|
| `homecore` | ADR-127 |
| `homecore-plugins` | ADR-128 |
| `homecore-automation` | ADR-129 |
| `homecore-api` | ADR-130 |
| `homecore-recorder` | ADR-132 |
| `homecore-assist` | ADR-133 |
| `homecore-migrate` | ADR-134 |

The frontend (`homecore-frontend`) is not a Rust crate — it is an npm package at `npm/homecore-frontend/`, mirroring the `npm/rvagent/` pattern from ADR-124.

### 5.6 HA wire-compatibility baseline

The HOMECORE REST and WebSocket API must be **compatible with HA 2025.1** as the baseline. HA 2025.1 introduced schema version 48 in the recorder. The API surface to replicate is:

- REST: `homeassistant/components/api/__init__.py` — 24 endpoints
- WebSocket: `homeassistant/components/websocket_api/` — the `connection.py` + `commands.py` handler pattern, the auth handshake, and the `subscribe_events` / `subscribe_trigger` / `call_service` commands
- Auth: `homeassistant/auth/` — the long-lived access token model
- Config entries: `.storage/core.config_entries` JSON schema (versioned, auto-migrated)

### 5.7 "Do not port" list

The following HA subsystems are explicitly **not** ported to HOMECORE:

| HA subsystem | Reason not ported | HOMECORE replacement |
|---|---|---|
| **SUPERVISOR** (`homeassistant/supervisor/`) | Manages add-on containers and OS upgrades. HOMECORE runs on a standard Linux/Pi OS managed by systemd. | ruflo + systemd service units + OTA via the existing Cognitum Seed OTA registry (ADR-116 §2.2) |
| **Home Assistant OS** (HAOS) | A custom embedded Linux image. HOMECORE targets standard Debian/Ubuntu on Pi 5 and standard Docker. | Standard OS + Docker Compose or systemd |
| **Nabu Casa Cloud** | Paid remote-access and Alexa/Google integration service. HOMECORE uses Tailscale for remote access and `@ruvnet/rvagent` for AI integration. | Tailscale + ADR-107 federation + SENSE-BRIDGE |
| **Add-on store** (Supervisor add-ons) | Docker container management. | Cognitum Seed cog registry (ADR-102) |
| **Legacy YAML-only integrations** (pre-config-flow, ~500 of 2,000) | These require Python `setup_platform` (deprecated in HA 2024.x). Only config-flow integrations (`async_setup_entry`) are ported. | Document upgrade path; unported integrations can run via `homecore-migrate` bridge mode |
| **Analytics / Nabu Casa telemetry** | Optional cloud telemetry. | Not replicated. HOMECORE is local-only. |
| **Home Assistant Yellow / Green hardware** | Specific hardware. HOMECORE targets Cognitum Seed, Pi 5, and x86_64. | Cognitum Seed hardware |

---

## 6. Compatibility contract

### 6.1 What works on day one (P3, wire-compat API stable)

| Client | Works? | Notes |
|---|---|---|
| **HA iOS companion app** | Yes | Connects to `/api/websocket`; authenticates with long-lived token; subscribes to state events |
| **HA Android companion app** | Yes | Same as iOS |
| **Home Assistant Dashboard (frontend)** | Yes (HA frontend served against HOMECORE API) | Until HOMECORE-UI (ADR-131) ships, serve the Python HA frontend binary against the HOMECORE API |
| **HACS** | Partial | HACS uses the WS API for integration management; custom component loading requires HOMECORE-PLUGINS (ADR-128) |
| **Node-RED HA integration** | Yes | Uses REST + WS API; wire-compat |
| **`homeassistant` Python client library** | Yes | Pure REST/WS client |
| **`ha-mqtt-discoverable` Python library** | Yes | Publishes MQTT discovery; HOMECORE consumes the same topics |
| **ESPHome devices** | Yes | ESPHome native API or MQTT; HOMECORE speaks both |
| **Nabu Casa Cloud** | **No** | Nabu Casa uses a proprietary remote-access tunnel to `nabucasa.com`. HOMECORE does not integrate with the Nabu Casa cloud proxy. Replace with Tailscale. |
| **M5Stack ATOM Echo / voice satellites** | Yes (P4) | Wyoming protocol is HOMECORE-ASSIST (ADR-133) scope |
| **HACS custom cards** | Yes (after ADR-131 P3) | Custom cards are served via the same `/hacsfiles/` static route |

### 6.2 What breaks and why

| HA feature | HOMECORE status | Reason |
|---|---|---|
| Nabu Casa remote access | Not supported | Proprietary tunnel; replace with Tailscale |
| HA Supervisor add-ons | Not supported | No container manager in HOMECORE |
| HAOS OTA updates | Not supported | HOMECORE runs on standard OS |
| Python custom integrations (non-WASM) | Not supported | WASM sandbox only; Python integrations cannot run natively |
| Legacy `setup_platform` integrations | Not supported | Config-flow (`async_setup_entry`) only |
| HA Cloud TTS/STT (Nabu Casa) | Not supported | Use Whisper + Piper locally |
| HA Cloud Alexa/Google skill | Not supported | Use ruflo agent instead |

---

## 7. Phase roadmap

```
Q3 2026    Q4 2026    Q1 2027    Q2 2027
   P1         P2         P3         P4         P5
scaffold   state+API  wire-compat  plugins+    full
           core       HA clients  automation  HOMECORE
```

### P1 — Scaffold (Q3 2026, 2 weeks)

- [ ] Create `v2/crates/homecore/` workspace member, empty state machine skeleton.
- [ ] Create `v2/crates/homecore-api/` skeleton, Axum server on port 8123 (HA default).
- [ ] Create `npm/homecore-frontend/` skeleton.
- [ ] CI: `cargo check -p homecore -p homecore-api --no-default-features` green.
- [ ] ADR-134 migration tool parses one `.storage/core.config_entries` fixture.

### P2 — State machine + API core (Q3 2026, 4 weeks)

- [ ] ADR-127 state machine: entity registry, state machine, event bus (Tokio broadcast), service registry.
- [ ] ADR-130 API: REST endpoints, WebSocket auth handshake, `subscribe_events`, `call_service`.
- [ ] ADR-132 recorder: SQLite schema (HA schema version 48 compatible), state write path.
- [ ] Integration test: HA companion app authenticates and receives state updates.

### P3 — Wire-compat + plugin scaffold (Q3–Q4 2026, 6 weeks)

- [ ] ADR-128 plugin system: WASM sandbox, manifest schema, first ported integrations (MQTT, HTTP).
- [ ] ADR-130 API: remaining WS commands, HACS support.
- [ ] ADR-134 migration: reads `automations.yaml`, `secrets.yaml`, config entries.
- [ ] ADR-132 recorder: ruvector dual-write, semantic search API.

### P4 — Automation + assist (Q4 2026, 4 weeks)

- [ ] ADR-129 automation engine: YAML parser, trigger evaluation, WASM expression evaluator.
- [ ] ADR-133 assist: ruflo agent orchestration, ruvector intent matching.
- [ ] ADR-131 frontend P1: TypeScript shell, WASM state machine in SharedWorker.

### P5 — Full HOMECORE (Q1 2027, 6 weeks)

- [ ] ADR-131 frontend: complete UI parity with HA Lovelace, custom cards.
- [ ] ADR-134 migration: side-by-side mode, one-click cutover.
- [ ] Full compatibility test suite against HA iOS/Android companion apps.
- [ ] Pi 5 performance benchmarks: startup < 1 s, idle < 50 MB RAM.

---

## 8. Alternatives rejected

### Alt-A: Contribute RuView sensing features upstream to Python HA

Add the HOMECORE features (WASM plugins, ruvector recorder, ruflo assist) as Python HA components via PRs to `home-assistant/core`.

**Rejected because**: HA's architecture board has strict policies against adding new runtimes (WASM, Rust FFI) to the core process. The GIL bottleneck cannot be resolved from within Python HA. CSI DSP at 100 Hz frame rate inside a Python process is not feasible. This path cedes architectural control permanently.

### Alt-B: Thin Rust wrapper that calls into Python HA via PyO3

Keep Python HA as the runtime; expose RuView sensing primitives via PyO3 bindings so they run at native speed inside the Python HA process.

**Rejected because**: the GIL is not resolved by PyO3 calls — the HA event loop still serialises all state changes. Startup time and memory footprint are unchanged. WASM plugin safety is unchanged. This is a tactical optimisation, not an architectural solution.

### Alt-C: OpenHAB or Domoticz as the base

Port RuView's sensing stack on top of an alternative hub (openHAB/Java, Domoticz/C++).

**Rejected because**: neither has HA's community network effects, companion app ecosystem, or HACS plugin catalog. A clean-room Rust implementation preserves the HA compatibility contract (the most valuable asset) without inheriting the Python runtime limitations.

### Alt-D: Extend the existing `wifi-densepose-sensing-server` into a full hub

Add automation, entity registry, and recorder features directly to the existing Axum sensing server.

**Rejected because**: the sensing server is a purpose-built single-concern binary (CSI → MQTT/WebSocket). Expanding it into a hub would violate the single-responsibility principle and couple hub release cycles to firmware release cycles. HOMECORE is a separate crate family that depends on but does not modify the sensing server.

---

## 9. Top-level risks

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| **API drift** — HA's REST/WS API evolves; HOMECORE must track it | High | High | Pin to HA 2025.1 baseline (schema 48); run the HA companion app integration tests against every HOMECORE release; ADR-130 owns the compat matrix |
| **WASM sandbox performance** — plugin calls through the WASM boundary add latency | Medium | Medium | Benchmark plugin roundtrip on Pi 5 before P3; reject if >5 ms; WASM3/Wasmtime both have sub-1 ms call overhead for compute-light integrations |
| **Core triad dependency** — ADR-128 and ADR-130 cannot start until ADR-127 is stable | High | High | ADR-127 is P2 start; freeze the state machine public API (entity_id, state, attributes, last_changed) before ADR-128 begins |
| **ruvector semantic recorder** — dual-write to SQLite + HNSW may impact write throughput under high-frequency sensing | Medium | High | ruvector writes are async (non-blocking tokio task); SQLite write is the hot path; benchmark at 100 state/s on Pi 5 before ADR-132 ships |
| **Nabu Casa gap** — users who depend on HA Cloud remote access have no HOMECORE replacement at P3 | High | Medium | Document Tailscale as the replacement prominently; provide ADR-134 migration wizard that detects Nabu Casa usage and offers Tailscale setup |
| **Frontend bundle size** — replicating the HA Lovelace card ecosystem in TS+WASM is a significant engineering effort | High | High | ADR-131 is off-critical-path; serve HA's Python frontend against the HOMECORE API until ADR-131 P3 ships |
| **License** — HA is Apache 2.0; the wire protocol is unencumbered; HA's UI assets and card components have separate licenses | Low | High | Clean-room Rust implementation does not use HA source; HA frontend is served as a binary (not embedded); review license before ADR-131 ships any reimplemented component |

---

## 10. Open questions

**Q1** (ADR-127): Should the HOMECORE state machine use a `DashMap<EntityId, State>` for lock-free concurrent reads, or a `RwLock<HashMap<EntityId, State>>` for simpler reasoning? The answer affects every integration's write pattern.

**Q2** (ADR-128): Does the WASM sandbox use Wasmtime (Cranelift JIT, ~5 MB binary) or WASM3 (interpreter, ~50 kB binary)? On a Pi 5 WASM3 is sufficient for integration logic; Wasmtime matters if integrations need near-native DSP speed.

**Q3** (ADR-130): The HA WebSocket API uses numeric IDs for command/response correlation. The HA 2025.1 baseline adds `subscribe_trigger` as a first-class WS command. Are there any commands in the HA companion app that require a newer baseline?

**Q4** (ADR-132): The ruvector HNSW index for state history — what embedding dimension represents a state snapshot? Options: (a) embed only numeric sensor states (scalar embedding), (b) embed `{entity_id, state, attributes}` as a text embedding via a local small model, (c) use a fixed schema encoding. The answer determines the semantic query fidelity.

**Q5** (ADR-134): HA's `.storage/core.config_entries` format is versioned but undocumented; it is hand-engineered from reverse-engineering the Python `StorageCollection` class in `homeassistant/helpers/storage.py`. Is this format stable enough to parse without upstream documentation, or does HOMECORE need to maintain a version matrix?

---

## 11. References

### This repo

- `docs/adr/ADR-115-home-assistant-integration.md` — HA-DISCO MQTT publisher; 21-entity surface; semantic primitives; competitive comparison table
- `docs/adr/ADR-116-cog-ha-matter-seed.md` — HA-COG Seed cog; cog packaging precedent (ADR-101)
- `docs/adr/ADR-117-pip-wifi-densepose-modernization.md` — PIP-PHOENIX PyO3 bindings; Python client surface
- `docs/adr/ADR-118-bfld-beamforming-feedback-layer-for-detection.md` — BFLD master; privacy class enforcement
- `docs/adr/ADR-124-rvagent-mcp-ruvector-npm-integration.md` — SENSE-BRIDGE; RUVIEW-POLICY §4.1a; multi-modal normalization §11.3
- `docs/adr/ADR-125-ruview-apple-home-native-hap-bridge.md` — APPLE-FABRIC HAP bridge
- `v2/crates/wifi-densepose-sensing-server/src/main.rs` — Axum server architecture; bearer auth pattern
- `v2/crates/wifi-densepose-ruvector/src/viewpoint/` — cross-viewpoint fusion (attention, coherence, geometry, fusion modules)
- `CLAUDE.md` — Project topology (hierarchical-mesh, 15 agents), ESP32 hardware table, crate publishing order

### HA upstream

- `homeassistant/core.py` — `HomeAssistant`, `StateMachine`, `EventBus`, `ServiceRegistry`, `Config`
- `homeassistant/helpers/entity_registry.py` — `EntityRegistry`, `RegistryEntry`
- `homeassistant/helpers/entity.py` — `Entity`, `async_write_ha_state`, entity lifecycle
- `homeassistant/components/api/__init__.py` — REST API handler (24 routes)
- `homeassistant/components/websocket_api/` — `connection.py` auth handshake; `commands.py` WS commands
- `homeassistant/components/recorder/` — SQLite schema; `migration.py` schema version 48
- `homeassistant/components/assist_pipeline/` — voice/text pipeline; Wyoming protocol
- `homeassistant/helpers/template.py` — Jinja2 template engine customisation
- `homeassistant/components/automation/__init__.py` — automation trigger/condition/action model
- `homeassistant/helpers/storage.py` — `.storage/*.json` persistence; `StorageCollection`
- `homeassistant/auth/` — long-lived access token model; `AuthManager`

### External

- [HA Developer Docs — Core Architecture](https://developers.home-assistant.io/docs/architecture/core/) — state machine, event bus, service registry overview
- [HA Developer Docs — WebSocket API](https://developers.home-assistant.io/docs/api/websocket/) — WS command catalog
- [DeepWiki HA core — Entity and Registry Management](https://deepwiki.com/home-assistant/core/2.2-entity-and-registry-management) — entity lifecycle
- [DeepWiki HA core — Data Management](https://deepwiki.com/home-assistant/core/3-data-management) — recorder schema version 48
- [HA recorder integration](https://www.home-assistant.io/integrations/recorder/) — SQLite default; schema migration overview
