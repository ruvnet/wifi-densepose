# homecore-server

Integrated HOMECORE server binary that wires state machine, API, recorder, plugins, automations, intent assistant, and HomeKit bridge into one process.

[![Crates.io](https://img.shields.io/badge/crates.io-workspace%20binary-inactive)](.)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MSRV: 1.89+](https://img.shields.io/badge/MSRV-1.89%2B-purple.svg)
[![ADR-126](https://img.shields.io/badge/ADR-126-orange.svg)](../../docs/adr/ADR-126-homecore-home-assistant-port.md)

The production-ready HOMECORE binary — boots all 7 subsystems (core, API, recorder, plugins, automation, assist, HAP bridge) in a single process listening on `:8123`.

## What this crate does

`homecore-server` is the integration point for the entire HOMECORE ecosystem. It orchestrates:

1. **HomeCore runtime** — state machine, event bus, service registry
2. **REST + WebSocket API** — Axum server on `:8123` (HA-compatible)
3. **SQLite Recorder** — persists all state changes to disk
4. **Plugin Registry** — loads and manages integrations (InProcessRuntime by default)
5. **Automation Engine** — evaluates triggers, conditions, and actions
6. **Assist Pipeline** — intent recognition and execution
7. **HAP Bridge** — exposes accessories to HomeKit

All subsystems share the same `HomeCore` instance, so state changes flow through the event bus and trigger automations, record history, and notify WebSocket subscribers in lockstep.

## Features

- **Single unified process** — no external microservices; run with `cargo run -p homecore-server`
- **HA-compatible REST API** — drop-in replacement for Home Assistant's `/api/` on `:8123`
- **SQLite state history** — persistent recording of all state changes
- **Automation engine** — YAML-driven trigger→condition→action execution
- **Intent assistant** — regex-based (P1) intent recognition + service calling
- **HomeKit bridge** — exposes HOMECORE entities as HomeKit accessories
- **Plugin system** — load first-party Rust plugins; Wasmtime WASM plugins (P2, `--features wasmtime`)
- **Configurable via CLI + env vars** — no YAML required; sensible defaults
- **Structured logging** — tracing output with `RUST_LOG` filtering
- **Feature-gated subsystems** — disable recorder (`--no-recorder`), enable ruvector/wasmtime as needed

## Subsystems

| Subsystem | Crate | Role | Notes |
|-----------|-------|------|-------|
| State Machine | `homecore` | Core domain model | All other subsystems depend on this |
| REST API | `homecore-api` | HTTP boundary | Listens on `:8123`; Axum framework |
| Recorder | `homecore-recorder` | Persistence | SQLite; optional `--no-recorder` |
| Plugins | `homecore-plugins` | Extension system | InProcessRuntime default; Wasmtime w/ feature |
| Automation | `homecore-automation` | Trigger execution | Subscribes to event bus; YAML-driven |
| Assist | `homecore-assist` | Intent pipeline | Regex recognizer (P1); semantic (P2) |
| HAP Bridge | `homecore-hap` | HomeKit export | Accessories + characteristics; mDNS (P2) |

## Usage

**Basic startup** (in-memory recorder):

```bash
cargo build -p homecore-server
./target/debug/homecore-server
# Listens on http://localhost:8123
```

**With persistent SQLite**:

```bash
./target/debug/homecore-server \
  --bind 0.0.0.0:8123 \
  --db sqlite:~/.homecore/home.db \
  --location-name "My Home"
```

**Full feature build** (ruvector semantic search + Wasmtime plugins):

```bash
cargo build -p homecore-server --features ruvector,wasmtime --release
```

**Via Docker** (Dockerfile planned P2):

```bash
docker run -p 8123:8123 \
  -e HOMECORE_DB=sqlite:///data/home.db \
  -v ~/.homecore:/data \
  homecore-server:latest
```

**Test the API**:

```bash
# List all entities
curl http://localhost:8123/api/states

# Set a light to "on"
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"state":"on","attributes":{"brightness":200}}' \
  http://localhost:8123/api/states/light.kitchen

# WebSocket subscription (real-time state changes)
wscat -c ws://localhost:8123/api/websocket
```

**Configuration via env**:

```bash
export HOMECORE_BIND="0.0.0.0:8123"
export HOMECORE_DB="sqlite:~/.homecore/home.db"
export HOMECORE_LOCATION="Living Room"
export RUST_LOG="homecore=debug,homecore_api=info"
./target/debug/homecore-server
```

## CLI Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `--bind` | `HOMECORE_BIND` | `0.0.0.0:8123` | REST API listen address |
| `--db` | `HOMECORE_DB` | `sqlite::memory:` | SQLite path (`:memory:` for ephemeral) |
| `--location-name` | `HOMECORE_LOCATION` | `Home` | Friendly name returned by `/api/config` |
| `--no-recorder` | — | off | Disable SQLite recorder (low-resource deployments) |
| `--ui-dir` | `HOMECORE_UI_DIR` | `<crate>/ui` | HOMECORE-UI asset dir served at `/homecore` (ADR-131); empty disables the mount |

## HOMECORE-UI dashboard (ADR-131)

This binary also serves the **HOMECORE-UI** — the complete operational dashboard
for the two-tier Cognitum stack (v0 Appliance → SEEDs → ESP32 nodes) — at
`/homecore`, alongside the HA-compat `/api` surface. It is a zero-dependency,
no-build-step vanilla TS/JS + CSS frontend living in `ui/`:

```bash
cargo run -p homecore-server          # then open http://localhost:8123/homecore/
```

It drives the live `/api` + `/api/websocket` (`subscribe_events`) endpoints; panels
backed by services not in this binary (SEED HTTPS API, calibration ADR-151,
federation ADR-105) render against a DEMO-flagged contract-conformant mock until
those endpoints land (ADR-131 §7.1). Frontend tests + benchmark run under plain
`node` (no `npm install`):

```bash
cd ui && npm test     # import graph + render-smoke + interaction (24 checks)
cd ui && npm run bench # bundle budget (~137 KB, ~37× smaller than HA) + render timing
```

## Comparison to Home Assistant

| Aspect | Home Assistant | homecore-server |
|--------|----------------|-----------------|
| Architecture | Python asyncio monolith | Rust async Tokio + component traits |
| API protocol | `/api/` REST (HA wire format) | Identical HA wire format |
| Persistence | SQLite + YAML files | SQLite (P1); Redis (P2) |
| Plugins | Python integrations in `homeassistant/components/` | Rust (P1) + WASM (P2) |
| Automation execution | Python asyncio event loop | Tokio async tasks + trait-based |
| HomeKit bridge | Via `homekit` integration | Built-in `homecore-hap` subsystem |
| CLI | `hass` command with config YAML | `homecore-server` with feature flags |
| Scalability | Single instance (HA Cloud for scale) | Can be load-balanced (future) |
| Binary size | ~200 MB (Python + deps) | ~50 MB (Rust, release build; 200 MB w/ wasmtime) |

## Performance Targets (unreleased; TBD)

- **Startup time** — < 2s to listen on `:8123`
- **REST endpoint latency** — p50 < 1 ms; p99 < 10 ms
- **Event bus throughput** — 10,000+ events/sec
- **Automation evaluation** — < 100 μs per trigger
- **Concurrent WebSocket connections** — 10,000+
- **Memory footprint** — ~100 MB (idle); ~500 MB with 1,000 recorded states

## Development

**Run tests**:

```bash
cargo test -p homecore-server
```

**Enable debug logging**:

```bash
RUST_LOG=debug cargo run -p homecore-server -- --bind 127.0.0.1:8123
```

**Build documentation**:

```bash
cargo doc -p homecore-server --open
```

## Relation to other HOMECORE crates

```
homecore-server (orchestration binary)
├── homecore (state machine)
├── homecore-api (REST + WS)
├── homecore-recorder (SQLite persistence)
├── homecore-plugins (extension system)
├── homecore-automation (trigger execution)
├── homecore-assist (intent pipeline)
└── homecore-hap (HomeKit bridge)
```

## References

- [ADR-126: HOMECORE Home Assistant Port (master)](../../docs/adr/ADR-126-homecore-home-assistant-port.md)
- [README — wifi-densepose](../../../README.md)
- [Dockerfile (planned P2)](Dockerfile.planned)
- [Docker Hub image (planned P2)](https://hub.docker.com/r/ruvnet/homecore-server)
