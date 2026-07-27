# HOMECORE capability status

This document describes the current alpha runtime. “Implemented” means the
capability is wired into `homecore-server` or its public protocol crate and is
covered by tests. It does not imply compatibility with every Home Assistant
integration or every Apple Home controller.

## Runtime capabilities

| Area | Current behavior |
|---|---|
| Core | Concurrent entity state machine; entity and device registries; system/domain event buses; service registry; causal contexts |
| Restore | Entity registry, device registry, and the most recent recorder state are restored in dependency order at startup; malformed rows are isolated and bounded |
| Recorder | SQLite state history listener; deterministic latest-state queries; broadcast lag recovery; optional ruvector semantic index |
| Migration | Entity/device registries and config entries are parsed with version checks, unknown forward-compatible fields are preserved, and output is written atomically without overwriting existing storage |
| Plugins | Compiled-in native plugins use an explicit registry. Packaged Wasm plugins are discovered only in configured directories, bounded and path-checked, signature-verified against configured Ed25519 publishers, and run through Wasmtime when the `wasmtime` feature is enabled |
| Plugin lifecycle | Plugin setup runs after restoration and built-in service registration; state changes are dispatched through a bounded queue; teardown runs during graceful shutdown |
| Automation | State, numeric, event, and time triggers; optional YAML loading at server boot |
| Voice | Bounded PCM16 audio types, async STT/TTS provider contracts, an STT → intent → TTS pipeline, and an authenticated transport-independent satellite session protocol |
| Assist | Authenticated local intent endpoint with bounded regex recognition and service-backed handlers |
| HAP network | With `homecore-server --features hap-server`, an explicitly configured bounded TCP listener, persisted pairing records, live entity/accessory synchronization, and `_hap._tcp` mDNS lifecycle are wired into the server |
| Dashboard/BFF | Static UI, calibration proxy, rooms, COG list, appliance metrics, and typed unavailable responses for absent upstreams |

## Home Assistant-compatible REST surface

- `GET /api/`
- `GET /api/config`
- `GET /api/components`
- `GET /api/states`
- `GET|POST|DELETE /api/states/:entity_id`
- `GET /api/services`
- `POST /api/services/:domain/:service`
- `GET /api/events`
- `POST /api/events/:event_type`
- `POST /api/template`
- `POST /api/config/core/check_config`
- `GET /api/error_log`
- `GET /api/history/period[/<start_time>]`
- `GET /api/logbook[/<start_time>]`
- `GET /api/calendars[/:entity_id]`
- `GET /api/camera_proxy/:entity_id`
- `GET /api/websocket`
- `POST /api/intent/handle`
- `GET /api/homecore/compatibility`

The WebSocket server implements authentication, feature negotiation, ping,
configuration/state/service queries, service calls, event firing,
subscribe/unsubscribe, template rendering, panels, and entity/device/area
registry list commands. Per-connection output is bounded; lagging event
subscribers resynchronize.

`GET /api/homecore/compatibility` is the machine-readable support matrix.
HOMECORE implements the core contract above, not every endpoint contributed by
Home Assistant integrations. History is available when the recorder is
enabled. Calendar discovery and interval queries are implemented, with events
supplied only by calendar integrations. Camera routing is implemented and
returns a typed unavailable response when the entity has no image provider.
Media and Lovelace behavior remain integration-dependent.
Registry mutations require a persistent registry mutation backend. The area
registry currently returns a valid empty list.

## Security and deployment constraints

- `HOMECORE_TOKENS` is required unless insecure development authentication is
  explicitly enabled.
- Unsigned Wasm packages are rejected unless the development-only override is
  explicitly supplied. Arbitrary native dynamic libraries are never loaded;
  native plugins must be linked into the binary and registered in code.
- HAP is disabled by default. Enabling it requires an explicit bind address,
  stable six-octet device identifier, advertised LAN address, hostname, and
  durable pairing-store path.
- The HAP listener fails closed for cryptographic phases that are unavailable
  in the selected build. Do not claim Apple Home interoperability unless the
  Pair-Setup, Pair-Verify, and encrypted transport conformance tests for that
  build pass.
- STT and TTS are provider contracts; a deployment must supply real providers.
  The built-in disabled providers return typed errors rather than fabricated
  speech results.
- Synthetic biometric/demo entities are opt-in and must not be interpreted as
  live sensing data.
