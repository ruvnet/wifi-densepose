# HOMECORE capability status

This document describes the implemented runtime surface as of the current
alpha. “Implemented” means wired into `homecore-server` and covered by tests;
workspace libraries alone are not presented as server capabilities.

## Implemented

| Area | Runtime behavior |
|---|---|
| Core | Concurrent entity state machine, entity registry API, shared system/domain event bus, service registry, contexts |
| Event flow | Committed state changes reach state subscribers and the shared event bus; service calls emit system events |
| REST | API root, config, list/get/set/delete state, list/call service |
| WebSocket | Auth handshake, ping, config/states/services queries, service calls, event subscribe/unsubscribe |
| Backpressure | Per-connection WebSocket output is bounded to 256 messages; slow clients are disconnected from overflowing subscriptions |
| Authentication | Token allowlist from `HOMECORE_TOKENS`; missing tokens fail startup unless insecure development mode is explicitly enabled |
| Recorder | SQLite state history listener; broadcast lag is recovered by resynchronizing current state; optional ruvector semantic index |
| Automation | State, numeric, event, and time triggers; optional YAML loading at server boot |
| Assist | Authenticated `POST /api/intent/handle` with bounded regex recognition and five local handlers |
| Built-in services | Real state mutation for `homeassistant`, `light`, and `switch` on/off/toggle; ping and state snapshot |
| Migration | HA entity registry v1/minor 1–13 parsing and atomic, no-overwrite persistence into a HOMECORE storage directory |
| Dashboard/BFF | Static UI, calibration proxy, rooms, COG list, appliance metrics, and typed unavailable responses for absent upstreams |

## Exact HA-style HTTP surface

- `GET /api/`
- `GET /api/config`
- `GET /api/states`
- `GET|POST|DELETE /api/states/:entity_id`
- `GET /api/services`
- `POST /api/services/:domain/:service`
- `GET /api/websocket`
- `POST /api/intent/handle`

This is a compatible subset, not full Home Assistant parity.

## Explicitly deferred

- Loading native or Wasmtime plugins from `homecore-server`
- A network HomeKit Accessory Protocol server, pairing, and mDNS advertisement
- Device-registry persistence and full config-entry conversion
- Full HA event/history/template/config/check endpoints
- Restore-state on server startup
- STT/TTS and satellite voice protocols
- SEED/federation/witness/privacy upstream services when their daemons are absent
- Claiming Home Assistant integration parity or production-scale performance

Unavailable BFF upstreams return typed `503 upstream_unavailable`. Unsupported
services are not registered. Synthetic biometric/demo entities are opt-in and
must not be interpreted as live sensing data.
