# ADR-130: HOMECORE-API — Wire-compatible REST and WebSocket API

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-API** |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE), [ADR-055](ADR-055-integrated-sensing-server.md) (sensing-server Axum pattern), [ADR-124](ADR-124-rvagent-mcp-ruvector-npm-integration.md) (SENSE-BRIDGE — bearer auth pattern) |
| **Tracking issue** | TBD |

---

## 1. Context

Home Assistant's HTTP and WebSocket APIs are the primary interface for every non-frontend client: the iOS companion app, the Android companion app, HACS, Node-RED, the `homeassistant` Python client library, ESPHome native API clients, external automation scripts, and the hundreds of third-party HA dashboard projects.

The API surface is defined in two Python modules:

1. **`homeassistant/components/api/__init__.py`** — 24 REST API routes mounted at `/api/`. Key routes: `GET /api/`, `GET /api/states`, `GET /api/states/<entity_id>`, `POST /api/states/<entity_id>`, `GET /api/events`, `POST /api/events/<event_type>`, `GET /api/services`, `POST /api/services/<domain>/<service>`, `GET /api/error_log`, `GET /api/config`, `POST /api/template`, `POST /api/check_config`, `GET /api/history/period/<datetime>` (deprecated — recorder), `POST /api/logbook/` (deprecated — recorder).

2. **`homeassistant/components/websocket_api/`** — the WebSocket API handler (`connection.py` handles auth handshake; `commands.py` handles 30+ command types). Key commands: `auth`, `subscribe_events`, `unsubscribe_events`, `call_service`, `get_states`, `get_services`, `get_config`, `subscribe_trigger`, `render_template`, `validate_config`, `subscribe_entities` (entity registry updates), `config/entity_registry/list`, and many more.

### 1.1 Auth model

HA uses **long-lived access tokens (LLAT)** as the primary auth mechanism for non-UI clients. Tokens are created in the HA user profile UI and stored in `.storage/auth`. The REST API accepts `Authorization: Bearer <token>` or the `api_password` legacy header (deprecated since HA 2022.x). The WebSocket API requires an `auth` message with `access_token` as the first message after connection.

### 1.2 Why wire-compat matters

The iOS and Android HA companion apps (>100,000 installs combined) hardcode the HA API paths and WebSocket command schemas. Any implementation that deviates from the exact JSON schemas causes the apps to fail silently — not with a meaningful error, but by returning empty entity lists or missing state updates. Wire-compat is therefore a hard requirement, not a nice-to-have.

The baseline for compatibility is **HA 2025.1** (the version that introduced SQLite recorder schema version 48). Any HOMECORE instance claiming compliance with this ADR must pass the companion app integration test suite.

---

## 2. Decision

Implement the `homecore-api` crate as an Axum-based server that replicates the HA REST and WebSocket API on port 8123. The implementation is informed by — but does not copy — `homeassistant/components/api/__init__.py` and `homeassistant/components/websocket_api/`.

The server reuses the Axum + Tokio architecture established in `v2/crates/wifi-densepose-sensing-server/src/main.rs` and its bearer auth pattern (`v2/crates/wifi-densepose-sensing-server/src/bearer_auth.rs`).

### 2.1 REST API route table

| Route | Method | HA source line (approx.) | HOMECORE status |
|---|---|---|---|
| `/api/` | GET | `api/__init__.py:74` | P2 — returns `{ "message": "API running." }` |
| `/api/config` | GET | `api/__init__.py:97` | P2 — returns `homecore.config` as JSON |
| `/api/states` | GET | `api/__init__.py:116` | P2 — returns `hass.states.all()` as JSON array |
| `/api/states/<entity_id>` | GET | `api/__init__.py:130` | P2 |
| `/api/states/<entity_id>` | POST | `api/__init__.py:145` | P2 — writes state; fires `state_changed` |
| `/api/events` | GET | `api/__init__.py:168` | P3 |
| `/api/events/<event_type>` | POST | `api/__init__.py:180` | P3 — fires domain event |
| `/api/services` | GET | `api/__init__.py:192` | P2 |
| `/api/services/<domain>/<service>` | POST | `api/__init__.py:206` | P2 |
| `/api/template` | POST | `api/__init__.py:222` | P3 — WASM MiniJinja evaluator (ADR-129) |
| `/api/check_config` | POST | `api/__init__.py:240` | P4 |
| `/api/error_log` | GET | `api/__init__.py:252` | P3 |
| `/api/history/period/<datetime>` | GET | `api/__init__.py:270` | P4 — recorder query (ADR-132) |
| `/api/logbook/` | POST | `api/__init__.py:310` | P4 — recorder query |
| `/api/camera_proxy/<entity_id>` | GET | `api/__init__.py:330` | P4 — proxy to camera integration |
| `/api/calendar/<entity_id>` | GET | `api/__init__.py:348` | P4 |
| `/api/webhook/<webhook_id>` | POST/GET | `api/__init__.py:368` | P3 — fires `webhook.<id>` event |
| `/api/intent/handle` | POST | `api/__init__.py:400` | P4 — HOMECORE-ASSIST (ADR-133) |
| `/auth/token` | POST | `auth/providers/__init__.py` | P2 — issue LLAT from username/password |
| `/auth/authorize` | GET/POST | `auth/providers/__init__.py` | P3 — OAuth2 flow |
| `/frontend/` static assets | GET | `frontend/__init__.py` | P1 — serve HA Python frontend static files until ADR-131 ships |

### 2.2 WebSocket API command table

| WS command type | HA source | HOMECORE status |
|---|---|---|
| `auth` (handshake) | `websocket_api/connection.py:55` | P2 |
| `subscribe_events` | `websocket_api/commands.py:120` | P2 |
| `unsubscribe_events` | `websocket_api/commands.py:145` | P2 |
| `call_service` | `websocket_api/commands.py:160` | P2 |
| `get_states` | `websocket_api/commands.py:200` | P2 |
| `get_services` | `websocket_api/commands.py:218` | P2 |
| `get_config` | `websocket_api/commands.py:230` | P2 |
| `subscribe_trigger` | `websocket_api/commands.py:250` | P3 |
| `render_template` | `websocket_api/commands.py:280` | P3 |
| `validate_config` | `websocket_api/commands.py:300` | P3 |
| `subscribe_entities` | `websocket_api/commands.py:320` | P3 — entity registry update stream |
| `config/entity_registry/list` | `websocket_api/commands.py:370` | P3 |
| `config/entity_registry/update` | `websocket_api/commands.py:400` | P3 |
| `config/area_registry/list` | `websocket_api/commands.py:450` | P3 |
| `config/device_registry/list` | `websocket_api/commands.py:480` | P3 |
| `config/config_entries/list` | `websocket_api/commands.py:510` | P3 |
| `lovelace/config` (dashboard) | `lovelace/dashboard.py` | P4 — reads from HOMECORE storage |
| `media_player/*` | `websocket_api/commands.py:600` | P4 |

### 2.3 Auth implementation

HOMECORE-API implements long-lived access tokens as JWTs signed with an Ed25519 key (generated at first startup, stored in `.homecore/auth_key.pem`). Token format:

```json
{
  "sub": "<user_id>",
  "iss": "homecore",
  "iat": <unix_timestamp>,
  "exp": <unix_timestamp or null for LLAT>,
  "type": "long_lived_access_token"
}
```

The HA companion app sends `Authorization: Bearer <token>` on every REST request. The WebSocket auth handshake sends `{ "type": "auth", "access_token": "<token>" }`. Both paths validate the JWT against the stored Ed25519 key.

Legacy `api_password` is deliberately not supported (removed in HA 2022.x and never properly secure).

---

## 3. HA-side reference table

| HA module / file | What it does | HOMECORE preserves | Changes | Drops |
|---|---|---|---|---|
| `components/api/__init__.py` | 24 REST routes + JSON response schemas | All response schemas byte-compatible with HA 2025.1 | Axum router instead of HA's custom HTTP component; `serde_json` instead of Python `json` | Python HTTP request context; HA's built-in CORS middleware (replicated in Axum) |
| `components/websocket_api/connection.py` | WS auth handshake; per-connection state; message dispatch | Auth handshake flow: `auth_required` → `auth` message → `auth_ok` or `auth_invalid` | Axum `WebSocketUpgrade` extractor; per-connection `tokio::task` | Python asyncio message handling |
| `components/websocket_api/commands.py` | 30+ WS command handlers | All command type strings; response envelope `{ id, type, result }` or error `{ id, type, error: { code, message } }` | Rust match dispatch; Tokio broadcast receiver per subscription | Python class-based command handler registration |
| `auth/providers/__init__.py` | Auth providers; LLAT issuance; OAuth2 flow | LLAT issuance; token validation | Ed25519 JWT instead of HA's custom token serializer; same token `type` field values | Nabu Casa cloud auth; multi-provider auth chain |
| `components/http/__init__.py` | Aiohttp-based HTTP server setup; CORS; trusted proxies | CORS headers; `X-Forwarded-For` trusted proxy handling | Axum Tower middleware | Aiohttp; Python SSL context |

---

## 4. Public API parity table

| HA API surface | HOMECORE exact equivalent |
|---|---|
| `GET /api/states` → `[{entity_id, state, attributes, last_changed, last_updated, context}]` | Identical JSON schema; `last_changed` / `last_updated` in ISO 8601 |
| `GET /api/services` → `{domain: {service: {description, fields}}}` | Identical schema; service descriptions read from plugin manifests |
| WS `subscribe_events` → `{type: "event", event: {event_type, data, origin, time_fired, context}}` | Identical envelope; `time_fired` in ISO 8601 |
| WS `call_service` → `{type: "result", success: true, result: {context}}` | Identical; `context.id` is a UUID |
| WS `get_states` → `{type: "result", result: [{entity_id, state, attributes, ...}]}` | Identical schema |
| REST `POST /api/services/<domain>/<service>` → 200 with called service list | Identical; same `target` field support |
| REST `POST /api/template` → 200 with evaluated string | Identical; same error response `{message: "..."}` on template error |
| Auth WS flow: `auth_required` → `auth` → `auth_ok` | Identical message type strings; same `ha_version` field in `auth_required` |
| REST `Authorization: Bearer <token>` | Identical header name; JWT instead of HA's opaque token format (transparent to clients) |

---

## 5. Phased implementation plan

### P1 — Axum skeleton + static frontend (1 week)

- [ ] Create `v2/crates/homecore-api/` workspace member.
- [ ] Axum router on port 8123; Tower CORS middleware (allow `http://homeassistant.local:8123`).
- [ ] Static file handler: serve HA's Python frontend build from a configurable path (default `./frontend/build/`). This allows using the Python HA frontend as-is until ADR-131 ships.
- [ ] `GET /api/` returns `{ "message": "API running." }`.
- [ ] CI: `cargo check -p homecore-api`; HTTP smoke test.

### P2 — Core REST + WebSocket auth + states (3 weeks)

- [ ] Axum WebSocket upgrade at `/api/websocket`.
- [ ] Auth: Ed25519 JWT issuance at `/auth/token`; validation middleware.
- [ ] WS auth handshake: `auth_required` → `auth` → `auth_ok` / `auth_invalid`.
- [ ] WS commands: `get_states`, `subscribe_events`, `unsubscribe_events`, `call_service`, `get_services`, `get_config`.
- [ ] REST: `/api/states`, `/api/states/<entity_id>` (GET + POST), `/api/services`, `/api/services/<domain>/<service>`, `/api/config`.
- [ ] Integration test: HA iOS companion app authenticates and displays entity list against HOMECORE.

### P3 — Remaining WS commands + entity registry API (3 weeks)

- [ ] WS: `subscribe_trigger`, `render_template`, `validate_config`, `subscribe_entities`, entity/area/device registry commands.
- [ ] REST: `/api/template`, `/api/webhook/<id>`, `/api/error_log`, `/api/events`, `/api/events/<type>`.
- [ ] `/auth/authorize` OAuth2 flow for UI login.
- [ ] HACS smoke test: HACS connects, lists integrations.

### P4 — Recorder + history API (2 weeks)

- [ ] `/api/history/period/<datetime>` backed by ADR-132 recorder SQLite.
- [ ] `/api/logbook/` backed by ADR-132 recorder.
- [ ] `/api/camera_proxy/`, `/api/calendar/`, `/api/intent/handle`.
- [ ] Companion app full feature test: automations, notifications, history charts.

---

## 6. Risks

| Risk | Likelihood | Severity | Mitigation | Cross-ADR impact |
|---|---|---|---|---|
| **JSON schema drift** — HA updates a response field name between 2025.1 and HOMECORE release | Medium | High | Maintain a JSON-schema test fixture set generated from HA 2025.1; run against HOMECORE in CI | ADR-134: migration tool depends on the same JSON schemas; must stay in sync |
| **WS subscription fan-out** — 50 concurrent HA companion app sessions each subscribed to `subscribe_events` ALL; every state change creates 50 serialization tasks | Medium | Medium | Broadcast serialized JSON once; clone the `Bytes` arc to each subscriber sender; do not re-serialize per subscriber | ADR-127: broadcast channel capacity must handle subscriber fan-out without lagging |
| **Auth token format** — HA companion apps may validate the token format (JWT vs opaque). HOMECORE uses JWT; HA uses a custom opaque token. Tokens are never decoded client-side in standard clients, but non-standard clients may inspect them | Low | Low | JWTs are base64url-encoded JSON; any client checking `token.startsWith("ey")` will see a JWT. HA's own tokens are also base64url but not JWTs. Document the difference; test with the iOS app specifically | None |
| **Port 8123 conflict** — HOMECORE runs on the same port as HA; side-by-side mode (ADR-134) requires HOMECORE on a different port until cutover | High | Medium | ADR-134 side-by-side mode runs HOMECORE on port 8124; companion app can be pointed at port 8124 for testing | ADR-134 owns the cutover mechanism |

---

## 7. Open questions

**Q1**: The HA WebSocket API uses incremental integer IDs (`id: 1, 2, 3, ...`) for command/response correlation within a session. HOMECORE uses the same scheme. What is the maximum `id` value the companion app supports before wrapping? If the app doesn't wrap and HOMECORE processes > 2^31 commands per session, this becomes an overflow issue in extremely long-lived sessions.

**Q2**: The `subscribe_entities` WS command (added in HA 2021.x) sends entity registry change events in addition to state change events. The iOS companion app uses this to maintain a local entity list without polling. Is the full `subscribe_entities` delta schema (including `action: "create" | "update" | "remove"`) fully documented, or must it be reverse-engineered from the companion app source?

**Q3**: HA's `/auth/token` endpoint accepts `grant_type=password` (username/password) and `grant_type=refresh_token`. HOMECORE's initial implementation supports password grant only. Is refresh token support required for the companion app (it caches tokens between sessions) or does the companion app re-authenticate on each launch?

**Q4**: CORS policy: HA's default CORS allows `http://localhost:*` and `http://homeassistant.local:*`. The HOMECORE-UI frontend (ADR-131) will be served from a different origin in development. What CORS policy should HOMECORE-API use in production vs development mode?

---

## 8. References

### HA upstream

- `homeassistant/components/api/__init__.py` — 24 REST routes with exact URL paths, methods, and JSON response schemas
- `homeassistant/components/websocket_api/connection.py` — auth handshake protocol; per-connection state management
- `homeassistant/components/websocket_api/commands.py` — 30+ command type handlers with exact type strings and result schemas
- `homeassistant/components/http/__init__.py` — CORS setup; trusted proxy handling; aiohttp-based server
- `homeassistant/auth/providers/__init__.py` — token issuance; `AuthManager`; LLAT format
- `homeassistant/auth/__init__.py` — `AuthManager.async_create_long_lived_access_token`

### This repo

- `v2/crates/wifi-densepose-sensing-server/src/main.rs` — Axum server architecture (REST + WebSocket); pattern for this ADR
- `v2/crates/wifi-densepose-sensing-server/src/bearer_auth.rs` — Bearer auth middleware pattern
- `docs/adr/ADR-127-homecore-state-machine-rust.md` — state machine that REST/WS routes read from
- `docs/adr/ADR-126-ruview-native-ha-port-master.md` — §6 compatibility contract with companion apps

### External

- [HA WebSocket API Developer Docs](https://developers.home-assistant.io/docs/api/websocket/) — authoritative command type catalog
- [HA REST API](https://developers.home-assistant.io/docs/api/rest/) — REST endpoint schemas
