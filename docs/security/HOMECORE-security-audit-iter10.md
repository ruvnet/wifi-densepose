# HOMECORE Security Audit — Iter-10

**Branch**: `feat/adr-126-homecore-impl`  
**Audit date**: 2026-05-25  
**Scope**: 8 new crates + integration binary (iter-1 through iter-9)  
**Auditor**: Security-audit agent (claude-sonnet-4-6)

---

## Executive Summary

HOMECORE's Rust codebase is structurally sound but ships with two pre-production
placeholders that are critical blockers for any production deployment: the HTTP
bearer-token validator accepts **any non-empty string as a valid token**, and the
WebSocket auth handshake does the same. Every protected endpoint is therefore fully
open to unauthenticated attackers who can reach port 8123.

`cargo audit` flagged **18 advisories** across three dependency trees. Two are
Critical (CVSS 9.0): both are Wasmtime sandbox-escape bugs in the Winch and
Cranelift compiler backends (RUSTSEC-2026-0095/0096). SQLx 0.7.4 carries a
binary-protocol misinterpretation bug (RUSTSEC-2024-0363). The Wasmtime
version must be upgraded before any WASM plugin is loaded in production.

Additional findings: `CorsLayer::permissive()` allows cross-origin requests from
any domain; the HAP service record hardcodes a predictable setup code and a
broadcast MAC address; `hc_log` writes plugin output directly to `eprintln!`
without going through `tracing`; and the WS `subscribe_events` command has no
per-connection subscription cap, enabling a resource-exhaustion DoS.

---

## Findings

| ID | Severity | Title | File : Line | Description | Remediation |
|----|----------|-------|-------------|-------------|-------------|
| HC-01 | **Critical** | Bearer auth accepts any non-empty token (REST) | `homecore-api/src/auth.rs:25` and `rest.rs` (all handlers) | `BearerAuth::from_headers` returns `Ok` for any non-empty string. All REST endpoints (`/api/config`, `/api/states`, `/api/services`, `call_service`) are fully open to any caller. | Implement a token store in P2 before deployment. Until then, enforce network-level ACL so port 8123 is unreachable from untrusted networks. |
| HC-02 | **Critical** | WebSocket auth handshake accepts any non-empty token | `homecore-api/src/ws.rs:61–68` | The WS `auth` phase validates only that `access_token` is non-empty. After passing this check the client reaches the full command loop including `call_service`. An attacker sending `{"type":"auth","access_token":"x"}` gets a fully authenticated session. | Same as HC-01; block at network until real token store is wired. |
| HC-03 | **Critical** | Wasmtime 25.0.3 — sandbox-escape via Winch backend (RUSTSEC-2026-0095) | `homecore-plugins/Cargo.toml` | The Winch compiler backend in Wasmtime 25.0.3 allows a sandboxed WASM plugin to perform out-of-sandbox memory writes (CVSS 9.0). | Upgrade `wasmtime` to `>=36.0.7` or `>=42.0.2`. |
| HC-04 | **Critical** | Wasmtime 25.0.3 — sandbox-escape via miscompiled heap access on aarch64 Cranelift (RUSTSEC-2026-0096) | `homecore-plugins/Cargo.toml` | Miscompiled guest heap access in Cranelift's aarch64 backend enables sandbox escape (CVSS 9.0). Production Pi 5 targets are aarch64. | Upgrade `wasmtime` to `>=36.0.7` or `>=42.0.2`. |
| HC-05 | **High** | `CorsLayer::permissive()` allows all cross-origin requests | `homecore-api/src/app.rs:25` | `CorsLayer::permissive()` sets `Access-Control-Allow-Origin: *` and allows all methods and headers. Any webpage on any origin can make authenticated API calls using a stored bearer token (when HC-01/02 are fixed). | Replace with an explicit allowlist: `CorsLayer::new().allow_origin(expected_origin).allow_methods([GET, POST])`. |
| HC-06 | **High** | SQLx 0.7.4 — binary protocol misinterpretation (RUSTSEC-2024-0363) | `homecore-recorder/Cargo.toml` | Truncating/overflowing casts in SQLx 0.7.4's binary protocol handling can cause values to be misread. Although HOMECORE only uses SQLite (not MySQL/Postgres), the vulnerable codepath is in the shared crate. | Upgrade `sqlx` to `>=0.8.1`. |
| HC-07 | **High** | No per-connection subscription cap on WS `subscribe_events` | `homecore-api/src/ws.rs:237–295` | A single authenticated WS connection can call `subscribe_events` in an unbounded loop. Each subscription spawns a Tokio task and takes one broadcast receiver slot. With the bus capacity at 4096 slots, a malicious client can exhaust OS thread/task resources before the bus fills. | Add a per-connection subscription ceiling (e.g., 50). Reject further `subscribe_events` commands with `"too_many_subscriptions"`. |
| HC-08 | **High** | Hardcoded HAP setup code and broadcast MAC in production binary | `homecore-server/src/main.rs:113–114`, `homecore-hap/src/bridge.rs:143–144` | The integration binary hard-codes `setup_code: "123-45-678"` and `device_id: "AA:BB:CC:DD:EE:FF"`. When real HAP pairing lands in P2 any attacker on the local network can pair with the bridge using the published setup code; the broadcast MAC address is also invalid per the HAP specification. | Generate a random setup code and a locally administered unicast MAC at startup (or require them as CLI arguments). Never use a known-fixed setup code. |
| HC-09 | **Medium** | Wasmtime 25.0.3 — 11 additional medium/low CVEs | `homecore-plugins/Cargo.toml` | RUSTSEC-2025-0046, -0118, -2026-0020, -0021, -0085, -0086, -0087, -0088, -0089, -0091, -0092, -0093, -0094 affect resource exhaustion, host data leakage, OOB reads/writes, and panics. All are fixed in wasmtime `>=36.0.7`. | Same fix as HC-03/04: upgrade wasmtime. |
| HC-10 | **Medium** | `hc_log` writes plugin output via `eprintln!` bypassing structured logging | `homecore-plugins/src/wasmtime_runtime.rs:297` | Plugin log messages are written directly to stderr via `eprintln!`, bypassing the `tracing` subscriber. This means: (a) log level filtering does not apply to plugin output; (b) log aggregation pipelines (e.g., JSON structured logs) miss plugin messages. A verbose or malicious plugin can flood stderr. | Replace `eprintln!` with `tracing::debug!/info!/warn!/error!` using the already-imported `LogLevel`. |
| HC-11 | **Medium** | No size bound on `set_state` body or `attributes` JSON | `homecore-api/src/rest.rs:95–108`, `ws.rs:222–235` | `POST /api/states/:entity_id` and the WS `call_service` / `get_states` paths accept a `serde_json::Value` body with no size limit beyond Axum's default (2 MB). Specially crafted deeply-nested JSON can cause quadratic parse time or high-memory allocation during serialization. | Apply `axum::extract::DefaultBodyLimit::max(65536)` on the route or globally; validate JSON depth before accepting. |
| HC-12 | **Medium** | `rsa 0.9.10` — Marvin Attack timing side-channel (RUSTSEC-2023-0071) | transitive via `sqlx-mysql 0.7.4` | The `rsa` crate's decryption is vulnerable to timing-based key recovery. Pulled in by `sqlx-mysql` even though HOMECORE only uses SQLite. No fix is available upstream. | Add `sqlx` features `sqlite` only (remove `mysql`/`postgres` from the feature list) to avoid pulling in `sqlx-mysql` and the `rsa` transitive dependency. |
| HC-13 | **Medium** | `shlex 0.1.1` — shell-injection via quote API (RUSTSEC-2024-0006) | transitive via `wasm3-sys 0.3.0 → wasm3 0.3.1 → homecore-plugins` | `shlex`'s quote function can produce unsafe shell strings. Pulled in by the `wasm3` build system. Not directly callable from HOMECORE Rust code but present in the binary's dependency tree. | Upgrade `shlex` to `>=1.3.0` or drop the `wasm3` dependency if `WasmtimeRuntime` is the production path. |
| HC-14 | **Low** | No TLS on the HTTP/WS listener | `homecore-server/src/main.rs:122–128` | The Axum listener binds plain TCP (`axum::serve`). Bearer tokens and all home automation data are transmitted in cleartext. On LAN deployments an attacker with ARP poisoning can intercept credentials. | Add `rustls`/`axum-server` TLS termination or document that a TLS-terminating reverse proxy (nginx/Caddy) is required. |
| HC-15 | **Low** | Migration CLI performs no symlink/traversal check on `.storage/` path | `homecore-migrate/src/storage.rs:36–37`, `main.rs:14–32` | `HaStorageDir::file_path` calls `self.path.join(name)` where `name` comes from hard-coded constants, so exploitation requires the `--storage` argument itself to point outside the intended tree. There is no `Path::canonicalize` + prefix check. While the current filenames are constants, if P2 makes `name` data-driven the surface widens. | Add `path.canonicalize()` + assert prefix after computing `file_path` if the name ever becomes user-controlled. Document this as a P2 gate. |
| HC-16 | **Low** | `AutomationEngine` uses `eprintln!` for action errors | `homecore-automation/src/engine.rs:93–95, 105` | Action errors and lag notices are emitted via `eprintln!`, not `tracing::warn!`. Same issues as HC-10: bypasses structured logging. | Replace with `tracing::warn!`/`tracing::error!`. |
| HC-17 | **Informational** | WS `call_service` authorization is contingent on fixing HC-01/HC-02 | `homecore-api/src/ws.rs:222–235` | `call_service` (including destructive calls such as `homeassistant.restart`) sits behind the WS auth handshake. Once HC-01 and HC-02 are fixed this path is properly guarded. No additional change needed here beyond those fixes. | No action required beyond HC-02. |
| HC-18 | **Informational** | `hc_state_subscribe` accumulates entity strings without eviction | `homecore-plugins/src/wasmtime_runtime.rs:263–268` | The `PluginStoreData.subscriptions` Vec grows without bound if a plugin repeatedly subscribes to the same entity. There is no deduplication. This is a plugin-local memory leak, not a sandbox escape. | Deduplicate on insert: `if !caller.data().subscriptions.contains(&eid)`. |

---

## Negative-Result Section (Surfaces Checked and Found Clean)

**SQL injection (homecore-recorder/src/db.rs)**: All queries use `sqlx::query`
with positional `?` bind parameters. No `format!`-constructed SQL was found in
any path (`record_state`, `record_event`, `get_state_history`, `search_semantic`,
`apply_schema`). Clean.

**WS bearer token in logs/error messages**: The bearer token is extracted and
immediately discarded after the non-empty check at ws.rs:62. It is not passed
to any `tracing` macro, `eprintln!`, or error-display path. The `access_token`
field is not part of any `Debug`-derived struct that enters a log path. Clean.

**REST bearer token in logs/error messages**: `BearerAuth(token)` is `Debug`
but no handler logs it or includes it in an error response. `ApiError` variants
do not capture the token. Clean.

**WASM linear-memory buffer overflow in `hc_state_get`/`hc_state_set`**: The
`read_str` helper validates `len < 0` and `len > MAX_ABI_BUFFER_BYTES (65536)`
before slicing, and uses `mem.get(ptr..ptr+len)?` which cannot panic. In
`hc_state_get` phase 3, the write is guarded by `json_bytes.len() > out_cap`
before attempting the slice. The `call_export_str` host-to-guest path also uses
`.get_mut(ptr..ptr+len).ok_or_else(...)` rather than unchecked indexing. No
buffer-overflow vector identified in the host ABI.

**WASM JSON ABI escape**: Plugins receive and emit plain UTF-8 JSON strings via
the linear-memory ABI. The host deserializes attribute JSON with
`serde_json::from_str` and defaults to `{}` on parse failure — no panic path.
No mechanism for a plugin to escape the Cranelift JIT sandbox via the JSON layer
alone was identified; the sandbox-escape risk is in the Cranelift/Winch compiler
backends (HC-03/04).

**Path traversal in homecore-migrate**: All `.storage/` filenames are currently
hard-coded constants (`"core.entity_registry"`, `"core.device_registry"`, etc.)
in the Rust source. The `--storage` and `--config-dir` arguments are user-supplied
but refer to the directory root, not individual filenames. No user-controlled
string is concatenated into a file path. Clean at P1 scope (noted as a P2 gate in HC-15).

**DoS via event-bus flood from a plugin**: A WASM plugin can call `hc_state_set`
in a tight loop. Each call fires a `broadcast::Sender::send` on the system channel
(capacity 4096). When the channel is full, `send` returns 0 (receivers are
dropped/lagged) but does not block or panic. Lagged receivers are notified via
`RecvError::Lagged`. The state machine itself does not back-pressure the sender.
The flood can cause the recorder and automation engine to lag, but it cannot crash
the host process. Noted as design-level concern; acceptable for P1.

**Secrets leakage in homecore-migrate InspectSecrets**: The CLI correctly prints
`<redacted>` for secret values and only logs key names.

---

## Critical-Path Remediation List (Required Before Production Deployment)

The following items MUST be resolved before `homecore-server` is reachable from
any untrusted network:

1. **HC-01 + HC-02 (Critical)** — Implement the token store and validate bearer
   tokens in both `BearerAuth::from_headers` and the WS `handle_socket` auth
   phase. Until this is done every REST and WS endpoint is completely open.

2. **HC-03 + HC-04 (Critical)** — Upgrade `wasmtime` in `homecore-plugins/Cargo.toml`
   from `25.0.3` to `>=36.0.7` (or `>=42.0.2`). The current version has two
   confirmed CVSS-9.0 sandbox-escape bugs; loading any third-party WASM plugin
   on the current version cannot be considered safe.

3. **HC-06 (High)** — Upgrade `sqlx` from `0.7.4` to `>=0.8.1` to eliminate the
   binary-protocol misinterpretation bug.

4. **HC-05 (High)** — Replace `CorsLayer::permissive()` with an explicit
   origin allowlist before any browser-accessible deployment.

5. **HC-08 (High)** — Replace the hardcoded HAP setup code and broadcast MAC
   address with randomly generated values before P2 real HAP pairing lands.

6. **HC-07 (High)** — Add per-connection subscription limit to the WS command
   loop before exposing the server to untrusted LAN clients.

---

## Dependency CVE Summary

`cargo audit` reported **18 advisories** against workspace `Cargo.lock`:

| Advisory | Crate | Severity | Affects HOMECORE |
|----------|-------|----------|------------------|
| RUSTSEC-2026-0096 | wasmtime 25.0.3 | Critical (9.0) | homecore-plugins |
| RUSTSEC-2026-0095 | wasmtime 25.0.3 | Critical (9.0) | homecore-plugins |
| RUSTSEC-2026-0093 | wasmtime 25.0.3 | Medium (6.9) | homecore-plugins |
| RUSTSEC-2026-0020 | wasmtime 25.0.3 | Medium (6.9) | homecore-plugins |
| RUSTSEC-2026-0021 | wasmtime 25.0.3 | Medium (6.9) | homecore-plugins |
| RUSTSEC-2024-0363 | sqlx 0.7.4 | (no CVSS) | homecore-recorder |
| RUSTSEC-2026-0091 | wasmtime 25.0.3 | Medium (6.1) | homecore-plugins |
| RUSTSEC-2026-0094 | wasmtime 25.0.3 | Medium (6.1) | homecore-plugins |
| RUSTSEC-2026-0089 | wasmtime 25.0.3 | Medium (5.9) | homecore-plugins |
| RUSTSEC-2026-0092 | wasmtime 25.0.3 | Medium (5.9) | homecore-plugins |
| RUSTSEC-2023-0071 | rsa 0.9.10 | Medium (5.9) | transitive via sqlx-mysql |
| RUSTSEC-2026-0085 | wasmtime 25.0.3 | Medium (5.6) | homecore-plugins |
| RUSTSEC-2026-0087 | wasmtime 25.0.3 | Medium (4.1) | homecore-plugins |
| RUSTSEC-2025-0046 | wasmtime 25.0.3 | Low (3.3) | homecore-plugins |
| RUSTSEC-2026-0086 | wasmtime 25.0.3 | Low (2.3) | homecore-plugins |
| RUSTSEC-2026-0088 | wasmtime 25.0.3 | Low (2.3) | homecore-plugins |
| RUSTSEC-2025-0118 | wasmtime 25.0.3 | Low (1.8) | homecore-plugins |
| RUSTSEC-2024-0006 | shlex 0.1.1 | (no CVSS) | transitive via wasm3-sys |

All 15 wasmtime advisories are resolved by upgrading to `wasmtime >= 36.0.7`.
