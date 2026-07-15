# ADR-161: HOMECORE Server Layer — WebSocket Auth Bypass, Reply-Theater & Documented-but-No-Op Automation (Security & Honest Labeling)

- **Status**: accepted
- **Date**: 2026-06-12
- **Deciders**: ruv
- **Tags**: homecore, http-ws-boundary, websocket-auth-bypass, security, automation-engine, documented-no-op, prove-everything, soundness, honest-labeling
- **Amends**: ADR-130 (HOMECORE-API WS protocol), ADR-129 (HOMECORE-AUTO automation engine), ADR-128 (plugin manifest)

## Context

Beyond-SOTA sweep **Milestone 7**, over the HOMECORE **server/network layer**
crates only — `homecore-api`, `homecore-server`, `homecore-automation`,
`homecore-hap`, `homecore-plugins` — executed under the project's
**prove-everything / anti-"AI-slop"** directive.

### Headline — the library cores are real, but the network boundary was unsound

The same audit pattern as ADR-160 held for the *library logic*: the automation
trigger/condition/template/action evaluators, the REST handlers, the HAP
mapping, and the plugin manifest parser are **real, tested code** — not stubs.
That is the anti-slop positive and it is cited here as such.

What the audit found was **not fake business logic but an unsound trust
boundary plus documented-but-no-op features**:

1. A **CRITICAL WebSocket authentication bypass** — the WS handshake accepted
   any non-empty token, ignoring the provisioned token whitelist the REST path
   enforces.
2. **Reply-theater** — WS command responses were computed, then logged and
   **discarded**; no `result`/`pong`/`event` ever reached the client.
3. **Documented-but-idle automation** — the engine was constructed and dropped
   (never started); time triggers, `RunMode`, `Choose` branches, and template
   conditions were each **documented as working but were no-ops in the live
   path**.

This is a worse class than ADR-160's over-naming: here the **doc claimed a
capability the code did not deliver** (auth enforcement, reply transport,
running automations). The fix is **implement where feasible, honestly relabel
where not — never leave a false doc.** Every fix is pinned by a test that
**fails on the old code**.

Grading vocabulary (ADR-152 / ADR-158 / ADR-160):
- **MEASURED** — reproduced in this worktree, command + failing-on-old test recorded.
- **NO-ACTION (already-honest/already-hardened)** — audited, found correct, cited as a positive.
- **ACCEPTED-FUTURE** — deliberately deferred, nothing dropped.

## Decision — Fixes Landed

### §A1 — WebSocket auth bypass (CRITICAL, security) — MEASURED

`homecore-api/src/ws.rs` handshake checked only `token.trim().is_empty()` and
sent `auth_ok` for **any** non-empty token. It never called
`state.tokens().is_valid()` — the check the REST path uses via
`auth::BearerAuth`. With a provisioned `HOMECORE_TOKENS` whitelist, **any
attacker-chosen non-empty token got full WS access** (read all states, call any
service, subscribe to all events).

**Real fix:** the handshake now calls
`state.tokens().is_valid(&token).await` (the *same* store + method as REST).
A wrong token receives `auth_invalid` and the socket closes. DEV (`allow_any`)
mode still accepts any non-empty bearer with a warn, so smoke tests keep
working; the empty token is rejected inside `is_valid`.

**Failing-on-old test** (`tests/ws_handshake.rs`):
`wrong_token_is_rejected` — provisions a real (non-dev) store with one good
token, sends a DIFFERENT non-empty token over the WS handshake, asserts
`auth_invalid`. On the old source the client received
`{"type":"auth_ok",…}` (verified: the test panics on old `ws.rs` with
`left: "auth_ok", right: "auth_invalid"`). Companion: `correct_token_is_accepted`.
**Grade: MEASURED. This is the milestone headline.**

### §A2 — WS replies never transmitted (HIGH, functional) — MEASURED

`ws.rs::Connection::run` moved the socket into a recv-only task; the only
consumer of the response mpsc just did `debug!("ws emit: {msg}")` and dropped
every message. No command reply ever reached the wire.

**Real fix:** the socket is split with `futures_util::StreamExt::split`. A
dedicated **writer task** drains the response channel onto `sink.send(...)`
(text frames; a `__pong:<n>` sentinel maps to a Pong control frame); the reader
task parses commands concurrently. On reader exit the senders drop and the
writer task ends cleanly.

**Failing-on-old tests:** `result_reply_is_received` (connect → auth →
`get_states` → assert a `result` reply is RECEIVED within 5s) and
`ping_pong_reply_is_received`. Both time out on the old source (verified:
`Elapsed` panic). **Grade: MEASURED.**

### §A8 — `homecore-api` bin: no env-token path, network-exposed (HIGH, security) — MEASURED

`homecore-api/src/bin/server.rs` bound `0.0.0.0:8123` with
`SharedState::new()` → `allow_any_non_empty()` and **no** `HOMECORE_TOKENS`
path (unlike `homecore-server`), so a provisioned operator had no way to lock
it down.

**Real fix:** the bin now mirrors `homecore-server`'s provisioning — prefer the
`HOMECORE_TOKENS` whitelist (`LongLivedTokenStore::from_env()`), fall back to an
**explicitly warn-logged** DEV mode only when unset. It also defaults the bind
address to **`127.0.0.1`** (loopback) so a bare `cargo run` is not
network-exposed, with `HOMECORE_BIND` to opt into LAN.

**Failing-on-old test** (`tests/server_bin_auth.rs`):
`provisioned_bin_rejects_wrong_bearer` reproduces the bin's exact provisioning
path (a populated, non-dev store) and asserts a wrong bearer → 401;
`from_env_path_enforces_whitelist` proves `from_env()` is not dev mode and
enforces the list. The old bin's `allow_any_non_empty()` accepted the wrong
bearer. **Grade: MEASURED.**

### §A3 — Automation engine never started (HIGH) — MEASURED

`homecore-server/src/main.rs` did `let _automation_engine = AutomationEngine::new(...)`
then dropped it immediately, while the header doc claimed "Automation engine
subscribed to the state machine."

**Real fix:** the engine is now built into a long-lived binding and `.start()`
is called, spawning the event loop + timer task; the header/log lines state it
is started with N automations and which trigger classes are active. (With A4–A7
the running engine is genuinely functional, not theater.)

**Evidence:** the engine-behavior tests below run against the same
`AutomationEngine::start()` path now wired into the bin. **Grade: MEASURED.**

### §A4 — `Trigger::Time` hard-coded `false`, no timer (HIGH) — MEASURED

`trigger.rs::matches_sync` returned `false` for `Time` and there was **no timer
task** anywhere, so time automations could never fire.

**Real fix:** `AutomationEngine::start_timer` — a 1 Hz tokio interval that
compares each `time:` automation's `at` (`HH:MM` or `HH:MM:SS`) against the
local wall-clock second and fires it once per match (conditions still gate it).
`matches_sync` returning `false` for `Time` is now **correct and documented**
(it is a wall-clock trigger with no state-change context); a public
`fire_time_for_test` exposes the same path deterministically.

**Failing-on-old test** (`tests/engine_behaviors.rs`):
`time_trigger_fires_via_timer_path` (+ unit `time_at_matches_handles_hh_mm_and_hh_mm_ss`).
The method does not exist on the old engine. **Grade: MEASURED.**

### §A5 — `RunMode` documented as AtomicBool-enforced but unbounded-parallel (HIGH) — MEASURED

`engine.rs` doc claimed "RunMode::Single is enforced via a per-automation
AtomicBool" — but no such code existed and **every** trigger spawned an
unbounded parallel task regardless of `mode`.

**Real fix:** each registered automation carries a `running: Arc<AtomicBool>`.
`Single`/`IgnoreFirst` modes `compare_exchange` the flag before spawning and
**skip** the trigger if a run is already in flight, clearing it on completion;
`Parallel` (and, for now, `Restart`/`Queued`) spawn on every trigger.

**Failing-on-old tests** (`tests/engine_behaviors.rs`):
`single_mode_does_not_double_fire_on_rapid_triggers` (two rapid triggers while
the first run sleeps → exactly **1** run; old code fired **2**, verified) and
`parallel_mode_does_fire_concurrently` (→ 2). **Grade: MEASURED (Single/Parallel
honored; bounded `Queued`/`Restart`/`max` ordering → ACCEPTED-FUTURE, see below).**

### §A6 — `Action::Choose` ignored branches (HIGH) — MEASURED

`action.rs` discarded `choices` and always ran `default`.

**Real fix:** `ChoiceBranch::matches` deserialises each branch's
`serde_yaml::Value` conditions into `Condition` and evaluates them (AND
semantics, against an `EvalContext` now carried on `ExecutionContext`). `Choose`
runs the **first matching branch's** sequence and falls to `default` only if
none match.

**Failing-on-old tests** (`action.rs` inline):
`choose_runs_matching_branch_not_default` (matching branch runs, default does
NOT — old code ran default, verified) and
`choose_falls_to_default_when_no_branch_matches`. **Grade: MEASURED.**

### §A7 — Template conditions always false in the live engine (MEDIUM) — MEASURED

`condition.rs` returned `false` for `Template` whenever `template_env` was
`None`, and the engine built every `EvalContext` with `template_env: None`
(`EvalContext::new`), so `template:` conditions could never be true in
production — only in unit tests that hand-built a template env.

**Real fix:** the engine constructs one `TemplateEnvironment` over the state
machine and threads it into every `EvalContext` via
`EvalContext::with_templates` (event loop, timer task, and
`ExecutionContext` for `Choose` branches).

**Failing-on-old tests** (`tests/engine_behaviors.rs`):
`template_condition_evaluates_true_in_engine` (a `{{ is_state(...) }}` condition
gates an action true) and `template_condition_evaluates_false_blocks_action`.
On the old engine the action never ran (template always false, verified).
**Grade: MEASURED.**

### §B5 — Plugin manifest sig/hash "verified before execution" doc was false (LOW, honesty) — relabeled

`homecore-plugins/src/manifest.rs` documented `wasm_module_hash` as "verified
before execution" and carried `wasm_module_sig` / `publisher_key`, but these
fields are **never read** for verification (only ever set to `None` in tests).

**Fix (honest labeling — no false capability claimed):** the three fields are
re-doc'd **"(P4 — not yet enforced, ADR-161/B5)"** — parsed and round-tripped,
but no integrity/signature check happens before a plugin runs. No verification
code was added (that is P4); the doc now matches the code.
**Grade: doc-honesty (no behavior change).** *(Superseded by ADR-162 §P4:
the hash/signature gate is now implemented and enforced.)*

## Negative Results (NO-ACTION positives — audited, found correct, cited not edited)

These were checked and are genuinely sound/honest; cited as positives, **not**
touched:
- **CSPRNG correctness** — all IDs are `uuid::v4`; the rng/`randn` suspicion was
  **REFUTED**. No weak-randomness issue exists.
- **CORS allowlist** (`app.rs`) — already hardened (explicit `AllowOrigin::list`,
  no `permissive()`, `allow_credentials(false)`, env override). NO-ACTION.
- **No path traversal in `homecore-migrate`** — audited, clean.
- **No secrets in logs** — audited, clean.
- **HAP pairing stub** — honestly disclaimed as a surface stub; not over-claimed.
- **`InProcessRuntime` "no sandbox" disclaimer** — honest; left as-is.

## Deferred Backlog (Nothing Dropped)

- **Plugin authority-isolation (P5)** — ~~`homecore_permissions` claims are parsed
  but not enforced at the host-call boundary.~~ **DONE — ADR-162 §P5.**
  `hc_state_set` now consults a `PermissionSet` distilled from the manifest;
  an undeclared write returns a typed `-3` to the guest.
- **Plugin signature/hash verification (P4)** — ~~implement the
  `wasm_module_hash`/`wasm_module_sig`/`publisher_key` gate that B5 now honestly
  says is absent.~~ **DONE — ADR-162 §P4.** `WasmtimeRuntime::load_plugin` now
  SHA-256-checks the module, Ed25519-verifies the signature against
  `publisher_key`, and enforces a `PluginPolicy` trust allowlist
  (secure-default rejects unsigned/untrusted/tampered modules).
- **HAP real pairing (P2)** — SRP/HKDF pairing + encrypted sessions; current
  bridge is an accessory-mapping surface. **ACCEPTED-FUTURE (honestly stubbed).**
- **`RunMode::Queued`/`Restart`/`max` ordering** — ~~`Single`/`Parallel` are
  honored; bounded queueing, restart-kill, and `max` concurrency are not yet
  wired (every non-Single mode is parallel).~~ **DONE — ADR-162 §A5.** Restart
  aborts the in-flight task, Queued serializes via a per-automation async mutex,
  and `max: N` caps concurrency via a per-automation semaphore.
- **Automation YAML load-at-boot** — the engine starts empty; a YAML loader is
  P-next. The bin log states "0 automations registered" honestly.

## Reproduction (MEASURED)

```bash
cd v2
cargo test -p homecore-api -p homecore-server -p homecore-automation -p homecore-hap --no-default-features
cargo test -p homecore-plugins --features wasmtime
cargo build --workspace --no-default-features
```

Result at time of writing (all 0 failed):
- **homecore-api** — **25 passed** (lib 18; `server_bin_auth` 3; `ws_handshake` 4)
- **homecore-automation** — **42 passed** (lib 37; `engine_behaviors` 5)
- **homecore-hap** — **17 passed**
- **homecore-server** — bin, **0 tests**
- (**homecore-plugins** — **15 passed**: lib 12; integration 3)
- Full workspace `cargo build --workspace --no-default-features` succeeds.

## Consequences

- The WebSocket path can no longer be entered with a forged token — it enforces
  the same `LongLivedTokenStore` whitelist as REST (A1).
- WS clients now actually receive `result`/`pong`/`event` frames (A2).
- The `homecore-api` dev bin defaults to loopback and honors `HOMECORE_TOKENS`
  (A8); it is no longer an open `0.0.0.0` accept-any endpoint by default.
- The automation engine is started for real and its time triggers, `Single`
  run-mode, `Choose` branches, and `template:` conditions all function — no doc
  claims a capability the code lacks (A3–A7).
- The plugin manifest no longer claims signature verification it does not
  perform (B5).
- Files kept under the 500-line guideline (`engine.rs` 462; behavioral tests
  moved to `tests/engine_behaviors.rs`).

## Addendum — `homecore-api` follow-up security review (beyond-SOTA pass)

A later network-facing review of `homecore-api` (the remote REST + WS attack
surface) — independent of the ADR-154–159 sweep — found and fixed two real
issues the original M7 pass (which focused on the WS auth bypass HC-WS-01, the
reply-theater HC-WS-02, and the bin token provisioning HC-WS-08) did not catch.
Both are LOW severity and reported at true severity.

### HC-API-AUTH-01 — `GET /api/` was unauthenticated (FIXED)

`rest::api_root` took no headers and unconditionally returned
`200 {"message":"API running."}`, while every sibling route gates on
`BearerAuth::from_headers`. HA's `APIStatusView` inherits `requires_auth = True`,
so `/api/` must return **401** for a missing/wrong bearer. HA clients use the
status route as a token-validation probe; a 200 told a bad-token client its
token was valid and let an unauthenticated party confirm a live endpoint.
LOW severity (the body is a static string; no entity/state data leaks).

**Fix:** `api_root(headers, State)` now validates the bearer like `get_config`.
**Pinned by** (fail-on-old, `tests/server_bin_auth.rs`):
`api_root_rejects_missing_bearer`, `api_root_rejects_wrong_bearer` (both 200→401),
guarded by `api_root_accepts_correct_bearer` (still 200 with a valid token).

### HC-WS-LAG-01 — `subscribe_events` killed the stream on a broadcast lag (FIXED)

The per-subscription task matched `Err(_) => break` on both broadcast
`recv()` arms. `RecvError::Lagged(n)` (a slow consumer falling
>`EVENT_CHANNEL_CAPACITY` = 4,096 events behind) is **recoverable** — the bus
doc says "Lagged receivers must re-sync" and HA keeps the subscription alive
across a lag. The old code treated the first lag as fatal, so after an event
burst the client's stream went permanently silent with no error frame — a
self-inflicted event-delivery DoS under load.

**Fix:** `Lagged(_) => continue` (skip the dropped window, re-sync),
`Closed => break`, on both the system and domain arms of the `select!`.
**Pinned by** `subscription_survives_broadcast_lag` (`tests/ws_handshake.rs`):
subscribes to a filtered event type, floods 6,000 unrelated events past the
4,096 capacity to force a `Lagged`, then asserts a subsequent subscribed event
is still delivered (old code: 5s-timeout panic).

### Dimensions confirmed clean (with evidence)

- **AuthN/AuthZ** — all 7 other REST handlers gate on `BearerAuth::from_headers`
  → `LongLivedTokenStore::is_valid` before any work; the WS handshake validates
  the `auth` token against the same store before the command loop, and
  privileged commands are unreachable pre-`auth_ok`. Token compare is
  `HashSet::contains` (content-independent timing — not the byte-`==` oracle of
  ADR-157 §B4), so no timing-oracle finding. No route skips the gate; no
  result-ignored check; no default/empty token accepted.
- **Path traversal** — no route maps user input to a filesystem path (state is an
  in-memory `DashMap`); `:entity_id` passes through `EntityId::parse`, a strict
  `[a-z0-9_]+\.[a-z0-9_]+` ASCII allowlist that rejects `..`, `/`, `\`, and
  absolute paths. No traversal surface.
- **Injection** — no SQL, no shell/subprocess, no `format!`-into-response;
  service/state bodies are typed `serde_json::Value` handed to the in-process
  registry (HA-equivalent).
- **Info-leak** — `ApiError` maps to fixed status + a typed `{message}`;
  `ServiceError::HandlerFailed(String)` is integration-controlled (HA surfaces
  the handler error too), never framework internals/paths/stack-traces — no
  ADR-080-class leak.
- **CORS** — explicit allowlist with `allow_credentials(false)` (HC-05),
  not `permissive()`.
- **De-magic** — no bare security-relevant literals in the crate worth
  extracting (`EVENT_CHANNEL_CAPACITY` is already named in `homecore`; CORS
  dev-default ports are documented).

**Tests:** `homecore-api --no-default-features` **25 → 29** (+2 api-root auth,
+1 api-root accept-guard, +1 WS lag-survival), 0 failed. Workspace green.
Python deterministic proof unchanged (homecore-api is off the signal proof
path).
