# ADR-162: HOMECORE Plugin Security (Signature + Capability Isolation) & Bounded Automation RunModes — Making ADR-161's Deferred Claims TRUE

- **Status**: accepted
- **Date**: 2026-06-12
- **Deciders**: ruv
- **Tags**: homecore, homecore-plugins, homecore-automation, plugin-security, wasm-signature-verification, ed25519, capability-isolation, runmode, prove-everything, soundness, honest-labeling
- **Amends**: ADR-161 (relabelled P4/P5 + §A5 deferrals → now enforced), ADR-128 (plugin manifest), ADR-129 (automation engine)

## Context

Beyond-SOTA sweep **Milestone 8**, scoped to `homecore-plugins` and
`homecore-automation` only, under the project's **prove-everything /
anti-"AI-slop"** directive.

ADR-161 (Milestone 7) did the honest thing with three plugin/automation
items it could not finish in that window: rather than fake them, it **relabelled
them as deferred** —

- **P4** (plugin signature verification): the manifest's `wasm_module_hash` /
  `wasm_module_sig` / `publisher_key` were re-doc'd "(P4 — not yet enforced,
  ADR-161/B5)" — parsed and round-tripped, but **never checked** before a
  plugin runs.
- **P5** (plugin authority isolation): `homecore_permissions` claims were
  parsed but **never consulted**; `hc_state_set` let any plugin write any
  entity, including `lock.*` / `alarm_control_panel.*`.
- **§A5** (`RunMode`): `Single`/`Parallel` were honored; `Restart`/`Queued`/
  `max: N` were honestly documented as still **unbounded-parallel**.

### Headline — the deferred security items are now ENFORCED + TESTED

M8 turns those honest deferrals into real, tested behavior. The plugin trust
boundary is now sound (a tampered module, an untrusted publisher, or an
unsigned module is rejected by the secure default), an over-privileged plugin
write is denied with a typed error, and the bounded run-modes actually bound.
**Every fix is pinned by a test that FAILS on the pre-M8 code** — each of the
three RunMode tests was additionally run against a simulated unbounded-parallel
dispatch and confirmed to panic.

The Ed25519 crypto reuses the in-repo `cog-ha-matter::witness_signing` pattern
(same `ed25519-dalek` 2.x API, same deterministic-test-key convention). SHA-256
matches the `sha256:` prefix the manifest already declared and the
`cog-ha-matter` cog manifest's `binary_sha256` hex convention. No new external
dependency tree was introduced — `ed25519-dalek` / `sha2` / `hex` / `base64`
were already in the workspace `Cargo.lock` (cog-ha-matter / bfld pull them in);
only new dependency *edges* were added to `homecore-plugins`.

Grading vocabulary (ADR-152 / ADR-158 / ADR-160 / ADR-161):
- **MEASURED** — reproduced in this worktree, command + failing-on-old test recorded.
- **ACCEPTED-FUTURE** — deliberately deferred, nothing dropped.

## Decision — Fixes Landed

### §P4 — Plugin signature & integrity verification (SECURITY) — MEASURED

`homecore-plugins/src/manifest.rs` declared `wasm_module_hash` /
`wasm_module_sig` / `publisher_key` but they were **never read** for
verification; the load path (`wasmtime_runtime.rs`) instantiated any `.wasm`
bytes handed to it.

**Real fix** (`src/verify.rs`, wired into `WasmtimeRuntime::load_plugin`):
before instantiation the runtime now —

1. computes the **SHA-256** of the actual `.wasm` bytes and rejects if it ≠ the
   manifest's `wasm_module_hash` (`sha256:<hex>`) — tamper detection;
2. verifies the **Ed25519** `wasm_module_sig` (`ed25519:<base64>`, 64-byte raw)
   over the 32-byte digest against `publisher_key` (`ed25519:<base64>`, 32-byte
   raw) and rejects on failure;
3. enforces a configurable **trust policy** — `PluginPolicy::trusted(&[keys])`
   is an allowlist of publisher verifying keys; `PluginPolicy::AllowUnsigned`
   is an explicit dev escape hatch that LOGS a loud `warn` on every load it
   waves through. The **secure default rejects unsigned and unknown-publisher
   modules.** `PluginPolicy::deny_all()` trusts no publisher.

A typed `PluginError::SignatureRejected` is returned (no host panic). The
legacy permission-free `load_wasm` is retained for first-party/trusted/test
modules; production loading goes through `load_plugin`.

**Failing-on-old tests** (`tests/integration.rs`, `--features wasmtime`) — all
drive `load_plugin`, which **did not exist** on the old code (so the gate is
genuinely new):
- `p4_tampered_module_is_rejected` — a byte-flipped `.wasm` → hash mismatch → rejected.
- `p4_valid_sig_from_trusted_key_loads` — a valid sig from an allowlisted key loads.
- `p4_valid_sig_from_untrusted_key_is_rejected` — a correctly-signed module from a key NOT on the allowlist is rejected.
- `p4_unsigned_module_rejected_by_default_loads_only_under_allow_unsigned` — unsigned rejected under `deny_all`, loads (with warn) only under `AllowUnsigned`.
- Unit (`src/verify.rs`): `valid_sig_from_trusted_key_passes`, `tampered_module_is_rejected`, `valid_sig_from_untrusted_key_is_rejected`, `forged_signature_is_rejected`, `unsigned_module_rejected_under_default_policy`.

A real deterministic keypair signs real `.wasm` bytes in the tests.
The manifest doc now reads **"(P4 — ENFORCED, ADR-162)"**. **Grade: MEASURED. Milestone headline.**

### §P5 — Plugin authority / capability isolation (SECURITY) — MEASURED

`wasmtime_runtime.rs::hc_state_set` applied any write a plugin requested,
ignoring the manifest's `homecore_permissions`.

**Real fix** (`src/permissions.rs` + `hc_state_set`): the manifest's
`homecore_permissions` (the `state:write:<glob>` form, or a bare entity glob
like `light.*`) are distilled into a `PermissionSet` installed in the plugin's
Wasmtime store. The `hc_state_set` host import consults
`permissions.may_write(entity_id)` before applying a write and returns a typed
`-3` (permission denied) to the guest on a violation — **the host is not
panicked.** Wasmtime already gives memory isolation; this adds **authority**
isolation. A plugin with **no** write grants can write nothing (secure default).

**Failing-on-old tests** (`tests/integration.rs`, `--features wasmtime`):
- `p5_declared_light_plugin_may_write_light_but_not_lock` — a `light.*` plugin writes `light.kitchen` (succeeds) but is REJECTED (`-3`, and the entity is not written) when it tries `lock.front_door`.
- `p5_plugin_with_no_permissions_can_write_nothing` — a plugin with empty `homecore_permissions` cannot write `light.kitchen`.
- Unit (`src/permissions.rs`): domain-glob, exact-grant, wildcard, read-grants-don't-confer-write, no-permissions, and explicit `state:write:` form.

The manifest doc now reads **"(P5 — ENFORCED, ADR-162)"**. **Grade: MEASURED.**

### §A5 — Bounded automation RunModes (Restart / Queued / max) — MEASURED

`homecore-automation/src/engine.rs` (per ADR-161) honored `Single`/`Parallel`
but spawned an unbounded parallel task for `Restart`/`Queued`/`max`.

**Real fix** (`src/runmode.rs`, a per-automation `RunState` the engine owns and
dispatches through at all three trigger sites — event loop, timer, test hook):
- **Restart** — aborts the in-flight action task via `tokio::task::AbortHandle`, then starts a fresh one.
- **Queued** — serializes runs in arrival order via a per-automation async `Mutex`: sequential, never concurrent, nothing dropped.
- **max: N** — caps concurrency at N via a per-automation `Semaphore`; triggers beyond N **queue** (await a permit) rather than running concurrently. (HA bounded `parallel`/`queued` semantics — chosen and documented as *queue beyond N*, not drop.)
- `Single`/`IgnoreFirst` re-entrancy guard and `Parallel` preserved.

`engine.rs` trimmed to **433 lines**; the run-mode machinery lives in the new
`runmode.rs` (153 lines) to keep both under the 500-line guideline.

**Failing-on-old tests** (`tests/engine_behaviors.rs`) — each was run against a
simulated unbounded-parallel dispatch and confirmed to panic:
- `restart_mode_cancels_prior_run` — prior run is aborted: exactly **1** completion (old: both ran → 2).
- `queued_mode_runs_sequentially_not_concurrently` — 3 rapid triggers all run, **max observed concurrency = 1** (old: 3).
- `max_two_caps_concurrency_at_two` — 4 rapid triggers all run, **max observed concurrency ≤ 2** (old: 4).

**Grade: MEASURED. Restart, Queued, and `max: N` all implemented — no remaining RunMode deferral.**

## Threat model closed

| Threat | Before (ADR-161) | After (ADR-162) |
|--------|------------------|-----------------|
| **Tampered module** — attacker swaps `.wasm` bytes after signing | loaded unconditionally (hash never checked) | rejected: SHA-256 mismatch |
| **Untrusted publisher** — valid sig from a key the host doesn't trust | loaded (sig/key never read) | rejected: publisher_key not on allowlist |
| **Unsigned module** — no integrity material at all | loaded | rejected by secure default; loads only under explicit `AllowUnsigned` (loud warn) |
| **Over-privileged plugin write** — a `light.*` plugin writes `lock.front_door` / `alarm_control_panel.*` | applied (permissions never consulted) | denied: typed `-3` to guest, write not applied |
| **Run-mode resource exhaustion** — `max`/`Queued` spawn unbounded tasks | unbounded parallel | bounded: Restart cancels, Queued serializes, `max: N` caps at N |

## Remaining honest deferral (Nothing Dropped)

- **Plugin-key provisioning / rotation** — the host's trust allowlist
  (`PluginPolicy::trusted`) is supplied by the caller; sourcing it from the
  Cognitum control-plane key store (as `cog-ha-matter` does for Seed keys) and
  key rotation are **ACCEPTED-FUTURE** (out of M8 scope — same boundary
  `witness_signing` draws).
- **`InProcessRuntime` (native first-party plugins)** — has no `.wasm` bytes to
  hash, so P4/P5 apply only to the WASM (`wasmtime`) path; native plugins remain
  trusted-by-compilation. Honestly noted, not over-claimed.
- **HAP real pairing (P2)** — unchanged from ADR-161; out of M8 scope.

## Reproduction (MEASURED)

```bash
cd v2
# P4/P5 (wasmtime feature needs rustc 1.91+; workspace pins 1.89 for the rest):
cargo +1.91.1 test -p homecore-plugins --features wasmtime
# Bounded RunModes:
cargo test -p homecore-automation --no-default-features
# Full workspace still builds (1.89 toolchain, no wasmtime):
cargo build --workspace --no-default-features
```

Result at time of writing (all 0 failed):
- **homecore-plugins** `--features wasmtime` — **32 passed** (lib 23; integration 9). (ADR-161 baseline was 15.)
- **homecore-automation** `--no-default-features` — **45 passed** (lib 37; `engine_behaviors` 8). (ADR-161 baseline was 42.)
- Full workspace `cargo build --workspace --no-default-features` succeeds.

## Consequences

- A HOMECORE WASM plugin can no longer be loaded with a tampered binary, an
  untrusted publisher, or (by default) no signature at all — the trust boundary
  ADR-161/B5 honestly said was absent is now real (P4).
- A plugin can no longer write entities outside its declared
  `homecore_permissions`; the lock/alarm escalation path is closed (P5).
- The automation engine's `Restart`, `Queued`, and `max: N` run-modes are now
  bounded as documented — no run-mode claims a capability the code lacks.
- No new external dependency tree (reuses the cog-ha-matter Ed25519 stack
  already in the lock); source files kept under the 500-line guideline
  (`engine.rs` 433, `runmode.rs` 153, `verify.rs` 397, `permissions.rs` 168;
  `wasmtime_runtime.rs` non-test source < 500, inline WAT tests as ADR-161 left
  them).
