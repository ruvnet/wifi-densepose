# ADR-186: Training progress API — wire the orphaned in-server trainer to `/ws/train/progress`

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-21 |
| **Deciders** | ruv |
| **Codename** | **TRAIN-RECONNECT** — connecting a trainer that was written, committed, and then never plugged in |
| **Relates to** | [ADR-051](ADR-051-sensing-server-decomposition.md) (main.rs decomposition into ~14 modules), [ADR-151](ADR-151-per-room-calibration.md) (`train-room` specialist bank), [ADR-152](ADR-152-wifi-pose-sota-2026.md) (MAE recipe / geometry conditioning), [ADR-166](ADR-166-quality-engineering-security-hardening.md) (WS auth + god-object decomposition) |
| **Tracking issue** | [#1233](https://github.com/ruvnet/wifi-densepose/issues/1233) — "Training does not start – /ws/train/progress returns 404 and no model is generated" (open) |

---

## 1. Context

### 1.1 The reported gap

A user starting training from the web dashboard hits
`ws://localhost:3000/ws/train/progress`, which **404s**, and the backend never
produces a trained `.rvf` model or any further log output beyond a single
"Training started" line. Issue #1233 is open, and the repo owner's own comment on
it states:

> The `/ws/train/progress` WebSocket endpoint is not yet exposed in the stable
> server — the training pipeline (room-calibration specialists, MAE pretraining)
> runs via the CLI (`wifi-densepose train-room`) rather than through the
> HTTP/WebSocket API, which is why the Docker image returns 404 for that path.

So the dashboard has a **"Start Training" button that silently no-ops**: it POSTs a
config, receives a `success: true` response, and then nothing happens — no error is
surfaced, no model is produced, no progress stream exists. A button that appears to
work but does nothing is the definition of slop, and this ADR exists to close that
gap honestly.

### 1.2 What the live server actually does today (evidence)

The stable server mounts **stub** training handlers. The POST handler flips a string
flag, logs one line, and returns success — it starts no job:

```rust
// v2/crates/wifi-densepose-sensing-server/src/main.rs:4986–5006
async fn train_start(
    State(state): State<SharedState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let mut s = state.write().await;
    if s.training_status == "running" { /* ... */ }
    s.training_status = "running".to_string();
    s.training_config = Some(body.clone());
    info!("Training started with config: {}", body);   // ← the one log line the issue reports
    Json(serde_json::json!({
        "success": true,
        "status": "running",
        "message": "Training pipeline started. Use GET /api/v1/train/status to monitor.",
    }))
}
```

These three stubs — and **nothing else training-related** — are wired into the live
router:

```rust
// v2/crates/wifi-densepose-sensing-server/src/main.rs:8068–8071
// Training endpoints
.route("/api/v1/train/status", get(train_status))
.route("/api/v1/train/start",  post(train_start))
.route("/api/v1/train/stop",   post(train_stop))
```

There is **no `/ws/train/progress` route in the live app** — hence the 404 that
issue #1233 reports. The stub state fields backing them are just:

```rust
// v2/crates/wifi-densepose-sensing-server/src/main.rs:1125–1127
training_status: String,               // "idle" | "running" | ...
training_config: Option<serde_json::Value>,
```

### 1.3 The surprising finding: a real trainer already exists, orphaned

The gap is **not** that training was never built for the server. A complete
in-server training pipeline **already exists in the tree** at
`v2/crates/wifi-densepose-sensing-server/src/training_api.rs` (1,860 lines). Its own
module doc describes what it does (`training_api.rs:1–25`):

- Loads recorded CSI from `.csi.jsonl` files, extracts signal features (subcarrier
  variance, temporal gradients, Goertzel frequency-domain power).
- Trains a regularised linear model via batch gradient descent.
- Exports a calibrated `.rvf` model container via `RvfBuilder` on completion.
- **"No PyTorch / `tch` dependency is required. All linear algebra is implemented
  inline using standard Rust math."** (`training_api.rs:11–13`)

It runs training on a **background tokio task** and streams progress over a
`tokio::sync::broadcast` channel to a real WebSocket handler:

- `start_training` spawns the job: `tokio::spawn(async move { ... })`
  (`training_api.rs:1564`, spawn at `:1610`).
- `ws_train_progress_handler` subscribes to `training_progress_tx` and forwards
  `{"type":"progress", "data": …}` frames (`training_api.rs:1778–1836`).
- A `routes()` factory wires the whole surface, **including the missing route**:

```rust
// v2/crates/wifi-densepose-sensing-server/src/training_api.rs:1841–1849
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/v1/train/start",    post(start_training))
        .route("/api/v1/train/stop",     post(stop_training))
        .route("/api/v1/train/status",   get(training_status))
        .route("/api/v1/train/pretrain", post(start_pretrain))
        .route("/api/v1/train/lora",     post(start_lora_training))
        .route("/ws/train/progress",     get(ws_train_progress_handler))
}
```

**This module is dead code.** There is no `mod training_api;` declaration anywhere
in the crate — a repo-wide search for `training_api` returns only a doc-comment
mention in `path_safety.rs:9`. Because Rust never sees the file without a `mod`
declaration, `training_api.rs` is **not compiled into the binary at all**, and
`training_api::routes()` is never merged into the app. It was written, committed
(last touched by commit `9b07dff29`), and then orphaned.

### 1.4 Why it would not even compile if naively wired in

The orphan was written against a **different state shape than the one that shipped**.
`training_api.rs` expects its parent to expose an `AppStateInner` carrying a training
sub-state and a broadcast sender:

```rust
// v2/crates/wifi-densepose-sensing-server/src/training_api.rs:249
pub type AppState = Arc<RwLock<super::AppStateInner>>;
// handlers read s.training_state.status, s.training_state.task_handle,
// s.training_progress_tx  (e.g. training_api.rs:1588, :1610, :1788)
```

But the **real** `AppStateInner` (`main.rs:1024`, aliased `SharedState` at
`main.rs:1249`) has none of those fields — only the `training_status: String` /
`training_config` stubs from §1.2. `training_state: TrainingState` is defined
locally in `training_api.rs:232`, and `training_progress_tx` exists nowhere on the
live state. So adding `mod training_api;` today produces a compile error: the module
references `AppStateInner` fields that do not exist. Wiring it in requires
**reconciling the state struct first**, not merely uncommenting a route.

### 1.5 The working path today

The path that actually trains a model is the CLI, exactly as the maintainer's
comment says:

- `wifi-densepose train-room` → `room.rs:241` `train_room(...)`, the ADR-151
  Stage-2–5 per-room specialist-bank trainer (`enroll → train-room → room-watch`).
- The heavier `wifi-densepose-train` crate exposes epoch-level metrics
  (`trainer.rs:43` `pub epoch: usize`, `trainer.rs:64` `best_epoch`) that a progress
  stream could surface directly — the data a WebSocket needs already exists in the
  training loop.

### 1.6 What this ADR is *not*

- Not a rewrite of the trainer. The pipeline in `training_api.rs` already exists;
  this ADR reconnects and hardens it.
- Not a move of GPU/`tch`-backed training into the Axum server. The in-server
  trainer is deliberately `tch`-free (§1.3). Heavy MAE/LoRA training stays in the
  CLI / `wifi-densepose-train` crate; the server streams progress for the light,
  pure-Rust specialist trainer and (optionally) proxies status for CLI-launched runs.
- Not a change to the `train-room` CLI contract (ADR-151). The CLI remains the
  authoritative path for offline / batch training.

---

## 2. Current state — evidence

| Artifact | Value | Source |
|---|---|---|
| Live POST handler | `train_start` — flips a flag, logs, returns `success:true`, starts no job | `main.rs:4986–5006` |
| The "Training started" log line from the issue | `info!("Training started with config: {}", body)` | `main.rs:5000` |
| Live training routes | `train/status`, `train/start`, `train/stop` (stubs only) | `main.rs:8068–8071` |
| `/ws/train/progress` in live app | **Absent** → 404 | (no route in `main.rs` router) |
| Live training state fields | `training_status: String`, `training_config: Option<Value>` | `main.rs:1125–1127` |
| Real in-server trainer | 1,860-line implemented pipeline, `tch`-free, exports `.rvf` | `training_api.rs:1–25` |
| Real WS progress handler | subscribes to broadcast, streams `progress` frames | `training_api.rs:1778–1836` |
| Real route factory (has the missing route) | `routes()` incl. `/ws/train/progress` | `training_api.rs:1841–1849` |
| Background job spawn | `tokio::spawn` of the training task | `training_api.rs:1564`, spawn `:1610` |
| `mod training_api;` declaration | **None in the crate** (only a doc mention) | `path_safety.rs:9` |
| State-shape mismatch | expects `super::AppStateInner.{training_state, training_progress_tx}` | `training_api.rs:249`, `:232` |
| Real `AppStateInner` / `SharedState` | has neither field | `main.rs:1024`, `:1249` |
| Working training path | CLI `train-room` (ADR-151 specialist bank) | `room.rs:241` |
| Epoch metrics available to stream | `TrainMetrics.epoch`, `best_epoch` | `train/src/trainer.rs:43`, `:64` |

---

## 3. Gap analysis

| Capability | Desired | Today | Gap severity |
|---|---|---|---|
| `/ws/train/progress` resolves | 101 Switching Protocols, streams epoch/loss/eta | 404 (route absent) | **Critical** — the reported bug |
| "Start Training" produces a model | background job trains and writes `.rvf` | flag flip + one log line, no job, no model | **Critical** |
| Error surfaced to the user | button reflects real state / disabled with reason | silent no-op, `success:true` | **Critical** (slop) |
| In-server trainer compiled | part of the crate, unit-tested | orphaned; not compiled (no `mod`) | **High** |
| State supports progress streaming | `training_state` + `training_progress_tx` on `AppStateInner` | absent — orphan won't compile as-is | **High** |
| WS auth on the training surface | `/ws/train/progress` under bearer gate (ADR-166 §Sprint-1) | n/a (route absent) | **High** |
| `dataset_ids` path safety | validated before file open | `path_safety.rs` exists but unreached by live routes | **Medium** |
| Server ↔ CLI parity | shared/consistent training semantics | two divergent trainers (stub vs CLI vs orphan) | **Medium** |

---

## 4. Decision

**Chosen path: wire the existing in-server trainer into the live server** — reconcile
the state struct, declare the module, merge `training_api::routes()`, delete the
stub handlers, and expose a real `/ws/train/progress` that streams epoch/loss/eta
events from the already-implemented background job.

This is called **TRAIN-RECONNECT**.

### 4.1 Why this path, and not "make the button honestly say CLI-only"

The task framing offered two honest options. Investigation decided it:

| Consideration | Evidence | Implication |
|---|---|---|
| Is server-side training genuinely GPU/`tch`-bound (→ keep CLI-only)? | The in-server trainer is explicitly **`tch`-free**, pure Rust, exports `.rvf` (`training_api.rs:11–13`) | The "too heavy for Axum" argument is contradicted by the code |
| Does a real streaming implementation already exist? | Full pipeline + broadcast + WS handler + `routes()` present (`training_api.rs:1564,1778,1841`) | The impressive-sounding option is also the *least* new code — it already exists |
| Why does it 404 then? | No `mod training_api;`; state-shape mismatch (`:249` vs `main.rs:1024`) | The fix is reconnection + reconciliation, not new invention |

Because the honest, code-supported reality is "a working trainer was written and left
unplugged," the right decision is to plug it in — this is not choosing the flashier
option over the code; it *is* what the code says.

**However**, path B is retained as a **mandatory fallback guarantee** (Phase P5): if,
for a given build/deployment, server-side training is disabled (e.g. behind a
feature flag, or on the lightweight appliance image where recordings aren't
available), the dashboard button MUST be disabled with a tooltip pointing at
`wifi-densepose train-room` — never a silent `success:true` no-op again. The slop is
eliminated in both the enabled and disabled configurations.

### 4.2 Scope boundary — light trainer streams, heavy trainer proxies

- The **pure-Rust specialist trainer** (`training_api.rs`, ADR-151 flavour) runs
  in-process and streams live epoch/loss/eta over `/ws/train/progress`.
- **Heavy MAE/LoRA training** (`wifi-densepose-train`, `tch`/GPU) stays CLI-launched.
  The server does not host it; at most `/api/v1/train/status` reports on a
  CLI-launched run if one registers itself. Streaming heavy training is out of scope
  for this ADR (noted as an open question, §8).

---

## 5. Detailed design

### 5.1 Reconcile `AppStateInner`

Replace the two stub fields (`main.rs:1125–1127`) with the sub-state the trainer
expects, so `training_api.rs` compiles against `super::AppStateInner`:

```rust
// main.rs — inside AppStateInner (replacing training_status / training_config)
training_state: training_api::TrainingState,          // status, epoch, best_pck, task_handle
training_progress_tx: tokio::sync::broadcast::Sender<String>,  // progress fan-out
```

`train_status` consumers that read `s.training_status` / `s.training_config` are
updated to read `s.training_state.status`. The broadcast sender is created at state
init (`main.rs:7826` region, where the stubs are seeded today).

### 5.2 Declare and merge the module

- Add `mod training_api;` to `main.rs` (or `pub mod` in `lib.rs` if the router is
  assembled there).
- Delete the stub handlers `train_start` / `train_stop` / `train_status`
  (`main.rs:4977–5023`) and their three route mounts (`main.rs:8069–8071`).
- Merge the real router **after** `.with_state(state.clone())`, the same pattern the
  RuField surface already uses (`main.rs:8104–8111`):

```rust
// main.rs router assembly
.merge(training_api::routes())
```

so that `/api/v1/train/*` and `/ws/train/progress` resolve against the shared state.

### 5.3 Auth and safety (ADR-166 alignment)

- `/api/v1/train/*` sits under the existing opt-in bearer gate (`main.rs:8095–8102`,
  `RUVIEW_API_TOKEN`). `/ws/train/progress` follows the same policy decision made for
  `/ws/sensing` — document explicitly whether the training WS is gated (recommended:
  gated when a token is set, since training reads/writes recordings and models).
- `dataset_ids` from `StartTrainingRequest` (`training_api.rs:126–130`) are resolved
  through `path_safety` before any file open — `path_safety.rs:9` already anticipates
  `{dataset_id}.csi.jsonl` under `RECORDINGS_DIR`; wire it in the load path.
- Single-job concurrency guard: `start_training` already rejects a second run while
  `training_state.status.active` (`training_api.rs:1571`) — keep it.

### 5.4 Progress event schema (already emitted)

The WS handler already frames messages as `{"type":"status"|"progress", "data": …}`
(`training_api.rs:1796–1815`). Confirm the `data` payload carries at minimum
`epoch`, `total_epochs`, `loss`, `best_pck`, and an `eta_seconds`; these map onto the
`TrainMetrics`/`TrainingStatus` fields already populated by the loop
(`training_api.rs:1251`, `train/src/trainer.rs:43,64`).

### 5.5 Dashboard honesty (both configurations)

- **Enabled build:** button POSTs `/api/v1/train/start`, then opens
  `/ws/train/progress`; the UI renders live epoch/loss/eta and a terminal
  success/failure with the output `.rvf` path.
- **Disabled build:** `/api/v1/train/start` returns a structured
  `{"enabled": false, "reason": "...", "cli": "wifi-densepose train-room"}` and the
  button renders disabled with a tooltip — no silent `success:true`.

---

## 6. Phase ledger

```
P0 ──► P1 ──► P2 ──► P3 ──► P4 ──► P5 ──► P6
repro   state   wire    stream  auth+   dash    tests+
+audit  recon   router  job     safety  honesty witness
```

### P0 — Reproduce & audit (evidence lock)
- [x] Confirmed the orphan: `grep -rn "mod training_api"` returned **nothing**; the only
      hit was a doc mention in `path_safety.rs`. `training_api.rs` was uncompiled.
- [x] Confirmed the stub no-op (`train_start` at `main.rs:4986` flipped a string + logged
      one line, no job, no `.rvf`) and the missing `/ws/train/progress` route.

### P1 — Reconcile `AppStateInner`
- [x] Replaced `training_status`/`training_config` with `training_state:
      training_api::TrainingState` + `training_progress_tx: broadcast::Sender<String>`.
- [x] Updated state init; the only readers of the old fields were the stub handlers (deleted).
- [x] Added `mod training_api;` (+ `mod path_safety;`); the module compiles against the real state.

### P2 — Wire the router, delete the stubs
- [x] Removed `train_start`/`train_stop`/`train_status` and their 3 route mounts.
- [x] `.merge(training_api::routes())` — merged **before** `.with_state(...)` (not after).
      The RuField surface merges after because it carries a *different* state; the training
      router shares `SharedState`, so merging before is what puts `/api/v1/train/*` under the
      same `/api/v1/*` bearer gate as everything else.
- [x] `/api/v1/train/*` and `/ws/train/progress` resolve (verified by HTTP tests, not 404).

### P3 — Confirm the real job streams and produces a model
- [x] The spawned job loads `.csi.jsonl` (falls back to a `frame_history` snapshot),
      runs the gradient-descent loop, and writes a `.rvf` under `data/models`.
- [x] Progress frames carry `epoch`, `total_epochs`, `train_loss`, `val_pck`, `eta_secs`.
- [x] Server-vs-CLI semantics documented as **intentionally divergent** (§4.2, §9.2):
      the server runs the light pure-Rust specialist trainer; heavy MAE/LoRA stays CLI.

### P4 — Auth & path safety
- [x] `/api/v1/train/*` sits under the existing `RUVIEW_API_TOKEN` bearer gate (merged
      before `.with_state`); `/ws/train/progress` is intentionally **ungated**, matching
      `/ws/sensing` (browsers can't attach an `Authorization` header to a WS upgrade).
- [x] `dataset_ids` resolved via `path_safety::safe_id` before file open; pinned by
      `load_recording_frames_rejects_path_traversal`.
- [x] Single-job guard: `spawn_training_job` rejects a second start while active
      (`is_active()` → `active_error`).

### P5 — Dashboard honesty (fallback guarantee)
- [x] Enabled build: `TrainingPanel` opens `/ws/train/progress` before the POST and renders
      live epoch/loss/PCK/ETA + a terminal Complete state (already wired; verified).
- [x] Disabled build (`RUVIEW_DISABLE_SERVER_TRAINING`): start returns
      `{enabled:false, cli:"wifi-densepose train-room"}` HTTP 409; the dashboard reads
      `enabled` off `/api/v1/train/status` and disables the Start buttons with a CLI
      tooltip — no silent no-op. Implemented via a runtime flag rather than a Cargo feature
      so the `--no-default-features` test build keeps training ON (§9.4 resolved this way).

### P6 — Tests & witness
- [x] Live-socket test `ws_train_progress_live_101_and_frame`: genuine 101 handshake + a real
      progress frame after POST start. Plus `ws_train_progress_route_is_wired_not_404`.
- [x] `http_train_start_produces_model_and_streams`: POST start → poll status → `.rvf` exists.
- [x] CHANGELOG updated. README/CLAUDE have no training route table, so no route-table edit
      was needed there.

*(All phases complete. Acceptance criteria verified below — this ADR is Accepted.)*

---

## 7. Acceptance criteria (concrete verification)

All must pass before ADR-186 is Accepted:

- [x] **Orphan is reconnected:**
      `grep -rn "mod training_api" v2/crates/wifi-densepose-sensing-server/src/`
      returns a hit (`main.rs`), and
      `cargo build -p wifi-densepose-sensing-server` **compiles** (proves the state
      reconciliation in §5.1 is correct — the module cannot compile against the
      current `AppStateInner`). **VERIFIED.**
- [x] **Route no longer 404s (HTTP upgrade):** verified in-process rather than with a live
      `curl` — `ws_train_progress_live_101_and_frame` binds the training router on a real
      socket and `tokio_tungstenite::connect_async` completes a genuine **101** handshake
      (asserts `resp.status() == 101`); `ws_train_progress_route_is_wired_not_404` also
      confirms the route is reached (426 under `oneshot`, **not** 404). **VERIFIED.**
- [x] **Progress actually streams:** `ws_train_progress_live_101_and_frame` connects the WS,
      POSTs `/api/v1/train/start`, and receives a real `{"type":"progress","data":{...}}`
      frame within the 10 s ceiling. **VERIFIED.**
- [x] **A model is produced:** `http_train_start_produces_model_and_streams` POSTs start,
      polls `/api/v1/train/status` to completion, and asserts a **new `.rvf`** appeared under
      `data/models/` (snapshot diff). Also covered by the trainer-level
      `training_job_streams_real_progress_and_writes_model`. **VERIFIED.**
- [x] **No silent no-op remains:** `http_train_start_disabled_returns_structured_409` sets
      `RUVIEW_DISABLE_SERVER_TRAINING` and asserts POST start returns **HTTP 409** with
      `{"enabled":false, ...,"cli":"wifi-densepose train-room"}` and never `success:true`.
      **VERIFIED.**
- [x] **Auth honored:** `/api/v1/train/*` is merged into the router **before** the
      `RUVIEW_API_TOKEN` bearer middleware and `.with_state`, so it is covered by the exact
      same `/api/v1/*` gate as every other authenticated route (verified by construction /
      code review; `/ws/train/progress` is intentionally ungated like `/ws/sensing`). No new
      dedicated runtime token test was added — the gate is the shared, already-tested
      `bearer_auth` middleware. **VERIFIED (by construction).**
- [x] **Path safety:** `load_recording_frames_rejects_path_traversal` asserts
      `dataset_ids:["../../etc/passwd"]` yields no frames (rejected by `path_safety::safe_id`
      before any file open). **VERIFIED.**
- [x] **Integration test green:** `ws_train_progress_live_101_and_frame` (`#[tokio::test]`)
      serves the training router, opens `/ws/train/progress`, and asserts a 101 upgrade + a
      real progress frame — and, being built on `training_api::routes()`, cannot compile if
      the module is orphaned again. **VERIFIED.**
- [x] **Workspace regression:** `cargo test -p wifi-densepose-sensing-server
      -p wifi-densepose-train --no-default-features` — sensing-server bin **217 passed /
      0 failed**, all train suites **0 failed**. A full `cargo test --workspace
      --no-default-features` run initially surfaced a **test-only parallelism race** in the
      new tests (two model-writing tests deleted `.rvf`s by directory-diff, occasionally
      removing a file a third test asserted existed) — fixed by removing the cross-test
      deletions (each test cleans only its own artifact; `data/models` is gitignored).
      Re-verified post-fix: `cargo test --workspace --no-default-features` — **0 failed**
      (exit 0). **VERIFIED.**

---

## 8. Consequences

### Positive
- Closes issue #1233: the dashboard button either trains-and-streams or honestly says
  "use the CLI" — the silent no-op is gone in every configuration.
- Reclaims 1,860 lines of already-written, already-committed trainer that were dead
  (uncompiled) code, and adds a test that keeps them wired.
- `/ws/train/progress` gives the UI real epoch/loss/eta, matching the maintainer's
  stated intent.
- Forces the state-shape reconciliation that the orphan implied but never landed,
  removing a latent "two competing training designs" trap in `AppStateInner`.

### Negative
- Editing `AppStateInner` (`main.rs:1024`) and the router (`main.rs:8068`) touches the
  large `main.rs`; merge-conflict risk with concurrent work on the same file (the
  ADR-166 decomposition is relevant here).
- Adds a live training code path to the server's attack surface — mitigated by the
  bearer gate and `path_safety`, but it must be reviewed (network/hardware boundary,
  per the pre-merge security checklist).
- Server and CLI now have two trainers that must be kept semantically consistent, or
  their divergence explicitly documented.

### Neutral
- Heavy MAE/LoRA/`tch` training remains CLI-only; the server streams only the
  light pure-Rust specialist trainer. Streaming heavy runs is deferred.
- The progress event schema (`epoch/loss/best_pck/eta`) is already emitted by the
  orphan; no new schema is invented, only confirmed and documented.

---

## 9. Open questions

1. **WS auth policy for `/ws/train/progress`:** gate it whenever `RUVIEW_API_TOKEN`
   is set (like `/api/v1/*`), or leave it open like `/ws/sensing`? *Tentative: gate
   it — training reads recordings and writes models.*
2. **Server ↔ CLI trainer parity:** should the in-server trainer and
   `wifi-densepose train-room` (ADR-151) share one code path, or remain deliberately
   separate (server = quick UI-driven specialist fit; CLI = full bank + geometry
   conditioning)? *Tentative: keep separate, document the split, share feature
   extraction where cheap.*
3. **Heavy-training progress:** can a CLI-launched `wifi-densepose-train` (`tch`)
   run register itself so `/api/v1/train/status` and the WS can report on it without
   hosting it in-process? *Tentative: out of scope here; a follow-on ADR.*
4. **Feature-flagging server training:** should in-server training be behind a Cargo
   feature (off on the lightweight appliance image), making the P5 disabled-button
   path the default there? *Tentative: yes — flag it; default the UI to the honest
   disabled state on images without recordings.*

---

## 10. References

- **Issue #1233**: https://github.com/ruvnet/wifi-densepose/issues/1233 — the reported bug.
- **Live stubs**: `v2/crates/wifi-densepose-sensing-server/src/main.rs:4977–5023` (handlers),
  `:8068–8071` (routes), `:1125–1127` (state fields), `:1024`/`:1249` (`AppStateInner`/`SharedState`).
- **Orphaned trainer**: `v2/crates/wifi-densepose-sensing-server/src/training_api.rs` —
  module doc `:1–25`, `TrainingState` `:232`, `AppState` alias `:249`, `start_training` `:1564`
  (spawn `:1610`), WS handler `:1778–1836`, `routes()` `:1841–1849`.
- **Not-a-module proof**: repo-wide `training_api` only in `path_safety.rs:9` (doc comment).
- **CLI working path**: `v2/crates/wifi-densepose-cli/src/room.rs:241` `train_room` (ADR-151).
- **Epoch metrics**: `v2/crates/wifi-densepose-train/src/trainer.rs:43`, `:64`.
- **ADR-166**: WebSocket authentication + `main.rs` decomposition (security context for this change).
- **ADR-151**: per-room calibration / `train-room` specialist bank.
