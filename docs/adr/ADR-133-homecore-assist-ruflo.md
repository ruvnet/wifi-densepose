# ADR-133: HOMECORE-ASSIST — Voice/Intent Pipeline + Ruflo Agent Bridge

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-25 |
| **Deciders** | ruv |
| **Codename** | **HOMECORE-ASSIST** |
| **Relates to** | [ADR-126](ADR-126-ruview-native-ha-port-master.md) (HOMECORE master), [ADR-127](ADR-127-homecore-state-machine-rust.md) (HOMECORE-CORE), [ADR-130](ADR-130-homecore-rest-websocket-api.md) (HOMECORE-API), [ADR-124](ADR-124-rvagent-mcp-ruvector-npm-integration.md) (SENSE-BRIDGE) |
| **Tracking issue** | TBD |
| **Crate** | `v2/crates/homecore-assist` |

---

## 1. Context

Home Assistant's Assist pipeline (`homeassistant/components/assist_pipeline/`) provides
voice-to-intent-to-response processing. It chains:

1. **STT** (speech-to-text) — Whisper, cloud, or satellite
2. **NLU** (natural language understanding) — intent recognition via regex/slots
3. **Intent handler** — maps intent to a HA service call
4. **TTS** (text-to-speech) — synthesises the response for the caller

HA's intent model (`homeassistant/helpers/intent.py`) is keyword/regex based. Every
intent is a named template with slot definitions and a handler that dispatches to HA
services. The built-in intents (`homeassistant/components/conversation/default_agent.py`)
cover `HassTurnOn`, `HassTurnOff`, `HassLightSet`, `HassNevermind`, `HassCancelAll`,
`HassGetState`, `HassGetWeather`, and many others.

HOMECORE needs a wire-compatible Assist pipeline so that:
- The HA iOS/Android companion app's "Assist" button works against HOMECORE.
- The HOMECORE-API WebSocket `assist` command (ADR-130 §2.2) has a handler.
- The ruflo agent toolchain (ADR-124) can provide LLM-grade intent disambiguation as a
  drop-in upgrade path for the P1 regex recognizer.

### 1.1 Ruflo integration approach

Ruflo's agent runner exposes an MCP-over-stdio interface (`node ruflo-agent.js`).
HOMECORE-ASSIST manages a long-lived subprocess (Q3 Windows concern below), sends
utterance JSON, and receives intent JSON back. In P1 we ship only the trait surface
and a `NoopRunner` stub; the real subprocess management is P2.

### 1.2 Ruvector semantic intent matching (P2)

`ruvector-core` provides embedding + cosine-similarity primitives. P2 will add a
`SemanticIntentRecognizer` that embeds the utterance and compares it to a HNSW index
of intent exemplars, falling back to the P1 regex recognizer when similarity < 0.75.
This is the mechanism that allows "dim the lights" to match `HassLightSet` without an
explicit regex entry.

---

## 2. Design

### 2.1 Module layout (`v2/crates/homecore-assist/`)

| Module | Contents |
|--------|----------|
| `intent` | `IntentName` newtype, `Intent` (name + slots), `IntentResponse` (speech + optional card + optional data) |
| `recognizer` | `IntentRecognizer` trait; `RegexIntentRecognizer` (P1); `SemanticIntentRecognizer` stub (P2) |
| `handler` | `IntentHandler` trait; built-in handlers: `HassTurnOn`, `HassTurnOff`, `HassLightSet`, `HassNevermind`, `HassCancelAll` |
| `runner` | `RufloRunner` trait + `RufloRunnerOpts`; `NoopRunner` (P1 stub); real subprocess runner (P2) |
| `pipeline` | `AssistPipeline`: wires recognizer → handler → response; exposes `async fn process(utterance, language) -> IntentResponse` |

### 2.2 Built-in intent handlers (P1)

| Handler | HA service call | Slot |
|---------|-----------------|------|
| `HassTurnOn` | `homeassistant.turn_on` / `light.turn_on` / `switch.turn_on` | `entity_id` |
| `HassTurnOff` | `homeassistant.turn_off` / `light.turn_off` / `switch.turn_off` | `entity_id` |
| `HassLightSet` | `light.turn_on` | `entity_id`, `brightness` (0–255), `color_name` |
| `HassNevermind` | — (no-op, returns acknowledgement) | — |
| `HassCancelAll` | — (fires `homeassistant_stop_all_scripts` domain event) | — |

### 2.3 IntentResponse

```rust
pub struct IntentResponse {
    pub speech: String,
    pub card: Option<Card>,
    pub data: Option<serde_json::Value>,
}

pub struct Card {
    pub title: String,
    pub content: String,
}
```

### 2.4 RufloRunner trait

```rust
#[async_trait]
pub trait RufloRunner: Send + Sync + 'static {
    async fn spawn(&mut self, opts: RufloRunnerOpts) -> Result<(), AssistError>;
    async fn send_request(&self, payload: serde_json::Value) -> Result<RufloResponse, AssistError>;
    async fn shutdown(&mut self) -> Result<(), AssistError>;
}
```

`RufloResponse` is `{ intent: Option<Intent>, speech: Option<String> }`.

### 2.5 Pipeline

```rust
pub struct AssistPipeline<R, H> {
    recognizer: R,
    handler: H,
    runner: Option<Box<dyn RufloRunner>>,
}

impl<R: IntentRecognizer, H: IntentHandler> AssistPipeline<R, H> {
    pub async fn process(&self, utterance: &str, language: &str, hc: &HomeCore)
        -> Result<IntentResponse, AssistError>;
}
```

---

## 3. Questions & Answers

### Q1 — Why not reuse HA's existing `homeassistant.helpers.intent` via PyO3?

PyO3 bridges add a GIL lock on every cross-language call; the Assist pipeline processes
hundreds of short utterances per day from voice satellites. A native Rust recognizer is
simpler and faster. Python HA can still connect as an external integration via MQTT or
the HOMECORE WebSocket API.

### Q2 — How does `RegexIntentRecognizer` handle ambiguity?

Patterns are tried in registration order; the first match wins. Slot extraction uses
named capture groups. A future P2 upgrade can run all patterns, score them by slot
completeness, and return the highest-scoring match.

### Q3 — Windows subprocess teardown (ruflo runner subprocess on Windows)

`tokio::process::Child` on Windows does not automatically kill the child process when
the `Child` struct is dropped — `SIGTERM` is not a Windows concept, and `TerminateProcess`
is not called automatically. Options for P2:

1. Call `child.start_kill()` in a `Drop` impl (requires a `Runtime` handle — tricky in sync Drop).
2. Wrap `Child` in an `Arc<Mutex<Option<Child>>>` and call `kill()` in an `async fn shutdown()`.
3. Use a Windows job object to bind the subprocess lifetime to the parent process.

**P2 decision**: implement option 2 (explicit `async shutdown()`) + register a `tokio::signal`
handler for `Ctrl+C` / `SIGINT` that calls `shutdown()` before exit. Document the Windows caveat
in the crate README and in `runner.rs`. Job object approach (option 3) is deferred to P3 only
if option 2 proves insufficient in fleet testing.

### Q4 — Why is `SemanticIntentRecognizer` a P2 stub?

The ruvector HNSW index requires the vector store to be populated at startup with intent
exemplars. That startup path requires deciding on a serialization format (HNSW index files
vs. an in-memory array at compile time), which intersects with ADR-084 (RabitQ) and ADR-067
(ruvector v2.0.5). P2 will define the exemplar format and populate the index.

---

## 4. Consequences

- **Positive**: HOMECORE-API `assist` WebSocket command gets a functional backend.
- **Positive**: Ruflo LLM pipelines can upgrade intent matching by swapping the `RufloRunner` impl.
- **Positive**: P1 ships with zero new heavy dependencies (no subprocess spawning, no ML runtime).
- **Negative**: Regex matching has limited coverage; long-tail utterances will return "I'm not sure".
- **Deferral**: ruvector semantic recognizer and real subprocess runner both land in P2.

---

## 5. Implementation phases

| Phase | Scope |
|-------|-------|
| **P1** (this ADR) | `intent`, `recognizer` (regex), `handler` (5 built-ins), `runner` (trait + noop), `pipeline` (end-to-end wiring), 10–15 tests |
| **P2** | Real `tokio::process::Child` runner with Windows-safe teardown; `SemanticIntentRecognizer` with ruvector HNSW |
| **P3** | STT/TTS bridge, satellite protocol, cloud fallback |
