# ADR-140: Semantic State Record Schema, Versioning, and Ruflo Agent Bridge

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see Implementation Status, commit `169a355bd`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-sensing-server/src/semantic/` (`bus.rs`, `common.rs`); `homecore/src/state.rs` + `event.rs`; `homecore-assist` |
| **Relates to** | ADR-115 (HA Integration / HA-MIND semantic primitives), ADR-127 (HOMECORE State Machine), ADR-129 (HOMECORE Automation Engine), ADR-133 (HOMECORE-ASSIST + Ruflo), ADR-136 (RuView Streaming Engine / FrameMeta), ADR-137 (Fusion Engine Quality Scoring / Evidence Refs), ADR-139 (WorldGraph Digital Twin), ADR-141 (BFLD Privacy Control Plane), ADR-021 (ESP32 Vital Signs), ADR-125 (Apple Home Native HAP Bridge) |

---

## 1. Context

### 1.1 The Gap

The HA-MIND semantic primitive layer landed under ADR-115 §3.12 and lives in `v2/crates/wifi-densepose-sensing-server/src/semantic/`. It is a real, tested, ten-primitive inference layer: `bus.rs` owns a `SemanticBus` that dispatches one `RawSnapshot` to each of ten FSMs (`sleeping`, `distress`, `room_active`, `elderly_anomaly`, `meeting`, `bathroom`, `fall_risk`, `bed_exit`, `no_movement`, `multi_room`) and collects `SemanticEvent`s. Each `SemanticEvent` carries exactly four fields (`bus.rs:44-50`):

```rust
pub struct SemanticEvent {
    pub kind: SemanticKind,
    pub state: PrimitiveState,
    pub node_id: String,
    pub timestamp_ms: i64,
}
```

and `PrimitiveState` (`common.rs:36-47`) is one of `Boolean { active, changed, reason }`, `Scalar { value, reason }`, `Event { event_type, reason }`, or `Idle`. The only provenance a downstream consumer receives today is the `Reason` tag list (`common.rs:50-65`) — a `Vec<String>` of human-readable debug strings such as `["motion<5%", "br=12bpm"]`.

That is the gap this ADR closes. Searching the workspace confirms three concrete absences:

- **No version provenance on a published state.** Grepping `v2/crates/` for `model_version` and `calibration_version` finds matches only in `wifi-densepose-bfld` and `wifi-densepose-signal` (frame-level metadata), never in the `semantic/` module. A `SemanticEvent` for `fall_risk_elevated` carries no record of *which* model or *which* empty-room baseline (ADR-135) produced it. A caregiver-escalation automation acting on that event cannot audit whether the signal came from a calibrated node or a stale one.
- **No `evidence_refs`, `confidence`, `expiry_at`, or `privacy_action` on a state.** `SemanticEvent` has no field tying its assertion back to the signal evidence that justified it, no machine-readable confidence (only the `Reason` tag strings), no time-to-live, and no privacy classification. `PrimitiveConfig` (`common.rs:71-100`) holds per-primitive thresholds but no per-primitive model/calibration metadata, and `Default` (`common.rs:102-122`) hardcodes them — there is no manifest load path.
- **No `Rest`/inactivity `SemanticKind`.** The `SemanticKind` enum (`bus.rs:29-41`) has ten variants. Inactivity is currently expressed only through `NoMovement` (`no_movement.rs`), which fires a *safety* signal (`presence == true` AND motion < 0.01 for ≥ 30 min — a possible-collapse alarm), and `ElderlyInactivityAnomaly`. Neither expresses the benign, expected state of a person at rest (reading, watching TV). Automations that want to *suppress* lighting/HVAC changes during rest have no primitive to subscribe to; they must reverse-engineer it from the absence of `RoomActive`, which is fragile.

The privacy boundary is likewise under-specified at the state layer. `mqtt/privacy.rs` makes a binary `PublishDecision::{Publish, Suppress}` keyed solely on `EntityKind::is_biometric()` and a global `--privacy-mode` flag (`privacy.rs:33-39`). Semantic primitives are always `Publish` in that path (`privacy.rs:84-102`) because they are inferred states, not raw biometrics. But there is no per-record privacy *action* — no way to say "publish this `BathroomOccupied` state but anonymize the room", or "strip the biometric attributes from this `PossibleDistress` while keeping the boolean". The privacy decision is made once, globally, at the wire boundary, and is invisible to the record itself.

Finally, the **Ruflo agent bridge** exists only as a P1 stub. `homecore-assist/src/runner.rs` defines the `RufloRunner` trait and a `NoopRunner` that returns an empty `RufloResponse` (`runner.rs:113-139`); the crate doc (`lib.rs:24-27`) explicitly defers the real subprocess runner and semantic embedding recognizer to P2/P3. There is no path today by which a `SemanticEvent` (or a *combination* of them) reaches a Ruflo agent so that an automation can route on **multi-signal agreement** — e.g. `fall_risk_elevated` AND `elderly_inactivity_anomaly` together escalating to a caregiver, which neither primitive can decide alone.

### 1.2 What "Semantic State Record" Means Here

A `SemanticStateRecord` is the unified, versioned, auditable envelope that every primitive emits *instead of* the bare `SemanticEvent`. It is the inference-layer analogue of what ADR-136 calls a `FrameMeta` at the signal layer and what ADR-137 calls an evidence-scored fusion output: a state assertion that carries its own provenance. It captures:

- **What** was asserted: the `SemanticKind`, the `PrimitiveState`, the `room`, and the `Reason` tags.
- **How confident**: a normalized `confidence ∈ [0, 1]` distinct from the human `Reason` tags.
- **From which model and calibration**: `model_version` and `calibration_version`, threaded from the ADR-136 `FrameMeta` of the frames that produced the snapshot.
- **Backed by what evidence**: `evidence_refs`, opaque handles into the ADR-137 fusion evidence store (and, where relevant, the ADR-139 WorldGraph node IDs).
- **For how long it is valid**: `expiry_at` — the wall-clock instant past which the record must not be acted upon without refresh.
- **Under what privacy classification**: `privacy_action`, an enum that *the record carries*, enforced downstream at the MQTT/Matter boundary.

What a `SemanticStateRecord` is **not**: it is not a replacement for the per-primitive FSMs, the `Reason` explainability contract, or the existing `--privacy-mode` wire filter. It is the schema that wraps their output so the rest of the system (HOMECORE state machine, automation engine, Ruflo agents, the recorder) can reason about provenance.

### 1.3 The Provenance Rule

This ADR honours the project-wide rule that **every semantic state traces to signal evidence + model version + calibration version + privacy decision.** Today a `SemanticEvent` honours none of those four. After this ADR, a `SemanticStateRecord` carries all four as first-class fields, and the witness/proof chain (ADR-028 style) can assert that no record reaches an HA controller without them.

### 1.4 Pipeline Position

```
CSI frames (per node)
  → signal pipeline → FrameMeta { model_version, calibration_version } (ADR-136)
  → fusion engine → quality score + evidence_refs (ADR-137)
  → RawSnapshot (semantic/common.rs)              ← unchanged projection
  → SemanticBus::tick()                            ← still runs 10+1 FSMs
  → SemanticStateRecord::from_event(meta, ev)      ← NEW: wraps each SemanticEvent
        carries model_version, calibration_version, confidence,
        room, evidence_refs, expiry_at, privacy_action
  ├─→ MQTT / Matter publisher  → privacy_action enforced at boundary (ADR-141 maps mode→action)
  ├─→ HOMECORE StateMachine::set()  → state_changed broadcast (ADR-127)
  │       → AutomationEngine triggers (ADR-129)
  └─→ SemanticAgentBridge::route()  ← NEW: feeds agreeing records to Ruflo (ADR-133)
          → RufloRunner::send_request()  → caregiver escalation / multi-signal automation
```

The `SemanticBus` is unchanged except that `tick()` returns records instead of bare events; the FSMs themselves do not move. The new code is the record wrapper, the manifest loader, the `Rest` primitive, and the agent bridge.

---

## 2. Decision

### 2.1 The `SemanticStateRecord` Schema

A new struct in `semantic/common.rs`, the canonical output type of the bus. It wraps the existing `SemanticKind` + `PrimitiveState` + `Reason` without changing them.

```rust
use std::time::{Duration, SystemTime};

/// Privacy classification carried by every record. The *action* is
/// chosen at the state layer; the *enforcement* happens at the MQTT /
/// Matter boundary (mqtt/privacy.rs). The mode→action mapping is owned
/// by ADR-141 (BFLD Privacy Control Plane); this enum is the action
/// vocabulary it maps onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyAction {
    /// Publish the record verbatim (room, attributes, all tags).
    Allow,
    /// Publish state + confidence, but replace `room` with a coarse
    /// bucket ("upstairs", "downstairs", or "home") before the wire.
    AnonymizeByRoom,
    /// Publish the boolean/scalar state only; drop any attribute that
    /// derives from a biometric channel (HR/BR-derived tags) and any
    /// evidence_ref. Used for healthcare deployments.
    StripBiometrics,
}

/// Opaque handle into the ADR-137 fusion evidence store, or an ADR-139
/// WorldGraph node id. Records what justified the assertion without
/// embedding the evidence itself (keeps records small + privacy-safe).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    /// "fusion" | "worldgraph" | "vitals" | "cir" — the producing layer.
    pub source: &'static str,
    /// Stable id within that source (e.g. fusion clip id, graph node id).
    pub id: String,
}

/// Versioned, auditable envelope around one primitive's output.
///
/// This is the inference-layer analogue of ADR-136's FrameMeta. It is
/// the type the SemanticBus emits and the type every downstream
/// consumer (MQTT, Matter, HOMECORE StateMachine, Ruflo bridge,
/// recorder) sees.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticStateRecord {
    // ---- what was asserted -------------------------------------------
    pub kind: SemanticKind,
    pub state: PrimitiveState,        // unchanged enum (Boolean/Scalar/Event/Idle)
    pub node_id: String,
    pub timestamp_ms: i64,
    /// Room/zone this assertion is scoped to. None for whole-home
    /// primitives (e.g. MultiRoom). Drawn from RawSnapshot.active_zones
    /// or the ADR-139 WorldGraph room node.
    pub room: Option<String>,

    // ---- how confident -----------------------------------------------
    /// Normalized confidence in [0,1], distinct from the Reason tags.
    /// Derived per-primitive (see §2.6); 1.0 for deterministic FSM
    /// transitions, < 1.0 when the producing fusion score was degraded.
    pub confidence: f32,

    // ---- provenance: model + calibration -----------------------------
    /// Threaded from ADR-136 FrameMeta of the frames behind this snapshot.
    pub model_version: String,
    /// Empty-room baseline version (ADR-135). "uncalibrated" if no
    /// baseline was loaded for node_id.
    pub calibration_version: String,
    /// Evidence handles (ADR-137 / ADR-139). Empty for pure-FSM
    /// transitions that used only RawSnapshot scalars.
    pub evidence_refs: Vec<EvidenceRef>,

    // ---- validity + privacy ------------------------------------------
    /// Wall-clock instant past which this record must not be acted upon
    /// without refresh. Computed as timestamp + per-kind TTL (§2.4).
    pub expiry_at: SystemTime,
    /// Privacy classification (enforced downstream, §2.3).
    pub privacy_action: PrivacyAction,
}
```

**Why a wrapper, not a field-extension of `SemanticEvent`.** `SemanticEvent` is a value type already serialized to the MQTT/Matter publishers and exercised by the proptest suite in `bus.rs` (the `bus_events_carry_node_id_and_ts` and `boolean_states_always_have_reason_tags` invariants). Replacing it outright would churn those tests. Instead, `SemanticEvent` becomes the *inner* assertion and `SemanticStateRecord` the *outer* envelope; the bus constructs records, and a `record.as_event()` accessor reproduces the old four-field shape for any caller that has not migrated. The proptest invariants are preserved verbatim and a new invariant — "every record carries a non-empty `model_version` and `calibration_version`" — is added.

### 2.2 Constructing a Record: `from_event`

The bus does not change the FSMs. It changes the assembly step in `SemanticBus::tick()` (`bus.rs:86-111`): the `filter_map` that builds `SemanticEvent`s now builds `SemanticStateRecord`s.

```rust
impl SemanticStateRecord {
    /// Wrap one primitive's event with the provenance from the frame
    /// metadata that produced the snapshot.
    pub fn from_event(
        ev: SemanticEvent,
        meta: &SnapshotMeta,        // see §2.6 — threaded with RawSnapshot
        cfg: &PrimitiveConfig,
    ) -> Self {
        let ttl = cfg.record_ttl(ev.kind);           // §2.4
        Self {
            kind: ev.kind,
            state: ev.state,
            node_id: ev.node_id,
            timestamp_ms: ev.timestamp_ms,
            room: meta.room.clone(),
            confidence: meta.confidence_for(ev.kind), // §2.6
            model_version: meta.model_version.clone(),
            calibration_version: meta.calibration_version.clone(),
            evidence_refs: meta.evidence_refs.clone(),
            expiry_at: meta.captured_at + ttl,
            privacy_action: cfg.privacy_action_for(ev.kind),
        }
    }

    /// Reproduce the legacy four-field event for un-migrated callers.
    pub fn as_event(&self) -> SemanticEvent {
        SemanticEvent {
            kind: self.kind,
            state: self.state.clone(),
            node_id: self.node_id.clone(),
            timestamp_ms: self.timestamp_ms,
        }
    }
}
```

`SnapshotMeta` is a small companion struct attached to each `RawSnapshot` carrying `model_version`, `calibration_version`, `evidence_refs`, `room`, `captured_at: SystemTime`, and the per-kind confidence inputs. It is populated by the snapshot projection step that already builds `RawSnapshot` from the `VitalsSnapshot` + `sensing_update` broadcast (`common.rs:5-33`). When the upstream frame metadata is absent (e.g. a synthetic test snapshot), `SnapshotMeta::unknown()` supplies `model_version = "unknown"`, `calibration_version = "uncalibrated"`, empty `evidence_refs`, and `confidence = 1.0` for deterministic FSM transitions — so existing tests that build a bare `RawSnapshot::default()` still pass.

### 2.3 `privacy_action` Semantics and the Boundary Contract

The record carries `privacy_action`, but the record layer **does not** redact anything. Redaction is enforced exactly where it is today — in `mqtt/privacy.rs` at the wire boundary — extended from a binary decision to one keyed on the record's action:

```rust
pub enum PublishDecision {
    Publish,                       // unchanged: send verbatim
    Suppress,                      // unchanged: drop silently
    Redact(PrivacyAction),         // NEW: send, but apply the action's transform
}

pub fn decide_record(rec: &SemanticStateRecord, mode_default: bool) -> PublishDecision {
    match rec.privacy_action {
        PrivacyAction::Allow            => PublishDecision::Publish,
        PrivacyAction::AnonymizeByRoom  => PublishDecision::Redact(PrivacyAction::AnonymizeByRoom),
        PrivacyAction::StripBiometrics  => PublishDecision::Redact(PrivacyAction::StripBiometrics),
    }
}
```

The existing biometric `EntityKind` filter (`privacy.rs:33-39`) is unchanged and runs first: raw HR/BR/pose entities are still `Suppress`ed under global `--privacy-mode`. The new `decide_record` path applies *only* to `SemanticStateRecord`s, which were never biometric and were always `Publish` (`privacy.rs:84-102`). The record's action therefore adds granularity *within* the always-published semantic class — it cannot weaken the existing global biometric suppression.

**The mode→action mapping is explicitly delegated to ADR-141.** This ADR defines the *action vocabulary* (`Allow`/`AnonymizeByRoom`/`StripBiometrics`) and the enforcement point. ADR-141 (BFLD Privacy Control Plane) owns the named privacy *modes* and the policy that maps a deployment's mode plus the primitive kind onto one of these actions — and the runtime attestation that the mapping was applied. `PrimitiveConfig::privacy_action_for(kind)` is the seam: in this ADR it returns a static default (`Allow` for all kinds, preserving today's behaviour); ADR-141 replaces the seam with its policy engine without re-touching the record schema.

### 2.4 Per-Kind TTL and `expiry_at`

`expiry_at` is computed as the record's `captured_at` plus a per-kind TTL drawn from `PrimitiveConfig`. The TTLs reflect each primitive's physical timescale, not a single global value, because acting on a stale `bed_exit` (a one-shot event) is very different from acting on a stale `someone_sleeping` (a sustained state).

| Kind | TTL | Rationale |
|------|-----|-----------|
| `BedExit`, `MultiRoom`, `FallRisk` (event) | 30 s | One-shot events; a consumer that acts more than 30 s late is acting on history, not state. |
| `RoomActive`, `BathroomOccupied`, `Rest` | 90 s | Occupancy states refresh on the 30 s `room_active_window`; 3× window before considered stale. |
| `SomeoneSleeping`, `NoMovement` | 10 min | Slow-changing states; the FSM dwell is minutes-to-hours. |
| `PossibleDistress`, `ElderlyAnomaly` | 5 min | Safety states; short enough that a missed refresh self-clears rather than persisting a false alarm. |
| `FallRisk` (scalar) | 5 min | Continuous score; recomputed every tick, so a 5 min TTL is generous. |

`record_ttl(kind)` returns these as `Duration`s; the values are config fields with the table above as `Default`. A consumer that reads a record past `expiry_at` MUST treat it as "unknown", not as the last asserted value — this is the contract the HOMECORE state machine and the automation engine rely on to avoid acting on stale safety states after a sensor outage.

### 2.5 The `Rest` Primitive — an Explicit v2 `SemanticKind`

The `SemanticKind` enum (`bus.rs:29-41`) gains one variant in this ADR:

```rust
pub enum SemanticKind {
    SomeoneSleeping, PossibleDistress, RoomActive, ElderlyAnomaly,
    Meeting, BathroomOccupied, FallRisk, BedExit, NoMovement, MultiRoom,
    Rest,   // NEW (v2)
}
```

`Rest` is the benign, expected inactivity state of a present, awake person (reading, watching TV): `presence == true` AND `motion < room_active_motion_threshold` AND NOT `someone_sleeping` AND breathing rate present and in the awake band, sustained for a dwell. It is added as a new primitive file `semantic/rest.rs` with its own FSM and tests, registered in the bus exactly as the existing ten are (one file change per the §3.12.6 "adding a primitive is one file change" contract documented in `mod.rs:18-22`).

**Why not alias `no_movement`.** `NoMovement` (`no_movement.rs`) is a *safety* primitive: it fires after 30 minutes of near-zero motion as a possible-collapse alarm, and the project doc (`no_movement.rs:1-6`) frames it that way. Aliasing `Rest` to it would conflate "person resting comfortably" with "person possibly collapsed" — the exact distinction caregivers need. `Rest` has a *shorter* dwell, a *higher* motion ceiling, and an explicit "awake breathing" gate, and crucially it carries the opposite automation intent: `Rest` should *suppress* environmental changes (don't turn the lights off on someone reading), whereas `NoMovement` should *escalate*. They are different states with different downstream consumers and must be different `SemanticKind`s.

**Deferral.** The remaining proposed v2 primitives — `child-play`, `pet-vs-human`, `agitation-gradient`, `circadian-phase` — are explicitly deferred to a follow-on ADR. They each require new signal inputs not present in `RawSnapshot` today (per-person classification embeddings, multi-day circadian baselines persisted across restart). `Rest` is the only v2 primitive that can be built from the existing `RawSnapshot` fields, so it is the only one promoted here.

### 2.6 Confidence Derivation and the Manifest

`confidence ∈ [0,1]` is per-record and per-kind. The rule:

1. A deterministic FSM transition that used only `RawSnapshot` scalars (e.g. `bed_exit` time-gate crossing) yields `confidence = 1.0` — the FSM is exact given its inputs.
2. When the producing snapshot carried an ADR-137 fusion quality score (degraded link, contradiction flag), `confidence` is the product of `1.0` and that fusion score, clamped to `[0,1]`. A `BathroomOccupied` derived from a node whose fusion score was 0.6 yields `confidence = 0.6`.
3. When the snapshot was produced on an `"uncalibrated"` node (no ADR-135 baseline), confidence is capped at `0.8` to flag that motion/amplitude thresholds were absolute rather than baseline-relative.

`PrimitiveConfig` is extended to load per-primitive **model/calibration metadata from a manifest**, so that the `model_version` and `calibration_version` stamped onto every record are auditable rather than hardcoded. Today `PrimitiveConfig::default()` hardcodes thresholds (`common.rs:102-122`); this ADR adds an optional manifest:

```rust
/// Loaded once at startup from `--semantic-manifest-file` (TOML). Maps a
/// model/calibration identity onto each primitive so records are auditable.
#[derive(Debug, Clone, Default)]
pub struct PrimitiveManifest {
    /// e.g. "ha-mind-v2.1" — the semantic-layer model bundle version.
    pub model_version: String,
    /// Build commit hash of the sensing-server that produced records.
    pub commit_hash: String,
    /// ISO-8601 date the model bundle was trained/released.
    pub model_date: String,
    /// Per-node calibration versions, keyed by node_id, from ADR-135
    /// baseline files. "uncalibrated" when absent.
    pub calibration_versions: std::collections::HashMap<String, String>,
}

impl PrimitiveConfig {
    pub fn manifest(&self) -> &PrimitiveManifest;          // NEW field accessor
    pub fn record_ttl(&self, kind: SemanticKind) -> Duration;       // §2.4
    pub fn privacy_action_for(&self, kind: SemanticKind) -> PrivacyAction; // §2.3
}
```

The manifest TOML:

```toml
[model]
version = "ha-mind-v2.1"
commit_hash = "850463818"
date = "2026-05-28"

[calibration]
"esp32s3-com9" = "baseline-2026-05-28T14:32:00Z"
"cognitum-seed-1" = "baseline-2026-05-27T09:10:00Z"
# nodes absent here are stamped "uncalibrated"
```

When no `--semantic-manifest-file` is supplied, `PrimitiveManifest::default()` stamps `model_version = "unknown"`, `commit_hash = ""`, and every node as `"uncalibrated"` — identical observable behaviour to today, but now explicit on every record.

### 2.7 The Ruflo Agent Bridge (ADR-133 Integration Path)

This ADR defines the path by which `SemanticStateRecord`s reach a Ruflo agent so that automations can route on **multi-signal agreement** — agreement no single primitive can decide. The motivating case: `FallRisk` (elevated) AND `ElderlyAnomaly` (firing) within a short window in the same room ⇒ caregiver escalation. `fall_risk.rs` cannot see `elderly_anomaly`'s state, and vice versa; only an aggregator over records can.

The bridge is a new component, `SemanticAgentBridge`, in `homecore-assist` (alongside the existing `RufloRunner` trait in `runner.rs`). It does **not** replace the voice/intent pipeline — it reuses the same `RufloRunner` subprocess transport.

```rust
/// Subscribes to the SemanticStateRecord stream and routes agreeing
/// records to a Ruflo agent for multi-signal automation decisions.
/// Reuses the existing RufloRunner transport (homecore-assist/runner.rs).
pub struct SemanticAgentBridge<R: RufloRunner> {
    runner: R,
    rules: Vec<AgreementRule>,
    /// Sliding window of recent records per (room, kind).
    recent: RecordWindow,
}

/// A multi-signal agreement that, when satisfied, sends a payload to the
/// agent. Declarative so ADR-129 automations and ADR-141 policy can
/// extend the set without code changes.
pub struct AgreementRule {
    pub name: &'static str,
    /// All of these kinds must have a *fresh* (non-expired), active
    /// record scoped to the same room within `window`.
    pub require: Vec<SemanticKind>,
    pub window: Duration,
    /// Minimum confidence each constituent record must clear.
    pub min_confidence: f32,
    /// Intent name handed to the Ruflo agent on satisfaction.
    pub agent_intent: &'static str,
}

impl<R: RufloRunner> SemanticAgentBridge<R> {
    /// Ingest one record. If it completes an AgreementRule, build a
    /// JSON payload (records + their provenance) and call
    /// RufloRunner::send_request(). Returns the agent's RufloResponse
    /// when a rule fired, else None.
    pub async fn route(&mut self, rec: SemanticStateRecord)
        -> Result<Option<RufloResponse>, AssistError>;
}
```

The default rule set ships one rule:

```rust
AgreementRule {
    name: "caregiver_escalation",
    require: vec![SemanticKind::FallRisk, SemanticKind::ElderlyAnomaly],
    window: Duration::from_secs(120),
    min_confidence: 0.7,
    agent_intent: "HassCaregiverEscalate",
}
```

**Provenance is mandatory on the agent payload.** The JSON sent to the agent via `send_request()` (`runner.rs:86-89`) includes, for each constituent record, its `model_version`, `calibration_version`, `confidence`, `room`, and `evidence_refs`. This is the project provenance rule applied to the agent boundary: the agent never sees a bare "fall risk is high" — it sees "fall risk is high, confidence 0.82, model ha-mind-v2.1, node esp32s3-com9 calibrated baseline-2026-05-28, evidence fusion#clip-1841." An agent declining or confirming an escalation does so against an auditable record.

**P1/P2 staging.** With the existing `NoopRunner` (`runner.rs:113-139`), `route()` returns `Ok(None)` and the bridge falls back to a deterministic local decision (fire the escalation event directly into the HOMECORE state machine). When the real subprocess `RufloRunner` lands (ADR-133 P2, `runner.rs:9-18` deferral), `route()` consults the agent. The bridge is written against the trait, so no bridge code changes when the runner is swapped — mirroring how the assist pipeline already swaps `NoopRunner` for the real runner.

### 2.8 Bridge to HOMECORE State Machine

`SemanticStateRecord`s also flow into the HOMECORE `StateMachine` (`homecore/src/state.rs`) so that ADR-129 automations can trigger on them via the existing `state_changed` broadcast. The mapping:

- Each record becomes a `StateMachine::set(entity_id, state, attributes, context)` call (`state.rs:75-110`). The `entity_id` is `binary_sensor.<room>_<kind>` (or `sensor.` for `FallRisk`), matching the HA entity naming the MQTT discovery already uses.
- The record's provenance (`model_version`, `calibration_version`, `confidence`, `expiry_at`, `privacy_action`, `evidence_refs`) is serialized into the `attributes: serde_json::Value` so it survives into the `StateChangedEvent` (`event.rs:101-106`) and is queryable by automations and the recorder.
- The `Context` (`event.rs:42-69`) is stamped with the bridge as origin so automations can detect and avoid self-trigger loops, exactly as HA's context does.

The HOMECORE state machine already suppresses no-op writes (`state.rs:92-99`); a record whose `state` and `attributes` are unchanged from the prior write does not re-fire the broadcast, so a primitive emitting the same `Scalar` confidence every tick does not spam the channel. A record's `expiry_at` is written into attributes; a consumer reading state past that instant treats it as `unknown` (§2.4).

### 2.9 Interface Boundaries (Summary)

| Boundary | Type crossing it | Owner |
|----------|------------------|-------|
| signal → semantic | `RawSnapshot` + `SnapshotMeta` (model/calibration/evidence) | `semantic/common.rs` (ADR-136 supplies meta) |
| semantic bus output | `SemanticStateRecord` | `semantic/bus.rs` (this ADR) |
| semantic → MQTT/Matter | `SemanticStateRecord` → `PublishDecision` | `mqtt/privacy.rs` (this ADR; mapping by ADR-141) |
| semantic → HOMECORE | `SemanticStateRecord` → `StateMachine::set` | `homecore/src/state.rs` (this ADR) |
| semantic → Ruflo | agreeing records → JSON payload → `RufloRunner::send_request` | `homecore-assist` `SemanticAgentBridge` (this ADR; transport from ADR-133) |
| legacy callers | `SemanticStateRecord::as_event()` → `SemanticEvent` | back-compat shim (this ADR) |

### 2.10 Test Plan

**Tier 1 — Record construction is total (unit test, `common.rs`).** For every `SemanticKind` variant (now 11 including `Rest`) and every non-`Idle` `PrimitiveState`, `SemanticStateRecord::from_event` produces a record with a non-empty `model_version`, non-empty `calibration_version`, a finite `confidence ∈ [0,1]`, and an `expiry_at > timestamp`. Assert `as_event()` round-trips the four legacy fields exactly.

**Tier 2 — Provenance proptest (extend `bus.rs` proptest suite).** Reuse the existing `arb_snapshot()` strategy. Assert a new invariant alongside the existing ones (`bus_events_carry_node_id_and_ts`, `boolean_states_always_have_reason_tags`): **every emitted `SemanticStateRecord` carries a non-empty `model_version` and `calibration_version`**, and `confidence` is in `[0,1]`. This wires the provenance rule into the property suite that already guards the bus.

**Tier 3 — Default behaviour unchanged (unit test).** With `PrimitiveManifest::default()` and `privacy_action_for` returning `Allow`, assert `decide_record` returns `Publish` for all 11 kinds — i.e. zero observable change from today's `privacy.rs:84-102` behaviour. This is the no-regression gate.

**Tier 4 — `Rest` distinct from `NoMovement` (unit test, `rest.rs`).** Feed a sequence: present, awake breathing (br ≈ 14 bpm), motion 0.05 for 3 minutes. Assert `Rest` fires `Boolean { active: true }` and `NoMovement` stays `Idle` (its 30-min dwell is not met and motion ≥ 0.01). Then drop motion to 0.005 for 30 minutes and assert `NoMovement` fires while `Rest` exits — proving the two states are not aliases.

**Tier 5 — TTL / staleness (unit test).** Build a `FallRisk` event record and a `SomeoneSleeping` record. Assert `expiry_at - captured_at == 30 s` and `10 min` respectively (per §2.4 table). Assert a helper `record.is_expired(now)` returns `true` past `expiry_at`.

**Tier 6 — `privacy_action` enforcement (unit test, `mqtt/privacy.rs`).** For a record with `privacy_action = AnonymizeByRoom`, assert `decide_record` returns `Redact(AnonymizeByRoom)` and that the redaction transform replaces `room = "bedroom"` with a coarse bucket. For `StripBiometrics`, assert HR/BR-derived `Reason` tags and `evidence_refs` are removed while the boolean state survives. For `Allow`, verbatim publish.

**Tier 7 — Multi-signal agreement bridge (async unit test, `homecore-assist`).** With a `NoopRunner`, feed a `FallRisk` record then an `ElderlyAnomaly` record for the same room within 120 s, both `confidence ≥ 0.7`. Assert `route()` recognises the `caregiver_escalation` rule and (since the runner is a no-op) falls back to firing the escalation locally. Feed the same two records > 120 s apart and assert no escalation. Feed them in *different* rooms and assert no escalation.

**Tier 8 — HOMECORE state-machine bridge (async unit test).** Route a record into a `StateMachine`; subscribe; assert a `StateChangedEvent` (`event.rs:101-106`) fires whose `new_state` attributes contain `model_version`, `calibration_version`, `confidence`, and `expiry_at`. Route an identical record again; assert the no-op suppression (`state.rs:92-99`) yields no second event.

### 2.11 Witness / Proof

Per ADR-028, three rows are added to `docs/WITNESS-LOG-028.md`:

| Row | Capability | Evidence |
|-----|-----------|----------|
| W-39 | Every `SemanticStateRecord` carries model + calibration version (proptest invariant) | `cargo test -p wifi-densepose-sensing-server semantic::` proptest passes |
| W-40 | `privacy_action` enforced at the MQTT boundary (Allow/AnonymizeByRoom/StripBiometrics) | `cargo test mqtt::privacy::tests::decide_record_*` passes |
| W-41 | Multi-signal agreement routes to Ruflo bridge (fall_risk + elderly_anomaly → escalation) | `cargo test -p homecore-assist bridge::tests::caregiver_escalation` passes |

`source-hashes.txt` in the witness bundle gains the SHA-256 of `semantic/common.rs`, `semantic/rest.rs`, and the new bridge module.

---

## 3. Consequences

### 3.1 Positive

- **Auditable states.** Every published semantic state now traces to a model version, a calibration version, signal evidence, and a privacy decision. A caregiver-escalation automation can refuse to act on records from an `"uncalibrated"` node, closing the silent-degradation hole where an uncalibrated node's absolute thresholds produced unreliable states with no flag.
- **Privacy granularity without weakening the existing guarantee.** The `privacy_action` enum adds room-anonymization and biometric-stripping *within* the always-published semantic class, while the existing global biometric `Suppress` filter (`privacy.rs`) is untouched and still runs first. Healthcare deployments gain `StripBiometrics` per-record without a new wire schema.
- **Multi-signal automations become possible.** The agent bridge enables decisions no single primitive can make (`fall_risk` + `elderly_anomaly` → caregiver), reusing the existing `RufloRunner` transport rather than inventing a new IPC path.
- **`Rest` unblocks suppression automations.** Automations can finally subscribe to "person resting comfortably" and suppress environmental changes, instead of fragilely inferring it from the absence of `RoomActive`.
- **Back-compatible.** `SemanticEvent` is preserved as the inner type; `as_event()` and `PrimitiveManifest::default()` mean un-migrated callers and existing tests observe no behaviour change.

### 3.2 Negative

- **Larger records on the wire.** A `SemanticStateRecord` carries five new fields plus `evidence_refs`. For high-rate `Scalar` primitives (`fall_risk` publishes every tick) this is more bytes; the HOMECORE no-op suppression (`state.rs:92-99`) and the per-kind TTL mitigate the rate, but MQTT payloads grow.
- **Manifest is a new operational artifact.** Operators must supply `--semantic-manifest-file` to get meaningful `model_version`/`calibration_version`; absent it, every node is stamped `"uncalibrated"`. This is not a regression (today there is no version at all) but it is a new step to get full auditability.
- **Bridge couples two crates.** `homecore-assist` now depends on the `SemanticStateRecord` type from the sensing server. The dependency is one-directional (assist depends on the semantic schema, not vice versa) and the schema is small, but it is a new cross-crate edge.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Confidence derivation is gamed by always returning 1.0 | Medium | Records look more trustworthy than they are; uncalibrated nodes' states acted on blindly | §2.6 caps confidence at 0.8 on `"uncalibrated"` nodes and multiplies by the ADR-137 fusion score; Tier 2 proptest asserts `confidence ∈ [0,1]` but a separate review must confirm the per-kind derivation is honest |
| Agreement rule fires on coincidental co-occurrence | Medium | Spurious caregiver escalation | `min_confidence` gate + same-room scoping + 120 s window; the agent (when present) makes the final call with full provenance, declining low-evidence escalations |
| `expiry_at` consumers ignore it and act on stale safety states | Low | Acting on a post-outage stale `possible_distress` | The contract is documented (§2.4) and the HOMECORE attributes carry `expiry_at`; Tier 5 tests `is_expired`; recorder can flag consumers that read past expiry |
| ADR-141 mode→action mapping not yet built; `privacy_action` defaults to `Allow` everywhere | High (until ADR-141 lands) | No room-anonymization until the policy engine ships | `privacy_action_for` seam returns `Allow` (today's behaviour) until ADR-141 replaces it; no record-schema change needed when it does |

---

## 4. Alternatives Considered

### 4.1 Extend `SemanticEvent` In Place Instead of Wrapping

Add the five provenance fields directly to `SemanticEvent`. Rejected: `SemanticEvent` is already serialized to MQTT/Matter and is the subject of five proptest invariants in `bus.rs`. Mutating it churns the wire format and the tests simultaneously. The wrapper + `as_event()` shim isolates the change, keeps the proptest suite green, and lets callers migrate incrementally.

### 4.2 Put Provenance in the `Reason` Tags

`Reason` is already a `Vec<String>` (`common.rs:50-65`); one could append `"model=ha-mind-v2.1"` tags. Rejected: tags are human-readable debug strings, not a machine schema. An automation would have to string-parse tags to find the model version, which is brittle and untyped. Provenance must be typed fields so consumers and the recorder can query them structurally.

### 4.3 Alias `Rest` to `NoMovement`

Reuse `NoMovement` for the rest state with a different threshold. Rejected in §2.5: `NoMovement` is a *safety/escalation* primitive (possible collapse), `Rest` is a *suppression* primitive (don't disturb). They carry opposite automation intent and different dwell/motion semantics; conflating them would make it impossible for an automation to distinguish "resting" from "possibly collapsed" — the exact distinction caregivers need.

### 4.4 Route All Records to the Agent

Send every `SemanticStateRecord` to the Ruflo agent and let the LLM decide everything. Rejected: most records (a single `room_active` toggle) need no LLM reasoning, and the agent subprocess (ADR-133) has a 5 s timeout (`runner.rs:51`) and per-call cost. The declarative `AgreementRule` set filters to the multi-signal cases that actually need cross-primitive reasoning, keeping the single-signal path deterministic and free.

### 4.5 Enforce Privacy at the Record Layer

Have `SemanticStateRecord` redact itself (drop `room`, strip biometrics) before publishing. Rejected: redaction must happen at the wire boundary so the same record can be published differently to different transports (full to a local trusted HOMECORE state machine, anonymized to an external MQTT broker). The record carries the *action*; `mqtt/privacy.rs` applies the *transform* per transport. This also keeps the enforcement point co-located with the existing biometric filter, so ADR-141's attestation can verify one place.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-115 (HA Integration / HA-MIND) | **Extended**: the ten §3.12 semantic primitives now emit `SemanticStateRecord`s; the `SemanticEvent` becomes the inner assertion |
| ADR-127 (HOMECORE State Machine) | **Consumer**: records bridge into `StateMachine::set` and surface as `StateChangedEvent` attributes |
| ADR-129 (HOMECORE Automation Engine) | **Consumer**: automations trigger on record attributes (confidence, expiry_at) via the state_changed broadcast |
| ADR-133 (HOMECORE-ASSIST + Ruflo) | **Path defined**: `SemanticAgentBridge` reuses the `RufloRunner` transport; multi-signal agreement routes records to the agent |
| ADR-135 (Empty-Room Calibration) | **Provenance source**: `calibration_version` is the ADR-135 baseline file version per node |
| ADR-136 (Streaming Engine / FrameMeta) | **Provenance source**: `model_version` and `calibration_version` thread from the ADR-136 `FrameMeta` |
| ADR-137 (Fusion Quality / Evidence Refs) | **Provenance source**: `evidence_refs` are handles into the ADR-137 evidence store; `confidence` multiplies the fusion quality score |
| ADR-139 (WorldGraph) | **Provenance source**: `room` and some `evidence_refs` resolve to ADR-139 WorldGraph node ids |
| ADR-141 (BFLD Privacy Control Plane) | **Delegates**: ADR-141 owns the mode→`PrivacyAction` mapping and runtime attestation; this ADR defines the action vocabulary and enforcement point |
| ADR-021 (ESP32 Vital Signs) | **Substrate**: HR/BR channels are the biometrics `StripBiometrics` strips and the awake-breathing gate `Rest` consumes |
| ADR-125 (Apple Home Native HAP Bridge) | **Consumer**: records reaching the HOMECORE state machine surface as HAP characteristics; `privacy_action` governs what the HAP bridge exposes |

---

## 6. References

### Production Code

- `v2/crates/wifi-densepose-sensing-server/src/semantic/bus.rs` — `SemanticBus`, `SemanticEvent`, `SemanticKind` (the bus this ADR wraps)
- `v2/crates/wifi-densepose-sensing-server/src/semantic/common.rs` — `RawSnapshot`, `PrimitiveState`, `Reason`, `PrimitiveConfig` (the schema home for `SemanticStateRecord`)
- `v2/crates/wifi-densepose-sensing-server/src/semantic/mod.rs` — the "adding a primitive is one file change" contract (§3.12.6) `Rest` follows
- `v2/crates/wifi-densepose-sensing-server/src/semantic/no_movement.rs` — the safety primitive `Rest` must not be aliased to
- `v2/crates/wifi-densepose-sensing-server/src/semantic/fall_risk.rs`, `elderly_anomaly.rs` — the two primitives whose agreement drives caregiver escalation
- `v2/crates/wifi-densepose-sensing-server/src/mqtt/privacy.rs` — `PublishDecision`, `decide`; extended with `decide_record` and `Redact`
- `v2/crates/homecore/src/state.rs` — `StateMachine::set`, no-op suppression, `state_changed` broadcast
- `v2/crates/homecore/src/event.rs` — `StateChangedEvent`, `Context`, `EventType`
- `v2/crates/homecore-assist/src/runner.rs` — `RufloRunner` trait + `NoopRunner`; transport reused by `SemanticAgentBridge`
- `v2/crates/homecore-assist/src/lib.rs` — ADR-133 P1 scope and the P2 deferral the bridge stages against
- `v2/crates/homecore-recorder/src/semantic.rs` — semantic index that will record record provenance (ADR-132 path)

### Related ADRs (this series)

- `docs/adr/ADR-136-ruview-streaming-engine-frame-contracts.md` — `FrameMeta` source of `model_version` / `calibration_version`
- `docs/adr/ADR-137-fusion-engine-quality-scoring-evidence.md` — evidence references and contradiction flags feeding `evidence_refs` + `confidence`
- `docs/adr/ADR-139-worldgraph-environmental-digital-twin.md` — room/node resolution for `room` and graph `evidence_refs`
- `docs/adr/ADR-141-bfld-privacy-control-plane-modes-attestation.md` — owns the mode→`PrivacyAction` mapping and attestation


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `169a355bd`, issue #844): `SemanticStateRecord` (provenance-carrying), `PrivacyAction`, and the `MultiSignalRule` agent bridge that fires only on multi-signal agreement. 4 tests.

**Integration glue -- not yet on the live path:** the `Rest` `SemanticKind` (deferred to avoid an enum-match cascade); subscribing `route_all()` to the broadcast bus -> ADR-133 HOMECORE-ASSIST; and loading the per-primitive model/calibration manifest into `RecordContext`.

**Trust contribution:** high-stakes actions (caregiver escalation) require *multiple independent signals to agree*, and every emitted record carries model + calibration + privacy provenance and an expiry.
