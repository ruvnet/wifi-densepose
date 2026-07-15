# ADR-262: RuField MFS ↔ RuView integration — a live SensingServerAdapter, a privacy/provenance bridge, MAPPED not papered-over

| Field | Value |
|-------|-------|
| **Status** | Proposed — **P1 + P3 implemented** (live `/api/field` + `/ws/field`; P3 signs with a **dedicated dev/sensing key**, deferring the §8 Q1 `cog-ha-matter` key-ownership decision to P2) |
| **Date** | 2026-06-14 |
| **Deciders** | ruv |
| **Codebase target** | New thin bridge crate `wifi-densepose-rufield` (v2 workspace member); taps `wifi-densepose-sensing-server` emit path + `wifi-densepose-engine` `TrustedOutput`; depends on `vendor/rufield/crates/rufield-*` via path (the `vendor/rvcsi` pattern) |
| **Relates to** | ADR-260 (RuField MFS spec + v0.1 reference stack), ADR-261 (RuVector graph-ANN), ADR-141 (BFLD privacy control-plane / modes / attestation), ADR-137 (fusion-engine quality scoring / contradiction), ADR-032 (multistatic mesh security hardening / witness), ADR-116 (cog tamper-evident audit log — `cog-ha-matter` SHA-256+Ed25519), ADR-095/096 (`rvcsi` vendored-submodule precedent) |
| **Scope** | Decide **how** RuView's live WiFi-CSI sensing-server emits RuField `FieldEvent`s, **whether** RuView's ruvsense fusion composes with or is wrapped by rufield-fusion, and **how** to reconcile RuView's existing privacy/witness/provenance machinery with RuField's P0–P5 + ed25519 `ProvenanceReceipt`. The privacy/provenance reconciliation is the crux. |

---

## 0. PROOF discipline (this ADR's contract)

This project has been publicly accused of "AI slop." This ADR answers with **evidence, not adjectives** — every "RuView already does X" carries a `file:line`, and every external/SOTA claim is graded.

- **No accuracy is claimed.** RuField v0.1 is **SYNTHETIC** end-to-end by its own admission (ADR-260 "Honest statement", line 386–390: *"Every metric here is simulator-based. No ESP32 CSI, mmWave, or thermal capture was used."*). RuView's only real-CSI rufield path today would be **replay of recorded `.csi.jsonl`, unlabeled** — `rufield-adapters::CsiReplayAdapter`'s own module doc (`vendor/rufield/crates/rufield-adapters/src/csi_replay.rs:19-31`) states it is *"real signal, replay from file not live hardware, unlabeled ⇒ proxy not validated accuracy."* This ADR therefore proposes **plumbing**, and grades its own claims as "ARCHITECTURE" (a design decision, testable by a round-trip/compile gate) vs "ACCURACY" (which it explicitly does not assert).
- The privacy/provenance section reports an **honest conflict**: RuView has **three** witness mechanisms across two hash algorithms, and **two** privacy enums, none of which map 1:1 onto RuField's P0–P5. We map them and recommend the cleanest reconciliation rather than asserting they already align.
- Each phase below ships an **independently testable gate** (a round-trip test, a privacy-monotonicity test, a signature-verify test) so the integration is provable, not aspirational.

---

## 0.1 Implementation status

**P1 (§4) is implemented** as the `wifi-densepose-rufield` bridge crate (`v2/crates/wifi-densepose-rufield/`, a new v2 workspace member; path-deps the `vendor/rufield` submodule per §5.4):

- **Input** — `SensingSnapshot` (owned primitives mirroring `SensingUpdate` features/classification/signal_field joined with the `TrustedOutput` `trust_class`/`demoted`/`identity_bound`); the bridge does **not** depend on `wifi-densepose-sensing-server` (anti-corruption layer).
- **Conversion** — `snapshot_to_field_event(&snap, &Signer)` emits a signed `FieldEvent` (`Modality::WifiCsi`, axis `[Frequency]`, real `timestamp_ns`); position derived from the signal-field peak when present (never fabricated); real sha256 `ProvenanceRef` + ed25519 signature, `synthetic = false`.
- **Privacy (§3.3 crux)** — `map_privacy()` maps by information content, **fail-closed**: `Raw → P0`, `Derived → P4` (or `P5` if identity-bound — **never P1**), `Anonymous → P2`, `Restricted → P2`; a `demoted` cycle floors egress to ≥ P2.
- **Gates that pass** (`tests/p1_gates.rs`, 15 tests / 0 failed = 5 unit + 9 integration + 1 doc): round-trip (snapshot → `FieldEvent` → serde → equal); `is_fusable` (verified ed25519 receipt); `RuFieldFusion::ingest` accept + `infer()` runs; **privacy-safety** (`gate_privacy_safety_derived_never_maps_to_low_privacy` — `Derived → P4/P5`, never P1; full §3.3 table; fail-closed demotion); determinism (same snapshot + same signer seed → byte-identical event).

**P3 (§4) is implemented** as the live RuField surface in `wifi-densepose-sensing-server` (the bridge is now wired into the running server):

- **Tap** — at the ESP32 governed-trust cycle (`main.rs` `observe_cycle` ~`:5886` / `SensingUpdate` build ~`:5938`), a new `emit_rufield_event` joins the cycle's `SensingUpdate` (features / classification / signal_field) with the engine's recorded `effective_class` / `demoted` trust state into a `wifi_densepose_rufield::SensingSnapshot`, then `snapshot_to_field_event(&snap, &signer)`. Existing endpoints (`/ws/sensing` etc.) are **unchanged** — purely additive.
- **Surface** — `GET /api/field` (latest signed `FieldEvent`s + signer pubkey + a `dev_signing_key` flag) and `GET /ws/field` (broadcast stream, mirroring `/ws/sensing`), both mounted on the HTTP port and `/ws/field` also on the WS port. A small bounded ring buffer (`FIELD_RING_CAPACITY = 64`) holds recent **network-surfaced** events. New handler code lives in `src/rufield_surface.rs`, not in the 8k-line `main.rs`.
- **Signer (defers the P2 key decision)** — a **dedicated standalone `Signer`** held in server state, seeded from `WDP_RUFIELD_SIGNING_SEED` (64-hex or ≥32-byte value), else a deterministic dev default with a logged `WARN`. Reusing the `cog-ha-matter` Ed25519 key (§8 Q1) is the **deferred P2** decision — P3 uses a standalone sensing key so it does not pre-empt that call.
- **Egress privacy (fail-closed)** — `network_egress_allowed` is *stricter* than `DefaultPrivacyGuard` for an unattended live surface: only **P1/P2** leave the box; P0 (raw) and P3/P4/P5 (identity/biometric/aggregate above the default P2 ceiling) are held edge-local. A `Derived` cycle maps to P4/P5 and is therefore **never** surfaced. No-presence cycles emit nothing (no phantom events).
- **Gates that pass** (`tests/rufield_surface_test.rs`, 4 integration via `tower::oneshot` + 4 module unit, 0 failed): a well-formed **signed** event (`Modality::WifiCsi`, P2 not P1, `is_fusable` ed25519-verified, real timestamp); **empty cycle → no phantom**; **privacy-safety** — an injected `Derived` trust never surfaces on `/api/field`; a mixed stream surfaces only egress-safe events.

**Deferred:** the §3.3 *provenance carrier* recommendation (reuse the `cog-ha-matter` SHA-256+Ed25519 chain + embed the BLAKE3 engine witness) is **not** in P1/P3 — both take a dedicated `Signer` (the §8 open question 1 key-ownership decision is unresolved; P3 uses a standalone dev/sensing key precisely so it does not pre-empt P2). P2's `cog-ha-matter` key reuse + BLAKE3-embed, and P4 (multi-modality), remain future work. **No accuracy is claimed** (§0 / §6) — P1/P3 are tested plumbing on a live endpoint + a safe privacy mapping; the live surface is single-link CSI with its existing caveats (no validated room-coordinate accuracy — `field_localize`).

---

## 1. Context — two architectures, mapped

### 1.1 RuField MFS (ADR-260, `vendor/rufield/`)

A standalone pure-Rust Cargo workspace (serde, serde_json, toml, sha2, ed25519-dalek; **no tch/ndarray/candle**), vendored here as a git submodule (`git submodule status vendor/rufield` → `ba66e2e…`), **not** a v2 workspace member — exactly the `vendor/rvcsi` precedent (ADR-095/096). **Not published to crates.io**: every internal dep is a path dep with a nominal `version = "0.1.0"` (`vendor/rufield/Cargo.toml:31-37`); the `docs.rs/rufield-*` URLs are aspirational.

The data model (graded ARCHITECTURE, evidence read directly):

- **`FieldEvent`** (`vendor/rufield/crates/rufield-core/src/event.rs:96-112`): `spec_version, event_id, timestamp_ns: u64, sensor: SensorDescriptor, tensor: FieldTensor, observation: Observation, provenance: ProvenanceRef`.
- **`Observation`** (`event.rs:25-51`): `zone_id, space_cell, range_m, velocity_mps, motion_vector, confidence: f32, features: BTreeMap<String,f32>` (the derived P1 scalars the fusion engine actually reads), `labels: Vec<String>` (ground-truth, **never read by fusion**), `privacy_class: PrivacyClass`.
- **`PrivacyClass`** (`rufield-core/src/privacy.rs:8-25`): `P0..P5`, `#[serde(rename_all="UPPERCASE")]`, `Ord` by declaration order so **P0 < P1 < … < P5** — higher = more private; `level()->u8` returns 0..=5 (`privacy.rs:27-40`).
- **`ProvenanceRef`** (on-wire, `event.rs:73-93`): `raw_hash, firmware_hash` (`sha256:…`), `model_id, calibration_id, synthetic: bool`, optional `signature_hex` / `signer_pubkey_hex` (detached ed25519).
- The four traits (`rufield-core/src/traits.rs`): **`FieldAdapter`** (`:26-38`, `next_event() -> Result<Option<FieldEvent>>`), **`FieldEncoder`** (`:41-51`, **unimplemented in v0.1** — an open seam), **`FusionEngine`** (`:54-63`, `ingest(event)` + `infer(&query)`), **`PrivacyGuard`** (`:86-97`, `authorize(class, Destination, consent, identity_bound) -> PrivacyDecision{Allow|Deny|RequiresConsent}`).
- **`CsiReplayAdapter`** (`rufield-adapters/src/csi_replay.rs`): constructed from **already-loaded text** (`from_jsonl(&str)` `:249-251`; `from_jsonl_with(text, device_id, &[u8;32])` `:254-323`) — **not** a path/`Read`/`Iterator`. Deserializes `CsiFrameRecord { timestamp: f64 (seconds), subcarriers: Vec<f64> }` (`:74-80`), buffers all frames into a `Vec<CsiFrame>`, then streams via a cursor (`next_event` `:550-557`). Maps each frame → `FieldEvent` with `Modality::WifiCsi`, axes `[Frequency]`, a Welford motion proxy, observation `privacy_class = P2 if presence else P1` (`:439-443`), real `sha256` raw-hash, and a **real ed25519 signature** (`signer.sign_event` `:507-510`). `max_privacy_class = P2`.
- **`RuFieldFusion`** (`rufield-fusion/src/engine.rs:55-78`): `ingest()` **rejects non-fusable events on its first line** — `if !is_fusable(&event) { return Err(NotFusable) }` (`:212-215`) — then reads `event.observation.features` into a bounded temporal window; `infer()` applies TOML rules (`WeightedBayes` noisy-OR / `TemporalWindow`) → `Vec<FieldInference>`. TOML rule struct: `inputs, method, feature, threshold, privacy_max, window_ms, requires_consent` (`rules.rs:17-35`).
- **`is_fusable`** (`rufield-provenance/src/lib.rs:179-184`): `synthetic == true` **OR** `verify_event().is_ok()` — the §11 invariant. Signing key is `ed25519_dalek 2.1`, deterministic from a 32-byte seed; raw hash is `sha256_hex` → `"sha256:<hex>"` (`:26-35`).
- **`DefaultPrivacyGuard`** (`rufield-privacy/src/lib.rs:38-110`): default `network_max = P2`, `allow_p0_network = false`. P5-no-identity → `Deny`; P4-no-consent → `RequiresConsent`; `EdgeLocal` → `Allow`; `Network` denies P0 and `class > network_max`.
- **`rufield-viewer`** (Axum 0.7): **self-contained, consumes `SyntheticSim` only** — all routes are read-only GET/SSE (`GET /api/run`, `GET /events`); **there is no ingest endpoint** (`vendor/rufield/crates/rufield-viewer/src/server.rs:63-72`). Feeding it a live stream requires adding a route.

### 1.2 RuView (the integration target)

- **Sensing-server is Axum** (`v2/crates/wifi-densepose-sensing-server/src/main.rs:7498-7629`), two listeners (WS `:8765`, HTTP). CSI does **not** arrive over WS/HTTP — it arrives over **UDP** from ESP32 nodes (`use tokio::net::UdpSocket`, `main.rs:53`; `recv_from` loop `main.rs:5286-5299`), parsed by magic `0xC511_0001` → **`Esp32Frame`** (`types.rs:84-100`: `node_id, n_subcarriers, ppdu_type, amplitudes: Vec<f64>, phases: Vec<f64>`, rssi/freq/sequence) → pushed into per-node `NodeState.frame_history: VecDeque<Vec<f64>>` (`main.rs:441-497`).
- **`/ws/sensing` emits a `SensingUpdate`** (`main.rs:267-317`), broadcast over a `tokio::sync::broadcast` channel (`s.tx.send(json)` `main.rs:5938-5991`; the WS handler just subscribes and forwards, `main.rs:3021-3073`). `SensingUpdate` carries `nodes`, `features`, `classification {motion_level, presence, confidence}`, `signal_field`, `persons: Vec<PersonDetection>` (17 COCO keypoints + `position:[f64;3]` from `field_localize`, `main.rs:403-428`), pose, vitals. **`field_localize` (PR #1050) is a module, not a route** (`mod field_localize` `main.rs:17`; honesty caveat `field_localize.rs:16-27` — a single ESP32 link cannot resolve true room position, `position` is "strongest field peak").
- **ruvsense fusion is strictly WITHIN-WiFi-modality.** `MultistaticFuser::fuse(&[MultiBandCsiFrame]) -> FusedSensingFrame` (`v2/crates/wifi-densepose-signal/src/ruvsense/multistatic.rs:285-288`) attention-weights **multiple WiFi CSI nodes/viewpoints** (every input is ESP32 CSI; `multistatic_bridge.rs:50-62` builds the frames from `NodeState` amplitude with `HardwareType::Esp32S3`). `coherence_gate.rs:18-37` is the `GateDecision{Accept|PredictOnly|Reject|Recalibrate}`; `pose_tracker.rs:255-263` is the 17-keypoint Kalman tracker with 128-dim AETHER re-ID; `field_model.rs:301-308` does SVD room-eigenstructure perturbation extraction. **No camera/mmWave/audio enters this path** — ruvsense is a multi-link WiFi-CSI fuser.
- **The governed-trust cycle** runs in the separate **`wifi-densepose-engine`** crate. `StreamingEngine::process_cycle` (`v2/crates/wifi-densepose-engine/src/lib.rs:409`, `run_cycle` `:434-533`) produces **`TrustedOutput`** (`:82-112`): `semantic_id, quality: QualityScore, effective_class: PrivacyClass, demoted: bool, provenance: SemanticProvenance, witness: [u8;32]` (BLAKE3 over `evidence‖model‖calibration‖privacy_decision‖class`, `witness_of` `:598-613`), `recalibration_recommended`. **Crucially, none of this trust metadata is on the `SensingUpdate` wire today** — it is exposed only out-of-band on `GET /api/v1/status` (`main.rs:4173-4178`) and as a single live effect: `EngineBridge::suppress_raw_outputs()` strips per-node amplitude when `effective_class >= Restricted` (`engine_bridge.rs:240-243`, applied `main.rs:5908-5932`). The honest scope is stated in `engine_bridge.rs:14-27`: the governed engine runs *alongside* the bare fusion path; derived outputs are "published ungoverned."

---

## 2. Decision

1. **Build a thin RuView-side bridge crate `wifi-densepose-rufield`** (a new v2 workspace member) that depends on `vendor/rufield/crates/rufield-core` (+ `rufield-provenance`, `rufield-privacy`, `rufield-fusion`) **via path** — mirroring the `vendor/rvcsi` pattern. RuView does **not** depend on published rufield crates (there are none) and does **not** vendor rufield into the v2 workspace; rufield stays a standalone submodule and the bridge is the only coupling point (an anti-corruption layer).
2. **Emit `FieldEvent`s from the live server via an in-process `SensingServerAdapter`**, not by re-using the file-based `CsiReplayAdapter` on the hot path. The bridge taps the existing `SensingUpdate` build site and the `EngineBridge` trust state, joins them, and emits one signed `FieldEvent` per cycle on a new `tokio::broadcast` topic / optional `/ws/field` endpoint. `CsiReplayAdapter` is retained for the **offline/replay** path (recorded `.csi.jsonl` → events) because it already reads RuView's recording format (`recording.rs` writes `{session}.csi.jsonl`).
3. **Compose the two fusion engines vertically, do not merge them.** ruvsense stays the **WiFi-modality node** (multi-link fusion → one fused WiFi belief); rufield-fusion sits **above** it as the **cross-modality** graph. ruvsense's `FusedSensingFrame`/`TrustedOutput` becomes one `FieldEvent` (modality `wifi_csi`); rufield fuses it against future mmWave/thermal/`rvcsi` events. They do not conflict because ruvsense has no cross-modality fusion to collide with (§1.2 evidence).
4. **Reconcile privacy/provenance with ONE canonical model + a documented mapping** (§3, the crux): RuView's `effective_class` is the **source of truth**, mapped onto RuField `PrivacyClass` at the bridge; RuView's existing **`cog-ha-matter` SHA-256+Ed25519 witness chain** (already RuField's exact crypto) is adopted as the carrier for RuField `ProvenanceReceipt`, with the live BLAKE3 engine witness embedded as a hashed field. We do **not** maintain two parallel signed-receipt systems.

---

## 3. Privacy & provenance reconciliation (the crux)

This is the most important section. RuView and RuField genuinely **overlap and partially conflict**. We map both honestly.

### 3.1 What RuView actually has (implemented, with evidence)

- **TWO privacy enums, not one ladder.** `PrivacyClass` — **4 variants** `Raw=0, Derived=1, Anonymous=2, Restricted=3` (`v2/crates/wifi-densepose-bfld/src/lib.rs:103-116`, `#[repr(u8)]`, higher byte = more private, **non-monotonic in information** — `Derived=1` carries *more* identity than `Anonymous=2`). And `PrivacyMode` — **5 variants** `RawResearch, PrivateHome, EnterpriseAnonymous, CareWithConsent, StrictNoIdentity` (`bfld/src/privacy_mode.rs:18-31`), each mapping to a `PrivacyClass` via `target_class()` (`:63-70`; two modes collapse to `Anonymous`).
- **THREE witness mechanisms across TWO hash algorithms:**
  - BFLD `PrivacyAttestationProof` — **BLAKE3, unsigned**, attests mode/class continuity only; **built but NOT on the live path** (ADR-141 status line ~597; `bfld/src/privacy_mode.rs:121-148`).
  - Engine-cycle `TrustedOutput.witness: [u8;32]` — **BLAKE3, unsigned**, over the full trust decision; **LIVE every cycle** (`wifi-densepose-engine/src/lib.rs:598-613`).
  - `cog-ha-matter::WitnessChain` — **SHA-256 hash chain + Ed25519 signatures** (`v2/crates/cog-ha-matter/src/witness.rs:138-151`; `witness_signing.rs:39-76`), JSONL-persisted, `verify()` + `verify_signature()`. Implemented for ADR-116 (cog/Matter audit log); **standalone, not wired to BFLD/engine**. Its `WitnessHash` newtype doc explicitly anticipates a hash-algo migration (`witness.rs:37-41`).
- **No numeric trust score.** "Trust" in code = `base_coherence: f32∈[0,1]` + `penalized_coherence()` (`signal/.../fusion_quality.rs:99,122-126`) + a **boolean** `forces_privacy_demotion()` (`:116`). Demotion is monotonic and irreversible (`demote_one` clamps at Restricted, `engine/src/lib.rs:617-619`).
- **Structured provenance exists, but no signed "receipt" on the sensing path.** `SemanticProvenance { evidence, model_version, calibration_version, privacy_decision }` (`v2/crates/wifi-densepose-worldgraph/src/model.rs:137-147`) is attached to every belief and is the *input* to the BLAKE3 witness — but it is unsigned and not called a receipt.

### 3.2 Side-by-side, graded

| Dimension | RuView (file:line) | RuField | Alignment |
|---|---|---|---|
| Privacy ladder | `PrivacyClass` 4 (`bfld/lib.rs:103`) **or** `PrivacyMode` 5 (`bfld/privacy_mode.rs:18`) | `PrivacyClass` 6 (P0–P5, `rufield-core/privacy.rs:8`) | **PARTIAL→CONFLICT** — no clean 1:1; counts differ (4/5 vs 6); RuView class ordering non-monotonic |
| Demotion direction | higher = more private, irreversible (`engine/lib.rs:617`) | higher P# = more private, `Ord` by decl order (`privacy.rs:8-25`) | **STRONG** (same direction) |
| Provenance receipt | `SemanticProvenance` unsigned (`worldgraph/model.rs:137`) | `ProvenanceRef` + ed25519 (`event.rs:73`) | **PARTIAL** — structured but unsigned |
| Witness crypto (live path) | BLAKE3 `[u8;32]`, unsigned (`engine/lib.rs:598`) | sha256 + ed25519 (`rufield-provenance/lib.rs:26,135`) | **CONFLICT** (algo + signing) |
| Witness crypto (cog-ha-matter) | **SHA-256 + Ed25519** (`cog-ha-matter/witness.rs`, `witness_signing.rs`) | **sha256 + ed25519** | **STRONG** — RuField's exact crypto, already in-repo, but unwired and in another bounded context |
| Trust / confidence | `penalized_coherence: f32` + boolean demote (`fusion_quality.rs:122`) | `confidence: f32` per observation | **WEAK** — RuView has no graded trust object; confidence maps, demotion is binary |

### 3.3 The recommendation (the key call)

**Adopt ONE canonical model with a documented, lossy-but-monotonic mapping — do not run two parallel schemes.** Concretely:

1. **Privacy: RuView `effective_class` is the source of truth; the bridge maps it onto RuField `PrivacyClass`** at the egress boundary. The honest mapping (graded ARCHITECTURE — it is a *policy* decision, and it is **monotonicity-testable**, not an accuracy claim):

   | RuView `PrivacyClass` | → RuField | Rationale |
   |---|---|---|
   | `Raw` (raw CSI amplitude) | `P0` | raw waveform |
   | `Derived` (identity embedding, LAN-only) | `P4` *(or P5 if identity-bound)* | derived **identity** features ⇒ biometric/identity tier, **not** P1 — RuView's non-monotonic `Derived=1` is the trap; map by *information content*, not byte value |
   | `Anonymous` (occupancy/aggregate) | `P2`/`P3` | occupancy → P2, room-count aggregate → P3 |
   | `Restricted` (zeroized) | `P2`-capped, raw suppressed | matches `suppress_raw_outputs` (`engine_bridge.rs:240`) |

   The bridge **must** map `Derived → P4/P5`, never P1, because RuView's `Derived` carries `identity_embedding` (§3.1) — this is the single most dangerous mapping mistake and gets a dedicated test (P2 in §4). `PrivacyMode` (5) is the better *operator-facing* join to RuField's 6 levels but the **class** is what gates egress, so the class mapping is canonical.

2. **Provenance: adopt `cog-ha-matter`'s SHA-256+Ed25519 chain as the carrier for RuField `ProvenanceReceipt`** — it is already RuField's exact crypto (graded STRONG above), already implemented, already tamper-evident. The bridge constructs the RuField `ProvenanceRef` by: `raw_hash = sha256(csi bytes)`, `model_id`/`calibration_id` from `SemanticProvenance`, and **embeds the live BLAKE3 engine witness `[u8;32]` as a hashed provenance field** (it is already computed every cycle — do not throw it away), then **signs with ed25519** so `is_fusable` passes for live (non-synthetic) events. We do **not** add a second BLAKE3-vs-ed25519 argument: BLAKE3 stays RuView's internal fast cycle-fingerprint; ed25519 is the *external* attestation RuField requires. One signer, one chain.

3. **Trust: map `penalized_coherence` → `Observation.confidence`; keep demotion binary.** RuView has no graded trust object to reconcile; the coherence scalar is the honest analog and the demotion boolean already drives `effective_class`.

This is a **bridge-with-canonical-source**, not "keep both forever." RuView owns the privacy decision (it has the live governed cycle); RuField owns the *external wire shape* (P0–P5 + signed receipt). The bridge is the one-directional translation, and it is the only place the two schemes meet.

---

## 4. Phased plan (each phase independently shippable + testable)

**P1 — `SensingServerAdapter` emitting `FieldEvent`s (ARCHITECTURE).**
New crate `wifi-densepose-rufield` with a `SensingServerAdapter` that consumes a `(SensingUpdate, TrustedOutput)` pair (tapped at `main.rs:5886`/`:5938`) and emits a signed `FieldEvent` (`Modality::WifiCsi`, axes `[Frequency]`, observation features from `SensingUpdate.features`, `confidence` from `penalized_coherence`). Offline path: keep `CsiReplayAdapter` for recorded `.csi.jsonl`. **Gate:** a round-trip test — emit a `FieldEvent` from a fixture `SensingUpdate`, assert it serializes, `is_fusable` passes (ed25519-signed), and `RuFieldFusion::ingest` accepts it. No server changes required beyond exposing the tap; the adapter is a library.

**P2 — privacy/provenance bridge (the crux, ARCHITECTURE).**
Implement the §3.3 mapping: `effective_class → PrivacyClass`, `cog-ha-matter` ed25519 signer for the receipt, BLAKE3 witness embedded. **Gates (three, all monotonicity/safety, not accuracy):** (a) `Derived → P4|P5` never P1 (the dangerous-mapping test); (b) privacy monotonicity — `demoted == true` ⇒ emitted `PrivacyClass >= P2` and raw suppressed; (c) signature round-trip — sign with the cog-ha-matter key, `rufield_provenance::verify_event` passes. This phase is shippable without P3 (events emitted on an internal topic, not yet on the public wire).

**P3 — surface in `/ws` + viewer (ARCHITECTURE).**
Add an opt-in `/ws/field` endpoint (or a `field_events` array on `SensingUpdate` behind a flag) carrying the signed `FieldEvent` + a privacy badge. Add an ingest route to `rufield-viewer` (it has none today — `server.rs:63-72`) so it can replay RuView's live feed instead of only `SyntheticSim`. **Gate:** a WS integration test asserting a connected client receives a privacy-badged, signature-verifiable `FieldEvent`; a viewer test asserting the new ingest route renders a live event. The `cognitum` appliance can speak RuField by consuming this endpoint (it already runs `ruview-vitals-worker`); deferred to its own ADR.

**P4 — fusion composition + multi-modality (ARCHITECTURE, optional).**
Wire a second modality (cheapest: an `rvcsi`-sourced event, or recorded mmWave) into `RuFieldFusion` alongside the WiFi event, proving cross-modality fusion above ruvsense. **Gate:** a fusion test with two modalities producing ≥1 cross-modal inference, with provenance coverage 100%.

---

## 5. Decision matrix

### 5.1 Data-path emission (P1)

| Option | Latency | Reuse | Live-fit | Risk | Verdict |
|---|---|---|---|---|---|
| Re-use `CsiReplayAdapter` on hot path | poor (file buffer, `&str` ctor) | high | **bad** — it's a file-cursor, not a live source | low | **Reject for live** (keep for replay) |
| In-process `SensingServerAdapter` (tap `SensingUpdate`+`TrustedOutput`) | good | medium | **good** — taps the real emit + real trust state | low | **CHOSEN** |
| Server publishes `FieldEvent` on its own topic (no adapter trait) | good | low | good | medium (bypasses `FieldAdapter` contract) | Reject — loses the trait seam |

### 5.2 Fusion relationship (P3/P4)

| Option | Verdict |
|---|---|
| Merge ruvsense into rufield-fusion | **Reject** — different scopes; ruvsense is within-WiFi multi-link, rufield is cross-modality |
| rufield-fusion wraps ruvsense (vertical compose) | **CHOSEN** — ruvsense → one WiFi `FieldEvent` → rufield cross-modality graph |
| Run both as peers, reconcile after | Reject — duplicates fusion semantics, two contradiction models |

### 5.3 Privacy/provenance reconciliation (P2)

| Option | Verdict |
|---|---|
| (a) Map RuView classes onto RuField P0–P5, RuView canonical | **CHOSEN (privacy)** — `effective_class` is the live source of truth |
| (b) Adopt RuField ed25519 receipts as RuView's provenance | **CHOSEN (provenance)** — via the already-present `cog-ha-matter` SHA-256+Ed25519 chain |
| (c) Keep both schemes with a permanent bridge | **Reject** — two signed-receipt systems is the duplication we must not ship |

### 5.4 Dependency direction

| Option | Verdict |
|---|---|
| Depend on published rufield crates | **Reject** — not published (`vendor/rufield/Cargo.toml:31-37`) |
| Make rufield a v2 workspace member | **Reject** — breaks the standalone-spec/`rvcsi` precedent |
| Thin `wifi-densepose-rufield` bridge → path deps on submodule | **CHOSEN** — anti-corruption layer, single coupling point |

---

## 6. Security & honesty notes

- **No accuracy claim.** Live RuField events from RuView are derived from the same single-link CSI whose own caveats are on record (`field_localize.rs:16-27`); the offline path is unlabeled replay (`csi_replay.rs:19-31`). This ADR ships **plumbing with monotonicity/signature gates**, not validated F1.
- **The dangerous mapping is `Derived → P1`.** RuView's `Derived` byte value (1) is numerically below `Anonymous` (2) but carries identity (`bfld/lib.rs`); a naive byte-mapping would leak identity-bearing features as low-privacy P1. P2's gate (a) exists specifically to prevent this.
- **One signer, not two.** Adding a second ed25519 keypair alongside `cog-ha-matter`'s would create two roots of trust. The bridge reuses the cog-ha-matter signing key (`witness_signing.rs`).
- **`is_fusable` is a real gate, not decoration** (`rufield-provenance/lib.rs:179-184`): live events that fail to sign are rejected by `RuFieldFusion::ingest` — we must not paper over a signing failure with `synthetic = true` on a real event (that would be the §11 invariant violation the spec forbids).
- BLAKE3 stays internal; ed25519 is the external attestation. We do not relitigate RuView's BLAKE3 cycle-witness — it is embedded, not replaced.

## 7. Consequences

**Positive:** RuView becomes one honest adapter in the larger RuField ecosystem (ADR-260 goal §9) without forking its fusion or privacy engine; the three witness mechanisms get a single external attestation path; cross-modality fusion becomes possible above the existing WiFi fusion; the `cognitum` appliance gains a standard wire format. The bridge is the only coupling point, so rufield can evolve as a standalone spec.

**Negative:** a fourth crate to maintain; the privacy mapping is lossy (4/5 → 6) and must be kept honest by tests; reusing the `cog-ha-matter` key crosses a bounded-context boundary (cog/Matter ↔ sensing) that ADR-116 kept separate — that coupling needs review. The live trust metadata (`witness`, `effective_class`) is **currently decoupled** from `SensingUpdate` (§1.2), so P1 must do real join work, not a field read.

## 8. Open questions

1. **Signer ownership:** should the bridge reuse the `cog-ha-matter` Ed25519 key, or mint a dedicated RuView-sensing key with its own rotation? (Reuse couples bounded contexts; a new key adds a second root of trust.)
2. **`PrivacyMode` vs `PrivacyClass` as the canonical map target:** class gates egress (chosen), but the 5-mode ladder is the cleaner join to 6 levels — do we expose mode in the receipt too?
3. **Where does the BLAKE3 engine witness live in the RuField receipt** — a `firmware_hash`-style field, an extension field, or a `CalibrationReceipt.data_hash`? (RuField's `ProvenanceRef` has no spare slot; needs a spec extension or reuse of `model_id`.)
4. **Should `field_localize` positions ride in `Observation.space_cell`/`motion_vector`** given the explicit single-link caveat, or stay RuView-only until multi-node calibration lands?
5. **`rvcsi` relationship:** `rvcsi` has its own `CsiFrame`/`CsiWindow` and could implement `FieldAdapter` directly — should the second modality in P4 be `rvcsi`, making RuField the convergence point for *both* vendored sensing runtimes?
6. **Transport:** RuField ADR-260 §29 leaves default transport open (MQTT/NATS/WS/MCP). RuView is WS + UDP + broadcast; does `/ws/field` suffice, or does the appliance need MQTT to match the cog stack?

## 9. Recommendation

Proceed with P1+P2 behind a feature flag. They are independently shippable, carry real gates (round-trip, monotonicity, signature-verify), and require no change to RuView's fusion or privacy engine — only a tap and a translation. Defer P3/P4 and the appliance/transport questions to follow-up ADRs once the bridge round-trips on recorded `.csi.jsonl` and on one live cycle.
