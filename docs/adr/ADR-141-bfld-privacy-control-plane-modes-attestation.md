# ADR-141: BFLD Privacy Control Plane: Named Modes, Actions, and Runtime Attestation

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see Implementation Status, commit `7d88eb84c`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-bfld` (new module `mode.rs` + `attestation.rs`; extends `lib.rs` `PrivacyClass`, `sink.rs`, `privacy_gate.rs`, `identity_risk.rs`, `emitter.rs`, `ha_discovery.rs`) |
| **Relates to** | ADR-010 (Witness Chains), ADR-118 (BFLD), ADR-120 (Privacy Class + Hash Rotation), ADR-121 (Identity-Risk Scoring), ADR-122 (RuView HA/Matter Exposure), ADR-136 (Streaming Engine), ADR-139 (WorldGraph), ADR-140 (Semantic State Record), ADR-143 (RF SLAM v2) |

---

## 1. Context

### 1.1 The Gap

The BFLD crate (`v2/crates/wifi-densepose-bfld/src/`) already implements a complete, structurally enforced privacy posture, but it does so entirely in terms of a **4-value numeric class** — there is no first-class concept of a deployment *mode* and no concept of a discrete privacy *action*. Reading the real code:

- `lib.rs` defines `PrivacyClass` as `#[repr(u8)]` with four variants `Raw = 0`, `Derived = 1`, `Anonymous = 2`, `Restricted = 3`, plus `allows_network()` / `allows_matter()` / `as_u8()` (`lib.rs:82-117`). This is the entire vocabulary the system has for "what is this deployment allowed to emit." Nothing names *why* a node is at class 2 vs class 3, nor records which privacy transformations were actually applied.
- `privacy_gate.rs` implements `PrivacyGate::demote()` — a monotonic, zeroizing transformer that strips payload sections (`compressed_angle_matrix`, `csi_delta`, `amplitude_proxy`, `phase_proxy`) on each class transition (`privacy_gate.rs:31-75`). The stripping is real and irreversible, but it is **silent**: nothing records *which* sections were zeroed for *which* frame. There is no audit trail and no way for a downstream verifier to prove what was stripped.
- `sink.rs` enforces I1 at compile time via `Sink::MIN_CLASS` and the runtime `check_class::<S>()` (`sink.rs:47-55`), with the three concrete `LocalKind`/`NetworkKind`/`MatterKind` tags. The MQTT topic router (`mqtt_topics.rs:109-157`) and HA discovery (`ha_discovery.rs:61-129`) hard-code the rule "publish only at class >= Anonymous, and `identity_risk` only at exactly Anonymous." This is an *implicit ACL* scattered across two files; it is not declared in one place and is not bound to a named mode.
- `identity_risk.rs` defines `GateAction { Accept, PredictOnly, Reject, Recalibrate }` (`identity_risk.rs:57-69`) — but these are *risk-gating* actions on a per-event basis, not *privacy* actions. There is no enum that names the privacy transformation a mode enforces (e.g., "suppress identity", "drop raw", "aggregate only").
- `emitter.rs` hard-codes `privacy_class: PrivacyClass::Anonymous` as the constructed default (`emitter.rs:82`) and the Soul Signature gate is controlled only by whether a `SoulMatchOracle` is supplied (`emitter.rs:138`, `coherence_gate.rs:71`). Whether Soul Signature is *enabled* for a deployment is not a declared policy — it is an implicit consequence of construction-site wiring.

The consequence: a deployment's privacy stance is encoded in **four separate places** — the constructed `PrivacyClass`, the presence/absence of a `SoulMatchOracle`, the class-gated MQTT/HA fan-out, and the `signature_hasher` install — with no single declared object that says "this node runs in *CareWithConsent* mode, which means class Derived, Soul Signature enabled, identity_risk published, raw never networked." There is no runtime artifact a regulator, a Home Assistant dashboard, or the WorldGraph (ADR-139) can read to learn the *effective* policy, and no cryptographic proof that the policy was actually enforced frame-by-frame.

ADR-140 (Semantic State Record) requires that every semantic state trace to a `privacy_action`. ADR-139 (WorldGraph) needs a `privacy_limited_by` annotation to compute which edges/zones are degraded by privacy. Neither has anything to bind to today: BFLD exposes a numeric class but no *action* and no *attestation*. This ADR closes that gap.

### 1.2 What "Mode", "Action", and "Attestation" Mean Here

- A **PrivacyMode** is a named, operator-facing deployment posture (e.g., `CareWithConsent`). It is the human-meaningful unit a regulator or installer reasons about. It is *not* a new enforcement primitive — it is a declarative selection that *maps to* the existing `PrivacyClass`, plus a Soul Signature gate decision, plus an MQTT/Matter ACL.
- A **PrivacyAction** is the discrete, machine-checkable privacy transformation that a mode enforces (e.g., `SuppressIdentity`, `DropRaw`). Actions are the bridge between the human mode and the byte-level stripping `privacy_gate.rs` already performs. They are what ADR-140's `privacy_action` field carries.
- A **PrivacyAttestationProof** is a hash-chained record (per ADR-010) of *which mode was active, which actions were enforced, and which fields were stripped per event*. It is the cryptographic continuity proof that the declared mode was honored, surfaced read-only to HA/Matter diagnostics.

What this ADR is **not**: it does not change the four `PrivacyClass` byte values, does not weaken any structural invariant (I1/I2/I3 from `lib.rs:8-11`), and does not replace `PrivacyGate::demote()` — it *records* what `demote()` did.

### 1.3 Pipeline Position

```
SensingInputs
  → BfldEmitter::emit()                       (identity_risk + CoherenceGate)
       ↑ consults
  PrivacyModeRegistry::active_mode()          ← NEW
       ↓ resolves to (PrivacyClass, Soul gate, ACL)
  → PrivacyGate::demote(frame, target_class)  (existing; now records stripped fields)
       ↓ emits per-frame
  PrivacyActionRecord { actions, fields_stripped }  ← NEW
       ↓ folded into
  PrivacyAttestationProof { mode, actions, fields_stripped_per_event, prev_hash }  ← NEW (hash-chained, ADR-010)
       ↓ surfaced
  mqtt_topics.rs / ha_discovery.rs            (active mode + proof hash diagnostic entity)
       ↓ consumed by
  ADR-139 privacy_limited_by  /  ADR-140 privacy_action
```

The registry is consulted once per class transition (not once per byte). The attestation chain is appended per emitted event window, not per frame, to bound chain growth (see §2.5).

---

## 2. Decision

### 2.1 `PrivacyMode`: Five Named Variants Layered Over `PrivacyClass`

Introduce `PrivacyMode` in a new module `mode.rs`. It is a *semantic abstraction* over the existing 4-class `PrivacyClass`; it adds zero new enforcement bytes on the wire.

```rust
// v2/crates/wifi-densepose-bfld/src/mode.rs

use crate::PrivacyClass;

/// Operator-facing deployment posture. Maps deterministically to a
/// `PrivacyClass`, a Soul Signature gate decision, and an MQTT/Matter ACL via
/// the `PrivacyModeRegistry`. Adds no new wire bytes — `PrivacyClass` remains
/// the only byte carried in `BfldFrameHeader`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrivacyMode {
    /// Local research: raw BFI retained, never networked. Maps to `Raw`.
    RawResearch = 0,
    /// Single-home production: anonymous sensing, Soul Signature OFF.
    /// Maps to `Anonymous`, no per-day rf_signature_hash.
    PrivateHome = 1,
    /// Multi-tenant / enterprise: anonymous + per-seed salt rotation so no
    /// two seeds can correlate. Maps to `Anonymous`, multiseed salt domain.
    EnterpriseAnonymous = 2,
    /// Care deployment with explicit consent: identity-derived fields enabled
    /// behind consent. Maps to `Derived`, Soul Signature ON.
    CareWithConsent = 3,
    /// Regulated / no-identity: strictest posture. Maps to `Restricted`.
    StrictNoIdentity = 4,
}

impl PrivacyMode {
    /// The `PrivacyClass` this mode resolves to. This is the *only* coupling
    /// to the existing enforcement layer.
    #[must_use]
    pub const fn privacy_class(self) -> PrivacyClass {
        match self {
            Self::RawResearch => PrivacyClass::Raw,
            Self::PrivateHome | Self::EnterpriseAnonymous => PrivacyClass::Anonymous,
            Self::CareWithConsent => PrivacyClass::Derived,
            Self::StrictNoIdentity => PrivacyClass::Restricted,
        }
    }

    /// Whether Soul Signature (`SignatureHasher` install + non-`Null` oracle)
    /// is enabled in this mode. See `emitter.rs:138` / `coherence_gate.rs:71`.
    #[must_use]
    pub const fn soul_signature_enabled(self) -> bool {
        matches!(self, Self::CareWithConsent)
    }

    /// Whether per-seed (multiseed) salt isolation is required so two seeds
    /// in the same site produce uncorrelated `rf_signature_hash` (invariant I3,
    /// `signature_hasher.rs:8-18`). Enterprise turns this on; single-home does not.
    #[must_use]
    pub const fn multiseed_salt(self) -> bool {
        matches!(self, Self::EnterpriseAnonymous)
    }

    /// Stable string token used in TOML config, MQTT diagnostics, and the
    /// attestation proof. Lowercase snake form of the variant.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::RawResearch => "raw_research",
            Self::PrivateHome => "private_home",
            Self::EnterpriseAnonymous => "enterprise_anonymous",
            Self::CareWithConsent => "care_with_consent",
            Self::StrictNoIdentity => "strict_no_identity",
        }
    }
}
```

The decision to keep `PrivacyMode` separate from `PrivacyClass` (rather than collapsing the two into a 5-variant class) is deliberate: `PrivacyClass` is a wire/sink-enforcement primitive with byte semantics relied on by `frame.rs`, `sink.rs::check_class`, and the on-NVS/MQTT representation. Two of the five modes (`PrivateHome`, `EnterpriseAnonymous`) resolve to the *same* class (`Anonymous`) but differ in salt domain — they are not separable at the class layer. Modes are a strictly higher-level concept and must not perturb the existing byte contract.

### 2.2 `PrivacyAction`: The Enforced-Transformation Vocabulary

```rust
// v2/crates/wifi-densepose-bfld/src/mode.rs (continued)

/// A discrete privacy transformation a mode enforces. These are the
/// machine-checkable bridge between a human `PrivacyMode` and the byte-level
/// stripping already performed by `PrivacyGate::demote()` (`privacy_gate.rs`).
///
/// ADR-140's semantic-state `privacy_action` field carries the *strongest*
/// action enforced for the event that produced the state.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PrivacyAction {
    /// No transformation: the frame is published as-is at its class.
    Allow = 0,
    /// Strip identity-derived fields (`identity_risk_score`, `rf_signature_hash`)
    /// — the `Restricted` strip in `event.rs:112-117`.
    SuppressIdentity = 1,
    /// Down-sample the angle/CSI surface (the `compressed_angle_matrix` /
    /// `csi_delta` zeroing in `privacy_gate.rs:48-55`).
    ReduceResolution = 2,
    /// Refuse to network a `Raw` frame (structural invariant I1, `sink.rs:35`).
    DropRaw = 3,
    /// Emit only aggregate sensing (presence/motion/count/confidence); no
    /// per-subject or per-cluster surface leaves the node.
    AggregateOnly = 4,
}
```

`PrivacyAction` is `Ord` so a per-event set can be reduced to its **strongest** action for ADR-140's single-valued `privacy_action` field (the maximum). The actions are intentionally orthogonal to `GateAction` (`identity_risk.rs:57`): `GateAction` answers "is this *event* too risky to publish?"; `PrivacyAction` answers "what privacy transformation does the active *mode* require on every event?" They compose — a mode may enforce `SuppressIdentity` while the per-event gate independently `Reject`s.

### 2.3 `PrivacyModeRegistry`: Single Source of Truth + Append-Only Audit Log

The registry is the one declared object that the gap (§1.1) is missing. It owns the active mode, the mode→actions mapping, the ACL, and an append-only audit log that the witness verifier can replay.

```rust
// v2/crates/wifi-densepose-bfld/src/mode.rs (continued)

use crate::sink::Sink;

/// Declares the active mode and the policy it implies. Consulted by the
/// emitter/gate on every class transition. Holds an append-only, witness-
/// checkable audit log of every mode resolution and action enforcement.
#[derive(Debug)]
pub struct PrivacyModeRegistry {
    active: PrivacyMode,
    /// Append-only; never mutated in place. Each entry is hashed into the
    /// attestation chain (§2.5).
    audit_log: Vec<ModeAuditEntry>,
}

/// One append-only audit record. ADR-010 §"Hash chain" linkage is applied at
/// the `PrivacyAttestationProof` layer, not here — this is the raw event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModeAuditEntry {
    /// Monotonic capture-clock ns (matches `BfldEvent::timestamp_ns`).
    pub timestamp_ns: u64,
    /// Mode active at the moment of this transition/resolution.
    pub mode: PrivacyMode,
    /// Class the mode resolved to.
    pub resolved_class: PrivacyClass,
    /// The set of actions enforced, sorted ascending (Ord), deduplicated.
    pub actions_enforced: Vec<PrivacyAction>,
}

impl PrivacyModeRegistry {
    /// Build a registry pinned to `mode`. The production-safe default is
    /// `PrivateHome` (resolves to `Anonymous`, matching `emitter.rs:82`).
    #[must_use]
    pub fn new(mode: PrivacyMode) -> Self {
        Self { active: mode, audit_log: Vec::new() }
    }

    /// The currently active mode.
    #[must_use]
    pub const fn active_mode(&self) -> PrivacyMode {
        self.active
    }

    /// The set of actions this mode enforces, sorted ascending. Pure function
    /// of `active` — the canonical mode→actions mapping (§2.4 table).
    #[must_use]
    pub fn enforced_actions(&self) -> Vec<PrivacyAction> {
        actions_for(self.active)
    }

    /// Whether a specific action is enforced under the active mode. This is the
    /// predicate ADR-139/ADR-140 query to decide `privacy_limited_by` and
    /// `privacy_action`.
    #[must_use]
    pub fn is_action_enforced(&self, action: PrivacyAction) -> bool {
        actions_for(self.active).contains(&action)
    }

    /// Whether the active mode's class may cross sink `S`. Re-uses the
    /// existing compile-time ACL (`sink.rs::check_class`). This is the
    /// declared-in-one-place MQTT/Matter ACL the gap (§1.1) lacked.
    #[must_use]
    pub fn allows_sink<S: Sink>(&self) -> bool {
        crate::sink::check_class::<S>(self.active.privacy_class()).is_ok()
    }

    /// Record a class transition / resolution into the append-only log and
    /// return the entry that was appended (so the caller can fold it into the
    /// attestation chain). Called by the emitter on every transition.
    pub fn record_transition(&mut self, timestamp_ns: u64) -> &ModeAuditEntry {
        let entry = ModeAuditEntry {
            timestamp_ns,
            mode: self.active,
            resolved_class: self.active.privacy_class(),
            actions_enforced: actions_for(self.active),
        };
        self.audit_log.push(entry);
        self.audit_log.last().expect("just pushed")
    }

    /// Read-only view of the audit log for the witness verifier.
    #[must_use]
    pub fn audit_log(&self) -> &[ModeAuditEntry] {
        &self.audit_log
    }
}

/// Canonical mode→actions mapping (§2.4). Pure, total, `const`-friendly.
#[must_use]
pub fn actions_for(mode: PrivacyMode) -> Vec<PrivacyAction> {
    use PrivacyAction::{Allow, AggregateOnly, DropRaw, ReduceResolution, SuppressIdentity};
    let v = match mode {
        PrivacyMode::RawResearch => vec![Allow], // local-only; I1 still blocks network in sink.rs
        PrivacyMode::PrivateHome => vec![SuppressIdentity, DropRaw],
        PrivacyMode::EnterpriseAnonymous => vec![SuppressIdentity, DropRaw, AggregateOnly],
        PrivacyMode::CareWithConsent => vec![DropRaw, ReduceResolution],
        PrivacyMode::StrictNoIdentity => {
            vec![SuppressIdentity, ReduceResolution, DropRaw, AggregateOnly]
        }
    };
    v // already authored in ascending Ord order
}
```

The audit log is `Vec`-backed and append-only by API surface (no `pop`, no index-mut). The registry requires `&mut self` only for `record_transition`; `active_mode`, `enforced_actions`, `is_action_enforced`, and `allows_sink` are `&self` reads safe to call from the publish path.

### 2.4 Mode → (Class, Soul Gate, MQTT ACL) Mapping

This is the explicit, single-place declaration the gap (§1.1) was missing. Each row is enforced by `PrivacyMode::privacy_class()`, `PrivacyMode::soul_signature_enabled()`, and the existing class-gated routers.

| Mode | `PrivacyClass` | Soul Signature | Salt domain | MQTT/HA exposure (existing routers) | Enforced actions |
|------|----------------|----------------|-------------|--------------------------------------|------------------|
| `RawResearch` | `Raw` (0) | off | per-node | none — class 0 never networked (`mqtt_topics.rs:111`, I1 `sink.rs:35`) | `Allow` |
| `PrivateHome` | `Anonymous` (2) | off | per-node | presence/motion/count/conf/`identity_risk` (`ha_discovery.rs:116`) | `SuppressIdentity`, `DropRaw` |
| `EnterpriseAnonymous` | `Anonymous` (2) | off | **multiseed** (`signature_hasher.rs` per-seed `site_salt`) | same as PrivateHome | `SuppressIdentity`, `DropRaw`, `AggregateOnly` |
| `CareWithConsent` | `Derived` (1) | **on** (`SoulMatchOracle` + `SignatureHasher`) | per-node | LAN/research only — class 1 not on public tree (`mqtt_topics.rs:111`) | `DropRaw`, `ReduceResolution` |
| `StrictNoIdentity` | `Restricted` (3) | off | per-node | presence/motion/count/conf only; `identity_risk` *not* published (`mqtt_topics.rs:147`, `event.rs:113`) | `SuppressIdentity`, `ReduceResolution`, `DropRaw`, `AggregateOnly` |

Two mappings warrant explanation:

- **`PrivateHome` vs `EnterpriseAnonymous` both → `Anonymous`.** The difference is salt isolation, not class. Enterprise enables `multiseed_salt()` so that two seeds observing the same person in adjacent units produce uncorrelated `rf_signature_hash` values, preserving I3 (`signature_hasher.rs:8-18`) across a shared tenant boundary. Single-home does not need this. Both publish `identity_risk` at class 2 per the existing `ha_discovery.rs:116` rule — Enterprise additionally enforces `AggregateOnly` semantically, suppressing any zone-level or per-cluster surface beyond the five aggregate entities.
- **`CareWithConsent` → `Derived` with Soul on.** This is the only mode that resolves to class `Derived`, matching `lib.rs:88-90`'s comment "Required for Soul Signature deployments." It enables `soul-signature` (the Cargo feature, `Cargo.toml:24-27`) and installs a real `SoulMatchOracle` so the gate's `Recalibrate` exemption (`coherence_gate.rs:71-84`) fires for enrolled subjects. Class `Derived` is *not* on the public MQTT tree (`mqtt_topics.rs:111` requires `>= Anonymous`), so consented identity data stays on LAN/research surfaces — `DropRaw` and `ReduceResolution` still apply.

### 2.5 `PrivacyAttestationProof`: Hash-Chained Per ADR-010

The attestation proof gives cryptographic continuity that the declared mode was honored. It reuses the ADR-010 witness-chain primitive directly: each proof entry includes the SHAKE-256/BLAKE3 hash of the previous entry (`ADR-010` §"Hash chain", `previous_hash`/`entry_hash` linkage), so any insertion, deletion, or reordering breaks verification.

```rust
// v2/crates/wifi-densepose-bfld/src/attestation.rs
#![cfg(feature = "std")]

use crate::mode::{PrivacyAction, PrivacyMode};
use blake3::Hasher; // already a dependency (Cargo.toml:33)

/// Per-event privacy enforcement record — the unit folded into the chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivacyActionRecord {
    /// Capture-clock ns of the event this record attests.
    pub timestamp_ns: u64,
    /// Strongest action enforced for this event (ADR-140 `privacy_action`).
    pub strongest_action: PrivacyAction,
    /// Names of payload/event fields stripped for this event, e.g.
    /// "compressed_angle_matrix", "rf_signature_hash". Sorted lexicographically
    /// so the canonical-bytes hash is deterministic.
    pub fields_stripped: Vec<&'static str>,
}

/// One link in the attestation hash chain. ADR-010-compatible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrivacyAttestationProof {
    /// Active mode at the time this link was sealed.
    pub mode: PrivacyMode,
    /// All actions enforced under `mode`, ascending (from the registry).
    pub actions_enforced: Vec<PrivacyAction>,
    /// Per-event strip records covered by this link (a window, see below).
    pub fields_stripped_per_event: Vec<PrivacyActionRecord>,
    /// BLAKE3 hash of the *previous* link's `entry_hash`; all-zero for genesis.
    pub prev_hash: [u8; 32],
    /// BLAKE3 over (mode token || actions || records || prev_hash). Computed by
    /// `seal()`; this is the value the next link references as `prev_hash`.
    pub entry_hash: [u8; 32],
}

impl PrivacyAttestationProof {
    /// Seal a new link given the previous link's `entry_hash` (or `[0u8; 32]`
    /// for the genesis link). The hash binds mode, actions, and per-event
    /// strips, so altering any field after sealing breaks the chain.
    #[must_use]
    pub fn seal(
        mode: PrivacyMode,
        actions_enforced: Vec<PrivacyAction>,
        fields_stripped_per_event: Vec<PrivacyActionRecord>,
        prev_hash: [u8; 32],
    ) -> Self {
        let mut h = Hasher::new();
        h.update(mode.token().as_bytes());
        for a in &actions_enforced {
            h.update(&[*a as u8]);
        }
        for rec in &fields_stripped_per_event {
            h.update(&rec.timestamp_ns.to_le_bytes());
            h.update(&[rec.strongest_action as u8]);
            for f in &rec.fields_stripped {
                h.update(f.as_bytes());
                h.update(&[0u8]); // length-free field separator
            }
        }
        h.update(&prev_hash);
        let entry_hash = *h.finalize().as_bytes();
        Self { mode, actions_enforced, fields_stripped_per_event, prev_hash, entry_hash }
    }

    /// Verify chain linkage against the previous link's `entry_hash` AND that
    /// `entry_hash` recomputes from the sealed fields (tamper evidence).
    #[must_use]
    pub fn verify_link(&self, expected_prev: [u8; 32]) -> bool {
        if self.prev_hash != expected_prev {
            return false;
        }
        let recomputed = Self::seal(
            self.mode,
            self.actions_enforced.clone(),
            self.fields_stripped_per_event.clone(),
            self.prev_hash,
        );
        recomputed.entry_hash == self.entry_hash
    }

    /// Short proof hash for diagnostics: `"blake3:<16 hex>"` (first 8 bytes of
    /// `entry_hash`). Surfaced on the HA diagnostic entity (§2.6).
    #[must_use]
    pub fn short_hash(&self) -> String {
        let mut s = String::with_capacity(7 + 16);
        s.push_str("blake3:");
        for b in &self.entry_hash[..8] {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}
```

**Chain granularity — per window, not per frame.** The proof links one *event window* (e.g., one emit cycle of the `BfldEmitter`, `emitter.rs:138`), not one CSI frame. A per-frame chain at 20 Hz would grow at 1,728,000 links/day; per-window keeps the chain bounded to the published-event rate while still attesting every strip (each window's `fields_stripped_per_event` enumerates the per-event strips inside it). BLAKE3 is reused (it is already a dependency, `Cargo.toml:33`) rather than introducing the SHAKE-256 used in ADR-010's MAT path — ADR-010 §"Hash chain" specifies a hash-linked chain but not a fixed algorithm; BFLD already keys its `rf_signature_hash` with BLAKE3 (`signature_hasher.rs:20`), so reusing it avoids a second crypto dependency in the no-`std`-capable crate.

### 2.6 Integration Into MQTT Discovery + a Read-Only HA Diagnostic Entity

The active mode and proof hash are surfaced as a **read-only diagnostic** so an operator, regulator, or the cognitum-v0 dashboard can see the live privacy posture without touching the sensing entities. This extends `ha_discovery.rs` and `mqtt_topics.rs`, both of which already class-gate every entity.

- A new discovery payload is rendered by `render_discovery_payloads()` (`ha_discovery.rs:61`) for a `sensor` with `entity_category = "diagnostic"`, unique-id `<node>_bfld_privacy_mode`, state topic `ruview/<node>/bfld/privacy_mode/state`. Its state is a compact JSON object `{"mode":"care_with_consent","class":"derived","proof":"blake3:<16hex>","actions":["drop_raw","reduce_resolution"]}`.
- The entity is published at every class `>= Anonymous` (same gate as the existing five diagnostic sensors) **and** additionally at class `Raw`/`Derived` on the LAN-only research surface — because a research/care deployment most needs to display its own attestation. The class gate for the *public* tree (`mqtt_topics.rs:111`) is unchanged; the diagnostic mode entity is added to the local diagnostic surface regardless of class so the proof is always inspectable on-node.
- It is strictly read-only: the entity has no `command_topic`. Mode changes are an operator/config action (TOML + restart, §2.7), never an MQTT write — consistent with the "no `promote`" posture of `privacy_gate.rs`.

The proof hash on this entity is the `short_hash()` of the most recently sealed `PrivacyAttestationProof`. A verifier with the full chain (exported via a future `attestation export` CLI) can confirm continuity from genesis to the displayed hash.

### 2.7 Registry Wiring Into the Emitter

`BfldEmitter` (`emitter.rs:65-88`) gains an owned `PrivacyModeRegistry` and seals one attestation link per emit window. The change is additive — the existing `emit()`/`emit_with_oracle()` signatures are unchanged; the registry is configured via a new builder.

```rust
// emitter.rs additions (sketch)
pub struct BfldEmitter {
    // ...existing fields (node_id, default_zone_id, privacy_class, gate, ring, signature_hasher)
    registry: PrivacyModeRegistry,         // NEW — single source of truth
    last_proof_hash: [u8; 32],             // NEW — chain tail; [0;32] genesis
}

impl BfldEmitter {
    /// Configure the emitter from a named mode. Sets `privacy_class` from
    /// `mode.privacy_class()`, installs/clears the signature hasher and Soul
    /// oracle per `mode.soul_signature_enabled()`, and pins the registry.
    #[must_use]
    pub fn with_mode(mut self, mode: PrivacyMode) -> Self {
        self.privacy_class = mode.privacy_class();
        self.registry = PrivacyModeRegistry::new(mode);
        self
    }

    /// Active mode + freshly sealed proof for the most recent emit window.
    /// Read by the HA diagnostic entity (§2.6).
    #[must_use]
    pub fn attestation(&self) -> Option<&PrivacyAttestationProof> { /* tail of sealed chain */ }
}
```

On each `emit()`, after the gate decision (`emitter.rs:171`), the emitter: (1) calls `registry.record_transition(ts)`; (2) builds a `PrivacyActionRecord` enumerating the fields the privacy gating actually stripped (e.g., at class `Restricted` the `identity_risk_score` + `rf_signature_hash` strip in `event.rs:112-117` yields `fields_stripped = ["identity_risk_score","rf_signature_hash"]`); (3) calls `PrivacyAttestationProof::seal(mode, actions, records, self.last_proof_hash)` and updates `last_proof_hash`. The configured baseline mode (default `PrivateHome`) preserves the current `Anonymous` default (`emitter.rs:82`), so an un-migrated caller sees identical behavior plus a populated attestation chain.

### 2.8 Downstream Consumers (ADR-139, ADR-140)

| Consumer | What it reads | Binding |
|----------|---------------|---------|
| ADR-140 Semantic State Record | `PrivacyActionRecord::strongest_action` | Populates the record's mandatory `privacy_action` field; the proof `entry_hash` populates the record's privacy-provenance reference |
| ADR-139 WorldGraph | `PrivacyModeRegistry::is_action_enforced(AggregateOnly)` / `ReduceResolution` | A zone/edge whose evidence was degraded by `ReduceResolution` or `AggregateOnly` is tagged `privacy_limited_by = <mode token>` so the digital twin can mark the region as privacy-degraded rather than sensor-blind |
| ADR-136 Streaming Engine | `attestation()` short hash | Stage-boundary frame contract may carry the active mode token for downstream stages without re-deriving it |
| `ha_discovery.rs` / `mqtt_topics.rs` | active mode + `short_hash()` | Read-only diagnostic entity (§2.6) |

This honors the project rule that every semantic state traces to **signal evidence + model version + calibration version + privacy decision**: ADR-141 supplies the *privacy decision* half — the `PrivacyActionRecord` (what was enforced) plus the chain `entry_hash` (proof it was enforced) — which ADR-140 records alongside the signal/model/calibration provenance from ADR-134/ADR-135.

---

## 3. Consequences

### 3.1 Positive

- **Single declared policy object.** A deployment's privacy stance is now one named `PrivacyMode` and a `PrivacyModeRegistry`, not four scattered wiring decisions. An installer selects `CareWithConsent`; the registry derives class, Soul gate, salt domain, and ACL deterministically.
- **Cryptographic continuity.** `PrivacyAttestationProof` makes "we ran in StrictNoIdentity and stripped identity on every event" a verifiable claim, not a code-review assertion. The chain reuses the ADR-010 primitive, so the existing witness verifier extends naturally.
- **Regulator/operator visibility.** The read-only HA diagnostic entity exposes the live mode and proof hash without widening the sensing surface — useful for care-home compliance audits.
- **Clean ADR-139/ADR-140 bindings.** `privacy_action` and `privacy_limited_by` now have a concrete, queryable source (`is_action_enforced`, `strongest_action`), closing the trace requirement for semantic state.
- **No wire/byte changes.** `PrivacyClass` byte values, `BfldFrameHeader`, `sink.rs` ACL, and the MQTT topic tree are untouched. Modes are purely additive.

### 3.2 Negative

- **Two same-class modes.** `PrivateHome` and `EnterpriseAnonymous` both resolve to `Anonymous`; the difference (salt domain, `AggregateOnly`) lives above the class layer and is only meaningful if downstream consumers honor the action set. A consumer that looks only at `PrivacyClass` will not distinguish them.
- **Chain growth.** Even per-window, a busy node accumulates attestation links. An export/prune policy (genesis re-anchoring after verified export) is needed and is deferred to a follow-up iter.
- **`emitter.rs` gains state.** The emitter now owns a registry and a chain tail, growing its memory footprint and making `emit()` no longer a pure transform of inputs→event. The seal cost (one BLAKE3 over a small buffer) is sub-microsecond but non-zero.
- **Mode change requires restart.** By design there is no MQTT command topic to change mode at runtime (mirrors `privacy_gate.rs`'s no-`promote` posture). Operators change mode via TOML config + restart, which is a heavier operation than a dashboard toggle.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Mode→action mapping drifts from what `privacy_gate.rs` actually strips, so the proof attests fields that were not really removed | Medium | Attestation lies — worse than no attestation | The `PrivacyActionRecord.fields_stripped` is populated from the *actual* gate output (`event.rs`/`privacy_gate.rs` return values), not from the mode table; a unit test asserts the recorded strips equal the bytes the gate zeroed |
| `EnterpriseAnonymous` multiseed salt not actually isolated (two seeds share a salt) → I3 broken under same class | Low | Cross-unit identity correlation | `multiseed_salt()` gates a per-seed `site_salt` derivation; an acceptance test asserts cross-seed Hamming distance ~128 bits (reusing ADR-120 §2.7 AC2 from `tests/signature_hasher.rs`) |
| Chain genesis confusion: a node that restarts mid-deployment starts a fresh genesis, breaking continuity from the prior chain | Medium | Verifier sees a discontinuity it cannot distinguish from tampering | Genesis links record `prev_hash = [0;32]` and a boot epoch; the verifier treats a genesis link with a logged restart event as a legitimate re-anchor, not a break |
| Operator selects `RawResearch` and assumes raw never networks, but a misconfigured custom `Sink` accepts class 0 | Low | I1 violation | `RawResearch`'s `DropRaw` action is redundant with the compile-time `sink.rs` ACL (`MIN_CLASS`); the registry's `allows_sink::<NetworkKind>()` returns `false` for `Raw`, giving a runtime second line of defense |

---

## 4. Alternatives Considered

### 4.1 Extend `PrivacyClass` to Five+ Variants Instead of Adding `PrivacyMode`

Collapsing modes into the class enum would avoid a second type. Rejected because `PrivacyClass` is a *wire and sink-enforcement* primitive: its byte values are serialized in `BfldFrameHeader`, switched on in `sink.rs::check_class`, the MQTT router, and the NVS/Matter representation. Two modes (`PrivateHome`, `EnterpriseAnonymous`) share the same class but differ only in salt domain — they are *not* separable at the byte layer, so they cannot be class variants without inventing byte semantics that the existing `frame.rs`/`sink.rs` code would have to learn. Modes are strictly higher-level and must not perturb the byte contract.

### 4.2 Per-Frame Attestation Chain

A chain link per CSI frame would attest every single frame. Rejected on growth grounds: 20 Hz × 86,400 s = 1.7 M links/day/node, unbounded. The per-window granularity (§2.5) attests every *strip* (each window enumerates its per-event records) at the published-event rate, which is orders of magnitude lower while losing no strip evidence.

### 4.3 Reuse `GateAction` Instead of a New `PrivacyAction` Enum

`GateAction { Accept, PredictOnly, Reject, Recalibrate }` already exists (`identity_risk.rs:57`). Rejected because it answers a different question — *per-event risk gating* — and overloading it would conflate "this event is risky" with "this mode strips identity on every event." They compose (a mode can `SuppressIdentity` while the gate independently `Reject`s); merging them would lose that orthogonality and break ADR-140's need for a stable `privacy_action` value independent of per-event risk.

### 4.4 Runtime Mode Changes via MQTT Command Topic

A `command_topic` would let a dashboard flip modes live. Rejected for the same reason `privacy_gate.rs` has no `promote`: a remote, unauthenticated-by-default MQTT write that *weakens* privacy (e.g., `StrictNoIdentity` → `RawResearch`) is a privilege-escalation surface. Mode is a config-time + restart decision; the diagnostic entity is read-only.

### 4.5 SHAKE-256 (Match ADR-010 Exactly) vs BLAKE3 Reuse

ADR-010's MAT path uses SHAKE-256. Adopting it here would mean a second crypto dependency in a crate that is `#![cfg_attr(not(feature = "std"), no_std)]` (`lib.rs:14`). Rejected: ADR-010 §"Hash chain" specifies a hash-*linked* chain, not a fixed algorithm, and BFLD already depends on BLAKE3 for `rf_signature_hash` (`signature_hasher.rs:20`, `Cargo.toml:33`). Reusing BLAKE3 keeps the no-std footprint minimal while satisfying the linkage/tamper-evidence contract.

---

## 5. Testing and Acceptance Criteria

### 5.1 Test Plan

**T1 — Mode→class/Soul/salt mapping (unit).** For each of the five `PrivacyMode` variants, assert `privacy_class()`, `soul_signature_enabled()`, and `multiseed_salt()` exactly match the §2.4 table. Assert `token()` round-trips through a `from_token()` parser.

**T2 — Canonical action set (unit).** For each mode, assert `actions_for(mode)` equals the §2.4 "Enforced actions" column, is sorted ascending (`Ord`), and is deduplicated. Assert `is_action_enforced` agrees with set membership for all 25 (mode, action) pairs.

**T3 — ACL agreement with `sink.rs` (unit).** For each mode, assert `registry.allows_sink::<LocalKind>()`, `::<NetworkKind>()`, `::<MatterKind>()` equal `check_class::<S>(mode.privacy_class()).is_ok()` — i.e., the registry ACL never disagrees with the compile-time sink ACL. In particular `RawResearch.allows_sink::<NetworkKind>() == false` (I1).

**T4 — Attestation chain linkage (unit).** Seal a genesis link (`prev_hash = [0;32]`), then three more, threading each `entry_hash` into the next `prev_hash`. Assert `verify_link()` passes for all four against the correct predecessors. Mutate one link's `mode` and assert `verify_link()` fails (tamper evidence). Insert/delete/reorder a link and assert verification breaks.

**T5 — Recorded strips equal actual gate output (unit).** Run `BfldEmitter::with_mode(StrictNoIdentity)`, emit an event that would carry `identity_risk_score` + `rf_signature_hash`, and assert: (a) the emitted `BfldEvent` has both fields `None` (existing `event.rs:113` behavior), AND (b) the sealed `PrivacyActionRecord.fields_stripped` equals `["identity_risk_score","rf_signature_hash"]` (sorted) — proving the proof attests what was really stripped, not what the table claims.

**T6 — Multiseed salt isolation (unit, reuses ADR-120 AC2).** Two emitters in `EnterpriseAnonymous` with distinct per-seed salts observing identical identity features produce `rf_signature_hash` values with Hamming distance in [112, 144] bits (≈128 expected). Same test in `PrivateHome` with a shared node salt is *not* required to isolate (documents the difference).

**T7 — Default-mode backward compatibility (unit).** A `BfldEmitter::new(node_id)` with no `with_mode()` call behaves identically to today (class `Anonymous`, `emitter.rs:82`) and its registry reports `active_mode() == PrivateHome`.

**T8 — HA diagnostic entity render (unit).** `render_discovery_payloads()` emits the `privacy_mode` diagnostic sensor with `entity_category = "diagnostic"`, no `command_topic`, and a state JSON containing the mode token, class, `short_hash()`, and action tokens. Assert the public sensing tree (presence/motion/etc.) is byte-identical to the pre-change output (no regression to `mqtt_topics.rs:109`).

**T9 — Determinism proof (CI, extends ADR-028).** Seal a fixed 4-link chain from a hard-coded mode sequence and assert the final `entry_hash` matches a recorded SHA-256-of-bytes constant in `archive/v1/data/proof/expected_features.sha256` under key `bfld_attestation_chain_v1`. Makes the attestation hash deterministic end-to-end.

### 5.2 Acceptance Criteria

- **AC1**: All five modes resolve to the exact (class, Soul, salt, ACL, actions) tuple in §2.4 — T1, T2, T3 green.
- **AC2**: The attestation chain is tamper-evident: any single-field mutation, insertion, deletion, or reorder fails `verify_link()` — T4 green.
- **AC3**: For every emitted event, `PrivacyActionRecord.fields_stripped` equals the set of fields the gate actually zeroed (no attestation lies) — T5 green.
- **AC4**: `EnterpriseAnonymous` preserves I3 across seeds (cross-seed Hamming ≈ 128 bits) — T6 green.
- **AC5**: An un-migrated `BfldEmitter::new()` is observationally identical to today, plus a populated attestation chain — T7 green; the public MQTT tree is byte-identical — T8 green.
- **AC6**: `is_action_enforced` and `strongest_action` are callable by ADR-139/ADR-140 with no `&mut` access to the registry (read path is `&self`).

### 5.3 Witness / Proof

Per ADR-028/ADR-010, three rows are added to the witness log:

| Row | Capability | Evidence |
|-----|-----------|----------|
| W-39 | Mode→action mapping is total and matches §2.4 | `cargo test -p wifi-densepose-bfld mode::tests::mapping_table` |
| W-40 | Attestation chain tamper-evidence | `cargo test -p wifi-densepose-bfld attestation::tests::tamper_breaks_chain` |
| W-41 | Recorded strips equal actual gate output | `cargo test -p wifi-densepose-bfld attestation::tests::strips_match_gate` |

`source-hashes.txt` in the witness bundle gains `SHA-256(mode.rs)` and `SHA-256(attestation.rs)`.

---

## 6. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-010 (Witness Chains) | **Reuses**: `PrivacyAttestationProof` adopts the hash-linked chain primitive (`previous_hash`/`entry_hash`); BFLD uses BLAKE3 rather than SHAKE-256 per §4.5 |
| ADR-118 (BFLD) | **Extended**: modes/actions/attestation layer over the existing pipeline; invariants I1/I2/I3 (`lib.rs:8-11`) unchanged |
| ADR-120 (Privacy Class + Hash Rotation) | **Extended**: `PrivacyMode` maps to `PrivacyClass`; `EnterpriseAnonymous` formalizes multiseed `site_salt` isolation (`signature_hasher.rs`) |
| ADR-121 (Identity-Risk Scoring) | **Composes with**: `PrivacyAction` is orthogonal to `GateAction` (`identity_risk.rs:57`); Soul gate exemption (`coherence_gate.rs:71`) is enabled by `CareWithConsent` |
| ADR-122 (HA/Matter Exposure) | **Extended**: read-only `privacy_mode` diagnostic entity added to `ha_discovery.rs`/`mqtt_topics.rs`; public tree unchanged |
| ADR-136 (Streaming Engine) | **Consumer**: active mode token may ride stage-boundary frame contracts |
| ADR-139 (WorldGraph) | **Consumer**: `is_action_enforced(ReduceResolution/AggregateOnly)` drives `privacy_limited_by` zone/edge tagging |
| ADR-140 (Semantic State Record) | **Consumer**: `strongest_action` populates `privacy_action`; chain `entry_hash` is the privacy-provenance reference |
| ADR-143 (RF SLAM v2) | **Constrains**: reflector/anchor surfaces are subject to `ReduceResolution`/`AggregateOnly` under the active mode |

---

## 7. References

### Production Code

- `v2/crates/wifi-densepose-bfld/src/lib.rs` — `PrivacyClass` (`:82-117`), `BfldError`, structural invariants I1/I2/I3 (`:8-11`)
- `v2/crates/wifi-densepose-bfld/src/sink.rs` — `Sink::MIN_CLASS`, `check_class` (`:47-55`), `LocalKind`/`NetworkKind`/`MatterKind`
- `v2/crates/wifi-densepose-bfld/src/privacy_gate.rs` — `PrivacyGate::demote` zeroizing strip (`:31-75`)
- `v2/crates/wifi-densepose-bfld/src/identity_risk.rs` — `GateAction` (`:57-69`), risk-score bands
- `v2/crates/wifi-densepose-bfld/src/emitter.rs` — `BfldEmitter` default class `Anonymous` (`:82`), gate consult (`:171`)
- `v2/crates/wifi-densepose-bfld/src/event.rs` — `BfldEvent` field exposure table, `apply_privacy_gating` (`:112-117`)
- `v2/crates/wifi-densepose-bfld/src/coherence_gate.rs` — `SoulMatchOracle`, `evaluate_with_oracle` Recalibrate exemption (`:71-84`)
- `v2/crates/wifi-densepose-bfld/src/signature_hasher.rs` — BLAKE3 keyed `rf_signature_hash`, I3 site isolation (`:8-18`)
- `v2/crates/wifi-densepose-bfld/src/ha_discovery.rs` — class-gated discovery render (`:61-129`)
- `v2/crates/wifi-densepose-bfld/src/mqtt_topics.rs` — class-gated topic router (`:109-157`)
- `v2/crates/wifi-densepose-bfld/Cargo.toml` — BLAKE3 dependency (`:33`), `soul-signature` feature (`:24-27`)

### Related ADR Documents

- `docs/adr/ADR-010-witness-chains-audit-trail-integrity.md` — hash-chain primitive
- `docs/adr/ADR-118-bfld-beamforming-feedback-layer-for-detection.md`
- `docs/adr/ADR-120-bfld-privacy-class-and-hash-rotation.md`
- `docs/adr/ADR-121-bfld-identity-risk-scoring.md`
- `docs/adr/ADR-122-bfld-ruview-ha-matter-exposure.md`


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `7d88eb84c`, issue #845): `PrivacyMode` / `PrivacyAction` / `PrivacyModeRegistry` plus the BLAKE3 hash-chained `PrivacyAttestationProof` (`verify_chain()` detects tamper). no_std-safe (registry is std-gated for the ESP32 path). 6 tests.

**Integration glue -- not yet on the live path:** wiring the registry into `PrivacyGate` class transitions, the MQTT discovery payload, and a read-only Home Assistant diagnostic entity exposing the active mode + proof hash.

**Trust contribution:** the *policy spine* -- privacy posture is a tamper-evident, auditable chain rather than a checkbox; an operator's mode choice actively governs whether identity data may even exist.

---

## Privacy Monotonicity Review (2026-06-14) — confirmed clean

A beyond-SOTA security review of the governed-trust cycle
(`wifi-densepose-engine::StreamingEngine::process_cycle_calibrated`) examined
the privacy-demotion path this ADR governs. **The monotonicity invariant holds:
demotion only ever makes the emitted class more restrictive, never less.**

Verification (no behaviour change, the result is a clean bill with evidence):

- Each cycle computes `effective_class` fresh from the active mode's
  `target_class()` (the floor) and applies at most a **single-step** demotion
  (`demote_one`, clamped at `Restricted`). There is no cross-cycle state that
  could let a permissive class overwrite a restrictive one.
- A forced contradiction (calibration mismatch / array-geometry insufficiency /
  mesh partition risk, ADR-032) raises the class byte; a clean cycle emits
  exactly the base class.
- Pinned by `forced_contradiction_never_relaxes_class`, a property test over
  **all five** `PrivacyMode`s asserting `effective_class.as_u8() >=
  base_class.as_u8()` (strictly greater unless already clamped at `Restricted`)
  under a forced contradiction, and `== base` on a clean cycle.

Fail-closed boundaries were also pinned: an empty cycle errors (no degenerate
over-permissive output, `empty_cycle_fails_closed`) and the single-node boundary
is characterized as a valid non-demoting mode (`single_node_cycle_is_well_formed`).

The related witness domain-separation fix from the same review is recorded in
ADR-137 (the witness folds `effective_class`, so the demotion is auditable).
## Security & Privacy Review (2026-06-14)

Beyond-SOTA privacy+security review of `wifi-densepose-bfld` (the crate was not in the ADR-154–159 sweep). Two real bugs fixed (each pinned by a fails-on-old test), several dimensions confirmed clean.

### Findings

| # | Severity | Site | Issue | Fix | Pinned by |
|---|----------|------|-------|-----|-----------|
| 1 | **privacy-bypass (HIGH)** | `pipeline.rs::process_to_frame` | The documented wire-bytes production path stamped the frame header with the active `PrivacyClass` but serialized the caller's `BfldPayload` **unchanged** via `BfldFrame::from_payload` — never routing through `PrivacyGate::demote`. A frame labeled `Anonymous`(2)/`Restricted`(3) carried the full `compressed_angle_matrix` (identity surface) + amplitude/phase + `csi_delta`. A `NetworkSink` accepts class ≥ `Derived`(1), so the identity surface could cross the node boundary despite the restrictive class byte — the byte lied about content. | Apply `PrivacyGate::demote(frame, active_class)` after construction: a same-class transition that strips the sections the class forbids; `Raw`/`Derived` keep the full payload. | `tests/pipeline_to_frame.rs::process_to_frame_at_anonymous_strips_identity_leaky_sections`, `…_in_privacy_mode_strips_amplitude_and_phase` (both FAILED pre-fix); `…_at_derived_preserves_full_payload` (over-strip guard) |
| 2 | **PII/injection (MEDIUM)** | `mqtt_topics.rs::render_events` | `zone_activity` payload built as `format!("\"{zone}\"")` with no JSON escaping (while `ha_discovery.rs` already escapes). A zone name with `"`/`\` produced malformed/injectable JSON on the HA state topic. | `json_string_literal()` escaper mirroring `ha_discovery::push_str_field`. Value-identical for normal zone names. | `tests/mqtt_topic_routing.rs::zone_payload_escapes_json_metacharacters` (FAILED pre-fix) |

### Dimensions confirmed clean (with evidence)

- **Event-field privacy gating** — `BfldEvent::apply_privacy_gating` nulls `identity_risk_score` + `rf_signature_hash` at `Restricted`, and `serde(skip_serializing_if = "Option::is_none")` omits them entirely. `render_events`/`render_discovery_payloads` refuse class < `Anonymous` (stricter than the `sink.rs` `NetworkKind` `MIN_CLASS = Derived` — defense in depth toward less leakage). Covered by `event_privacy_gating.rs`, `mqtt_topic_routing.rs`, `ha_discovery.rs`.
- **Witness/hash framing (the engine `witness_of` bug class)** — CLEAN. `SignatureHasher::compute` prefixes a **fixed 4-byte** `day_epoch` then a **fixed-width canonical-f32** feature block (`IdentityFeatures`: Embedding = `EMBEDDING_DIM*4`, RiskFactors = 16 B). `PrivacyAttestationProof::compute` hashes a fixed 32-byte `prev_hash` + three fixed 1-byte values. No variable-length operator-influenceable string is concatenated into any digest — no length-prefix-framing collision is possible.
- **Fail-closed** — `payload.rs::from_bytes` rejects truncated/overflowing/trailing-byte sections (`checked_add`, bounds checks); `frame.rs::from_bytes` validates magic/version/length/CRC; `PrivacyClass::try_from` rejects unknown bytes; `identity_risk::score` maps NaN/degenerate factors → 0.0 (privacy-conservative). The `from_score(NaN) → Accept` choice is a documented, deliberate publish-aggregate-only fallback (NaN never reaches it from `score()`); risk-driven NaN cannot leak identity because identity gating is class-byte-driven, not risk-driven.

### Observation (not a bug)

The ADR-141 control plane (`PrivacyMode`/`PrivacyModeRegistry`) is **not yet wired into the emit path** — the emitter/pipeline enforce the raw `PrivacyClass` directly; the registry is exported + unit-tested but advisory. This matches the "Integration glue — not yet on the live path" status above. The class-byte enforcement (emitter + event + renderers + the now-fixed `process_to_frame`) is the live guarantee. Wiring the registry is the documented next step.
