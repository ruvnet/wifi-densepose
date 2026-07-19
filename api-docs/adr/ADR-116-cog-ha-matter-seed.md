# ADR-116: Home Assistant + Matter as a Cognitum Seed cog (`cog-ha-matter`)

| Field | Value |
|-------|-------|
| **Status** | Proposed — P1 research complete ([`docs/research/ADR-116-ha-matter-cog-research.md`](../research/ADR-116-ha-matter-cog-research.md)). P2 cog scaffold compiles (`v2/crates/cog-ha-matter`, 2/2 unit tests green). |
| **Date** | 2026-05-23 |
| **Deciders** | ruv |
| **Codename** | **HA-COG** — HA + Matter, packaged for the Seed |
| **Relates to** | [ADR-110](ADR-110-esp32-c6-firmware-extension.md) (C6 firmware substrate), [ADR-115](ADR-115-home-assistant-integration.md) (HA-DISCO + HA-MIND + HA-FABRIC), [ADR-102](ADR-102-edge-module-registry.md) (cog catalog), [ADR-101](ADR-101-pose-estimation-cog.md) (cog packaging precedent) |
| **Tracking issue** | TBD — file under RuView issue tracker once research dossier lands |

---

## 1. Context

ADR-115 shipped the Home Assistant + Matter integration as a **`--mqtt` flag on `wifi-densepose-sensing-server`** — a Rust binary that runs on a Pi / Linux box, consumes UDP frames from the ESP32 fleet, and publishes MQTT for any Home Assistant install to discover. That works, but it makes HA+Matter a *configuration of the aggregator*, not an *installable artifact* a Cognitum Seed user can drop into their existing fleet.

The Cognitum Seed already has a [105-cog catalog](https://seed.cognitum.one/store) — packaged Seed apps (`cog-pose-estimation`, `cog-quantum-vitals`, `cog-person-matching`, etc.) that anyone can install from `app-registry.json`. **There is no `cog-ha-matter` yet.** That's the gap this ADR closes.

The cog packaging precedent is ADR-101 (`cog-pose-estimation`) which ships signed aarch64 + x86_64 binaries on GCS with a `pose_v1.safetensors` weight blob — same shape we'd want for the HA cog.

### 1.1 Why a cog, not just the existing flag?

| Path | Distribution | Discovery | Update | Witness | Local AI |
|---|---|---|---|---|---|
| `--mqtt` on `sensing-server` | manual install of the Rust binary | none | manual | none | external |
| **`cog-ha-matter` Seed cog** | `app-registry.json` listing, one-click install | mDNS / cog browser | OTA via cog runtime | Ed25519 witness chain | local ruvllm + RuVector |

The cog ships HA+Matter as a first-class Seed feature — same UX as installing a pose estimator or person matcher.

### 1.2 What this ADR is *not*

- Not a deprecation of the `--mqtt` flag on sensing-server. The flag stays for Pi / Linux deployments without a Seed; the cog is the Seed-native option.
- Not a port of HA-MIND / HA-DISCO logic to a different language. The Rust crate already exists; the cog *wraps* it as a Seed-installable artifact + adds Seed-specific surfaces (witness, RuVector, ruvllm-driven thresholds).
- Not a Matter SDK ship. ADR-115 §9.10 deferred the matter-rs SDK wiring to v0.7.1; this ADR continues that deferral and focuses on the *cog packaging* + *first-class Seed integration*, with Matter Bridge mode shipping in v0.8 once the SDK is ready.

## 2. Decision (provisional — to be refined by the research dossier)

Build **`cog-ha-matter`** as a Cognitum Seed cog with these surfaces:

### 2.1 Core entity surface (unchanged from ADR-115)

The cog republishes the same 21 entities per node (11 raw + 10 semantic primitives) over MQTT auto-discovery, so HA installations behave identically whether the source is a Seed cog or an external sensing-server.

### 2.2 Seed-native enhancements

- **Self-contained MQTT broker (optional)** — if the user doesn't already run mosquitto, the cog can host an embedded broker on `cognitum-seed.local:1883` and act as the HA endpoint directly.
- **mDNS service advertisement** — `_ruview-ha._tcp` so HA's discovery integration finds the Seed without manual config.
- **RuVector-backed semantic-primitive thresholds** — instead of static `semantic-thresholds.yaml`, the cog learns per-home thresholds via a SONA-adapted RuVector model (matches the Seed's local-first AI story).
- **Ed25519 witness chain** — every state transition logged with a Seed signature so care-home / regulated deployments can audit decisions.
- **OTA firmware coordination** — the cog manages C6 firmware updates for ESP32-C6 nodes in the mesh (ADR-110 substrate).

### 2.3 Matter dimensions (depend on research findings)

The research dossier covers (a) Matter Bridge vs Matter Device mode, (b) Thread Border Router on the Seed's ESP32-S3 (if feasible), (c) CSA certification path, (d) which Matter device classes map cleanly to which entities. **Decision deferred** until the dossier lands; this ADR will be updated in §3 with the specific Matter feature set.

### 2.4 Multi-Seed federation

Multiple Seeds in adjacent rooms coordinate via:
- ESP-NOW mesh (ADR-110 substrate) for time alignment
- mDNS for service discovery
- Witness chain replication for cross-Seed event provenance

The federation model is the natural extension of ADR-110's mesh substrate into the application layer. Specifically: ADR-110 gives us ≤100 µs cross-board sync; this ADR uses that to deduplicate cross-Seed events (one fall, one alert) and reconstruct multi-room transitions (one occupant, room A → hallway → room B).

## 3. Research dossier findings (P1 complete)

Full dossier: [`docs/research/ADR-116-ha-matter-cog-research.md`](../research/ADR-116-ha-matter-cog-research.md). The eight research questions are now answered:

1. **Matter Bridge vs Matter Root** — Matter 1.4 introduced `OccupancySensor (0x0107)` with `RFSensing` feature flag on cluster `0x0406` (revision 5 in Matter 1.4). That's the correct device class for WiFi-CSI sensing — no health/vitals cluster exists in Matter 1.4.2 and won't soon. **Seed acts as Bridge** with N dynamic OccupancySensor endpoints, **not Commissioner** (the C6 sensing nodes stay Accessories only — 320 KB SRAM no PSRAM rules out commissioning).
2. **Thread Border Router** — ESP32-C6 single-chip TBR confirmed working; `CONFIG_OPENTHREAD_BORDER_ROUTER=y` is the only config step. ADR-110's `c6_timesync.c` already initialises 802.15.4 — TBR is a Kconfig flag away. Real value: HA's Improv-style commissioning works without a separate Thread border router box.
3. **HACS value-add** — config flow (UI setup wizard), Repairs API (structured error cards), re-authentication, diagnostics download, typed service actions (`set_privacy_mode`, `calibrate_zone`), i18n translations. **Bronze is the minimum bar; Gold (repairs + diagnostics + reconfiguration) is the target.** Start from `hacs.integration_blueprint` template.
4. **CSA certification** — ~$30-42k first year ($22.5k membership + $10-19k ATL lab fees). **Skippable for v1** by publishing as "Works with HA" instead. CSA re-evaluate at v0.9+ after HACS adoption data lands.
5. **Cog RAM budget** — 128 MB RAM / 15 % CPU on the Seed appliance (Pi 5 + Hailo-10 variant has more headroom). 10 KB INT8 semantic-primitive classifier fits without PSRAM. Long-lived supervised process with capability scopes `network.mqtt + network.matter + api.ruview_vitals`.
6. **ruvllm + RuVector latency** — `ruvllm-esp32` v0.3.3 confirms SONA self-optimising adaptation under 100 µs per query. 8→10 INT8 classifier ~10 KB quantised. Per-home threshold tuning via HA thumbs-up/thumbs-down feedback as LoRA-style gradient steps — closes the top user complaint (false positives) without cloud round-trips.
7. **HIPAA / FDA** — FDA January 2026 General Wellness guidance explicitly classifies HR / sleep / activity-anomaly alerts as **wellness devices** (outside FDA jurisdiction) when marketed without diagnostic claims. Frame fall detection as **"activity anomaly notification"** not "fall diagnosis". `--privacy-mode` audit-only tier (no MQTT state messages, only SHA-256 digests on-Seed) creates a technical PHI barrier. `OccupancySensor (0x0107)` device class keeps the product in the same regulatory category as a smart motion sensor.
8. **Competitor moat** — Aqara FP300 (Nov 2025): 5 entities, no person count, no vitals, no fall detection. TOMMY: zones only, no vitals, closed-source, paywalled. ESPectre: motion only. **RuView's differentiation** — HR/BR + 17-keypoint pose + 10 semantic primitives + witness chain + SONA adaptation — has no competitor equivalent.

## 4. Recommended v1 scope (from dossier §8)

Ranked by build cost × user impact:

| # | Feature | Cost | Impact | Phase |
|---|---|---|---|---|
| 1 | **`--privacy-mode` audit-only tier** (no MQTT state, SHA-256 digests on-Seed) | ~1 week | Closes care / GDPR deployments | P3 (this cog) |
| 2 | **Seed cog manifest + Ed25519 signing + store listing** | ~1-2 weeks | Enables one-click distribution | P2 + P8 (this cog) |
| 3 | **Local SONA fine-tuning loop** (HA feedback → LoRA gradient steps) | ~2-3 weeks | Reduces false positives, closes #1 user complaint | P5 (this cog) |
| 4 | **HACS gold-tier integration** (config flow + repairs + diagnostics) | ~4-6 weeks | Removes MQTT prerequisite for mainstream users | P9 (separate repo `hass-wifi-densepose`) |
| 5 | **Matter Bridge with OccupancySensor + dynamic endpoints** | ~6-8 weeks | Apple Home / Google Home / Alexa native | **v0.8** dedicated sprint (after HACS adoption data) |
| 6 | **Embedded MQTT broker (rumqttd) inside the cog** | ~1 week | "Works without external broker" but every HA install already has mosquitto / built-in | **v0.7** deferred — adds ~2 MB binary + ACL config surface for marginal user benefit. Dossier ranking did not include this in the prioritised v1 scope. |

## 4. Implementation phases

| Phase | Scope | Status |
|---|---|---|
| **P1** | Research dossier ([`docs/research/ADR-116-ha-matter-cog-research.md`](../research/ADR-116-ha-matter-cog-research.md)) | ✅ **done** — 8 sections, 30+ citations, v1 scope ranked |
| **P2** | Cog crate scaffold (`v2/crates/cog-ha-matter/`) — Cargo.toml + `src/{lib,main,manifest}.rs`, workspace member, CLI args, `--print-manifest` flag, 2 manifest unit tests | ✅ **done** — `cargo check` + `cargo test` green |
| **P3** | Wrap existing ADR-115 MQTT publisher as cog entry point | ✅ **wiring done** — `main.rs` boots ADR-115's `publisher::spawn` via `runtime::spawn_publisher` thin wrapper, holds a long-lived `broadcast::Sender<VitalsSnapshot>`, awaits Ctrl-C. Live-handle test green without a broker. Next (P3.5): subscribe to sensing-server `/v1/snapshot` WS and republish into the channel. |
| **P4** | Seed-native enhancements (mDNS, witness; embedded broker deferred) | ✅ **shipped** — mDNS half: record-builder + ServiceInfo conversion + live responder wired into `main.rs` (HA auto-discovery on `_ruview-ha._tcp` works out of the box, `--no-mdns` flag for restrictive networks). Witness half: hash-chain + JSONL + file persistence + chain-level verify + Ed25519 signing. **Embedded rumqttd broker deferred to v0.7** per dossier §8 ranking — not in the prioritised v1 scope; v1 ships with external-broker only (mosquitto or HA's built-in broker). See §4 v1 scope table. |
| **P5** | RuVector-backed threshold learning (SONA adaptation) | pending |
| **P6** | Multi-Seed federation (cross-Seed dedup + witness) | pending |
| **P7** | Matter Bridge mode (depends on matter-rs / esp-matter readiness) | pending |
| **P8** | Cog signing + `app-registry.json` listing + Seed Store entry | pending |
| **P9** | HACS integration repo (`hass-wifi-densepose`) for HA-side install path | pending |
| **P10** | Witness bundle + CSA-style spec compliance check | pending |

## 4.1 Crypto/security review notes (§2.2 witness chain — ADR-262 P2 prerequisite)

Beyond-SOTA crypto+security review of the SHA-256 + Ed25519 witness chain
(`witness.rs` / `witness_signing.rs`) and the manifest signature surface
(`manifest.rs`), because ADR-262 P2 proposes to **reuse this exact signing
chain**. Top priority was the sibling `wifi-densepose-engine` bug class —
unframed boundary-to-boundary concatenation of operator-influenceable strings
into a signed/hashed digest.

- **Engine bug class ABSENT (good result, reported with byte evidence).**
  `canonical_bytes` is `DOMAIN_TAG ‖ prev_hash[32] ‖ seq:u64-be ‖ ts:u64-be ‖
  kind_len:u32-be ‖ kind ‖ payload_len:u32-be ‖ payload`. The two
  variable-length operator-influenceable fields (`kind`, `payload`) are
  **length-prefixed**; the fixed-width fields are self-delimiting → the
  encoding is injective (no two distinct event tuples share a preimage). The
  Ed25519 signature signs the **identical** bytes the SHA-256 chain commits to.
  No separate unframed concatenation exists; the manifest `binary_signature`
  is signed at build time (Makefile) over a single fixed-length `binary_sha256`
  hex value, not in-crate.

- **CHM-WIT-01 (FIXED) — domain-separation tag added.** The engine fix
  prescribed *domain-tag + length-prefix*; length-prefix was present, the
  domain tag was not. Added a versioned, NUL-terminated
  `WITNESS_DOMAIN_TAG = b"cog-ha-matter/witness-event/v1\x00"` prefix so the
  witness message can never be replayed as a message for another Ed25519
  context that shares key infrastructure (notably the manifest signature).
  **Witness bytes change by design** (prior on-disk hashes/signatures
  invalidated, as with the engine fix); verified safe because no in-repo crate
  consumes cog-ha-matter witness bytes programmatically (doc-mentions only).

- **CHM-WIT-02 (HARDENED) — `verify_signature` now uses `verify_strict`.** For
  an audit chain the signature is the attestation, so non-canonical encodings
  and small-order keys are rejected (RFC 8032 strict), giving the "one
  canonical signature per event" property. Not a forgery fix — the verifying
  key is caller-pinned, never read from the event.

- **Confirmed clean (with evidence):** verify-before-trust + key-pinning
  (`verify_signature` takes the verifying key as a parameter; `read_jsonl`
  re-derives every hash and chain-verifies); key handling (the crate never
  generates/stores/logs/serializes a signing key — only a documented test-only
  fixed seed; production keys come from the Seed secure store, out of scope);
  determinism (positional bytes, deterministic Ed25519, alphabetically-locked
  JSONL field order, sorted TXT records — no HashMap/float nondeterminism feeds
  any digest); fail-closed parsing (structured errors, no panics; `main.rs`
  reads no untrusted files/paths).

Tests: `cog-ha-matter --no-default-features` 64 → **68**, 0 failed (CHM-WIT-01
pinned by 4 fails-on-old tests across `witness.rs`/`witness_signing.rs`;
CHM-WIT-02 guarded by a key-pinning test). Python deterministic proof
unchanged (cog-ha-matter is off the signal proof path).

## 5. References

- ADR-101 — `cog-pose-estimation` packaging precedent (signed binaries on GCS, .cog manifest)
- ADR-102 — edge module registry (`app-registry.json` surfaces all cogs)
- ADR-110 — ESP32-C6 firmware substrate (mesh time alignment that multi-Seed federation depends on)
- ADR-115 — HA-DISCO + HA-MIND + HA-FABRIC (the Rust crate this cog wraps)
- `docs/research/ADR-116-ha-matter-cog-research.md` — companion research dossier (deep-researcher agent in progress)
- Cognitum Seed store: https://seed.cognitum.one/store
- Matter spec: https://csa-iot.org/all-solutions/matter/
- HACS integration target: https://github.com/ruvnet/hass-wifi-densepose (planned)
