# ADR-277: Edge sensing control plane — purposes, zones, retention, and a trust boundary raw RF cannot cross

| Field | Value |
|-------|-------|
| **Status** | Accepted — **P1 implemented** (`ruview-unified/src/policy.rs`; 5 unit tests + the acceptance-pipeline export test) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 |
| **Relates to** | ADR-153 (802.11bf protocol model), ADR-141/120 (BFLD privacy control plane + privacy classes), ADR-262 §3.3 (RuField P0–P5 fail-closed mapping — the same philosophy, applied to sensing outputs), ADR-032 (mesh security hardening) |

## 0. PROOF discipline

Grades per ADR-273 §0. Standards status (EXTERNAL, checkable): IEEE 802.11bf-2025 published 2025-09; IEEE 802.11bk addresses ≤ 320 MHz positioning; ETSI published an ISAC architecture 2026-02 (monostatic/bistatic/multistatic/network/device sensing) followed by a security report identifying **19 privacy and security issue classes**; 3GPP Release 20 sensing studies are active. The OpenAirInterface SRS-xApp demo (0.12 m MAE under a **random** split) is EXTERNAL-UNVERIFIED and its split methodology is exactly the leakage ADR-273 §4 rejects — we cite the *implementation path*, not the number.

## 1. Context

Sensing purposes and sensing zones are becoming first-class authorization objects in the standards (802.11bf sensing sessions; ETSI ISAC purposes/exposure). Meanwhile the ETSI security report's issue classes make one thing clear: a sensing stack without a policy plane is a liability. RuView already fails closed at other boundaries (ADR-262 §3.3 maps privacy by information content, never byte value); this ADR gives sensing *outputs* the same discipline, on-device, before any transport.

## 2. Decision — three structural rules

### 2.1 Raw RF never leaves the trust boundary

The only exportable type is `BoundedEvent` — typed verdicts only (`Presence(bool)`, `ActivityClass(u8)`, `RespirationBpm(f64)`, `Location([f64;3])`, `AnomalyScore(f64)`). **No variant can carry RF samples, so raw CSI/radar export is unrepresentable, not merely forbidden**; `TrustBoundary::export` is the single egress and there is deliberately no API that serializes an `RfTensor` outward. External systems receive bounded events + uncertainty, never signal history.

### 2.2 Fail closed, everywhere

`PolicyEngine::authorize`: unknown zone ⇒ deny; purpose not granted in the zone ⇒ deny; **identity recognition is double-gated** — it must be in the zone's `allowed_purposes` *and* the zone must set `identity_explicitly_enabled` (either alone denies). Retention: an event older than the zone's `retention_s` at export time is dropped with a typed `PolicyDenied`. Tests cover every branch, including the manufactured cases (identity granted-but-not-enabled; enabled-but-not-granted; stale event).

### 2.3 Every output is accountable (ADR-273 acceptance item 8)

`BoundedEvent::new` is the only constructor and *fails* without: uncertainty ∈ [0,1], provenance (device + `synthetic` flag — the ADR-276 honest label survives export), a non-zero model version, timestamp, purpose, and zone. The acceptance test (`outputs_leave_only_through_the_policy_boundary_fully_attributed`) runs the full pipeline — synthetic world → encoder → presence head → event → export — and asserts the attribution and the denial of an ungranted purpose on the same zone.

## 3. Purpose taxonomy

`SensingPurpose`: `Presence, Activity, Vitals, Localization, PoseTracking, IdentityRecognition, ChannelDiagnostics` — deliberately aligned with the ETSI ISAC sensing-service classes and WLAN-sensing use cases so a future 802.11bf sensing-session negotiation or ISAC exposure API maps 1:1 onto zone grants. Person *identity* is additionally kept out of the ADR-275 scene graph by type (`EntityKind::PersonClass`, never a person id) — the graph cannot leak what it cannot store.

## 4. O-RAN / cellular path (roadmap, seams shipped)

The P1 control plane is transport-agnostic and already fronts the cellular seam:

- `CellularSrsAdapter` (`oai-srs-xapp`, ADR-274 §2.3) normalizes comb-sampled SRS frequency responses into the canonical tensor — the data-plane contract an OAI xApp needs.
- P4 (ADR-273 §7) places the sensing application beside the DU for sub-ms I/Q–CSI–SRS access, with the xApp performing wider-area fusion; **every output of that path still exits through this ADR's `TrustBoundary`**, and its localization claims will be reported only under strict splits (the OAI demo's random split is the cautionary example, not the target).

## 5. Alternatives considered

- **Reuse BFLD's privacy classes directly** — rejected: BFLD (ADR-120) classifies *captures*; this plane authorizes *outputs by purpose and zone*. They compose (a BFLD-classified capture feeding a head still exits through `TrustBoundary`), and ADR-262's `map_privacy` remains the capture-side mapping.
- **Config-file allow-lists without types** — rejected: the 19 ETSI issue classes are mostly "the code path existed" failures; unrepresentability beats configuration.

## 5.5 Boundary hardening (property-tested)

`tests/security_boundaries.rs` drives every validated constructor and every authorization gate with `proptest` over arbitrary values — including NaN/±inf smuggled via `f64::from_bits` — and asserts the *contract* (valid object **or** typed error, never a panic, never a permissive default). Three real defects surfaced and were fixed, all input-controlled denial-of-service or NaN-propagation:

1. `ble_cs_range` unwrap looped forever on a **non-finite** phase (`+inf − x = +inf`); a **finite-but-huge** phase (1e300 rad) made the same loop run ~1e299 iterations. Fixed by rejecting implausible phases (> 1e6 rad) and replacing the loop-based unwrap with O(1) modular arithmetic.
2. A **subnormal** Gaussian scale (5e-324) passed `> 0` but overflowed `1/σ²` to ∞, making the density at the primitive's own centre NaN. Fixed with physical plausibility bounds (σ ∈ [1e-6, 1e4] m, occupancy ∈ [0, 1e6] nepers/m).

The eight properties now proven: tensor/Gaussian/BoundedEvent constructors never panic; `ble_cs_range` never panics and yields only finite non-negative distances; the policy engine is fail-closed for every (purpose, grants, zone) triple; raw export is unreachable for every task configuration; coherent fusion rejects every non-finite or out-of-bounds sync state; occupancy representations can never retain identity.

## 6. Consequences

- Enterprise/telecom conversations get a concrete artifact: a privacy manifest is a serialization of zones + purposes + retention (all types already `serde`).
- Every future surface (sensing-server WS, RuField bridge, SRS xApp, MCP tools) must route sensing outputs through `TrustBoundary` — added to the pre-merge security-review checklist item 12.
- Cost: purposes are coarse (no per-consumer grants yet); P3 adds consumer identity when the sensing-server wiring lands.
