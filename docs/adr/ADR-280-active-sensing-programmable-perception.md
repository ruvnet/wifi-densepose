# ADR-280: Active sensing and programmable perception — tasks, freshness, coherence, and governed actuation

| Field | Value |
|-------|-------|
| **Status** | Accepted — **implemented** (`ruview-unified/src/control.rs`; 6 test suites incl. a measured ≥70 % traffic-reduction gate) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273; extends ADR-277 |
| **Relates to** | ADR-277 (policy engine — every contract here composes with it), ADR-262 (P0–P5 privacy classes, reused verbatim), ADR-148 (`ruview-swarm` — the mobile-agent consumer of sensing actions) |

## 0. PROOF discipline

Grades per ADR-273 §0. External motivators — ESI-Bench's act-to-uncover formalization, LuLIS's 256-coherent-RF-chain distributed aperture, ETSI's cooperative-ISAC and AI/data-handling work items, age-of-information digital-twin scheduling, semantic/task-sufficient communication architectures — are all EXTERNAL-UNVERIFIED. Everything asserted about *our* behavior is a named test.

## 1. Context

The important shift is from **passive sensing** (accept whatever measurements arrive) to **programmable perception**: the system chooses where, when, how, and at what fidelity to sense, then changes the radio environment or moves sensing agents to resolve uncertainty. Simultaneously, the dominant failure mode across the emerging systems is **hidden synchronization and calibration dependence** — shared clocks, known antenna poses, stable phase silently assumed, confidently wrong when violated. Both belong in the control plane, fail-closed, before capture begins.

## 2. Decision — the evidence-aware sensing task (`SensingTask`)

ETSI-ISAC-vocabulary contract: purpose, target zone, modalities, requested resolution, latency bound, minimum confidence (below which results become *no decision*), raw + result retention, authorized consumers, consent reference. `admit_task` composes with the ADR-277 engine and is fail-closed on every branch; two rules deserve record:

- `raw_export_allowed` **exists in the contract** (ISAC vocabulary compatibility) but is **always refused** (`task_admission_is_fail_closed`): ADR-277 §2.1 made raw export unrepresentable, and a config flag does not reopen it.
- Identity-purpose tasks without a consent reference are refused before the zone check even runs.

## 3. Decision — sensing actions (`SensingAction` + `InformationGoal`)

An action is a deliberate act of evidence-gathering against a stated hypothesis ("the east corridor holds one stationary person or two closely spaced people"), bounded by latency, energy, and a **privacy ceiling** (`PrivacyClass` P0–P5, the ADR-262 ladder). Actions are what the planner (§4), a MetaHarness agent, or a swarm drone consume.

## 4. Decision — age-of-information scheduler (`ActiveSensingPlanner`)

A spatial twin is only useful when it knows which parts are stale. Per region: `SpatialStateFreshness` (last observation, expected change rate, uncertainty growth, business criticality, sensing cost), with

```text
priority = uncertainty(age) × change_rate × criticality ÷ cost
```

The planner emits at most the highest-priority action above threshold per cycle. **Measured** (`planner_reduces_sensing_traffic_versus_uniform_refresh`): 20 regions / 100 ticks, one hot region — 100 observations vs 2,000 under uniform refresh = **95 % sensing-traffic reduction** while the hot region stays observed. (The brief's "50–90 %" was an architectural estimate; this is a synthetic-scenario measurement, sensitive to how concentrated change is.) Priority ordering is proven separately (`planner_prioritizes_stale_critical_regions`: emergency-exit > server-room > storage).

## 5. Decision — coherent distributed apertures fail closed (`CoherentSensorGroup`)

No coherent fusion unless the group can *prove* compatibility: every member must report sync state, be within the group's time-error and phase-error bounds, and match the calibrated baseline geometry hash; unknown reporters are rejected too. Five denial paths, each tested (`coherent_fusion_fails_closed`): missing member, clock drift, phase drift, geometry change since calibration, non-member injection. This is the antidote to the hidden-synchronization failure mode — a building-scale WiFi aperture (the LuLIS direction) degrades to incoherent processing rather than producing confident nonsense.

## 6. Decision — programmable radio environments are governed actuators

RIS / movable / fluid antennas change **which rooms and people are observable**, so actuation is governed like sensing: `request_actuation` is the only way to obtain an `ActuationReceipt`, it verifies the state is supported *and* that the affected zone grants the purpose under the ADR-277 engine (`actuation_requires_policy_authorization`: steering a beam for an ungranted purpose is denied). Receipts carry requested/applied state, time, controller, purpose — the audit trail the RIS governance requirement demands.

## 7. Decision — task-sufficient representations are leakage-checked

Semantic compression ("transmit occupancy uncertainty, not CSI") must remain **task-scoped**: a representation sufficient for anonymous occupancy may not retain identity. `TaskSufficientRepresentation` carries source lineage, an information bound, an explicit `excluded_information` list, and a privacy class; `validate_representation` enforces per-purpose ceilings (Presence/Diagnostics ≤ P2 excluding identity+vitals; Activity/Localization ≤ P3 excluding identity; Vitals/Pose ≤ P4; Identity = P5) and refuses lineage-free orphans (`task_sufficient_representation_is_leakage_checked`).

## 8. Standards alignment (the strongest strategic seam)

The vocabulary here — sensing task/service/entity, measurement configuration, sensing data/result/consumer/purpose, retention, result exposure — is deliberately the emerging ETSI ISAC data-plane vocabulary, positioning this crate as an open reference implementation candidate for ISAC data handling rather than a parallel dialect. Charging/mobility management are explicitly out of scope until a cellular deployment exists.

## 9. Consequences

- MetaHarness/OaK-style agents get a typed surface: read freshness, plan actions, receive receipts — spatial memory meets agentic planning without touching raw RF.
- Distributed-aperture work (P4+) inherits a fusion gate that already fails closed.
- Not implemented (honest scope): information-gain *estimation* is caller-supplied (the planner uses staleness heuristics, not mutual information); RIS drivers, actual multi-AP coherence measurement, and OTFS waveform control are hardware-dependent roadmap items.
