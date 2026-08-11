# ADR-297: RuView perception substrate — a phased program for the calibration, evidence, trust, and deployment layer

- **Status**: Accepted — program framing; child ADRs carry their own status
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: program, architecture, calibration, evidence, provenance, fusion, fleet, epic

## Context

Three independent analyses converged on the same conclusion in 2026: a deep
research sweep of the WiFi-sensing state of the art, an external technical and
industry review, and an internal strategic assessment. All three found that
RuView's gap is **not another sensing modality** but the horizontal layer that
turns RF research into repeatable spatial infrastructure — measurement,
calibration, out-of-distribution awareness, evidence accounting, authenticated
identity, a canonical spatial model, and fleet deployment.

Several of these primitives already have foundations in the tree and should be
**unified and made to produce signed, expiring certificates**, not rebuilt:

- `wifi-densepose-calibration` (enrollment, bank, anchor, runtime, specialist).
- `frame::EvidenceLevel` L0–L5 as mandatory policy (ADR-282).
- AetherArena benchmark infrastructure — v0 complete, CI-gated, witness ledger,
  live HF Space (ADR-149); board intentionally empty (benchmark-first).
- RuField provenance/signature types (ADR-260/262/277/279) and BFLD
  attestation (ADR-141).
- `worldgraph` crate; `wifi-densepose-mat/tracking` (tracker, fingerprint).
- The in-flight ADR-292 (provenance state machine), ADR-293 (authenticated
  data plane, step one), ADR-295 (model sanity gates) — the first bricks.

## What RuView is optimizing for

Not inference capability — **epistemic reliability**:

```
signal → observation → calibration → inference → uncertainty → evidence
       → certificate → policy → governed action
```

That pipeline is the product. The defensible category is not "RuView perceives
the physical world" but "RuView determines what machines are justified in
believing about it, proves why, and constrains what they may do with that
belief."

### Four non-negotiable program rules

Every child ADR and implementation is bound by these:

1. **UNKNOWN is a first-class output, never an error condition.** A surface that
   cannot answer says UNKNOWN and stays legible; it does not throw, default to a
   confident class, or silently hold a stale value.
2. **Capability certificates bind cryptographically.** Hardware, environment,
   model, calibration, metrics, expiry, and evidence level are bound under one
   signature (ADR-315/ADR-302). An unsigned or partially-bound certificate is
   not a certificate.
3. **One canonical semantics downstream.** Every surface (MQTT, REST, WebSocket,
   RuField, Matter, agents, UI) consumes the same Observation → Inference →
   GovernedEvent types (ADR-303). No transport- or UI-specific reinterpretation.
4. **Benchmarks expose worst-domain performance and confidence intervals.**
   Pooled accuracy is never sufficient for promotion (ADR-314).

### Certificate conditionality (the staleness guard)

The central architectural risk is **certificate staleness**: a room can remain
syntactically calibrated while its RF distribution has drifted enough to
invalidate the certificate. Therefore a capability certificate is **conditional
on a continuously evaluated domain signature** (ADR-299), not a one-time stamp.
Crossing the OOD threshold automatically degrades state and triggers
recalibration rather than silently continuing:

```
VALID → DEGRADED → UNKNOWN   (auto-degrade on domain drift; triggers recalibration)
```

This binds ADR-298 (calibration), ADR-299 (OOD), ADR-315 (certificate), and
ADR-318 (policy): a degraded/unknown domain must invalidate the affected
capability *before* a false confident inference reaches an actuator.

### Commercial framing — three primitives, not one product

- **RuView Runtime** — provides perception.
- **RuView Certify** — establishes what a deployment can legitimately claim
  (calibration + evidence + capability certificate + policy).
- **RuView Trust / Fleet** — keeps that claim valid across hardware, firmware,
  models, and environmental drift (ADR-313).

Certify and Trust are the parts that are hard to commoditize; presence
detection alone is not.

## Decision

Adopt a **21-primitive phased program**. Each primitive gets a child ADR
(ADR-298…ADR-318) that owns its detailed decision, status, and validation.
This ADR owns the framing, the dependency order, and the phase assignment.

### Primitive → ADR map

| # | Primitive | ADR | Phase |
|---|---|---|---|
| 1 | Automatic domain calibration | ADR-298 | 1 |
| 2 | Out-of-distribution detection | ADR-299 | 1 |
| 3 | Ground-truth synchronization | ADR-300 | 2 |
| 4 | Evidence engine | ADR-301 | 1 |
| 5 | Authenticated sensor identity | ADR-302 | 1 |
| 6 | Canonical spatial ontology | ADR-303 | 1 |
| 7 | Persistent identity & tracking | ADR-304 | 2 |
| 8 | Sensor placement optimizer | ADR-305 | 3 |
| 9 | Active sensing | ADR-306 | 3 |
| 10 | 802.11bf-native architecture | ADR-307 | 2 |
| 11 | Real sensor fusion | ADR-308 | 2 |
| 12 | Long-term spatial memory | ADR-309 | 3 |
| 13 | Counterfactual inference | ADR-310 | 3 |
| 14 | Information-gain scheduler | ADR-311 | 3 |
| 15 | Digital RF twin | ADR-312 | 3 |
| 16 | Fleet control plane | ADR-313 | 2 |
| 17 | Real benchmark service (multi-domain scorecard) | ADR-314 | 1 |
| 18 | Capability certificates | ADR-315 | 1 |
| 19 | Witness chain | ADR-316 | 1 |
| 20 | RuView sensor HAL | ADR-317 | 2 |
| 21 | Decision policy — action authorization | ADR-318 | 1 |

### Dependency order (why phase, not score, drives sequencing)

```
        ADR-303 spatial ontology ──┐
        ADR-302 auth identity ─────┼──► ADR-298 calibration cert ──► ADR-299 OOD gating
                                   │             │                        │
                                   └──► ADR-316 witness chain             │ (VALID→DEGRADED→UNKNOWN)
                                                 │                        ▼
                                   ADR-301 evidence engine ──► ADR-315 capability certificate
                                                 │                        │ (conditional on domain signature)
                                                 │                        ▼
                                                 │              ADR-318 decision policy ──► governed action
                                                 └──► ADR-314 benchmark scorecard (per-PR gate)
```

- **Phase 1 (the certificate spine, built now):** foundational roots 303, 302,
  301, 298 (implemented first, in their own crates); then the dependent wave
  316, 299, 315, 314, 318. This set is exactly the acceptance test decomposed
  and is buildable without new hardware (types, logic, signatures, tests). The
  dependent wave adds the staleness guard (299 auto-degrades 315) and the
  action gate (318) that denies at the actuator on a degraded/unknown domain.
- **Phase 2 (integration & operations):** 300 ground truth, 304 tracking, 307
  802.11bf-native, 308 fusion, 313 fleet, 317 HAL. Depends on the spine.
- **Phase 3 (higher-ceiling, research-forward):** 305 placement optimizer, 306
  active sensing, 309 spatial memory, 310 counterfactual, 311 info-gain
  scheduler, 312 RF twin. Sit on top of the fused world state.

Phase-2 and phase-3 child ADRs are authored as **Proposed** (design intent,
validation plan) and are not implemented by the phase-1 swarm.

### Acceptance test A — onboarding (from the strategic assessment)

> Connect a new sensor type in an unseen room. Within 30 minutes RuView should
> identify the hardware (HAL, ADR-317), calibrate the environment (ADR-298),
> quantify whether it can reliably sense the requested phenomenon (ADR-299),
> generate a signed capability certificate (ADR-315), expose governed spatial
> events (ADR-303), and return UNKNOWN whenever evidence falls outside that
> certificate (ADR-299).

### Acceptance test B — drift invalidation (the staleness guard)

> Deliberately change the room after certification — move furniture, change the
> AP channel, or substitute hardware. RuView should detect distribution drift
> (ADR-299), invalidate the affected capability (ADR-315) **before** a false
> confident inference reaches an actuator (ADR-318 denies with the specific
> failed condition), emit UNKNOWN, preserve the complete witness chain
> (ADR-316), and explain exactly which certificate condition failed.

Test B is the load-bearing one: it proves the substrate fails safe, not just
that it perceives well. Phase 1 makes every clause except HAL testable in
software; HAL (phase 2)
closes the "identify the hardware" clause.

## Consequences

- One coherent substrate replaces overlapping ad-hoc schemas; every surface
  (MQTT, REST, WebSocket, RuField, Matter, agents) eventually consumes the
  ADR-303 ontology and the ADR-315 certificate.
- Headline applications (pose/vitals/pointcloud models) are explicitly **not**
  the investment focus during this program, per the strategic direction.
- Later ADRs may be revised as the spine lands; that is expected for a phased
  program and is why phase-2/3 ADRs ship as Proposed.

## Validation

- Each child ADR defines its own tests. The program-level exit is the
  acceptance test above, run end-to-end once phase 1 lands, and encoded as an
  AetherArena scenario (ADR-314).
