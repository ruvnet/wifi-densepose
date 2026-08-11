# ADR-318: Decision policy — action authorization conditioned on certificate class, freshness, uncertainty, and evidence

- **Status**: Accepted — initial implementation planned (ADR-297 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: policy, authorization, safety, certificates, governed-action, phase-1

## Context

The perception substrate (ADR-297) makes RuView state *what it knows* and
*how well* — the capability certificate (ADR-315) binds hardware, environment,
model, calibration, metrics, expiry, and evidence level. But a certificate is a
statement of knowledge, not a grant of action. The same certificate that is
adequate to dim a light is wholly inadequate to release a door lock or clear an
industrial stop condition.

Without an explicit authorization layer, every consumer re-implements its own
(inconsistent, usually optimistic) rule for "is this good enough to act on,"
and a confident-but-out-of-domain inference can reach an actuator. That is the
exact failure the substrate exists to prevent. Decision policy therefore
belongs in **phase 1**, alongside the certificate it gates, not later.

This ADR realizes program invariant #1 (UNKNOWN is a first-class output, never
an error) and the action-side of the refined acceptance test: a drift-
invalidated capability must be *denied at the actuator* before a false
confident inference is acted upon.

## Decision

Introduce a `ruview-policy` crate providing an **action authorization gate**
that sits between governed spatial state and any actuator.

### 1. Assurance requirements per action

An `ActionClass` declares the assurance an action demands:

- `min_certificate_class` — the required `CapabilityCertificate` class (ADR-315).
- `max_certificate_age` / `min_domain_freshness` — the certificate must be
  currently valid **and** the live domain signature (ADR-299) must not be in a
  DEGRADED/UNKNOWN state (this is the staleness guard, program invariant on
  certificate conditionality — see ADR-297).
- `max_uncertainty` — inference uncertainty ceiling.
- `min_evidence_level` — the L0–L5 floor (ADR-282/ADR-301); e.g. a safety
  action may require ≥ L3 (held-out room+subject validation).

Reference action classes (illustrative, configurable):

| Class | Example | Typical floor |
|---|---|---|
| `Convenience` | lighting, scenes | tolerant: L1+, higher uncertainty ok |
| `Security` | alerts, arming | stricter: valid cert, L2+, bounded uncertainty |
| `SafetyCritical` | door lock, machine stop | strict: fresh cert, L3+, low uncertainty, KNOWN domain only |

### 2. The authorization decision

`authorize(action, capability_certificate, live_state) -> Authorization` where
`live_state` carries the current `SourceState` (ADR-292), OOD/domain state
(ADR-299), and inference uncertainty. Rules:

- **Fail-closed.** Any unmet condition → `Deny { failed_condition }`. The denial
  names the *specific* condition (expired cert, domain DEGRADED, uncertainty
  over ceiling, evidence below floor, certificate class too low).
- **UNKNOWN denies high-assurance actions.** A domain in UNKNOWN (ADR-299)
  cannot authorize `Security`/`SafetyCritical` actions; it may still authorize
  `Convenience` if that class's policy permits, but the authorization records
  that it proceeded under UNKNOWN.
- The decision is a **pure function** of (action class, certificate, live
  state) — deterministic and unit-testable without a clock or actuator.
- Every authorization (allow or deny) is emitted as the terminal stage of the
  witness chain (ADR-316), so "why was this actuator allowed/denied" is
  auditable end-to-end.

### 3. No silent optimism

A missing certificate, an expired certificate, or an unrecognized action class
all deny by default. Absence of a policy is not permission.

## Consequences

- Action authorization becomes uniform and centrally reasoned instead of
  per-consumer and optimistic; this is the "RuView Certify → constrains action"
  boundary that is hard to commoditize.
- A behavior change for existing automations that acted directly on presence:
  they now pass through the gate. Convenience-class defaults keep low-stakes
  automations working; high-stakes actions must opt into stricter classes.
- Depends on ADR-315 (certificate), ADR-299 (domain/OOD state), ADR-292
  (source state), ADR-301 (evidence). Built in the phase-1 dependent wave after
  those types land.

## Validation

- Unit tests: each action class authorizes/denies correctly across the matrix
  of (valid/expired/degraded cert × KNOWN/DEGRADED/UNKNOWN domain × uncertainty
  above/below ceiling × evidence above/below floor); UNKNOWN denies
  safety-critical; every deny names its failed condition; absence-of-policy
  denies; determinism.
- Integration: the acceptance-test scenario (ADR-297) — post-certification room
  change drives domain to DEGRADED→UNKNOWN, and a `SafetyCritical` authorization
  is denied with `failed_condition = domain_not_known` *before* the inference
  reaches the actuator, witness chain preserved.
- `cargo test -p ruview-policy`.
