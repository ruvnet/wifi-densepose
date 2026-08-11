# ADR-306: Active sensing — closed-loop RF experiment control

- **Status**: Proposed (ADR-297 phase 3)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: active-sensing, control-plane, closed-loop, information-gain, actuation, phase-3

## Context

This ADR is a child of **ADR-297** and owns primitive #9, *active sensing*. In
the ADR-297 phasing it is a phase-3 primitive that sits on top of the fused
world state produced by **ADR-308** (real sensor fusion) and is driven by the
information budget of **ADR-311** (information-gain scheduler). It is authored
as **Proposed**: design intent and validation plan, not a phase-1 build.

The default posture of every current RuView path is **passive**: RF traffic
happens for its own reasons (a device transmits, a beacon fires), RuView
observes whatever CSI/CIR arrives, and the pipeline extracts what it can from
that incidental signal. The strategic assessment behind ADR-297 named the next
step: move from *RF-happens → observe* to **RuView-controls-RF → observe the
response → optimize the next measurement**. That turns sensing into a
closed-loop experiment — the system chooses what to measure to resolve the
uncertainty it currently has, rather than accepting the measurements the
environment happens to offer.

Substantial control-plane scaffolding already exists and must be
**reused/extended, not rebuilt**:

- **ADR-280** (active sensing / programmable perception, *implemented* in
  `ruview-unified/src/control.rs`) already defines the governed control surface
  this ADR closes the loop over: `SensingTask` (evidence-aware, fail-closed
  admission), `SensingAction` + `InformationGoal` (a deliberate act of
  evidence-gathering against a stated hypothesis, bounded by a `PrivacyClass`
  P0–P5 ceiling), `ActiveSensingPlanner` (age-of-information scheduler),
  `CoherentSensorGroup` (coherent fusion fails closed), and `request_actuation`
  → `ActuationReceipt` for governed RIS/movable/fluid-antenna actuation.
- ADR-280 explicitly recorded that **information-gain *estimation* is not
  implemented** — "the planner uses staleness heuristics, not mutual
  information; RIS drivers, actual multi-AP coherence measurement, and OTFS
  waveform control are hardware-dependent roadmap items." ADR-306 is the ADR
  that closes exactly those gaps, in coordination with ADR-311.

The missing piece is not the actuation surface — ADR-280 built that and made it
fail closed — but the **loop**: a controller that reads the current fused-state
uncertainty, selects a *controllable measurement configuration* expected to
reduce it most, requests it through the ADR-280 governed surface, observes the
response, and updates its belief before choosing the next measurement.

## Options considered

1. **Stay passive; only schedule which incidental observations to keep.** This
   is roughly today's `ActiveSensingPlanner` (staleness-priority over regions).
   Rejected as the endpoint: it optimizes *attention* over uncontrolled RF, not
   the *measurement* itself. It remains the fallback when nothing is
   controllable.
2. **Open-loop measurement scripting** (a fixed sweep of channels/bandwidths).
   Rejected: a fixed sweep spends the RF/energy/privacy budget the same way
   regardless of what is already known; it cannot concentrate measurement where
   uncertainty actually is.
3. **Closed-loop experiment control** — read uncertainty, pick the controllable
   configuration with highest expected information gain per unit cost/privacy,
   actuate through the ADR-280 governed surface, observe, update, repeat.
   Chosen.

## Decision

Adopt **closed-loop RF experiment control** as a phase-3 controller layered on
the ADR-280 surface. RuView selects and drives the controllable degrees of
freedom of the RF measurement, then optimizes the next measurement from the
observed response.

### 1. Controllable degrees of freedom

Define an `ExperimentControl` vocabulary over the configuration axes RuView can
influence on hardware that exposes them (each axis is optional and
capability-gated by ADR-317's HAL, so an ESP32-only deployment simply has an
empty controllable set and degrades to the passive planner):

- **Channel / band** and **bandwidth** (which spectrum to probe; reuses the
  ADR-289 wideband subcarrier-agnostic metadata).
- **Packet timing / cadence** (when to solicit a sounding, and at what rate).
- **Antenna / chain selection** (which subset of a distributed aperture to
  activate — bounded by the ADR-280 `CoherentSensorGroup` compatibility proof).
- **Beam / RIS configuration** (which rooms and people become observable —
  governed exactly as ADR-280 §6 requires, via `request_actuation` and an
  `ActuationReceipt`).
- **802.11bf measurement parameters** (TB/non-TB, reporting config) once
  ADR-307 exposes standardized sensing as a native measurement type.

### 2. The loop

```
fused-state uncertainty (ADR-308)
        │
        ▼
info-gain ranking of ExperimentControl options (ADR-311)
        │  select argmax  E[ΔI] / (cost, energy, privacy ceiling)
        ▼
governed request  (ADR-280 admit_task / request_actuation, fail-closed)
        │
        ▼
observe response → update belief (ADR-308) → repeat
```

The controller never bypasses the ADR-280 admission and actuation gates: every
solicited measurement is a `SensingTask`/`SensingAction`, every environment
change is an `ActuationReceipt`, and every step composes with the ADR-277
policy engine. Information gain is what **ADR-311** supplies (the mutual-
information estimate ADR-280 deferred); ADR-306 owns the *control loop* that
consumes that estimate and drives the hardware.

### 3. Governance and honesty boundary

- Actuation and solicitation stay fail-closed and privacy-ceilinged: a
  closed-loop experiment cannot widen the P0–P5 ceiling of the task it serves,
  and cannot steer a beam into a zone that does not grant the purpose (ADR-280
  `actuation_requires_policy_authorization`).
- Any accuracy or "traffic-reduction" claim from the closed loop is tagged
  **MEASURED** only with a named reproducer over a stated scenario, **SYNTHETIC**
  for simulated apertures, and **CLAIMED** otherwise. Real multi-AP coherent
  measurement and RIS actuation remain **hardware-dependent** and require
  real-silicon evidence (a captured runtime log) before any hardware claim, per
  CLAUDE.md. No number is invented here.

## Consequences

- Sensing becomes an experiment: RuView spends its RF/energy/privacy budget on
  the measurements that most reduce current uncertainty, instead of processing
  whatever incidental traffic arrives.
- The loop is only as strong as its two dependencies: ADR-308 must expose a
  usable uncertainty surface and ADR-311 must produce trustworthy information-
  gain estimates. Where either is absent, the controller degrades to the
  ADR-280 staleness planner rather than acting on a fabricated gain estimate.
- Controllability is hardware-bounded. On commodity ESP32 sensors the
  controllable set may be limited to cadence; the full loop (bandwidth, antenna,
  beam) needs NICs/RIS that expose those axes, surfaced through ADR-317.
- This ADR adds a controller; it does not re-open ADR-280's raw-export or
  actuation-governance decisions, which remain authoritative and fail-closed.

## Validation

- Design-level acceptance (phase 3): a simulated closed loop over a synthetic
  scene reduces terminal fused-state uncertainty faster than (a) the passive
  ADR-280 staleness planner and (b) an open-loop fixed sweep, at equal
  measurement budget — reported **SYNTHETIC**, with the scenario and seed named.
- Governance tests: every solicited measurement and actuation in the loop is
  admitted through the ADR-280 fail-closed path; a loop step that would exceed
  the task's privacy ceiling or steer into an ungranted zone is denied.
- Degradation test: with an empty controllable set (ESP32-only), the controller
  falls back to the staleness planner with no error and no fabricated gain.
- Hardware validation of bandwidth/antenna/beam actuation is explicitly out of
  scope until real silicon exposes those axes and produces a captured log.
