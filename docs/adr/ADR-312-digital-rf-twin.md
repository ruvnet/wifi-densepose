# ADR-312: Digital RF twin — persistent per-deployment RF model

- **Status**: Proposed (ADR-297 phase 3)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: rf-twin, digital-twin, propagation, calibration, spatial-memory, worldgraph, phase-3

## Context

This ADR is a child of **ADR-297** (perception substrate program) and owns
primitive #15, *digital RF twin*. In the ADR-297 DAG it is a phase-3,
research-forward primitive that underpins several other phase-3 primitives:
**ADR-305** (placement optimizer) plans against the twin's propagation model,
**ADR-310** (counterfactual inference) uses it as the generative forward model,
and **ADR-311** (information-gain scheduler) uses it to predict per-sensor
informativeness. It ties directly to **ADR-298** (calibration), **ADR-305**
(placement), and **ADR-309** (long-term spatial memory). It is authored as
Proposed and is not implemented by the phase-1 swarm.

RuView today has no persistent, per-deployment model of the RF environment.
Calibration state, observed multipath, and radio geometry exist transiently
inside a running session; when the process restarts or a change happens
overnight, there is nothing that says "this is what this room's RF looked like
yesterday." Without a persistent baseline, a physical change — furniture moved,
a wall opened, a machine relocated, an intruder present — has nothing to be a
*delta against*. It is just a different measurement, indistinguishable from
noise or drift.

The **digital RF twin** is that persistent baseline: a per-deployment model
holding

- **geometry and radio locations** (from the ADR-303 scene / worldgraph),
- **propagation history** and **observed multipath** structure,
- **calibration state** (from ADR-298),
- **expected measurement distributions** for each link and phenomenon.

Once the twin exists, a physical change becomes a **measurable delta against the
twin** rather than an unexplained measurement. This is what connects RuView to
facility management (what changed in this space?), security (is there an
unexplained presence?), robotics (has the map drifted?), and industrial
monitoring (did the plant layout change?) — the applications the strategic
assessment named as the value beyond a single detector.

Relevant existing assets to build on rather than duplicate:

- The `worldgraph` crate already models the physical scene — `Room`/`Space`
  with `bounds_enu`, `Wall { rf_attenuation_db }`, `Doorway`, `Zone`, and
  `Sensor` nodes (ADR-303). The twin *annotates and persists* this scene with RF
  state; it does not invent a second geometry.
- `wifi-densepose-calibration` (enrollment, bank, anchor, runtime, specialist)
  holds the calibration state the twin persists; the twin references and
  versions calibration records, it does not reimplement calibration.
- **ADR-309** (long-term spatial memory, phase 3) is the persistence and
  temporal-history substrate; the twin is a *structured occupant* of that
  memory, not a separate database.
- **ADR-302** (authenticated identity) and **ADR-292** (provenance) mean the
  measurements that update the twin carry verified lineage, so a delta is
  attributable rather than anonymous.

## Options considered

1. **No persistent RF model (status quo).** Rejected: every change looks like
   noise; nothing supports "what changed since yesterday?", which is the
   question the facility/security/industrial applications actually ask.
2. **A full electromagnetic digital twin (per-site ray-tracing / FDTD kept in
   sync in real time).** Rejected for the default path: far heavier than the
   coarse `rf_attenuation_db` scene RuView actually has and impractical on edge
   hardware. A high-fidelity solver is retained as an *optional backend* the
   twin can call, not the baseline.
3. **A persistent, per-deployment RF model layered over the ADR-303 scene and
   ADR-309 memory: geometry + radio locations + calibration state + observed
   multipath + expected measurement distributions, updated by verified
   measurements, exposing changes as deltas.** Chosen.

## Decision

Define the **digital RF twin** as a persistent, versioned, per-deployment model
of the RF environment, layered over existing scene, calibration, and memory
assets.

### 1. State the twin holds

- **Geometry and radio locations** referenced from the ADR-303 / worldgraph
  scene (not copied).
- **Calibration state** referenced and versioned from
  `wifi-densepose-calibration` (ADR-298), so the twin knows *which* calibration
  a stored distribution was captured under.
- **Observed multipath and propagation history** — a bounded temporal summary
  of per-link channel structure, stored in ADR-309 spatial memory.
- **Expected measurement distributions** per link and phenomenon — the forward
  model ADR-305, ADR-310, and ADR-311 consume.

### 2. Update and delta

- Verified measurements (ADR-302 identity, ADR-292 provenance) update the twin's
  distributions online, bounded by ADR-298 calibration validity. A new
  observation is compared to the twin's expected distribution; the **delta** —
  and its statistical significance against the twin's own variance — is the
  primary output. A change large relative to the twin's modelled variance is a
  *detected physical change*, not noise.
- The twin is **versioned**: a calibration event, a deliberate geometry edit, or
  an accepted physical change advances the twin version, so history is
  auditable and a delta is always relative to a named baseline.

### 3. Consumers

- **ADR-305** queries the twin's propagation model to plan placements.
- **ADR-310** uses the twin's expected distributions as the generative forward
  model for hypothesis scoring.
- **ADR-311** uses per-sensor expected informativeness from the twin.
- Facility/security/robotics/industrial integrations read the twin's change
  deltas as governed ADR-303 spatial events.

### Evidence discipline

- The twin's expected distributions and any propagation simulation are
  **simulation** (evidence level L0 per ADR-282), labelled `SYNTHETIC`. A delta
  computed against them is a model-relative statement.
- A change/anomaly detection *claim* (e.g. "detects furniture-scale changes")
  requires real-silicon measurement against a leakage-free protocol with a
  reproducer before it is tagged `MEASURED` (CLAUDE.md hardware rule). The twin
  never presents a modelled expected distribution as evidence that a physical
  state *is* the case; it presents a *delta and its significance*. This ADR
  asserts **no** detection-accuracy number.

## Consequences

- RuView gains a persistent per-deployment baseline, turning "a different
  measurement" into "a measurable, attributable, versioned change" — the bridge
  from a sensing runtime to facility management, security, robotics, and
  industrial monitoring.
- The twin is the shared forward model for ADR-305/310/311, so those primitives
  speak one propagation model rather than three inconsistent ones — a
  deliberate reason to build the twin before its consumers mature.
- Quality is bounded by the coarseness of the worldgraph scene and the fidelity
  of the forward model; the twin reports deltas *with significance against its
  own variance* rather than asserting confident change detection on a coarse
  model. The optional high-fidelity backend is where higher accuracy lives.
- Hard dependency on ADR-303 (scene), ADR-298 (calibration state), and ADR-309
  (persistence); it reuses `worldgraph` and `wifi-densepose-calibration` rather
  than rebuilding geometry or calibration.
- Being phase 3, this is design intent; it is expected to be revised as the
  phase-1 spine, ADR-308 fusion, and ADR-309 memory land.

## Validation

- Unit tests: the twin's expected distribution is a deterministic function of
  scene + calibration + propagation history; delta computation and its
  significance against stored variance are correct on synthetic distributions;
  versioning advances on calibration/geometry/accepted-change events and history
  is retained.
- Integration test: on a synthetic deployment, an injected physical change (a
  wall attenuation shift) produces a significant delta against the twin while
  ordinary noise does not; the delta surfaces as a governed ADR-303 event with
  provenance (ADR-302/292).
- Field validation (deferred, real-silicon): change detection on an instrumented
  real deployment with a controlled physical-change protocol, reported as
  `MEASURED` with a reproducer. Until then all twin distributions and deltas are
  `SYNTHETIC`/L0. No detection-accuracy number is asserted by this ADR.
