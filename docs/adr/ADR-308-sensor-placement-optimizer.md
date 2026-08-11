# ADR-308: Sensor placement optimizer — floorplan + inventory → recommended positions

- **Status**: Accepted — initial implementation (ADR-300 phase 3)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: placement, planning, rf-twin, coverage, worldgraph, phase-3

## Context

This ADR is a child of **ADR-300** and owns primitive #8, *sensor placement
optimizer*. In the ADR-300 DAG it is a phase-3, research-forward primitive that
sits on top of the fused world state and is tightly coupled to **ADR-315**
(digital RF twin): the twin provides the propagation simulation this optimizer
plans against. It reads the **ADR-306** canonical ontology for the physical
scene and, after install, compares its predictions against ADR-302 observability
and the ADR-318 capability certificate.

The problem it solves is the single most common cause of a bad RuView
deployment: sensors placed by guesswork. Whether a room can be reliably sensed
depends on AP/sensor geometry relative to walls, Fresnel-zone clearance,
multipath structure, and where people actually move. Today an installer has no
principled way to answer "where do I put the two nodes I have so the kitchen is
observable?" — and no way, after install, to know whether reality matched the
plan. This is a genuine **differentiator**: it turns RuView from "sense
whatever the given placement happens to allow" into "recommend the placement
that makes the requested sensing feasible."

Relevant existing assets to build on rather than duplicate:

- The `worldgraph` crate models the physical scene the optimizer plans over:
  `Room`/`Space` with `bounds_enu`, `Wall { rf_attenuation_db }` (drywall ≈ 3
  dB, brick ≈ 12 dB), `Doorway`, and `Zone` — enough geometry and coarse RF
  attenuation to seed a coverage model, plus `Sensor` nodes (ADR-306) for
  candidate positions.
- **ADR-315** (RF twin, phase 3) is the propagation/multipath simulator; this
  optimizer is a *consumer* of the twin, not a second simulator.
- **ADR-302** (OOD/observability) and **ADR-318** (capability certificate)
  define what "reliably sense the requested phenomenon" means, so the optimizer
  can optimize against the same observability metric the runtime later gates on.
- **ADR-029** (multistatic) and **ADR-063** (mmWave fusion) inform which link
  geometries are useful for which phenomena.

## Options considered

1. **Static placement guidelines in docs (e.g. "one node per room, opposite
   the door").** Rejected: ignores the specific floorplan, wall materials, and
   the actual hardware inventory; gives no uncertainty and no post-install
   feedback.
2. **Full electromagnetic solver per site.** Rejected for the default path:
   too heavy for an installer workflow and overkill relative to the coarse
   `rf_attenuation_db` scene RuView actually has; reserved as an optional
   high-fidelity backend inside ADR-315.
3. **A coverage optimizer that consumes the ADR-315 RF twin over the ADR-306
   scene, then validates predicted vs. measured observability after install.**
   Chosen.

## Decision

Define a **placement optimizer** that takes a floor plan (ADR-306 scene) and a
hardware inventory and recommends sensor positions, then closes the loop after
install.

### 1. Inputs

- The ADR-306 canonical scene: `Space`/`Zone` bounds, `Wall` segments with
  `rf_attenuation_db`, `Doorway` topology, and any already-placed `Sensor`
  nodes.
- A hardware inventory: the count and type of available radios (ESP32-S3/C6
  nodes, mmWave, adapters) with their capability envelopes (what each can
  sense, per ADR-318 / ADR-320 HAL descriptors).
- A sensing objective: which phenomenon must be observable in which
  `Space`/`Zone` (presence, vitals, pose), expressed against the ADR-302
  observability metric.

### 2. Prediction

- For a candidate placement, query the **ADR-315 RF twin** for simulated RF
  coverage: path loss through `Wall` attenuation, **Fresnel-zone clearance**
  between link endpoints, and coarse **multipath** structure. From that derive
  an **expected observability** and an **uncertainty** for each objective in
  each space — reusing the same observability definition ADR-302 gates on so the
  plan and the runtime speak one language.
- Search over candidate positions (the inventory bounds the count; the scene
  bounds the geometry) to recommend the placement that maximizes objective
  observability, reporting expected observability **and its uncertainty** per
  space — never a single confident number for a simulated result.

### 3. Post-install loop

- After install, compare **predicted vs. measured** observability using the
  ADR-302 runtime observability signal from the freshly enrolled (ADR-305),
  calibrated (ADR-301) sensors. Where measurement disagrees with prediction,
  recommend adjustments (move, re-aim, add a node) and feed the residual back
  to improve the ADR-315 twin's scene parameters (e.g. a wall's effective
  attenuation).

### Evidence discipline

- Predicted coverage is a **simulation** (evidence level L0 per ADR-282) and is
  labelled `SYNTHETIC`; it is a *recommendation*, never a sensing claim.
- The predicted-vs-measured comparison is the only place a `MEASURED` statement
  appears, and only with a reproducer and real-silicon observability data
  (CLAUDE.md hardware rule). The optimizer never presents a simulated coverage
  map as evidence that a room *is* being sensed.

## Consequences

- Installers get a principled, floorplan-specific placement plan and, crucially,
  a post-install check that says whether reality matched the plan — a
  differentiating capability over guess-and-check deployment.
- Quality is bounded by the fidelity of the ADR-315 RF twin and the coarseness
  of the `worldgraph` scene (2D walls, coarse attenuation). The optimizer
  reports uncertainty rather than overstating a coarse model; higher fidelity
  is an ADR-315 concern.
- Hard dependency on ADR-315 (twin), ADR-302 (observability metric), and
  ADR-306 (scene); this ADR does not build a simulator or an observability
  metric of its own.
- Being phase 3, this is design intent sitting on the fused world state; it is
  expected to be revised as ADR-315 and the phase-1 spine land.
- No claim that recommended placement *guarantees* sensing — it maximizes
  modelled observability subject to inventory and geometry, with explicit
  uncertainty.

## Validation

- Unit tests: coverage/observability prediction is a deterministic function of
  scene + placement + twin parameters; Fresnel-zone and wall-attenuation math
  against known analytic cases; search returns the modelled-optimal placement on
  small synthetic scenes.
- Integration test: on a synthetic floorplan with a known-good and a
  known-bad placement, the optimizer ranks them correctly and reports higher
  uncertainty for the marginal case.
- Post-install loop test: injected predicted-vs-measured disagreement produces a
  sensible adjustment recommendation and a twin-parameter residual.
- Field validation (deferred, real-silicon): predicted vs. measured
  observability on an instrumented real site, reported as `MEASURED` with a
  reproducer. Until then all coverage output is `SYNTHETIC`/L0. No coverage or
  accuracy number is asserted by this ADR.
