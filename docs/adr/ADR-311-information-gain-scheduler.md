# ADR-311: Information-gain scheduler — sample the most informative radios

- **Status**: Accepted — initial implementation (ADR-297 phase 3)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: scheduling, active-sensing, information-gain, edge, energy, fusion, phase-3

## Context

This ADR is a child of **ADR-297** (perception substrate program) and owns
primitive #14, *information-gain scheduler*. In the ADR-297 DAG it is a phase-3,
research-forward primitive that sits on top of the fused world state and
**pairs with ADR-306** (active sensing): ADR-306 decides *what to probe*
(waveform, sensing task); this ADR decides *which radios/modalities to spend
budget on next*. It is authored as Proposed and is not implemented by the
phase-1 swarm.

With multiple sensors, processing every stream at full rate is wasteful: many
radios are, at any moment, contributing little to the current estimate while
consuming compute, energy, and bandwidth — the three scarce resources on the
edge nodes RuView targets (ESP32-S3/C6 and small gateways). Treating all sensors
equally is precisely the design that does not survive a real deployment of
"hundreds of sensors."

The scheduler assigns each candidate sensor/modality a value

```
Value(sensor) ≈ expected uncertainty reduction / (compute + energy + bandwidth)
```

and spends the next sampling/processing budget on the highest-value sensors.
Expected uncertainty reduction is estimated *before* paying for the measurement,
which is why the scheduler needs a model of what each sensor is likely to tell
it — supplied by the fused state's covariance and the RF twin's forward model,
not by actually sampling.

Relevant existing assets to build on rather than duplicate:

- **ADR-308** (fusion) maintains the fused state and its covariance — the
  current uncertainty the scheduler is trying to reduce. Expected uncertainty
  reduction is computed against that covariance, not a private one.
- **ADR-312** (RF twin) provides the per-sensor forward model used to predict a
  candidate measurement's expected informativeness before sampling.
- **ADR-317** (RuView sensor HAL, phase 2) exposes each radio's real
  compute/energy/bandwidth cost descriptors; the denominator is read from the
  HAL, not guessed per platform.
- **ADR-306** (active sensing) is the paired actuator: the scheduler ranks
  sensors, ADR-306 chooses the probe on the chosen sensor.
- **ADR-299** (observability) defines the phenomenon the estimate is *for*, so
  the scheduler prioritizes uncertainty reduction on the objective that matters,
  not on nuisance dimensions.

## Options considered

1. **Round-robin / process-everything scheduling.** Rejected: burns edge
   compute and energy on redundant streams and does not scale to large fleets;
   the strategic and external reviews named exactly this as an edge-deployment
   blocker.
2. **Static priority per sensor type (e.g. always prefer mmWave).** Rejected:
   ignores that a sensor's *current* informativeness depends on the scene and
   the present uncertainty — a well-placed WiFi link can dominate an occluded
   mmWave node in a given moment.
3. **A value-of-information scheduler that ranks sensors by expected uncertainty
   reduction per unit cost, using the ADR-308 covariance and ADR-312 forward
   model, with costs from the ADR-317 HAL.** Chosen.

## Decision

Define an **information-gain scheduler** that allocates the next
sampling/processing budget across available radios by value of information.

### 1. Value function

- For each candidate sensor/modality, estimate **expected uncertainty
  reduction** on the ADR-299 objective by evaluating how much a predicted
  measurement (via the **ADR-312** forward model) would shrink the **ADR-308**
  fused-state covariance — a value-of-information estimate made *before* paying
  for the measurement.
- Divide by the sensor's **cost** — compute + energy + bandwidth — read from the
  **ADR-317** HAL descriptors. The exact weighting of the three cost terms is a
  deployment policy (a battery node weights energy heavily; a wired gateway
  weights bandwidth), configured, not hardcoded.

### 2. Allocation

- Rank candidates by value and spend the budget on the top set, subject to a
  configurable floor that guarantees each sensor is sampled at least
  occasionally (so a sensor whose value is currently low is not starved into
  permanent blindness and can be re-evaluated as the scene changes).
- The scheduler emits an allocation, not a measurement; **ADR-306** active
  sensing chooses the probe/waveform on each selected sensor, and the fusion
  layer (ADR-308) incorporates the result.

### 3. Governance and honesty

- Skipping a sensor for a cycle is a *deliberate* reduction in coverage; the
  scheduler records which sensors were sampled so downstream evidence (ADR-301)
  reflects the actual sensing that occurred, and observability (ADR-299) can
  raise `UNKNOWN` for a zone that went under-sampled rather than reporting a
  stale estimate as current.

### Evidence discipline

- Expected-uncertainty-reduction estimates are model predictions from the
  ADR-312 twin (simulation, L0 per ADR-282, `SYNTHETIC`); a scheduling decision
  is a resource choice, never a sensing claim.
- Any energy/latency/throughput improvement figure requires real-silicon
  measurement with a reproducer before it is tagged `MEASURED` (CLAUDE.md
  hardware rule). This ADR asserts **no** efficiency number.

## Consequences

- Edge deployments spend scarce compute, energy, and bandwidth where they buy
  the most certainty, making "hundreds of sensors" operationally tractable — a
  capability the reviews flagged as critical for edge deployment.
- Quality is bounded by the accuracy of the ADR-312 forward model (informativeness
  prediction) and ADR-317 cost descriptors; a poor forward model degrades to
  near-round-robin, which is safe but not optimal. The sampling floor bounds the
  worst case.
- Hard dependency on ADR-308 (covariance), ADR-312 (forward model), and ADR-317
  (cost descriptors), and paired with ADR-306; this ADR builds none of those.
- Being phase 3, this is design intent sitting on the fused world state and is
  expected to be revised as ADR-306, ADR-308, ADR-312, and the ADR-317 HAL land.

## Validation

- Unit tests: the value function is a deterministic function of covariance +
  forward model + cost descriptors; a sensor predicted to reduce objective
  uncertainty more per unit cost ranks above one that reduces it less; the
  sampling floor guarantees eventual re-evaluation of a low-value sensor.
- Integration test: on a synthetic multi-sensor scene, the scheduler reduces
  objective uncertainty faster per unit modelled cost than round-robin, and
  raises ADR-299 UNKNOWN for a deliberately starved zone rather than reporting a
  stale estimate.
- Field validation (deferred, real-silicon): energy/latency/throughput on an
  instrumented multi-node deployment, reported as `MEASURED` with a reproducer.
  Until then all informativeness and cost figures are `SYNTHETIC`/L0. No
  efficiency number is asserted by this ADR.
