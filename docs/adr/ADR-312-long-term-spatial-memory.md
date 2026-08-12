# ADR-312: Long-term spatial memory — learn the normal physics of a location

- **Status**: Accepted — initial implementation (ADR-300 phase 3)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: spatial-memory, ruvector, anomaly-detection, temporal, world-state, phase-3

## Context

This ADR is a child of **ADR-300** and owns primitive #12, *long-term spatial
memory*. In the ADR-300 phasing it is a phase-3 primitive that sits on the fused
world state produced by **ADR-311** (real sensor fusion) and **ties to ADR-315**
(digital RF twin): spatial memory is the *learned normal* that a twin can
simulate against and that anomaly detection compares against. It is authored as
**Proposed**.

The capability is to **learn the normal physics of a location** so anomalies
surface *without training a detector for every anomaly*. Concretely, the system
should learn statements like: "a chair is normally here"; "this bedroom is
usually occupied between these hours"; "the RF propagation of this space
changed"; "this machine's vibration signature changed"; "a new reflector
appeared." None of these is a labeled anomaly class — they are *deviations from
a learned baseline of normality*. This is the difference between supervised
anomaly detection (which needs examples of every failure) and **baseline-relative
anomaly detection** (which needs only a well-characterized normal).

Substantial substrate already exists and must be **reused/extended, not
rebuilt**:

- **RuVector** (`v2/crates/wifi-densepose-ruvector`) is the designated substrate
  in the ADR-282 layer stack ("persistent objects, Gaussian fields, scene
  graphs, temporal memory"). It already provides the vector/temporal machinery
  this ADR needs — HNSW indexing (`hnsw.rs`, `hnsw_quantized.rs`), an event log
  (`event_log.rs`), coverage and estimator surfaces, and the `crv`/`mat`
  temporal sub-modules — so long-term spatial memory is a *consumer and
  organizer* of RuVector primitives, not a new store.
- **ADR-306** supplies the entity vocabulary the memory is indexed by (`Space`,
  `Object`, `Sensor`, `Track`, `Event`); **ADR-311** supplies the fused,
  uncertainty-carrying `WorldState` snapshots that memory accumulates over time.
- **ADR-135** (empty-room baseline calibration) and **ADR-301** (automatic
  domain calibration) already establish a *calibration-time* baseline of a
  space; ADR-312 extends that from a one-shot baseline to a **continuously
  learned, time-of-day-aware** model of normal.

What is missing is the **temporal normality model**: a per-`Space` learned
distribution of fused world states over time (including periodicity — hour of
day, day of week), plus RF-propagation and modality-signature baselines, against
which a live fused state is scored for deviation.

## Options considered

1. **Supervised anomaly classifiers per anomaly type.** Rejected: it needs
   labeled examples of every anomaly (fall, intrusion, machine fault, moved
   furniture), which do not exist for most spaces and do not transfer between
   rooms; it also cannot catch a *novel* anomaly it was never trained on.
2. **Single static baseline** (the ADR-135 empty-room snapshot, used forever).
   Rejected as the endpoint: it cannot express *when* a space is normally
   occupied, cannot track slow legitimate drift (furniture rearranged on
   purpose), and flags every diurnal change as anomalous.
3. **Continuously learned, time-aware normality model on the RuVector
   substrate**, scoring live fused state against learned normal. Chosen.

## Decision

Adopt a **long-term spatial memory** that learns each location's normal physics
on the RuVector substrate and scores live fused state against it.

### 1. What "normal" is learned over

Per ADR-306 `Space` (and the entities within it), accumulate the ADR-311 fused
`WorldState` over time into a learned normality model covering:

- **Occupancy / activity periodicity** — the distribution of presence and
  activity by hour-of-day and day-of-week (the "bedroom usually occupied certain
  hours" case).
- **Static scene layout** — persistent `Object` positions and the expected
  reflector set (the "chair normally here" / "new reflector appeared" cases),
  building on the ADR-135/298 baseline.
- **RF-propagation baseline** — the space's normal multipath/propagation
  signature (the "RF propagation changed" case).
- **Per-modality signatures** — e.g., a machine's normal vibration/acoustic/IMU
  signature (the "vibration signature changed" case).

Each learned baseline carries its own uncertainty and an `EvidenceLevel`
(ADR-282); a baseline learned from replay is L1, from a field pilot L4, and is
never presented above the evidence of the observations it was learned from.

### 2. Substrate: RuVector, temporally compressed

- The memory is stored and indexed on RuVector (HNSW for nearest-normal recall,
  the event log for the temporal stream, the temporal sub-modules for
  compression). Long-horizon history is temporally compressed — recent detail
  retained, older history summarized — so memory cost is bounded rather than
  growing linearly forever.
- The memory is *keyed by* the ADR-306 ontology, so "normal for this `Space` at
  this hour" is a first-class query, and slow legitimate drift updates the
  baseline (with provenance) instead of accumulating as permanent anomaly.

### 3. Anomaly = deviation from learned normal

- A live fused `WorldState` is scored against the applicable learned baseline
  (matched by space and time context). A deviation beyond the baseline's
  uncertainty is surfaced as an ADR-306 `Event` — *without* a per-anomaly
  detector — carrying the baseline it deviated from, the deviation magnitude,
  and its evidence level. Whether that event is actionable is a policy/consumer
  decision (ADR-277), not this layer's.
- The learned normal is exactly what **ADR-315** (RF twin) can simulate against:
  the twin proposes an expected state, spatial memory supplies the learned
  actual-normal, and their divergence is a physically grounded anomaly signal.

## Consequences

- Anomaly detection generalizes: a space gets deviation detection from its own
  learned normal, so a novel anomaly (never labeled anywhere) still registers as
  a deviation, and the model transfers to a new room by *learning that room's*
  normal rather than importing a foreign detector.
- Bounded memory: temporal compression keeps long-horizon memory finite; the
  trade-off is that fine detail of old history is summarized, which is acceptable
  for a normality baseline.
- Legitimate change is not a permanent false positive: slow drift updates the
  baseline with provenance, distinguishing "furniture deliberately rearranged"
  (baseline shifts) from "reflector appeared unexpectedly" (deviation event).
- **No accuracy claim is made.** Deviation-detection quality is not asserted
  here; any detection-rate or false-positive number requires a named reproducer
  tagged MEASURED / SYNTHETIC / CLAIMED, and a health/safety framing stays within
  the ADR-282 bounded-claims discipline (decision support, not diagnosis). No
  number is invented.
- The memory is governed: learned baselines are observations of a space, subject
  to the same ADR-277 retention/privacy policy as the fused state they summarize;
  no raw P0 RF is retained to build a baseline.

## Validation

- `cargo test -p wifi-densepose-ruvector` — the normality model builds on the
  existing HNSW/event-log/temporal primitives; nearest-normal recall and
  temporal-compression bounds are exercised on synthetic streams.
- Baseline/deviation tests: a synthetic scene with a known injected change (moved
  `Object`, altered propagation, altered modality signature) produces a deviation
  `Event` against the learned normal *without* a per-anomaly detector; an
  unchanged diurnal cycle produces none (no false positive on normal periodicity).
- Drift test: a slow legitimate change updates the baseline (with provenance)
  rather than emitting a persistent anomaly; an abrupt change does emit one.
- Evidence test: a learned baseline carries the evidence level of its source
  observations and is never presented above it; retention honors ADR-277.
- Twin-linkage design check (with ADR-315): divergence between a twin-simulated
  expected state and the learned normal is expressible as a deviation signal.
