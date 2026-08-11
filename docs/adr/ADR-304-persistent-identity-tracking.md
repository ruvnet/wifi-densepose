# ADR-304: Persistent identity & tracking — privacy-preserving probabilistic tracks

- **Status**: Accepted — initial implementation (ADR-297 phase 2)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: tracking, identity, privacy, fusion, worldgraph, phase-2

## Context

This ADR is a child of **ADR-297** and owns primitive #7, *persistent identity
& tracking*. In the ADR-297 DAG it is a phase-2 primitive sitting on the
phase-1 spine: it **consumes the ADR-303 ontology** (producing and updating the
`Track` and `Person` node types defined there), it relies on **ADR-302**
authenticated identity so that the observations it associates have a verified
origin, and its outputs are governed `Event`s that ADR-315/ADR-316 can certify
and witness.

The product need is to reason about *persistent entities* — "person_7 entered
the kitchen, then the hallway, then the bedroom" — across radios, modalities,
rooms, and time. The hard constraint is that this must happen **without
establishing civil identity**. RuView is camera-free (ADR-282), and a
persistent pseudonymous track must never become, or be joinable to, a real-
world named individual. This is a privacy property to be enforced *by
construction*, not a policy footnote.

Substantial scaffolding already exists in
`v2/crates/wifi-densepose-mat/src/tracking` and must be **reused/extended, not
rebuilt**:

- `SurvivorTracker`, `TrackedSurvivor`, `TrackId`, `TrackerConfig`,
  `TrackLifecycle`, and `TrackState` — a multi-target tracker with lifecycle
  (tentative/active/lost/terminal) and a `TrackId` backed by a UUID
  (`as_uuid`).
- `KalmanState` with `predict`/`update`, `position`, `velocity`,
  `position_uncertainty`, and `mahalanobis_distance_sq` — the motion model and
  gating distance.
- `CsiFingerprint`, `DetectionObservation`, `AssociationResult`, and the
  `can_reidentify`/`matches`/`mark_rescued`/`rescue` re-identification surface —
  the appearance/fingerprint channel for track continuity.

What is missing is (a) continuity **across radios, modalities, and rooms** (the
tracker today reasons within a node/room context), (b) a **persistent** entity
that survives track loss and hand-off between spaces, and (c) an explicit
**privacy boundary** that guarantees no civil-identity binding.

## Options considered

1. **Per-room independent trackers, no cross-room identity.** Rejected: cannot
   express "person_7 moved kitchen → hallway → bedroom"; loses the entity at
   every room boundary.
2. **Global identity keyed on a strong biometric fingerprint.** Rejected: a
   fingerprint strong enough to re-identify across long gaps trends toward a
   civil-identity-grade biometric — exactly what the privacy constraint
   forbids.
3. **Probabilistic persistent tracks with bounded, decaying pseudonymous
   association, built on the existing MAT tracker.** Chosen.

## Decision

Extend `wifi-densepose-mat/tracking` into a **cross-domain persistent track
layer** that produces ADR-303 `Track`/`Person` nodes.

### 1. Persistent probabilistic entity

- A persistent entity is a pseudonymous `Person` node (ADR-303) with a stable
  synthetic id (e.g. `person_7`) backed by the existing `TrackId`/UUID. It
  aggregates one or more `SurvivorTracker` tracks over time and space and holds
  a **probabilistic** continuity belief — association is never asserted as
  certain, and every hand-off carries a confidence.
- Continuity across a track-loss gap reuses the existing re-identification
  surface (`can_reidentify`, `CsiFingerprint`, `AssociationResult`), extended
  with a **time- and distance-decayed** association prior so that confidence in
  "same entity" falls with the size of the gap. Beyond a bounded horizon the
  association is dropped and a new pseudonym is minted rather than forcing a
  join — under-linking is the privacy-safe failure mode.

### 2. Cross-radio / cross-modality / cross-room continuity

- Association operates over the ADR-303 ontology graph: `Observes` edges from
  multiple `Sensor`s and `AdjacentTo`/`Doorway` topology constrain plausible
  hand-offs (a person can only move between adjacent spaces). The existing
  `mahalanobis_distance_sq` gating extends to a fused observation across
  modalities rather than a single node's detections.
- Fusion here is track-level association; the underlying multi-modality fusion
  (radar/mmWave per ADR-063, multistatic per ADR-029, and real sensor fusion
  per ADR-308) supplies the observations. This ADR depends on those for the raw
  cross-modality evidence and does not re-implement sensor fusion.

### 3. Privacy boundary (by construction)

- **No civil-identity binding.** The persistent id is a synthetic pseudonym
  with no field, edge, or join key to any name, account, phone, MAC, or other
  civil identifier. The type carries no such field, so binding is impossible in
  the schema, not merely discouraged.
- The `CsiFingerprint` used for re-identification is **bounded and decaying**:
  it is scoped to short-horizon continuity, is not persisted as a long-term
  biometric template, and expires. This keeps re-identification useful for
  "same person across the hallway" while structurally unable to serve "this is
  the same person who visited last month."
- Every `Track`/`Person`/`Event` produced carries `SemanticProvenance` and an
  `EvidenceLevel` (ADR-282), and honors the ADR-277/ADR-280 edge governance and
  ADR-141 attestation — a pseudonymous track is still governed P-class data.
  Tracking accuracy is a per-domain claim to be tagged MEASURED/CLAIMED/
  SYNTHETIC with a reproducer; **this ADR claims no accuracy number.**

## Consequences

- RuView can express persistent, cross-room trajectories for automation and
  analytics while remaining camera-free and civil-identity-free.
- The privacy-safe failure mode is **under-linking** (mint a fresh pseudonym
  when unsure), which will fragment a trajectory across long gaps or sparse
  coverage. This is a deliberate trade: a fragmented pseudonym is safe, a
  wrong civil-identity join is not.
- Extends an existing tracker rather than forking one; single-room single-radio
  deployments keep the current behavior (one entity = one track).
- Cross-modality quality depends on ADR-308/ADR-063/ADR-029 landing; until then
  continuity is WiFi-primary and its limits are stated, not hidden.
- Being phase 2, this ADR is design intent; it will be revised as the ADR-303
  ontology and ADR-302 identity spine finalize.

## Validation

- Unit tests (`cargo test -p wifi-densepose-mat`): decayed association prior
  (confidence falls with gap; drops beyond horizon → new pseudonym);
  topology-constrained hand-off (no association across non-adjacent spaces);
  schema check that a `Person`/`Track` carries no civil-identifier field.
- Integration test against a synthetic multi-room, multi-radio scenario:
  a scripted walk kitchen → hallway → bedroom yields one persistent pseudonym
  with per-hand-off confidence, and a deliberately ambiguous crossing produces
  two pseudonyms rather than a false join.
- Evidence discipline: any tracking-continuity accuracy is reported only with
  the ADR-288 leakage-free protocol and an evidence tag; no number is asserted
  here.
- Privacy review: confirm no persisted long-term biometric template and no
  civil-identity join path, as an explicit checklist item before any pilot.
