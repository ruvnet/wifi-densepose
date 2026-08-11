# ADR-303: Canonical spatial ontology — one Site→…→Event model for every surface

- **Status**: Accepted — initial implementation planned (ADR-297 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: ontology, worldgraph, schema, mqtt, matter, rufield, phase-1

## Context

This ADR is a child of **ADR-297** and owns primitive #6, *canonical spatial
ontology*. In the ADR-297 DAG it is a spine root alongside **ADR-302**
(authenticated identity) and feeds every downstream primitive that must speak
about *where* and *what*: **ADR-298** (calibration), **ADR-304** (tracking,
consumes `Track`/`Person`), **ADR-316** (witness chain), and every external
surface named in the ADR-297 consequences (MQTT, REST, WebSocket, RuField,
Matter, agents).

RuView currently expresses "where something is" in several overlapping,
per-surface schemas: the MQTT/Home-Assistant mapper has its own node/room
shapes (**ADR-294** just introduced `NodeInference`/`RoomInference` to
disambiguate node vs. room state); the `worldgraph` crate models a spatial
graph; RuField carries `SemanticProvenance`; Matter/HomeKit has its own area
model. The same physical fact — "a person is in the kitchen" — is re-encoded
differently on each surface, and the review called for "one canonical
`NodeInference`/`RoomInference` contract" (ADR-294 consequences). Without a
single semantic model, every new surface multiplies the translation matrix and
each translation is a place where provenance and evidence level (ADR-282) can
be silently dropped.

Substantial scaffolding already exists and must be **reused/extended, not
rebuilt**. `v2/crates/worldgraph/wifi-densepose-worldgraph` already defines:

- `WorldNode` variants including `Room { area_id, name, bounds_enu, floor }`,
  `Zone { parent_room, … }`, `Wall { rf_attenuation_db }`, and `Doorway`.
- `WorldEdge` variants including `Observes { quality, last_seen_unix_ms }`,
  `LocatedIn { since_unix_ms }`, `AdjacentTo { via_doorway }`, and `Supports`.
- `WorldGraph`, `WorldGraphSnapshot`, `WorldId`, `SemanticProvenance`,
  `PersonPosition`, and a HomeCore `area_id` linkage join key (ADR-127).

The `worldgraph` crate is therefore the natural home for the canonical model.
What is missing is (a) the full `Site → Building → Floor → Space → Zone`
containment spine above `Room`, (b) first-class `Sensor`, `Object`,
`Observation`, `Track`, and `Event` node types, (c) one canonical serialization
that every surface consumes, and (d) a documented migration path from the
existing per-surface schemas.

## Options considered

1. **Leave each surface with its own schema; add adapters pairwise.** Rejected:
   O(surfaces²) translations, and provenance/evidence loss at each hop.
2. **Invent a new top-level ontology crate.** Rejected: `worldgraph` already
   models rooms, zones, walls, doorways, observation edges, and HomeCore
   linkage; a parallel crate would fork the world model.
3. **Extend `worldgraph` into the canonical ontology and make every surface a
   projection of it.** Chosen.

## Decision

Adopt **one canonical spatial ontology**, hosted in the `worldgraph` crate,
that every RuView surface reads from and writes to.

### 1. The containment spine and entity types

Define the full node taxonomy as an extension of the existing `WorldNode`:

```
Site ▸ Building ▸ Floor ▸ Space ▸ Zone
                                   └─▸ { Sensor, Person, Object,
                                         Observation, Track, Event }
```

- `Site`, `Building`, `Floor`, `Space` are new containment `WorldNode`
  variants above the existing `Room` (mapped to `Space`, keeping its `area_id`
  and `bounds_enu`) and `Zone`. `Wall`/`Doorway` remain as topological
  elements. Containment reuses the existing `LocatedIn`/`AdjacentTo` edge
  vocabulary; a new `PartOf` edge expresses the pure hierarchy
  (Zone `PartOf` Space `PartOf` Floor …).
- `Sensor` is the entity **ADR-302** authenticates (`DeviceId` as its stable
  identity) and **ADR-317** (HAL, phase 2) describes the hardware of. `Person`,
  `Object`, `Observation`, `Track`, and `Event` are first-class nodes.
  `Observes`/`LocatedIn` edges already carry quality and dwell timestamps.
- `Track` and `Person` are defined **here** as the ontology contract that
  **ADR-304** (persistent tracking) produces and updates. `Observation` is what
  an authenticated frame (ADR-302) becomes after calibration (ADR-298), and
  `Event` is the governed output that ADR-315 certifies and ADR-316 witnesses.

### 2. Canonical serialization

- A single, versioned serialization (serde-based, stable field names) is the
  one wire/at-rest representation. Every surface — MQTT/Home-Assistant, REST,
  WebSocket, RuField observations, Matter/HomeKit, agent queries — is a
  **projection** of this model, not an independent schema. `NodeInference` and
  `RoomInference` (ADR-294) become projections of `Sensor→Observes` and the
  `Space`-level fused inference respectively, so ADR-294's node/room separation
  is preserved by construction rather than re-encoded per surface.
- Every node and edge carries `SemanticProvenance` and exactly one
  `EvidenceLevel` (L0–L5, ADR-282 policy): the evidence ladder travels *with*
  the fact across every projection, so no surface can silently upgrade or drop
  it.

### 3. Migration path

- Each existing per-surface schema gets a documented, tested bidirectional
  mapping to/from the canonical model, plus a migration accessor for consumers
  reading the old shape (mirroring ADR-294's migration accessor). Surfaces are
  cut over one at a time; a surface is "canonical" once its projection is the
  only encoder it uses. Until cutover, the mapping layer is authoritative and
  round-trip-tested so no fact is lost in translation.
- The `worldgraph` HomeCore `area_id` linkage (ADR-127) remains the join key
  between the ontology's `Space` and external area registries.

## Consequences

- The translation matrix collapses from O(surfaces²) to O(surfaces): each
  surface implements one projection. New surfaces (ROS 2, OpenUSD, OPC UA per
  ADR-282's roadmap) plug in as additional projections.
- Provenance and evidence level are carried uniformly; a fact cannot cross a
  surface boundary and lose its lineage or its L-level.
- A schema change reaching every surface; managed by the versioned
  serialization and per-surface migration accessors. Single-node deployments
  keep working (one `Sensor`, one `Space`).
- The ontology is a *representation*, not an inference engine: it says nothing
  about *how* a `Track` or `Event` is produced — that is owned by ADR-304,
  ADR-298, ADR-299, and the model layer. This ADR does not itself make any
  accuracy claim to grade.
- Extending `worldgraph` grows one crate's surface rather than forking a second
  world model; the geo/worldmodel sub-crates continue to build on the same node
  vocabulary.

## Validation

- Unit tests (`cargo test -p wifi-densepose-worldgraph`): containment-spine
  construction and invariants (a `Zone` is `PartOf` exactly one `Space`, a
  `Space` on exactly one `Floor`, etc.); round-trip serialization of every node
  and edge type; every node/edge carries exactly one `EvidenceLevel`.
- Migration tests: each per-surface schema maps to the canonical model and back
  with no loss of provenance or evidence level; `NodeInference`/`RoomInference`
  (ADR-294) project and re-project identically.
- Contract test: a single canonical `Event` renders correctly through the MQTT,
  REST, and WebSocket projections from one source of truth.
- No accuracy numbers are claimed; this ADR delivers the shared representation
  the rest of the phase-1 spine writes into.
