# ADR-297: Multi-node semantic correctness — per-node inference, node-keyed rate limiting, stale state

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: multi-node, mqtt, home-assistant, correctness, sensing-server

## Context

The external review confirmed three defects on the multi-node path — the core
mechanism RuView uses to reduce blind spots and room dependence:

1. The active `NodeInfo` payload carries RSSI/position/subcarrier/sync but **no
   per-node classification**; the MQTT mapper reads `node.classification` and
   falls back to the room aggregate when absent, so every node can publish the
   same aggregate presence value (issues 1540, 1554).
2. The MQTT `RateLimiter` is keyed by `EntityKind` only
   (`mqtt/state.rs:65`), so one node consumes the numeric publish slot and the
   others are suppressed until the interval expires, while availability still
   says online (issue 1541).
3. In the UDP vital path, top-level classification is taken from the
   latest-arriving node while other features are fused, so with disagreeing
   nodes room presence can flip at packet frequency (issue 1555).

## Decision

- **Separate the types.** Introduce `NodeInference` (per-node classification +
  confidence + freshness) distinct from `RoomInference` (the fused room
  aggregate). `NodeInfo` carries a `NodeInference`; the room aggregate is
  computed explicitly and never overwrites node state. No silent fallback from
  node to room.
- **Key the rate limiter by (node, entity).** `RateLimiter` becomes keyed on
  `(NodeId, EntityKind)` so nodes no longer starve each other; per-entity
  behavior per node is preserved.
- **Deterministic fusion.** Room classification is a pure function of the set
  of current per-node inferences (e.g. freshness-weighted vote), not
  last-writer-wins; identical inputs yield identical room state.
- **Stale entities cannot stay online.** An entity whose backing node has not
  reported within N expected publish intervals transitions to unavailable/
  stale rather than holding a frozen value while availability says online.

## Consequences

- Multi-node HA/MQTT output becomes semantically correct; distinct nodes
  report distinct state and no longer suppress one another.
- Schema change to `NodeInfo`/the MQTT contract; existing single-node
  deployments keep working (one node = one inference). Consumers reading the
  old aggregate-only shape need the migration accessor.
- Aligns with ADR-295 (freshness) and the review's call for one canonical
  `NodeInference`/`RoomInference` contract.

## Validation

- Unit/integration tests: per-node classification round-trips through the MQTT
  mapper with no room fallback; two nodes with different rates both publish
  (no starvation); disagreeing nodes produce deterministic, non-flapping room
  state; a silent node's entities go stale, not frozen-online.
- `cargo test -p wifi-densepose-sensing-server`.
