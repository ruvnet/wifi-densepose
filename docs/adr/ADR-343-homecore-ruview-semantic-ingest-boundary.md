# ADR-343: HomeCore RuView semantic ingest boundary

| Field | Decision |
|---|---|
| **Status** | Accepted; software contract and lifecycle tests implemented, physical Apple-controller validation pending |
| **Date** | 2026-08-24 |
| **Scope** | `homecore-server` ingestion of RuView's per-node Apple Home projection endpoints into HomeCore state and the optional HAP bridge |
| **Depends on** | ADR-115, ADR-118, ADR-125, ADR-127, ADR-272 |

## Context

RuView exposes privacy-bounded, per-node snapshots at
`/api/v1/vitals/{node_id}/latest` and
`/api/v1/semantic-events/{node_id}/latest`. HomeCore already maps ordinary
binary-sensor state changes into HAP accessories, but no server lifecycle code
consumed those endpoints. Consequently a running sensing server could not
materially update HomeCore or Apple Home.

Embedding the sensing pipeline or accepting its broad WebSocket messages would
unnecessarily enlarge HomeCore's authority. Those surfaces may contain raw CSI,
CIR, pose, identity, camera, LiDAR, or vital waveform data that is outside the
ambient home-state boundary.

## Decision

`homecore-server` provides an explicitly opt-in RuView poller. It requires one
exact node identifier and a read-only bearer token. The token is held in a
redacting type, sent only in the authorization header, and never included in
logs or entity attributes. Redirects are disabled so credentials cannot follow
an upstream redirect.

Each poll retrieves both authoritative projection endpoints under bounded body,
timeout, and frequency limits. A snapshot is accepted only when:

1. both requests return HTTP 200 and strict JSON schemas;
2. both node identifiers match the configured node;
3. privacy class is P2 or P3 and agrees across responses;
4. timestamps are fresh, plausible, and mutually aligned;
5. numeric values are finite and within configured implementation bounds;
6. occupancy evidence agrees across the two endpoints;
7. the semantic response declares the required identity-field redactions; and
8. neither response contains a prohibited raw or identifying field.

One accepted evidence unit produces only these HomeCore entities:

- anonymous occupancy;
- thresholded motion; and
- policy-derived unexpected-occupancy and unrecognized-activity threshold
  events.

They are ordinary `binary_sensor` states. The existing HomeCore state-change
subscription therefore remains the sole HAP synchronization path. HomeCore does
not create a second HAP mapper or connect directly to an Apple controller.

Any request, status, schema, privacy, validation, freshness, or agreement
failure removes all entities owned by this poller. Removal also removes their
HAP accessories through the existing listener. There is no cached "last known"
occupancy, synthetic fallback, or fabricated offline state. Retries use bounded
exponential backoff and shutdown is explicit.

## Exclusions

This integration never requests, stores, or maps raw CSI, CIR, RF tensors,
recordings, pose frames, camera or LiDAR data, vital waveforms, identity
observations, biometric rates, person identity, or RF signatures. Heart and
breathing rate fields present in the bounded vitals schema are range-validated
to reject malformed envelopes and then discarded.

The HAP server remains behind the `hap-server` Cargo feature and separate
explicit runtime configuration. Enabling RuView ingest does not enable HAP.

## Consequences and validation

The bridge now has a source-backed RuView-to-HomeCore lifecycle, with tests for
authorization, allowed projections, prohibited fields, staleness, upstream
failure, entity removal, bounded configuration, and secret redaction. Software
tests do not establish physical Apple Home compatibility. Final release
evidence still requires a real RuView node, a HomeCore host on the same LAN, a
current iPhone Home app, and an Apple Home hub/controller, with retained
discovery, pairing, update, offline, and recovery logs.
