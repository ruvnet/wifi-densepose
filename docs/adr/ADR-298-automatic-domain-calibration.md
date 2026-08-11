# ADR-298: Automatic domain calibration — signed, versioned, invalidatable room fingerprint

- **Status**: Accepted — initial implementation planned (ADR-297 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: calibration, provenance, drift, evidence, honesty, substrate

## Context

This ADR is primitive 1 of the perception-substrate program (ADR-297) and the
first brick of that program's "certificate spine" (ADR-297 phase 1). It depends
on the canonical spatial ontology (ADR-303) to name *which space* it
characterizes, on authenticated sensor identity (ADR-302) to bind a fingerprint
to *which signed device* produced it, and on the witness chain (ADR-316) to
anchor the resulting artifact. Its output is consumed directly by
out-of-distribution detection (ADR-299).

WiFi sensing is only reproducible inside the environment it was tuned for.
Multipath, furniture geometry, transceiver placement, and AP channel all shape
the CSI distribution, so a model that reads a room correctly one week can drift
silently the next. RuView already has the raw ingredients for room-aware
sensing but not a single portable, signed, expiring artifact that says "this is
the room, here is when it was measured, and here is the evidence that it is
still the same room."

Existing scaffolding to build on, not rebuild (`v2/crates/wifi-densepose-calibration`):

- `enrollment` / `anchor` — guided human anchors with an adaptive quality gate.
- `bank` / `specialist` / `runtime` — a versioned bank of small specialist
  models and a confidence-gated mixture runtime (`RoomState`), including the
  crate's existing honest `STALE` degradation when the ADR-135 empty-room
  baseline drifts.
- `geometry` / `geometry_embedding` — transceiver-geometry record and its
  fixed-length conditioning featurization (ADR-152).

What is missing is (a) an *automatic* observe-only characterization phase that
does not require a human enrollment ritual, (b) empty-vs-occupied baseline
separation as a first-class pair, (c) a signed, versioned, comparable
`CalibrationCertificate` artifact, and (d) explicit invalidation on drift rather
than a soft `STALE` flag buried in the runtime.

## Options considered

1. **Keep calibration internal to the runtime (status quo).** Rejected: the
   room characterization exists only as in-process state; it cannot be signed,
   shipped, compared across time, or presented as evidence to ADR-299/ADR-315.
2. **Build a new calibration crate.** Rejected: `wifi-densepose-calibration`
   already owns enrollment, the specialist bank, geometry embedding, and the
   baseline-drift concept. A parallel crate would fork the room model.
3. **Extend `wifi-densepose-calibration` with an automatic characterization
   phase and a signed certificate artifact.** Chosen.

## Decision

Extend `v2/crates/wifi-densepose-calibration` with an `autocal` characterization
phase and a `certificate` artifact module. The target UX is:

> install → observe (~10 min) → room fingerprint → calibration certificate →
> sensing.

### 1. Automatic characterization (`autocal`)

- An observe-only pass (default ~10 minutes, configurable) that collects CSI
  without requiring guided human anchors, reusing the `anchor` quality gate to
  reject frames it cannot trust. It layers on the existing ADR-135 empty-room
  baseline rather than replacing it.
- Produces a `RoomFingerprint`: a bounded, fixed-length statistical summary of
  the room's CSI distribution (subcarrier amplitude/phase moments, multipath
  structure, occupancy-band energy), plus the `geometry_embedding` when a
  geometry record is present. The fingerprint is the distance-comparable object
  ADR-299 measures against; its schema is versioned.

### 2. Empty / occupied baseline pair

- Characterization establishes a paired baseline: an **empty** distribution
  (no occupant motion) and an **occupied** distribution (motion present),
  separated by the existing occupancy signal rather than a manual label. Both
  are stored on the fingerprint so downstream OOD gating can distinguish "the
  empty room changed" (furniture/geometry drift) from "occupancy statistics
  changed" (different subject dynamics).

### 3. `CalibrationCertificate` artifact

- A serializable `CalibrationCertificate` binding: the `RoomFingerprint`; a
  space identifier from the ADR-303 ontology; the signing sensor identity from
  ADR-302; `captured_at_unix_s`; a monotonic `version`; a schema version; the
  calibration `tier`; and an `EvidenceLevel` (L0–L5, ADR-282) — an automatic
  characterization on real captured CSI is at most L1/L2 and is labelled as
  such, never L3+.
- The certificate is **signed** using RuField provenance/signature types
  (ADR-260/262/277/279) and anchored in the witness chain (ADR-316). Signature
  and witness anchoring are mandatory: an unsigned certificate is not a valid
  certificate.
- Two certificates for the same space are **comparable**: `distance(a, b)`
  returns a bounded fingerprint distance, which is the primitive ADR-299 uses
  to gate KNOWN → DEGRADED → UNKNOWN.

### 4. Invalidation and continuous drift compensation

- A certificate carries an explicit validity policy: it is invalidated when
  fingerprint distance against live traffic exceeds a threshold, when the AP
  channel or transceiver geometry changes, when the signing device identity
  changes, or on age expiry. Invalidation is an explicit state transition that
  emits a witness record (ADR-316), not a silent `STALE` flag.
- Continuous drift compensation runs as a bounded online update of the
  fingerprint within a **compatibility envelope**: small drift is absorbed and
  logged; drift beyond the envelope invalidates the certificate and forces
  re-characterization. Compensation never silently rewrites a signed
  certificate — it produces a new version, preserving the append-only history.

### Provenance and honesty discipline

- No accuracy number is claimed by this ADR; it delivers the artifact and the
  distance/invalidation machinery. Any certificate produced from generated CSI
  is L0/`Synthetic` by construction; the constructor rejects labelling
  synthetic characterization as measured (ADR-279 invariant 6, ADR-282 ladder).
- Certificates never leave the edge except through the governed control plane
  (ADR-277); a room fingerprint is treated as potentially sensitive spatial
  data, not free telemetry.

## Consequences

- Room characterization becomes a portable, signed, versioned artifact that
  ADR-299 (OOD), ADR-315 (capability certificates), and ADR-314 (benchmark)
  can consume without re-deriving room state.
- The automatic observe-only path lowers deployment friction (no mandatory
  enrollment ritual) but yields a weaker evidence level than guided enrollment;
  the certificate states which path produced it so consumers can weight it.
- Explicit invalidation means RuView will sometimes refuse to sense a changed
  room until re-characterization. That refusal is the intended honest behavior,
  surfaced by ADR-299, not a regression.
- The existing enrollment/bank/runtime path is preserved; `autocal` is an
  additional entry point that produces the same `RoomFingerprint` object the
  guided path can also emit.

## Validation

- `cargo test -p wifi-densepose-calibration` — fingerprint determinism from
  fixed synthetic CSI; empty/occupied separation on synthetic occupancy;
  certificate signing/verification round-trip and tamper rejection;
  `distance()` monotonicity on progressively perturbed fixtures; invalidation
  transitions (channel change, geometry change, age, drift-envelope breach)
  each emit the expected witness record; constructor rejects synthetic→measured
  mislabeling.
- Cross-ADR: an ADR-299 test consumes a certificate and asserts the gating
  state transitions on a drifted fingerprint.
- Real-silicon characterization (ESP32 capture over a real 10-minute window)
  remains a follow-up requiring hardware evidence per CLAUDE.md; a successful
  build or synthetic run is not hardware evidence.
