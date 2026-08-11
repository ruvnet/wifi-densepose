# ADR-299: Out-of-distribution detection — KNOWN / DEGRADED / UNKNOWN gating

- **Status**: Accepted — initial implementation planned (ADR-297 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: ood, calibration, uncertainty, quality, evidence, honesty, substrate

## Context

This ADR is primitive 2 of the perception-substrate program (ADR-297) and part
of the phase-1 certificate spine. It sits directly downstream of automatic
domain calibration (ADR-298): the `CalibrationCertificate` and its
`RoomFingerprint` are the reference distribution this ADR measures against. It
reuses fusion-layer quality scoring (ADR-137) as one of its inputs and feeds
its state into the evidence engine (ADR-301) and capability certificates
(ADR-315).

The central unsolved problem of WiFi sensing is cross-domain generalization: a
model trained (or calibrated) in one room degrades unpredictably in another, or
in the same room after furniture moves, the AP changes channel, or the radio
hardware is swapped. A model that keeps returning confident classifications
under these conditions is the single most misleading failure mode in the field,
and it is the failure the strategic assessment (ADR-297) named explicitly.
Confidence alone is insufficient: a softmax head is perfectly capable of being
confidently wrong on out-of-distribution input. RuView must be able to say
"I do not recognize this situation" instead of guessing.

Today RuView has partial signals but no unified gate:

- ADR-298 produces a comparable `RoomFingerprint` and a `distance()` metric.
- ADR-137 `QualityScore` carries fusion coherence, evidence references, and
  contradiction flags per fused frame.
- Model heads emit confidence/uncertainty, but nothing combines domain
  distance, signal quality, calibration compatibility, and uncertainty into a
  single decision, and nothing forces a model to stop emitting confident labels
  when it leaves its calibrated domain.

## Options considered

1. **Threshold on model confidence alone.** Rejected: confidently-wrong OOD
   predictions are exactly the failure mode; confidence is necessary but not
   sufficient.
2. **A per-model bespoke OOD check inside each task head.** Rejected:
   duplicates logic, cannot be audited uniformly, and does not compose with the
   calibration certificate or the evidence engine.
3. **A shared OOD gate that every inference passes through, fusing four signals
   against the ADR-298 certificate.** Chosen.

## Decision

Add an out-of-distribution gate — implemented in a shared crate consumed by the
task-head runtime (`wifi-densepose-calibration::runtime` and the model serving
path) — that attaches a `DomainState` to **every** inference.

### 1. Four inputs, one decision

Each inference carries four measured quantities:

1. **Domain distance** — fingerprint distance (ADR-298 `distance()`) between
   live traffic and the active `CalibrationCertificate`, split into the
   empty-baseline and occupied-baseline components so geometry drift and
   occupancy-statistics drift are distinguishable.
2. **Signal quality** — reuse the ADR-137 quality scoring signals (fusion
   coherence, contradiction flags) plus per-frame SNR/validity.
3. **Calibration compatibility** — is a valid, non-invalidated certificate
   present for this space (ADR-303) and this signed device (ADR-302)? An
   expired, invalidated, or device-mismatched certificate is itself a
   compatibility failure.
4. **Uncertainty** — the model head's own predictive uncertainty.

### 2. State machine: KNOWN → DEGRADED → UNKNOWN

- **KNOWN** — domain distance within the certificate's compatibility envelope,
  quality above threshold, certificate valid and compatible, uncertainty low.
  Confident classifications are returned.
- **DEGRADED** — one or more signals crossed a soft threshold (e.g. moderate
  fingerprint drift within the envelope, elevated uncertainty, a tolerated
  ADR-137 contradiction flag). Classifications are returned but flagged
  degraded with the specific reason; downstream consumers must treat them as
  lower-evidence.
- **UNKNOWN** — the room changed materially (empty-baseline drift beyond the
  envelope, AP channel change, transceiver-geometry change, hardware/device
  change, or an invalidated/absent certificate). RuView **stops returning
  confident classifications** and returns UNKNOWN with the triggering cause.
  This is the required behavior, not an error.

State transitions are hysteretic (separate enter/exit thresholds) so the gate
does not flap on noise. The state, the four input values, and the triggering
cause are all reported — never a bare label.

### 3. Certificate-bound, honest by construction

- The gate is meaningless without a certificate: with no valid ADR-298
  certificate for the current space/device, the default state is UNKNOWN, not
  KNOWN. Absence of evidence is treated as absence of capability.
- The `DomainState` and its inputs are emitted to the evidence engine
  (ADR-301) as part of every inference record, and are an input to the ADR-315
  capability certificate (a model's capability is bounded by the domain it can
  hold KNOWN in).
- No accuracy number is claimed here; the ADR delivers the gating machinery.
  The gate's own thresholds are calibration parameters, reported with each
  decision.

## Consequences

- RuView gains a uniform, auditable answer to "should I trust this inference?"
  that combines domain, quality, calibration, and uncertainty rather than
  confidence alone.
- Deployments will see more DEGRADED/UNKNOWN results than a
  confidence-only system, especially right after a room changes. That increase
  is the product working: it is the difference between honest RF perception and
  confidently-wrong output.
- Every task head that opts into the substrate must route through the gate;
  heads that bypass it cannot claim a KNOWN state or earn an ADR-315
  certificate.
- The gate couples model serving to the presence of a live calibration
  certificate, making ADR-298 a hard dependency of confident inference — the
  intended coupling.

## Validation

- `cargo test` on the OOD crate — state-machine transitions on synthetic
  fixtures: in-envelope drift stays KNOWN; soft-threshold breach → DEGRADED;
  empty-baseline drift beyond envelope, channel change, geometry change,
  device mismatch, and invalidated/absent certificate each → UNKNOWN;
  hysteresis prevents flapping under injected noise; missing certificate
  defaults to UNKNOWN.
- Cross-ADR: consumes an ADR-298 certificate and asserts a drifted fingerprint
  drives the expected transition; asserts the `DomainState` is present on every
  emitted inference record consumed by ADR-301.
- No confident classification is emitted in the UNKNOWN state in any test —
  enforced as an assertion, not a convention.
- Real-silicon OOD behavior (moving furniture / changing AP channel on a live
  ESP32 capture and observing the transition) remains a follow-up requiring
  hardware evidence per CLAUDE.md.
