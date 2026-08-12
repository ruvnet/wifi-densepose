# ADR-318: Capability certificates — validated-for-this-environment claims

- **Status**: Accepted — initial implementation planned (ADR-300 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: capability, certificate, evidence, provenance, signature, honesty, substrate

## Context

This ADR is primitive 18 of the perception-substrate program (ADR-300) and,
per the strategic assessment, among the strongest ideas in the program: it is
where the whole certificate spine becomes a consumable contract. In the ADR-300
dependency DAG it **consumes the evidence engine (ADR-304)** — a capability
certificate is a signed attestation minted over a slice of that ledger — the
**calibration certificate (ADR-301)** for the environment it is validated
against, and the **RuField signature types (ADR-305 / ADR-260/262/277/279)** to
sign it. It reports domain state via ADR-302 and is anchored in the witness
chain (ADR-319).

RuView must stop making unconditional capability claims. "Supports presence" is
not a true statement — presence detection works in some rooms, on some hardware,
for some subject dynamics, and fails on a stationary subject at range in an
uncalibrated room. A capability is only ever *validated for a specific
environment*, and the honest unit of that claim is a signed, expiring
certificate, not a feature flag in a README.

The ingredients now exist across the phase-1 spine: ADR-304 accumulates
per-`(room, device, subject)` accuracy, false-positive rate, drift, and domain
state; ADR-301 produces the signed room fingerprint the environment is keyed to;
ADR-305 provides the authenticated device identity and `CapabilityAttestation`
(BFLD, ADR-141) that bounds *what a device is even attested to sense*; ADR-282
provides the mandatory `EvidenceLevel`. What is missing is the artifact that
binds them into a single, verifiable "validated here, until then" claim and the
consumer-side rule that refuses capabilities lacking one.

## Options considered

1. **Static capability flags / a `supports_presence` boolean.** Rejected: it is
   the exact dishonest claim — environment-independent, unsigned, non-expiring,
   and false the moment the room, device, or subject dynamics differ.
2. **Report raw ledger accuracy to consumers directly.** Rejected: the ledger
   (ADR-304) is the source of truth but not a portable, signed, bounded contract;
   handing consumers raw records pushes evidence-weighting and expiry logic into
   every consumer and drops the single verifiable object.
3. **Mint a signed, expiring `CapabilityCertificate` over an ADR-304 ledger
   slice, and make consumers refuse capabilities without a valid one.** Chosen.

## Decision

Introduce a signed **`CapabilityCertificate`**: a bounded attestation that a
specific capability has been validated for a specific environment, for a bounded
time.

### 1. The certificate

A serializable `CapabilityCertificate` binding:

- `capability` — the phenomenon (e.g. `presence`, `pose`), which must be within
  the device's ADR-305/ADR-141 `CapabilityAttestation` (a device cannot be
  certified for something it is not even attested to sense).
- `room` — the ADR-306 space identifier, tied to the ADR-301 calibration
  certificate version the validation was performed against.
- `hardware` — the ADR-305 authenticated `DeviceId` (and, in phase 2, the
  ADR-320 HAL descriptor of the sensor).
- `model` — the model version scored.
- `calibrated_date` — the calibration certificate age at validation time.
- `moving_recall`, `stationary_recall`, `false_presence_per_24h` — the measured
  operating metrics, sliced from the ADR-304 ledger for this exact context (not
  a global average), each honestly labelled. These are per-capability; a pose
  certificate carries pose metrics with the mean-pose baseline and a
  leakage-free split (CLAUDE.md) or it is not issued.
- `valid_until` — an explicit expiry; a certificate is never open-ended.
- `evidence_level` — exactly one L0–L5 (ADR-282). A certificate minted from a
  synthetic ledger slice is L0/`Synthetic`; a MEASURED metric requires an
  ADR-303 reference and a reproducer. The certificate cannot upgrade the level
  of the ledger it is minted from (ADR-304 honesty rule).
- `signature` — a RuField `SignatureBlock` (ADR-305 / ADR-260/262/277/279) over
  the canonical serialization; an unsigned certificate is not a valid
  certificate. The certificate is anchored in the witness chain (ADR-319).

### 2. Minting

- A certificate is minted from a slice of the ADR-304 evidence ledger for one
  `(room, device, subject-class, model)` context. If the ledger reports "no
  evidence" for that context, **no certificate is issued** — absence of evidence
  is never a capability. Minting is a pure function over the append-only ledger
  at mint time; the metrics are frozen into the signed object.
- Expiry (`valid_until`) is derived from calibration validity (ADR-301) and an
  evidence-freshness policy: a certificate cannot outlive the calibration it was
  validated against, and drift beyond the ADR-301 envelope invalidates both.

### 3. Consumer refusal rule

- Applications and surfaces **refuse to consume a capability that lacks a valid
  certificate for the current environment**. "Valid" means: signature verifies,
  `room`/`hardware`/`model` match the running context, `valid_until` is in the
  future, and the referenced calibration certificate is itself still valid
  (ADR-301 not invalidated). A failed check yields UNKNOWN via ADR-302, not a
  best-effort guess.
- This makes the ADR-300 acceptance clause "quantify whether it can reliably
  sense the requested phenomenon → generate a signed capability certificate"
  a hard gate rather than a hope.

## Consequences

- RuView can no longer claim a capability it has not validated for the caller's
  environment; the honest failure — "not certified here" → UNKNOWN — is
  surfaced by construction rather than by discipline.
- OEM/integrator diligence gets a single verifiable artifact ("presence,
  validated in *this* room, on *this* device, with *these* recall/false-alarm
  numbers, until *this* date, at *this* evidence level, signed") — the strongest
  commercial output of the spine.
- Certificates expire and get refused; some environments will have no
  certificate and therefore no capability until validated. That refusal is the
  intended honest behavior, not a regression.
- Key management and expiry policy are operational responsibilities, reusing the
  ADR-305 enrollment/rotation and ADR-301 validity machinery rather than new
  infrastructure; fleet distribution of certificates is owned by ADR-316.
- No capability number is invented here; every metric on a certificate is sliced
  from the ADR-304 ledger at its honest evidence level.

## Validation

- `cargo test` on the certificate crate — mint from a ledger slice produces the
  frozen metrics; "no evidence" context yields no certificate; signature
  round-trip and tamper rejection; `valid_until` and calibration-linked expiry
  enforced; consumer refusal on room/hardware/model mismatch, expiry, or
  invalidated calibration resolves to UNKNOWN (ADR-302), not a guess; evidence
  level is inherited from the ledger and cannot be upgraded; a certificate
  cannot be issued for a capability outside the device's ADR-305/ADR-141
  attestation.
- Cross-ADR: an ADR-304 ledger fixture mints a certificate; an ADR-302 test
  asserts an expired/mismatched certificate gates to UNKNOWN; the ADR-300
  acceptance test consumes a minted certificate end-to-end.
- Real-deployment certificates (minted from a populated ledger with ADR-303
  references on live ESP32 captures) are the maturity milestone and require
  hardware evidence per CLAUDE.md; a certificate minted from a synthetic ledger
  is L0 by construction.
