# ADR-313: Fleet control plane — provisioning to audit trails

- **Status**: Proposed (ADR-297 phase 2)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: fleet, operations, provisioning, firmware, updates, audit, identity, phase-2

## Context

This ADR is a child of **ADR-297** (perception substrate program) and owns
primitive #16, *fleet control plane*. In the ADR-297 DAG it is a phase-2
integration-and-operations primitive that sits on the phase-1 spine: it
**consumes ADR-302** (authenticated sensor identity) for per-device identity and
enrollment, and **ADR-315** (capability certificate) for the signed models,
calibration validity, and capability envelopes a device is allowed to run. It is
authored as Proposed and is not implemented by the phase-1 swarm.

The external and internal reviews both named the same operational gap: RuView
has strong per-device primitives but no **release identity** and no **bill of
materials** binding a fielded sensor to the exact firmware, model, and
calibration it is running — and no plane to manage that across many devices.
Without this, a handful of nodes is fine but *hundreds* of sensors become an
operational nightmare: no coherent way to provision, roll certificates, verify
firmware compatibility, distribute signed models, track calibration lifecycle,
watch health, stage updates, roll back, diagnose remotely, enforce data
retention, or produce an audit trail. This ADR addresses that release-identity /
BOM gap directly.

The scope is deliberately the **control plane**, not the data plane. The
authenticated measurement path is **ADR-293** (bind + allowlist) plus **ADR-302**
(signed envelope); this ADR governs the *devices and artifacts*, not the
per-frame stream.

Relevant existing assets to build on rather than duplicate:

- **ADR-302** already defines per-device keypairs, the `DeviceId → public key →
  capabilities` enrollment record, key rotation and revocation *semantics* — and
  explicitly deferred their **fleet distribution** to this ADR. The control
  plane is the distribution and lifecycle layer over ADR-302 identity, not a new
  identity scheme.
- **ADR-315** (capability certificate) defines the signed, expiring artifact a
  device is authorized to run; the fleet plane is what *distributes, stages, and
  revokes* those certificates and the signed models they point at.
- **ADR-298** (calibration) owns calibration validity/expiry; the fleet plane
  tracks calibration *lifecycle* across the fleet (which nodes are due, which are
  stale) rather than redefining calibration.
- **ADR-316** (witness chain) provides the append-only, re-verifiable record;
  fleet audit trails are witness-chain entries, not a parallel log format.
- **ADR-317** (RuView sensor HAL, phase 2) provides hardware/firmware capability
  descriptors used for firmware-compatibility checks before staging an update.
- `wifi-densepose-bfld` `CapabilityAttestation` (ADR-141) is the device-side
  attestation the plane checks against declared cohort capabilities.

## Options considered

1. **Manual per-device operations (SSH/flash by hand).** Rejected: does not
   scale past a handful of nodes, produces no release identity, no audit trail,
   and no safe rollback — exactly the operational nightmare the reviews named.
2. **Adopt a generic third-party IoT device-management platform wholesale.**
   Rejected as the core: generic platforms do not understand RuView's signed
   capability certificate, calibration validity, or witness chain, and would
   fork trust away from the phase-1 spine. A generic transport/agent *may* be a
   backend, but identity, certificates, and audit remain RuView's.
3. **A RuView-native control plane layered on ADR-302 identity, ADR-315
   certificates, ADR-298 calibration lifecycle, and ADR-316 audit — covering
   provisioning through rollback and retention.** Chosen.

## Decision

Define a **fleet control plane** that manages RuView sensors and their signed
artifacts across their lifecycle, built on the phase-1 identity/certificate
spine.

### 1. Release identity and bill of materials

- Each fielded device has a **BOM record** binding `DeviceId` (ADR-302) → exact
  firmware version → signed model set → active capability certificate (ADR-315)
  → current calibration record (ADR-298) → HAL/hardware descriptor (ADR-317).
  This *is* the release identity the reviews found missing: given a device you
  can state precisely what it is running and prove it is signed.

### 2. Provisioning, certificates, firmware compatibility

- **Provisioning** is the authorized ADR-302 enrollment step at fleet scale:
  minting a keypair, registering the public key and capabilities, and issuing
  the initial ADR-315 certificate. A device is untrusted until provisioned.
- **Certificate lifecycle**: issue, rotate, expire, and **revoke** ADR-315
  certificates and the ADR-302 keys behind them; revocation lists are
  distributed here (the distribution ADR-302 deferred).
- **Firmware compatibility**: before staging a firmware or model, check the
  target's ADR-317 HAL descriptor and ADR-141 capability attestation so an
  incompatible or under-capable device is never sent an artifact it cannot
  honestly run.

### 3. Cohorts, staged updates, rollback

- Devices group into **cohorts** (by site, hardware, capability). Updates —
  signed models and firmware — roll out **staged** (canary → cohort → fleet)
  with health gates between stages, and **roll back** to the previously recorded
  BOM on a failed health check. Only signed artifacts are ever staged.

### 4. Health telemetry, remote diagnostics, retention, audit

- **Health telemetry** and **remote diagnostics** report device liveness,
  calibration staleness (ADR-298), certificate expiry (ADR-315), and error
  state — read-only diagnostics by default, mutations authorized explicitly.
- **Data retention** policy is enforced per cohort, and P0/CSI/person data never
  leaves the edge except under the ADR-277/280 governance already in force
  (CLAUDE.md: never commit or exfiltrate CSI/person data).
- Every lifecycle action — provision, rotate, revoke, stage, roll back — is
  written as an **ADR-316 witness-chain** entry, giving a re-verifiable **audit
  trail** rather than a mutable log.

### Authority and least privilege

- The control plane is default-deny (CLAUDE.md: default to least authority).
  Provisioning, key rotation, revocation, staging, and rollback are each
  separately authorized operations; no fleet action is implied by another.
  Credentials and private keys are never logged or committed.

## Consequences

- Hundreds of sensors become operable: coherent release identity, signed-artifact
  distribution, staged updates with rollback, and a re-verifiable audit trail —
  closing the release-identity / BOM gap the reviews raised.
- The plane concentrates operational authority; that is mitigated by
  default-deny, per-action authorization, signed-only artifacts, and
  witness-chained audit. A compromised plane must still forge signatures the
  phase-1 spine verifies.
- Hard dependency on ADR-302 (identity), ADR-315 (certificate), ADR-298
  (calibration lifecycle), ADR-316 (audit), and ADR-317 (firmware/HAL
  compatibility). This ADR distributes and sequences those artifacts; it does
  not redefine identity, certificates, calibration, or the witness format.
- Being phase 2, this is design intent depending on the spine; it is expected to
  be revised as ADR-315, ADR-316, and ADR-317 land.
- **No fielded fleet-operation claim is MEASURED without real-silicon evidence**
  (CLAUDE.md hardware rule): staged update and rollback on real nodes require a
  captured runtime log. A passing simulation is not fleet evidence.

## Validation

- Unit tests: BOM records bind identity/firmware/model/certificate/calibration
  consistently and reject inconsistent bindings; certificate issue/rotate/revoke
  transitions are correct; a firmware-incompatible target is refused staging;
  every lifecycle action emits a well-formed ADR-316 witness entry.
- Integration test: a synthetic cohort undergoes a canary→cohort→fleet staged
  update; an injected health failure triggers rollback to the prior BOM; the
  full sequence is re-verifiable from the witness chain offline; a revoked
  certificate is rejected fleet-wide.
- Security test (`npm run test:security` analogue for the plane): default-deny
  is enforced; unauthorized provision/rotate/revoke/stage is rejected and
  counted; no credential or P0 data appears in telemetry or audit output.
- Field validation (deferred, real-silicon): a real multi-node staged update and
  rollback with a captured boot/runtime log, reported as `MEASURED` with a
  reproducer. Until then all fleet-operation results are simulator-level. No
  fielded reliability number is asserted by this ADR.
