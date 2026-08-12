# ADR-305: Authenticated sensor identity — RF chain of custody

- **Status**: Accepted — initial implementation planned (ADR-300 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: security, identity, provenance, sensor-ingest, attestation, phase-1

## Context

This ADR is a child of **ADR-300** (perception substrate program) and owns
primitive #5, *authenticated sensor identity*. In the ADR-300 dependency DAG it
is a spine root that, together with **ADR-306** (canonical spatial ontology),
feeds **ADR-301** (calibration certificate) and **ADR-319** (witness chain).

RuView's inference outputs are only as trustworthy as the measurements that
produced them, yet today a measurement's origin is essentially assertional. The
UDP data plane accepts frames from any reachable host: **ADR-296** shipped step
one — a loopback-default bind (`--udp-bind`) and an optional source
IP/CIDR allowlist — and explicitly deferred to a follow-up ADR "per-device
provisioned keys, MAC/AEAD, device identifiers, monotonic sequence numbers,
freshness window, and replay rejection." **This ADR is that step two.** ADR-296
correctly documented that an IP allowlist does not stop LAN spoofing; a
cryptographic device identity is what closes that gap.

Foundations already exist in the tree and must be reused rather than rebuilt:

- `wifi-densepose-rufield` provides `DeviceId`, `Signature`, `SignatureBlock`,
  `FrameProvenance`, `ProvenanceClass`, and `SignatureVerifyError` — the type
  vocabulary for a signed frame.
- `wifi-densepose-bfld` provides `CapabilityAttestation` and
  `PrivacyAttestationProof` (BFLD attestation, ADR-141) — the device-side
  attestation surface.
- **ADR-295** defines the source-provenance state machine and freshness
  (`SpatialStateFreshness`); a monotonic sequence and freshness window slot
  into that machine rather than duplicating it.

The gap is not new primitives but an **end-to-end chain of custody**: a frame
must be traceable as `device → signed measurement → sequence → timestamp →
calibration → inference → signed event`, with every link verified at the
ingest boundary per CLAUDE.md ("validate untrusted input at every network,
hardware, and FFI boundary; default to least authority").

## Options considered

1. **Stop at ADR-296 (bind + IP allowlist).** Rejected: ADR-296 itself names
   this insufficient on a trusted LAN; any on-subnet host can still spoof a
   device.
2. **TLS/DTLS transport authentication only.** Rejected: authenticates the
   *channel*, not the *measurement*. It does not survive store-and-forward,
   does not bind a sequence number into the signed object, and gives the
   downstream evidence/witness layers nothing to re-verify offline.
3. **Per-device signing keys with a signed measurement envelope, monotonic
   sequence, and freshness window, reusing the RuField/BFLD types.** Chosen.

## Decision

Introduce an **authenticated frame envelope** carried through the sensing
server, built from existing RuField/BFLD types.

### 1. Per-device provisioned identity

- Each radio (ESP32-S3/C6 node or adapter) is provisioned with a keypair; the
  device holds the private key, the server holds the enrolled public key bound
  to a `DeviceId`. Provisioning is an explicit, authorized enrollment step — a
  device is untrusted until an operator enrolls its public key. Private keys are
  never logged or committed (CLAUDE.md credential rule); the ESP32 side follows
  `firmware/esp32-csi-node` key-handling notes.
- The enrollment record binds `DeviceId → public key → capabilities`
  (via `CapabilityAttestation`, ADR-141), so a device can only assert
  measurements for phenomena it is attested to sense. This is what **ADR-318**
  (capability certificate) later consumes.

### 2. Signed measurement envelope

- A frame on the wire becomes a `SignatureBlock` over the canonical
  serialization of `{DeviceId, sequence, timestamp, measurement-hash}`. The
  measurement itself (CSI/CIR payload) is covered by the hash so tampering is
  detectable without embedding the whole payload twice.
- Verification uses `Signature`/`SignatureVerifyError` from
  `wifi-densepose-rufield`. A frame that fails signature verification is
  dropped and counted, exactly as ADR-296 drops disallowed sources — an `Err`
  at the boundary, never a warning that proceeds.

### 3. Monotonic sequence + freshness (replay defense)

- Each device maintains a strictly monotonic per-device sequence number. The
  server tracks the last accepted sequence per `DeviceId`; a non-increasing
  sequence is rejected as a replay.
- A freshness window bounds `timestamp` against the server clock skew budget;
  stale frames are rejected. This reuses ADR-295's `SpatialStateFreshness`
  rather than inventing a parallel notion of staleness, and composes with
  ADR-297's stale-node handling.

### 4. Chain of custody into the event

- On successful verification the frame's `FrameProvenance` records the verified
  `DeviceId`, sequence, and timestamp. Calibration (ADR-301) and inference
  annotate their transforms, and the emitted spatial event (ADR-306 ontology)
  carries a signed provenance lineage. `ProvenanceClass` still enforces the
  synthetic/measured invariant from ADR-282/ADR-279 (invariant 6): a measured
  chain of custody can never be aliased to synthetic and vice-versa.
- This end-to-end signed lineage is the substrate the **ADR-319** witness chain
  serializes and the **ADR-318** capability certificate points at as evidence.

### Compatibility

- The envelope is **opt-in per deployment** and negotiated at enrollment. An
  un-enrolled single-node desktop deployment keeps working unauthenticated
  behind ADR-296's loopback default; a routable, multi-node, or fleet
  deployment (ADR-316) requires enrolled identities. The startup security log
  (ADR-296) is extended to state whether frame authentication is active.

## Consequences

- LAN spoofing and replay — the residual risks ADR-296 named plainly — are
  closed for enrolled deployments. The measurement, not merely the channel, is
  authenticated, so the guarantee survives store-and-forward into the witness
  chain.
- Enrollment/key-management is now an operational responsibility (provisioning,
  rotation, revocation). This is documented as a deployment step; key rotation
  and revocation lists are specified here but their fleet distribution is
  owned by ADR-316.
- Signature verification adds per-frame CPU cost at ingest; bounded and
  measured in validation below. It is a deliberate cost for a verifiable chain
  of custody.
- A schema addition to the frame contract; un-enrolled deployments are
  unaffected, and the migration accessor mirrors ADR-297's approach.
- **No spoof-resistance claim is MEASURED until validated on real silicon**
  (CLAUDE.md hardware rule): a passing unit/integration suite demonstrates the
  logic, not the fielded device path.

## Validation

- Unit tests (`cargo test -p wifi-densepose-sensing-server`,
  `-p wifi-densepose-rufield`): valid envelope accepted; bad signature
  rejected and counted; non-monotonic sequence rejected as replay; out-of-
  window timestamp rejected; un-enrolled `DeviceId` rejected; measured/synthetic
  provenance aliasing rejected (ADR-279 invariant 6).
- Integration test: a captured/synthesized multi-frame stream produces a
  verifiable `device → … → signed event` lineage that ADR-319 can serialize and
  re-verify offline.
- Benchmark (`cargo bench`): per-frame verification cost, to bound ingest
  overhead.
- **Real-silicon evidence required** before any deployment-grade
  authentication claim: a captured boot/runtime log from an enrolled ESP32 node
  signing frames end-to-end. A successful build or simulator run is not
  hardware evidence.
