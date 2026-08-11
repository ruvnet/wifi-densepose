# ADR-319: Witness chain — epistemic infrastructure for physical AI

- **Status**: Accepted — initial implementation planned (ADR-300 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: provenance, witness, evidence, signature, epistemics, ontology, substrate

## Context

This ADR is primitive 19 of the perception-substrate program (ADR-300) and a
spine root of its phase-1 certificate stack. In the ADR-300 dependency DAG it
**extends the source-provenance state machine (ADR-295)** and the RuField
provenance types, **ties to the signature machinery (ADR-305 /
ADR-260/262/277/279)**, and anchors the artifacts produced by ADR-301
(calibration certificates), ADR-304 (evidence records), ADR-317 (benchmark
scorecards), and ADR-318 (capability certificates). In phase 2 it carries the
independent-corroboration link from ADR-303.

The strategic assessment (ADR-300) framed RuView's real product as **epistemic
infrastructure for physical AI**: the value is not the claim "a person is
present" but the *auditable reasoning* behind it. A bare boolean output discards
everything a downstream system needs to trust or contest it — which radio
observed it, what DSP evidence supported it, which model inferred it, whether an
independent sensor agreed, what spatial state it updated, and what policy acted
on it. Once the answer is a boolean, "why do you believe that?" has no answer.

RuView already has the pieces of a chain but not the chain itself. ADR-295
defines a canonical `SourceState` (`Synthetic` / `LiveVerified` /
`LiveUnverified` / `Stale` / `Disconnected`) with `Unknown` structurally
forbidden from collapsing to live. ADR-305 defines the signed
`device → measurement → sequence → timestamp → … → signed event` chain of
custody. RuField carries `FrameProvenance`, `SemanticProvenance`, and signature
types; the AetherArena witness ledger (ADR-149) demonstrates an append-only,
witness-anchored ledger. What is missing is a single **staged, signed envelope**
that travels the whole pipeline and records, at each stage, the confidence and
provenance of that stage.

## Options considered

1. **Keep provenance as scattered per-stage fields (status quo).** Rejected:
   `FrameProvenance`, `SourceState`, calibration state, and model uncertainty
   live in different structures and are re-encoded per surface; there is no
   single object a consumer can re-verify offline to answer "why."
2. **Log a free-form audit trail alongside the output.** Rejected: mutable,
   unsigned, and not structurally tied to the output — the classic
   dashboard-that-overwrites-yesterday failure the evidence engine (ADR-304)
   already rejects.
3. **A staged, signed witness envelope carried through the pipeline, each stage
   appended and signed, anchored in an append-only ledger.** Chosen.

## Decision

Define the **witness chain**: a staged, append-only, signed envelope that
accompanies an observation from radio to policy decision. Instead of emitting
"person present," RuView emits a chain whose stages are:

```
RF observation ▸ DSP evidence ▸ model inference ▸ independent corroboration
              ▸ spatial state ▸ policy decision
```

### 1. The staged envelope

- Each stage is a signed record carrying its **confidence** and its
  **provenance**:
  - **RF observation** — the ADR-305 authenticated frame envelope
    (`DeviceId`, sequence, timestamp, measurement hash) and its ADR-295
    `SourceState`. This is the root link; a `Synthetic` root can never present
    as a `LiveVerified` one (ADR-295 invariant).
  - **DSP evidence** — the deterministic signal features and the ADR-137
    quality signals that support (or fail to support) an inference.
  - **model inference** — the model version, its raw output, and its predictive
    uncertainty; the ADR-302 `DomainState` (KNOWN / DEGRADED / UNKNOWN) gate
    result, so a low-confidence or out-of-distribution inference is recorded as
    such, not silently promoted.
  - **independent corroboration** — the phase-2 ADR-303 agreement link
    (a reference/second modality that agreed or disagreed); absent in phase 1,
    the stage records "no corroboration," never a fabricated one.
  - **spatial state** — the ADR-306 ontology `Observation`/`Track`/`Event` the
    inference updated, carrying `SemanticProvenance` and its `EvidenceLevel`.
  - **policy decision** — the governed action taken (or withheld), with the
    certificate (ADR-318) it relied on.
- Each stage carries exactly one `EvidenceLevel` (L0–L5, ADR-282); the envelope's
  effective level is the **minimum** across its stages — a synthetic root or an
  unreferenced inference caps the whole chain, so the chain cannot claim more
  than its weakest link.

### 2. Signing and anchoring

- Each stage is signed with RuField signature types (ADR-305 /
  ADR-260/262/277/279) over the canonical serialization of that stage plus the
  hash of the prior stage, so the chain is tamper-evident end to end and any
  broken link is detectable. The completed chain is anchored in an append-only,
  witness-anchored ledger following the AetherArena pattern (ADR-149); it is the
  same anchoring ADR-301/ADR-304/ADR-317/ADR-318 write into.
- The chain is **append-only**: a correction is a new chain referencing the
  prior one, never an in-place edit (mirroring ADR-304 and CLAUDE.md's "source
  over summaries").

### 3. Offline re-verification

- A consumer with the enrolled public keys (ADR-305) can re-verify a chain
  offline: check each stage signature, check each prior-stage hash, and read the
  per-stage confidence and evidence level — answering "why do you believe this?"
  without trusting the emitting host. This is the property store-and-forward
  channel authentication (rejected in ADR-305) cannot provide.

### Provenance and honesty discipline

- The witness chain never manufactures confidence: a stage that lacks evidence
  records the absence. A `Synthetic` root, a missing corroboration, or an
  UNKNOWN gate is carried faithfully and caps the chain's evidence level. No
  accuracy number is invented here; the chain records the numbers the other
  primitives produce at their honest level.

## Consequences

- Every RuView output becomes contestable and auditable: a downstream physical-AI
  system can inspect the reasoning, weight it by per-stage confidence, and reject
  a chain whose weakest link is too weak — the defining property of epistemic
  infrastructure the strategic assessment asked for.
- The certificate spine (ADR-301/301/314/315) gains a single anchoring substrate;
  each of those artifacts is a specialization of a witness record rather than a
  bespoke signed blob.
- Carrying and signing a staged envelope adds per-observation size and CPU cost;
  bounded by reusing RuField signatures and the existing ledger, and by the
  minimum-level rule keeping the object honest rather than exhaustive.
- The chain will frequently reveal weak links (synthetic root, no corroboration,
  DEGRADED gate). Surfacing that is the point; the envelope must never smooth a
  weak stage into a confident summary.

## Validation

- `cargo test` on the witness-chain crate — stage-by-stage signature round-trip
  and tamper rejection (a mutated stage or a broken prior-stage hash fails
  verification); effective evidence level equals the minimum across stages; a
  `Synthetic` root caps the chain and cannot present as `LiveVerified`
  (ADR-295 invariant); an UNKNOWN gate (ADR-302) and a "no corroboration" stage
  are recorded faithfully; append-only correction produces a new chain
  referencing the prior one.
- Cross-ADR: an ADR-305 signed frame lineage serializes into a chain that
  re-verifies offline with only the enrolled public keys; ADR-301/301/314/315
  artifacts anchor into the same ledger.
- Real-deployment chains (from live ESP32 captures with ADR-303 corroboration)
  are the maturity milestone and require hardware evidence per CLAUDE.md; a
  chain rooted in synthetic input is L0 by construction.
