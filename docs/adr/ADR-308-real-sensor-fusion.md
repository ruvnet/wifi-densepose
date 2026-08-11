# ADR-308: Real sensor fusion — uncertainty-aware, multiple observations → one world state

- **Status**: Accepted — initial implementation (ADR-297 phase 2)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: fusion, uncertainty, multimodal, world-state, ontology, phase-2

## Context

This ADR is a child of **ADR-297** and owns primitive #11, *real sensor fusion*.
In the ADR-297 DAG it is a phase-2 integration primitive: it **consumes ADR-303**
(canonical spatial ontology) and **produces the single fused world state** that
the phase-3 primitives build on — **ADR-309** (long-term spatial memory),
**ADR-310** (counterfactual inference), and **ADR-312** (digital RF twin). It is
authored as **Proposed**.

The defining invariant is not "support more modalities" but the *shape of the
output*: **multiple observations must resolve to one probabilistic world state,
not many feeds into a visualization.** A dashboard that shows a WiFi layer, a
mmWave layer, and a BLE layer side by side is not fusion; it pushes the
reconciliation onto the human. Real fusion produces one uncertainty-aware state
that every downstream consumer reads, with each contributing observation's
provenance and confidence still recoverable.

Substantial scaffolding already exists and must be **reused/extended, not
rebuilt**:

- **ADR-063** (60 GHz mmWave ↔ WiFi CSI fusion, *Proposed*) established the
  first cross-modal fusion case: pairing noisy CSI-derived vitals with clinical-
  grade mmWave FMCW radar (Seeed MR60BHA2 over UART, with a **live hardware
  capture** logged on 2026-03-15). ADR-308 generalizes that pairwise case into
  an N-modality, uncertainty-aware fusion.
- **ADR-137** (fusion-engine quality scoring, *Accepted — partial*) already
  built the auditable-quality building block: it identified that the multistatic
  fusers (`wifi-densepose-signal/src/ruvsense/multistatic.rs`,
  `wifi-densepose-ruvector/src/viewpoint/fusion.rs`) discarded the evidence they
  used, and specified a single auditable record — "this fused output is
  trustworthy because X, Y, Z, but be aware of contradiction C" — with evidence
  references and contradiction flags. ADR-308 reuses that record as the
  provenance/quality carrier of the fused state.
- **ADR-280** `CoherentSensorGroup` (fail-closed coherent fusion) and
  **ADR-303** `Observation`/`Track`/`Event` node types are the input and output
  vocabulary respectively.

What is missing is the **uncertainty-aware combiner across heterogeneous
modalities**: a fusion stage that takes authenticated observations from WiFi,
BLE, UWB, mmWave, acoustic, IMU, lidar, and cameras (only where policy permits),
each with its own uncertainty, and emits one probabilistic `WorldState` — with
per-observation contradiction flags, not a stack of independent feeds.

## Options considered

1. **Per-modality feeds rendered together** (today's implicit model on some
   surfaces). Rejected: it is visualization, not fusion; contradictions are
   never reconciled and there is no single state to reason over.
2. **Hard-switch "best modality wins"** (e.g., always prefer mmWave vitals over
   CSI vitals). Rejected: throws away corroborating evidence and cannot express
   *disagreement* — the very thing ADR-137's contradiction flags exist to
   surface — and degrades badly when the preferred modality is absent or OOD.
3. **Uncertainty-weighted probabilistic fusion into one world state**, reusing
   ADR-137's auditable quality record and ADR-280's fail-closed coherence gate.
   Chosen.

## Decision

Adopt **uncertainty-aware multimodal fusion** whose invariant output is one
probabilistic world state.

### 1. Inputs: authenticated, ontology-typed observations

- Inputs are ADR-303 `Observation` nodes from **ADR-302-authenticated** sensors.
  Supported modalities: WiFi (CSI / 802.11bf native reports via ADR-307), BLE,
  UWB, mmWave (ADR-063), acoustic, IMU, lidar, and cameras. Cameras and any
  higher privacy-class modality enter fusion **only where the ADR-277 policy
  engine permits** — camera-free coverage is a RuView invariant (ADR-282), so
  cameras are an opt-in, policy-gated input, never assumed present.
- Each observation carries its own uncertainty and exactly one `EvidenceLevel`
  (ADR-282). An observation flagged out-of-distribution by **ADR-299** is
  down-weighted or excluded per its OOD verdict rather than silently averaged in.

### 2. Combiner: uncertainty-weighted, contradiction-aware

- Observations are combined by their uncertainty into one probabilistic
  `WorldState` over the ADR-303 entities (`Person`, `Object`, `Track`, and the
  per-`Space` inference). The combiner does **not** collapse disagreement: when
  modalities conflict beyond their stated uncertainty, the fused output carries
  ADR-137 **contradiction flags** and the evidence references that produced
  them, so a consumer can see *that* WiFi and mmWave disagree and *why*.
- Coherent multi-node fusion inherits ADR-280's fail-closed
  `CoherentSensorGroup` gate: no coherent combination unless sync, phase, and
  geometry compatibility are proven; otherwise the group degrades to incoherent
  combination rather than producing confident nonsense.

### 3. Output: one world state, provenance preserved

- The output is a single `WorldState` written into the ADR-303 ontology, with
  every fused value retaining recoverable per-observation provenance and the
  ADR-137 quality record. This is the state ADR-309/310/312 consume; they read
  one probabilistic world, not a modality stack.
- The fused state carries an aggregate uncertainty and an evidence level derived
  from its inputs (never upgraded above the weakest contributing L-level for a
  given claim).

## Consequences

- Downstream primitives (spatial memory, counterfactual, RF twin) build on one
  probabilistic world state with uniform uncertainty and provenance, instead of
  re-implementing reconciliation per consumer.
- Contradictions become first-class signal, not noise: ADR-137's record means a
  disagreement between mmWave and CSI is surfaced and auditable, which is also
  what lets ADR-299 and the evidence engine (ADR-301) reason about reliability.
- Fusion is uncertainty-honest: an OOD or low-evidence observation is
  down-weighted, not averaged in as if trustworthy; a fused claim never presents
  a stronger evidence level than its weakest necessary input.
- **No accuracy or "camera-grade" claim is made.** ADR-063's mmWave path has a
  real-silicon capture; the multimodal combiner's accuracy is not asserted here.
  Any fused-accuracy number requires a named reproducer tagged MEASURED /
  SYNTHETIC / CLAIMED, and WiFi sensing is never presented as camera-grade
  (CLAUDE.md, ADR-282). No number is invented.
- Cameras remain a governed, opt-in input; enabling them does not weaken the
  camera-free coverage guarantee for deployments that exclude them.

## Validation

- `cargo test -p wifi-densepose-ruvector` / `-p wifi-densepose-signal` — the
  ADR-137 quality record and contradiction flags travel with the fused output;
  the ADR-280 `CoherentSensorGroup` gate still fails closed under
  clock/phase/geometry violation.
- Fusion invariant test: N modality observations over one scene resolve to a
  single `WorldState` node in the ADR-303 ontology (not N feeds), with
  per-observation provenance recoverable and one aggregate evidence level.
- Uncertainty tests: a high-uncertainty or ADR-299-flagged-OOD observation is
  down-weighted/excluded; conflicting modalities produce a contradiction flag
  rather than a silently averaged value; the fused evidence level never exceeds
  the weakest necessary input.
- Governance test: a camera or higher-privacy modality is admitted into fusion
  only when the ADR-277 policy engine permits; otherwise it is excluded and the
  fused state notes the exclusion.
- Any accuracy comparison (e.g., fused vitals vs. mmWave-only) is reported with
  its evidence tag and reproducer; none is asserted in this ADR.
