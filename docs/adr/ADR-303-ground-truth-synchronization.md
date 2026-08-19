# ADR-303: Ground-truth synchronization — reference sensors as a formal validation plane

- **Status**: Accepted — initial implementation (ADR-300 phase 2)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: ground-truth, validation, fusion, evidence, benchmark, honesty, substrate

## Context

This ADR is primitive 3 of the perception-substrate program (ADR-300), authored
as **Proposed** in phase 2: it is design intent and a validation plan, not
implemented by the phase-1 swarm. It sits on top of the phase-1 certificate
spine and feeds the evidence engine (ADR-304) and the real benchmark service
(ADR-317). It generalizes the vitals ground-truth rig (ADR-293) from a single
measurand to a modality-agnostic plane.

RuView's evidence discipline (CLAUDE.md; ADR-282 ladder) requires MEASURED
accuracy claims to be backed by an independent reference. ADR-293 built exactly
this for vitals: reference-series ingest, time alignment (cross-correlation
lag + optional clock-drift fit), and agreement statistics (MAE/RMSE/bias/
Bland–Altman/within-tolerance), with an `EvidenceGrade` that is only
constructible as `Measured` when a real reference, non-zero paired samples,
minimum coverage, and a reproducer are present. That machinery is measurand- and
device-shaped: it knows about heart rate and breathing rate.

The substrate needs the same discipline for *every* phenomenon RuView senses —
presence, count, localization, pose, posture, activity — and for reference
sources of many modalities (cameras, mmWave, pressure mats, wearables, pulse
oximeters, microphones, manual labels). The critical design decision is that
these reference sensors form a **validation plane**, not additional inference
inputs.

## Options considered

1. **Fuse reference sensors as extra inference inputs.** Rejected on principle:
   folding cameras/mmWave into the estimator would make RuView's RF claims
   unfalsifiable — the reference would be training the thing it is meant to
   check, and a camera-fed result is no longer a camera-free RF result. It
   would also violate the ADR-282 layering (RuView is probabilistic
   exteroception, never ground truth) and the honesty rule against presenting
   fused-with-camera output as WiFi sensing.
2. **One-off rigs per measurand (extend ADR-293 ad hoc each time).** Rejected:
   duplicates alignment/agreement code per phenomenon and never yields a shared
   validation surface for the benchmark.
3. **A first-class, modality-agnostic `GroundTruth` API that is strictly a
   validation plane.** Chosen.

## Decision

Introduce a `GroundTruth` API — a modality-agnostic validation plane that
compares RF inference against independent observation and never feeds it.

### 1. Modality-agnostic reference ingest

- A `ReferenceObservation` generalizing ADR-293's `ReferenceSeries`: a
  timestamped, typed observation of a `Phenomenon` (presence, count,
  localization, pose keypoints, posture, activity, heart rate, breathing rate)
  from a `ReferenceModality` (camera, mmWave, pressure, wearable, pulse
  oximeter, microphone, manual label), with device/source metadata and the
  measurement principle recorded.
- Untrusted reference files are validated at the boundary (row-numbered
  rejections, non-monotonic timestamps are errors), reusing ADR-293's ingest
  discipline. Camera/mmWave references arrive as exported label/keypoint
  streams, not live model feeds.

### 2. Synchronization

- Generalize ADR-293's time alignment (bounded-lag normalized cross-correlation
  + optional linear clock-drift fit) to arbitrary measurands on a common
  resampled grid, with no interpolation across gaps beyond a configurable
  limit. Alignment parameters are always reported, never silently applied.
- Spatial synchronization where relevant: reference observations are expressed
  in the ADR-306 spatial ontology so an RF localization/pose result and a
  camera/mmWave observation are compared in one coordinate frame.

### 3. Agreement as validation, not fusion

- A modality-appropriate `AgreementReport` per phenomenon: continuous
  measurands reuse ADR-293's MAE/RMSE/bias/Bland–Altman/within-tolerance;
  categorical/detection phenomena (presence, activity) report confusion-matrix
  metrics; spatial phenomena report localization error percentiles and pose
  PCK **with the mandatory mean-pose baseline and leakage-free split**
  (CLAUDE.md; ADR-291).
- Session scope is mandatory metadata (subject count, motion state, LOS/NLOS/
  through-wall, distance band) — a report without scope cannot be constructed,
  as in ADR-293.

### 4. Evidence and isolation guarantees

- The plane is one-directional by type: the inference path has no read access
  to `GroundTruth` at runtime. A build/test-time isolation check (and the type
  boundary) prevents a reference observation from becoming an estimator input.
- Reports carry an `EvidenceLevel` (ADR-282) and an `EvidenceGrade`
  constructible as `Measured` only with a real reference, paired samples,
  coverage, and a reproducer (ADR-293 rule). Reports feed the ADR-304 evidence
  engine and are the substrate ADR-317 scores against.

## Consequences

- Every phenomenon RuView senses gets the same MEASURED-vs-independent-observer
  discipline vitals already has, in one shared surface.
- Keeping references strictly as validation preserves the falsifiability and
  the camera-free identity of RF results; it costs the (tempting) accuracy a
  camera-fused estimator would show, which is the correct trade.
- Reference capture is an operational burden (a camera/mmWave rig per validated
  session); acceptable because it is a validation activity, not a runtime
  requirement, and it is what turns CLAIMED into MEASURED.
- Because this is Proposed (phase 2), the API shape may be revised once the
  phase-1 spine (ADR-301/299/301/303) lands and the benchmark (ADR-317)
  exercises it.

## Validation

- Unit tests (planned): modality-agnostic ingest rejection cases; alignment
  recovery of known synthetic offsets/drifts across measurands; agreement math
  per phenomenon against hand-computed fixtures; pose PCK path requires a
  mean-pose baseline and rejects leaky splits; evidence-grade constructibility;
  the isolation check fails a build that wires a reference into the inference
  path.
- Cross-ADR: an ADR-317 benchmark scenario consumes `GroundTruth` reports as
  its scored reference; ADR-304 ingests the agreement reports as evidence
  records.
- Real-session validation (RF capture synchronized with a real camera/mmWave/
  pressure/wearable reference) is the phase-2 exit and requires hardware
  evidence per CLAUDE.md; a synthetic run is not hardware evidence.
