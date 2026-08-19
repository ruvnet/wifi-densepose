# ADR-323: Native Rust physics-constrained pose refinement

- **Status**: Proposed
- **Date**: 2026-08-15
- **Deciders**: ruv
- **Owners**: RuView perception and edge runtime maintainers
- **Tags**: pose, physics, rust, uncertainty, provenance, abstention, edge
- **Numbering note**: ADR-323 is the next free number in the authoring checkout. Re-run the ADR index/collision check immediately before merge and rename if needed.
- **Extends**: ADR-020, ADR-027, ADR-079, ADR-101, ADR-135, ADR-145, ADR-150, ADR-273, ADR-279, ADR-282, ADR-295, ADR-296, ADR-297, ADR-298, ADR-302, ADR-303, ADR-304, ADR-305, ADR-306
- **Supersedes**: None

## Executive decision

RuView will add a clean-room native Rust boundary between RF pose inference and
semantic publication. It will preserve the immutable RF observation, publish a
physics assessment, optionally produce a bounded corrected candidate, and
abstain when required evidence is absent. It must never increase observational
confidence merely because a pose is physically plausible.

Three independently gated layers are adopted:

1. A deterministic kinematic auditor and bounded covariance-weighted projector
   using Rust and `nalgebra`.
2. An optional articulated-body dynamics auditor using `rapier3d`.
3. A later optional supervised residual model using Burn.

The first production milestone is deterministic audit. It is not a GRIP port,
not PPO, and not evidence that the current pose observer is production-ready.

## Context

ADR-101's committed Cog emits 17 COCO keypoints as normalized 2D coordinates.
Its model has no per-joint uncertainty head and publishes a constant confidence.
The sensing server also contains renderer-oriented EMA and bone clamping. These
surfaces cannot establish metric 3D physics and can make weak evidence look
more convincing.

Pose output can violate bone length, floor, velocity, acceleration, and temporal
continuity constraints. Downstream consumers also cannot reliably distinguish
observed coordinates from derived correction. The rejected premise is:
"physically plausible means more likely correct." Plausibility is only a prior;
many incorrect poses are plausible.

GRIP is architectural inspiration for an observer/controller split, but it
observes four wearable IMUs and pressure insoles and drives a simulator. RuView
observes RF, so GRIP weights are not input-compatible. External code, weights,
simulators, and datasets require independent license review and never enter the
runtime dependency graph by implication.

## Outcome and actors

For every accepted person track/timestamp, the engine returns exactly one
`PoseRefinementV1`, including off, timeout, rejection, and abstention paths:

- immutable `PoseObservationV2` content hash;
- constraint residuals and quality disposition;
- an optional bounded candidate and an explicit `selected` bit;
- a typed reason when correction is unavailable;
- model, calibration, configuration, and optional learned-artifact provenance.

The RF observer owns observations and calibrated uncertainty; tracking owns
identity stability; physics owns assessment/correction only; the sensing server
owns deadlines, modes, publication, and rollback; the evidence engine owns
release evaluation; clients choose raw/both/refined without silent fallback.

## Input and coordinate contract

Metric correction requires a monotonic nanosecond timestamp, session-scoped
track ID, sequence and sensor epoch, 17 ordered COCO joints in metric X/Y/Z,
per-joint positive-semidefinite covariance calibrated on held-out data, a
versioned right-handed Z-up room frame, a normalized upward floor plane, model
and calibration hashes, ADR-302 trust state, and authenticated/replay-protected
source provenance.

`Image2d` observations may be audited for image-plane ratios and continuity but
must never enter 3D projection/dynamics or be called physically corrected.
Unknown trust, missing calibration, missing uncertainty, stale/non-monotonic
input, non-finite values, invalid covariance, excessive tracks, and room-bound
violations fail to raw output with a typed reason.

## Public contracts

`wifi-densepose-core` owns `PoseObservationV2` and `PoseRefinementV1`; no
duplicate server/Cog contract is permitted. Public output remains COCO17. The
engine derives pelvis and thorax virtually and never labels them observed.

The raw content hash is deterministic and excludes its own hash field. The
idempotency key is `(sensor_epoch, sequence, track_id, raw_hash, config_hash)`.
An exact duplicate returns the cached result; same sequence with different
content is a replay rejection.

Contact is `hypothesis` unless a measured sensor and its provenance say
otherwise. Raw, derived, hypothesis, and unknown labels must survive every
projection.

## Confidence invariant

For upstream calibrated confidence `c_obs`, normalized residual `r`, and
normalized intervention `i`:

```text
c_physics = exp(-(beta_r * r + beta_i * i))
c_effective = min(c_obs, c_obs * c_physics)
0 <= c_effective <= c_obs <= 1
```

Only a separately witnessed multimodal fusion contract may increase fused
confidence.

## Deterministic projector

The default `kinematic` feature has no Rapier, Burn, ONNX, libtorch, Python,
CUDA, or network dependency. Per bounded iteration it:

1. projects observed parent/child distances toward anonymous track-scoped
   bone-length posteriors;
2. applies broad joint/trunk validity checks without an upright prior;
3. bounds temporal motion and resets derivatives after gaps;
4. resolves floor penetration only, allowing seated, kneeling, prone, child-
   scale, mobility-aid, and genuine-fall poses;
5. recomputes residuals and stops below epsilon.

Initial operator-owned caps are four iterations (hard maximum eight), 0.20 m
single-joint correction, 0.10 m root correction, 250 ms derivative gap, 500 ms
track reset, ten known joints, a 100 m metric room bound, a separate 16,384
image-coordinate audit bound, and a 5 ms one-track Pi 5 p95 gate. Keeping image
and metric bounds separate prevents legitimate pixel observations from
weakening the physical room bound. A candidate over either correction cap is
discarded in full.

Bone posteriors are initialized only from high-confidence frames, anonymous,
memory-only, track-scoped, and deleted on expiry. Persistent personalization is
outside this ADR and requires consent/retention/deletion governance.

## Optional dynamics and learned layers

`dynamics` adds a process-owned Rapier humanoid and begins audit-only. Network
input may never provide Rapier snapshots, bodies, constraints, solver limits,
or arbitrary geometry. Dynamics approval is independent of kinematic approval.

`learned` uses first-party Burn 0.21 core/NN components without `burn-tch`
because this workspace already has a different native libtorch link.
`learned-cpu` adds the ndarray backend. The implemented two-layer GRU uses a
20-frame history and width 128 to predict bounded residuals, uncertainty,
foot-contact hypotheses, and abstention. Verified model records can be loaded
from bytes and executed natively; no trained artifact is shipped or approved.
The resolved Burn/CubeCL graph declares Rust 1.92, while the workspace file
pins Rust 1.89 and the authoring host provides Rust 1.91.1.
`--ignore-rust-version` is diagnostic evidence only: learned activation remains
blocked until an approved Rust 1.92 release-toolchain change builds it without
that override. Residuals are hard-clipped to deterministic caps and cannot
bypass validation or confidence monotonicity. PPO is deferred until measured
evidence identifies a failure supervised residual learning cannot address.

## Feature boundary

```text
default = kinematic
dynamics = rapier3d
learned = burn-core + burn-nn
learned-cpu = learned + burn-ndarray
learned-train = learned + burn-train
learned-wgpu = learned-train + burn-wgpu
learned-cuda = learned-train + burn-cuda
deterministic = rapier3d?/enhanced-determinism
```

The lockfile is release authority. The learned feature currently requires the
toolchain supported by Burn/CubeCL's resolved graph; this does not change the
default edge build.

## Runtime modes and API

Rollout is `OFF -> AUDIT -> SHADOW_CORRECT -> OPT_IN_CORRECT -> DEFAULT_CORRECT`.
Evidence permits forward transitions; any regression returns immediately to
audit/off. Correct selection additionally requires authenticated sensor
identity and replay protection from ADR-305. High model confidence cannot
override missing source authentication.

Existing pose fields stay unchanged and raw remains the migration default:

```text
GET /api/v1/pose/current?view=raw
GET /api/v1/pose/current?view=both
GET /api/v1/pose/current?view=refined
```

Refined-only returns HTTP 409 with `pose_refined_unavailable` when no selected
candidate exists. It never silently returns raw labeled refined.

## Security, privacy, and availability

All frames, model output, geometry, and pre-verification artifacts are
untrusted. Calibration/config/model artifacts become trusted only after signed,
hash-addressed verification and atomic activation. Runtime inference performs
no model retrieval or other network access.

Fixed arrays/caps, bounded iterations, a maximum track count, room geometry
limits, deadlines, and track expiry constrain denial of service. Timeout drops
partial refinement, never raw publication. Backpressure retains the newest raw
frame per track, drops intermediate refinement work, resets derivatives after
250 ms, and never extrapolates beyond 500 ms.

Metrics contain only allowlisted aggregate scalars: mode/disposition/reason,
stage latency, iterations, maximum correction, residuals, confidence delta,
track resets, invalid input, timeout, and raw/refined divergence. They exclude
joint arrays, body dimensions, room coordinates, CSI, and persistent person
identifiers. Bone/gait state is memory-only and excluded from logs.

Refined output is not a sole medical, emergency, industrial-safety, or
autonomous-control source. A real fall is valid state and must never be made
upright to stabilize a simulator.

## Threat model summary

| Threat | Primary control | Residual risk |
|---|---|---|
| Spoofed/replayed sensor | ADR-305 identity, MAC, sequence and replay window; correction gate | Compromised legitimate sensor |
| Altered model/floor/config | Signed hashes, authenticated configuration, atomic activation | Authorized unsafe configuration |
| Poisoned data/splits | Immutable manifests, strict split validator, witnessed benchmarks | Subtle label poisoning |
| Operator repudiation | Append-only witnessed transition with actor/old/new hash/reason | Compromised signer |
| Biometric/log leakage | Track-local retention and fixed metric allowlist | Aggregate inference |
| Track/geometry CPU flood | Authentication, cardinality/geometry/allocation/deadline caps | Valid dense-scene overload |
| Remote mode escalation | Capability-scoped local control plane, deny by default | Compromised operator capability |
| Derived output relabeled observed | Required schema/provenance and signed event envelope | Malicious downstream stripping |

The implementation review records commit, lockfile hash, Rust toolchain,
scanner versions, and advisory-feed timestamp.

## Evidence protocol

Evidence levels are L0 deterministic synthetic, L1 public measured replay, L2
controlled RuView RF plus optical truth, L3 subject/room/hardware/session-
disjoint RuView, L4 privacy-safe shadow fleet aggregates, and L5 independent
vertical validation outside this ADR.

No sequence, contiguous take, subject, room, or calibration session may cross
train/test for the generalization gate. Preprocessing, body priors, and
uncertainty calibration fit training data only. Reports include raw observer,
renderer smoothing, audit, deterministic correction, dynamics audit, and
learned residual on identical observations, plus empty-room, prone/fall,
missing-joint, and OOD subsets.

Primary metrics are 3D MPJPE, declared-threshold PCK, per-joint error, foot
slide, floor penetration, jerk, uncertainty calibration, abstention coverage,
and selective risk. Learned runs use at least five fixed seeds and report mean,
median, standard deviation, and 95% bootstrap intervals. All frames count;
selective metrics report risk and coverage.

## Acceptance gates

- **G0 contract**: real metric 3D/covariance output, round-trip raw hash,
  versioned frame/floor, 2D compatibility, non-stub observer, ADR-298 artifact
  sanity, and the ADR-079 PCK@20 >=35% gate or adopted successor. The current
  committed Cog does not pass G0, so correction remains unavailable.
- **G1 deterministic audit**: property/fuzz tests, deterministic hashes per
  platform class, 24-hour accelerated replay without panic/growth, Pi 5 p95
  <=5 ms, and universal confidence monotonicity.
- **G2 shadow correction**: strict-disjoint measured median MPJPE improvement
  >=10% with positive 95% CI lower bound; foot slide >=30% and jerk >=25%
  better; no joint median >5 mm worse; fall/prone sensitivity change <=2 pp;
  >=95% corrections below 0.10 m; every correction above 0.20 m abstains.
- **G3 opt-in**: >=30 subjects, 10 rooms, 3 hardware configurations, and 3
  independent sessions/room; UNKNOWN never selected; confidence monotonic;
  live disable; REST/WebSocket/MQTT/Home Assistant/replay compatibility.
- **G4 default visualization only**: 30 shadow days under 0.1% timeout/internal
  error, no open severity 1/2 incidents, and gates still valid for current
  model/calibration.

Dynamics and learned engines each repeat G2-G4; approval is not inherited.

## Testing and completion evidence

Unit/property/fuzz/integration/security coverage maps to requirements R1-R13:
raw hash, confidence, modes, malformed/stale/frame/covariance input, caps and
deadlines, provenance, dependency graph, pose diversity/fall preservation,
strict splits, fail-to-raw faults, no network capability, and authenticated
source/replay selection.

Release commands include focused core/physics tests, default/dynamics/learned
feature checks, format/clippy, benches, `cargo deny`, `cargo audit`, strict split
verification, and golden replay verification. Completion also requires JSON
schemas, measured Pi 5/x86 rows, strict manifest hashes, raw/refined metrics,
SBOM/license report, rollback drill, and residual-risk owners. Missing measured
or operational evidence leaves status Proposed and runtime in audit.

## Rollback

Rollback is an authenticated mode transition to audit/off, not a binary
downgrade. Stop selection immediately, keep raw publication and disposition
records, discard track state, and retain only aggregate incident metrics plus
signed configuration history. Failed artifact activation leaves the previous
engine atomically active. Additive schemas remain; refined-only callers receive
the typed unavailable response.

## Consequences

### Positive

- Explicit anti-hallucination and provenance boundary after RF inference.
- Reusable native Rust consistency primitive with measurable abstention.
- Python/CUDA remain absent from the production default.
- Cross-modal teacher data remains possible without wearable runtime inputs.

### Negative

- Full value requires a real metric 3D observer and calibrated uncertainty.
- Stateful tracks add latency/memory; optional backends add supply-chain surface.
- A constrained but wrong pose can look more credible.
- Strict data collection costs more than the software implementation.

### Neutral

- This ADR does not improve RF observability or current weight evidence.
- Existing 2D consumers continue to function.

## Implementation phases

P0 contracts/schemas; P1 deterministic audit; P2 bounded shadow correction; P3
server/Cog publication and evidence ledger; P4 Rapier audit; P5 Burn residual
training/inference. Code may land ahead of evidence, but runtime authority
advances only through the gates above.

## Implementation status at proposal

- P0-P3 are implemented on this branch: canonical contracts, strict schemas,
  deterministic audit/projection, authenticated correction receipts,
  idempotency, bounded track state, latest-frame backpressure, additive HTTP
  and WebSocket publication, live legacy-2D audit, privacy-safe metrics, golden
  replay, and strict-split checks.
- P4 is implemented as an optional persistent per-track Rapier dynamics auditor
  and remains audit-only pending independent G2-G4 evidence.
- P5 inference architecture, artifact verification, serialization, and native
  CPU execution are implemented. Training data, a signed trained artifact, and
  G2-G4 accuracy/calibration evidence do not exist, so the layer has no runtime
  selection authority. Its resolved Rust 1.92 requirement is also an explicit
  activation blocker on the current Rust 1.91.1 release host.
- The live Cog honestly emits `Image2d`, degraded trust, and uncalibrated
  uncertainty. It can be audited but cannot be selected for 3D correction.
  G0 therefore remains open until an independently released metric-3D observer
  with calibrated covariance is integrated.
- Local x86 latency and synthetic contract checks are recorded in the append-
  only evidence ledger. Pi 5 measurements, 24-hour replay, 100-million-case
  fuzzing, held-out RF/optical accuracy, fleet shadowing, and vertical safety
  validation remain release evidence gates rather than software claims.

## References

- [GRIP project](https://ryosukehori.github.io/grip-project/)
- [GRIP paper (arXiv:2603.16233)](https://arxiv.org/abs/2603.16233)
- [Rapier documentation](https://docs.rs/rapier3d/)
- [Burn documentation](https://docs.rs/burn/0.21.0/burn/)
- [ADR-020](./ADR-020-rust-ruvector-ai-model-migration.md)
- [ADR-079](./ADR-079-camera-ground-truth-training.md)
- [ADR-101](./ADR-101-pose-estimation-cog.md)
- [ADR-150](./ADR-150-rf-foundation-encoder.md)
- [ADR-273](./ADR-273-unified-rf-spatial-world-model.md)
- [ADR-279](./ADR-279-native-rf-frame-contract.md)
- [ADR-298](./ADR-298-model-release-sanity-gates.md)
- [ADR-302](./ADR-302-out-of-distribution-detection.md)
- [ADR-303](./ADR-303-ground-truth-synchronization.md)
- [ADR-304](./ADR-304-evidence-engine.md)
- [ADR-305](./ADR-305-authenticated-sensor-identity.md)
- [ADR-306](./ADR-306-canonical-spatial-ontology.md)
