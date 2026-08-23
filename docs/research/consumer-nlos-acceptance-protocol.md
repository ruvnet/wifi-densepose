# Preregistered RuView consumer NLOS reproduction and CSI-fusion protocol

**Protocol ID:** `ruview-consumer-nlos-v1`
**Status:** Template to freeze before the first scored capture
**Governed by:** ADR-328, ADR-329, ADR-330, ADR-331
**Primary claim scope:** controlled, identity-free tracking of a known hidden
rigid target using an externally enrolled, histogram-capable ST transient sensor
whose exact silicon/firmware/API are recorded; not built-in iPhone
NLOS, through-wall optical sensing, unrestricted people tracking, or safety use

## 1. Research question and decision rule

This protocol answers two ordered questions:

1. Can the pinned consumer-NLOS baseline track a hidden target from live
   commodity transient LiDAR at roughly 30 frames per second?
2. After that is established, does independently captured RuView CSI improve
   target-position error or lost-track rate by enough to justify fusion?

The program passes only when all provenance/privacy/security gates pass and:

1. the LiDAR-only arm emits at least **27 accepted end-to-end track updates per
   second** on live hardware; and
2. over at least **100 paired live sequences**, the fused arm reduces either
   mean target-position error or lost-track rate by **at least 25 percent**
   relative to LiDAR-only, with its adjusted interval excluding zero and all
   frozen guardrails passing. Fused update rate remains a reported guardrail,
   not an added substitute for the requested improvement endpoint.

`SYNTHETIC`, simulator, generated, duplicated, or captured replay frames cannot
pass. They may be used for software QA and pilot power planning only. A build,
UI frame rate, configured sensor frequency, or upstream paper result is not the
measured endpoint.

## 2. Hypotheses

### H1: live reproduction

For the primary controlled stratum, the LiDAR-only pipeline's accepted
sensor-to-track update rate is at least 27 Hz. Failure stops confirmatory fusion
interpretation. Diagnostics may continue but are labeled exploratory.

### H2a: position-error improvement

\[
G_e = \frac{\bar E_L - \bar E_F}{\bar E_L} \ge 0.25,
\]

where \(\bar E_L\) and \(\bar E_F\) are sequence-weighted mean Euclidean target
position errors for LiDAR-only and fused arms, using an external ground-truth
coordinate frame.

### H2b: lost-track improvement

\[
G_\ell = \frac{\ell_L - \ell_F}{\ell_L} \ge 0.25,
\]

where \(\ell\) is the fraction of evaluable time that satisfies the frozen
lost-track definition.

Fusion succeeds if H2a **or** H2b meets its magnitude and multiplicity-adjusted
uncertainty gate. Both metrics and all guardrails are reported. A zero baseline
denominator cannot establish gain.

## 3. Roles and separation of duties

| Role | Responsibility | Must not do |
|---|---|---|
| Protocol owner | freeze protocol, strata, splits and decision rule | inspect sealed confirmatory results before freeze |
| Capture operator | approved setup, consent, identity/calibration, run manifest | tune model/thresholds during scored capture |
| Ground-truth owner | independent system, clock/transform checks, sealed labels | feed labels into online LiDAR or fusion arms |
| Model owner | freeze upstream/Rust/CSI/fusion artifacts | alter artifacts after test partition opens |
| Analyst | run committed reproducer and report all endpoints/strata | delete trials or change exclusions post hoc |
| Security/privacy reviewer | approve collection, access, retention, threats | waive live provenance or identity/actuation boundaries |
| Witness reviewer | verify digests, randomization, exclusions and analysis | equate acceptance JSON arithmetic with physical audit |

One person may hold multiple roles in a pilot, but protocol/model ownership and
ground-truth/confirmatory analysis should be independently reviewed.

## 4. Hardware and software freeze

Complete and sign this table before scored capture:

| Item | Frozen value |
|---|---|
| ST kit and sensor | VL53L8CH silicon read from hardware; raw/CNH API compatibility witnessed; scoped enrollment/certificate reference stored in the restricted manifest rather than a guessable raw-serial hash |
| Firmware | source/release, compiler/toolchain, binary SHA-256 |
| Sensor configuration | zones, bins, bin width, requested rate, integration/subsampling, ambient settings |
| Upstream baseline | `sidsoma/consumer-nlos` commit `15314de422a765a2d1b72ea7037dfafb2f908d7c` and clean/patch manifest |
| RuView | full Git commit SHA; `ruview-nlos` crate feature/config digest |
| CSI nodes | authenticated IDs, hardware/firmware, channel/bandwidth/subcarrier configuration |
| Optical calibration | wall points/plane, direct-return masks/peaks, background, timestamps, SHA-256, expiry |
| RF calibration | room/link fingerprint, coordinate transform, timestamp, SHA-256, expiry/OOD threshold |
| Ground truth | device/camera/tag firmware/software, calibration digest and measured clock uncertainty |
| Models | canonical target response, particle count/motion prior/score, CSI model, fusion weights and digests |
| Hosts | CPU/GPU/RAM/OS, power mode, process priority, compiler/runtime versions |
| Analysis | script/lockfile/container digest, bootstrap seed list and report template |

Changing any frozen item starts a new protocol version or invalidates the
affected block. No silent patch is permitted.

## 5. Physical setup and primary stratum

### 5.1 Geometry

1. Mount the transient sensor so a matte, light-colored planar relay surface
   fills its field of view.
2. Place an opaque occluder so no sensor zone, phone camera, or operator-facing
   optical path directly sees the scored target. Record a setup photograph/mesh
   for review; do not publish participant imagery by default.
3. Start with the upstream-friendly geometry: sensor-to-wall less than 1 m and
   wall-to-target approximately 1 to 1.5 m, then record exact distances.
4. Define a right-handed world coordinate frame, units in metres, transform
   chain, uncertainties, and a hidden-region boundary before capture.
5. Place CSI nodes/APs in a fixed documented configuration. Confirm CSI is not
   derived from the LiDAR, target-control signal, or ground-truth system.

### 5.2 Primary target

The confirmatory reproduction target is a known rigid approximately 25 cm
retroreflective patch or the exact upstream canonical target. The target shape
and response are frozen before the confirmatory split. This supports a scoped
known-shape tracking claim only.

Diffuse rigid objects, hands/people, multiple objects, other sizes, longer
ranges, sunlight, non-planar relay surfaces and moving sensors are separate
exploratory or later confirmatory strata. Never pool them to imply generalized
human/scene reconstruction.

### 5.3 Ground truth

Use an independent externally calibrated system, for example an overhead camera
with a rigid AprilTag/active marker or a surveyed motion stage. The system must
observe the target while remaining unavailable to the online algorithms. Measure
spatial transform error and clock offset/jitter before and after each block.

Ground-truth labels remain sealed until capture and all three online arms are frozen.
A label interpolated beyond the frozen maximum gap makes that time point
unevaluable; it is not imputed from the NLOS output.

## 6. Calibration and negative controls

For every block:

1. enroll/verify sensor and CSI identities;
2. fit the relay wall and verify per-zone direct-return distance against a
   physical measurement;
3. capture the frozen-duration empty-scene optical background with no person or
   target in the hidden region;
4. capture RF empty-room calibration under the approved protocol;
5. verify clocks, transform chain and calibration digests;
6. run an empty hidden-region negative sequence;
7. run a direct-line-of-sight exclusion check with the occluder; and
8. mark calibration `VALID` only after all bounds pass.

Negative controls include sensor disconnected, CSI disconnected, stale/replayed
frame injection in a non-scored software run, target absent, static distractor,
and calibration mismatch. The live confirmatory stream contains no injected
synthetic/replay frames.

## 7. Trial unit, sample size and randomization

### 7.1 Paired sequence

A paired sequence is one continuous, live, preregistered target trajectory whose
accepted transient-histogram and CSI frames are delivered simultaneously to frozen
online arms:

1. **L:** LiDAR-only MAS tracker; CSI is unavailable to every decision in this
   arm; and
2. **C:** CSI-only ablation, reported to establish independent RF information;
   it need not satisfy a centimetre-localization promotion threshold; and
3. **F:** the same optical inputs and initialization plus the frozen calibrated
   CSI likelihood.

All three arms use the same synchronized live interval; L/F share the offered
optical fan-out and C/F share the offered CSI fan-out. They do not take turns on
different captures. One-arm drops remain outcomes, not exclusions. If resource contention is material, run them
on matched isolated hosts fed by the authenticated live fan-out and record fan-
out latency/drop parity. Replaying a recording later does not satisfy the live
gate.

### 7.2 Minimum and power

Capture at least 100 valid paired sequences in the primary stratum. Before
confirmatory capture, use a disjoint pilot or synthetic/replay data to estimate
cluster variance and document power for detecting a 25 percent gain at family-
wise alpha 0.05. If the calculated requirement exceeds 100, use the larger
number. Pilot sequences, rooms and target paths do not enter confirmatory
metrics.

Each sequence duration, initialization window and trajectory family is frozen.
Include translation directions, speeds and positions across the hidden volume,
not 100 copies of one favorable path. Randomize trajectory order and block order
with a committed seed. Counterbalance any host assignment.

### 7.3 Grouping and splits

Calibration/training, pilot and confirmatory partitions are grouped by capture
session, time block, path family, target instance, room and sensor configuration.
Adjacent frames from one sequence cannot cross partitions. If a learned CSI
model uses LiDAR supervision, no optical target/embedding from the confirmatory
partition is used for training, thresholding, normalization or early stopping.

## 8. Online quality and exclusion rules

Freeze numeric values for each placeholder before capture:

| Rule | Frozen value |
|---|---|
| Valid calibration age and OOD bounds | `<fill before capture>` |
| Maximum optical/CSI/ground-truth clock uncertainty | `<fill>` ms |
| Maximum optical-to-CSI pairing skew | `<fill>` ms |
| Maximum ground-truth interpolation gap | `<fill>` ms |
| Track association radius | `<fill>` m |
| Lost-track consecutive interval | `<fill>` frames or ms |
| Posterior quality/entropy/effective-particle threshold | `<fill>` |
| Saturation/underexposure and minimum valid zones | `<fill>` |
| Maximum sequence frame-loss fraction | `<fill>` |
| Warmup/initialization exclusion | `<fill>` frames, applied to all three arms |

Pre-capture exclusions only:

1. consent/safety/indicator failure;
2. sensor/CSI/ground-truth identity or clock failure;
3. calibration invalid before sequence start;
4. direct line of sight or physical setup outside tolerance;
5. target controller/ground truth did not execute the randomized trajectory;
6. raw capture corruption affecting all affected arms; or
7. host failure prevents paired operation.

Algorithm failure, low signal, lost track, high error, drift, overload, one-arm
drop, poor target position, unfavorable reflectivity or unexpected but in-scope
motion are outcomes, not exclusions. Report all excluded sequences with reason
and arm-independent timing.

## 9. Endpoint definitions

### 9.1 Accepted update rate

For each sequence and arm, use the entire frozen evaluable wall-time window:

\[
f = \frac{N_{valid,new}}{T_{evaluable}},
\]

where `valid,new` means a newly computed, schema-valid, calibrated, fresh track
from a unique live sensor frame after the symmetric warmup. Duplicates, replay,
late/stale frames, renderer frames and cached outputs do not count. Report
the accepted sensor-frame count as well and require track updates not to exceed
it. If an implementation instead uses first-to-last span, its estimator is
`(N-1)/(t_last-t_first)`, not `N/span`. Report sequence distribution and overall accepted updates divided by evaluable wall
time. H1 and the fused performance guard use the lower preregistered aggregate
definition, not the maximum instantaneous rate.

### 9.2 Position error

At each evaluable matched timestamp:

\[
e_{a,t}=\|\hat p_{a,t}-p^{GT}_t\|_2.
\]

Compute a mean within each sequence first, then the equally weighted mean across
sequences so long sequences do not dominate. Report median, p95 and axis-wise
error as secondary metrics. Invalid/missing intervals contribute to lost-track
rate and cannot simply disappear from the report. For the confirmatory position
endpoint, every paired sequence receives a score in both arms. A sequence with
no valid position receives the preregistered worst-case/censoring penalty; it is
not dropped. Consequently each arm's `position_error_sample_count` must equal
`paired_sequences`, and the shared scoring-mask/penalty rules are bound by
`endpoint_pairing_sha256`.

### 9.3 Lost-track rate

A track is lost when the arm has no `VALID` matched hypothesis within the frozen
association radius for at least the frozen consecutive interval after warmup.
Lost-track rate is lost evaluable time divided by total evaluable time. Report
number/duration of episodes and reacquisition time. Track-ID changes without
position loss are reported separately.

### 9.4 Empty-region false tracks and safety guardrails

Report confident-track time and event count during preregistered empty-region
sequences. Also report p95 latency, frame loss, calibration/OOD rejection and
non-winning primary metric. A fusion gain accompanied by materially worse empty-
region false tracks, severe latency/update-rate loss, provenance failure or a
privacy/security failure is not promoted even if the 25 percent arithmetic
passes. `offered_optical_frame_count` is the number of eligible optical frames
offered to both paired arms. Its rate is recomputed against the shared
LiDAR/fused evaluable duration and cannot exceed `sensor_configured_max_hz`.
Fused frame loss is recomputed as
`(offered_optical_frame_count - fused.accepted_sensor_frame_count) /
offered_optical_frame_count`; a supplied rate that differs fails closed.

## 10. Statistical analysis

1. Calculate paired per-sequence deltas and relative gains. Do not treat frames
   within a sequence as independent samples.
2. Use a paired cluster bootstrap over sequences with at least 10,000 resamples
   and committed seeds. Report point estimate and two-sided interval for every
   metric.
3. Because H2a/H2b are alternative success endpoints, control family-wise error
   with Bonferroni-adjusted 97.5 percent confidence intervals or a frozen
   equivalent procedure. The successful endpoint needs point gain at least 25
   percent and its adjusted interval must exclude zero improvement.
4. Report all primary/secondary/stratified results, exclusions and missingness.
   No optional stopping; capture count is frozen by power/minimum before the
   confirmatory set opens.
5. Sensitivity analyses vary the frozen association/lost-track thresholds only
   as clearly labeled exploratory analysis after the primary result.

If only one endpoint passes, state exactly which one. Do not summarize it as
“25 percent more accurate” when the passing endpoint was lost-track rate.

## 11. Execution sequence

### Gate A: software before participants/live capture

1. Run the NLOS Rust unit/property/golden tests and workspace gate.
2. Run Swift core tests and macOS iOS simulator build where available.
3. Run mobile web tests, typecheck, lint and web export.
4. Run harness tests/security/brain/flywheel/manifest/package gates.
5. Run secret/raw-data/dependency/license scans and close confirmed high/critical
   findings.
6. Exercise deterministic `SYNTHETIC` replay, stale/disconnect/oversize/replay
   rejection and empty fixture. Label all outputs software/L0.

Convenience check:

```bash
cd harness/ruview
node bin/cli.js nlos verify --repo ../.. --run-builds
```

This reports only the available build/discovery subset, with explicit skips. It
cannot pass full Gate A, Gate B, or a release gate by itself; steps 1–6 remain
required.

### Gate B: upstream live reproduction

1. Freeze/sign protocol, privacy approval, identities, software/hardware table,
   randomization and analysis container.
2. Inspect geometry and direct-line-of-sight exclusion.
3. Calibrate optical/RF/ground truth and run negative control.
4. Execute live primary-stratum sequences with LiDAR-only online output and
   ground-truth labels sealed.
5. Verify accepted end-to-end update rate >=27 Hz and report tracking/error/
   empty-region diagnostics.
6. If Gate B fails, stop confirmatory fusion interpretation; fix path under a
   new protocol version.

### Gate C: paired live fusion

1. Freeze the CSI likelihood, fusion weights, arm fan-out and host assignment.
2. Execute at least the powered minimum of paired randomized live sequences.
3. Open ground truth once capture/artifacts/exclusions are immutable.
4. Run the committed analysis, bootstrap and stratified report.
5. Retain fusion only if update-rate, 25 percent endpoint, provenance,
   privacy/security and guardrail review pass.

### Gate D: Apple/iOS claim

Native/web software may ship as external-track clients after Gate A. A claim that
built-in Apple LiDAR performs transient NLOS needs a separate adapter ADR,
documented public histogram API, physical-device capture, privacy/App Store
review and fresh Gates B/C. ARKit scene depth alone cannot enter Gate B.

## 12. Required artifacts

| Artifact | Contains | Excludes |
|---|---|---|
| Frozen protocol | signed/versioned text, hypotheses, thresholds, seeds | post-result edits |
| Capture manifest | content digests, identities, configurations, timing, strata, exclusions | raw credentials/private keys |
| Raw store | encrypted transients/CSI/ground truth under access/retention policy | Git/package/harness inclusion |
| Split manifest | grouped calibration/pilot/confirmatory IDs and hash | participant identity in public artifact |
| Analysis reproducer | locked dependencies, script, seeds, exact tables/plots | manual spreadsheet-only results |
| Security/privacy record | approvals, threat verification, retention/deletion | blanket “camera-free is safe” claim |
| Acceptance JSON | bounded aggregate fields and SHA-256 references | raw trajectories/sensor data/tokens |
| Witness report | reviewer checks and exact evidence level | higher-level field/production implication |

## 13. Acceptance JSON

The repository-contained record consumed by `ruview nlos verify` has this exact
top-level and per-arm key set. Unknown fields are rejected; values below are
placeholders, not results:

```json
{
  "schema": "ruview.nlos.acceptance.v1",
  "source": "LIVE_HARDWARE",
  "claim_tag": "MEASURED",
  "evidence_level": "L2",
  "ground_truth": "EXTERNAL",
  "protocol_frozen_before_capture": true,
  "witness_reviewed": true,
  "privacy_review_passed": true,
  "security_review_passed": true,
  "los_exclusion_verified": true,
  "independent_csi_verified": true,
  "sensor_model": "VL53L8CH",
  "transient_kind": "COMPACT_NORMALIZED_HISTOGRAM",
  "sensor_configured_max_hz": "<frozen-number>",
  "upstream_commit": "15314de422a765a2d1b72ea7037dfafb2f908d7c",
  "protocol_sha256": "<nonzero-64-hex>",
  "capture_manifest_sha256": "<nonzero-64-hex>",
  "sensor_identity_sha256": "<nonzero-64-hex-scoped-enrollment-reference>",
  "calibration_sha256": "<nonzero-64-hex>",
  "firmware_sha256": "<nonzero-64-hex>",
  "analysis_sha256": "<nonzero-64-hex>",
  "endpoint_pairing_sha256": "<nonzero-64-hex-shared-position/lost-track-evaluable-mask-manifest>",
  "sensor_configuration_sha256": "<nonzero-64-hex>",
  "witness_report_sha256": "<nonzero-64-hex>",
  "privacy_review_sha256": "<nonzero-64-hex>",
  "security_review_sha256": "<nonzero-64-hex>",
  "guardrail_report_sha256": "<nonzero-64-hex>",
  "csi_capture_manifest_sha256": "<nonzero-64-hex>",
  "csi_sensor_identity_sha256": "<nonzero-64-hex-scoped-enrollment-reference>",
  "csi_calibration_sha256": "<nonzero-64-hex>",
  "synthetic_frames": 0,
  "replay_frames": 0,
  "paired_sequences": "<integer-at-least-100>",
  "csi_source_count": "<positive-integer>",
  "offered_optical_frame_count": "<positive-integer>",
  "lidar_only": {
    "sequence_count": "<equals-paired-sequences>",
    "accepted_sensor_frame_count": "<integer>",
    "accepted_update_count": "<integer>",
    "evaluable_duration_s": "<number>",
    "update_hz": "<accepted-update-count/evaluable-duration>",
    "position_error_sum_m": "<sum-of-evaluable-sequence-means>",
    "position_error_sample_count": "<equals-paired-sequences-after-frozen-missing-output-penalty>",
    "position_error_m": "<sum/sample-count>",
    "lost_track_duration_s": "<number>",
    "evaluable_track_duration_s": "<number>",
    "lost_track_rate": "<lost/evaluable-duration>"
  },
  "csi_only": {
    "sequence_count": "<equals-paired-sequences>",
    "accepted_sensor_frame_count": "<integer>",
    "accepted_update_count": "<integer>",
    "evaluable_duration_s": "<number>",
    "update_hz": "<number>",
    "lost_track_duration_s": "<number>",
    "evaluable_track_duration_s": "<number>",
    "lost_track_rate": "<number>"
  },
  "fused": {
    "sequence_count": "<equals-paired-sequences>",
    "accepted_sensor_frame_count": "<integer>",
    "accepted_update_count": "<integer>",
    "evaluable_duration_s": "<number>",
    "update_hz": "<accepted-update-count/evaluable-duration>",
    "position_error_sum_m": "<sum-of-evaluable-sequence-means>",
    "position_error_sample_count": "<equals-paired-sequences-after-frozen-missing-output-penalty>",
    "position_error_m": "<sum/sample-count>",
    "lost_track_duration_s": "<number>",
    "evaluable_track_duration_s": "<number>",
    "lost_track_rate": "<lost/evaluable-duration>"
  },
  "confidence": {
    "bootstrap_resamples": "<integer-at-least-10000>",
    "bootstrap_seed_list_sha256": "<nonzero-64-hex>",
    "familywise_confidence_level": 0.975,
    "position_error_reduction_lower": "<number>",
    "position_error_reduction_upper": "<number>",
    "lost_track_reduction_lower": "<number>",
    "lost_track_reduction_upper": "<number>"
  },
  "guardrails": {
    "empty_false_track_rate": "<number>",
    "empty_false_track_rate_max": "<frozen-number>",
    "fused_p95_latency_ms": "<number>",
    "fused_p95_latency_max_ms": "<frozen-number>",
    "fused_frame_loss_rate": "<(offered-fused-accepted)/offered>",
    "fused_frame_loss_rate_max": "<frozen-number>",
    "exclusion_fraction": "<number>",
    "exclusion_fraction_max": "<frozen-number>",
    "fused_update_rate_ratio_min": "<frozen-number>",
    "nonwinning_position_error_regression_max": "<frozen-number>",
    "nonwinning_lost_track_regression_max": "<frozen-number>"
  }
}
```

Run the hard arithmetic/provenance gate only after witness review:

```bash
cd harness/ruview
node bin/cli.js nlos verify --repo ../.. \
  --evidence-file evidence/nlos/acceptance.json \
  --require-research-pass
```

Do not commit a fabricated placeholder file merely to exercise this command;
unit tests already cover synthetic fixtures.

## 14. Reporting language

Permitted after software only:

> `SYNTHETIC`: the Rust, Swift, and web contracts pass deterministic replay and
> boundary tests. No live NLOS accuracy or iPhone hardware claim was evaluated.

Permitted after a passing L2 capture:

> `MEASURED` on capture `<digest>` under protocol `ruview-consumer-nlos-v1`, the
> external `<sensor_model>/<transient_kind>` LiDAR-only arm emitted `<rate>` accepted updates/s. Fusion
> reduced `<exact endpoint>` by `<gain>` relative to LiDAR-only over `<n>` paired
> controlled sequences. This is L2 controlled-laboratory evidence for the named
> target/geometry, not built-in iPhone, diffuse-human, field, or safety evidence.

Forbidden:

1. “iPhone sees around corners” from ARKit depth, native/web build or external
   track display;
2. “30 fps” from requested sensor/display rate rather than accepted end-to-end
   live tracks;
3. “25 percent more accurate” when only lost-track rate passed;
4. “hardware validated” without sensor/firmware/capture/ground-truth witness;
5. pooled human/general-scene claims from the retroreflective target stratum; or
6. identity, intent, through-wall optical, collision-avoidance or safety claims.

## 15. Stop and rollback criteria

Stop capture immediately for consent/indicator failure, eye/electrical hazard,
credential compromise, raw-data leak, unapproved person entry, cross-tenant
join, provenance ambiguity, calibration/clock failure or confirmed high/critical
security issue. Quarantine the affected capture and never repair its label.

Stop promotion when H1 fails, neither H2 endpoint passes, fusion guardrails are
materially worse, exclusions exceed the frozen tolerance, confidence analysis
cannot be reproduced, or any required artifact is absent. Roll back to
LiDAR-only, explicit replay, or unavailable. Issue a new protocol version before
recapture; do not tune against and reuse the opened confirmatory set.

## 16. Primary sources

1. Somasundaram et al., [Nature paper](https://doi.org/10.1038/s41586-026-10502-x).
2. [Author manuscript and MAS/particle-filter methods](https://arxiv.org/html/2605.17865v1).
3. [MIT project and reported real-time demonstration](https://cornar.media.mit.edu/).
4. [Upstream implementation and hardware procedure](https://github.com/sidsoma/consumer-nlos).
5. STMicroelectronics, [VL53L8CH histogram interface](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html).
6. STMicroelectronics, [P-NUCLEO-53L8A1](https://www.st.com/en/evaluation-tools/p-nucleo-53l8a1.html).
