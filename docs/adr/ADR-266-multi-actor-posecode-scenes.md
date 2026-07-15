# ADR-266: Multi-Actor PoseCode Scenes

| Field | Value |
|-------|-------|
| Status | Accepted, crate implementation complete |
| Date | 2026-07-15 |
| Deciders | ruv |
| Crate | `wifi-densepose-posecode` |
| Related | ADR-029, ADR-037, ADR-082, ADR-101, ADR-170 |

## Context

RuView already maintains multiple persistent `PoseTrack` aggregates. Each track
contains 17 COCO keypoints, a lifecycle, an AETHER identity embedding and a
stable `TrackId`. The missing layer is a compact, agent-readable representation
of what those tracked people do together over time.

PoseCode 0.1 provides a useful vocabulary for one actor, sequential movement
phases, semantic joint actions, contacts and browser rendering. Its current
protocol does not represent multiple actors, shared world coordinates,
inter-person contact, observation confidence or track provenance.

Encoding every 20 Hz RuView frame as text would also be incorrect. It would
produce 1,200 steps per person per minute, discard observation uncertainty and
couple the sensing hot path to a presentation language.

## Decision

Create the leaf crate `wifi-densepose-posecode` and define a backwards-minded
0.2 scene extension with these invariants:

1. A `Scene` owns named actors and synchronized phases.
2. Each actor can carry the originating RuView `TrackId` and confidence.
3. Every joint target and inter-person contact carries confidence.
4. Actor positions and travel use one room-calibrated three-dimensional frame.
5. Raw `SceneFrame` observations remain separate from compact semantic phases.
6. `PhaseSegmenter` emits a phase on actor membership change, motion settling
   or a bounded maximum duration.
7. Serialization is canonical and deterministic.
8. The parser has hard document, line, actor, phase and target limits.
9. General range-of-motion violations are errors for authored scenes but only
   warnings for observed WiFi scenes. RF pose estimates are not medical joint
   measurements and must not create false safety claims.

The text form is:

```posecode
posecode scene "Assisted squat"
source observed_wifi_csi
actor patient:
  rig humanoid
  pose start = standing
  track 7
  confidence 0.82
  position 0 0 0
actor therapist:
  rig humanoid
  pose start = standing
  track 12
  confidence 0.76
  position 1.2 0 0
step "Lower" 1.5s flow:
  patient.knee_left: flex 95 0.8
  therapist.shoulder_left: flex 30 0.7
  contact therapist.hand_left patient.shoulder_right 0.7
repeat 1
```

## Architecture

`RuViewAdapter` consumes references to existing `PoseTrack` values. It does not
duplicate assignment or identity tracking. It calculates elbows and knees from
three-point interior angles, estimates hip and shoulder sagittal movement in a
calibrated coordinate frame and maps the hip midpoint to actor position.
The adapter is exposed by the crate's `ruview` feature so the protocol, parser
and segmenter remain usable without pulling the full signal and RuVector graph.

The adapter uses keypoint confidence when available. Current trackers that have
not populated that field receive an explicit configurable fallback confidence,
degraded by track staleness. Non-finite coordinates and terminated tracks fail
closed.

`PhaseSegmenter` consumes ordered frames. High-rate frames remain available for
live rendering and evidence storage while the resulting scene records only
meaningful synchronized targets. This preserves both fidelity and readability.

## Security and Resource Bounds

The public parser accepts at most 1 MiB per document and 4,096 bytes per line.
The scene validator defaults to 32 actors, 10,000 phases and 64 joint targets
per actor per phase. All coordinates, angles and confidence values must be
finite. References to undeclared actors are rejected. Durations, repeats and
timeline addition are bounded.

No network, filesystem, model or renderer capability is present in the crate.
It is a deterministic transformation leaf over owned data.

## Consequences

RuView can now produce privacy-preserving multi-person replays, structured fall
and interaction evidence, exercise comparison inputs and agent-readable motion
records without video.

The crate does not solve RF source separation. Its multi-person accuracy is
bounded by the upstream pose model and `PoseTracker`. Contact inference is also
not fabricated; contacts must be observed by a future calibrated classifier or
authored explicitly.

The current COCO skeleton has no toe keypoints, so ankle rotation and precise
foot contact cannot be inferred honestly. Those targets are omitted rather
than guessed.

## Acceptance Criteria

1. A scene containing two actors, simultaneous targets and a cross-actor
   contact parses and serializes deterministically.
2. Unknown actors, non-finite values, reversed timestamps and unbounded inputs
   fail closed.
3. One RuView track converts to finite semantic targets with the same TrackId.
4. Observed out-of-range angles produce warnings, not medical errors.
5. Phase count is smaller than input frame count for a normal motion sequence.
6. Focused crate tests and the complete workspace test suite pass before merge.

## Measured Results

On the implementation container with Rust 1.88, Criterion measured the canonical
two-actor scene parser at 3.40 microseconds and serializer at 3.29 microseconds
per operation. The protocol-only dependency surface is three runtime crates:
`serde`, `serde_json` and `thiserror`.

The protocol-only configuration passes 13 unit tests, documentation tests,
Rustfmt and Clippy with warnings denied. The `ruview` feature passes 14 unit
tests, including native `PoseTrack` conversion, under the repository's RuVector
AVX512-capable build configuration.

## References

1. PoseCode repository and protocol, MIT licensed:
   <https://github.com/posecode-dev/posecode>
2. RuView multi-person pose decision: ADR-037.
3. RuView persistent pose tracking: ADR-029 and `pose_tracker.rs`.
