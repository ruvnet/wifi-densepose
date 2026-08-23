# ADR-328: Consumer NLOS raw-transient pipeline and upstream reproduction boundary

| Field | Decision |
|---|---|
| **Status** | Accepted for staged implementation; software contracts may be validated in CI, live-hardware reproduction remains pending until a witness capture passes the protocol |
| **Date** | 2026-08-22 |
| **Owners** | RuView Labs maintainers and sensing research reviewers |
| **Scope** | Commodity optical transient acquisition, calibration, normalization, provenance, upstream reproduction |
| **Extends** | ADR-295, ADR-303, ADR-305, ADR-319, ADR-320 |
| **Related** | ADR-329, ADR-330, ADR-331 |
| **Primary implementation** | `v2/crates/ruview-nlos`, upstream `sidsoma/consumer-nlos`, `harness/ruview` advisory verification |

## Context

Somasundaram et al. introduce motion-induced aperture sampling (MAS) for
consumer time-of-flight LiDAR. Their measurement is not an ordinary depth map.
Each sensor zone records light intensity over time. The strong direct relay-wall
return is followed by much weaker multipath returns whose path lengths constrain
hidden geometry. Multiple frames supply redundant and spatially diverse samples
that improve signal-to-noise ratio and synthesize a larger virtual aperture.

The [Nature paper](https://doi.org/10.1038/s41586-026-10502-x),
[author manuscript](https://arxiv.org/html/2605.17865v1), and
[MIT project page](https://cornar.media.mit.edu/) report 3D reconstruction,
single and multi-object tracking, camera localization, and real-time tracking at
30 Hz. These are upstream `CLAIMED` results until RuView reproduces them with a
named live capture. They are not evidence that Apple exposes the required
measurement to an ordinary iOS application.

The upstream [consumer-nlos implementation](https://github.com/sidsoma/consumer-nlos)
targets ST's P-NUCLEO-53L8A1 research path and captures per-zone histograms. ST
documents the [VL53L8CH](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html)
compact-normalized-histogram interface and configuration-dependent zone/bin/rate
limits. The exact silicon, expansion board, firmware/API and transient format
must be read from and bound to each capture rather than inferred from the kit
name. The
[P-NUCLEO-53L8A1](https://www.st.com/en/evaluation-tools/p-nucleo-53l8a1.html)
combines the expansion board with an STM32 Nucleo host.

RuView needs an explicit boundary because flattening the input to point clouds,
`Observation.value`, ARKit depth, or CSI destroys the delayed transient that the
inverse problem needs. A successful build or replay also cannot validate the
photon path, relay geometry, ambient-light behavior, or target reflectivity.

## Decision

### 1. Reproduce upstream before modifying its inference state

Phase 1 uses upstream commit
`15314de422a765a2d1b72ea7037dfafb2f908d7c` and a documented,
upstream-compatible ST assembly with verified raw or compact-normalized
histogram access. The current reference adapter targets VL53L8CH framing, but
the acceptance record stores the observed board/silicon identity and never
promotes a model name inferred from packaging. RuView records the exact upstream
commit, firmware/API digest, scoped enrollment/certificate reference, configuration, calibration
digest, and capture-manifest digest. Upstream code runs as an isolated research
sidecar or separately reviewed tool; it is not silently vendored into the
production Rust dependency graph.

The first geometry follows the upstream plug-and-play arrangement:

1. a planar relay surface fills the sensor field of view;
2. an opaque occluder prevents direct line of sight to the target;
3. a known rigid retroreflective target supplies the initial high-SNR case;
4. an independent ground-truth system measures the hidden trajectory; and
5. an empty-scene background is captured before each experimental block.

Diffuse rigid objects and people are later strata. They cannot be pooled into
the retroreflective acceptance result. The manuscript notes weaker diffuse
returns and an approximately fourth-power distance falloff in its diffuse model;
the project therefore reports performance by target material and range.

### 2. Preserve a first-class transient frame

`v2/crates/ruview-nlos` owns the initial versioned transient scaffold. The table
below is the capture-manifest/next-contract requirement for live promotion, not
a claim that every field already exists in the current v1 Rust/track schema.
The current scaffold implements bounded histogram/provenance/calibration pieces;
missing clock-domain, certificate, world-frame, configuration, and capture
bindings must land in a reviewed schema before L2. The eventual wire/storage
representation retains, at minimum:

| Field group | Required content | Rejection rule |
|---|---|---|
| Identity | schema version, sensor ID or certificate reference, session ID, sequence | unknown schema, unauthenticated identity, duplicate or regressing sequence |
| Time | sensor monotonic timestamp, host receive timestamp, clock domain and uncertainty | future, stale, non-finite, or unbounded clock error |
| Histogram | zone layout, temporal bin count and width, signed/unsigned count encoding, ambient estimate | zero/oversized dimensions, non-finite values, count overflow |
| Geometry | per-zone ray or relay-wall point, sensor intrinsics, world transform, coordinate-frame ID | non-invertible transform, unit mismatch, incompatible frame |
| Calibration | direct-return peak, mask, background reference, calibration and configuration digests | absent, expired, mismatched, or out-of-distribution calibration |
| Provenance | source `LIVE_HARDWARE`/`REPLAY`/`SYNTHETIC`, firmware and capture digests, evidence level | unknown never becomes live; replay and synthetic are visibly distinct |

The HAL may wrap the frame as a modality-specific payload, but it must not
reduce the histogram to a scalar before NLOS preprocessing. Raw transient input
has its own size/rate limits and parser fuzz surface.

### 3. Deterministic normalization stages

The reference normalization pipeline is ordered and individually testable:

1. verify identity, schema, bounds, sequence, timestamps, and calibration;
2. subtract a compatible empty-scene background;
3. locate and mask the strong one-bounce relay-surface return per zone;
4. align each zone's direct peak to the agreed temporal origin;
5. reject saturated, underexposed, or calibration-incompatible zones;
6. transform time/depth into the light-cone coordinate used by MAS; and
7. emit a normalized transient plus quality flags, never an unconditional track.

Calibration is an explicit state machine: `UNAVAILABLE`, `CAPTURING`, `VALID`,
`DEGRADED`, `EXPIRED`, or `REJECTED`. Only `VALID` calibration can produce a
live optical likelihood. Loss of calibration produces `unknown`, not a cached
or synthetic fallback.

### 4. Separate acquisition evidence from inference evidence

The capture manifest is append-only and content-addressed. It binds the raw
stream, exclusions, firmware, configuration, calibration, clock synchronization,
target/relay-surface stratum, and ground-truth source. Track records reference
the capture and model digests. RuVector or a particle filter may improve
temporal inference, but neither can upgrade acquisition provenance or fabricate
unobserved photons.

Raw captures are local research data by default and are not committed to Git.
Only schemas, small non-person fixtures, checksums, and aggregate metrics may be
reviewed in the repository under the data policy.

## Performance decision

The first live reproduction targets end-to-end track updates at **at least 27
Hz**, a preregistered operational definition of “roughly 30 fps.” The rate is
measured from accepted sensor frame through emitted track, not the configured
sensor clock. Dropped, duplicated, replayed, or late frames are not counted.
The frozen zone/bin/integration configuration must itself be documented and
demonstrated capable of this rate; ST configurations documented at 25 Hz cannot
pass merely because another configuration has a 30 Hz maximum.

Implementation budgets are:

| Stage | Budget and behavior |
|---|---|
| Parser and provenance | bounded allocation; reject before copying oversized dimensions |
| Normalization | one pass over zones × bins; reusable buffers; no unbounded capture queue |
| Tracking | bounded particle count and search volume; overload drops stale work rather than growing latency |
| End-to-end | report p50/p95 latency, update rate, frame loss, CPU and memory with named reproducer |

The paper describes 1,000 particles and a 5 cm proximity prior at 30 Hz. Those
are starting parameters, not RuView guarantees. Optimization must preserve a
golden likelihood/track tolerance and may not hide quality loss behind pooled
throughput.

## Security and privacy

1. USB/serial is the initial acquisition transport. A future network bridge
   requires authenticated sensor identity, encryption, replay protection,
   explicit bind configuration, and a separate threat review. No new
   unauthenticated UDP listener is introduced.
2. Firmware and upstream code are supply-chain inputs. Pin commits and toolchain
   versions, review licenses, scan dependencies, and verify build/capture
   digests. Flashing remains an explicit, confirmed hardware mutation.
3. Malformed frames, NaN/Inf values, integer products, timestamp wrap,
   decompression bombs, calibration substitution, and coordinate transforms are
   fail-closed parser cases.
4. NLOS presence and trajectory data are sensitive even without imagery. The
   approved operator/controller and every required participant receive purpose/
   space/time notice and consent controls, pause/withdrawal and a persistent
   indicator; raw retention is bounded and session track IDs never claim identity.
5. Safety-critical actuation is outside this ADR. A hypothesis never directly
   drives a lock, vehicle, medical device, or emergency decision.

The full attacker/asset analysis is in
`docs/security/consumer-nlos-threat-model.md`.

## Alternatives considered

### Feed ARKit scene depth directly into the MIT algorithm

Rejected. Apple's documented scene-depth surface is processed distance data,
not the per-zone photon-arrival histogram required by the image formation
model. ARKit remains useful for pose and visible geometry under ADR-330.

### Treat CSI as a synthetic optical transient

Rejected. CSI and optical time-of-flight measure different physical channels.
CSI can contribute an independently calibrated likelihood under ADR-329; it
cannot replace the optical measurement or supervise itself.

### Port upstream code to Rust before reproducing it

Rejected for Phase 1. Simultaneously changing hardware, forward model, state
estimator, and language makes failures uninterpretable. The Rust port follows a
pinned upstream baseline and golden captures.

### Vendor all upstream firmware, Python, and data into RuView

Rejected. It increases supply-chain, license, binary-size, and data-governance
risk. The reproduction boundary uses pins, manifests, adapters, and small legal
fixtures instead.

## Consequences

### Positive

1. RuView gains an honest optical-transient modality without corrupting CSI or
   scalar HAL semantics.
2. Upstream reproduction and RuView extensions remain experimentally separable.
3. Provenance, calibration, and failure states survive into every downstream
   hypothesis.
4. The same contract can later support another histogram-capable ToF sensor.

### Costs and limitations

1. Initial use requires an external ST board and ground-truth rig.
2. Raw histograms and CSI increase sensitive data volume and governance burden.
3. Calibration, reflectivity, relay geometry, ambient light, and motion can
   dominate algorithm changes.
4. This decision does not establish through-wall optical sensing, unrestricted
   human reconstruction, identity, or production safety.

## Rollout and rollback

| Phase | Enable condition | Rollback trigger | Rollback action |
|---|---|---|---|
| R0 fixtures | schema/parser/property tests pass | parser panic, unbounded allocation, provenance ambiguity | disable crate feature and retain fixtures only |
| R1 upstream live reproduction | approved protocol, verified histogram interface, external ground truth, and `MEASURED` >=27 Hz accepted-update witness | calibration drift, direct line of sight, saturation, provenance gap | invalidate capture; return to controlled geometry |
| R2 Rust shadow | R1 remains valid and golden-capture agreement is within frozen tolerance | likelihood/track divergence or latency regression | keep upstream sidecar authoritative; disable Rust output |
| R3 candidate live | independent witness plus ADR-331 confidence/security/privacy/guardrail gate | research endpoint fails or privacy controls regress | emit `unknown`; withdraw capability certificate |

Rollback never relabels a failed live capture as a passing replay. Existing CSI,
RuVector, WorldGraph, and other RuView runtime paths remain independently usable.

## Objective acceptance mapping

| ID | Requirement | Evidence |
|---|---|---|
| NLOS-328-01 | Preserve bounded per-zone timing histograms and calibration provenance | Rust round-trip, boundary, fuzz/property, golden-vector tests |
| NLOS-328-02 | Reject stale, duplicate, oversized, unauthenticated, non-finite, and calibration-mismatched input | Negative parser and state-machine tests |
| NLOS-328-03 | Keep `LIVE_HARDWARE`, `REPLAY`, and `SYNTHETIC` mutually exclusive | provenance transition tests and capture manifest review |
| NLOS-328-04 | Reproduce hidden-target tracking at roughly 30 fps | `MEASURED` live capture, external ground truth, >=27 accepted updates/s |
| NLOS-328-05 | Prevent synthetic-only acceptance | `ruview nlos verify --require-research-pass` rejection tests |
| NLOS-328-06 | Make upstream/Rust comparison reproducible | pinned commit, firmware/config/calibration/capture SHA-256 digests |
| NLOS-328-07 | Preserve normal RuView operation when NLOS is absent | workspace tests with crate/sensor disabled; advisory harness reports `ABSENT` |

The authoritative experimental steps and statistical gate are in
`docs/research/consumer-nlos-acceptance-protocol.md`.

## References

1. Somasundaram et al., [“Imaging Hidden Objects with Consumer LiDAR via Motion Induced Sampling”](https://doi.org/10.1038/s41586-026-10502-x), Nature 653, 693–699 (2026).
2. [Author manuscript and methods](https://arxiv.org/html/2605.17865v1).
3. [MIT Consumer NLOS project](https://cornar.media.mit.edu/).
4. [Upstream implementation](https://github.com/sidsoma/consumer-nlos).
5. STMicroelectronics, [VL53L8CH product specification](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html).
6. STMicroelectronics, [P-NUCLEO-53L8A1 evaluation kit](https://www.st.com/en/evaluation-tools/p-nucleo-53l8a1.html).
