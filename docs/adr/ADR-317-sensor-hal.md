# ADR-317: RuView sensor HAL — abstract all sensing hardware to one Observation type

- **Status**: Accepted — initial implementation (ADR-297 phase 2)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: hal, sensor-abstraction, ontology, fusion, adapters, category, phase-2

## Context

This ADR is primitive 20 of the perception-substrate program (ADR-297) and a
phase-2 integration primitive; it is authored as **Proposed**. In the ADR-297
DAG it **consumes the canonical spatial ontology (ADR-303)** — its output is an
ontology `Observation` bound to a `Sensor` entity — and **feeds real sensor
fusion (ADR-308)**, which resolves many observations into one world state. It
closes the "identify the hardware" clause of the ADR-297 acceptance test that
phase 1 leaves open.

RuView's strategic ceiling is set by how tightly it is coupled to WiFi CSI.
Every new modality today lands as a bespoke ingest path with its own frame
shape, its own provenance handling, and its own place in the pipeline. That is
the difference between "a WiFi-DensePose project" and "an open
spatial-intelligence operating layer": the category changes the moment *any*
sensing hardware — {CSI, 802.11bf, BLE, UWB, mmWave, acoustic, camera, lidar,
IMU, custom} — enters through one abstraction and becomes one `Observation`
feeding one world model.

Crucially this is a *unification*, not a green field. Adapters already exist and
must be reused, not rebuilt:

- ADR-279's native RF frame contract (`RfFrameV2`) already unifies ESP32,
  Intel, Atheros, PicoScenes, Realtek radar, and 320 MHz 802.11bk producers as
  `RfFrameV2` producers into a shared latent — "lightweight per-device adapters
  into a shared latent, not a shared tensor." The HAL generalizes that lesson
  beyond RF.
- Existing CSI adapters (ESP32/Nexmon/FeitCSI paths), the mmWave fusion path
  (ADR-063), and the multistatic WiFi path (ADR-029) are concrete producers to
  bring under one trait.
- ADR-302 already authenticates a `Sensor`/`DeviceId`; ADR-303 already defines
  `Sensor`, `Observation`, `Track`, and `Event` as first-class node types. The
  HAL is the trait that turns a heterogeneous device into that authenticated
  `Sensor` emitting those `Observation`s.

The gap is a single **`SensorHal` trait and one `Observation` type** that every
modality implements, so the world model never sees a modality-specific frame —
only a provenance-bearing, evidence-labelled `Observation`.

## Options considered

1. **Continue adding per-modality ingest paths.** Rejected: O(modalities) bespoke
   pipelines, each re-encoding provenance and evidence, each a place the ladder
   can be dropped — and it keeps RuView categorically a WiFi project.
2. **Force every modality into the ADR-274/279 RF tensor/frame.** Rejected: the
   ADR-279 lesson is precisely that premature canonicalization discards
   information (bandwidth, antenna structure, phase). A camera, lidar, or IMU
   has no meaningful `RfFrameV2` projection; forcing one is the same mistake at a
   larger scale.
3. **Define a `SensorHal` trait producing one `Observation` type, with existing
   adapters as implementations feeding a shared latent and the ADR-303
   ontology.** Chosen.

## Decision

Introduce a **`SensorHal` trait** and a single **`Observation`** type. Every
sensing modality is an implementation of the trait; the world model consumes
only `Observation`s.

### 1. The `SensorHal` trait

- A `SensorHal` describes a device's **capabilities** (which phenomena it can
  sense — reusing the ADR-302/ADR-141 `CapabilityAttestation`), its **native
  frame** (kept native, not canonicalized, per the ADR-279 shared-latent
  lesson), and a method that lifts a native frame into an `Observation`.
- Implementations wrap the existing producers: CSI (ESP32/Nexmon/FeitCSI via the
  ADR-279 `RfFrameV2` path), 802.11bf (ADR-307, phase 2), BLE, UWB, mmWave
  (ADR-063), acoustic, camera, lidar, IMU, and `custom`. RF modalities reuse the
  ADR-279 per-device latent adapters wholesale; the HAL adds the non-RF and
  ranging modalities under the same trait.
- The trait is the boundary where untrusted hardware input is validated
  (CLAUDE.md: validate at every hardware/FFI boundary; default to least
  authority). A device is authenticated as an ADR-302 `Sensor` before its
  observations are trusted.

### 2. The `Observation` type

- One provenance-bearing `Observation`: a measurement plus its `SensorHal`
  source descriptor, its ADR-302 authenticated `DeviceId`, its ADR-292
  `SourceState`, its native-frame reference (not a lossy projection), and
  exactly one `EvidenceLevel` (L0–L5, ADR-282). A camera-derived `Observation`
  and a CSI-derived `Observation` are the same type with different provenance —
  and a camera observation never lifts WiFi output to camera-grade; each carries
  its own honest evidence level (CLAUDE.md: never present WiFi sensing as
  camera-grade).
- The `Observation` maps directly onto the ADR-303 ontology `Observation` node
  attached to its `Sensor`, so the ontology is the one representation and the
  HAL is its ingest funnel.

### 3. Feeding fusion

- Observations from any set of modalities flow into ADR-308 fusion, which
  resolves them into one probabilistic world state. The HAL guarantees fusion
  never sees a modality-specific frame — only `Observation`s with uniform
  provenance and evidence — which is what makes ADR-308's "many observations →
  one world state" invariant implementable across heterogeneous hardware.

### Category and honesty discipline

- This ADR changes RuView's category from a WiFi-DensePose pipeline to an open
  spatial-intelligence operating layer, but it makes **no accuracy claim**: the
  HAL delivers a uniform ingest boundary, not a detector. Any capability of a
  newly-connected sensor is still gated by ADR-299 and certified by ADR-315 for
  its specific environment — connecting a camera does not grant a validated
  capability by itself.
- Hardware support for a given modality is CLAIMED until demonstrated on real
  silicon with captured evidence per CLAUDE.md; a passing trait test proves the
  abstraction, not a fielded device.

## Consequences

- New sensing hardware lands as one `SensorHal` implementation instead of a
  bespoke pipeline; the translation matrix stays O(modalities), mirroring how
  ADR-303 collapsed the surface matrix.
- The ADR-297 acceptance clause "identify the hardware" becomes implementable:
  a new sensor type is described by its HAL, authenticated as an ADR-302
  `Sensor`, calibrated (ADR-298), gated (ADR-299), and certified (ADR-315)
  through the same phase-1 spine, closing the last open clause.
- A trait boundary and an `Observation` type are added; existing RF adapters
  are re-expressed as implementations rather than rewritten, preserving the
  ADR-279 native-frame/shared-latent design.
- Non-RF modalities (camera, lidar, acoustic) enter the governed plane with the
  same provenance and privacy discipline as RF; a camera is not a privacy-free
  shortcut — it inherits the ADR-277 governance and its own evidence level.
- As a phase-2 Proposed ADR, the trait shape may be revised as ADR-308 fusion
  and ADR-307 802.11bf land; that revision is expected for a phased program.

## Validation

- `cargo test` on the HAL crate (design-time, Proposed) — a fixture `SensorHal`
  for each of at least two modalities (CSI via ADR-279, plus one non-RF)
  produces uniform `Observation`s; every `Observation` carries a `DeviceId`,
  `SourceState`, native-frame reference, and exactly one `EvidenceLevel`; a
  synthetic source yields L0/`Synthetic` and cannot alias to measured
  (ADR-279 invariant 6); an unauthenticated device's observations are rejected
  at the trait boundary (ADR-302).
- Cross-ADR: an `Observation` maps round-trip to an ADR-303 ontology
  `Observation` node with no provenance loss, and a set of `Observation`s from
  distinct modalities is accepted by an ADR-308 fusion fixture.
- Real-silicon evidence is required before any modality's hardware support is
  claimed beyond CLAIMED: a captured boot/runtime log from the real device
  emitting `Observation`s. A successful build or simulator run is not hardware
  evidence (CLAUDE.md).
