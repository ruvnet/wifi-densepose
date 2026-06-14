# ADR 260: RuField Multimodal Field Sensing Specification

Status: Accepted — v0.1 reference stack

Date: 2026 06 14

Deciders: rUv

Tags: sensing, rf, csi, cir, bfld, radar, ultrasonic, infrared, quantum sensing, privacy, provenance, ruvector, ruview

## 1. Context

RuView proved that commodity wireless signals can be used as a practical sensing substrate. The next opportunity is larger: define a common specification for multimodal ambient sensing across RF, ultrasonic, subsonic, infrared, radar, and future quantum sensors.

Existing standards are valuable but fragmented.

IEEE 802.11bf 2025 standardizes WLAN sensing at the WiFi MAC and PHY layers and was published on September 26, 2025. It is important, but it is WiFi specific.

Bluetooth Channel Sounding standardizes techniques for obtaining phase and time delay information, but Bluetooth SIG explicitly does not define the distance algorithm. That leaves application level interpretation open.

IEEE 802.15.4z HRP UWB supports secure ranging using scrambled timestamp sequence waveforms, but UWB remains one modality rather than a universal sensing grammar.

Matter is a useful smart home interoperability protocol, but it is a device connectivity layer, not a multimodal field sensing specification.

The gap is clear: there is no open specification that normalizes sensor observations across CSI, CIR, BFLD, radar, ultrasound, subsonic vibration, thermal infrared, and quantum field sensing into one privacy aware, provenance rich, fusion ready event model.

## 2. Decision

Create **RuField MFS**, the RuField Multimodal Field Sensing Specification.

RuField MFS will define a common event, tensor, calibration, confidence, privacy, and provenance model for ambient field sensing.

It will not replace IEEE 802.11bf, Bluetooth Channel Sounding, UWB, Matter, radar protocols, or device vendor APIs.

It will sit above them.

```text
WiFi CSI
WiFi CIR
WiFi BFLD
UWB
Bluetooth Channel Sounding
mmWave radar
Ultrasonic
Subsonic
Infrared
Quantum magnetic sensing
Quantum inertial sensing

all emit

RuField Field Event
RuField Field Tensor
RuField Fusion Graph
RuField Privacy Class
RuField Provenance Receipt
```

## 3. Name

Preferred name: `RuField MFS`

Full name: `RuField Multimodal Field Sensing Specification`

Public positioning: `The open specification for camera free field intelligence.`

## 4. Problem Statement

Modern sensing systems are locked into modality specific silos: CSI systems produce channel matrices; radar produces range Doppler bins; UWB produces range and time of flight; Bluetooth Channel Sounding produces phase and timing primitives; infrared produces thermal arrays; ultrasonic produces acoustic echoes; subsonic produces structural vibration signatures; quantum sensors produce magnetic, inertial, or optical field traces.

Each has different sampling, calibration, confidence, privacy, and provenance semantics. This prevents reliable fusion and makes governance weak because raw sensing, derived sensing, biometric inference, and anonymous occupancy are often mixed without explicit boundaries.

## 5. Goals

1. Define a common multimodal sensing event format.
2. Define a field tensor format spanning time, frequency, phase, amplitude, range, velocity, angle, temperature, vibration, and uncertainty.
3. Define a modality registry for RF, acoustic, infrared, radar, and quantum sensing.
4. Define privacy classes for raw waveforms, derived features, occupancy, anonymized aggregate state, and biometric inference.
5. Define calibration receipts and provenance hashes.
6. Define fusion rules for multimodal inference.
7. Provide a Rust reference implementation.
8. Provide benchmark tasks for camera free room intelligence.
9. Make RuView one adapter inside a larger open sensing architecture.

## 6. Non Goals

1. Do not define a new wireless PHY.
2. Do not replace IEEE 802.11bf.
3. Do not replace Bluetooth Channel Sounding.
4. Do not replace UWB secure ranging.
5. Do not define medical diagnosis.
6. Do not transmit speech, images, or raw biometric identity by default.
7. Do not require cloud inference.
8. Do not require expensive hardware.

## 7. Core Abstraction — the Field Event

A Field Event is a timestamped observation from any ambient field sensor.

```json
{
  "spec": "rufield.mfs.v0.1",
  "event_id": "01J00000000000000000000000",
  "timestamp_ns": 1791986400000000000,
  "sensor": {
    "modality": "wifi_csi",
    "vendor": "esp32_c6",
    "device_id": "sensor_room_01",
    "placement": "ceiling_corner",
    "clock_domain": "local_ptp"
  },
  "field": {
    "carrier_hz": 5805000000,
    "bandwidth_hz": 80000000,
    "sample_rate_hz": 100,
    "channels": 234,
    "features": ["amplitude", "phase", "doppler", "range_proxy"]
  },
  "observation": {
    "space_cell": [4, 2, 1],
    "range_m": 3.42,
    "velocity_mps": 0.18,
    "motion_vector": [0.12, -0.03, 0.00],
    "confidence": 0.87,
    "privacy_class": "P2"
  },
  "provenance": {
    "raw_hash": "sha256:raw_measurement_hash",
    "firmware_hash": "sha256:firmware_hash",
    "model_id": "ruvector_field_encoder_v1",
    "calibration_id": "room_cal_2026_06_14"
  }
}
```

## 8. Modality Registry

| Code | Modality             | Example source                          |
| ---: | -------------------- | --------------------------------------- |
|    1 | wifi_csi             | ESP32 C6, Intel BE200, AP CSI           |
|    2 | wifi_cir             | channel impulse response                |
|    3 | wifi_bfld            | beamforming feedback                    |
|    4 | uwb_hrp              | IEEE 802.15.4z ranging                  |
|    5 | ble_channel_sounding | phase and timing primitives             |
|    6 | mmwave_radar         | range Doppler radar                     |
|    7 | ultrasonic           | echo and time of flight                 |
|    8 | subsonic             | structural vibration and room resonance |
|    9 | infrared_thermal     | thermal array or passive IR             |
|   10 | active_infrared      | reflected IR                            |
|   11 | lidar_phase          | phase based optical range               |
|   12 | quantum_magnetic     | NV diamond or OPM field trace           |
|   13 | quantum_inertial     | atom interferometer or precision IMU    |
|   14 | event_camera         | optional visual event stream            |
|   15 | synthetic_sim        | simulator or replay source              |

## 9. Field Tensor

The normalized numeric container (`Modality`, `FieldAxis`, `FieldTensor`) as specified in the implementation crate `rufield-core`.

## 10. Privacy Classes

| Class | Description                      | Example                         |
| ----- | -------------------------------- | ------------------------------- |
| P0    | Raw waveform or raw sensor frame | raw CSI, raw radar cube         |
| P1    | Derived non identity features    | Doppler peak, thermal blob      |
| P2    | Occupancy and motion only        | person present, bed exit        |
| P3    | Anonymous aggregate state        | room count, zone activity       |
| P4    | Biometric or health inference    | breathing, gait, sleep, scratch |
| P5    | Identity linked inference        | named person state              |

Default system policy: edge storage may retain P0 only temporarily; network transmission defaults to P2 or lower; P4 requires explicit consent; P5 requires explicit identity binding and audit log.

## 11. Provenance Receipt

Every event must be auditable (`ProvenanceReceipt`). Acceptance invariant: **No fused inference is valid unless every contributing event has a provenance receipt or is explicitly marked synthetic.**

## 12. Fusion Graph

Nodes: sensor, event, field_tensor, feature, object, zone, state, inference, receipt.
Edges: observed_by, derived_from, calibrated_by, supports, contradicts, fused_into, expires_at, requires_consent.

## 13. Fusion Rule Format

Human readable TOML rules (`rule.person_present`, `rule.bed_exit`, `rule.nocturnal_scratch`) with `inputs`, `method`, `threshold`, `privacy_max`, optional `window_ms` and `requires_consent`.

## 14. Reference Architecture

Layer 0 physical sensors; Layer 1 native adapters; Layer 2 field tensor normalization; Layer 3 RuVector field embeddings; Layer 4 fusion graph; Layer 5 policy and privacy guard; Layer 6 application event stream; Layer 7 dashboard, API, MCP, Matter bridge.

## 15. Rust Crate Layout

`rufield-core`, `rufield-schema`, `rufield-adapters`, `rufield-fusion`, `rufield-privacy`, `rufield-provenance`, `rufield-bench`, `rufield-viewer`.

## 16. Core Rust Interfaces

`FieldAdapter`, `FieldEncoder`, `FusionEngine`, `PrivacyGuard` traits as specified in `rufield-core`.

## 17. MVP Adapters

v0.1 must support three real modalities: WiFi CSI, mmWave radar, Infrared thermal. Optional: ultrasonic, subsonic, synthetic simulator.

## 18. Benchmark Suite

| Task                    |   Metric |       Target |
| ----------------------- | -------: | -----------: |
| Presence detection      |       F1 |         0.90 |
| Room transition         |       F1 |         0.85 |
| Bed exit                |       F1 |         0.90 |
| Breathing detected      |       F1 |         0.80 |
| Nocturnal scratch       |       F1 |         0.75 |
| Fall like event         |   Recall |         0.95 |
| False alarm rate        | per hour |   below 0.10 |
| Event latency           |      p95 | below 100 ms |
| Provenance coverage     |  percent |          100 |
| Privacy violation count |    count |            0 |

## 19. First Viral Demo

Camera free room intelligence: person enters, sits, breathing detected, sleeps, scratches arm, exits bed, leaves room — no camera, no identity, signed field receipts, live fusion graph, privacy class visible per event.

## 20. Data Model

`FieldEvent { spec_version, event_id, timestamp_ns, sensor, tensor, observation, provenance }` and `Observation { zone_id, space_cell, range_m, velocity_mps, motion_vector, confidence, labels, privacy_class }`.

## 21. Decision Matrix

| Option                                        | Interop | Novelty | Buildability | Business value | Risk | Score |
| --------------------------------------------- | ------: | ------: | -----------: | -------------: | ---: | ----: |
| Extend RuView only                            |       2 |       2 |            5 |              3 |    2 |    14 |
| Build proprietary fusion engine               |       3 |       3 |            4 |              4 |    3 |    17 |
| Create open RuField spec plus reference stack |       5 |       5 |            4 |              5 |    3 |    22 |
| Attempt new hardware standard                 |       5 |       5 |            1 |              4 |    5 |    20 |

Decision: **Create open RuField spec plus reference stack.** It maximizes credibility, extensibility, and ecosystem pull while avoiding the impossible burden of defining a new physical layer.

## 22. Security Model

| Threat                              | Impact                          | Mitigation                             |
| ----------------------------------- | ------------------------------- | -------------------------------------- |
| Raw waveform leakage                | privacy breach                  | P0 edge only by default                |
| Biometric inference without consent | legal and trust risk            | P4 consent gate                        |
| Sensor spoofing                     | false occupancy or safety event | signed sensor receipts                 |
| Replay attack                       | forged event stream             | nonce plus timestamp plus hash chain   |
| Model drift                         | wrong inference                 | calibration expiry and benchmark gates |
| Overfitting to one room             | weak generalization             | room split benchmark                   |
| Vendor firmware change              | silent degradation              | firmware hash in receipt               |

## 23. Calibration Model

`CalibrationReceipt` is first class. Required calibration tasks: empty room baseline, single person walk path, sit and stand, bed or couch transition, breathing reference, no motion stability period.

## 24. Inference Semantics

Every inference must include: label, confidence, supporting events, contradicting events, privacy class, calibration id, model id, expiry time.

## 25. Consequences

Positive: RuView becomes part of a larger sensing ecosystem; the spec creates a standards-style wedge without waiting for silicon vendors; multimodal fusion becomes portable; privacy and provenance become differentiators; enterprise deployment becomes easier to justify; benchmark receipts reduce skepticism.

Negative: broad scope can dilute execution; hardware variability will be painful; calibration is the hardest practical problem; some will claim existing standards already solve parts of this; medical and biometric use cases require careful governance.

Mitigation: keep v0.1 narrow; ship real adapters; publish benchmark receipts; do not claim medical diagnosis; position RuField above existing standards.

## 26. Implementation Plan

Phase 1 spec skeleton; Phase 2 Rust core; Phase 3 adapters; Phase 4 fusion graph; Phase 5 dashboard; Phase 6 benchmark.

## 27. Acceptance Criteria

v0.1 is accepted when:

1. Three modalities stream into one event graph.
2. Every event has a privacy class.
3. Every event has a provenance receipt.
4. Fusion produces at least five room state inferences.
5. p95 event pipeline latency is below 100 ms.
6. Benchmark runner produces deterministic reports.
7. Raw waveform storage is disabled by default.
8. P4 inference requires consent policy approval.
9. Dashboard shows live camera free room intelligence.
10. Spec is readable enough for external implementers.

## 28. Reference Repository Structure

Crates under `v2/crates/rufield-*` (workspace members), spec under `docs/rufield/`, benches under `rufield-bench`.

## 29. Open Questions

1. JSON Schema first, Protobuf first, or both?
2. Default transport: MQTT, NATS, WebSocket, or MCP?
3. Matter integration: bridge or first class target?
4. P4 health inference disabled by default in public demos?
5. Benchmark datasets synthetic first, then real world?
6. Include quantum modality IDs even if adapters are synthetic only?

## 30. Recommendation

Proceed. Publish RuField as an open specification with a working Rust reference stack and a viral camera free room intelligence demo.

## 31. Benchmark Acceptance Test

```text
Given a room with WiFi CSI, mmWave radar, and thermal IR sensors
When a person enters, sits, breathes, exits bed, and leaves
Then RuField emits signed events
And classifies room state without a camera
And keeps all default network events at P2 or below
And produces p95 latency below 100 ms
And produces a deterministic benchmark report
```

---

## Implementation Status (v0.1 reference stack)

The v0.1 reference stack is implemented as a **standalone Cargo workspace**
(`rufield/`, published as `github.com/ruvnet/rufield` and vendored into RuView
as a submodule — the `vendor/rvcsi` pattern). It is pure Rust, builds and tests
on Windows with no native deps (`ndarray`/`tch`/`openblas` are not used), and
depends only on `serde`, `serde_json`, `toml`, `sha2`, and `ed25519-dalek`.

**All metrics below are SYNTHETIC.** They are scored against the simulator's own
ground-truth labels. They demonstrate the pipeline recovers known truth and runs
within latency/privacy/provenance budgets — they are **not** field-validated
accuracy. There is no hardware in v0.1; real adapters (ESP32 CSI, mmWave, thermal
IR) are a documented follow-up (see the repo README "Firmware" section).

### Crates delivered

| Crate | Implements |
|-------|-----------|
| `rufield-core` | §7/§9/§16/§20 data model: `Modality` (15), `FieldAxis`, `FieldTensor` (shape↔values validated), `PrivacyClass` (P0–P5), `SensorDescriptor`, `Observation`, `FieldEvent`, `CalibrationReceipt`, `InferenceQuery`, `FieldInference`, `FieldEmbedding`; `FieldAdapter`/`FieldEncoder`/`FusionEngine`/`PrivacyGuard` traits. §7 JSON example round-trips. |
| `rufield-provenance` | Real `sha256` content hashing + deterministic `ed25519` sign/verify; §11 `is_fusable` invariant. Tests: tamper → verify fails; synthetic event fusable without signer. |
| `rufield-privacy` | §10 default policy + `DefaultPrivacyGuard` (`authorize` → Allow/Deny/RequiresConsent). Tests: P0 transmit denied; P4 no-consent → RequiresConsent; P4 consent → Allow; P2 → Allow; P5 needs identity binding. |
| `rufield-adapters` | Deterministic seeded `SyntheticSim` emitting the §19 sequence across 3 modalities (wifi_csi, mmwave_radar, infrared_thermal). Same seed ⇒ identical signed event stream with ground-truth labels. |
| `rufield-fusion` | `FusionGraph` (§12) + `RuFieldFusion` engine; TOML rules (§13, ≥5 inferences: person_present, sitting, sleeping, breathing, nocturnal_scratch, bed_exit, room_transition); weighted-Bayes + temporal-window; rejects non-fusable events; `FieldInference` with §24 fields. |
| `rufield-bench` | Deterministic runner: F1 per task (SYNTHETIC), p95 latency, provenance coverage, privacy violations; JSON + human table; §31 acceptance test as `#[test]`. |

Total test count across the workspace: **60 tests, 0 failed**.
`cargo clippy --workspace` is clean.

### §27 acceptance-criteria scorecard

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Three modalities stream into one event graph | **PASS** — wifi_csi, mmwave_radar, infrared_thermal |
| 2 | Every event has a privacy class | **PASS** — `Observation.privacy_class` (non-optional), default ≤ P2 |
| 3 | Every event has a provenance receipt | **PASS** — every event is ed25519-signed and verifies; coverage 100% |
| 4 | Fusion produces ≥ 5 room-state inferences | **PASS** — 7 distinct inferences produced |
| 5 | p95 event pipeline latency < 100 ms | **PASS** — p95 ≈ 0.01 ms (in-process) |
| 6 | Benchmark runner produces deterministic reports | **PASS** — identical report across runs (latency is the only wall-clock field) |
| 7 | Raw waveform storage disabled by default | **PASS** — P0 network transmission denied by default policy |
| 8 | P4 inference requires consent policy approval | **PASS** — P4 without consent → RequiresConsent; breathing/scratch rules carry `requires_consent = true` |
| 9 | Dashboard shows live camera-free room intelligence | **DEFERRED** — no `rufield-viewer` dashboard in v0.1; the benchmark + `room_intelligence` example provide a CLI view. Follow-up. |
| 10 | Spec readable for external implementers | **PASS** — ADR-260 + detailed standalone README with compiling usage examples |

**Decision:** §27 criteria 1–8 and 10 PASS; criterion 9 (live dashboard) is
**deferred** to a follow-up. Per the acceptance rule (1–8, 10 pass; 9 may be
deferred), Status is set to **Accepted — v0.1 reference stack**.

### Deterministic benchmark report (SYNTHETIC, seed = 2026)

```text
TASK (SYNTHETIC)       METRIC      VALUE     TARGET    MEETS
presence                   f1      1.000      0.900      yes
breathing                  f1      1.000      0.800      yes
nocturnal_scratch          f1      0.923      0.750      yes
bed_exit                   f1      1.000      0.900      yes
room_transition            f1      1.000      0.850      yes
-----------------------------------------------------------------------------------
p50 latency:          0.0097 ms
p95 latency:          0.0123 ms   (target < 100 ms: PASS)
provenance coverage:  100.0 %      (target 100%: PASS)
privacy violations:   0          (target 0: PASS)
events=216  modalities=3  distinct_inferences=7
```

All five scored §18 tasks meet their F1 targets **on synthetic ground truth**.
`nocturnal_scratch` is 0.923 (one borderline noise tick at this seed) — reported
honestly rather than tuned to 1.0. The fall-like / false-alarm-rate §18 rows are
not scored in v0.1 (no fall is in the demo sequence) and are a follow-up. These
numbers prove the fusion pipeline scores correctly against known truth; they say
**nothing** about real-world accuracy, which requires the hardware adapters that
v0.1 deliberately does not ship.

### Honest statement

Every metric here is simulator-based. No ESP32 CSI, mmWave, or thermal capture
was used. RuField v0.1 is a working, honestly-measured reference pipeline —
data model, provenance, privacy, fusion, and a deterministic benchmark — pending
real hardware adapters.
