# ADR 343: iPhone 17 Pro Spatial Capture, Calibration, Fusion, and External Transient Sensing

| Field | Value |
| --- | --- |
| Status | Proposed |
| Date | 2026 08 24 |
| Owners | RuView maintainers, iOS sensing, NLOS research, web visualization |
| Decision scope | Capabilities 1 through 10 for an iPhone 17 Pro based RuView field node |
| Depends on | ADR 318, ADR 319, ADR 330, ADR 340, ADR 341, ADR 342 |
| Research basis | arXiv 2605.17865, Motion Induced Aperture Sampling for Consumer LiDAR NLOS Imaging |
| Evidence state | Architecture specified. Software, accessory, and physical device validation are pending. |

## 1. Decision summary

RuView will treat the iPhone 17 Pro as a native visible space capture, calibration, inference, and multimodal fusion node. It will not represent the built in Apple LiDAR scanner as a transient sensor or as an optical around the corner camera.

The native application will own ARKit depth, scene mesh, RoomPlan, device pose, optional visible body ground truth, Nearby Interaction ranging, accessory transport, local inference, recording, and release of privacy filtered observations. The web application will own setup, replay, remote monitoring, and Three.js visualization. It will not own hardware capture.

True optical non line of sight reconstruction will require an external transient histogram sensor. The reference production accessory is the ST VL53L8CH because its public product specification exposes normalized histograms. The VL53L8CX named by the research paper remains a reproduction target until the authors' released code and the public driver establish that the required transient representation is available.

The product value is a portable deployment instrument that can:

1. Produce a measured room coordinate system for RuView nodes.
2. Capture visible spatial ground truth without exporting camera images.
3. Align RF, UWB, Bluetooth, LiDAR, and transient evidence.
4. Run privacy preserving inference at the edge.
5. Collect reproducible evidence for whether fusion improves hidden target tracking.

## 2. Context and correction

The referenced paper combines motion, known scene information, and transient LiDAR measurements to synthesize a larger aperture. Its reconstruction depends on time resolved multipath information. Public Apple frameworks expose derived depth, confidence, pose, and mesh products. They do not expose raw SPAD photon counts, picosecond transient histograms, laser timing control, or the internal multipath measurements required to reproduce the paper.

Therefore these are separate claims:

| Claim | Support on iPhone 17 Pro |
| --- | --- |
| Visible line of sight depth | Supported through ARKit on a LiDAR capable device |
| Visible room mesh and semantic room model | Supported through ARKit scene reconstruction and RoomPlan |
| Device pose and inertial motion | Supported through ARKit and Core Motion |
| Passive WiFi CSI capture | Not exposed through a public iOS API |
| UWB range and direction to compatible peers or accessories | Supported through Nearby Interaction, subject to device and session capabilities |
| Bluetooth Channel Sounding to a compatible paired accessory | Potentially supported through Nearby Interaction when the runtime capability reports support |
| Optical NLOS using only the built in LiDAR | Not implementable through public Apple APIs |
| Optical NLOS using an external histogram sensor | Implementable as a RuView accessory research path |

The engineering goal is not to imply that the phone sees through walls. The goal is to make the phone the best available portable truth, calibration, fusion, and presentation node while a separate sensor supplies any transient optical evidence.

## 3. Vocabulary and evidence policy

### 3.1 Evidence labels

Every quantitative result in code, documentation, logs, dashboards, and release notes MUST carry one of these labels:

| Label | Meaning |
| --- | --- |
| MEASURED | Produced by the exact hardware, software revision, scene, and test procedure identified in the evidence record |
| CLAIMED | Reported by Apple, ST, the research paper, or another named source |
| SYNTHETIC | Produced by a simulator, fixture, replay, or generated dataset |
| TARGET | An acceptance threshold that has not yet been demonstrated |

### 3.2 Spatial terms

| Term | Definition |
| --- | --- |
| Apple world frame | The ARKit session coordinate frame |
| Room frame | A stable local frame selected during calibration and stored with a room identifier |
| Node frame | The coordinate frame of a RuView RF, UWB, or transient node |
| Observation | A time stamped measurement with source, units, confidence, covariance, and provenance |
| Hypothesis | An inferred target state that may combine several observations |
| Ground truth | A MEASURED visible reference with a documented error bound, never an unqualified synonym for an ARKit result |

## 4. Specification for capabilities 1 through 10

### 4.1 Capability 1: Live line of sight depth and confidence

#### Public interface

Use `ARWorldTrackingConfiguration` with `frameSemantics` containing `sceneDepth` or `smoothedSceneDepth` after checking `supportsFrameSemantics`. Consume `ARFrame.sceneDepth` or `ARFrame.smoothedSceneDepth` as `ARDepthData`.

#### Required inputs

1. Depth `CVPixelBuffer` in meters.
2. Confidence `CVPixelBuffer` with the Apple confidence level for each pixel.
3. Camera intrinsics and image resolution.
4. Camera transform in the Apple world frame.
5. AR frame timestamp and tracking state.

#### Required output

`VisibleDepthFrameV1` MUST preserve the source resolution, units, confidence, capture timestamp, pose association, calibration version, and evidence label. Derived point clouds MUST be labeled `reconstruction` and MUST NOT be labeled raw LiDAR returns.

#### Runtime behavior

1. Use a dedicated capture actor or serial queue.
2. Hold no more than two unprocessed depth frames.
3. Drop the oldest unprocessed frame when the queue is full.
4. Never block the AR session delegate or main UI thread.
5. Record observed resolution and cadence because Apple does not guarantee a fixed public depth format across devices and operating systems.
6. Stop or downgrade capture on critical thermal state or memory pressure.

#### Privacy

Raw camera images and raw depth frames remain on the phone by default. Export requires an explicit per session user action and a visible recording indicator. The default network product is a redacted point sample, mesh delta, or fused track.

### 4.2 Capability 2: Classified room meshes and RoomPlan output

#### Public interface

Use ARKit `sceneReconstruction` with `meshWithClassification` when supported. Use RoomPlan for room dimensions, detected architectural elements, and USD or USDZ export.

#### Required output

1. `SpatialMeshDeltaV1` for incremental rendering and fusion.
2. `RoomCalibrationV1` for persistent room origin, scale, axis convention, landmarks, and node transforms.
3. Optional RoomPlan artifact for user approved export.

#### Requirements

1. Mesh vertices MUST include the anchor transform and unit definition.
2. Mesh classification MUST be stored as an observation, not an immutable truth.
3. RoomPlan and ARKit mesh coordinates MUST be converted through an explicit transform.
4. A calibration is valid only while its landmark residuals remain within the recorded tolerance.
5. Relocalization, map reset, and changed room geometry MUST invalidate affected transforms.

#### Product use

The room mesh becomes the coordinate reference for node placement, occlusion visualization, relay surface selection, and measurement reporting. It does not make hidden geometry visible.

### 4.3 Capability 3: Continuous device pose and motion

#### Public interface

Use `ARCamera.transform`, `ARCamera.trackingState`, and the AR frame timestamp for visual inertial pose. Use Core Motion only when higher cadence inertial samples are required by an algorithm or accessory synchronization experiment.

#### Requirements

1. Use monotonic clocks internally.
2. Associate every sensor observation with the closest pose and a timestamp skew value.
3. Reject fusion when pose age exceeds the configured budget.
4. Mark output unavailable during initial alignment, relocalization, or unusable tracking.
5. Send Core Motion callbacks to a non main queue and stop updates when the session ends.
6. Do not integrate raw acceleration as a long term position source without visual or external correction.

#### Initial budgets

| Metric | TARGET |
| --- | --- |
| Pose association skew | 10 ms or less for local Apple sensors |
| Accessory association skew | 20 ms or less after clock correction |
| Fusion pause after tracking becomes unusable | 100 ms or less |
| Main thread work caused by pose ingestion | 2 ms or less per display frame |

### 4.4 Capability 4: Visible people and body joints for evaluation

#### Public interface

Use ARKit body tracking on supported devices and operating systems. If the selected AR configuration cannot deliver the required depth and body products together, switch between explicit session modes or use a Vision body pose request as a visible camera fallback.

#### Requirements

1. This capability applies only to a visible consenting participant.
2. Store joints, confidence, frame transform, and timestamps. Do not store identity.
3. Do not perform face recognition or biometric identification.
4. Show an on screen ground truth capture indicator.
5. Delete ephemeral joint data at session end unless the tester opts into an evidence recording.
6. Detect runtime feature compatibility. Never infer simultaneous support from separate API availability.

#### Evaluation role

Visible joints provide a comparison reference before a participant walks behind an occluder. They do not remain visible or become ground truth after occlusion. Hidden estimates retain their RF, UWB, or NLOS provenance.

### 4.5 Capability 5: RuView CSI track fusion in the room frame

The iPhone does not capture WiFi CSI. It receives authenticated observations or tracks produced by RuView nodes and aligns them with the measured room frame.

#### Calibration model

For each node, estimate and store:

```text
T_room_node
covariance_room_node
calibration_method
landmark_residuals
calibration_timestamp
calibration_version
```

The camera pose is represented as `T_room_camera`. A node observation `x_node` is transformed into the room frame as:

```text
x_room = T_room_node * x_node
Sigma_room = J * Sigma_node * transpose(J) + Sigma_calibration
```

#### Fusion model

The first implementation uses an extended Kalman filter for a single approximately linear track and a particle filter for ambiguous or multimodal tracks. A learned model may score observations but MUST NOT erase source provenance or covariance.

```text
P(X_t | O_1:t) proportional to
P(O_lidar_t | X_t) *
P(O_rf_t | X_t) *
P(O_range_t | X_t) *
P(X_t | X_t_minus_1)
```

#### Requirements

1. Reject unknown schema versions, invalid units, stale timestamps, invalid covariance, and unauthenticated senders.
2. Preserve RF observations even when Apple depth is absent.
3. Preserve Apple visible observations even when RF is absent.
4. A fused track MUST identify every contributing source and its last observation age.
5. Never label an RF position as LiDAR verified merely because it is rendered inside a LiDAR mesh.
6. Report fusion residuals and source disagreement.

#### Acceptance direction

The initial gate is relative improvement because a universal absolute error claim is not defensible across rooms. Fusion MUST reduce median position error or lost track rate by at least 25 percent against the best single modality baseline in the same recorded scene. An absolute median error of 0.20 m is a stretch TARGET, not a current claim.

### 4.6 Capability 6: Second generation UWB ranging

#### Public interface

Use Nearby Interaction with `NISession`. Use `NINearbyPeerConfiguration` for a compatible Apple peer and `NINearbyAccessoryConfiguration` for a compatible certified accessory. Exchange discovery tokens only over an authenticated application channel.

#### Supported claim

UWB can measure distance and, when the session and hardware expose it, direction to a participating peer or compatible accessory. It is tagged ranging. It is not passive person sensing and it does not produce WiFi CSI.

#### Requirements

1. Check device and session capabilities at runtime.
2. Expose unavailable, initializing, active, interrupted, invalidated, and suspended states.
3. Support foreground ranging first.
4. Treat background behavior as a separate capability with stricter operating system constraints.
5. Record distance, direction when available, quality, peer identifier alias, timestamp, and session state.
6. Expire ranging evidence immediately when the session invalidates or the peer disappears.

### 4.7 Capability 7: Bluetooth 6 Channel Sounding

#### Public interface

Use Nearby Interaction Bluetooth Channel Sounding only when `supportsBluetoothChannelSounding` is true and the compatible accessory has been paired through the required Apple accessory flow. The exact operating system version, accessory support, and entitlement or program constraints MUST be confirmed during implementation.

#### Fallback order

1. UWB when the target accessory supports it.
2. Bluetooth Channel Sounding when the runtime and paired accessory support it.
3. BLE RSSI for coarse proximity only.
4. Explicit unavailable state.

#### Requirements

1. Do not translate RSSI into a precision distance claim without a scene specific calibrated model and uncertainty.
2. Do not expose a Channel Sounding control when the runtime capability is false.
3. Do not claim access to raw Bluetooth channel measurements unless Apple explicitly exposes them.
4. Keep pairing and device authorization user initiated.

### 4.8 Capability 8: Local inference with Core ML, Metal, and RuVector

#### Decision

Use Swift and a statically linked Rust or C ABI for production native state, transforms, filters, and RuVector integration. Use Core ML or Core AI for supported learned models. Use Metal for projection, reduction, rendering, and compute kernels that show a measured benefit. Use RuVector WebAssembly in the web viewer and parity tests, not as the default native iOS runtime.

#### Rationale

The iPhone 17 Pro has an A19 Pro, a 6 core CPU, 6 core GPU with neural accelerators, and a 16 core Neural Engine. Apple frameworks schedule supported models across CPU, GPU, and Neural Engine. A native library has a clearer App Store execution model and lower bridge overhead than embedding a WebAssembly runtime merely for code reuse.

#### Performance budgets

| Metric | TARGET |
| --- | --- |
| Live presentation cadence | 30 frames per second or better when thermal state is nominal |
| Fusion update p95 | 50 ms or less for 10 active tracks |
| UI main thread work p95 | 8 ms or less per display frame |
| Point cloud visible budget | 100,000 points or fewer on device, adaptively reduced |
| Application memory during standard capture | 500 MB or less |
| Sustained nominal or fair thermal run | 10 minutes or longer |

These are TARGET values. They become MEASURED only after a physical iPhone 17 Pro test with a recorded build, operating system version, and scene.

#### Compute policy

1. Start with clear CPU reference implementations.
2. Add signposts and measure.
3. Move only proven hot paths to Accelerate, Metal, Core ML, or Core AI.
4. Compare every optimized output against the reference implementation.
5. Reduce sampling and rendering quality before dropping sensor integrity.

### 4.9 Capability 9: External transient histogram accessory

#### Decision

Use the ST VL53L8CH as the reference implementable accessory because its public product material specifies normalized histogram output. Maintain a VL53L8CX adapter as a research compatibility target for reproducing the paper after the promised code establishes the required interface.

#### Reference sensor characteristics

The ST VL53L8CH product page CLAIMS 64 zones, up to 128 bins, a minimum bin width equivalent to 37 mm, up to 30 Hz, and I2C or SPI host communication. Documented operating modes include smaller zone and bin combinations at specific rates. The implementation MUST read the exact selected mode from device configuration and MUST NOT hard code marketing maxima as the active format.

#### Accessory topology

```text
VL53L8CH
    |
    | I2C or SPI
    v
Accessory MCU
    |
    | framed histogram protocol
    v
BLE for control and reduced data
or WiFi with TLS for full data
    |
    v
iPhone native accessory client
    |
    v
pose association, aperture accumulation, transient inversion
```

USB C transport may be evaluated later. It is not the first beta path because accessory interoperability and program requirements can increase release risk. WiFi is the preferred full histogram transport. BLE is suitable for pairing, configuration, health, and reduced data modes.

#### Data rate examples

| Mode | Payload estimate using 16 bit bins | Rate estimate before framing |
| --- | --- | --- |
| 64 zones by 18 bins at 15 Hz | 2,304 bytes per frame | about 34.6 KB per second |
| 32 zones by 36 bins at 15 Hz | 2,304 bytes per frame | about 34.6 KB per second |
| 16 zones by 48 bins at 25 Hz | 1,536 bytes per frame | about 38.4 KB per second |
| 64 zones by 128 bins at 30 Hz | 16,384 bytes per frame | about 491.5 KB per second |

The last row is a transport sizing case, not a claim that the sensor supports that zone, bin, and rate combination simultaneously.

#### Accessory security

1. Each accessory has a unique device identity and provisioning secret.
2. Pairing requires physical proximity and an explicit user action.
3. Network transport uses authenticated encryption.
4. Frames include sequence number, capture timestamp, configuration identifier, and integrity protection.
5. Reject replayed, duplicated, out of window, oversized, or malformed frames.
6. Firmware update packages are signed and version checked.
7. No fixed shared fleet password is permitted.

#### Reconstruction boundary

The first goal is to reproduce a documented paper configuration. The paper uses motion induced aperture sampling and reports experiments for reconstruction, tracking, and localization. The system may require two of object shape, object motion, and sensor pose to be known. These assumptions MUST be explicit in each test.

The production UI MUST distinguish:

1. Direct visible return.
2. Transient multipath observation.
3. Candidate hidden target.
4. Temporally maintained track.
5. Fused multimodal track.

### 4.10 Capability 10: Web viewer and operator console

#### Decision

The web application is a privacy filtered viewer, replay tool, setup assistant, and remote stream consumer. It uses Three.js for room mesh, reconstructed point cloud, nodes, tracks, uncertainty volumes, relay surfaces, and provenance overlays.

#### Requirements

1. The browser does not claim direct access to ARKit, Nearby Interaction, or the transient accessory.
2. The native app advertises a versioned capability manifest.
3. Remote streams use authenticated WSS or an equivalent secure channel.
4. Web rendering has adaptive point count and pixel ratio budgets.
5. A point cloud is labeled according to its source: visible reconstruction, transient reconstruction, RF hypothesis, or fused track.
6. The viewer exposes evidence state, last update age, confidence, covariance, and source disagreement.
7. Browser storage MUST NOT retain bearer secrets or raw sensitive captures.
8. Replay files are immutable inputs and do not masquerade as live hardware.
9. The explainer link is available from an information screen: `https://ruview-nlos.ruv.chatgpt.site`.

## 5. Capability matrix

| Number | Capability | Native implementation | Web implementation | Required hardware | Fallback | Hard claim boundary |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Depth and confidence | ARKit capture | Stream and replay | iPhone LiDAR | Unavailable state | Derived depth, not raw transient |
| 2 | Room mesh | ARKit and RoomPlan | Three.js mesh | iPhone LiDAR | Manual room axes | Visible surfaces only |
| 3 | Pose and motion | ARKit and Core Motion | Pose playback | iPhone sensors | Pause fusion | Drift and tracking state apply |
| 4 | Visible body joints | ARKit or Vision | Evaluation overlay | Visible camera view | Manual marker | No hidden joint truth |
| 5 | RF fusion | Receive and fuse RuView tracks | Inspect provenance | External RuView nodes | Single modality | iPhone captures no CSI |
| 6 | UWB | Nearby Interaction | Status and replay | Compatible peer or accessory | Bluetooth or unavailable | Tagged participant only |
| 7 | Bluetooth Channel Sounding | Nearby Interaction when supported | Status and replay | Compatible paired accessory | UWB then RSSI | Runtime support required |
| 8 | Local compute | Swift, Rust, Core ML, Metal | RuVector WASM, Three.js | A19 Pro | CPU reference path | Budgets pending measurement |
| 9 | Optical NLOS | Accessory client and inference | Results and replay | External histogram sensor | Simulation | Built in LiDAR is insufficient |
| 10 | Web console | Secure stream producer | Viewer and control | Any modern browser | Recorded replay | No native sensor capture |

## 6. Versioned data contracts

All contracts use SI units, right handed coordinate frames, monotonic capture timestamps, explicit wall clock mapping when available, and additive versioning. Unknown required fields or unsupported major versions fail closed.

### 6.1 Common observation header

```json
{
  "schema": "ruview.observation.v1",
  "source_id": "opaque-local-id",
  "source_kind": "arkit_depth",
  "sequence": 42,
  "capture_time_ns": 1234567890,
  "receive_time_ns": 1234569999,
  "clock_model_id": "clock-v3",
  "room_frame_id": "room-opaque-id",
  "calibration_id": "cal-opaque-id",
  "evidence": "MEASURED",
  "privacy": "derived-only"
}
```

### 6.2 `VisibleDepthFrameV1`

```text
header
width, height
depth_unit = meter
depth_buffer_reference
confidence_buffer_reference
camera_intrinsics_3x3
T_room_camera_4x4
tracking_state
pose_age_ns
```

### 6.3 `SpatialMeshDeltaV1`

```text
header
anchor_id
revision
operation = add | update | remove
T_room_anchor_4x4
vertices_m
triangle_indices
classification_per_face
```

### 6.4 `DevicePoseV1`

```text
header
T_room_camera_4x4
linear_velocity_mps optional
angular_velocity_radps optional
tracking_state
position_covariance_3x3
orientation_covariance_3x3
```

### 6.5 `BodyGroundTruthV1`

```text
header
participant_alias
visible = true
joints[] = name, position_m, confidence
T_room_body_4x4
retention = ephemeral | evidence-recording
```

### 6.6 `RangingObservationV1`

```text
header
technology = uwb | bluetooth_cs | ble_rssi
peer_alias
distance_m optional
direction_unit_vector optional
rssi_dbm optional
quality
session_state
covariance
```

### 6.7 `TransientHistogramFrameV1`

```text
header
sensor_model
firmware_version
configuration_id
zone_count
bin_count
bin_width_ps or vendor_native_bin_width
histogram_encoding
histogram_payload
ambient_per_zone optional
sensor_temperature_c optional
laser_state
T_room_sensor_4x4
pose_age_ns
integrity_tag
```

### 6.8 `FusedSpatialTrackV1`

```text
header
track_id
state = tentative | confirmed | coasting | lost
position_m
velocity_mps
position_covariance_3x3
contributing_sources[]
source_observation_age_ms[]
fusion_residuals[]
visible_state = visible | occluded | unknown
evidence_level
expiration_time_ns
```

## 7. Pseudocode

### 7.1 Capability negotiation

```text
function probeCapabilities():
    result.depth = ARWorldTracking supports sceneDepth
    result.smoothedDepth = ARWorldTracking supports smoothedSceneDepth
    result.mesh = ARWorldTracking supports mesh
    result.meshClassification = ARWorldTracking supports meshWithClassification
    result.body = ARBodyTracking is supported
    result.uwb = NearbyInteraction reports peer or accessory support
    result.bluetoothCS = NearbyInteraction reports Bluetooth Channel Sounding support
    result.externalTransient = accessory handshake returns compatible schema
    publish result with OS version, hardware model, and application build
```

### 7.2 Capture and normalization

```text
onARFrame(frame):
    if frame.trackingState is unusable:
        fusion.pauseApplePose()
        UI.showTrackingLimited(frame.reason)
        return

    pose = normalizePose(frame.camera.transform, frame.timestamp)
    poseStore.append(pose)

    if depth frame exists:
        if depthQueue is full:
            depthQueue.dropOldest()
            metrics.increment(depthDrops)
        depthQueue.enqueue(depth, confidence, intrinsics, pose)
```

### 7.3 Accessory frame handling

```text
onAccessoryFrame(bytes, authenticatedPeer):
    require authenticatedPeer
    frame = boundedDecode(bytes)
    require supportedMajorVersion(frame.schema)
    require validDimensions(frame.zoneCount, frame.binCount)
    require frame.sequence is newer inside replayWindow
    require integrityTagIsValid(frame)

    pose = poseStore.interpolate(correctClock(frame.captureTime))
    if pose missing or pose.age exceeds budget:
        quarantine frame with reason stalePose
        return

    normalized = normalizeHistogram(frame, pose)
    apertureAccumulator.insert(normalized)
```

### 7.4 Fusion

```text
onObservation(observation):
    require validUnits(observation)
    require finiteCovariance(observation)
    require sourceIsAuthorized(observation.source)

    roomObservation = transformToRoom(observation)
    if calibrationResidual is too large:
        mark calibration invalid
        do not fuse

    hypotheses = tracker.predict(observation.captureTime)
    associations = associate(roomObservation, hypotheses)
    tracker.update(associations, preservingProvenance = true)
    expireTracksBySourceAge()
    publish privacyFilteredTracks()
```

### 7.5 Failure behavior

```text
if Apple tracking is limited:
    stop calibration updates
    retain RF tracks in their own frame
    show limited state

if accessory authentication fails:
    disconnect
    increment security counter without storing payload

if thermal state is serious:
    reduce point rendering
    reduce optional model cadence
    preserve sensor timestamps and recording integrity

if thermal state is critical:
    stop capture safely
    finalize evidence manifest
    show cooldown guidance
```

## 8. Architecture

```text
Apple LiDAR depth and confidence       ARKit mesh and RoomPlan
                |                              |
                +--------------+---------------+
                               |
                         Capture Coordinator
                               |
ARKit pose and Core Motion -----+----- Capability and Quality Gate
                               |
                      Spatial Normalization
                               |
                   Room Calibration and Pose Graph
                               |
RuView RF tracks --------------+-------------- UWB and Bluetooth ranging
                               |
External transient accessory --+-------------- Transient inversion
                               |
                    Provenance Preserving Fusion
                               |
                 RuVector temporal state and WorldGraph
                               |
               +---------------+----------------+
               |                                |
       Native SwiftUI instrument          Privacy filtered WSS
                                                |
                                      Web and Three.js viewer
```

### 8.1 Module boundaries

| Module | Responsibility |
| --- | --- |
| `SpatialCapture` | ARKit configuration, frames, depth, mesh, RoomPlan, tracking state |
| `MotionClock` | Core Motion, monotonic timestamps, clock models, pose interpolation |
| `Ranging` | Nearby Interaction, Bluetooth capability negotiation, RSSI fallback |
| `AccessoryTransport` | Pairing, authenticated transport, bounded frame decode |
| `TransientCore` | Histogram normalization, aperture accumulation, inversion |
| `CalibrationCore` | Landmarks, transforms, residuals, invalidation |
| `FusionCore` | Association, filters, covariance, provenance, expiration |
| `RuVectorBridge` | Temporal memory and similarity operations |
| `WorldGraphBridge` | Persistent room, node, observation, and track relationships |
| `EvidenceRecorder` | Versioned records, manifest, redaction, export |
| `ViewerStream` | Privacy filtered capability, mesh, track, and metric stream |

## 9. Security and privacy specification

### 9.1 Trust boundaries

1. Apple sensor frameworks to native process.
2. External accessory to native process.
3. RuView nodes or server to native process.
4. Native process to local evidence storage.
5. Native process to web viewer or remote service.
6. Imported replay file to viewer and analysis pipeline.

### 9.2 Permission strategy

Ask for permissions progressively at the moment the tester selects a capability. Provide a plain language purpose before the operating system prompt.

| Capability | Permission or declaration |
| --- | --- |
| AR capture and body evaluation | Camera usage description |
| Higher cadence inertial capture | Motion usage description when required |
| RuView node discovery | Local network usage description and Bonjour service declaration if used |
| Bluetooth accessory | Bluetooth usage description and Apple accessory setup flow |
| Nearby Interaction | Nearby Interaction usage description and required capability declarations |

### 9.3 Data minimization

1. Default to derived geometry and tracks.
2. Do not send camera frames off device by default.
3. Do not collect face crops, names, contact data, or stable biometric templates.
4. Use opaque session, room, node, and participant aliases.
5. Default evidence retention to off.
6. Require an explicit tester action to start and stop recording.
7. Display source and recording state continuously.

### 9.4 Secrets and transport

Store tokens and accessory keys in Keychain with an accessibility class suitable for an unlocked interactive application. Use App Transport Security and authenticated TLS. Pinning may be considered only with an operational rotation plan. Never place production secrets in the application bundle, repository, logs, web local storage, or exported evidence.

### 9.5 Import hardening

Replay and accessory inputs are untrusted. Parsers MUST enforce maximum message size, dimensions, allocation count, recursion depth, numeric finiteness, schema major version, and decompression ratio. Fuzz the native and Rust decoders.

## 10. Alternatives considered

| Alternative | Advantages | Costs and risks | Decision |
| --- | --- | --- | --- |
| Built in iPhone LiDAR as the complete NLOS sensor | Simple product story, no accessory | Public APIs lack raw transient data, claim would be false | Rejected |
| Native phone as visible truth and fusion node | Uses supported APIs, good field workflow, privacy can remain local | Does not independently reproduce paper | Selected |
| VL53L8CX accessory immediately | Matches the paper name | Required public raw interface is not yet established | Research adapter only |
| VL53L8CH reference accessory | Public normalized histograms, explicit modes, practical data rate | Different sensor and processing from paper setup | Selected for implementable prototype |
| Browser only application | Easy distribution | Cannot own ARKit or Nearby Interaction capture | Rejected for sensing, selected for viewing |
| Cloud first inference | More compute and centralized updates | Latency, privacy, connectivity, and operating cost | Optional later path |
| On device native inference | Privacy, lower latency, works offline | Thermal and memory limits | Selected default |
| WebAssembly as native iOS core | One artifact across platforms | Runtime and bridge complexity, weaker native integration | Rejected as default; retained for web parity |

## 11. Implementation guide

### Phase 0: Capability probe and claim guardrails

Estimated effort: 2 engineer days.

1. Add a runtime capability screen.
2. Record hardware model, operating system, app build, AR frame semantic support, mesh support, body support, UWB support, Bluetooth Channel Sounding support, and accessory protocol support.
3. Add evidence labels to the shared presentation model.
4. Add copy tests that reject prohibited phrases such as built in optical NLOS.

Exit gate: a tester can export a capability report without granting unrelated permissions.

### Phase 1: Visible depth, mesh, pose, and room model

Estimated effort: 5 engineer days.

1. Implement `SpatialCaptureCoordinator` as an actor.
2. Configure ARKit only after capability checks.
3. Normalize depth, confidence, intrinsics, pose, and tracking state.
4. Build incremental AR mesh storage.
5. Add RoomPlan capture as a separate workflow.
6. Render adaptive visible depth points and mesh in the native interface.

Exit gate: a physical iPhone 17 Pro records a room session whose replay reproduces pose, depth, confidence, and mesh alignment within the recorded calibration tolerance.

### Phase 2: Shared schemas, evidence recorder, and replay

Estimated effort: 3 engineer days.

1. Define versioned Rust or Protocol Buffer schemas for the contracts in section 6.
2. Generate Swift bindings or implement a bounded hand written codec with golden fixtures.
3. Add an evidence manifest containing application build, device model, operating system, sensor modes, clock model, calibration version, and checksums.
4. Add explicit redacted and research recording profiles.
5. Implement deterministic replay.

Exit gate: native and Rust implementations pass the same golden fixtures and reject the same invalid fixtures.

### Phase 3: RuView node calibration and CSI track fusion

Estimated effort: 5 to 10 engineer days.

1. Define authenticated RuView track input.
2. Create a guided three landmark node alignment workflow.
3. Store `T_room_node`, residuals, and covariance.
4. Implement a CPU reference filter.
5. Preserve observation provenance and show source disagreement.
6. Compare RF only, Apple visible only, and fused results against a visible evaluation path.

Exit gate: on the same recorded trajectories, fusion improves median error or lost track rate by at least 25 percent versus the best single modality baseline. Label result MEASURED and attach the dataset manifest.

### Phase 4: UWB and Bluetooth ranging

Estimated effort: 5 engineer days plus accessory integration time.

1. Implement Nearby Interaction session lifecycle.
2. Add authenticated discovery token exchange.
3. Add the Bluetooth Channel Sounding runtime check and paired accessory path.
4. Add UWB, Channel Sounding, RSSI, and unavailable states.
5. Normalize all ranging into `RangingObservationV1`.
6. Test interruption, peer loss, background transition, and permission denial.

Exit gate: a compatible tagged accessory produces distance and available direction with explicit technology and quality. Unsupported devices show a truthful fallback.

### Phase 5: External histogram accessory and NLOS reproduction

Estimated effort: 10 to 20 engineer days after hardware availability.

1. Build the VL53L8CH MCU reference firmware.
2. Implement physical pairing, authenticated control, and WiFi histogram transport.
3. Add frame sequence, time synchronization, configuration identifier, and firmware identity.
4. Build a controlled relay wall and target fixture.
5. Implement paper compatible preprocessing and motion induced aperture accumulation.
6. Begin with retroreflective known shape tracking, then diffuse objects, then unknown motion.
7. Add the VL53L8CX adapter only when the public research implementation identifies the required sensor output.

Exit gate: reproduce one paper task with documented hardware, scene, assumptions, rates, error, and source code revision. Do not claim general around the corner imaging from a single fixture result.

### Phase 6: On device optimization and RuVector memory

Estimated effort: 5 to 10 engineer days.

1. Profile the CPU reference implementation with signposts and Metal tooling.
2. Add RuVector temporal embeddings and trajectory memory through a native bridge.
3. Move proven projection, accumulation, or scoring hot paths to Metal or Core ML.
4. Add quality degradation steps for thermal pressure.
5. Compare optimized output against the deterministic reference.

Exit gate: meet the section 4.8 performance budgets on a physical iPhone 17 Pro or revise the budgets with MEASURED evidence and an ADR amendment.

### Phase 7: Web viewer and remote stream

Estimated effort: 3 to 5 engineer days.

1. Publish a versioned capability manifest and privacy filtered observation stream.
2. Render room mesh, point cloud, nodes, tracks, uncertainty, and evidence state with Three.js.
3. Add adaptive performance tiers.
4. Add deterministic replay and screenshot tests.
5. Link the explainer and testing instructions.

Exit gate: current Safari, Chrome, and iPhone Safari render the same recorded scene without claiming browser sensor capture.

### Phase 8: TestFlight beta and operating procedure

Estimated effort: 5 engineer days plus review time.

1. Add first run permission education.
2. Add guided room calibration.
3. Add evidence export with user review.
4. Add structured tester feedback that includes capability report and optional redacted diagnostics.
5. Produce a release checklist, privacy disclosure, and rollback plan.

Exit gate: three external testers complete setup, scan, calibration, recording, replay, and feedback without developer assistance.

### 11.1 Planned file layout

```text
ui/ios-nlos/App/SpatialCapture/
ui/ios-nlos/App/Ranging/
ui/ios-nlos/App/AccessoryTransport/
ui/ios-nlos/App/Evidence/
ui/ios-nlos/Sources/RuViewNLOSApple/
v2/crates/ruview-nlos-schema/
v2/crates/ruview-nlos-fusion/
v2/crates/ruvector-bridge/
firmware/vl53l8ch-bridge/
ui/mobile/src/nlos/
docs/schemas/nlos/
docs/testing/iphone-spatial-node/
```

Actual paths MUST be reconciled against the repository before implementation. This ADR specifies ownership boundaries, not permission to create duplicate frameworks.

## 12. Verification metaharness

### 12.1 Test layers

| Layer | Purpose | Evidence |
| --- | --- | --- |
| Swift unit tests | Capability state, transforms, time association, permission state, redaction | XCTest report |
| Rust unit and property tests | Schema, covariance, filters, bounded decode | Cargo test report |
| Golden contract tests | Cross language byte and semantic parity | Checked fixtures and hashes |
| Fuzz tests | Accessory, network, and replay parser resilience | Fuzz corpus and crash summary |
| Simulator UI tests | Navigation, denial states, replay, copy | XCUITest screenshots |
| Web component tests | Evidence states and performance tiers | Test report |
| Web end to end tests | Stream, replay, Three.js, responsive UI | Playwright traces and screenshots |
| Physical device tests | ARKit, thermal, camera, depth, RoomPlan | Device and OS manifest |
| Accessory bench tests | UWB, Bluetooth, histogram transport | Firmware and hardware manifest |
| Controlled scene tests | Accuracy, lost track rate, occlusion | Dataset manifest and analysis notebook |

### 12.2 Required synthetic fixtures

1. Static room with known mesh and exact transforms.
2. Slowly moving single target.
3. Crossing targets with ambiguous association.
4. Clock drift and out of order frames.
5. Tracking limited and relocalization sequence.
6. Corrupt histogram dimensions and oversized payload.
7. Accessory replay attack.
8. Missing confidence and invalid covariance.
9. Thermal degradation state changes.
10. Stream disconnect and reconnection.

### 12.3 Physical test scenes

1. Visible room scan with tape measured landmarks.
2. Visible participant walking a measured path.
3. Participant moving behind a wall while RuView nodes continue tracking.
4. UWB tagged accessory moving on the same path.
5. Retroreflective hidden target at a relay wall for transient reproduction.
6. Diffuse hidden target for a harder transient test.
7. Low light, bright ambient light, reflective surfaces, and sparse texture.
8. Ten minute sustained run for thermal and battery characterization.

### 12.4 Test evidence manifest

Every MEASURED result includes:

```text
application commit and build
iPhone model and operating system
accessory model, hardware revision, firmware commit
RuView node models and firmware
room identifier and diagram
calibration procedure and residuals
test procedure version
raw or redacted evidence checksum
analysis code commit
metric definition
known limitations
```

## 13. Observability and performance

Allowed production metrics are operational and aggregate. They include capture cadence, dropped frames, pose age, tracking state, clock skew, calibration residual, message rejection reason, active tracks, source age, fusion residual, render cadence, memory pressure, thermal state, and battery impact.

Production logs MUST NOT contain camera images, depth buffers, histogram payloads, precise room geometry, precise person trajectories, Keychain material, discovery tokens, or stable personal identifiers.

Optimization order:

1. Establish deterministic correctness.
2. Measure producer and consumer cadence.
3. Bound memory and queues.
4. Reduce copies and allocations.
5. Move measured hot paths to vectorized or GPU execution.
6. Reduce optional visualization quality under load.
7. Recheck physical accuracy after every optimization.

## 14. Rollout and rollback

### 14.1 Feature flags

| Flag | Default beta state |
| --- | --- |
| Visible depth capture | On after camera consent |
| Room mesh | On after camera consent |
| RoomPlan export | Off until user selects workflow |
| Body evaluation | Off |
| RF fusion | Off until a room calibration exists |
| UWB | Off until a peer is selected |
| Bluetooth Channel Sounding | Off until runtime and accessory support are confirmed |
| Raw research recording | Off |
| External transient processing | Off until accessory authentication succeeds |
| Remote viewer stream | Off until a destination is approved |

### 14.2 Rollback

Each sensor adapter can be disabled independently without invalidating recorded schema support. If a decoder or fusion defect is found, disable live ingestion, preserve replay for diagnostics, and fall back to visible geometry or single modality tracks. Schema readers remain backward compatible for supported major versions.

## 15. Requirements to evidence mapping

| Requirement | Implementation evidence | Test evidence | Release evidence |
| --- | --- | --- | --- |
| Built in LiDAR is not presented as transient NLOS | Claim guard and source enum | Copy tests and UI screenshots | Release notes |
| Depth preserves confidence and pose | `VisibleDepthFrameV1` | Golden fixture and physical replay | Evidence manifest |
| Mesh is aligned to room frame | Calibration core | Landmark residual test | Calibration report |
| Hidden tracks preserve provenance | Fusion core | Source removal and disagreement tests | Track inspector screenshot |
| iPhone does not claim CSI capture | RF client boundary | Capability report test | Permission and capability documentation |
| UWB and Bluetooth are runtime gated | Ranging capability state machine | Unsupported and interruption tests | Device matrix |
| External frames are authenticated and bounded | Accessory transport | Fuzz and replay tests | Security review |
| Sensitive raw data remains local by default | Redaction profile | Network capture test | Privacy disclosure |
| Performance claims are measured | Evidence labels and manifest | Physical benchmark | Published results with manifest |
| Web is a viewer, not a native sensor | Capability manifest | Browser end to end tests | Explainer and UI copy |

## 16. Acceptance gates

### Gate A: Truthful capability report

A physical iPhone 17 Pro reports the exact available depth, mesh, body, UWB, Bluetooth Channel Sounding, and accessory capabilities. Unsupported features have an understandable state and no active control.

### Gate B: Visible spatial capture

The phone captures depth, confidence, pose, and classified mesh; records a versioned evidence file; and replays the scene with transform and timestamp consistency. Build success or simulator output alone does not satisfy this gate.

### Gate C: Calibration and fusion value

Three RuView nodes are aligned to a measured room. On a held out trajectory, the fused system reduces median position error or lost track rate by at least 25 percent versus the best single modality baseline. The 0.20 m median error goal remains a stretch TARGET until measured.

### Gate D: Ranging truthfulness

A participating accessory produces a technology labeled range observation. Direction is shown only when available. BLE RSSI is labeled coarse proximity unless a calibrated uncertainty model is present.

### Gate E: External optical NLOS

The reference histogram accessory reproduces at least one declared paper task in a documented fixture with explicit assumptions, frame rate, latency, error, and failure cases. The result is not generalized beyond the tested geometry and target class.

### Gate F: Privacy and security

A network inspection confirms that the default session exports no camera frames, raw depth buffers, precise mesh, histogram payloads, or secrets. Parser fuzzing produces no known crash or unbounded allocation. Permission denial leaves the app usable.

### Gate G: Sustained physical performance

A ten minute iPhone 17 Pro session records cadence, p95 fusion latency, memory, battery delta, and thermal state. All published values use the MEASURED label and identify the test manifest.

### Gate H: Web viewer

The web viewer renders the recorded mesh, point reconstruction, uncertainty, nodes, and fused tracks in responsive desktop and mobile layouts. Screenshot and end to end tests verify evidence labels, disconnect behavior, and the explainer link.

## 17. Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Public Apple LiDAR lacks transient output | Built in optical NLOS is impossible | Keep claim boundary and use external histogram accessory |
| VL53L8CX research interface remains unavailable | Paper reproduction stalls | Use VL53L8CH public histograms and isolate sensor adapter |
| Bluetooth Channel Sounding support varies | Feature absent on some systems | Runtime check, capability matrix, UWB and RSSI fallback |
| AR configuration products cannot run together | Missing depth or body output | Explicit session modes and runtime compatibility tests |
| RF baseline error is large | Absolute 0.20 m goal is unrealistic | Require relative improvement first and publish measured distributions |
| Cross device clock skew | Motion aperture and fusion blur | Clock model, timestamp skew, hardware assisted sync where available |
| Thermal throttling | Cadence and latency regress | Adaptive render and model cadence, physical sustained tests |
| Privacy leakage | Tester harm and adoption failure | Local first processing, opt in evidence, redaction, network tests |
| Visual polish hides uncertainty | Misleading operator decisions | Persistent provenance, covariance, evidence state, and stale age |
| Accessory supply or certification delay | Beta schedule slips | Separate visible spatial beta from transient accessory beta |

## 18. Open questions

1. Which operating system minimum provides the required Bluetooth Channel Sounding surface for the target accessory?
2. Does the released paper code use a public VL53L8CX output mode that can be reproduced without vendor restricted interfaces?
3. What RuView track schema and covariance are available from the current CSI pipeline?
4. Which room calibration workflow gives the best tester completion rate: landmarks, fiducials, measured node placement, or a combination?
5. Should research evidence use a local encrypted package, a user selected Files export, or both?
6. Which portion of transient inversion can sustain the target cadence on A19 Pro without critical thermal pressure?
7. What accuracy and lost track baselines are achievable in representative homes before setting an absolute release threshold?

## 19. Reference materials

References were accessed on 2026 08 24. Product and framework behavior MUST be checked again against the deployment operating system and hardware.

### Research

1. Motion Induced Aperture Sampling for Consumer LiDAR NLOS Imaging, arXiv abstract: <https://arxiv.org/abs/2605.17865>
2. Motion Induced Aperture Sampling for Consumer LiDAR NLOS Imaging, HTML paper: <https://arxiv.org/html/2605.17865v1>
3. MIT Media Lab overview: <https://www.media.mit.edu/posts/mit-media-lab-researchers-turn-everyday-lidar-into-an-around-the-corner-camera/>

### Apple hardware and spatial frameworks

1. iPhone 17 Pro technical specifications: <https://www.apple.com/iphone-17-pro/specs/>
2. ARKit: <https://developer.apple.com/documentation/arkit>
3. `ARFrame.smoothedSceneDepth`: <https://developer.apple.com/documentation/arkit/arframe/smoothedscenedepth>
4. ARKit scene reconstruction mesh: <https://developer.apple.com/documentation/arkit/arconfiguration/scenereconstruction/mesh>
5. RoomPlan: <https://developer.apple.com/augmented-reality/roomplan/>
6. `ARBodyTrackingConfiguration`: <https://developer.apple.com/documentation/arkit/arbodytrackingconfiguration/>
7. Core Motion: <https://developer.apple.com/documentation/coremotion>
8. `CMMotionManager` device motion updates: <https://developer.apple.com/documentation/coremotion/cmmotionmanager/startdevicemotionupdates%28using%3Ato%3Awithhandler%3A%29>

### Apple ranging, compute, network, and security

1. Nearby Interaction: <https://developer.apple.com/documentation/nearbyinteraction>
2. `NINearbyPeerConfiguration`: <https://developer.apple.com/documentation/nearbyinteraction/ninearbypeerconfiguration>
3. Third party Nearby Interaction accessories: <https://developer.apple.com/documentation/nearbyinteraction/implementing-spatial-interactions-with-third-party-accessories>
4. Bluetooth Channel Sounding capability: <https://developer.apple.com/documentation/nearbyinteraction/nidevicecapability/supportsbluetoothchannelsounding>
5. Nearby Interaction human interface guidance: <https://developer.apple.com/design/human-interface-guidelines/nearby-interactions>
6. Core Bluetooth discovery and RSSI: <https://developer.apple.com/documentation/corebluetooth/cbcentralmanagerdelegate/centralmanager%28_%3Adiddiscover%3Aadvertisementdata%3Arssi%3A%29>
7. Core ML: <https://developer.apple.com/documentation/coreml>
8. Core AI: <https://developer.apple.com/documentation/coreai>
9. Metal: <https://developer.apple.com/metal/>
10. Network framework: <https://developer.apple.com/documentation/network>
11. `NWProtocolWebSocket`: <https://developer.apple.com/documentation/network/nwprotocolwebsocket>
12. Apple Security framework: <https://developer.apple.com/documentation/security/>
13. Keychain items with Face ID or Touch ID: <https://developer.apple.com/documentation/localauthentication/accessing-keychain-items-with-face-id-or-touch-id>

### External transient sensor

1. ST VL53L8CH product page and public specifications: <https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html>

## 20. Consequences

### Positive

1. Product claims match public API reality.
2. Native and web responsibilities are clear.
3. Visible geometry and tagged ranging can ship before the transient accessory.
4. The sensor adapter boundary allows paper reproduction without coupling the product to one undocumented device.
5. Evidence labeling makes physical validation and investor communication auditable.
6. Privacy preserving local inference remains the default.

### Negative

1. Full optical NLOS requires additional hardware.
2. Two beta tracks are required: visible spatial fusion and external transient research.
3. Cross device synchronization and room calibration add meaningful complexity.
4. Absolute localization accuracy cannot be promised before representative physical tests.
5. Bluetooth Channel Sounding and accessory distribution may introduce program and operating system dependencies.

## 21. Final decision and next action

Adopt the iPhone 17 Pro as the RuView mobile spatial truth and fusion node. Keep optical NLOS behind an external transient sensor interface. Select VL53L8CH as the implementable reference accessory and retain VL53L8CX as a paper reproduction adapter pending public code confirmation.

The next implementation action is Phase 0 followed by Phase 1: ship the capability probe, evidence labels, and physical iPhone depth, confidence, pose, and mesh recorder before adding new fusion claims.

The key risk is not compute capacity. It is truthful access to the required measurement and reliable time and coordinate calibration across heterogeneous devices.

The first acceptance test is a physical iPhone 17 Pro session that captures and replays aligned depth, confidence, pose, and classified mesh with a complete evidence manifest. The first fusion acceptance test is a 25 percent improvement over the best single modality baseline in the same documented room.
