# ADR 340: iPhone LiDAR Sensor Bridge

Status: Proposed

## Context

RuView needs a low cost mobile geometry sensor that can contribute calibrated spatial observations without coupling the perception substrate to Apple frameworks.

ARKit exposes rear LiDAR scene depth through `ARFrame.sceneDepth` and `smoothedSceneDepth` on supported devices. Ordinary mobile web pages do not receive this ARKit depth surface directly, so native capture and web visualization must be separated.

## Decision

Use a two layer architecture.

1. Native Swift and ARKit perform acquisition.
2. A modality neutral wire frame transports geometry into browser tools and, next, the RuView HAL.

The native client will capture depth, confidence, camera intrinsics, and world tracking pose. RGB imagery is excluded from the default transport.

The protocol identifier is `ruview.lidar.depth.v1`.

Depth samples are quantized to UInt16 millimeters for transport. Confidence remains UInt8. The default sender downsamples by two spatially and caps transmission at 15 FPS. Full fidelity depth remains available locally for future on device inference.

## RuView integration boundary

The transport must not become a second world model. The production receiver converts each packet into the canonical `ruview-hal::Observation`, then passes it through authenticated sensor identity, provenance, OOD gating, uncertainty aware fusion, spatial memory, and WorldGraph adapters.

Rules:

1. `source=live` is valid only for frames produced by an active ARKit session.
2. Sequence numbers are monotonic per sensor session.
3. Wall clock timestamp is separate from ARKit monotonic frame timing.
4. RGB is off by default and requires an explicit higher privacy capability.
5. Browser clients consume geometry but are not treated as authoritative sensors.
6. Unsupported devices fail closed rather than substituting simulated depth.

## Performance target

`[SYNTHETIC]` A 256 x 192 Float32 depth map is about 196 KB before confidence and metadata. Downsampling to 128 x 96 and encoding each sample as two byte depth plus one byte confidence yields about 36.9 KB raw. At 15 FPS the raw sensor payload is about 553 KB/s. Base64 raises this to roughly 737 KB/s before JSON metadata. These values are arithmetic sizing estimates, not device measurements.

The `[CLAIMED target]` for local network latency is below 150 ms p95. A later binary WebSocket or QUIC transport can remove base64 overhead; the exact end-to-end reduction must be measured before it is claimed.

## Security

The development relay is LAN-facing, requires a random per-run bearer token, bounds message size, and restricts the files it serves. Its default `ws://` transport is not encrypted, so it is not a production trust boundary.

Production requires WSS, authenticated sensor identity, replay protection, message size limits, per tenant authorization, provenance receipts, and explicit retention policy before persistence.

## Consequences

Benefits include commodity hardware, metric depth, tracked camera pose, rapid room scanning, calibration support for RF sensing, and a practical ground truth source for RuView experiments.

The main limitation is that Apple provides processed scene depth rather than the underlying raw transient LiDAR waveform. Therefore this implementation supports direct geometry and sensor fusion now, but does not reproduce research systems that require raw multipath time of flight transients for non line of sight reconstruction.

## Acceptance criteria

A physical LiDAR capable iPhone must stream live geometry to the browser viewer with monotonically increasing sequence numbers, no RGB payload, valid confidence maps, and below 150 ms p95 local network latency over a 60 second run.

CI type-checking and simulator runs do not satisfy this criterion. Until a captured physical-device run records the environment and results, the hardware behavior and latency remain unverified.
