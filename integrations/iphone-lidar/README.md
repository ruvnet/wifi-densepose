# RuView iPhone LiDAR

This experimental integration provides the native and browser components needed to use a LiDAR-capable iPhone as a RuView geometry sensor. The native source is type-checked against the iOS SDK in CI; physical-device validation is tracked separately below.

## Architecture

```text
iPhone LiDAR
  -> ARKit sceneDepth
  -> depth + confidence + camera intrinsics + device pose
  -> compact u16 millimeter wire frame
  -> WebSocket relay
  -> RuView HAL normalization + signed RuField P1 summary
  -> browser/native sensor-fusion display
```

The native path is the sensor. The web path is a receiver and visualization surface. Mobile Safari does not expose ARKit scene depth directly to ordinary web pages, so the browser cannot replace the native capture layer on iPhone today.

## Native iPhone path

The production mobile client under `ui/mobile` now embeds this capture path as the local Expo module `modules/ruview-lidar`. Its NLOS commissioning panel adds capability checks, live visible-scene point rendering, RoomPlan room geometry, ESP32 pose marking in the same coordinate frame, a SHA-256 calibration artifact, and direct transmission of this integration's `ruview.lidar.depth.v1` packets to the authenticated relay. The relay endpoint and token are held in memory only. The sensing server bounds the packet, normalizes it through `ruview-hal`, emits a signed RuField P1 geometry summary, and drops raw depth bytes.

An optional pose-teaching stage reuses the completed RoomPlan AR session, runs Vision body-pose detection at a bounded rate, and lifts coarse visible joints with the scene-depth buffer from the same `ARFrame`. A hands-up gesture estimates the phone/CSI clock offset; samples are rejected when measured timing residual or association skew exceeds 20 ms. Ten sequences are fixed as seven training and three held-out tests. Only a room-specific student that improves PCK@20cm by at least 25% without increasing lost poses is marked `VALID` and used by the Live overlay. Raw camera, depth, CSI training frames, fingers, and biometric identity are not stored in the resulting model artifact.

The standalone SwiftUI target below remains a focused protocol/reference client and browser-relay test fixture.

Create an iOS SwiftUI app target in Xcode, deployment target iOS 17 or newer, then add the files under `native/RuViewLiDAR/` to the target.

Add this Info.plist value:

```xml
<key>NSCameraUsageDescription</key>
<string>RuView uses the camera and LiDAR scanner to capture local depth geometry.</string>
```

Run on a physical LiDAR capable iPhone or iPad. The simulator does not provide LiDAR scene depth.

The app requests `ARWorldTrackingConfiguration` with `.sceneDepth`, checks `supportsFrameSemantics`, extracts `ARDepthData.depthMap` and `confidenceMap`, and never transmits RGB camera frames. The explicit Live video-overlay option creates a bounded, transient local JPEG preview; it is not included in the relay packet or persisted.

## Browser path

```bash
cd integrations/iphone-lidar/web
npm ci
npm test
npm start
```

The relay prints a random per-run access token. Open the printed browser URL and set the iPhone endpoint to the printed native URL. They have this form:

```text
http://HOST:8787/?token=TOKEN
ws://HOST:8787/ws/lidar?token=TOKEN
```

Set `RUVIEW_LIDAR_TOKEN` to supply the token explicitly. The token only prevents unauthenticated peers from joining the development relay; because `ws://` does not encrypt it, production use requires TLS and `wss://`.

## Wire format

Schema: `ruview.lidar.depth.v1`

Depth is downsampled by 2 in each dimension by default and streamed at a maximum of 15 FPS. Each depth sample is encoded as little endian UInt16 millimeters plus one UInt8 confidence value. `[SYNTHETIC]` Arithmetic sizing reduces the depth payload from roughly 196 KB per 256 x 192 Float32 frame to roughly 37 KB per 128 x 96 frame before base64 and JSON overhead.

`[SYNTHETIC]` At 15 FPS that is approximately 0.75 MB/s after base64 overhead, versus roughly 8 MB/s for uncompressed Float32 JSON at full resolution. These are sizing estimates, not device or network measurements.

## Privacy and governance

The initial implementation labels provenance as `source=live` and `privacyClass=geometry-only`. It sends depth geometry, confidence, camera intrinsics, pose, sequence, and wall clock timestamp. It does not send RGB imagery.

The standalone development relay requires an ephemeral token and bounds each WebSocket message, but it is not a production trust boundary. The integrated RuView endpoint requires an admin-scoped bearer or single-use ticket, converts each valid frame into `ruview-hal::Observation`, and signs the derived RuField summary before fusion.

## Validation status

- `[MEASURED]` The committed Node tests cover wire decoding, malformed inputs, relay authentication, static-file restrictions, and live WebSocket forwarding.
- `[MEASURED]` GitHub Actions type-checks the native sources with strict concurrency against the iOS 17 SDK.
- Physical iPhone capture, end-to-end rendering, confidence-map behavior, and the latency target are not yet measured. A simulator or CI compile does not satisfy the hardware acceptance test.

## Acceptance test

1. Run the relay and browser viewer.
2. Run the native app on a LiDAR capable iPhone.
3. Start LiDAR capture and enable streaming.
4. Move the phone through a room.
5. Verify the browser shows a changing point cloud, sequence increases monotonically, latency stays below the `[CLAIMED target]` of 150 ms p95 on a local WiFi network, and no RGB payload is present in captured WebSocket frames.
