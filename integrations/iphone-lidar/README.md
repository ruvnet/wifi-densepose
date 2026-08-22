# RuView iPhone LiDAR

This experimental integration provides the native and browser components needed to use a LiDAR-capable iPhone as a RuView geometry sensor. The native source is type-checked against the iOS SDK in CI; physical-device validation is tracked separately below.

## Architecture

```text
iPhone LiDAR
  -> ARKit sceneDepth
  -> depth + confidence + camera intrinsics + device pose
  -> compact u16 millimeter wire frame
  -> WebSocket relay
  -> browser point cloud
  -> future RuView HAL / fusion ingest
```

The native path is the sensor. The web path is a receiver and visualization surface. Mobile Safari does not expose ARKit scene depth directly to ordinary web pages, so the browser cannot replace the native capture layer on iPhone today.

## Native iPhone path

Create an iOS SwiftUI app target in Xcode, deployment target iOS 17 or newer, then add the files under `native/RuViewLiDAR/` to the target.

Add this Info.plist value:

```xml
<key>NSCameraUsageDescription</key>
<string>RuView uses the camera and LiDAR scanner to capture local depth geometry.</string>
```

Run on a physical LiDAR capable iPhone or iPad. The simulator does not provide LiDAR scene depth.

The app requests `ARWorldTrackingConfiguration` with `.sceneDepth`, checks `supportsFrameSemantics`, extracts `ARDepthData.depthMap` and `confidenceMap`, and never transmits RGB camera frames.

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

The development relay requires an ephemeral token and bounds each WebSocket message, but it is not a production trust boundary. Production integration should terminate the WebSocket inside RuView, authenticate the device using the existing sensor identity path, convert each frame into `ruview-hal::Observation`, and attach witness receipts before fusion or persistence.

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
