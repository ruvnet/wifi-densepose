# RuView NLOS for iOS

RuView NLOS is a native SwiftUI beta tester assistant and monitor for authenticated hidden target hypotheses produced by the RuView consumer time of flight pipeline. The first screen guides a tester through a real ARKit visible depth validation before the monitor. It does not claim that Apple LiDAR or ARKit can perform optical non line of sight reconstruction.

The package has two libraries:

| Product | Responsibility |
|---|---|
| `RuViewNLOSCore` | Typed wire model, strict validation, freshness policy, sequence and replay guard, secure endpoint validation, bounded diagnostic model and aggregate phase metrics |
| `RuViewNLOSApple` | Apple capability probe, active ARKit visible depth validator, Keychain credential storage, authenticated WebSocket transport |

`RuViewNLOS.xcodeproj` contains the directly buildable iOS SwiftUI app and links both local package products.

## Evidence boundary

The app consumes JSON envelopes with schema `ruview.nlos.track.v1`. A frame is displayed only after the following checks pass:

1. The UTF 8 JSON frame is no larger than 256 KiB and has at most 16 unique tracks.
2. Every object has exactly the versioned keys. The session identifier, interoperable sequence, algorithm version, provenance strings, hashes, timestamps, vectors, covariance, confidence, entropy, signal quality, and modality contributions are bounded.
3. The expiry is after capture, no more than 5 seconds after capture, still in the future, and capture is no more than 1 second ahead of the local clock.
4. A connection binds to one session and each sequence must increase. A session change requires an explicit reconnect.
5. Live evidence must be at least `l1_measured`, must preserve a raw or compact normalized transient histogram, and cannot use replay transport.
6. `depth_only` provenance can never enter the live NLOS display path.
7. Synthetic evidence must be `l0_synthetic`, use the all zero calibration hash and replay transport, and receives a persistent `SYNTHETIC` watermark.
8. Tracks whose state is `unknown` are never displayed. A stale, malformed, oversized, replayed, or unsupported frame clears the entire current display.

The client also schedules a local expiry for the last accepted frame. If the stream stalls without closing, the visualization is cleared at the envelope deadline. Decode and sequence processing run on a dedicated Swift actor, while the main actor receives only the newest bounded display frame.

## Apple capability boundary

The native probe reports these capabilities separately:

| Apple signal | Public API status | NLOS interpretation |
|---|---|---|
| Scene depth | Probed with `ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)` | Derived visible surface depth only |
| Smoothed scene depth | Probed with `.smoothedSceneDepth` | Derived visible surface depth only |
| Scene mesh | Probed with `supportsSceneReconstruction(.mesh)` | Visible environment geometry only |
| World pose | Probed with `ARWorldTrackingConfiguration.isSupported` | Motion and registration context only |
| Raw photon timing histograms | Reported unavailable | Required from an external supported transient sensor for this pipeline |

The app never upgrades ARKit depth, mesh, or pose into optical NLOS evidence. The beta setup starts an `ARSession` only after the tester presses the start button and grants camera permission. It uses public `ARWorldTrackingConfiguration`, scene depth, camera pose, and tracking state APIs.

## Beta tester flow

The opening setup card links to the [interactive explainer](https://ruview-nlos.ruv.chatgpt.site) and [feedback issue 1690](https://github.com/ruvnet/RuView/issues/1690). The run takes 45 seconds after ARKit begins delivering frames:

1. Press **Start 45 second validation**. No sensor access begins before this action.
2. Grant camera permission. The camera is required by ARKit, but images are never displayed, stored, exported, or uploaded.
3. For 15 seconds, point at a well lit, textured, directly visible surface and move the phone slowly. This calibrates world tracking and visible depth coverage.
4. For 30 seconds, keep a directly visible wall in frame and move slowly from side to side. This tests sustained scene depth and pose stability.
5. Watch live frames per second, depth coverage, ARKit tracking state, movement in metres per second, thermal state, and remaining time.
6. On completion, optionally enable the explicit export consent toggle, prepare the local JSON, and use the iOS share sheet. Share it with issue 1690 only if you choose to do so.

Cancellation and ARKit interruption stop the session, preserve only bounded aggregate phase summaries, and produce a locally shareable cancellation or failure diagnostic. A new run always uses a new random session identifier.

Every validation result has evidence label `direct_depth`. It is visible surface evidence only and never NLOS evidence.

### Diagnostic contract

The JSON is capped at 64 KiB and contains only:

* Random session identifier
* Creation time
* Device model family
* OS and app versions
* Public capability flags
* At most two aggregate phase summaries
* Peak coarse thermal state and camera permission outcome
* Local validation and export consent flags
* Completion status and a bounded failure reason
* The invariant evidence label `direct_depth`
* The invariant physical NLOS status `blocked_raw_transients_unavailable`

It contains no image, raw depth map, camera transform, raw sample, hostname, endpoint, credential, token, or analytics identifier. The app has no upload endpoint. Preparing an export writes only this diagnostic JSON to the app's protected temporary directory so iOS can present its local share sheet.

### Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Scene depth unavailable | Device has no supported LiDAR scene depth API | Use a LiDAR equipped iPhone Pro or iPad Pro and confirm with the capability card |
| Camera permission declined | Permission was denied or restricted | Open Settings, select RuView NLOS, enable Camera, return, and start a new run |
| Tracking says `limited_insufficient_features` | Blank wall, darkness, or too little texture | Include a textured visible object at the wall edge and improve room lighting |
| Tracking says `limited_excessive_motion` | Phone movement is too fast | Move at roughly 5 to 15 centimetres per second |
| Depth coverage is near zero | Reflective, transparent, distant, or poorly lit surface | Use a matte wall within approximately 0.5 to 4 metres |
| Thermal state is serious or critical | Sustained sensing has heated the device | Cancel, let the phone cool for 5 to 10 minutes, then retry without a case |
| Session interrupted | App backgrounded, phone call, or ARKit interruption | Keep the app foregrounded and start a new run |
| Share button is unavailable | Export consent is off or no result exists | Complete or cancel a run, enable the consent toggle, then prepare the JSON |

## Security model

Only `wss` endpoints are accepted. URLs containing embedded user credentials or fragments are rejected, and HTTP redirects are not followed. Pairing tokens must be 32 to 512 visible ASCII characters, are sent in the `Authorization: Bearer` header, and are never placed in the URL or frame body.

On Apple platforms, the token is stored as a generic password with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`. It is not written to `UserDefaults`, logs, source, fixtures, or crash messages. The URL session is ephemeral with cookies and caches disabled. A production endpoint needs a certificate trusted by iOS; this client does not bypass TLS validation or accept self signed certificates.

The visualization is advisory. It must not directly trigger physical actuation or safety critical decisions.

The app has no analytics or position telemetry and its privacy manifest declares no tracking or collected data. ARKit images, depth maps, and poses remain transient in memory and are never persisted. Only aggregate numeric phase summaries can be exported after explicit opt in. Track frames remain in memory only and are replaced by the newest valid frame. Leaving the active foreground disconnects the monitor stream and clears track state; the visualization is also marked privacy sensitive for system snapshots.

## Build and test

Requirements:

1. Swift 5.9 or newer for the package tests.
2. Xcode 15 or newer for local development; Xcode 26 or newer for App Store Connect uploads after Apple's April 2026 requirement.
3. iOS 16 or newer for deployment.

Run the deterministic protocol and security tests on macOS or Linux:

```bash
cd ui/ios-nlos
swift test
```

Build the unsigned simulator app on macOS:

```bash
cd ui/ios-nlos
xcodebuild \
  -project RuViewNLOS.xcodeproj \
  -scheme RuViewNLOS \
  -sdk iphonesimulator \
  -destination 'generic/platform=iOS Simulator' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

For a physical iPhone, open `RuViewNLOS.xcodeproj`, choose a development team and a unique bundle identifier, then build to the device. Complete the opening visible depth validation before configuring the optional monitor. Enter an explicitly provisioned `wss` track endpoint and pairing token. Do not put the token in the endpoint query string.

## Validation limits

A successful Swift test, simulator build, or completed `direct_depth` run is software and visible depth evidence only. It is not evidence that Apple hardware exposes photon timing histograms and it is not a reproduction of the MIT consumer NLOS result. Real NLOS hardware validation requires both an external supported time of flight sensor and captured RuView server output with reviewed calibration and provenance. The simulator normally reports ARKit sensor capabilities as unavailable.
