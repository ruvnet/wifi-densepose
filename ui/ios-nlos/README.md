# RuView NLOS for iOS

RuView NLOS is a native SwiftUI monitor for authenticated hidden target hypotheses produced by the RuView consumer time of flight pipeline. It is deliberately a display and transport adapter. It does not claim that Apple LiDAR or ARKit can perform optical non line of sight reconstruction.

The package has two libraries:

| Product | Responsibility |
|---|---|
| `RuViewNLOSCore` | Typed wire model, strict validation, freshness policy, sequence and replay guard, secure endpoint validation |
| `RuViewNLOSApple` | Apple capability probe, Keychain credential storage, authenticated WebSocket transport |

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

The app never upgrades ARKit depth, mesh, or pose into optical NLOS evidence. The current monitor performs only static capability checks, does not start an `ARSession`, and does not request camera permission.

## Security model

Only `wss` endpoints are accepted. URLs containing embedded user credentials or fragments are rejected, and HTTP redirects are not followed. Pairing tokens must be 32 to 512 visible ASCII characters, are sent in the `Authorization: Bearer` header, and are never placed in the URL or frame body.

On Apple platforms, the token is stored as a generic password with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`. It is not written to `UserDefaults`, logs, source, fixtures, or crash messages. The URL session is ephemeral with cookies and caches disabled. A production endpoint needs a certificate trusted by iOS; this client does not bypass TLS validation or accept self signed certificates.

The visualization is advisory. It must not directly trigger physical actuation or safety critical decisions.

The app has no analytics or position telemetry and its privacy manifest declares no tracking or collected data. Track frames remain in memory only and are replaced by the newest valid frame. Leaving the active foreground disconnects the stream and clears track state; the visualization is also marked privacy sensitive for system snapshots.

## Build and test

Requirements:

1. Swift 5.9 or newer for the package tests.
2. Xcode 15 or newer for the iOS app.
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

For a physical iPhone, open `RuViewNLOS.xcodeproj`, choose a development team and a unique bundle identifier, then build to the device. Enter an explicitly provisioned `wss` track endpoint and pairing token. Do not put the token in the endpoint query string.

## Validation limits

A successful Swift test or simulator build is software evidence only. It is not evidence that Apple hardware exposes photon timing histograms and it is not a reproduction of the MIT consumer NLOS result. Real hardware validation requires both an external supported time of flight sensor and captured RuView server output with reviewed calibration and provenance. The simulator normally reports ARKit sensor capabilities as unavailable.
