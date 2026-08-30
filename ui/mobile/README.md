# RuView Mobile

RuView Mobile is the iPhone, Android, and web instrument for installing, calibrating, and observing a RuView sensing deployment. It combines local RuView evidence with calibrated WorldGraph topology, optional Cognitum Spaces context, and explicitly authorized Cognitum Meta-LLM interpretation.

The application is source-honest: disconnected sensors remain disconnected, empty cloud responses remain empty, synthetic replay is visibly labeled, and no UI state is promoted to measured evidence without a validated source.

## Capabilities

- **Welcome** — workflow-oriented entry point and system status.
- **Live** — local camera/LiDAR and RuView sensor-fusion display with evidence-level controls.
- **Calibration** — guided room scan, RuView node registration, coordinate alignment, visible-path capture, and calibration validation.
- **Vitals** — measured breathing and heart evidence with a fail-closed Apple Home availability boundary.
- **Zones** — anonymous RF occupancy, calibrated WorldGraph topology, Cognitum Spaces, and consent-gated spatial interpretation.
- **MAT** — governed RuView/WorldGraph incident evidence with explicit source health and no fabricated detections.
- **Settings** — sensing and calibration server configuration, transport diagnostics, privacy controls, and app preferences.

## Cognitum One integration

The mobile client uses two deliberately separate OAuth grants:

| Capability | OAuth client | Scope | Data boundary |
| --- | --- | --- | --- |
| Cognitum Spaces | `ruview` | `sensing:read spaces:read` | Tenant/workspace semantic P2/P3 resources only |
| Meta-LLM | `meta-proxy` | `inference` | User-approved, bounded anonymous spatial summary |

Both flows use Authorization Code with PKCE and Cognitum's supported one-time-code exchange. Native refresh tokens are stored in the OS keychain through Expo SecureStore; access tokens are short-lived. Web sessions use session storage and end with the browser session.

Spaces responses are schema-validated and must preserve the HomeCore Edge boundary. The client rejects a response if it does not explicitly exclude raw CSI, CIR, RF tensors, recordings, pose frames, vital waveforms, and identity observations.

Meta-LLM is off until the operator enables cloud interpretation. It receives only the semantic payload displayed in Zones and returns an OpenAI-compatible completion plus the governed `x_cognitum` routing receipt. Local sensing remains usable when Cognitum is unavailable or unauthorized.

Upstream contracts:

- [Cognitum API](https://github.com/cognitum-one/api)
- [Cognitum Meta-LLM](https://github.com/cognitum-one/meta-llm)
- [RuView](https://github.com/ruvnet/RuView)

## Requirements

- Node.js 20 or newer
- npm
- Xcode 16 or newer for iOS builds
- Android Studio for Android builds
- A RuView sensing server for measured sensing data
- A LiDAR-equipped iPhone or iPad for native room scanning
- Cognitum One account/workspace access for private Spaces or Meta-LLM calls

## Setup

```bash
git clone https://github.com/cognitum-one/ruview-mobile.git
cd ruview-mobile
npm ci
cp .env.example .env.local
npm run ios
```

For a web development build:

```bash
npm run web
```

`EXPO_PUBLIC_DEFAULT_SERVER_URL` is only the default local RuView server address. Cognitum API, authorization, and inference origins are fixed in the governed service implementation and are not accepted from user input.

## Native iOS module

`modules/ruview-lidar` contains the Expo native module for ARKit/LiDAR capture and local Apple Home capability bridging. Native projects are generated artifacts and are intentionally ignored:

```bash
npx expo prebuild --platform ios
npx expo run:ios
```

The app requests camera access for visible room geometry and calibration. It does not claim that consumer LiDAR can reconstruct hidden people, and it does not retain raw camera imagery through this module.

## Validation

```bash
npm run typecheck
npm run lint
npm test -- --runInBand
npm run e2e:web
npx expo-doctor
```

The browser suite exercises welcome navigation, fixed header/footer behavior, calibration, LiDAR gating, live rendering, Vitals, Zones, MAT, settings, and scroll-to-top behavior. Native release validation should additionally build through Xcode and run the physical-device calibration workflow against enrolled RuView hardware.

## Security and privacy invariants

- No bearer token is placed in a URL or persisted in ordinary app settings.
- Cloud interpretation requires a local consent receipt and a separate `inference` authorization.
- Cognitum authorization never grants actuator or write authority to this application.
- Raw sensing and identity-class data do not enter the Spaces or Meta-LLM payloads.
- Malformed, oversized, stale, unauthenticated, or boundary-violating responses fail closed.
- Synthetic evidence is visibly watermarked and never represented as a live measurement.

## License

See the upstream RuView project for applicable licensing and notices. A repository-specific license should be added before public redistribution.
