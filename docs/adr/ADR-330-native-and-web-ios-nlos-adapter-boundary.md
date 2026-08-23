# ADR-330: Native and web iOS NLOS adapters and Apple API boundary

| Field | Decision |
|---|---|
| **Status** | Accepted; native and web software surfaces are implementable now, direct built-in iPhone transient-NLOS remains unsupported until a documented Apple API and live device evidence exist |
| **Date** | 2026-08-22 |
| **Owners** | RuView Labs iOS, mobile web, API, security, and sensing maintainers |
| **Scope** | Swift package/app, ARKit context adapter, authenticated track transport, web replay/live UI, capability claims |
| **Depends on** | ADR-295, ADR-305, ADR-319, ADR-328, ADR-329 |
| **Related** | ADR-034, ADR-035, ADR-331 |
| **Implementation** | `ui/ios-nlos`, `ui/mobile`, `ruview.nlos.track.v1` |

## Context

The phrase “smartphone-grade LiDAR” describes a performance/cost class; it does
not guarantee that an App Store process can read every internal sensor signal.
The MIT technique needs each SPAD zone's photon-arrival histogram. Apple
documents ARKit APIs for processed
[`sceneDepth`](https://developer.apple.com/documentation/arkit/arframe/scenedepth),
[`smoothedSceneDepth`](https://developer.apple.com/documentation/arkit/arframe/smoothedscenedepth),
world tracking, and
[`sceneReconstruction`](https://developer.apple.com/documentation/arkit/arworldtrackingconfiguration/scenereconstruction).
Apple's [scene-depth point-cloud sample](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth)
shows how applications request and unproject processed depth. These public
surfaces do not document the per-zone transient histogram used by the MAS
measurement model.

This is an API assessment, not a claim about undisclosed Apple hardware or
future operating systems. It must be reviewed against official documentation
for each supported iOS/Xcode release. Until the needed signal is documented and
validated on a physical device, RuView must not present ARKit depth as an MIT
NLOS reproduction.

The user still needs two useful iOS surfaces:

1. a native app that compiles a shared contract, uses public ARKit depth/pose for
   visible context, and consumes external RuView NLOS tracks; and
2. a web-capable mobile UI that receives authenticated tracks or deterministic
   replay without pretending to capture the phone's LiDAR.

## Decision

### 1. Publish one contract with explicit capability levels

All iOS surfaces consume `ruview.nlos.track.v1`. Capability is an enum, not
inferred from device marketing:

| Capability | Meaning | Permitted label |
|---|---|---|
| `unavailable` | no valid track source | unavailable/unknown |
| `replay` | deterministic fixture or recorded stream | `SYNTHETIC` or `REPLAY`, persistently watermarked |
| `arkit_context` | public ARKit pose/depth/mesh only | line-of-sight context; never NLOS |
| `external_live` | authenticated live track from histogram-capable external pipeline | live external NLOS, subject to evidence/certificate |
| `apple_transient_live` | future documented raw-transient Apple adapter | disabled until separate ADR, API proof, device witness, and ADR-331 gate |

Unknown values fail closed. UI copy names the actual source and evidence level.
It never shortens `external_live` to “iPhone sees around corners.”

### 2. Keep the Swift core platform-neutral

`ui/ios-nlos/Package.swift` defines:

1. `RuViewNLOSCore`, a pure Swift contract/validation library that can be unit
   tested without ARKit; and
2. `RuViewNLOSApple`, an Apple-only adapter behind `canImport(ARKit)` and runtime
   availability/capability checks.

The direct iOS app and shared `RuViewNLOS` scheme use the same validated model.
The current v1 core decoder rejects unknown/excess JSON shape, non-finite or
bounded-range numeric values, invalid covariance diagonals, expired/future
tracks, excessive arrays, duplicate IDs, and illegal provenance/evidence
transitions. It has no world-frame or capture-manifest field yet; those are
live-promotion contract requirements, not claims about this G0 scaffold.

The current Apple adapter is deliberately a static capability probe: it does
not start an `ARSession`, request camera permission, or capture/export any depth.
It documents the public-API boundary and supports the authenticated external
track client. A later, separately reviewed line-of-sight context adapter may
export camera pose, intrinsics, visible scene depth/confidence, smoothed depth,
and mesh metadata with explicit `arkit_context` labeling. Neither implementation
synthesizes photon histograms or changes context into an NLOS evidence source.

### 3. Treat the native app as a client of the external NLOS pipeline

The production data path is intended to be an authenticated, versioned RuView
track endpoint, not direct access to the ST board from UI code. The current v1
client verifies:

1. TLS and the configured RuView service identity;
2. a scoped, revocable pairing token or short-lived session authorization;
3. authenticated server session, envelope session, and schema bindings;
4. monotonic sequence and bounded clock skew;
5. track expiry, covariance, modality contributions, and calibration hash.

L2/live promotion additionally requires tenant/workspace authorization,
coordinate-frame and capture-manifest bindings, scoped enrolled sensor identity,
and a valid witness/capability certificate. Those fields are not silently
inferred from v1.

Network loss, app suspension, sensor unavailability, decode error, and stale
data immediately degrade the capability
and clear or visually expire the live track. Cached data is never silently live.

The initial Swift client further pins concrete bounds: `wss` only; no embedded
URL credentials or fragments; redirects refused; ephemeral URLSession state; a
256 KiB maximum message; an exact bounded JSON model; sequence no larger than
the JavaScript safe integer and strictly increasing within a bound session; at
most a 5 second track lifetime; and pairing tokens of 32–512 visible ASCII bytes
stored as `WhenUnlockedThisDeviceOnly` Keychain data. `SYNTHETIC` frames require
replay transport and the reserved zero calibration hash. Live provenance must
retain a raw/CNH transient kind; a nonzero calibration hash is enforced at L2
calibrated and above, not for every L1 live envelope. These are wire/client
safety rules, not proof that the stream is physically honest.

### 4. Web iOS consumes tickets and replay; it does not capture transients

The NLOS surface in `ui/mobile` implements the same schema and freshness rules.
Live mode first obtains an authenticated, short-lived, single-purpose ticket
from `/api/v1/nlos/ws-ticket`, then connects to the NLOS stream. Long-lived
OAuth tokens and credentials are not placed in query strings, logs, local
storage, or replay files. The current server binds the ticket to its server
session, a 30-second expiry, and one use; the client pins the returned WSS URL
to the configured same authority. Tenant/workspace/audience/origin claims are
required before L2 deployment but are not present in the v1 ticket schema.

Deterministic replay is a first-class developer/demo mode. It is visibly marked
`SYNTHETIC` throughout the view, uses fixed seeds/fixtures, cannot update live
spatial memory, and cannot satisfy the research gate. Bounded exact-key
validation and same-authority ticket checks are implemented. Deployment CSP,
server-enforced origin policy, and reconnect backoff remain required hardening
before live promotion.

The web view visualizes plan/perspective geometry, covariance/quality,
freshness/expiry, modality/provenance, and disconnected/degraded state. It does
not call an undocumented WebKit or ARKit bridge.

Both clients accept at most 1,000 ms of future clock skew and a 5,000 ms
envelope lifetime. The web profile is intentionally stricter under receive
silence: it clears a frame after 1,500 ms without a replacement even when the
publisher supplied a longer TTL. Native clears at the signed envelope expiry.
This conservative display-liveness difference does not alter wire acceptance,
evidence level, or research metrics.

### 5. Keep pose/context fusion separate from evidence fusion

ARKit pose may help align the phone display or a separately calibrated external
sensor. That transform is valid only when extrinsic calibration, timestamps,
coordinate conventions, and uncertainty pass. Scene depth may display the
relay wall or visible geometry. It is not added to the optical NLOS likelihood
unless a future reviewed model defines and evaluates that factor.

If the phone and external sensor are not rigidly mounted, phone pose cannot be
treated as sensor pose without an independently measured time-varying transform.
The UI may still display both frames separately.

### 6. Re-evaluate Apple support through a documented gate

At each major iOS/Xcode intake, a reviewer searches Apple's official SDK headers,
documentation, entitlements, privacy manifest requirements, and App Store rules.
Direct Apple transient support advances only if all are true:

1. a public supported API exposes timing histogram/count data with documented
   units, dimensions, timestamps, and device support;
2. use requires no private symbols, jailbreak, reverse engineering, or hidden
   entitlement;
3. a physical-device capture proves that the data includes usable multipath;
4. privacy/security review and user disclosure pass; and
5. the same live reproduction/fusion protocol passes under a new adapter ADR.

No marketing article, simulator API, depth-map correlation, or successful
compile satisfies this gate.

## Performance decision

| Surface | Software target | Measurement note |
|---|---|---|
| Swift core | decode/validate without blocking the main actor; bounded memory | benchmark representative max-size track frames |
| Native rendering | newest-frame policy; 30 fps-capable presentation where device allows | rendering rate is not sensing rate |
| Web validation/store | bounded per-message parsing and no unbounded history | reject oversized messages before state update |
| Web rendering | responsive recent-track visualization and backoff under loss | requestAnimationFrame rate is not NLOS update rate |
| Transport | p50/p95 server-to-view latency, reconnects and dropped/stale counts | report separately from sensor-to-track latency |

UI optimization may decimate display history, but it cannot decimate or reorder
the evidence record. Performance tests use `SYNTHETIC` fixtures and are labeled
software evidence only.

## Security and privacy

1. NLOS tracking can reveal a person outside direct view. Native and web apps
   require explicit consent, purpose text, a persistent indicator, pause/stop,
   and clear source/provenance. Background capture is disabled by default.
2. App Transport Security/TLS and short-lived scoped authorization are required.
   Debug cleartext/local exceptions are not release defaults.
3. WebSocket messages are untrusted. Enforce maximum message size, schema,
   numeric bounds, sequence, freshness, origin, rate, tenant, and coordinate
   frame before rendering or storage.
4. Replay files contain no credentials and no raw person/CSI/transient data by
   default. Fixtures are synthetic or approved/de-identified and immutable.
5. Native and web telemetry excludes positions, raw sensor frames, tokens,
   calibration secrets, and stable person identifiers. Diagnostics use bounded
   counters and redacted errors.
6. The display is advisory. It cannot directly actuate a device or certify that
   a hidden region is safe.

## Alternatives considered

### Make the first milestone a direct iPhone NLOS app

Rejected. It makes the research depend on an undocumented measurement. The
external sensor proves the architecture independently while iOS remains a
transport, context, and presentation adapter.

### Use only a native app

Rejected. A web/mobile view lowers review and demo friction, exercises the
versioned protocol, and can run deterministic fixtures. It still must not claim
direct sensor access.

### Use only a web app and skip native Swift

Rejected. ARKit pose/depth and physical-device capability checks require the
native SDK. The pure Swift core also gives an independent decoder implementation.

### Embed a permanent bearer token in the app or WebSocket URL

Rejected. Tokens leak through logs, browser history, proxies, crash reports,
and screenshots. Use short-lived, scoped, one-use tickets.

### Treat replay as a transparent fallback during disconnect

Rejected. It would misrepresent stale/synthetic state as live. Replay is an
explicit operator mode with persistent labeling and separate state.

## Consequences

### Positive

1. Native and web iOS deliver useful live/replay experiences without blocking
   the core research on Apple's API choices.
2. The same schema, freshness, provenance, and coordinate rules apply across
   Rust, Swift, and TypeScript.
3. Public ARKit pose and visible geometry remain valuable but honestly scoped.
4. Future Apple transient access has a precise, reviewable activation gate.

### Costs and limitations

1. The first live iOS experience needs an external histogram sensor and RuView
   host; the phone alone is not the sensing system.
2. Cross-language contract tests and release CI add maintenance.
3. Web presentation depends on an authenticated RuView backend; offline mode is
   replay only.
4. Simulator and Linux Swift tests cannot validate ARKit, LiDAR hardware,
   App Store behavior, or a physical iOS build.

## Rollout and rollback

| Phase | Enablement | Rollback trigger | Action |
|---|---|---|---|
| I0 | pure Swift/TypeScript contract and deterministic fixtures | decoder disagreement, unbounded input, provenance drift | disable view; fix contract/golden vectors |
| I1 | future separately reviewed ARKit context on supported device | permission/session/transform failure | capability `unavailable`; clear context |
| I2 | authenticated external live tracks with valid, unexpired evidence/capability certificate and privacy approval | stale/replay/auth/tenant/evidence mismatch or approval withdrawal | disconnect; clear live state; retain explicit replay option |
| I3 | future Apple transient adapter | any of five activation gates absent/regressed | remove capability certificate and adapter flag |

The NLOS tab/app can be removed or disabled without affecting core RuView CSI,
Rust sensing, memory, MCP, or orchestration.

## Objective acceptance mapping

| ID | Requirement | Evidence |
|---|---|---|
| NLOS-330-01 | Swift core builds/tests without ARKit | `cd ui/ios-nlos && swift test` on supported Swift host |
| NLOS-330-02 | Native iOS scheme builds with public APIs | macOS `xcodebuild` simulator gate and physical-device smoke evidence, separately labeled |
| NLOS-330-03 | Current Apple capability probe, absent a new adapter ADR, never emits raw-transient/live-NLOS provenance | capability/provenance unit tests and source review |
| NLOS-330-04 | Web contract accepts valid `ruview.nlos.track.v1` and rejects malformed/stale/oversized input | Jest/TypeScript boundary tests |
| NLOS-330-05 | Live web transport uses short-lived authenticated tickets | client/server integration and replay/origin/expiry negative tests |
| NLOS-330-06 | Replay remains persistently `SYNTHETIC` and cannot pass live gate | UI/store tests plus harness research rejection |
| NLOS-330-07 | Missing Xcode/hardware or NLOS backend degrades honestly | advisory verifier skips and offline/disconnect UI tests |
| NLOS-330-08 | Web and native builds remain separable from core runtime | build matrix with NLOS surfaces disabled/absent |
| NLOS-330-09 | Future Apple activation uses only documented public access | official SDK/header diff, entitlement scan, privacy/App Store review, device witness, and separate adapter ADR |

## References

1. Apple, [`ARFrame.sceneDepth`](https://developer.apple.com/documentation/arkit/arframe/scenedepth).
2. Apple, [`ARFrame.smoothedSceneDepth`](https://developer.apple.com/documentation/arkit/arframe/smoothedscenedepth).
3. Apple, [`ARWorldTrackingConfiguration.sceneReconstruction`](https://developer.apple.com/documentation/arkit/arworldtrackingconfiguration/scenereconstruction).
4. Apple, [Displaying a point cloud using scene depth](https://developer.apple.com/documentation/arkit/displaying-a-point-cloud-using-scene-depth).
5. Somasundaram et al., [consumer-NLOS measurement model](https://arxiv.org/html/2605.17865v1).
6. STMicroelectronics, [VL53L8CH raw compact normalized histogram interface](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8ch.html).
