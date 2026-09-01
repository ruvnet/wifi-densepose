# ADR-341: Consumer NLOS beta tester delivery, setup assistant, and bounded diagnostics

| Field | Decision |
|---|---|
| **Status** | Proposed; release governance and CI dry gates are defined, TestFlight and physical-device evidence remain operator-gated |
| **Date** | 2026-08-23 |
| **Owners** | RuView Labs iOS, release, privacy, security, and research maintainers |
| **Scope** | TestFlight delivery, post-install setup, capability checks, calibration and wall-scan workflow, diagnostic package, observability, rollback |
| **Depends on** | ADR-295, ADR-305, ADR-318, ADR-319, ADR-328 through ADR-331, ADR-340 |
| **Implementation** | `ui/ios-nlos`, `.github/workflows/consumer-nlos-ci.yml`, `docs/research/consumer-nlos-beta-test-protocol.md` |

## Context

Beta testers need to install RuView NLOS without Xcode, understand what their
device can actually measure, complete a short test, and return actionable
feedback. Calling this an “installer” hides two different responsibilities:

1. **TestFlight is the distribution and update mechanism.** Apple controls app
   installation, signing, tester invitations, build expiry, and update delivery.
2. **The RuView setup assistant starts after installation.** It checks device
   support, requests narrowly scoped permissions, guides calibration and a wall
   scan, labels the resulting evidence, and creates an optional diagnostic.

RuView must not ship an ad hoc IPA installer, enterprise signing workaround, or
credential-bearing upload workflow. Those paths add device registration or
certificate custody, create revocation risk, and still cannot overcome the core
sensor limitation: public ARKit exposes processed scene depth, not the raw
transient histograms required by the consumer NLOS reconstruction in ADR-328.

The product value of the beta is therefore evidence collection, onboarding
quality, direct line-of-sight LiDAR characterization, and external NLOS client
validation. It is not proof that an iPhone alone sees around corners.

## Specification

### Outcome and actors

The outcome is a governed beta path that a nontechnical tester can complete in
under five minutes without Xcode. Actors are the tester, App Store Connect
release operator, privacy/security reviewer, research owner, and support
maintainer.

### Inputs

1. A TestFlight build produced from a reviewed commit.
2. A supported iPhone or iPad and current OS permitted by the build.
3. Tester consent and only the permissions needed by the selected test mode.
4. Optional short-lived authenticated RuView endpoint configuration for an
   external NLOS track source.
5. A local session identifier generated per run and not derived from an Apple
   account, advertising identifier, device serial, or person identity.

### Outputs

1. A terminal setup state with an explicit capability label.
2. A 15 second calibration result and a 30 second wall-scan result when the
   device supports public ARKit scene depth.
3. A local summary showing frame count, effective frame rate, depth coverage,
   pose stability, thermal state, permission state, and failure codes.
4. An opt-in diagnostic JSON document no larger than 65,536 bytes.
5. A tester-controlled path to copy or share the diagnostic and open the
   RuView explainer and feedback issue.

### Constraints and invariants

1. The full guided path targets less than 300 seconds from first launch to
   diagnostic-ready state. Calibration is 15 seconds and wall scan is 30
   seconds, each measured by monotonic time.
2. Raw camera images, depth maps, point clouds, CSI, transient histograms,
   audio, precise location, access tokens, and stable person identifiers are
   never included in diagnostics.
3. No raw capture is uploaded by the beta assistant. Diagnostic export is
   explicit, inspectable, and initiated by the tester.
4. The diagnostic is rejected locally if canonical UTF-8 JSON exceeds 64 KiB,
   contains a forbidden key, has non-finite numbers, or violates the schema
   contract below.
5. `physical_nlos` remains `blocked_raw_transients_unavailable` unless a future
   public sensor adapter passes ADR-330's activation gate and ADR-331's live
   research gate. ARKit scene depth can report `direct_depth` only.
6. Permission denial, unsupported hardware, thermal pressure, interruption,
   stale frames, or transport failure must produce a recoverable degraded state,
   never synthetic substitution presented as live data.
7. TestFlight publication requires human authorization and App Store Connect
   credentials held by Apple or the approved release environment. Pull request
   CI never reads signing or App Store Connect secrets.
8. A CI build, unsigned archive, simulator run, or TestFlight installation is
   software evidence only. Physical LiDAR behavior requires a named device run.

### Exclusions

This decision does not authorize App Store release, enterprise distribution,
remote device management, background sensing, raw-data collection, safety
actuation, medical use, identity inference, or an iPhone-only NLOS claim.

### Success criteria

1. At least 90 percent of recruited compatible-device testers complete the
   guided path in under five minutes in a pilot of at least 20 participants.
2. Median completion time is at most 180 seconds and p95 is at most 300 seconds.
3. Every exported diagnostic is at most 64 KiB and contains zero forbidden raw
   fields in automated and manual review.
4. Permission denial and unsupported-device tests reach useful guidance in at
   most two taps after the failure is detected.
5. Zero releases are uploaded from pull request CI and zero long-lived release
   credentials are stored in repository workflows.

## Pseudocode and state transitions

### State model

```text
NOT_INSTALLED
  -> TESTFLIGHT_INSTALLED
  -> CONSENT_REQUIRED
  -> CAPABILITY_CHECK
     -> UNSUPPORTED
     -> PERMISSION_REQUIRED
        -> PERMISSION_DENIED
        -> CALIBRATING
           -> CALIBRATION_FAILED
           -> WALL_SCAN_READY
              -> WALL_SCANNING
                 -> SCAN_INTERRUPTED
                 -> SUMMARY_READY
                    -> DIAGNOSTIC_PREVIEW
                       -> EXPORTED
                       -> DISCARDED
```

`UNSUPPORTED`, `PERMISSION_DENIED`, `CALIBRATION_FAILED`, and
`SCAN_INTERRUPTED` are explicit terminal states for the current attempt, but
each offers a bounded retry or support path. A retry creates a new session ID
and does not merge evidence across attempts.

### Setup control flow

```text
on_first_launch:
  show purpose, privacy boundary, evidence labels, and explainer link
  require affirmative consent before requesting sensor permissions
  transition CONSENT_REQUIRED -> CAPABILITY_CHECK

check_capability:
  inspect public runtime capability APIs
  if scene depth unsupported:
    label capability = unavailable
    physical_nlos = blocked_raw_transients_unavailable
    transition -> UNSUPPORTED
  else:
    transition -> PERMISSION_REQUIRED

request_permission_just_in_time:
  request camera permission for ARKit session only
  request local-network permission only if external_live mode is selected
  do not request precise location for the default beta workflow
  if denied:
    record only permission status and bounded reason code
    transition -> PERMISSION_DENIED

run_calibration:
  start monotonic timer for 15 seconds
  aggregate frame count, valid-depth ratio, pose deltas, interruption count
  retain no frame payload after aggregate update
  if minimum quality or continuity rule fails:
    transition -> CALIBRATION_FAILED
  else:
    transition -> WALL_SCAN_READY

run_wall_scan:
  start monotonic timer for 30 seconds
  aggregate the same bounded metrics plus thermal state samples
  drop raw frame immediately after aggregate update
  on app suspension or sensor interruption:
    stop session, discard partial evidence, transition -> SCAN_INTERRUPTED
  on completion:
    label mode = direct_depth
    label physical_nlos = blocked_raw_transients_unavailable
    transition -> SUMMARY_READY

build_diagnostic:
  construct exact schema from allowlisted aggregates
  reject unknown or forbidden keys recursively
  canonicalize JSON and reject size > 65536 bytes
  render preview before enabling share sheet
  never upload automatically
```

### Invariants walked through

Success case: a supported device grants camera permission, completes 15 seconds
of calibration and 30 seconds of wall scanning, then exports a 20 KiB aggregate
diagnostic. Every frame is reduced to counters before the next frame, the mode
is `direct_depth`, and `physical_nlos` stays blocked. All invariants hold.

Failure case: the app is backgrounded 12 seconds into the wall scan. Capture
stops, partial scan evidence is discarded, state becomes `SCAN_INTERRUPTED`, and
the diagnostic records only the reason and duration. It cannot label the test
complete or silently switch to replay. All invariants still hold.

## Architecture

### Components and ownership

| Component | Responsibility | Owner | Trust level |
|---|---|---|---|
| TestFlight | Signed beta distribution, invitations, expiry, updates | Apple plus release operator | External distribution boundary |
| Setup coordinator | State machine, timers, retry, evidence label | iOS maintainers | App process |
| Capability probe | Public API and device capability checks | iOS maintainers | Untrusted device/runtime inputs |
| Aggregate collector | Streaming calculation with no frame retention | Sensing plus privacy owners | Sensitive ephemeral boundary |
| Diagnostic builder | Exact allowlist, redaction, size limit, preview | Security plus support owners | Export boundary |
| External track client | Optional authenticated RuView NLOS tracks | API plus security owners | Network trust boundary |
| Feedback handoff | Opens explainer, issue, and system share UI | Product plus support owners | User-authorized external action |
| Xcode Cloud | Recommended signed archive and TestFlight delivery | Release operator | Privileged release boundary |

### Data lifecycle

```text
ARKit frame
  -> in-memory aggregate update
  -> immediate frame release
  -> bounded session summary
  -> local diagnostic preview
  -> tester exports or discards
```

The default retention target is the current app session. If the tester chooses
to save a diagnostic, the operating system share destination controls later
retention. RuView does not upload it automatically. Any future support portal
must define tenant, encryption, deletion, access logging, and retention in a
separate accepted decision before accepting diagnostics.

### Diagnostic schema contract

The on-device encoder emits the exact camel-case object below, defined by
`VisibleDepthDiagnostic` and
`docs/schemas/ruview-ios-visible-depth-diagnostic-v1.schema.json`. Unknown keys
cannot enter the strongly typed encoder and are rejected by schema consumers.

```json
{
  "schema": "ruview.ios.visible-depth-diagnostic.v1",
  "sessionId": "random UUID generated for this attempt",
  "createdAt": "RFC 3339 UTC timestamp",
  "deviceModelFamily": "coarse model family",
  "osVersion": "public OS version",
  "appVersion": "public app version and build",
  "capabilities": {
    "worldTracking": true,
    "sceneDepth": true,
    "smoothedSceneDepth": true,
    "sceneMesh": true,
    "rawPhotonHistograms": false
  },
  "phases": [{
    "phase": "calibration",
    "plannedDurationSeconds": 15,
    "observedDurationSeconds": 14.98,
    "frameCount": 450,
    "averageFPS": 30.0,
    "averageDepthCoverage": 0.91,
    "averageMovementMetersPerSecond": 0.08,
    "finalTrackingState": "normal",
    "peakThermalState": "nominal"
  }],
  "consent": {
    "localValidation": true,
    "diagnosticExport": true,
    "rawSensorExport": false
  },
  "evidenceLabel": "direct_depth",
  "physicalNLOSStatus": "blocked_raw_transients_unavailable",
  "cameraPermission": "granted",
  "completionStatus": "completed"
}
```

The optional `failureReason` is bounded to 240 characters and contains only a
public user-facing reason. Allowed values and numeric ranges are versioned with
the app. Free-form logs, stack traces, URLs, IP addresses, WiFi names, precise
device identifiers, file paths, and user-entered notes are excluded. The
canonical encoded document must remain at or below 65,536 bytes.

### Trust boundaries and threats

1. **Apple distribution boundary:** only the approved release operator can
   select a reviewed archive for external testing. Branch code cannot grant
   release authority.
2. **Sensor boundary:** ARKit frames are sensitive and untrusted. Validate
   dimensions, timestamps, finite numeric values, and continuity before
   aggregation. Never persist raw buffers by default.
3. **Network boundary:** external NLOS mode uses ATS, WSS, scoped short-lived
   authorization, server identity validation, replay protection, tenant binding,
   maximum message sizes, and immediate stale-track clearing.
4. **Export boundary:** diagnostic preview and affirmative share action are
   mandatory. The export contains no token, raw sensor payload, stable identity,
   precise location, or person trajectory.
5. **Claim boundary:** UI and diagnostics carry source provenance and evidence
   level. A successful calibration cannot mutate `direct_depth` into NLOS.
6. **Support boundary:** issue comments are public by default. The app warns the
   tester not to post private scene details and offers aggregate JSON only.

### Observability

Local observability is privacy-minimized and bounded:

1. state transition and elapsed duration;
2. completion and failure code counts;
3. aggregate frame rate, valid-depth ratio, pose stability, and interruptions;
4. peak coarse thermal state;
5. diagnostic encoded byte count and export/discard action; and
6. application version, build number, and coarse OS/device capability.

No remote analytics SDK is required for the beta. If aggregate fleet telemetry
is later enabled, it requires separate consent, documented retention, tenant
isolation, deletion controls, a data protection review, and a kill switch.

## Release architecture

### Pull request CI

GitHub Actions uses an Xcode 26 compatible macOS runner, verifies the selected
Xcode major version, runs Swift tests and an unsigned simulator build, and
creates an unsigned generic iOS archive as a dry gate. The job uses
`CODE_SIGNING_ALLOWED=NO`, does not export an IPA, does not upload an archive,
and receives no signing or App Store Connect credentials.

The archive proves that the reviewed project can reach the archive phase under
the selected SDK. It does not prove signing, TestFlight processing, installation,
camera permission behavior, LiDAR support, or NLOS sensing.

### Signed beta delivery

Use Xcode Cloud as the recommended privileged path for signed archive and
TestFlight delivery because Apple hosts the signing and App Store Connect
integration. The release workflow must:

1. trigger from an approved protected branch or reviewed tag;
2. rerun tests and archive with the declared Xcode version;
3. require a human release decision before external distribution;
4. attach the commit, build number, privacy manifest, test notes, and known
   capability limits; and
5. keep `physical_nlos` blocked unless separate live evidence is approved.

An equivalent manually operated Xcode Organizer path is acceptable for an
initial pilot, but CI must never contain reusable signing certificates or
automatic pull-request uploads.

### Rollback

Rollback is capability-first:

1. Disable external tester availability for the affected TestFlight build.
2. Publish a corrected build with a higher build number when safe.
3. Remotely disable external track mode only through an already reviewed,
   authenticated configuration path; otherwise fail closed locally.
4. Preserve direct-depth and replay features only if the fault is isolated and
   their evidence labels remain correct.
5. Revoke pairing tokens and service sessions for transport incidents.
6. Notify testers of affected versions, data exposure scope, remediation, and
   diagnostic deletion instructions.

If raw data is observed in any diagnostic, stop distribution immediately,
disable export, treat the file as a privacy incident, and require security and
privacy review before another build.

## Alternatives considered

| Alternative | Tester friction | Release risk | Privacy/security | Decision |
|---|---:|---:|---:|---|
| TestFlight plus in-app setup | About 2 to 5 minutes | Low to medium | Strong Apple signing boundary; bounded diagnostics | Selected |
| Ad hoc IPA distribution | 10 to 30 minutes plus device registration | High | Certificate and device-list custody | Rejected |
| Enterprise-signed public beta | Low initially | Critical | Misuses enterprise trust and broadens revocation blast radius | Rejected |
| Xcode source build by testers | 30 to 90 minutes | Medium | Exposes developer workflow and excludes nontechnical users | Developer fallback only |
| Web app only | Under 2 minutes | Low | Cannot directly access the required ARKit depth surface | Companion explainer/view only |
| Custom diagnostic upload service | About 1 minute | Medium to high | Creates a new personal-data trust boundary | Deferred pending separate governance |

TestFlight wins because it removes local signing and update complexity while
keeping release authority outside pull request CI. Its main cost is Apple beta
review and processing latency, which should be measured per build rather than
promised. Xcode Cloud is preferred over credential-bearing GitHub upload jobs;
the operational cost is an additional Apple-hosted workflow and usage budget.

## Refinement and failure handling

Deliver in reversible increments:

1. Add the local setup state machine and synthetic tests with capture disabled.
2. Add capability and permission checks with negative-path UI tests.
3. Add streaming aggregate collectors and prove raw buffers are not serialized.
4. Add diagnostic allowlist, forbidden-key scan, 64 KiB gate, and preview.
5. Run a physical direct-depth pilot on named devices and record evidence.
6. Configure Xcode Cloud and TestFlight only after privacy and release review.
7. Recruit external testers after internal completion time and failure recovery
   meet the protocol.

Retries are manual and bounded to three attempts per session screen. Network
reconnect uses capped exponential backoff and never preserves a live label
across authentication, freshness, or tenant failure. Calibration and wall scan
do not auto-retry after interruption because combining partial runs makes the
quality result ambiguous.

## Consequences

### Positive

1. Nontechnical testers can install and update without Xcode.
2. Support receives small, structured, comparable diagnostics instead of raw
   scenes or unbounded logs.
3. Release, software, device, and research evidence remain distinct.
4. The setup flow teaches the Apple API limitation before a tester can mistake
   direct depth for around-the-corner reconstruction.

### Costs and limitations

1. Xcode Cloud and TestFlight add Apple processing, beta review, and operational
   coordination. Budget and latency vary by account and must be measured.
2. A 64 KiB aggregate diagnostic is safer but may omit rare low-level failures;
   maintainers reproduce those through an explicitly approved development build.
3. The five minute target requires physical usability testing, not CI.
4. The largest uncertainty remains access to raw iPhone LiDAR transients. The
   fix path is to validate external histogram-capable hardware independently and
   keep the iPhone as a direct-depth, pose, transport, and presentation adapter.

## Requirement-to-evidence mapping

| ID | Requirement | Evidence | Promotion rule |
|---|---|---|---|
| NLOS-341-01 | TestFlight distributes; setup begins after install | release runbook review plus TestFlight install witness | Human release approval required |
| NLOS-341-02 | Full setup under 5 minutes | at least 20 physical-device tester timing records; median at most 180 s, p95 at most 300 s | Cannot be inferred from simulator |
| NLOS-341-03 | Calibration lasts 15 seconds | monotonic timer unit test plus physical run record | 14.5 to 16.5 s allowed for scheduling |
| NLOS-341-04 | Wall scan lasts 30 seconds | monotonic timer unit test plus physical run record | 29.5 to 32.0 s allowed for scheduling |
| NLOS-341-05 | No raw capture in diagnostic | serializer allowlist tests, forbidden-key fixtures, manual sample review | Any violation stops distribution |
| NLOS-341-06 | Diagnostic at most 64 KiB | boundary tests at 65,536 and 65,537 bytes plus CI fixture validation | Oversized document cannot export |
| NLOS-341-07 | NLOS claim blocked without raw transients | capability state tests plus diagnostic assertion | Must remain blocked until ADR-330 and ADR-331 gates pass |
| NLOS-341-08 | Permission and interruption failures degrade honestly | state-machine negative tests and physical background/denial runs | No live or completed label after failure |
| NLOS-341-09 | Pull request CI consumes no release credentials | workflow permission and secret-reference audit | Unsigned dry archive only |
| NLOS-341-10 | Xcode 26 toolchain is explicit | CI runner and `xcodebuild -version` major-version assertion | Tool mismatch fails before build |
| NLOS-341-11 | Tester controls diagnostic disclosure | preview and share/discard UI test plus usability observation | No automatic upload |
| NLOS-341-12 | Rollback is operable | internal TestFlight removal exercise and token-revocation drill | Record owner, time, and outcome |

## Acceptance test

A nontechnical tester installs the approved TestFlight build, reads the evidence
boundary, grants only the required permission, completes a measured 15 second
calibration and 30 second wall scan, sees `direct_depth` with physical NLOS
blocked, previews an aggregate diagnostic smaller than 64 KiB, opens the
explainer, and chooses whether to share or discard the diagnostic. Total elapsed
time must be under five minutes, and packet inspection plus diagnostic review
must find no raw camera, depth, point-cloud, CSI, transient, token, location, or
person-trajectory payload.

## References

1. Apple, [TestFlight overview](https://developer.apple.com/testflight/).
2. Apple, [Invite external testers](https://developer.apple.com/help/app-store-connect/test-a-beta-version/invite-external-testers/).
3. Apple, [Distribute builds using Xcode Cloud](https://developer.apple.com/documentation/xcode/distributing-your-app-for-beta-testing-and-releases).
4. Apple, [ARFrame scene depth](https://developer.apple.com/documentation/arkit/arframe/scenedepth).
5. Somasundaram et al., [consumer NLOS measurement model](https://arxiv.org/html/2605.17865v1).
6. GitHub, [GitHub-hosted runner images and macOS 26 labels](https://github.com/actions/runner-images).
