# ADR-342: Cognitum-inspired mobile instrument UI for RuView NLOS

| Field | Decision |
|---|---|
| **Status** | Proposed; software implementation and deterministic review evidence are part of a stacked UI-only pull request, while physical-device validation remains operator-gated |
| **Date** | 2026-08-23 |
| **Owners** | RuView Labs mobile, iOS, design, accessibility, privacy, security, and research maintainers |
| **Scope** | Presentation, responsive layout, evidence-state projection, accessible target and LiDAR point-cloud views, mobile end-to-end review flow, and screenshot baselines for the Expo and native SwiftUI NLOS clients |
| **Depends on** | ADR-295, ADR-318, ADR-319, ADR-330, ADR-340, ADR-341 |
| **Stacked pull request base** | `feat/consumer-nlos-ruview`, the head branch for pull request 1687; retarget to `main` after the dependency merges |
| **Implementation boundary** | `ui/mobile` presentation and tests, `ui/ios-nlos/App` presentation, the mobile UI screenshot baselines, and the UI validation portion of `.github/workflows/consumer-nlos-ci.yml` |

## Context

RuView NLOS already exposes strict source provenance, frame freshness, setup,
diagnostic, and transport behavior in its Expo and native SwiftUI clients. The
mobile presentation does not yet have one coherent instrument language. Its
most important state can consequently compete with setup copy, controls, and
technical detail on a narrow phone viewport.

The requested direction is similar to the visual language of Cognitum Explain
Studio: a near-black technical canvas, fine grid, restrained cyan and green
accents, compact monospaced labels, thin luminous outlines, and an orbital or
radar motif. This decision adopts that visual language as design provenance,
not as a product or code dependency. No Cognitum name, logo, wordmark, copy,
source code, private asset, authentication behavior, or investor-deck branding
is reused. RuView keeps its own name, logo, evidence vocabulary, information
architecture, and privacy boundary.

The authenticated Studio route was not available to the automated cloud review
session. The initial contract is therefore derived from the product owner's
direction and the visual characteristics observable on the available hosted
Cognitum Explain reference surface. Pixel equivalence with the authenticated
Studio route is neither claimed nor required. A maintainer with authorized
Studio access can later provide an approved reference screenshot and request a
bounded visual refinement without changing the epistemic or privacy rules in
this decision.

This is a UI-only decision. It must not change sensing, reconstruction,
filtering, transport, credential storage, permissions, diagnostic contents,
data retention, or claims. Existing source-validation and fail-closed behavior
remain authoritative. The UI projects that state; it does not invent or
upgrade it.

## Specification

### Outcome and actors

The outcome is a unified mobile instrument surface that lets a tester identify
the active evidence state, understand the privacy boundary, inspect current
hidden-target hypotheses when allowed, and reach the next safe action without
reading the entire screen.

Actors are beta testers, researchers, accessibility users, support maintainers,
design reviewers, privacy and security reviewers, and the release operator.

### Inputs

1. Existing validated Expo NLOS store values: frame source, freshness, stream
   status, accepted tracks, credential availability, and rejection reason.
2. Existing native `AppModel` connection, capability, frame, track, and visible
   depth validation state.
3. Current device safe-area insets, viewport dimensions, Dynamic Type or browser
   text scaling, reduced-motion preference, and high-contrast settings.
4. Local user actions such as selecting a view, opening setup guidance,
   starting synthetic replay, connecting an authenticated live source, or
   forgetting a credential.

### Outputs

1. A narrow-screen overview with a single prominent evidence state.
2. A visual instrument region that shows only tracks already allowed by the
   existing validation and freshness boundary.
3. Setup, provenance, privacy, and interpretation guidance in a stable reading
   order.
4. Reviewable screenshot baselines for overview, synthetic, setup, and Three.js
   point-cloud states.
5. A production-browser Playwright end-to-end suite and a mobile Maestro flow
   that verify navigation, state visibility, synthetic watermarking, setup
   affordances, and the primary local action.

### Honest evidence-state taxonomy

The presentation exposes exactly five top-level states. These are display
projections over existing domain state and do not create a second sensing state
machine.

| Display state | Minimum condition | Required presentation | Forbidden presentation |
|---|---|---|---|
| `SYNTHETIC` | A fresh accepted frame is explicitly sourced from synthetic replay | Persistent amber label and watermark in the instrument region | Green live treatment, verified language, or removal of the watermark |
| `LIVE VERIFIED` | A fresh accepted live frame has authenticated transport and satisfies the existing provenance and evidence gate | Green and cyan live label plus source and freshness details | Inferring verification from connection status alone |
| `LIVE UNVERIFIED` | A live transport is active or attempting to provide data, but no fresh frame satisfies the full verification gate | Amber caution label, withheld target visualization, and corrective guidance | Calling the source verified or silently substituting replay |
| `STALE` | A previously displayable frame has exceeded the existing freshness threshold | High-priority stale label, age context when available, and suppression of hidden-target geometry | Continuing to show cached targets as current |
| `DISCONNECTED` | No eligible fresh source is active and no stale frame needs a stronger warning | Neutral disconnected label and explicit connection or replay actions | Implied live availability or retained target geometry |

Unknown, contradictory, or partially initialized input resolves to `LIVE
UNVERIFIED` only when an authenticated live attempt is active. It resolves to
`DISCONNECTED` otherwise. No unknown state may resolve to `LIVE VERIFIED`.

### Layout and interaction requirements

1. The primary reference viewport is 390 by 844 CSS pixels or iOS points. It
   must have no horizontal document or root-scroll overflow at 100 percent text
   scale.
2. The layout must remain functional at 320 points wide and with text enlarged
   to 200 percent. Secondary rows may wrap vertically; essential state, privacy,
   and action text may not be clipped.
3. Interactive targets have a minimum hit area of 44 by 44 points. Adjacent
   targets maintain at least 8 points of separation unless their combined
   control supplies equivalent accessible grouping.
4. Evidence state is encoded by text and icon or shape as well as color. Color
   is never the sole differentiator.
5. The semantic reading order is identity and evidence state, instrument,
   metrics, primary actions, setup, provenance, privacy, then technical detail.
6. The existing explainer and tester-feedback destinations remain visible from
   setup without collecting credentials or diagnostics.
7. Motion is decorative and bounded. Reduced-motion mode removes continuous
   orbit, sweep, pulse, and parallax effects without removing information.
8. The web point-cloud mode uses the already installed Three.js runtime. It
   renders only deterministic relay samples and target returns derived from
   tracks that have already passed the display gate. The native surface uses an
   equivalent SwiftUI Canvas projection. Both identify the view as a rendered
   reconstruction rather than raw iPhone LiDAR output.

### Performance budgets

1. Initial usable render is under 1.5 seconds in a production browser build on
   a representative modern iPhone under the recorded test conditions. Usable
   means the evidence state and primary action are visible and respond to input.
2. Local UI state interactions have p95 input-to-committed-visual latency under
   100 milliseconds over at least 30 repetitions on the same device.
3. Continuous decorative animation must not be required for state recognition.
   It pauses when the app or page is inactive and honors reduced motion.
4. The redesign adds no network request, remote font, remote image, analytics
   client, sensor subscription, or background timer.
5. Screenshot rendering is deterministic: fixed viewport, fixed fixtures,
   reduced motion, local assets, and no dependency on a live endpoint.
6. Point-cloud geometry is bounded to 288 schematic relay points plus 96
   target-return points for each of at most 16 validated tracks, or 1,824 total
   points. Browser device pixel ratio is capped at 1.5. GPU resources, animation
   frames, resize observers, and pointer listeners are released on unmount.

The timing budgets are targets until a physical iPhone report records device,
OS, build commit, build mode, browser, network conditioning, repetition count,
median, and p95. CI and desktop browser results are software evidence and must
not be relabeled as physical-device measurements.

### Explicit exclusions

This decision does not authorize or alter optical NLOS sensing, ARKit capture,
CSI ingestion, WebSocket protocol behavior, endpoint validation, token handling,
Keychain behavior, browser credential persistence, permission prompts,
diagnostic schema, upload behavior, raw-data retention, background sensing,
identity inference, health or safety use, or camera-equivalence claims.

## Pseudocode and state transitions

### Evidence-state projection

```text
projectEvidenceState(domain):
  if domain.freshness == stale and domain.previouslyDisplayableFrameExists:
    return STALE

  if domain.freshness == fresh and domain.acceptedFrame.source == synthetic:
    return SYNTHETIC

  if domain.liveAttemptActive:
    if domain.freshness == fresh
       and domain.acceptedFrame.source == live
       and domain.transportAuthenticated
       and domain.provenanceGatePassed
       and domain.evidenceGatePassed:
      return LIVE_VERIFIED
    return LIVE_UNVERIFIED

  return DISCONNECTED
```

The projection consumes authoritative state without mutating it. A source
cannot become verified because a user tapped a control, because a socket is
open, or because a previous frame was verified.

### Presentation control flow

```text
onRender:
  state = projectEvidenceState(authoritativeDomainState)
  announce state when it changes, but do not repeatedly announce frame updates
  render state text, redundant icon or shape, and provenance summary

  if state == SYNTHETIC:
    render only accepted fresh replay tracks
    render persistent SYNTHETIC watermark above the instrument
  else if state == LIVE_VERIFIED:
    render only accepted fresh live tracks
  else:
    render no hidden-target geometry
    render safe recovery guidance for the current state

  if selectedView == pointCloud:
    build bounded deterministic cloud from the same displayable tracks only
    render schematic relay points as non-evidence context
    label the surface as a reconstruction view, not raw LiDAR
    use Three.js WebGL on web and SwiftUI Canvas on native

  preserve setup, privacy, interpretation, and feedback paths
  disable decorative motion when reduced motion is requested
```

### Responsive layout flow

```text
layout(viewport, textScale, safeArea):
  availableWidth = viewport.width - safeArea.left - safeArea.right
  apply bounded horizontal inset
  place identity and evidence status before the instrument
  stack metrics and actions when their measured width does not fit
  allow labels to wrap; never reduce essential text below token minimum
  assert rootScrollWidth <= viewport.width
  assert every interactive hit rectangle >= 44 by 44 points
```

### Screenshot flow

```text
captureBaseline(name, fixture):
  build production web bundle
  serve local immutable bundle
  set viewport to 390 by 844
  enable reduced motion and deterministic fixture mode
  navigate to NLOS screen
  apply fixture and wait for stable UI contract marker
  assert evidence-state marker and state-specific safety cues
  assert root has no horizontal overflow
  save PNG with expected dimensions
```

### Success and failure walks

Success case: synthetic replay provides a fresh accepted frame. The projection
returns `SYNTHETIC`, the instrument shows only accepted tracks, a visible amber
label and watermark remain present, and no live wording appears. The user can
switch local views without changing provenance. All invariants hold.

Failure case: an authenticated live frame becomes stale while targets were
visible. The next projection returns `STALE`, removes target geometry, displays
the stale warning, and offers recovery guidance. A reconnect cannot restore
`LIVE VERIFIED` until a new accepted frame passes the existing gates. All
invariants hold.

Contradiction case: transport reports connected while the frame provenance is
missing. The projection returns `LIVE UNVERIFIED`, does not render target
geometry, and never infers verification from connection state. All invariants
hold.

## Architecture

### Component boundaries

| Component | Responsibility | May change in this decision | Must not change |
|---|---|---|---|
| Expo NLOS screen and presentation components | Responsive hierarchy, tokens, evidence-state label, accessible controls, instrument composition | Yes | Store semantics, service requests, credential lifecycle |
| Expo Three.js point-cloud adapter | Deterministic bounded BufferGeometry, local WebGL rendering, orbit input, reduced motion, resize, and disposal | Yes | Raw sensor access, network fetches, provenance promotion, persistence |
| Native SwiftUI shell and presentation helpers | Equivalent hierarchy, tokens, state labels, Dynamic Type, reduced motion, hit targets | Yes | `AppModel`, Apple capability probe, stream guard, Keychain, diagnostics |
| Native SwiftUI point-cloud projection | Accessible Canvas rendering from the same displayable track list | Yes | Three.js embedding, WebView, raw ARKit depth, or new entitlement |
| Existing Expo store and hooks | Authoritative source, freshness, frame, rejection, and connection inputs | No | Any sensing or transport behavior |
| Existing native model and core packages | Authoritative connection, frame, track, diagnostic, and capability inputs | No | Any sensing, validation, persistence, or permission behavior |
| Playwright production-browser suite | Black-box responsive navigation, overflow assertions, local state actions, and screenshot reproduction | Yes | Live endpoint use, credentials, or physical-device claims |
| Maestro mobile review flow | Black-box native navigation and visible safety-contract assertions | Yes | Production state or credentials |
| Screenshot harness and baselines | Deterministic visual review evidence | Yes | Live endpoint access or physical-hardware claims |
| Consumer NLOS CI | Build, test, static UI-contract and baseline validation | UI validation only | Signing, upload, secrets, or release authority |

### Design token contract

The platforms implement equivalent tokens using their native systems rather
than a new shared runtime package:

1. near-black navy canvas and elevated graphite surfaces;
2. subtle grid lines with low contrast and no information meaning;
3. cyan primary accent and green verified accent;
4. amber synthetic and unverified accent, and red stale or blocked accent;
5. thin low-opacity outlines, restrained glow, and high-contrast text;
6. editorial sans-serif hierarchy with monospaced instrument labels; and
7. orbital or radar geometry that remains decorative and can be disabled.

Exact color values may differ enough to satisfy platform contrast and material
behavior. Semantic state, contrast, hierarchy, and spacing are the contract;
pixel identity between React Native Web and SwiftUI is not.

### Data lifecycle and trust boundaries

```text
existing validated domain state
  -> pure evidence-state projection
  -> displayable track allowlist
  -> local target view or deterministic point-cloud projection
  -> pixels and accessibility semantics
```

The redesign adds no storage and no export. Sensitive values remain in their
existing owners. Credential controls may display length-independent readiness
or masked state only; tokens never enter labels, logs, screenshots, test
fixtures, accessibility values, analytics, or error copy. Screenshot fixtures
contain synthetic, non-personal data and cannot connect to a live endpoint.

External links remain explicit user actions handled by the operating system or
browser. Opening an explainer or feedback page cannot attach a diagnostic,
credential, scene detail, endpoint, or referrer payload created by the app.

### Deployment and stacking

The UI pull request is stacked on `feat/consumer-nlos-ruview` because the NLOS
clients do not yet exist on `main`. Its diff against that base contains only UI
presentation, UI tests, screenshots, this ADR, and bounded UI CI validation.
After pull request 1687 merges, maintainers retarget the UI pull request to
`main`, verify that the diff remains scoped, rerun required checks, and merge it
independently. No sensing commit is cherry-picked into the UI branch.

## Decision

Adopt a RuView-owned mobile instrument system inspired by the available
Cognitum Explain visual language on both Expo and native SwiftUI. Preserve one
explicit evidence-state surface, a fail-closed instrument, prominent privacy
and interpretation boundaries, and platform-native accessibility behavior.

Use mirrored semantic tokens and state projection tests instead of sharing a
runtime UI package across TypeScript and Swift. Use deterministic Expo web
screenshots as cross-platform review references, not as proof of native pixel
output. Require an Xcode 26 build gate and a named physical-device review before
accepting native behavior or performance claims.

For the point-cloud surface, use Three.js directly in the Expo web build because
it is already a locked runtime dependency and can render bounded BufferGeometry
without a remote asset. Use SwiftUI Canvas for the native counterpart. Both
consume only the existing fail-closed displayable track list. Do not add
`expo-gl`, React Three Fiber, a WebView, a raw point-cloud schema, or a new sensor
adapter in this UI-only decision.

## Alternatives with quantified tradeoffs

Alternatives are scored from 1, poor, to 5, strong. Weighted score is out of
5.0. Product clarity and epistemic safety each receive 25 percent; delivery
cost, native accessibility, and long-term maintainability each receive 15, 20,
and 15 percent respectively.

| Alternative | Clarity 25% | Epistemic safety 25% | Delivery cost 15% | Native accessibility 20% | Maintainability 15% | Weighted score | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| RuView-owned mirrored instrument system | 5 | 5 | 4 | 5 | 4 | 4.70 | Selected |
| Restyle Expo only and leave native SwiftUI unchanged | 3 | 4 | 5 | 3 | 2 | 3.40 | Rejected because two beta surfaces would disagree on the primary state and review burden would increase |
| Embed one web UI in the native app | 4 | 4 | 3 | 2 | 4 | 3.40 | Rejected because WebView startup, focus, Dynamic Type, offline behavior, and native sensor workflow integration add operational risk |
| Create a cross-platform generated token and component package now | 5 | 5 | 1 | 4 | 3 | 3.90 | Deferred because generation infrastructure is disproportionate to one screen and would expand the UI-only diff materially |
| Keep the current generic card layout | 2 | 4 | 5 | 4 | 4 | 3.65 | Rejected because the evidence state remains visually subordinate and does not meet the requested design outcome |

The selected approach is expected to add roughly two small platform-specific
presentation implementations and one shared behavioral contract, while adding
zero runtime dependency. The cost is duplicated token maintenance. The control
is a screenshot review plus identical state names and accessibility assertions
on both platforms.

### Point-cloud renderer alternatives

The same weights apply to the narrower renderer choice.

| Alternative | Clarity 25% | Epistemic safety 25% | Delivery cost 15% | Native accessibility 20% | Maintainability 15% | Weighted score | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Three.js web plus SwiftUI Canvas native | 5 | 5 | 4 | 4 | 4 | 4.50 | Selected because it satisfies the requested Three.js view while preserving platform-native accessibility and adding zero runtime dependency |
| Three.js inside a native WebView | 5 | 4 | 3 | 2 | 4 | 3.70 | Rejected because focus, Dynamic Type, startup, memory, and lifecycle behavior are weaker |
| Add `expo-gl` and React Three Fiber for native Expo | 5 | 4 | 2 | 3 | 3 | 3.60 | Rejected because new native dependencies and build risk are disproportionate to a UI-only change |
| Static SVG or Canvas cloud on every platform | 3 | 5 | 5 | 4 | 5 | 4.30 | Rejected as the sole implementation because it does not provide the requested interactive Three.js web experience |

## Accessibility

1. Every interactive element has a descriptive label, role or trait, and a hit
   target of at least 44 by 44 points.
2. Evidence state is exposed as text and an accessible value. It is not encoded
   by color, glow, motion, or spatial position alone.
3. State-change announcements are debounced to meaningful transitions. Frame
   rate and coordinate updates do not continuously interrupt screen readers.
4. Expo supports browser text scaling and native font scaling. SwiftUI uses
   Dynamic Type and avoids fixed-height text containers for essential content.
5. At 200 percent text scale, controls may stack and cards may grow vertically;
   evidence state, privacy copy, warnings, and primary actions remain complete.
6. Reduced motion removes continuous decorative animation. Increased contrast
   retains distinct borders and semantic text.
7. The visual grid, glow, radar sweep, and orbital decorations are hidden from
   the accessibility tree.
8. The synthetic watermark has a programmatic label in addition to its visual
   rendering.
9. The WebGL canvas is hidden from the accessibility tree and its containing
   view exposes one concise label with evidence freshness, hypothesis count,
   and gated target-return count. Orbit gestures are optional and convey no
   exclusive information.

## Security and privacy

1. The redesign adds no permission, entitlement, endpoint, network request,
   storage key, cookie, analytics event, remote asset, background mode, or data
   retention path.
2. Existing credential rules remain unchanged: the Expo client holds an
   ephemeral credential in memory, while the approved native client uses its
   existing Keychain boundary. UI code cannot log, render, snapshot, export, or
   persist the credential.
3. Screenshot and E2E fixtures are synthetic and contain no CSI, image, raw
   depth, captured point cloud, location, person, device identifier, token,
   endpoint, or private diagnostic data. The displayed point cloud is generated
   locally from the deterministic synthetic track fixture and schematic relay
   geometry.
4. Stale, malformed, unauthenticated, replayed, unknown, or contradictory input
   fails closed through the existing validation layer and the state projection.
5. The synthetic watermark remains visible in every synthetic visualization
   and captured synthetic baseline. No style token can disable it.
6. The privacy and interpretation boundary is visible without opening a menu.
   Raw RF, audio, camera, depth, and transient retention remain off or absent as
   defined by prior decisions.
7. External links open only after a user action and do not append app state,
   credentials, diagnostics, or scene context.
8. CI uses read-only repository permission and receives no signing, App Store
   Connect, endpoint, or test-user secret for the UI contract.
9. Three.js receives no URL, shader text, texture, model, worker, or external
   input. It allocates local typed arrays from already gated numeric track data
   and releases WebGL resources when the view changes or unmounts.

## Performance budgets

The normative performance thresholds are defined in Specification. Validation
uses three evidence classes:

| Evidence | What it can establish | What it cannot establish |
|---|---|---|
| Deterministic CI production build | Bundle validity, type and unit tests, baseline dimensions, state-contract presence | iPhone render timing, thermal behavior, touch latency |
| Desktop browser capture at 390 by 844 | Responsive composition, overflow, fixture states, screenshot review | Mobile Safari GPU behavior or physical touch response |
| Named physical iPhone run | Mobile Safari usable-render time, local interaction p95, Dynamic Type, VoiceOver, reduced motion, safe areas | General performance across every supported device |

No measured value is documented without its reproducer and evidence label. A
desktop or CI measurement is `MEASURED_SOFTWARE` and not a physical-device
claim. A threshold without a completed physical run remains `TARGET`.

## Refinement and failure handling

### Increment plan

1. Introduce platform-local semantic colors, typography, spacing, and
   instrument primitives without changing data owners.
2. Project and render the five evidence states with unit coverage for success,
   stale, disconnected, contradiction, and synthetic watermark cases.
3. Recompose the Expo NLOS screen at the reference viewport and add the
   production-browser Playwright suite plus Maestro mobile flow.
4. Add the bounded Three.js point-cloud adapter, deterministic generator,
   accessible fallback, and fail-closed zero-return tests.
5. Recompose the native SwiftUI surface using equivalent semantics, Dynamic
   Type, and reduced-motion behavior.
6. Capture deterministic Expo overview, synthetic, setup, and point-cloud
   baselines.
7. Run type, lint, unit, bundle, Swift package, Xcode 26, contract, accessibility,
   security, and diff-scope review gates.
8. Complete the named physical iPhone gate before calling native behavior or
   mobile performance validated.

### Failure handling

1. If state inputs conflict, use `LIVE UNVERIFIED` during an active live attempt
   or `DISCONNECTED` otherwise, hide targets, and retain the rejection reason
   supplied by the authoritative layer.
2. If a frame becomes stale, remove its geometry on the same committed state
   transition. Reconnection alone does not restore it.
3. If decorative rendering fails, retain plain text state, controls, privacy,
   and provenance. Decoration cannot block operation. If WebGL creation or its
   context fails, identify the static fallback without promoting evidence.
4. If the viewport overflows, stack secondary content and remove nonessential
   decoration before reducing essential text or hit areas.
5. If the performance target fails, profile render count, SVG complexity,
   shadow or blur cost, and animation scheduling. Reduce decorative work before
   changing semantic content.
6. If a screenshot differs, the reviewer must classify it as an intended design
   change, platform rendering variance, fixture drift, or regression. Baselines
   are never updated solely to make CI pass.
7. If the native Xcode or physical-device gate fails, the web UI may remain
   reviewable, but the pull request is not described as fully validated on iOS.
8. If authenticated Studio reference access later reveals a material mismatch,
   refine nonsemantic tokens in a follow-up. Evidence and privacy behavior stay
   unchanged unless a new accepted ADR explicitly changes them.

## Requirement to evidence mapping

| Requirement | Automated evidence | Human or physical evidence | Pass condition |
|---|---|---|---|
| UI-only scope | Pull-request diff allowlist and service or model tests | Reviewer checks no sensing, transport, permissions, persistence, or retention diff | Zero out-of-scope behavior changes |
| Cognitum-inspired, RuView-owned language | Four screenshot baselines and token review | Product reviewer compares hierarchy and general visual language | Requested characteristics present; zero Cognitum branding or private assets |
| Five honest states | Unit tests plus `nlos-evidence-state` Playwright and Maestro selectors | Reviewer verifies wording and fail-closed hierarchy | All five exact labels are representable; unknown never becomes verified |
| Synthetic safety | Unit and E2E assertions for `nlos-synthetic-watermark` | Screenshot review of synthetic baseline | Label and watermark both visible |
| LiDAR point-cloud UI | Deterministic generator tests, exact point budgets, Three.js canvas readiness, reduced-motion capture, and zero-return fail-closed assertion | Browser and native visual review plus physical iPhone GPU check | WebGL renders 288 schematic relay points plus 96 returns per gated track; unverified states render zero target returns; native equivalent is labeled as a projection |
| 390 by 844 layout | PNG dimension contract and browser overflow assertion | Screenshot review | Exact dimensions and no horizontal overflow |
| 44 point targets | Style and interaction assertions where supported | iPhone accessibility inspector and manual target review | Every interactive hit rectangle is at least 44 by 44 points |
| Accessibility | Semantic tests, lint, and reduced-motion fixture | VoiceOver, Dynamic Type at 200 percent, contrast, and reduced-motion review | State and primary actions remain understandable and operable |
| Initial usable render under 1.5 seconds | Production build and instrumentation contract | Named modern iPhone run with recorded conditions | Measured usable render is below 1.5 seconds |
| Local interaction p95 under 100 milliseconds | Deterministic action loop instrumentation | Named iPhone run over at least 30 repetitions | Recorded p95 is below 100 milliseconds |
| Screenshot baselines | CI validates PNG format, names, dimensions, and synthetic marker contract | Reviewer approves overview, synthetic, setup, and point-cloud compositions | All four approved and traceable to the commit |
| No security or privacy expansion | Unit suite, dependency audit, secret scan, workflow permission review | Privacy and security diff review | No new collection, permission, storage, network, secret, or retention path |
| Native equivalence | Swift tests and Xcode 26 unsigned simulator build | Named physical iPhone inspection | Equivalent hierarchy and state semantics; no pixel-equivalence claim |

## Rollout and rollback

### Rollout

1. Review the stacked diff against `feat/consumer-nlos-ruview` and confirm the
   scope is limited to presentation, UI tests, screenshots, this ADR, and UI CI.
2. Require the deterministic Expo checks, screenshots, Maestro contract, Swift
   tests, and Xcode 26 unsigned build to pass.
3. After pull request 1687 merges, retarget to `main`, recheck the diff, and
   rerun required checks.
4. Install the resulting beta on a named supported iPhone and complete the
   physical accessibility, layout, state, and performance protocol.
5. Admit beta testers only after release governance in ADR-341 remains green.

### Rollback

Rollback is presentation-only. Revert the mobile UI commit while preserving
the NLOS state, service, validation, credential, and diagnostic implementations
from the base feature. If only decoration is defective, disable the affected
grid, glow, orbit, or animation and retain the plain evidence state, actions,
privacy boundary, and setup path. No rollback may remove the synthetic
watermark, stale target suppression, or visible evidence label.

## Acceptance test

The decision is accepted only when all software gates pass and the physical
gate is either complete or explicitly recorded as pending without a claim of
full iPhone validation.

### Deterministic software gate

1. Build the Expo production web and iOS bundles, run type checking, lint, unit
   tests, dependency audit, the production-browser Playwright suite, and the
   NLOS mobile UI Maestro contract validation.
2. Render the NLOS route from a production web bundle at 390 by 844 with reduced
   motion and deterministic fixtures. Assert `scrollWidth <= clientWidth`.
3. Capture and review:
   `docs/screenshots/consumer-nlos-mobile-ui/overview-390x844.png`,
   `docs/screenshots/consumer-nlos-mobile-ui/synthetic-390x844.png`,
   `docs/screenshots/consumer-nlos-mobile-ui/setup-390x844.png`, and
   `docs/screenshots/consumer-nlos-mobile-ui/point-cloud-390x844.png`.
4. Assert each PNG is exactly 390 by 844. Assert the synthetic baseline and E2E
   flow preserve the `SYNTHETIC` label and watermark.
5. Select LiDAR cloud, assert the Three.js canvas reports ready, assert the
   deterministic fixture renders 96 gated target returns, and confirm the
   390-wide cloud has no horizontal overflow.
6. Exercise `SYNTHETIC`, `LIVE VERIFIED`, `LIVE UNVERIFIED`, `STALE`, and
   `DISCONNECTED` projections. Assert stale, unverified, and disconnected states
   render no hidden-target geometry or point-cloud target returns.
7. Run Swift package tests and the Xcode 26 unsigned simulator build. Confirm
   zero changes to core packages, sensing, transport, credentials, permissions,
   diagnostic schema, or retention.

### Physical iPhone gate

On a named representative modern iPhone, record model, OS, commit, build mode,
browser or native client, and test conditions. Verify no overflow in portrait,
44 point hit targets, VoiceOver reading order, Dynamic Type at 200 percent,
reduced motion, all five states, synthetic watermarking, stale target removal,
initial usable render under 1.5 seconds, and local interaction p95 under 100
milliseconds over at least 30 repetitions.

Simulator, unsigned archive, desktop browser, or screenshot success does not
substitute for this physical gate. Until the report is attached to the pull
request or linked issue, the correct status is software-validated, with
physical iPhone validation pending.
