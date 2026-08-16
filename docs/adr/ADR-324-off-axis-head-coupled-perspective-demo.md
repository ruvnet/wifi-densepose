# ADR-324: off-axis-mode — RF-assisted head-coupled perspective for the three.js realtime demo

| Field | Value |
|-------|-------|
| **Status** | Proposed (core implemented — see §2.5) |
| **Date** | 2026-08-16 |
| **Deciders** | ruv |
| **Codename** | **off-axis-mode** |
| **Scope** | New `examples/three.js/demos/07-off-axis-window.html` (client-side only); no server changes |
| **Relates to** | ADR-019 (sensing-only UI), ADR-035 (live sensing UI accuracy), ADR-169 (adam-mode), ADR-170 (yoga-mode), ADR-282 (L0–L5 evidence ladder), ADR-295 (source provenance), ADR-306 (spatial ontology), ADR-307 (persistent tracking), ADR-323 (pose refinement) |
| **Prior art** | [`icurtis1/off-axis-sneaker`](https://github.com/icurtis1/off-axis-sneaker) (reference only — see §2.1 licensing) |
| **Numbering note** | ADR-324 is the next free number in the authoring checkout (322 is unused, 323 is the latest on disk). Re-run the ADR index/collision check immediately before merge and rename if needed. |
| **Tracking issue** | none yet |

---

## 1. Context

### 1.1 The question this ADR answers

"Can we use [`icurtis1/off-axis-sneaker`](https://github.com/icurtis1/off-axis-sneaker)
with RuView?" The answer is: **yes for the technique, no for the code, and
only honestly for the RF part.** This ADR records the research behind each of
those three clauses and defines the integration that is actually defensible.

### 1.2 What off-axis-sneaker is

`off-axis-sneaker` is a React + TypeScript + Vite web app that renders a GLB
model (a sneaker) in three.js and creates a *head-coupled perspective*
("fish-tank VR" / "window into the screen") illusion:

- **Tracking input**: MediaPipe Face Mesh (468 facial landmarks) from a
  webcam. Head (x, y) comes from the eye midpoint; depth (z) is proxied by
  inter-ocular distance. An exponential moving average (default factor 0.3)
  smooths jitter; sensitivity multipliers are `strengthX: 4`, `strengthY: 3`,
  `strengthZ: 2`.
- **Projection**: `src/utils/offAxisCamera.ts` builds a **true asymmetric
  (off-axis) frustum** — `makePerspective(left, right, top, bottom, near, far)`
  with `left/right/top/bottom = (screenBound − eyePosition) · (near /
  viewerToScreenDistance)` — i.e. Kooima's generalized perspective projection,
  plus a matching camera translation. Constants: `nearPlane 0.05`,
  `farPlane 1000`, `worldScale 0.01` (cm → world units), `movementScale 1.5`.
- **Calibration**: a wizard captures physical screen width/height (cm),
  typical viewing distance, and pixel density, stored locally, so eye position
  is computed relative to the *physical* display.

The technique descends from Johnny Chung Lee's 2007 Wii-remote desktop VR
demo and the fish-tank VR literature (Ware, Arthur & Booth, CHI '93). The
projection math is Robert Kooima's "Generalized Perspective Projection"
(2008). Both are public, well-documented techniques independent of any one
implementation.

### 1.3 What the illusion physically requires

The head-coupled illusion is only convincing when the tracked eye position is
**accurate to roughly centimeters** and **low-latency**. The VR literature
puts comfortable motion-to-photon latency below ~20 ms for head-mounted
displays; desktop fish-tank VR tolerates more, but visible lag between head
motion and parallax response is exactly what breaks the "window" illusion.
`CLAIMED` (literature values; no RuView measurement exists for this demo yet).

### 1.4 What RuView RF sensing can actually supply today

This is where honesty is mandatory (repo rule: never present WiFi sensing as
camera-grade).

- **Field-peak position, not metric localization.**
  `wifi-densepose-sensing-server/src/field_localize.rs` derives a position
  from the strongest peak of the 20×20 `signal_field` carried on
  `/ws/sensing` `sensing_update` frames. Its own module doc states the
  caveat: the subcarrier→angle mapping is a *representation*; "a single ESP32
  link cannot resolve a true (x, z) room position." The emitted position is
  "strongest field peak in the room model," mapped with `X_SCALE 0.6`,
  `Z_SCALE 0.5`, gated by `PEAK_THRESHOLD 0.35` — real, live, motion-tracking,
  but **not a calibrated person fix** and nowhere near eye-position precision.
- **RF pose is 2-D, normalized, constant-confidence.** The committed Cog
  (ADR-101, restated by ADR-323) emits 17 COCO keypoints as normalized 2-D
  coordinates with a constant confidence and no per-joint uncertainty. A
  "nose" keypoint exists (COCO index 0), but it is not a metric 3-D head fix.
- **Tracks are coarse and pseudonymous by design.** `ruview-track` (ADR-307)
  maintains `person_N` tracks with container-level ("kitchen → hallway")
  continuity, coarse non-reversible features, and asserts **no accuracy
  number** — outputs default to evidence level `L1`.
- **Update cadence and latency are unmeasured for this purpose.** The demo
  pipeline runs at ~30 Hz on the MediaPipe side (ADR-170), but no end-to-end
  RF motion-to-photon latency has been measured. Any figure quoted for the RF
  path must be tagged `MEASURED` with a reproducer before it appears in docs
  or UI.

Conclusion of the capability match: **RF cannot drive a convincing fish-tank
illusion by itself today**, and this ADR does not claim it can. RF *can*
supply things a webcam cannot: camera-free presence, zone-level position,
person count, approach direction, and pseudonymous continuity — including
when the camera is off.

### 1.5 What this ADR is *not*

- Not a vendoring of `off-axis-sneaker` (see §2.1 — the repo has no license).
- Not a claim of camera-grade RF head tracking, at any tier.
- Not a backend change: no new server endpoints, no new auth surface, no
  schema changes. Purely additive client-side HTML/JS, like ADR-169/170.
- Not a React/Vite/Tailwind adoption. The `examples/three.js/demos/*` are
  dependency-light single-file HTML demos and stay that way.

## 2. Decision

### 2.1 Licensing: adopt the technique, not the code

`off-axis-sneaker` publishes **no license**. Under default copyright, its
source cannot be copied, vendored, or translated into this repository.
Decision:

1. **No code, assets, or models from `off-axis-sneaker` enter this repo.**
   The GLB sneaker model is likewise unlicensed for reuse; demos use assets
   already present in `examples/`.
2. The off-axis projection is implemented **clean-room from the public
   sources**: Kooima's "Generalized Perspective Projection" (2008) — the
   `pa/pb/pc` screen-corner formulation — and three.js's documented
   `PerspectiveCamera.projectionMatrix` override path. The repository is cited
   as prior art in this ADR only.
3. If upstream later adds a permissive license, revisiting reuse requires a
   new ADR note, not silent copying.

### 2.2 Tiered integration — each tier labeled by what it really is

**Tier A (ships first): webcam-fine + RF-context hybrid.**
`07-off-axis-window.html` uses MediaPipe Face Landmarker (already the pattern
in demo 05) for fine head tracking and the Kooima frustum for rendering —
functionally what off-axis-sneaker does, reimplemented. RuView RF adds the
camera-free layer around it:

- **Presence-gated camera**: the webcam pipeline starts only when the RF
  presence signal (`/ws/sensing` `sensing_update`) says someone is in the
  zone, and stops after a configurable RF-vacancy timeout. The privacy
  posture improves: the camera is *off* until physics says there is someone
  to track.
- **Multi-person arbitration**: when RF reports more than one person, the HUD
  says so and the demo holds the last stable perspective instead of jumping
  between faces.
- **Pre-warm**: RF approach direction (field-peak trajectory) warms up
  MediaPipe and the scene before the person sits down.

**Tier B (demo mode, prominently labeled): RF-only coarse parallax.**
A toggle drives the off-axis eye position from RF alone — field peak (x, z)
plus the pose nose keypoint when present — through a one-euro filter, a
deadband, and a hard gain clamp. The HUD labels it **"RF coarse body
parallax — not head tracking"** and shows the live evidence level (`L1`
heuristic unless a certificate says otherwise, per ADR-282/ADR-318). The
expected experience is a slow, body-scale parallax sway — a demonstrative
"the room model moves because *you* moved, with no camera" — not a stable
fish-tank illusion. The demo must never present Tier B as equivalent to
Tier A.

**Tier C (future, explicitly gated, not promised): metric RF head position.**
Only a calibrated multistatic deployment (ADR-297 multi-node semantics,
ADR-311 fusion, ADR-303 ground-truth sync) with an evidence-engine ledger
entry (ADR-304) and a capability certificate (ADR-318) could justify feeding
RF positions into the fine path. No current data supports this; Tier C exists
in this ADR solely so nobody ships it informally without those gates.

### 2.3 Implementation surface

- New file `examples/three.js/demos/07-off-axis-window.html` (07, not 06 —
  ADR-170 reserves `06-yoga-mode.html`). Single-file demo following the 01–05
  conventions: same CSS custom properties, same HUD/helper-panel pattern,
  served from the existing static demo server
  (`http://127.0.0.1:8765/examples/three.js/demos/…`).
- A small clean-room module (inline `<script type="module">` or
  `examples/three.js/lib/off-axis-camera.js` if shared later) that, given
  screen corners `pa, pb, pc` (from calibration) and eye point `pe`, sets
  `camera.projectionMatrix` via the Kooima formulation each frame.
- Data inputs are the **existing** streams only: `/ws/sensing`
  (`sensing_update` → `signal_field` → field peak, using the same
  `X_SCALE`/`Z_SCALE`/`PEAK_THRESHOLD` mapping as `field_localize.rs`) and,
  when available, `/api/v1/stream/pose` for the nose keypoint. WebSocket
  access uses the existing ticket flow (`ws_ticket.rs` / `bearer_auth.rs`);
  no endpoint is exempted or added.
- Calibration mirrors the sneaker app's concept without its code: screen
  width/height in cm, viewing distance, persisted in `localStorage` under a
  demo-scoped key. No calibration data leaves the browser.
- Provenance discipline: if the demo is pointed at a synthetic or replayed
  source, the ADR-295 provenance state must surface in the HUD exactly as the
  Observatory does — synthetic can never present as live.

### 2.4 Honesty and evidence rules binding this feature

1. Every user-visible latency, accuracy, or precision statement in the demo,
   README, or docs carries a `MEASURED` (with reproducer), `CLAIMED`, or
   `SYNTHETIC` tag. This ADR itself contains no `MEASURED` claims.
2. Tier B is labeled coarse body parallax in the HUD at all times; there is
   no configuration that hides the label while RF drives the camera.
3. No PCK or pose-accuracy number may be quoted for the RF path without the
   mean-pose baseline and a leakage-free held-out split (repo rule).
4. The webcam feed never leaves the browser; no frames, landmarks, or
   embeddings are sent to the server. RF data continues to obey ADR-307's
   privacy invariants (pseudonymous, coarse, rotatable).

### 2.5 Implementation status (2026-08-16 amendment)

The projection core shipped as a **Rust crate compiled to WASM** rather than
the inline JS module §2.3 anticipated — a strict upgrade with the same
surface: `v2/crates/ruview-offaxis` (dependency-free native core; wasm-bindgen
only on wasm32) implements the Kooima projection, the one-euro filter, the
field-peak mapping (constants mirroring `field_localize.rs`), and the Tier B
coarse-parallax stage with its deadband/gain/clamp bounds enforced in Rust.
`examples/three.js/demos/07-off-axis-window.html` consumes the wasm-bindgen
output (built locally per the crate README; generated artifacts are not
committed). Validation and `MEASURED` benchmarks live in the crate README.
The demo ships with a `SYNTHETIC`-labeled mouse simulator and the labeled
Tier B RF mode; a Tier A fine tracker connects through
`OffAxisCamera.update_normalized` and remains host-provided.

## 3. Options considered

| Option | Verdict | Why |
|---|---|---|
| Vendor `off-axis-sneaker` (or fork + point at RuView) | **Rejected** | No license ⇒ no redistribution rights. Also React/Vite stack conflicts with the repo's single-file demo convention. |
| Clean-room Kooima off-axis demo, webcam-fine + RF-context (Tier A/B) | **Chosen** | Legally clean, matches demo conventions, uses RF for what it is actually good at, and demonstrates camera-free presence value honestly. |
| RF-only head-coupled perspective as the headline | **Rejected** | Over-claim. Single-link field peaks are a representation, not metric localization (`field_localize.rs` caveat); shipping this as "head tracking" violates the camera-grade rule. Survives only as the labeled Tier B toggle. |
| Wait for multistatic metric localization (Tier C) before any demo | **Rejected** | Blocks a useful, honest demo on a phase-2/3 program (ADR-303/311/318) with no delivery date. The gates are recorded instead. |
| Add a dedicated server endpoint for head position | **Rejected** | Unnecessary — existing `/ws/sensing` + `/api/v1/stream/pose` suffice; a new endpoint would expand the auth surface for no capability gain. |

## 4. Consequences

**Improves**

- A publicly legible demo of RF sensing's actual differentiator: the scene
  knows you are there, where you roughly are, and how many of you there are —
  before and without any camera.
- Privacy posture of the head-tracking demo class: camera duty-cycle is
  bounded by RF presence instead of always-on.
- Canonical, licensed off-axis projection code the Observatory or future UI
  can reuse.

**Costs / risks**

- Tier B can underwhelm viewers primed by webcam demos; the mitigation is the
  labeling and the side-by-side toggle, not inflated gain.
- MediaPipe CDN dependency (same as demo 05) remains a network-availability
  risk for Tier A; the demo must degrade to Tier B with a visible notice.
- Screen-calibration friction (cm measurements) may deter casual users; a
  "skip calibration (approximate)" path with degraded-accuracy labeling is
  acceptable.
- Upstream `off-axis-sneaker` may change or add a license; tracking that is
  manual.

**Follow-ups (not in this ADR's scope)**

- Measure end-to-end RF motion-to-parallax latency with a reproducer and
  publish it `MEASURED`.
- If/when ADR-303/311 land, evaluate Tier C against the ADR-318 certificate
  gate.
- Consider promoting the off-axis camera module into the Observatory 3D view.

## 5. Validation

- Demo checklist (manual, per ADR-169/170 practice): loads from the static
  server; Tier A activates only on RF presence; Tier B label visible whenever
  RF drives the camera; provenance badge correct against a synthetic source;
  no network requests carrying webcam-derived data (verified in devtools).
- `rg` gate before merge: no file under `examples/` contains code originating
  from `icurtis1/off-axis-sneaker`.
- No workspace, harness, or firmware validation rows are triggered — the
  change is a static HTML demo plus this document.

## 6. References

- [`icurtis1/off-axis-sneaker`](https://github.com/icurtis1/off-axis-sneaker) — prior-art reference (unlicensed; technique only)
- Robert Kooima, *Generalized Perspective Projection*, 2008 — off-axis frustum math
- Johnny Chung Lee, *Head Tracking for Desktop VR Displays using the Wii Remote*, 2007
- Ware, Arthur & Booth, *Fish Tank Virtual Reality*, CHI '93 — head coupling vs. stereo
- `v2/crates/wifi-densepose-sensing-server/src/field_localize.rs` — field-peak honesty caveat and coordinate mapping
- `v2/crates/wifi-densepose-sensing-server/src/ws_ticket.rs`, `bearer_auth.rs` — WebSocket auth pattern
- `v2/crates/ruview-track/src/lib.rs` — ADR-307 privacy invariants and evidence discipline
- ADR-169, ADR-170 — demo-scoped ADR pattern for `examples/three.js/demos/`
- ADR-282 — L0–L5 evidence ladder; ADR-295 — provenance state machine
