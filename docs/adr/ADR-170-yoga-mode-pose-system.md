# ADR-170: yoga-mode — pose detection, classification, and scoring for the three.js realtime demo

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-06-02 |
| **Deciders** | ruv |
| **Codename** | **yoga-mode** |
| **Scope** | `examples/three.js/demos/05-skinned-realtime.html` (primary); new `examples/three.js/demos/06-yoga-mode.html` (secondary, slimmed-down) |
| **Relates to** | ADR-169 (adam-mode light theme), ADR-019 (sensing-only UI), ADR-035 (live sensing UI accuracy) |
| **Tracking issue** | none yet |

---

## 1. Context

`examples/three.js/demos/05-skinned-realtime.html` already runs the full MediaPipe Pose Heavy pipeline at ~30 Hz: 33 BlazePose landmarks flow through a one-euro-filter bank into joint-angle extraction and then into a Mixamo X Bot IK retarget. The `#pose-panel` HUD shows landmark count, visibility, and pose FPS. The `#helpers` panel (ADR-097) has adam-mode (ADR-169) and eight visualisation toggles.

This infrastructure is complete. Every frame, per-joint angles are already computable from the existing `liveKp` world-space landmark array. What does not yet exist is any layer that interprets those angles as a known yoga pose, scores the user's alignment against a target shape, and guides the user through a structured sequence.

### 1.1 Why yoga-mode in this demo

Three concrete use-cases drive this:

1. **Developer self-test for the retargeting pipeline.** Cycling through a Sun Salutation A is a systematic, reproducible way to exercise every major joint (shoulder, elbow, hip, knee, spine). A pose-scoring overlay makes regression immediately visible — if a code change breaks elbow retargeting, the yoga classifier will output a depressed alignment score on Chaturanga even before a visual inspection.

2. **Public demonstration value.** The demo is served at `http://127.0.0.1:8765/examples/three.js/demos/05-skinned-realtime.html` and shown to evaluators. A guided instructional mode that scores real-time body alignment against Tadasana or Downward Dog is immediately intelligible to a non-technical audience in a way that raw CSI amplitude bars are not.

3. **Future bridge to the Rust host.** The Rust-side `wifi-densepose-signal/src/ruvsense/pose_tracker.rs` maintains a 17-keypoint Kalman tracker in COCO convention. yoga-mode in the demo operates on the 33-landmark MediaPipe convention. These are not the same: MediaPipe indices 0–32 (BlazePose) map non-trivially to COCO 0–16. Deciding the mapping now — even in a pure-JS context — canonicalises it for the eventual Rust integration.

### 1.2 What this ADR is *not*

- Not a backend service. No WebSocket endpoint, no session record, no cloud upload. Pure client-side HTML.
- Not a fitness-app competitor. The scope is Sun Salutation A (8 poses). The full 84-asana classical corpus is out of scope.
- Not an integration with the Rust `pose_tracker.rs`. That bridge is documented here as a future consequence, not an immediate deliverable.
- Not a redesign of demo 05. Panel layout, three.js scene geometry, and the CSI overlay are unchanged.
- Not a new design system. yoga-mode inherits every existing CSS custom property.

### 1.3 COCO-17 ↔ BlazePose-33 mapping note

The Rust tracker uses COCO 17-keypoint indices (0=nose, 5=left-shoulder, 6=right-shoulder, 7=left-elbow, 8=right-elbow, 9=left-wrist, 10=right-wrist, 11=left-hip, 12=right-hip, 13=left-knee, 14=right-knee, 15=left-ankle, 16=right-ankle). MediaPipe BlazePose-33 uses a different, denser scheme where shoulders are at 11–12, elbows at 13–14, wrists at 15–16, hips at 23–24, knees at 25–26, ankles at 27–28.

The mapping for the 13 joints used in yoga-mode angle computation is:

| Joint role | COCO idx | BlazePose idx |
|---|---|---|
| nose | 0 | 0 |
| left shoulder | 5 | 11 |
| right shoulder | 6 | 12 |
| left elbow | 7 | 13 |
| right elbow | 8 | 14 |
| left wrist | 9 | 15 |
| right wrist | 10 | 16 |
| left hip | 11 | 23 |
| right hip | 12 | 24 |
| left knee | 13 | 25 |
| right knee | 14 | 26 |
| left ankle | 15 | 27 |
| right ankle | 16 | 28 |

When the Rust host integration is implemented, the joint-angle features extracted by yoga-mode in JS and by `pose_tracker.rs` in Rust will be computed from the same physical joints via this table. No translation layer is needed at runtime — yoga-mode always uses BlazePose indices; `pose_tracker.rs` always uses COCO indices.

### 1.4 Biomechanical basis for joint-angle targets

The joint-angle targets in this ADR are grounded in peer-reviewed measurements. Perez-Testor et al. (2019, PMC6521759) captured 10 trained practitioners performing Surya Namaskar A on a 12-camera Vicon system at 100 Hz, reporting sagittal-plane joint angles at each pose transition. Key ranges: elbow 22°–116°, hip 15° extension to 134° flexion, knee 3° hyperextension to 140° flexion, spine 44° extension to 58° flexion, shoulder 56°–183°. These empirical ranges set the upper and lower bounds for the tolerance bands in this ADR's pose templates. Where Perez-Testor does not report a joint (e.g. wrist flexion for Chaturanga arm angle), the Iyengar geometry — "elbows at 90° bent close to the body" — supplies the target value. A 2023 PMC yoga-pose review (PMC10280249) confirming angle-heuristic approaches as the most reliable real-time classification method validates the algorithmic choice.

---

## 2. Decision

### 2.1 Pose taxonomy — Sun Salutation A, 8 poses

Sun Salutation A is chosen for the first ship. It satisfies three criteria simultaneously: the poses are geometrically distinct from each other (no two share the same joint-angle signature), they form a complete bilateral sequence (both left and right sides are exercised), and they are among the best-documented asanas in the biomechanics literature. The Sanskrit and English names are unambiguous in the Ashtanga tradition.

The 8 poses in sequence order with their one-line joint-angle signatures:

| Stage | Sanskrit | English | Joint-angle signature |
|---|---|---|---|
| 1 | Tāḍāsana | Mountain Pose | All limbs extended: knees 180°, hips 180°, elbows 180°, spine vertical |
| 2 | Ūrdhva Hastāsana | Upward Salute | Arms overhead: shoulders ~180° abducted, elbows 180°, torso elongated |
| 3 | Uttānāsana | Standing Forward Fold | Hips ~0–30° (full fold), knees 180°, elbows relaxed, spine flexed |
| 4 | Ardha Uttānāsana | Half Lift / Flat-Back | Hips ~90° (parallel torso), knees 180°, spine neutral (horizontal) |
| 5 | Catvāri (Chaturanga Daṇḍāsana) | Four-Limbed Staff | Hips 180° (plank line), elbows ~90°, shoulders depressed, body horizontal |
| 6 | Ūrdhva Mukha Śvānāsana | Upward-Facing Dog | Hips extended ~160°+, shoulders over wrists, spine extended, knees off floor |
| 7 | Adho Mukha Śvānāsana | Downward-Facing Dog | Hips ~80–110° (inverted V), knees 180°, shoulders ~180° (arms overhead), spine long |
| 8 | Uttānāsana | Standing Forward Fold (return) | Same as stage 3 — mirrors the descent; re-classified as stage 8 for sequence tracking |

"All 84 classical asanas" is explicitly rejected. Even the 26-pose Bikram set is rejected — the goal is a complete, self-contained instructional sequence for a 2–3 minute demo session, not exhaustive coverage. Eight poses are the minimum for a meaningful sequence narrative and the maximum that fits a single UI strip without horizontal scrolling on a 1080p screen.

### 2.2 Detection algorithm — joint-angle threshold matching with weighted scoring

**Chosen: joint-angle threshold matching.** For each frame, compute the angle at 6–10 named joints (one angle per joint, defined as the interior angle at the vertex formed by three landmarks). Compare each computed angle to the per-pose target. Score by weighted absolute deviation. Classify the argmax.

**Why not the alternatives:**

| Alternative | Verdict | Reason |
|---|---|---|
| Skeleton-as-vector cosine similarity | Rejected | Position-sensitive: a person standing 2 m from the camera vs. 1 m produces different vectors. Joint angles are translation- and scale-invariant by construction. |
| Small MLP trained on a labelled dataset | Rejected | No labelled dataset exists in this codebase. Training a reliable MLP for 8 poses would require hundreds of labelled examples per class, a train/test split, and a model serialization format — none of which belongs in a single-file demo HTML. Joint-angle matching achieves the same discrimination for 8 geometrically distinct poses with zero training data. |
| MediaPipe Tasks PoseClassifier (EfficientNet-based) | Rejected | Requires loading a separate `.task` bundle (~4 MB), adds a network dependency to the demo's offline-capable design, and uses a black-box embedding — undebuggable when a pose is misclassified. Threshold matching is fully inspectable in DevTools. |
| DTW template matching on full landmark sequences | Rejected | Appropriate for gesture recognition over time (ADR-014's `gesture.rs`), not static pose classification. Sun Salutation transitions are slow (2–5 seconds per pose); per-frame angle scoring is sufficient. |

**Joint angle computation.** For three landmark positions A (proximal), B (vertex), C (distal), the interior angle at B is:

```
angle_B = arccos( dot(A-B, C-B) / (|A-B| * |C-B|) )   in degrees
```

This is computed in world-space from the existing `liveKp` THREE.Vector3 array. The computation is purely arithmetic — no matrix inversion, no DFT. At 30 Hz on any modern laptop it is unmeasurably fast relative to the MediaPipe inference cost.

**Named joints used in yoga-mode.** Joint names, their three-landmark triplets (proximal-vertex-distal), and the BlazePose indices:

| Joint name | Triplet (P-V-D) | Indices |
|---|---|---|
| `left_elbow` | shoulder→elbow→wrist | 11→13→15 |
| `right_elbow` | shoulder→elbow→wrist | 12→14→16 |
| `left_knee` | hip→knee→ankle | 23→25→27 |
| `right_knee` | hip→knee→ankle | 24→26→28 |
| `left_hip` | shoulder→hip→knee | 11→23→25 |
| `right_hip` | shoulder→hip→knee | 12→24→26 |
| `left_shoulder` | hip→shoulder→elbow | 23→11→13 |
| `right_shoulder` | hip→shoulder→elbow | 24→12→14 |
| `torso_lean` | hip-midpoint→shoulder-midpoint→vertical | synthetic |

`torso_lean` is the angle between the hip-to-shoulder axis and the world vertical (Y axis). It distinguishes standing-upright (≈0°) from folded-forward (≈90°) from plank-horizontal (≈90° in a different axis pattern). In practice, it is implemented as `acos(dot(hipToShoulder.normalize(), UP_VECTOR))` where `UP_VECTOR = (0,1,0)`.

### 2.3 Pose template format — inline JSON, single-file portable

Templates live as a JS object literal inside the `<script>` block of the demo file. A sibling `poses.json` would break the single-file portability that makes demos easy to share and locally serve. The inline approach imposes no additional HTTP request and no CORS constraint.

**Schema** (one template per pose):

```js
{
  id: "tadasana",              // machine-readable ID, localStorage key fragment
  name_en: "Mountain Pose",    // English common name
  name_sa: "Tāḍāsana",        // Sanskrit with diacritics
  stage: 1,                    // position in the Sun Salutation A sequence (1-8)
  joint_targets: {
    left_elbow:    { angle_deg: 180, tolerance_deg: 15, weight: 0.5 },
    right_elbow:   { angle_deg: 180, tolerance_deg: 15, weight: 0.5 },
    left_knee:     { angle_deg: 180, tolerance_deg: 10, weight: 1.0 },
    right_knee:    { angle_deg: 180, tolerance_deg: 10, weight: 1.0 },
    left_hip:      { angle_deg: 180, tolerance_deg: 12, weight: 0.8 },
    right_hip:     { angle_deg: 180, tolerance_deg: 12, weight: 0.8 },
    torso_lean:    { angle_deg:   0, tolerance_deg: 12, weight: 1.2 },
  },
  instruction: "Stand tall. Feet hip-width, weight even. Arms relaxed at your sides. Lengthen through the crown.",
  min_hold_s: 3,               // seconds the pose must be held to count as completed
}
```

**Schema decisions:**

- `tolerance_deg` is the half-width of the pass band. An angle within `[target - tolerance, target + tolerance]` contributes full score for that joint. Beyond the tolerance band the score degrades linearly to zero at `target ± (tolerance * 3)`, then clamps to zero. This linear-outside-band behaviour prevents cliff edges where being 16° off scores identically to 90° off.

- `weight` carries the importance signal. High-weight joints (torso_lean 1.2, knees 1.0) dominate the aggregate score. Low-weight joints (elbows 0.5 in Tadasana, where arm position is relaxed) have less influence. A weight of 0 would mask a joint entirely — used when the joint is not visible (see §2.7 graceful degradation).

- `min_hold_s` is per-template. Tadasana and Uttanasana are grounding poses that benefit from a 3-second hold. Chaturanga is a strength pose where 2 seconds is already challenging. The value lives in the template, not as a global constant, so future operators can tune it per pose without touching logic.

- There is no `max_hold_s`. Holding a pose longer than `min_hold_s` does not penalise the score.

**Why `tolerance_deg` over explicit pass/fail thresholds.** A binary pass/fail at a hard threshold creates a jarring UX: the alignment bar slams between 0% and 100% at a single degree of motion. Linear-outside-band degradation provides smooth visual feedback that guides the user toward the target incrementally.

### 2.4 Scoring formula

Per-frame alignment score for pose *p*, given measured angle `θ_j` at joint *j*:

```
delta_j = |θ_j  −  target_j.angle_deg|

band_score_j =
    1.0                                           if delta_j ≤ tolerance_j
    1.0 − (delta_j − tolerance_j) / (2 * tolerance_j)   if delta_j ≤ 3 * tolerance_j
    0.0                                           otherwise

raw_score_p = Σ_j ( weight_j * band_score_j ) / Σ_j ( weight_j )

alignment_score_p = clamp(raw_score_p, 0.0, 1.0)
```

`alignment_score_p` is a value in [0, 1]. Displayed in the `#yoga-panel` as an integer percentage (0–100) with one decimal place for the progress ring to animate smoothly.

**Hold-time component.** The classifier reports a pose as *completed* when two conditions are simultaneously true:
1. The pose has been the argmax classifier output for a contiguous streak of `K = 6` frames (see §2.5).
2. Within that streak, the alignment score has remained above 0.6 (60%) for at least `min_hold_s` seconds.

Completion is a one-shot event per pose per sequence pass. It fires once, advances the sequence indicator, and triggers the audible cue. The user must drop out of the pose and re-enter it to re-trigger completion — this prevents accidental re-completion during a rest pause.

**Why 60% as the hold threshold.** At 60%, the user's joint angles are within the tolerance band on the majority of weighted joints. A strict 80% threshold would frustrate beginners; a lenient 40% threshold would fire on casual near-misses. 60% is consistent with the threshold used in the Google ML Kit PoseClassifier sample and the Perez-Testor study's reported inter-practitioner variance (mean joint-angle SD of ~10° across joints, which maps to roughly a 30% score drop relative to a perfect practitioner on a 15° tolerance band).

**Why not include a velocity component (punish fast transitions).** Velocity would require a second derivative of the landmark positions, which is already noisy from MediaPipe jitter even after the one-euro filter. Minimum hold time (2–3 s) implicitly penalises rushing through poses without adding noise sensitivity.

### 2.5 Pose classification flow and debounce

Every frame, after `ingestPoseLandmarks()` populates `liveKp`:

```js
function classifyPose() {
    if (!yogaMode.enabled || !liveValid) return;
    computeJointAngles();        // fills yogaMode.angles from liveKp
    for (const p of yogaMode.activePoses) {
        p.frameScore = scorePose(p);   // per-frame alignment_score_p
    }
    const best = yogaMode.activePoses.reduce((a, b) =>
        b.frameScore > a.frameScore ? b : a
    );
    if (best.frameScore > SCORE_NO_POSE_FLOOR) {
        yogaMode.streak = (yogaMode.candidate === best.id)
            ? yogaMode.streak + 1 : 1;
        yogaMode.candidate = best.id;
    } else {
        yogaMode.streak = 0;
        yogaMode.candidate = null;
    }
    if (yogaMode.streak >= K_FRAMES && yogaMode.candidate !== yogaMode.current) {
        yogaMode.current = yogaMode.candidate;
        onPoseTransition(yogaMode.current);
    }
    updateYogaHUD();
}
```

**K = 6 frames** (debounce depth). At 30 Hz this corresponds to a 200 ms lag from first matching pose to classification announcement. This is long enough to suppress a one-frame flicker from a mediocre landmark result but short enough to feel instantaneous to a human moving at yoga pace (typical transition speed: 1–3 seconds).

Lowering K to 3 creates flickering when the user is near a pose boundary. Raising K to 12 introduces a 400 ms lag that makes the HUD feel unresponsive on quick transitions (e.g. Uttanasana → Ardha Uttanasana takes ~1 second in a vigorous practice). K = 6 is the correct value given the ~30 Hz landmark update rate.

**SCORE_NO_POSE_FLOOR = 0.40.** If no pose scores above 40%, yoga-mode reports "no recognised pose" and does not transition. This prevents the classifier from latching onto the closest-matching pose during, say, walking across the room or sitting at a desk. At 40%, at least a plurality of the weighted joints must be within their tolerance band — a constraint that a non-yoga posture reliably fails.

### 2.6 UI surfaces

**Toggle in `#helpers` panel.** Added below the adam-mode row:

```html
<label class="yoga-toggle">
    <input type="checkbox" id="yoga-mode-toggle">
    <span>yoga-mode (instructional)</span>
    <span class="swatch" style="color: var(--green)"></span>
</label>
```

yoga-mode is orthogonal to adam-mode: both can be active simultaneously. It uses `data-yoga="on"` on `<body>`, not `data-theme`. The attribute is distinct so that CSS selectors like `:root[data-theme="adam"]` and `:root[data-yoga="on"]` compose without conflict.

**`#yoga-panel` — bottom-centre overlay.** A new `<div id="yoga-panel" class="panel">` appears at the bottom centre of the viewport when yoga-mode is enabled. It is hidden (`display: none`) when yoga-mode is off, so it does not interfere with the existing layout.

The panel contains:

1. **Current pose name** — large (18px), Sanskrit name above English name below, amber colour. Falls back to "—" when no pose is recognised.
2. **Alignment score ring** — a small SVG `<circle>` progress ring (r=22, stroke-dasharray) updating on every classified frame. Score 0–100 shown as integer inside the ring.
3. **Hold-time progress bar** — a `<div class="bar-track">` identical in style to the CSI bars, filling from 0% to 100% as the hold-time accumulates. Resets on pose transition.
4. **Instruction text** — one line from the current pose's `instruction` field, `font-size: 10px`, `color: var(--text-mute)`.
5. **Visibility warning** — a `<span class="yoga-warn">` shown in `var(--red)` when `torso_not_visible` is true (see §2.7).

**Sequence strip — top-centre.** A horizontal strip of 8 thumbnail slots (`<div class="yoga-strip">`) spanning the top of the viewport (z-index above the titlecard, below `#info`). Each slot contains the pose's stage number and a 3-letter abbreviation (TAD, URD, UTT, ARD, CAT, UPD, DOG, UT2). Slots are styled:

- **Dimmed** (opacity 0.3, `var(--text-mute)` text) — not yet reached.
- **Active** (opacity 1.0, `var(--amber)` border glow, pulsing) — current pose.
- **Completed** (opacity 0.7, `var(--green)` checkmark `✓`, no glow) — held for `min_hold_s` seconds.

The strip does not scroll. Eight slots at ~90px each fit a 720px-wide viewport. On narrower screens the strip compresses gracefully because the slots use `flex: 1` within a `display: flex` container.

**Audible cue.** A single `<audio id="yoga-bell" src="data:audio/wav;base64,..." preload="auto">` element. The WAV is a 0.4-second C5 bell tone encoded inline as base64 (~12 KB). This preserves the single-file portability. It fires once on pose completion via `yogaBell.currentTime = 0; yogaBell.play()`. A `muted` toggle in `#helpers` (beneath the yoga-mode checkbox) allows the user to silence it: `<label><input type="checkbox" id="yoga-mute-toggle"> mute bell</label>`. The bell is muted by default (`yogaBell.muted = true`) to avoid startling first-time users.

**Theme compatibility.** `#yoga-panel` and the sequence strip use only existing custom properties: `var(--bg-panel)`, `var(--border)`, `var(--amber)`, `var(--amber-hot)`, `var(--text)`, `var(--text-mute)`, `var(--green)`, `var(--red)`. No new CSS variables are introduced. The panel therefore inherits both the default dark theme and adam-mode automatically — the same mechanism described in ADR-169 §2.1.

### 2.7 Camera / MediaPipe assumptions and graceful degradation

**Expected input:** front-facing camera, full body from head to ankles in frame, neutral indoor lighting. The demo's existing camera pipeline already requests `{ video: { facingMode: 'user', width: 640, height: 480 } }`. No change to the MediaPipe setup.

**Graceful degradation when body is partially out of frame.** MediaPipe assigns a `visibility` score in [0, 1] to each landmark. When a landmark's visibility drops below 0.35, yoga-mode treats that joint as missing:

```js
function effectiveWeight(jointName, angles) {
    const vis = jointVisibility(jointName);   // min visibility of the 3 landmarks
    if (vis < 0.35) return 0.0;              // joint masked — not counted
    if (vis < 0.65) return angles.weight * (vis / 0.65);   // partial weight
    return angles.weight;
}
```

When two or more of the high-weight joints (knees, hips, torso_lean) are masked simultaneously, `Σ_j(weight_j)` falls below a minimum viable total, and `alignment_score_p` is set to 0 regardless of the numerator. This prevents spurious high scores from a partially visible body where only one or two low-weight joints (e.g. elbows) are visible and happen to match a pose.

The `#yoga-panel` surfaces a `torso_not_visible` warning ("Move back — full body not in frame") in `var(--red)` whenever `liveVis[23] < 0.35 || liveVis[24] < 0.35` (left or right hip not visible). The hips are the reference joint for torso_lean and for hip-angle computation; their absence makes the entire classifier unreliable.

### 2.8 Cross-demo applicability

**yoga-mode ships in demo 05 only for the first iteration.** Demos 03 and 04 do not have a MediaPipe pipeline; there are no `liveKp` landmarks to score. Adding yoga-mode to them would require pulling in the entire MediaPipe Pose Heavy CDN script — changing those demos' character and load time.

**New demo: `06-yoga-mode.html`.** A new file `examples/three.js/demos/06-yoga-mode.html` is introduced as a slimmed-down variant of demo 05 where yoga-mode is the primary focus rather than an optional overlay. Differences from demo 05:

- The CSI panel (`#csi`) and the tomography sweep are hidden by default (`display: none`).
- The `#yoga-panel` is expanded to a larger centre-screen layout with a bigger score ring (r=44) and larger pose name text (24px).
- The sequence strip is rendered larger (100px slot width).
- The `#helpers` panel shows only the yoga-related toggles (yoga-mode, adam-mode, mute bell).
- The titlecard text reads "RuView · Yoga Mode".

This file is created from a copy of demo 05 with the CSI and tomography sections stripped. It shares the `YogaMode` object and pose templates verbatim — no logic is duplicated.

The decision to introduce a sixth demo file rather than making demo 05's yoga features more prominent is: demo 05 is a complete multi-feature demo (CSI + MediaPipe + IK retarget); demo 06 is a single-purpose instructional demo. Evaluators who want to show the yoga system without the RF sensing noise get demo 06.

### 2.9 Persistence

User settings are persisted in `localStorage` under the `ruview.yoga.*` namespace:

| Key | Type | Value shape | Default |
|---|---|---|---|
| `ruview.yoga.enabled` | boolean string | `"true"` or `"false"` | `"false"` |
| `ruview.yoga.muted` | boolean string | `"true"` or `"false"` | `"true"` |
| `ruview.yoga.tolerance_scale` | float string | `"0.5"` to `"2.0"` | `"1.0"` |
| `ruview.yoga.sequence` | JSON string | `["tadasana","urdhva_hastasana",…]` | full 8-pose sequence |

`tolerance_scale` is a global multiplier applied to every `tolerance_deg` value in every template. A scale of 0.5 makes the classifier strict (tight bands); a scale of 2.0 makes it forgiving (wide bands). The HUD exposes this as a simple "Difficulty" slider: Easy (2.0×), Normal (1.0×), Strict (0.5×). The default is Normal.

`ruview.yoga.sequence` allows an operator to load a custom subset or reordering of the 8 poses, or to load additional poses added via `YogaMode.addPose()`. The array contains pose `id` strings. On load, yoga-mode resolves each ID against the registered template map; unknown IDs are skipped with a console warning.

All `localStorage` accesses are wrapped in try/catch to handle privacy-restricted origins.

### 2.10 JS API surface

yoga-mode exposes a clean internal module object. Because the demo is a single-file HTML with no ES module bundler, the pattern is a plain object literal assigned to a local `const`:

```js
const YogaMode = {
    // ---- Lifecycle ----
    init(opts = {}) {},       // wire up UI, register pose templates, restore localStorage
    enable() {},              // set data-yoga="on", show #yoga-panel, start classifying
    disable() {},             // remove data-yoga="on", hide #yoga-panel, reset state

    // ---- Classification callbacks ----
    onPoseChanged(cb) {},     // cb(poseId: string | null) — fires on confirmed transition
    onPoseScored(cb) {},      // cb(scores: {[poseId]: number}) — fires every frame
    onPoseCompleted(cb) {},   // cb(poseId: string, holdMs: number) — fires on hold completion

    // ---- Template management ----
    addPose(template) {},     // validate and register a custom pose template
    removePose(id) {},        // remove a template by id (built-ins can be removed)
    poses() {},               // returns Array<PoseTemplate> — current registered set

    // ---- State accessors ----
    currentPose() {},         // returns current confirmed pose id or null
    currentScore() {},        // returns alignment score [0,1] of current pose or 0
    angles() {},              // returns the latest computed joint angles object

    // ---- Sequence control ----
    resetSequence() {},       // clears all completion state, restarts from stage 1
    setSequence(ids) {},      // replace active sequence with a custom id array

    // Internal state — not part of the public API:
    _state: { enabled, candidate, current, streak, holdStart, completedSet }
};
```

`onPoseChanged`, `onPoseScored`, and `onPoseCompleted` follow the same pattern as the demo's existing event hooks: they register a single callback (last-writer wins, not an array). This is sufficient for a single-file demo where there is at most one consumer per event. A future multi-listener pattern would need a `listeners` array; that is out of scope.

`addPose(template)` validates the template schema before registering it. A template missing `joint_targets` or with an `id` that contains non-alphanumeric characters is rejected with a `console.error` and returns `false`. Valid templates return `true`.

### 2.11 Pose templates — Sun Salutation A joint targets

The full 8-pose template set. Angle targets are derived from Perez-Testor et al. (2019) Vicon measurements and Iyengar alignment geometry. Tolerances are set to twice the reported inter-practitioner SD (~10°) rounded to the nearest 5°, then scaled by the user's `tolerance_scale`.

**Stage 1 — Tāḍāsana (Mountain Pose)**

All joints extended. Body in anatomical position. Baseline for comparison.

```js
{ id: "tadasana", name_en: "Mountain Pose", name_sa: "Tāḍāsana", stage: 1,
  min_hold_s: 3,
  joint_targets: {
    left_knee:    { angle_deg: 180, tolerance_deg: 10, weight: 1.0 },
    right_knee:   { angle_deg: 180, tolerance_deg: 10, weight: 1.0 },
    left_hip:     { angle_deg: 180, tolerance_deg: 12, weight: 0.8 },
    right_hip:    { angle_deg: 180, tolerance_deg: 12, weight: 0.8 },
    torso_lean:   { angle_deg:   0, tolerance_deg: 10, weight: 1.2 },
    left_elbow:   { angle_deg: 180, tolerance_deg: 20, weight: 0.4 },
    right_elbow:  { angle_deg: 180, tolerance_deg: 20, weight: 0.4 },
  },
  instruction: "Stand tall. Feet hip-width, weight even. Arms at sides. Lengthen through the crown.",
}
```

**Stage 2 — Ūrdhva Hastāsana (Upward Salute)**

Arms sweep overhead. Shoulders maximally abducted. Distinguishing feature: both elbows extended and arms overhead (shoulder angle approaches 180° abduction). Perez-Testor reports shoulder elevation of 183° at peak overhead position.

```js
{ id: "urdhva_hastasana", name_en: "Upward Salute", name_sa: "Ūrdhva Hastāsana", stage: 2,
  min_hold_s: 2,
  joint_targets: {
    left_shoulder:  { angle_deg: 165, tolerance_deg: 20, weight: 1.2 },
    right_shoulder: { angle_deg: 165, tolerance_deg: 20, weight: 1.2 },
    left_elbow:     { angle_deg: 180, tolerance_deg: 15, weight: 0.8 },
    right_elbow:    { angle_deg: 180, tolerance_deg: 15, weight: 0.8 },
    left_knee:      { angle_deg: 180, tolerance_deg: 12, weight: 0.8 },
    right_knee:     { angle_deg: 180, tolerance_deg: 12, weight: 0.8 },
    torso_lean:     { angle_deg:   0, tolerance_deg: 15, weight: 0.7 },
  },
  instruction: "Inhale. Sweep arms overhead. Palms face each other. Gaze forward or slightly up.",
}
```

**Stage 3 — Uttānāsana (Standing Forward Fold)**

Deep hip flexion. Torso approaches vertical-inverted. Perez-Testor reports hip flexion of 134°. The angle at the hip joint as computed by our triplet (shoulder→hip→knee) goes to ~30° as the torso folds toward the legs. Knees remain extended.

```js
{ id: "uttanasana", name_en: "Standing Forward Fold", name_sa: "Uttānāsana", stage: 3,
  min_hold_s: 3,
  joint_targets: {
    left_hip:    { angle_deg:  40, tolerance_deg: 25, weight: 1.2 },
    right_hip:   { angle_deg:  40, tolerance_deg: 25, weight: 1.2 },
    left_knee:   { angle_deg: 175, tolerance_deg: 15, weight: 1.0 },
    right_knee:  { angle_deg: 175, tolerance_deg: 15, weight: 1.0 },
    torso_lean:  { angle_deg:  85, tolerance_deg: 20, weight: 1.0 },
  },
  instruction: "Exhale. Fold forward from the hips. Let the crown of the head drop toward the floor.",
}
```

**Stage 4 — Ardha Uttānāsana (Half Lift / Flat-Back)**

Torso lifts to horizontal. Hip angle opens to ~90°. Spine neutral. This is the most distinctive pose for classification: it is the only one where the torso is neither upright nor fully folded — the `torso_lean` angle is ~90° and the hips are also ~90°. Perez-Testor reports the half-lift as an intermediate transition posture; the distinguishing cue is the simultaneous hip angle and spine neutral (not flexed).

```js
{ id: "ardha_uttanasana", name_en: "Half Lift", name_sa: "Ardha Uttānāsana", stage: 4,
  min_hold_s: 2,
  joint_targets: {
    left_hip:    { angle_deg:  90, tolerance_deg: 20, weight: 1.2 },
    right_hip:   { angle_deg:  90, tolerance_deg: 20, weight: 1.2 },
    left_knee:   { angle_deg: 175, tolerance_deg: 12, weight: 0.8 },
    right_knee:  { angle_deg: 175, tolerance_deg: 12, weight: 0.8 },
    torso_lean:  { angle_deg:  90, tolerance_deg: 15, weight: 1.2 },
    left_elbow:  { angle_deg: 180, tolerance_deg: 20, weight: 0.5 },
    right_elbow: { angle_deg: 180, tolerance_deg: 20, weight: 0.5 },
  },
  instruction: "Inhale. Lift the chest. Flat back. Fingertips on the shins or floor. Gaze forward.",
}
```

**Stage 5 — Catvāri / Chaturanga Daṇḍāsana (Four-Limbed Staff)**

Plank lowered. Elbows at 90°. Body horizontal. This is the hardest pose to classify from a front-facing camera alone: the body is horizontal and the depth axis is ambiguous. The key discriminator is `elbow_angle ≈ 90°` combined with `hip ≈ 180°` (no flexion) and `torso_lean ≈ 90°`. Note: from a front-facing camera, a person in Chaturanga facing the camera appears foreshortened. yoga-mode accepts this limitation and primarily tracks Chaturanga as the transition between Ardha Uttanasana and Upward Dog in the sequence, with lower weight on spatial cues and higher weight on elbow angle. Iyengar geometry specifies elbows at 90° against the body.

```js
{ id: "chaturanga", name_en: "Four-Limbed Staff", name_sa: "Catvāri / Chaturanga Daṇḍāsana", stage: 5,
  min_hold_s: 2,
  joint_targets: {
    left_elbow:   { angle_deg:  90, tolerance_deg: 20, weight: 1.5 },
    right_elbow:  { angle_deg:  90, tolerance_deg: 20, weight: 1.5 },
    left_hip:     { angle_deg: 175, tolerance_deg: 15, weight: 0.8 },
    right_hip:    { angle_deg: 175, tolerance_deg: 15, weight: 0.8 },
    left_knee:    { angle_deg: 175, tolerance_deg: 15, weight: 0.6 },
    right_knee:   { angle_deg: 175, tolerance_deg: 15, weight: 0.6 },
    torso_lean:   { angle_deg:  90, tolerance_deg: 20, weight: 0.7 },
  },
  instruction: "Lower down. Elbows at 90°, hugged to the ribs. Body in one straight line.",
}
```

**Stage 6 — Ūrdhva Mukha Śvānāsana (Upward-Facing Dog)**

Hips extend, spine extends (backbend), shoulders over wrists, knees off floor. Distinguishing feature: hips are near 160–180° (extended), which is the opposite of Uttanasana's deep flexion. The `torso_lean` reverses from ~90° horizontal to approaching 0° or slightly past vertical (slight backbend). Perez-Testor's spine extension of 44° is the reference for the backbend component; the hip angle opens to near-full extension.

```js
{ id: "urdhva_mukha_svanasana", name_en: "Upward-Facing Dog", name_sa: "Ūrdhva Mukha Śvānāsana", stage: 6,
  min_hold_s: 2,
  joint_targets: {
    left_hip:     { angle_deg: 165, tolerance_deg: 20, weight: 1.2 },
    right_hip:    { angle_deg: 165, tolerance_deg: 20, weight: 1.2 },
    left_elbow:   { angle_deg: 170, tolerance_deg: 20, weight: 0.8 },
    right_elbow:  { angle_deg: 170, tolerance_deg: 20, weight: 0.8 },
    left_knee:    { angle_deg: 170, tolerance_deg: 20, weight: 0.6 },
    right_knee:   { angle_deg: 170, tolerance_deg: 20, weight: 0.6 },
    torso_lean:   { angle_deg:  15, tolerance_deg: 20, weight: 0.8 },
  },
  instruction: "Press the tops of the feet down. Lift the chest. Shoulders away from the ears. Gaze forward.",
}
```

**Stage 7 — Adho Mukha Śvānāsana (Downward-Facing Dog)**

Hips high. Inverted V. The most geometrically distinct pose in the sequence: high hips, extended knees, arms overhead-ish (shoulder angle ~150° relative to torso), torso_lean ~90° but in the opposite direction to Chaturanga (body weight shifted back over the heels). The hip angle as measured by our shoulder→hip→knee triplet is ~80–110° (the pelvis is high, creating a roughly right-angle fold at the hip). Perez-Testor reports the hip-angle transition from Chaturanga to Downward Dog as the largest single-frame angle change in the sequence (~120° excursion), making it the easiest pose to classify correctly.

```js
{ id: "adho_mukha_svanasana", name_en: "Downward-Facing Dog", name_sa: "Adho Mukha Śvānāsana", stage: 7,
  min_hold_s: 5,
  joint_targets: {
    left_hip:     { angle_deg:  90, tolerance_deg: 25, weight: 1.2 },
    right_hip:    { angle_deg:  90, tolerance_deg: 25, weight: 1.2 },
    left_knee:    { angle_deg: 180, tolerance_deg: 15, weight: 1.0 },
    right_knee:   { angle_deg: 180, tolerance_deg: 15, weight: 1.0 },
    left_shoulder: { angle_deg: 150, tolerance_deg: 25, weight: 0.8 },
    right_shoulder: { angle_deg: 150, tolerance_deg: 25, weight: 0.8 },
    torso_lean:   { angle_deg:  90, tolerance_deg: 20, weight: 0.7 },
  },
  instruction: "Hips up and back. Heels reaching toward the floor. Arms and ears in one line. Breathe.",
}
```

**Stage 8 — Uttānāsana (Standing Forward Fold, return)**

Identical to stage 3 in geometry. Classified as stage 8 for sequence-tracking purposes only — same template joint targets, different `id` and `stage` value.

```js
{ id: "uttanasana_return", name_en: "Standing Forward Fold (return)", name_sa: "Uttānāsana", stage: 8,
  min_hold_s: 2,
  joint_targets: { /* same as stage 3 */ },
  instruction: "Step or jump to the front. Exhale. Release the head. Return to stillness.",
}
```

Distinguishing stages 3 and 8 is handled by the sequence-tracking layer, not by the classifier. If yoga-mode is in stage 7 (Downward Dog) and detects a forward-fold shape, it advances to stage 8 rather than regressing to stage 3. If yoga-mode is in stages 1–2 and detects a forward-fold shape, it advances to stage 3. The sequence tracks forward direction only; there is no backward regression in the first implementation.

### 2.12 Test plan

**Manual — live camera:**
Stand in front of the workstation USB camera (ruvzen, confirmed front-facing in CLAUDE.local.md). Enable yoga-mode from `#helpers`. Cycle through all 8 poses in order. For each pose: verify the HUD shows the correct Sanskrit and English name within 2 frames (~67 ms) of entering the pose, the alignment score exceeds 60%, and the sequence strip advances. Verify no pose is misclassified when standing in a casual at-rest position (score should be below 40% floor for all 8 poses).

**Synthetic — test mode triggered by `?test=1` URL parameter:**
When `location.search` includes `test=1`, yoga-mode enters a headless test mode: instead of reading from `liveKp`, it reads from a pre-recorded `YOGA_TEST_FIXTURES` object — one synthetic landmark array per pose, generated at authoring time by capturing the real `liveKp` values during a manual demo session.

```js
if (new URLSearchParams(location.search).has('test')) {
    for (const fixture of YOGA_TEST_FIXTURES) {
        ingestPoseLandmarks(fixture.landmarks);
        classifyPose();
        const result = YogaMode.currentPose();
        console.assert(result === fixture.expected_id,
            `FAIL: ${fixture.expected_id} got ${result}`);
    }
    console.log('YogaMode tests complete');
}
```

The fixture set is 8 entries (one per pose). Each entry is a hard-coded `landmarks` array of 33 objects with `{x, y, z, visibility}` values. These fixtures are inlined in the `<script>` block, gated behind `if (urlParams.has('test'))` so they are never executed in normal operation.

**Negative test:** A ninth fixture entry with the user standing in a neutral at-rest position (arms at sides but knees slightly bent, casual posture — not a yoga pose). Assert `YogaMode.currentPose() === null` (no pose above the 0.40 floor).

**Regression guard for joint-angle computation:** A tenth fixture that hard-codes known landmark positions forming a right angle at the left knee (three points forming a precise 90° angle). Assert `YogaMode.angles().left_knee` is within ±0.5° of 90.

### 2.13 Rejected alternatives

| Alternative | Rejected because |
|---|---|
| Train a custom MLP on a labelled yoga dataset | No labelled dataset in this codebase. Training requires hundreds of examples per class, a train/test pipeline, and a serialized model file — all incompatible with a single-file demo. Joint-angle matching achieves equivalent discrimination for 8 geometrically distinct poses with zero training data. |
| Use a paid SaaS pose-classification API (e.g. a commercial yoga scoring cloud service) | Introduces an external network dependency, a per-request cost, and a privacy concern (camera frames leaving the browser). Pure client-side is a hard requirement. |
| Ship audio/video instructional content (video of an instructor demonstrating each pose) | Massively increases the demo's asset footprint. A single instructor video per pose at 15 fps, 10 seconds, compressed, is ~500 KB × 8 = 4 MB minimum. The inline base64 bell (~12 KB) is the correct granularity of embedded media for this demo. |
| Ship a backend yoga-tracking session record (store per-session completion data to a server) | No backend endpoint exists or is planned for the demos. Client-only; persistence via `localStorage`. |
| Integrate with the Rust `pose_tracker.rs` now | Convention mismatch (BlazePose-33 vs COCO-17) documented in §1.3 but the cost of bridging it outweighs the benefit for a demo. The bridge is deferred: yoga-mode in JS is valuable without it. Rust integration becomes tractable once a WebSocket protocol for streaming joint angles (not raw CSI) from the sensing server is defined — a separate ADR. |
| Use MediaPipe Tasks `PoseLandmarker` with a built-in `PoseClassifier` task | The Tasks API requires loading a `.task` bundle (~4 MB) from CDN at runtime. Demo 05 already uses the older `@mediapipe/pose@0.5` CDN script; switching APIs would require rewriting the entire landmark ingest pipeline. The classifier task is a black box undebuggable in DevTools. Threshold matching is fully transparent. |
| Put yoga-mode on `data-theme` alongside adam-mode | yoga-mode is not a theme — it is a feature toggle. Mixing it with the theme attribute would prevent simultaneous adam-mode + yoga-mode activation and would conflate presentation with functionality. Separate `data-yoga="on"` attribute is the correct model. |

---

## 3. Consequences

### 3.1 Positive

- The retargeting pipeline in demo 05 gains a per-pose regression test harness (`?test=1`) at no additional tooling cost.
- yoga-mode operates on the existing `liveKp` stream — zero additional CPU cost beyond a few arctangent calls per frame (~50 µs at 30 Hz).
- The pose-scoring formula is fully deterministic and inspectable: `console.log(YogaMode.angles())` in DevTools shows every joint angle on every frame.
- Demo 06 provides a clean instructional-first presentation that separates yoga-mode from the RF sensing visualisations, making the feature accessible to a fitness-context audience.
- The `YogaMode.addPose()` API allows operators to extend the template library without touching core logic — enabling future pose sets (Warrior series, Yin postures) as a follow-on.
- The `tolerance_scale` persistence allows the same demo codebase to serve both beginners (2× tolerance) and experienced practitioners (0.5× tolerance) without code changes.

### 3.2 Negative

- Two HTML files to maintain (`05` and `06`) where previously there was one. Mitigated by the fact that yoga-mode logic is identical between them — demo 06 is a layout variant, not a code fork.
- Chaturanga Dandasana classification is inherently degraded from a front-facing camera (the body is horizontal; the depth axis is ambiguous). The classifier can detect the pose if the user faces the camera sideways (profile view), but the existing camera setup on ruvzen is front-facing. This is a known limitation, documented in the instruction text ("face the camera from the side for best Chaturanga detection").
- The inline base64 bell WAV adds ~12 KB to the HTML file size. Negligible at the scale of the demo but noted.
- `localStorage` namespace `ruview.yoga.*` adds four keys per origin. No conflict with `ruview.theme` from adam-mode.

### 3.3 Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| MediaPipe visibility scores are unreliable for floor-level landmarks (ankles, feet) during Dog poses | Medium | `effectiveWeight()` already masks low-visibility joints; Dog-pose templates weight knees (visible) more than ankles (may be occluded). |
| The `?test=1` fixture landmarks become stale if the coordinate-space transform in `ingestPoseLandmarks()` changes | Low | Fixtures store raw `liveKp` world-space values, not normalized MediaPipe coords. If `ingestPoseLandmarks()` changes its output schema, the fixtures will produce obviously wrong joint angles in the assertion step — the failure is loud, not silent. |
| Sequence-strip animation (CSS pulsing glow on the active stage) triggers repaint on every frame at 30 Hz | Low | The pulse is a CSS `animation` on `opacity` — composited by the GPU, no layout reflow. Negligible cost. |
| User's camera position cuts off the hips (e.g. laptop on a desk) — `torso_not_visible` fires immediately | High for laptop use | The warning instructs the user to step back. This is the correct behaviour. Future: add a "camera too close" heuristic based on the ratio of shoulder distance to image width. |
| Stage 8 (Uttanasana return) is classified identically to stage 3 by the angle classifier alone — the sequence layer must correctly disambiguate them | Medium | The sequence-tracking layer uses monotonic forward-only progression. Stage 3 can only fire when the current sequence position is 2 (after Urdhva Hastasana); stage 8 can only fire when the current sequence position is 7 (after Downward Dog). The classifier produces the angle score; the sequence layer decides which stage to credit. If the user skips a pose, the sequence layer waits — it does not leap to stage 8 from stage 2 even if a forward-fold shape is detected. |

---

## 4. Implementation plan

Moderate scope — two HTML files, no build step, no new external dependencies.

1. **Define the `YOGA_POSES` array** — 8 template objects as specified in §2.11, inline in the `<script>` block of demo 05.
2. **Implement `computeJointAngles()`** — read from the existing `liveKp` array, fill a `yogaAngles` object using the 9 joint triplets in §2.2.
3. **Implement `scorePose(template)`** — the weighted-sum formula from §2.4, respecting `effectiveWeight()` for visibility masking.
4. **Implement `classifyPose()`** — argmax with K=6 debounce as in §2.5; call from the existing `requestAnimationFrame` loop after `applyRetargeting()`.
5. **Add `#yoga-panel` markup and CSS** — bottom-centre panel, score ring, hold-time bar, instruction text, visibility warning. All styles via existing custom properties.
6. **Add the sequence strip** — `#yoga-strip` top-centre, 8 flex slots, 3-state styling (dimmed/active/completed).
7. **Wire the `#helpers` toggle** — `yoga-mode-toggle` checkbox and `yoga-mute-toggle` checkbox; `localStorage` persistence.
8. **Add `YogaMode` object** — wrapping steps 1–7 with the API surface from §2.10.
9. **Add `YOGA_TEST_FIXTURES` and the `?test=1` harness** — 10 fixture entries (8 positive, 1 negative, 1 angle-computation).
10. **Create `06-yoga-mode.html`** — copy of demo 05 with CSI/tomography sections hidden, larger yoga panel layout.
11. **Manual validation** — stand in front of ruvzen camera, cycle all 8 poses, verify classification and sequence advancement.

Acceptance criteria:

- All 8 poses classified correctly in the `?test=1` synthetic harness (assertions pass with no console errors).
- The negative fixture (casual stand) produces `currentPose() === null`.
- The angle-computation fixture (`left_knee` at a known 90°) asserts within ±0.5°.
- Manual: each of the 8 Sun Salutation A poses classified within 2 frames when held correctly.
- Alignment score exceeds 60% when the user matches the pose by self-assessment.
- Sequence strip advances in order; completed poses show green checkmark.
- Bell fires on completion (when unmuted).
- adam-mode + yoga-mode simultaneously active: both panels visible, correct theme.
- `localStorage` persists enabled-state and tolerance-scale across page reloads.

---

## 5. Related ADRs

| ADR | Relationship |
|---|---|
| [ADR-169](ADR-169-adam-mode-light-theme.md) | Sibling demo-side feature. yoga-mode toggle lives in the same `#helpers` panel. Both are orthogonal and must compose. |
| [ADR-019](ADR-019-sensing-only-ui-mode.md) | Sensing-only UI — yoga-mode is the opposite: camera-first, sensing secondary. |
| [ADR-035](ADR-035-live-sensing-ui-accuracy.md) | Live sensing UI accuracy norms. yoga-mode scores the user's body against templates, not CSI accuracy — but the same principle of not misrepresenting measurement quality applies. |
| [ADR-014](ADR-014-sota-signal-processing.md) | The Rust-side `gesture.rs` uses DTW for gesture recognition. yoga-mode explicitly rejects DTW for static pose classification (§2.2). The two systems are complementary: DTW for motion gestures, angle-threshold for static poses. |
| [ADR-029](ADR-029-ruvsense-multistatic-sensing-mode.md) | The Rust `pose_tracker.rs` (COCO-17) that yoga-mode defers integrating with. The COCO↔BlazePose mapping in §1.3 is the foundation for the future bridge. |

---

## 6. References

### Production code
- `examples/three.js/demos/05-skinned-realtime.html` — primary implementation target; `liveKp`, `liveVis`, `ingestPoseLandmarks()`, `#helpers`, `#pose-panel`, `RETARGETS`, `visForRetarget()` are all anchors for yoga-mode integration
- `examples/three.js/demos/04-skinned-fbx.html` — sibling demo; lighting reference
- `v2/crates/wifi-densepose-signal/src/ruvsense/pose_tracker.rs` — Rust COCO-17 tracker; convention mapping in §1.3 of this ADR targets this module

### External references

1. **Perez-Testor, S. et al. (2019).** "Kinematics of Suryanamaskar Using Three-Dimensional Motion Capture." *PMC6521759*. 10 trained practitioners, 12-camera Vicon, 100 Hz, sagittal-plane joint angles for each of the 12 standard Surya Namaskar positions. Primary source for angle targets and tolerance bounds in §2.11.

2. **Chidamber, S. and Harikumar, K. (2023).** "A novel approach for yoga pose estimation based on in-depth analysis of human body joint detection accuracy." *PMC10280249*. Validates joint-angle threshold matching as the dominant reliable real-time method for small-to-medium yoga pose sets; reports average inter-joint angle error of 10.017° across six common daily poses — the empirical basis for the ±10–25° tolerance bands in the templates.

3. **Lugaresi, C. et al. (2020 / MediaPipe team).** "On-device, Real-time Body Pose Tracking with MediaPipe BlazePose." Google Research Blog and arXiv:2006.10204. Defines the 33-landmark BlazePose topology used throughout §1.3 and §2.2. Confirms the landmark visibility score semantics used in §2.7.

4. **Google ML Kit team.** "Pose classification options." developers.google.com/ml-kit/vision/pose-detection/classifying-poses. Documents the `PoseClassifier` EfficientNet approach that this ADR rejects in §2.13; the 60% alignment threshold in §2.4 is consistent with the sample thresholds in this guide.

5. **Iyengar, B.K.S. (2001).** *Light on Yoga* (Schocken Books, revised edition). Chaturanga Dandasana description pp. 102–104: "elbows at right angles along the body" — the 90° elbow target for stage 5. Tadasana pp. 61–63: anatomical position as baseline. The Iyengar descriptions supply angle targets where Perez-Testor's Vicon study does not explicitly report a joint.
