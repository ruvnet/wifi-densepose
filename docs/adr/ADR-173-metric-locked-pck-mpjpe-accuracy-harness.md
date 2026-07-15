# ADR-173: Metric-Locked PCK/MPJPE Accuracy Harness

| Field | Value |
|-------|-------|
| **Status** | Accepted — implemented, deterministically tested |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **METRIC-LOCK** |
| **Amends** | ADR-155 (generalizes the torso-only `metrics_core::pck_canonical` to a selectable normalization) |
| **Motivated by** | `docs/research/sota-nn-train-benchmark-brief.md` (PR #1090) |

## Context

The beyond-SOTA SOTA-research brief (PR #1090) identified the single biggest
threat to any "beyond-SOTA" accuracy claim this project makes: **metric
ambiguity**. Three PCK@20 numbers circulate, computed under three *different and
unstated* normalizations, so they cannot be compared:

- **96.09–96.61%** — WiFlow-STD reproduction, **image/bounding-box-normalized** PCK (the looser convention).
- **81.63%** — an internal MM-Fi number reported as **"torso-PCK"** (tighter).
- **61.1%** — GraphPose-Fi (arXiv 2511.19105), **standard torso-diameter** PCK on the MM-Fi random split (the academic frontier).

The project has been burned by this twice: a previously-published 92.9% was
retracted because it used **absolute-pixel** normalization, not torso. Until
there is *one canonical, documented, tested* PCK definition — and every reported
number carries the definition it was computed under — no accuracy comparison is
credible, and the "prove everything" bar cannot be met for the benchmark half of
the work.

This is measurement infrastructure, not an accuracy claim. The deliverable's job
is to make the metric **unambiguous and reproducible**, so future numbers are
comparable and an unlabeled PCK is structurally impossible.

## Decision

Add a metric-locked accuracy harness as a new module
`v2/crates/wifi-densepose-train/src/accuracy.rs` (404 non-test lines; inline
deterministic tests bring the file to 708), re-exported at the crate root. It
**extends, not duplicates** — it reuses `metrics_core`'s geometric primitives
(`bounding_box_diagonal`, canonical hip indices `CANON_LEFT_HIP/RIGHT_HIP`), so
there remains exactly one implementation of each geometric reference; the
existing ADR-155 `pck_canonical` (torso-only) is unchanged and this generalizes
it.

### Public API

- `enum PckNormalization { TorsoDiameter, BoundingBoxDiagonal, AbsolutePixels(f32) }`
  — the three conventions the three historical numbers used, now **explicit and
  selectable**. `.label()` / `.tolerance(...)`.
- `pck_at(pred, gt, vis, k, norm) -> (correct, total, pck)` — PCK@k =
  fraction of *visible* keypoints whose predicted-vs-GT distance ≤ the tolerance,
  where tolerance = `k%` of the chosen normalizer (or an absolute threshold for
  `AbsolutePixels`).
- `mpjpe(pred, gt, vis) -> f32` — mean per-joint position error (2D/3D, coordinate
  units; mm for mm inputs). Re-exported crate-root as `pck_mpjpe` to avoid
  colliding with the existing `eval::mpjpe`.
- `struct PoseAccuracy { pck_at: BTreeMap<u8,f32>, mpjpe, normalization, n_keypoints, n_frames }`
  — **a reported number always carries its `normalization`**; an unlabeled PCK is
  structurally impossible to produce through this surface.
- `struct PoseFrame { pred, gt, visibility }` + `accuracy_report(frames, ks, norm) -> PoseAccuracy`
  (micro-averaged over keypoints).

### Correctness is proven by hand-computed deterministic tests (no GPU, no data)

The tests construct synthetic keypoint sets whose PCK/MPJPE can be computed by
hand, and assert the harness matches. Highlights (all pass):

| Test | Construction | Expected |
|------|--------------|----------|
| perfect_prediction | pred==gt | PCK=1.0 (all 3 norms), MPJPE=0 |
| all_just_outside | every error just past τ@20 | PCK=0.0 |
| half_in_half_out | 2 exact, 2 just outside | PCK=0.5 |
| **three_normalizations (KEY PROOF)** | identical pred; nose err .06, shoulder .10, hips exact | torso=**0.50**, bbox=**1.00**, abs(.08)=**0.75** |
| mpjpe_2d / mpjpe_3d | (3,4)→5 / (1,2,2)→3 | 2.5 / 3.0 |
| mpjpe_excludes_invisible | invisible joint err 100 ignored | 5.0 |
| zero_torso_unscoreable | coincident hips | `(0,0,0.0)`, **not** false-perfect |
| no_visible_keypoints | vis=∅ | `(0,0,0.0)` |
| nan_coords | one NaN pred coord | counted wrong, **no panic** |
| empty report | no frames | 0.0, **not** NaN |
| bbox≥torso ordering | same frames | bbox-PCK ≥ torso-PCK |

### The key proof (the ambiguity is real and quantified)

Identical predictions, three declared normalizations → **0.50 / 1.00 / 0.75**.
Mechanism: the bbox diagonal `√(0.20² + 0.80²) = 0.825` is ~4× the hip-span torso
`0.20`, so τ@20 is 0.165 (bbox) vs 0.040 (torso) — the looser image-normalized
convention passes joints the strict torso convention rejects. This is *exactly*
why 96% / 81.6% / 61% cannot be lined up without declaring the enum, demonstrated
in-code.

## Validation

- `cargo test -p wifi-densepose-train --no-default-features` → lib **191 → 206**
  (+15), `test_metrics` **12 → 14** (+2), doc-tests 8 — **0 failed**.
- `cargo test --workspace --no-default-features` → **exit 0**, 0 failed.
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash
  `f8e76f21a0f9852b70b6d9dd5318239f6b20cbcb4cdd995863263cecdc446f7a` **unchanged**
  (off the signal proof path — confirms no pipeline alteration).

## Consequences

### Positive
- The three historical PCK numbers can now be **recomputed under one declared
  definition** and compared honestly. The retracted-number class of error
  (silent normalization mismatch) is structurally prevented going forward.
- Establishes the measurement substrate for the beyond-SOTA target: GraphPose-Fi
  cross-environment **PCK@20 = 12.9%** (standard torso PCK) is now a number this
  harness can produce comparably.

### Negative
- None functional. The harness is additive; no existing metric path changed.

### Neutral
- Producing actual model numbers under this harness requires the trained models +
  datasets (MM-Fi) and, for cross-domain splits, is the next sub-deliverable of
  the benchmark/optimization milestone — out of scope here (this ADR is the
  *instrument*, not the *reading*).

## Links
- ADR-155 — metric core (`pck_canonical`, torso-only) — generalized here
- ADR-152 — WiFi-Pose SOTA 2026 intake / WiFlow-STD benchmark
- `docs/research/sota-nn-train-benchmark-brief.md` — the motivating gap analysis
- GraphPose-Fi — arXiv 2511.19105 (verified cross-env PCK@20 = 12.9% anchor)
