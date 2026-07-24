# ADR-187: `archive/v1` Deprecation & Model-Weights Honest Labeling

- **Status**: Accepted
- **Date**: 2026-07-21
- **Deciders**: ruv
- **Tags**: archive-v1, deprecation, densepose-head, model-weights, honest-labeling, prove-everything, credibility, pip-tombstone
- **Refs**: [#509](https://github.com/ruvnet/RuView/issues/509) (missing model weights / reproducibility), [#1125](https://github.com/ruvnet/RuView/issues/1125) ("has anyone got this to work?")
- **Relates to**: [ADR-117](ADR-117-pip-wifi-densepose-modernization.md) (pip modernization + 1.99.0 tombstone), [ADR-160](ADR-160-edge-skill-library-honest-labeling.md) (honest-labeling precedent), [ADR-079](ADR-079-camera-ground-truth-training.md) (camera-supervised pose target), [ADR-152](ADR-152-wifi-pose-sota-2026-intake.md) (WiFlow-STD PCK@20 measurement), [ADR-175](ADR-175-int8-quantization-half-pose-model-measured.md) (int8 pose trade-off), [ADR-101](ADR-101-pose-estimation-cog.md) (pose cog)

---

## Context

Two open GitHub issues are, at root, the same complaint: the project's public surface
lets a reader believe a WiFi→17-keypoint pose model exists and produces real accuracy,
when the specific code they land on cannot back that claim.

- **#509** — a detailed technical review states: *"While the network architecture for
  DensePoseHead is defined in the code, there are no pre-trained weights (.pth or .onnx
  files) available in the repository,"* and questions whether ESP32 1×1 SISO antennas
  can match the multi-antenna NIC research this project is inspired by.
- **#1125** — a user asks for anyone to testify the project actually runs and returns
  real data. A pure credibility complaint.

This ADR follows the **prove-everything / anti-"AI-slop"** directive and the
**honest-labeling** precedent set by ADR-160: the fix is to make the labels TRUE, not
to fabricate a capability. Grading vocabulary (from ADR-152 / ADR-160):

- **MEASURED** — reproduced in this worktree; the file/absence was directly inspected.
- **DATA-GATED** — a real code path exists; honestly flagged where the accuracy is not validated.
- **NO-ACTION (already-honest)** — audited, found correct, cited as a positive.

### What the investigation actually found (MEASURED in this worktree)

The situation is **more nuanced than either issue implies** — worse in one place, and
distinctly *better* in others. Forcing a uniformly negative narrative would itself be
dishonest. The findings:

**1. `archive/v1` — the issue reporter is correct here.**
- `archive/v1/src/models/densepose_head.py` defines `DensePoseHead` (segmentation +
  UV-regression heads). Its `_initialize_weights()` uses **`kaiming_normal_` random
  initialization only** — there is no checkpoint-loading path in the class.
- `Glob archive/v1/**/*.{pth,onnx,safetensors,pt,ckpt,bin}` → **zero files**. There are
  **no trained weights anywhere under `archive/v1/`.** The "architecture defined, no
  weights" claim is TRUE for this tree.
- `archive/v1/README.md` calls the tree "the legacy Python implementation" in a single
  closing note but does **not** loudly warn users off it, and there is **no
  `archive/v1/DEPRECATED.md`.** This is the dead-but-present code that shows up in greps
  and search and reads as if it were the live implementation.
- Per ADR-117, this exact tree is the source of the tombstoned pip package
  `wifi-densepose 1.x` (1.99.0 raises an `ImportError` telling users to migrate). The
  code is already tombstoned *on PyPI* but not *in the repo*.

**2. `v2` (the current, maintained system) — real weights DO exist; the "no weights
anywhere" reading is FALSE at the project level.** Git-tracked, committed checkpoints:
- `v2/crates/cog-pose-estimation/cog/artifacts/pose_v1.safetensors` (507 KB) +
  `pose_v1.onnx` (12 KB) + `train_results.json` — a **real committed 17-keypoint
  model**, trained with Candle on an RTX 5080.
- `v2/crates/cog-person-count/cog/artifacts/count_v1.{safetensors,onnx}` — a committed
  person-count model.
- Externally published on Hugging Face (not committed, but real and released):
  `ruvnet/wifi-densepose-pretrained` (CSI encoder + presence head, honestly re-labeled
  at **82.3% held-out temporal-triplet accuracy** — the older "100% presence" figure was
  already retracted, an existing honest-labeling win) and `ruvnet/wifi-densepose-mmfi-pose`
  (a pose model reporting **82.69% torso-PCK@20** on the MM-Fi `random_split` protocol).
- ADR-152 measurement (a): the *external* WiFlow-STD (DY2434) model was reproduced at
  **96.09% PCK@20** on an RTX 5080 (graded MEASURED-EQUIVALENT). That is an external
  baseline, not RuView's own weights.

**3. The honest gap is narrow and specific — the live, on-device ESP32 17-keypoint
pose path.** Per `v2/crates/cog-pose-estimation/cog/README.md` (already an exemplary
"Honest reading" section):
- The committed `pose_v1` scores **PCK@20 = 3.0% / PCK@50 = 18.5%** on a 217-sample
  holdout — **below the ADR-079 target of PCK@20 ≥ 35%.** It learns coarse structure
  (`r_hip` 77% PCK@50) but distal/face joints are near-random. `encoder_init` was
  `random`; it was trained on a single 30-min seated-at-desk recording (1,077 samples,
  avg confidence 0.44).
- The cog's **runtime inference path is still a centred-skeleton stub returning
  `confidence=0`** — the `pose_v1.safetensors` weights are not yet wired into
  `src/inference.rs`.
- ADR-079 records the proxy-supervised baseline at **PCK@20 = 2.5%**, and ADR-152
  **retracted** the internal camera-supervised 92.9% PCK@20 figure (it was a
  constant-output model scored under an absolute threshold on near-static frames; a mean
  predictor scores 100% under the same broken protocol).

### The real problem to fix

Not "the project has no weights" (false) and not "there is a validated pretrained
DensePoseHead" (false for the live ESP32 path). The real problem is a **labeling and
navigation gap**:
1. `archive/v1`'s random-init `DensePoseHead` is indistinguishable, to a grepping
   reader, from the live implementation, and carries no deprecation notice.
2. Nowhere is the split stated plainly: *which* checkpoints are real and validated
   (presence 82.3%, MM-Fi pose 82.69% torso-PCK@20), *which* are real-but-weak and
   honestly labeled (`pose_v1` 3% PCK@20, runtime stubbed), and *which* are
   architecture-only with no weights at all (`archive/v1` `DensePoseHead`).

## Decision

Two coordinated honest-labeling actions. Neither invents a capability; both make the
public surface match what the code and checkpoints actually deliver.

### (a) Formally deprecate `archive/v1` in the repo — MEASURED gap, proposed fix

- **Add `archive/v1/DEPRECATED.md`** — a loud tombstone stating that `archive/v1` is the
  original pure-Python implementation, is **unmaintained and superseded**, that its
  `DensePoseHead` is **architecture-only with random-initialized weights and ships no
  trained checkpoint**, and that the maintained path is the `v2/` Rust workspace + the
  `wifi-densepose 2.x` / `ruview` pip wheel (ADR-117). Mirror the disclaimer tone of
  ADR-160's `//!` headers and the pip 1.99.0 tombstone text.
- **Prepend a loud notice to `archive/v1/README.md`** (the file exists) — a `> ⚠️
  DEPRECATED` block at the very top pointing to `DEPRECATED.md`, `v2/`, and the pip
  wheel, before any of the existing "how to install v1" content.
- **Rule:** no doc outside `archive/v1/` may reference `archive/v1` code (other than the
  ADR-028 deterministic proof at `archive/v1/data/proof/verify.py`, which is a
  legitimately live signal-pipeline witness and stays) as if it were current. The two
  README references verified (`README.md` lines 139/198/204; `docs/user-guide.md`
  proof/swift-compile lines) are all proof/utility invocations, not implementation
  claims — they are acceptable and out of scope.

### (b) Model-weights honest labeling — state the three tiers explicitly

Add a **"Model weights: what's real, what's not"** subsection to `README.md` and
`docs/user-guide.md` that names the three tiers verified above, so no reader can infer
"a pretrained 17-keypoint DensePoseHead produces real pose accuracy on my ESP32":

| Tier | Checkpoint(s) | Honest status |
|------|---------------|---------------|
| **Real & validated** | `ruvnet/wifi-densepose-pretrained` (encoder + presence, 82.3% held-out temporal-triplet); `ruvnet/wifi-densepose-mmfi-pose` (82.69% torso-PCK@20, MM-Fi `random_split`); `count_v1` | MEASURED / published; keep current honest labels |
| **Real but weak (honestly labeled)** | committed `pose_v1.safetensors` in `cog-pose-estimation` | **PCK@20 = 3.0%**, below the ADR-079 ≥35% target; runtime path is a `confidence=0` stub until weights are wired into `src/inference.rs`. Already disclosed in the cog README; surface the same caveat wherever the live ESP32 pose feature is advertised |
| **Architecture only, no weights** | `archive/v1` `DensePoseHead` | random-init, no checkpoint; deprecated per (a) |

- The existing MM-Fi/presence honest labels (retraction of "100% presence", the cog
  "Honest reading") are **NO-ACTION positives** — cite them, do not weaken them.
- The live ESP32 17-keypoint claim stays **DATA-GATED**: the path to a first
  *reproducible* on-device baseline is ADR-079 (multi-session, full-body-framed,
  camera-supervised, ≥30K paired samples at conf ≥0.7, target PCK@20 ≥35%), tracked in
  [#645]. Do not advertise the live ESP32 pose feature without the "first-cut / below
  target / runtime stub" caveat until that baseline is MEASURED.
- Directly answer #509's ESP32-SISO question in the docs, honestly: single-antenna 56-
  subcarrier CSI at a 20-frame window does **not** carry the fine-grained spatial
  information the multi-antenna NIC research relies on (the cog README already shows
  distal/face joints near-random) — the shippable pose accuracy the project *can* stand
  behind today is the **MM-Fi benchmark** number, not a live single-ESP32 number.

## Phase ledger

| Phase | Action | State |
|-------|--------|-------|
| **P0** | This ADR (investigation + decision) | **DONE** (this file) |
| **P1** | Add `archive/v1/DEPRECATED.md` + loud notice atop `archive/v1/README.md` | **DONE** (1fb5397dd) |
| **P2** | Add "Model weights: what's real, what's not" tier table to `README.md` + `docs/user-guide.md`; add the caveat wherever the live ESP32 17-keypoint feature is advertised | **DONE** (1fb5397dd; follow-up caveated the hardware table, hero caption, and live-pipeline note) |
| **P3** | Answer #509's SISO/no-weights question and #1125's "does it run" in `docs/user-guide.md` (point to the reproducible proofs: MM-Fi arena, `archive/v1/data/proof/verify.py`, cog `train_results.json`) | **DONE** (1fb5397dd) |
| **P4** | Close the DATA-GATED live-pose gap via ADR-079 first reproducible on-device baseline (PCK@20 ≥35%) + wire `pose_v1.safetensors` into `cog-pose-estimation/src/inference.rs` | ACCEPTED-FUTURE ([#645]) |

## Acceptance criteria

- [x] `archive/v1/DEPRECATED.md` exists and names `v2/` + the pip wheel as the maintained path.
- [x] `archive/v1/README.md` opens with a `> ⚠️ DEPRECATED` block before any install instructions.
- [x] `README.md` and `docs/user-guide.md` no longer let a reader infer that `archive/v1`
      or an untrained/random-init `DensePoseHead` produces real pose accuracy without the
      caveats added here.
- [x] The live ESP32 17-keypoint pose feature is nowhere advertised without its
      "first-cut, PCK@20 = 3.0%, below ADR-079 target, runtime stub" caveat.
- [x] The three real/published checkpoints (presence 82.3%, MM-Fi pose 82.69% torso-PCK@20,
      `count_v1`) keep their existing honest labels — nothing is weakened or overclaimed.
- [x] No claim is added that is not MEASURED or explicitly DATA-GATED.

## Consequences

### Positive
- A grepping reader can no longer mistake `archive/v1`'s random-init `DensePoseHead` for
  the live system; the dead code is loudly tombstoned in the repo, matching its PyPI 1.99.0 tombstone.
- #509 and #1125 get an honest, verifiable answer: real trained weights *do* exist
  (presence + MM-Fi pose are published and benchmarked), the *specific* file the reporter
  found is architecture-only, and the live ESP32 pose path is honestly weak-and-in-progress.
- Reinforces the ADR-160 honest-labeling discipline: the project's credibility comes from
  precise labels, not from a suppressed or inflated narrative.

### Negative
- The docs must openly state that the live single-ESP32 17-keypoint pose is not yet at a
  citable accuracy — a short-term "looks less finished" cost, paid for by not overclaiming.
- Two more files to keep in sync (`DEPRECATED.md`, the tier table) as the checkpoints evolve.

### Neutral
- No code or model behavior changes; `archive/v1` stays in the tree as a research archive
  (ADR-117 §1.3) and its ADR-028 proof witness is untouched.
- Purely documentation/labeling; no crate, wheel, or firmware rebuild required.

## References

- `archive/v1/src/models/densepose_head.py` — `DensePoseHead`, random `_initialize_weights()`, no checkpoint load.
- `archive/v1/README.md` — legacy note; no loud deprecation (target of P1).
- `v2/crates/cog-pose-estimation/cog/README.md` — the "Honest reading" precedent (PCK@20 = 3.0%, runtime stub).
- `v2/crates/cog-pose-estimation/cog/artifacts/{pose_v1.safetensors,pose_v1.onnx,train_results.json}` — committed first-cut pose model.
- `v2/crates/cog-person-count/cog/artifacts/count_v1.{safetensors,onnx}` — committed count model.
- `ruvnet/wifi-densepose-pretrained`, `ruvnet/wifi-densepose-mmfi-pose` — published, benchmarked checkpoints.
- ADR-079 §Target (PCK@20 ≥35%), ADR-152 measurement (a) (96.09% PCK@20 external; internal 92.9% retracted), ADR-160 (honest-labeling method), ADR-117 (pip 1.99.0 tombstone).
