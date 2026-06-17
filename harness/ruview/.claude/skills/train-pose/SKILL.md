---
name: train-pose
description: Train/evaluate WiFi pose models honestly — camera-supervised (MediaPipe + CSI) and camera-free (WiFlow), always checked against the mean-pose baseline before any PCK is quoted.
---

# train-pose

Build a CSI→pose model without overstating it. The project has a **retracted 92.9%/100%**
history — the discipline below exists so it never recurs.

## The non-negotiable: mean-pose baseline first

A pose model that always predicts the dataset's *mean pose* already scores ~50% PCK.
**Quote PCK only as a delta over that baseline**, on a held-out split with no subject
or temporal leakage. Example honest result (ADR-181):

> Held-out PCK@20 **59.5%** vs a 50% mean-pose baseline = **+9.4 pp real signal** — MEASURED.

## Paths

- **camera-supervised** (ADR-079) — MediaPipe Pose labels the camera frame; paired CSI
  trains the net. Train/infer in one camera frame so the skeleton aligns.
- **camera-free** (WiFlow, ADR-152) — no camera at inference; geometry-conditioned.
- **in-browser** (ADR-181) — WebGPU/WASM trainer; the active backend is shown as a badge
  (honest about what's executing).

## Before you publish a number

1. Run the mean-pose baseline on the same split.
2. Report `(model − baseline)` in pp, with the split definition (chronological /
   blocked-gap / grouped-bucket; no leakage).
3. `ruview.claim_check` the writeup — it flags any untagged or 100%/perfect claim.
4. If it's a benchmark vs SOTA, tag MEASURED-EQUIVALENT only with the reproducer.
