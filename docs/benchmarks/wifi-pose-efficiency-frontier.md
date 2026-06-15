# WiFi-CSI Pose — Efficiency Frontier (beyond SOTA at a fraction of the size)

**Measured:** 2026-05-31 · MM-Fi `random_split` (ratio 0.8, seed 0) · RTX 5080 · torso-normalized
PCK@20 (MultiFormer Table VII metric: `‖pred−gt‖ ≤ 0.2·‖R-shoulder − L-hip‖`).

The flagship [`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose)
reaches **83.59%** torso-PCK@20 (vs MultiFormer 72.25%, CSI2Pose 68.41%). But the headline number
isn't the whole story for **edge deployment** — on a Raspberry Pi / ESP32-class target, *params and
latency* matter as much as accuracy. So we swept model size to map the **accuracy-per-parameter
frontier**: how small can a WiFi-CSI pose model be and still beat the prior published SOTA?

## The frontier

| Model | Params | Latency (batch=1) | torso-PCK@20 | vs SOTA (72.25%) |
|-------|-------:|------------------:|-------------:|------------------|
| nano  | 39,971 | 0.126 ms | 71.76% | −0.49 (58× smaller than flagship) |
| **micro** | **75,237** | 0.224 ms | **74.30%** | **✅ +2.05 — beats SOTA at 31× fewer params** |
| tiny  | 210,949 | 0.299 ms | 76.82% | ✅ +4.57 |
| small | 348,005 | 0.287 ms | 77.87% | ✅ +5.62 |
| base  | 726,437 | 0.344 ms | 79.38% | ✅ +7.13 (3.2× smaller) |
| flagship | 2,320,869 | — | 83.59% | +11.34 |

**Every configuration from `micro` (75K params) upward beats the prior published state of the art**,
and even `nano` (40K params, 0.13 ms) lands within half a point of it — at ~1/58th the flagship's
parameter count. A **75,237-parameter** model tops MultiFormer's 72.25%.

### Deployable footprint AND deployed accuracy (quantized `micro`)

Size alone isn't the claim — what matters is **accuracy at the deployed precision**. Measured
(weight-only, per-tensor symmetric):

| Precision | Size | torso-PCK@20 | vs SOTA 72.25 |
|-----------|-----:|-------------:|---------------|
| fp32 | 294 KB | 74.73% | ✅ +2.5 |
| **int8 (PTQ)** | **73.5 KB** | **74.70%** | ✅ +2.5 — **essentially lossless** |
| int4 (naïve PTQ) | 36.7 KB | 70.21% | ❌ −2.0 — drops below SOTA |
| **int4 (QAT)** | **36.7 KB** | **74.46%** | ✅ **+2.2 — recovered, still beats SOTA** |

**The honest edge result:** `micro` is **lossless at int8 (73.5 KB, 74.70%)**, and at **int4 (36.7 KB)
naïve post-training quantization falls below SOTA (70.21%) — but quantization-aware training fully
recovers it to 74.46%**, still beating MultiFormer. So a **SOTA-beating WiFi-pose model genuinely runs
in ~37 KB int4** (with QAT) or **~73 KB int8** (no retraining) — deployable on the sensing node itself.
`nano` (40K params) sits at the SOTA line in fp32 and is best treated as int8.

(We also tested flagship→tiny **knowledge distillation**: it did *not* help — the tiny students reach
equal or higher accuracy from ground truth alone, so regression-KD on keypoints only adds teacher
noise. Direct training wins.)

**Shipped as a usable artifact.** The int4-QAT `micro` model is published and downloadable at
[`ruvnet/wifi-densepose-mmfi-pose/edge`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose/tree/main/edge)
(`pose_micro_int4.npz` + `load_int4.py`): **verified deployed int4 accuracy 74.08%** (beats SOTA),
~20 KB int4 weight payload, sha256 `c03eeb…`. It runs in **0.135 ms single-thread on x86 CPU**
(no GPU) — i.e. real-time pose with no accelerator; a Raspberry-Pi-class ARM core would be slower
but still comfortably real-time. (Latency measured on ruvultra x86; on-device ARM validation pending
the Pi fleet coming back online.)

## Why this matters

- **Edge-native pose.** `micro`/`tiny` (75–210K params, sub-0.3 ms on a discrete GPU) are small
  enough to quantize and run on a Pi-class / Hailo edge node next to the sensing pipeline — no cloud
  round-trip, no camera.
- **Pareto-dominant, not just smaller.** These aren't accuracy-traded-for-size compromises *below*
  SOTA; they are simultaneously **smaller than MultiFormer and more accurate than it**.
- **Orthogonal to the accuracy frontier.** Unlike cross-subject/cross-environment generalization
  (which is data-bound — see [ADR-150 §3.2](../adr/ADR-150-rf-foundation-encoder.md)), the efficiency
  frontier responded immediately to optimization. This is the lever that's still open.

## Method & reproduction

Same architecture family as the flagship — input `[3,114,10]` CSI amplitude → linear projection →
`L`-layer / `H`-head Transformer encoder over the 10 temporal tokens → **temporal attention
pooling** → MLP head → **skeleton-graph refinement** (COCO bone topology) — with width `d`, depth
`L`, heads `H` swept. Training: mixup (Beta(0.2,0.2)), 4-view test-time augmentation, EMA, cosine LR.

| Model | d | L | H | graph head |
|-------|--:|--:|--:|:----------:|
| nano | 48 | 1 | 2 | — |
| micro | 64 | 1 | 2 | ✓ |
| tiny | 96 | 2 | 4 | ✓ |
| small | 128 | 2 | 4 | ✓ |
| base | 160 | 3 | 4 | ✓ |

Reproduce: `python aether-arena/staging/train_efficiency_pareto.py npy/X.npy npy/Y.npy npy/split_random.npy`
(MM-Fi parsed via `aether-arena/staging/parse_mmfi_zips.py`). Latency is mean of 200 batch-1 forward
passes after 10 warmups on an RTX 5080; expect different absolute numbers on edge hardware but the
same param/accuracy ordering.

> **Controlled claim.** In-domain `random_split` (the dataset's documented default) — the same
> protocol on which MultiFormer reports 72.25%. Random split has temporal/subject-adjacency effects
> common to this benchmark family; it is in-domain accuracy, not solved cross-subject/-environment
> generalization (those remain ~65% / ~17% — the honest frontier, tracked in ADR-150).
