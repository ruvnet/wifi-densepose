# We audited a state-of-the-art WiFi pose model. Here's what broke, what reproduced, and the 30× smaller model that nearly matches it.

*RuView team, June 2026. All numbers measured; full scripts and forensics in the
[RuView repo](https://github.com/ruvnet/RuView/tree/main/benchmarks/wiflow-std).*

## The setup

WiFi sensing is having a moment: a 2026 preprint ("WiFlow", arXiv 2602.08661)
claims **97.25% pose-estimation accuracy (PCK@20) from WiFi signals alone**,
with a tiny 2.23M-parameter model — and unlike most papers, it ships
everything: code, trained weights, and a 360,000-sample dataset.

We build WiFi sensing systems, so before adopting any external number we run
it through a simple rule: **a claim is "CLAIMED" until we reproduce it, then
it's "MEASURED."** Here's what happened when we tried.

## Day 1: nothing works

- **The code doesn't run.** The package imports a class that doesn't exist.
  (One-line fix.)
- **The released model scores 0.08%, not 97.25%.** The shipped checkpoint was
  trained under a different data normalization than the shipped dataset —
  it's a real trained model, just not *this* pipeline's model. Even letting it
  cheat with a fitted per-keypoint correction only reaches 72%.
- **The dataset is corrupted.** Its last 13 files contain garbage values up to
  3.4×10³⁸ (float32's maximum). Subtle consequence: the training loop uses
  fp16 mixed precision with no guards, so the first corrupted batch overflows
  and **permanently poisons the model's BatchNorm statistics**. Training from
  the public download produces NaN from epoch 1, every time.
- The training script also crashes before its own test phase ever runs
  (calls an undefined function), and ignores its `--data_dir` flag.

At this point a less patient reader concludes "fraud." That would be wrong.

## Day 1, later: actually, the science is real

We repaired the artifacts — fixed the import, zeroed the 9,072 corrupted
windows, retrained from scratch with the authors' own code and
hyperparameters on one GPU (~50 minutes):

| Metric | Published | Our retrain |
|---|---|---|
| PCK@20 | 97.25% | **96.1–96.6%** |
| PCK@50 | 99.48% | 99.0–99.1% |
| Params | 2.23M | 2,225,042 (exact) |

**The claims reproduce.** What didn't survive contact was the *packaging*:
wrong checkpoint, corrupted upload, broken glue code. This distinction —
**artifact rot vs. bad science** — is the single most useful thing a
reproduction can establish, and you can't establish it without actually
running the thing.

(We filed all six defects upstream with fixes:
[issue #3](https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling/issues/3).
And to be clear: the authors released more than 90% of papers do. That's the
only reason this audit was possible.)

## Day 2: the model is also 2.6× too big

Once we could train, we asked: does the architecture need 2.23M parameters?

| Variant | Params | Accuracy (PCK@20) | Size on disk |
|---|---|---|---|
| Original | 2,225,042 | 96.61% | 8.97 MB |
| **Half** | **843,834** | **96.62%** ✨ | — |
| Quarter | 338,600 | 96.05% | — |
| **Tiny** | **56,290** | **94.11%** | **295 KB** |

The half-width model **matches the original exactly** (and converges faster).
The tiny one — 1/39th the parameters — gives up 2.5 points and runs at
**0.66 ms per inference on a laptop CPU** (~1,500 poses/second) as a 295 KB
ONNX file. For edge devices, that's the interesting end of the curve.

Quantization footnote: the paper's "~2.2 MB int8" estimate is reachable
(we measured 2.44–2.53 MB) but only via conv-capable toolchains — PyTorch's
one-line dynamic quantization converts *literally nothing* on this model
(it has no Linear layers), a trap worth knowing about.

## What we took away

1. **Run the artifact, not the README.** Every number in a paper is one
   `git clone` away from being either confirmed or understood. Both outcomes
   are valuable; only one is publishable by the original authors.
2. **fp16 + unvalidated data = silent model death.** Mixed-precision training
   with no NaN/inf guards doesn't fail loudly — it corrupts BatchNorm buffers
   and ships a broken model with a green progress bar. Validate inputs, or
   train in fp32, or guard the autocast.
3. **Evidence-grade your own claims too.** Mid-audit, the same forensics
   tooling caught one of *our own* published accuracy numbers resting on a
   degenerate evaluation (a constant-output model scored with a flawed
   metric). We retracted it the same day. The rule has to cut both ways or
   it's marketing, not measurement.
4. **Over-parameterization hides in SOTA tables.** Nobody publishes the
   half-size ablation that matches their headline model. Run it yourself;
   it's an hour of GPU time and sometimes it *is* the result.

*Reproduction scripts, corruption masks, the efficiency-sweep configs, and a
numerically parity-proven Rust port (max divergence 1.2e-7) are all in
[`benchmarks/wiflow-std/`](https://github.com/ruvnet/RuView/tree/main/benchmarks/wiflow-std).*
