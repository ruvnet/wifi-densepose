# WiFlow-STD (DY2434) Benchmark Results — ADR-152 §2.2

Upstream: <https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling>
pinned at `06899d29` (2026-04-05), Apache-2.0. Dataset: Kaggle `kaka2434/wiflow-dataset`
(12.8 GB archive → 15.5 GB extracted; 360,000 windows of 540×20 CSI + 15-keypoint 2D labels).

Published claims (README "Setting 1"): PCK@20 97.25%, PCK@30 98.63%, PCK@40 99.16%,
PCK@50 99.48%, MPJPE 0.007 m, 2.23M params, 0.07 GFLOPs.

## Measurement (a): their model on their data

### Artifact verification (MEASURED, 2026-06-10, this repo `eval_repro.py`)

| Check | Result |
|---|---|
| Parameter count | **2,225,042 (2.23M) — matches claim** |
| FLOPs (torch profiler, batch 1) | ~0.055 GFLOPs — consistent with 0.07B claim |
| CPU latency (Windows box, torch 2.12 CPU) | 13.2 ms/window @ batch 1 (76/s); 2.48 ms/sample @ batch 64 (403/s) |
| Checkpoint load | `weights_only=True` (no pickle code execution) |

### Released checkpoint does NOT reproduce the claims — REFUTED as shipped

Running the released `best_pose_model.pth` through the released code on the released
dataset with the released split procedure (seed-42 file-level 70/15/15; 54,000 test
samples) yields:

| Metric | Published | Measured (shipped checkpoint) |
|---|---|---|
| PCK@20 | 97.25% | **0.08%** |
| PCK@30 | 98.63% | 0.78% |
| PCK@40 | 99.16% | 5.53% |
| PCK@50 | 99.48% | 15.42% |
| MPJPE | 0.007 | **NaN** (dataset contains NaN CSI windows) |

Raw output: `results/repro_a.json`.

Diagnostics (on 2,000 NaN-free windows from the first files of the dataset, i.e.
mostly would-be *training* data — so this is not a split mismatch):

- Predictions correlate with targets (Pearson r ≈ 0.76) — the checkpoint is a trained
  model, but in a **different keypoint normalization/order** than the released data.
- Best-case post-hoc global per-axis affine correction: PCK@20 ≈ 20%.
- Best-case per-keypoint affine correction (15×2 fitted transforms — generous
  cheating): PCK@20 ≈ 72%, still far below 97.25%.
- Pred↔target keypoint correspondence matrix is degenerate (multiple predicted
  keypoints best-match the same target joint) — keypoint convention mismatch.

### Reproducibility defects in the released artifacts

1. `models/__init__.py` imports `TemporalConvNet`, which `models/tcn.py` does not
   define — **the published code does not import/run as-is**.
2. The released root checkpoint uses pre-rename module names (`att.*`, `final_conv.*`)
   vs the published code (`attention.*`, `decoder.*`) — same shapes/param count, but
   confirms the checkpoint predates the published code.
3. The second shipped checkpoint (`cross_dataset_test/WiFlow/best_pose_model.pth`) is
   a **different architecture** (342-channel input = MM-Fi layout, 3 TCN layers,
   3-channel/3D decoder) — not usable on their own dataset.
4. `run.py` ignores `--data_dir` and hardcodes `../preprocessed_csi_data`.
5. The released dataset's final 13 files (indices 487–499; 9,072 windows, 2.52%)
   are corrupted: NaN values plus garbage amplitudes up to 3.4e38 (float32 max) in
   data that is otherwise [0,1]-normalized. Upstream code has no NaN/inf handling;
   training as published on this download diverges — the first corrupted batch
   overflows fp16 autocast and permanently poisons BatchNorm running statistics
   (GradScaler step-skipping does not protect BN). The authors' training curves
   show normal convergence, so their local data evidently differed from the
   Kaggle upload. Window masks: `results/nan_windows_mask.npy`,
   `results/big_windows_mask.npy`.

### Reproducing the corruption masks

The two mask files (9,070 NaN/Inf windows, 9,072 with |amplitude| > 1.5;
union 9,072, all in dataset files 487–499) are **committed ground truth**
(gitignore-negated, ~352 KB each). They can only be regenerated from a
**pristine** Kaggle download: `remote/clean_v2.py` repairs the dataset by
zeroing the corrupted windows in place, after which the corruption evidence
is gone and a rescan returns all-False. `generate_corruption_masks.py`
re-derives them (chunked scan, criteria: any non-finite value OR
max |finite| > 1.5 per 540×20 window) and refuses to write all-False masks,
which indicate a cleaned copy. Verified 2026-06-11: a regeneration from the
local pristine download is bit-identical to the committed masks.

### Retraining result (MEASURED, 2026-06-10): claims APPROXIMATELY REPRODUCED

Since the shipped checkpoint is unusable, measurement (a) fell back to retraining
with upstream code + defaults (seed 42, batch 64, early-stopped at epoch 41 of 50,
best epoch 36, ~75 s/epoch) on ruvultra (RTX 5080). Deviations, all forced and
documented: one-line fix for defect (1); torch 2.x+cu128 instead of pinned 2.3.1
(Blackwell sm_120 unsupported); the 9,072 corrupted windows (defect 5) zeroed
entirely — without this the published pipeline produces NaN from epoch 1 (observed).
Scripts mirrored in `remote/`; raw metrics in `results/eval_retrained.json`.

| Metric | Published | Retrained (full test, 54,000) | Retrained (corruption-free, 52,560) |
|---|---|---|---|
| PCK@20 | 97.25% | **96.09%** | **96.61%** |
| PCK@30 | 98.63% | 97.89% | 98.23% |
| PCK@40 | 99.16% | 98.58% | 98.79% |
| PCK@50 | 99.48% | 98.99% | 99.11% |
| MPJPE | 0.007 | 0.0098 | 0.0094 |

Within ~0.6–1.2 PCK points of every published figure (single run, corrupted train
windows zeroed, different torch/GPU). **Verdict: the accuracy claims are credible
and approximately reproducible — but only after repairing the released dataset and
code.** Val best: PCK@20 96.99%, MPJPE 0.0086 (epoch 36).

One more defect found during the run:

6. `train.py` calls `plot_training_history`, which is not defined anywhere — the
   built-in post-training test evaluation is unreachable as published (crashes
   with NameError after training completes).

## ADR-152 §2.2 citation rule

Evidence grade for the WiFlow-STD accuracy claims after measurement (a):
**MEASURED-EQUIVALENT (96.1–96.6% PCK@20 reproduced by retraining; shipped
checkpoint REFUTED; dataset/code require repairs)**. RuView docs may cite
"~96% PCK@20 (our reproduction)" — still **not comparable** to our 17-keypoint
ESP32 numbers (different hardware, 5 subjects, in-domain random split,
15 keypoints).

## Edge optimization (measured)

ADR-152 "optimize beyond SOTA" track, 2026-06-10, this Windows box (Windows 11,
16 torch threads, torch 2.12.0+cpu, onnxruntime 1.26.0). Subject: the retrained
checkpoint `results/retrained_best_pose_model.pth` (2,225,042 fp32 params).
Scripts: `quantize_bench.py`, `onnx_bench.py`, `eval_ort_accuracy.py`.
Raw numbers: `results/edge_optimization.json`.

Accuracy is on a **10,000-window seed-42 random subset** of the corruption-free
test split (same seed-42 file-level 70/15/15 split as `eval_repro.py`; 54,000
test windows, 1,440 corrupted excluded via `results/nan_windows_mask.npy` |
`results/big_windows_mask.npy`, leaving 52,560; subset drawn with
`np.random.default_rng(42)`). The fp32 subset PCK@20 (96.68%) matches the full
clean-test figure (96.61%), so the subset is representative.

Latency is CPU ms/window, median of repeated runs, 3 interleaved repetitions
per variant (medians below; run-to-run spread on this box is large, roughly
±20-40% at batch 1 — reps are in the JSON).

| Variant | Disk size | Batch 1 (ms/win) | Batch 64 (ms/win) | PCK@20 | PCK@50 | MPJPE |
|---|---|---|---|---|---|---|
| torch fp32 (baseline) | 9.07 MB | 11.0 | 2.27 | 96.68% | 99.15% | 0.00936 |
| torch fp16 (`.half()`) | **4.58 MB** | 24.3 | 2.42 | 96.68% | 99.15% | 0.00946 |
| torch int8 dynamic | 9.07 MB (unchanged) | 15.6 | 2.06 | 96.68% (identical) | 99.15% | 0.00936 |
| ONNX fp32 (onnxruntime) | 8.97 MB | **3.2** | **2.0** | 96.68% | 99.15% | 0.00936 |
| ONNX int8 (ORT dynamic, supplementary) | **2.44 MB** | 6.5 | 5.8 | 96.52% | 99.15% | 0.01108 |

Findings:

- **torch dynamic INT8 quantizes nothing on this model.** The architecture has
  **zero `nn.Linear` layers** — it is entirely Conv1d (21) + Conv2d (22) +
  BatchNorm. `torch.ao.quantization.quantize_dynamic` (requested over
  `{Linear, Conv1d, Conv2d}`) converted **0 modules / 0.0% of params**: dynamic
  quantization only has kernels for Linear/RNN-family modules and silently
  skips convolutions. The "int8" model is bit-identical to fp32 (same outputs,
  same 9.07 MB). Conv quantization would require static (PTQ) quantization
  with calibration — out of scope here; the ORT dynamic path below is the
  honest int8 datapoint.
- **fp16 halves size for free accuracy-wise** (PCK@20 −0.005 pt, MPJPE
  +0.0001) but is *slower* on CPU at batch 1 (~2.2×) — torch CPU fp16 conv
  kernels are emulated. fp16 is a storage/transport format here, not a CPU
  runtime win.
- **ONNX Runtime is the real batch-1 latency win: ~3.4× faster than torch**
  (3.2 vs 11.0 ms/window) at identical accuracy (parity 2.4e-7).

### Verdict on the paper's "~2.2 MB int8" claim

**Plausible but not free, and unreachable by the obvious PyTorch route.**
2,225,042 params × 1 byte ≈ 2.2 MB assumes *every* parameter quantizes.
PyTorch dynamic quantization — the one-liner most readers would reach for —
yields **9.07 MB (0% quantized)** because the model has no Linear layers.
ONNX Runtime dynamic quantization, which does have int8 conv weight support,
gets **2.44 MB** (close to the claim; the overhead is BatchNorm params/buffers
and quantization scales kept in fp32) at a measurable accuracy cost:
PCK@20 96.68 → 96.52% (−0.16 pt) and MPJPE 0.00936 → 0.01108 (+18%), and
~2× slower inference than ONNX fp32 (ConvInteger kernels). The paper does not
state a method or an int8 accuracy; treat "2.2 MB" as a weight-arithmetic
estimate, achievable in practice only via conv-capable quantization toolchains
and with a small accuracy penalty.

### ONNX export status

**Works.** Exported via the TorchScript exporter (`dynamo=False`), opset 17,
with a dynamic batch axis — `results/retrained_fp32_dynamic.onnx` (8.97 MB),
verified to run at batch 1/2/64. The axial attention's
`view(N*W, C, H)` reshape traced correctly (sizes recorded as graph ops, not
baked constants). The dynamo exporter also captures the graph but crashed on
this box writing a ✅ to a cp1252 console (cosmetic Windows encoding issue, not
a model blocker). Parity vs torch on the stored fixture
(`results/parity_fixture.npz`, batch 2, seed 42): **max abs diff 2.4e-7 —
PASS** (< 1e-4). ORT-quantized int8 model: `results/retrained_int8_ort_dynamic.onnx`.

### Static PTQ (calibrated) — follow-up

Follow-up to the dynamic-int8 row above (2026-06-10, same box, onnxruntime
1.26.0): ONNX Runtime **static** post-training quantization
(`quantize_static`, QDQ format, per-channel int8 weights + int8 activations)
of the same fp32 export, calibrated on **corruption-free TRAINING-split
windows only** (seed-42 file-level split, same masks; 1,000 windows for
MinMax, 512 for the histogram calibrators; never test windows). Scopes:
"conv-only" (`op_types_to_quantize=["Conv"]` — the attention path exports as
Einsum/Softmax, which ORT never quantizes anyway, so "all-ops" additionally
quantizes the elementwise Mul/Sigmoid/Add/AveragePool glue). Accuracy on the
identical 10k-window seed-42 corruption-free test subset; latency median of
3 interleaved reps (fp32/dynamic re-benched in-session as references).
Script: `static_ptq_bench.py`; raw: `results/edge_optimization.json`
(`onnx_static_ptq`).

| Variant | Disk size | Batch 1 (ms/win) | Batch 64 (ms/win) | PCK@20 | PCK@50 | MPJPE |
|---|---|---|---|---|---|---|
| ONNX fp32 (reference) | 8.97 MB | 2.5 | 1.9 | 96.68% | 99.15% | 0.00936 |
| ORT dynamic int8 (baseline) | **2.44 MB** | 5.7 | 4.6 | 96.52% | 99.15% | 0.01108 |
| static QDQ **Percentile(99.99) conv-only** | 2.53 MB | 5.3 | 4.7 | 96.61% | 99.16% | **0.01031** |
| static QDQ MinMax conv-only | 2.53 MB | 5.2 | 3.3 | **96.63%** | 99.19% | 0.01084 |
| static QDQ Entropy conv-only | 2.53 MB | 5.2 | 3.1 | 96.60% | 99.19% | 0.01078 |
| static QDQ MinMax all-ops | 2.60 MB | 6.5 | 3.9 | 95.45% | 99.14% | 0.01486 |
| static QDQ Entropy all-ops | 2.60 MB | 5.7 | 4.1 | 95.30% | 99.13% | 0.01510 |
| static QDQ Percentile all-ops | 2.60 MB | 5.3 | 4.3 | 96.39% | 99.17% | 0.01218 |

**Verdict: static PTQ (conv-only) is the new best int8 point on accuracy —
but only modestly, and it does not fix int8's latency penalty.**

- **Accuracy: beats dynamic.** All three conv-only calibrations land at
  PCK@20 96.60–96.63% (vs dynamic 96.52%, fp32 96.68% — recovers ~⅔ of the
  dynamic gap) and MPJPE 0.0103–0.0108 (vs dynamic 0.01108). Best MPJPE:
  Percentile conv-only, +10% over fp32 instead of dynamic's +18%.
- **Size: slightly worse.** 2.53 MB vs 2.44 MB (+3.6%) — QDQ nodes and
  per-channel scales cost a little; BatchNorm stays fp32 in both (the 12 BNs
  follow Slice/Einsum/Reshape, never Conv, so they cannot be folded).
- **Latency: a wash vs dynamic, still ~2× slower than ONNX fp32 at batch 1.**
  Batch-1 medians 5.2–5.3 vs dynamic 5.7 ms/win in-session — within this
  box's ±20–40% noise. Batch 64 leans static (3.1–3.3 for MinMax/Entropy
  conv-only vs 4.6), same caveat.
- **All-ops QDQ is strictly worse**: up to −1.4 pt PCK@20 and +60% MPJPE for
  zero size/latency benefit — int8 activations through the elementwise glue
  around the attention blocks is where the damage is. Conv-only is the right
  scope.
- Negative result worth recording: **Entropy calibration is a no-op here** —
  on an identical calibration set it selects full-range thresholds
  bit-identical to MinMax (all 247 scales equal; verified on a 64-window
  smoke set). Also, ORT 1.26's `CalibMaxIntermediateOutputs` raises a
  spurious "No data is collected" when the batch count divides the chunk
  size (worked around in the script).

Deployment guidance: need speed → ONNX fp32 (3.2 ms b1). Need int8 weights
for size → static QDQ conv-only (Percentile or MinMax,
`results/retrained_int8_static_percentile_conv.onnx`), which strictly
dominates dynamic int8 on accuracy at ~equal latency and +0.09 MB.

## Efficiency sweep (MEASURED, overnight 2026-06-10/11)

ADR-152 beyond-SOTA track: compact purpose-built variants of the WiFlow-STD
architecture, trained from scratch on the same cleaned dataset, identical
seed-42 file-level split, loss and protocol as the measurement-(a) reference
(fp32, batch 64, ≤50 epochs, patience 5; RTX 5080, ~22–29 min/variant).
Variant transforms are pure channel/group/stride scalings of an
architecture-exact parameterized model (validated: reproduces 2,225,042 params
at the reference config). Scripts: `remote/sweep/`; raw:
`results/efficiency_sweep.jsonl`; checkpoints `results/{half,quarter,tiny}_best.pth`
(gitignored).

| Variant | Params | vs 2.23M | Clean-test PCK@20 | PCK@50 | MPJPE | Best epoch |
|---|---|---|---|---|---|---|
| full (reference, meas. a) | 2,225,042 | 1× | 96.61% | 99.11% | 0.0094 | 36 |
| **half** | **843,834** | **0.38×** | **96.62%** | **99.47%** | **0.00898** | 23 |
| quarter | 338,600 | 0.15× | 96.05% | 99.43% | 0.00928 | 50 |
| tiny | 56,290 | 0.025× | 94.11% | 99.36% | 0.0125 | 47 |

Findings:

- **The half model (843k params) strictly dominates the full reference** on
  this dataset — equal PCK@20, better PCK@50 and MPJPE, converges in fewer
  epochs. The published 2.23M architecture is over-parameterized for its own
  benchmark.
- **tiny (56k params, 1/39.5) holds 94.11% PCK@20** — a ~220 KB fp32 /
  ~60 KB int8-class model in reach of severely constrained edge targets,
  at −2.5 pt from the full reference.
- Caveats: in-domain (5-subject random-file split) like every number on this
  dataset; single run per variant; corruption-free test subset (52,560).
  Cross-domain behavior of compact variants is untested — ADR-150's evidence
  says capacity *hurts* cross-subject, so the compact end may generalize no
  worse, but that is a hypothesis, not a measurement.

### Compact-variant edge artifacts (MEASURED, 2026-06-11)

Edge pipeline for the **tiny** checkpoint (56,290 params), same machinery and
protocol as the full-model edge rows above (this Windows box, torch
2.12.0+cpu, onnxruntime 1.26.0; dynamic-batch opset-17 TorchScript export;
static QDQ **Percentile(99.99) conv-only** int8 calibrated on **512**
corruption-free TRAIN-split windows; accuracy on the identical 10k-window
seed-42 clean test subset; latency = median ms/window over 3 interleaved
reps, with the full-model fp32/int8 sessions interleaved as same-session
references). Script: `tiny_edge_bench.py`; raw:
`results/edge_optimization.json` (`tiny_variant`). Torch-vs-ORT parity on the
stored fixture input: **max abs diff 1.5e-7 — PASS** (< 1e-4). The tiny fp32
subset PCK@20 (94.11%) matches the full clean-test sweep figure (94.11%)
exactly, so the subset remains representative.

Two forced deviations, both recorded in the JSON:

1. **Adaptive-pool export rewrite.** tiny's derived stride schedule
   `[2,1,1,1]` leaves feature width 16, and the TorchScript exporter rejects
   `AdaptiveAvgPool2d((15,1))` when 15 is not a factor of the input height
   (the full model never hit this — its width was exactly 15). Since the
   pool over a fixed-size map is a fixed linear operator, the export wrapper
   replaces it with `mean(-1)` (W axis, a factor) + a constant averaging
   matmul using PyTorch's exact bin rule; the parity check (vs the original
   torch model with the real pool) proves exactness.
2. **Calibration count 512, not "~500"**: ORT 1.26's histogram collector
   `np.asarray()`'s the per-batch maxima, so the calibration count must be a
   multiple of the 64-window calibration batch or the ragged last batch
   crashes it (the earlier static-PTQ run dodged this by using exactly 512).

| Variant | Disk size | Batch 1 (ms/win) | Batch 64 (ms/win) | PCK@20 | PCK@50 | MPJPE |
|---|---|---|---|---|---|---|
| full ONNX fp32 (same-session ref) | 8.97 MB | 2.27 | 1.42 | 96.68% | 99.15% | 0.00936 |
| full static QDQ Percentile conv-only (same-session ref) | 2.53 MB | 5.53 | 3.82 | 96.61% | 99.16% | 0.01031 |
| **tiny ONNX fp32** | **0.295 MB** | **0.66** | **0.24** | **94.11%** | 99.37% | 0.01253 |
| tiny static QDQ Percentile conv-only | 0.248 MB | 0.85 | 1.03 | 92.68% | 99.33% | 0.01491 |

(tiny torch `.pth` checkpoint for reference: 0.34 MB on disk; 56,290 fp32
params ≈ 225 KB of weights.)

Findings:

- **The smallest deployable WiFlow-class model is the tiny ONNX fp32
  artifact: ~295 KB on disk, 0.66 ms/window batch-1 CPU (~1,500 windows/s),
  94.1% PCK@20** — 30× smaller and ~3.4× faster (in-session) than the full
  ONNX fp32 model for −2.6 pt PCK@20.
- **int8 is a bad trade at this scale.** Static QDQ conv-only — the recipe
  that cost the full model only 0.07 pt — costs tiny **−1.43 pt** PCK@20
  (94.11 → 92.68%) and +19% MPJPE, saves only 47 KB (−16%; QDQ scales and
  the fp32 BN/attention glue are proportionally larger in a small graph),
  and is *slower* than tiny fp32 (0.85 vs 0.66 ms b1; 1.03 vs 0.24 ms b64 —
  QDQ kernel overhead dominates when the convs are this small). A 56k-param
  model has little redundancy left to absorb weight+activation rounding.
- Deployment guidance, compact edition: ship tiny as **ONNX fp32** — at
  295 KB the int8 size saving solves no real constraint and costs accuracy
  and speed. If ~250 KB vs ~295 KB ever matters, weight-only quantization
  would be the thing to try next, not QDQ.

## Measurement (b): BLOCKED-ON-DATA (attempted 2026-06-10)

The fine-tune-on-ESP32 measurement stopped at dataset characterization, per the
pre-registered stop rule (<2,000 paired windows). Findings (MEASURED):

- **Only one trainable paired dataset exists**: `ruvultra:~/work/cog-pose-train/paired.jsonl`
  — 1,077 windows (one subject, one room, one 29.9-min session, single node;
  CSI [56, 20]; 17 COCO keypoints, MediaPipe confidence mean 0.44 — only 264
  windows pass ADR-079's own conf>0.5 training filter). Prior measured attempts
  on this exact set: 0–3% torso-PCK@20 (temporal splits, three independent
  pipelines). Fine-tuning a 2.23M-param model on ~860 train windows would
  measure memorization, not transfer.
- **The April session behind the old "92.9% PCK@20" claim is lost** (345
  samples, 35 subcarriers; raw CSI gone from ruvzen/ruvultra/cognitum-v0; only
  a 69-sample predictions+GT holdout survives at `models/wiflow-real/eval-holdout.jsonl`).
- **Forensic recheck of that holdout RETRACTS the 92.9% figure**: the trainer's
  `pck()` used an absolute 0.2 image-unit threshold (not torso-normalized) and
  the model output a **constant pose** (pred std 0.0000 across 69 near-static
  frames; a mean predictor scores 100% under the same protocol). The
  torso-normalized PCK@20 on the same holdout is 19.1%. This corroborates the
  2026-05-11 audit retraction (CHANGELOG, PR #535); stale doc citations were
  removed 2026-06-10 (user-guide, readme-details, ADR-152 §2.1.3). The §2.2
  no-citation rule now applies to ADR-079 accuracy claims.

Unblock criteria: a paired collection session of ≥2k windows (≈35+ min at the
observed stride; multi-pose, conf>0.5, ideally with the §2.1.3 two-checkerboard
calibration), plus a re-baselined our-pipeline number under torso-PCK@20 on the
same split. WiFlow-STD assets stand ready on ruvultra (`~/wiflow-std-bench/`).
Also worth investigating: ADR-079's protocol predicts ~9k windows per 30 min;
the May session under-delivered ~8× (aligner drop rate?).

## Measurement (b) (MEASURED 2026-06-10/11)

The data baseline unblocked: the 2026-06-10 22:10–22:40 collection session produced
**2,046 paired windows** (`ruvultra:~/wiflow-std-bench/paired-20260610.jsonl`; ONE
subject, ONE room, ONE ESP32 node, varied poses: walk/raise/squat/kick/wave/turn/
jump/sit; aligner `scripts/align-ground-truth.js`, non-overlapping 20-frame windows
~0.42 s; 17 COCO keypoints in normalized [0,1] camera coords; MediaPipe confidence
mean 0.802, min 0.692 — all windows pass the conf>0.5 filter). The −4 h timestamp
bug and the empty-frame confidence-dilution aligner findings are recorded
separately; results only here. Trained on ruvultra (RTX 5080, torch 2.11+cu128,
fp32, batch 32, GPU shared with the efficiency sweep). Scripts mirrored in
`remote/measb/`; raw metrics + full training curves in `results/measurement_b.json`.

### Two new aligner/dataset findings (forced deviations, MEASURED)

1. **`csi_shape` is heterogeneous, not [70, 20]**: 1,347× [70,20], 284× [134,20],
   243× [26,20], 130× [12,20], 42× [20,20]. The ESP32 stream emits mixed frame
   types and `extractCsiMatrix` stamps each window's subcarrier count from
   `window[0].subcarriers`, zero-padding/truncating the other frames — even
   native-70 windows contain ~20.4% internally zero-padded short frames
   (subcarriers 40–69 all-zero). Handling: the primary suite ("all 2,046")
   linearly resamples every frame's subcarrier axis to 70 bins (identity for
   native-70 frames) so the pre-registered n and split sizes hold; a secondary
   suite restricts to the 1,347 native [70,20] windows as a homogeneity check.
2. **Aligner layout bug**: `extractCsiMatrix` fills `matrix[f * nSc + s]`
   (frame-major) but declares `shape: [nSc, nFrames]` — the stored shape label is
   transposed relative to the data. Confirmed by coherent per-frame zero-tails;
   corrected on load (`reshape(nFrames, nSc).T`).

### Protocol (pre-registered, followed)

Temporal split, no shuffling across time: first 70% train (1,432), next 15% val
(307), last 15% test (307); seed 42 elsewhere. Model: learned 1×1 Conv1d 70→540
adapter prepended to the upstream WiFlow-STD trunk; K=17 via the parameter-free
adaptive pool (`AdaptiveAvgPool2d((17,1))` — pretrained weights load strict for
any K). CSI normalized by the TRAIN-split p99 amplitude (129.7 all / 130.9
native-70), clipped to [0,1]. Three runs, ≤60 epochs, early-stop patience 8 on
val MPJPE, AdamW (adapter lr 1e-4; pretrained trunk lr 1e-5, 10× lower; scratch
all 1e-4), fp32. Pretrained init = the measurement-(a) **retrained** checkpoint
(`upstream/test/best_pose_model.pth`, ~96% PCK@20 on WiFlow data; the
`att.`/`final_conv.` key remap from `eval_repro.py` applied defensively — a no-op,
that checkpoint already uses post-rename keys). Frozen-trunk run: trunk
`requires_grad=False` **and** held in `.eval()` so BatchNorm running stats cannot
drift — a pure transfer probe; only the 70→540 adapter (38,340 params) trains.

PCK is torso-normalized with **torso = ‖l_shoulder(5) − l_hip(11)‖** (upstream
`calculate_pck` math — per-frame norm clamped at 0.01, mean over keypoints ×
frames — but upstream's `NECK_IDX/PELVIS_IDX = 2, 12` is a 15-keypoint
convention; on 17-kp COCO those indices are right_eye/right_hip, so the indices
were replaced, not the math). MPJPE is in normalized image units (not meters).

### Results — primary suite, all 2,046 windows (test = last 307)

| Run | PCK@10 | PCK@20 | PCK@30 | PCK@40 | PCK@50 | MPJPE | pred std | best ep |
|---|---|---|---|---|---|---|---|---|
| **mean-pose baseline** (honesty bar) | **73.1%** | **95.9%** | **98.7%** | 99.3% | 99.3% | **0.0148** | 0 (by constr.) | — |
| (i) pretrained-init, full fine-tune | 26.0% | 65.0% | 88.0% | 96.4% | 98.9% | 0.0313 | 0.0113 | 58/60 |
| (ii) scratch | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2554 | 0.0002 | 4 (stop @13) |
| (iii) frozen-trunk (adapter only) | 0.0% | 0.0% | 0.2% | 3.2% | 14.4% | 0.1260 | 0.0073 | 59/60 |

Secondary suite (native [70,20] windows only, n=1,347, test=202) reproduces the
same ordering: mean-baseline 96.0% / pretrained 67.1% / scratch 0.0% /
frozen-trunk 0.0% PCK@20 (MPJPE 0.0153 / 0.0318 / 0.2236 / 0.1343) — the
subcarrier-resampling choice does not change any conclusion.

### Interpretation

- **Did pretraining-transfer happen? Partially — as optimization transfer, not
  feature transfer, and not past the honesty bar.**
  - *Pretrained vs scratch*: dramatic (65.0% vs 0.0% PCK@20). The pretrained init
    is the only configuration that trains at all under the pre-registered budget.
  - *Frozen-trunk*: near-zero (0.0% PCK@20, 14.4% @50). WiFlow-STD's frozen
    features do **not** transfer to our ESP32 domain through a linear subcarrier
    adapter — the pretrained benefit is a well-conditioned initialization (incl.
    calibrated BN/output scales), not reusable CSI→pose features.
  - *Everything vs mean-pose baseline*: **no run beats it.** A constant
    train-mean pose scores 95.9% torso-PCK@20 / 0.0148 MPJPE on this test split,
    because a single subject in one camera frame barely moves in normalized
    coordinates. The fine-tuned model is a real, non-constant model
    (pred std 0.0113 > 0 — passes the constant-pose detector that retracted the
    old 92.9% figure) but its deviations from the mean hurt: it fits train-period
    temporal dynamics that do not generalize across the temporal split.
- **Verdict for ADR-152 §2.2(b): fine-tuning WiFlow-STD on this dataset does not
  demonstrate CSI→pose signal beyond the mean pose.** Until a model beats the
  mean-pose baseline on a temporal split, no PCK number from this line may be
  cited as pose-estimation capability.

### Caveats (honest, pre-registered)

- Single subject, single room, single session (30 min), single ESP32 node —
  in-domain temporal split only; nothing here speaks to cross-room or
  cross-subject generalization.
- 2k windows vs the 360k-window WiFlow-STD corpus — **NOT comparable** to the
  ~96% in-domain measurement-(a) number, and the published 97.25% even less so.
- The scratch run's total collapse (it cannot even reach the mean pose; its
  output BatchNorm/SiLU head must learn output scale from random init at lr 1e-4)
  is an optimization outcome under the fixed budget, not proof the architecture
  cannot learn from scratch — the pretrained-vs-scratch gap partially reflects
  this conditioning advantage.
- Mixed-subcarrier frames (finding 1) mean even the "clean" windows carry ~20%
  zero-padded frames; collection-side frame-type filtering should precede the
  next session.
- Mean-baseline PCK is inflated by low pose variance relative to torso size
  (~0.2–0.3 image units); PCK@10 (73.1%) shows the same ceiling effect at a
  stricter threshold — the bar is the bar, but a livelier dataset would lower it.

## Pending

- (b) fine-tune on our ESP32 17-keypoint eval set — **MEASURED 2026-06-10/11**,
  see above: no run beats the mean-pose baseline; pretraining transfers as
  optimization aid only.
- (c) our internal WiFlow on their dataset (15-keypoint subset mapping) — also
  affected: there is currently no validated internal pose model to compare
  (the 92.9% artifact is retracted; the MM-Fi SOTA models in ADR-150 §3 are a
  different input domain).
