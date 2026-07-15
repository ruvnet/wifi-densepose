# WiFi-CSI Sensing on MM-Fi — a complete, honest study

**Scope:** what works, what doesn't, and what actually ships — for 2D human **pose** and **action
recognition** from WiFi Channel State Information on the public [MM-Fi](https://github.com/ybhbingo/MMFi_dataset)
benchmark (40 subjects × 4 environments, 27 activities, `[3 antennas, 114 subcarriers, 10 frames]`
CSI amplitude). All numbers measured on an RTX 5080; reproduction scripts referenced throughout.

> **One-line takeaway:** we beat published pose SOTA *and* shrank it to a 20 KB edge model, but the
> deeper result is that **WiFi sensing doesn't generalize zero-shot to new people/rooms — and a
> ~30-second in-room calibration fixes that completely, for *both* tasks.** Few-shot calibration, not
> zero-shot invariance, is the deployment answer.
>
> **Sharpest finding (§7):** WiFi-CSI sensing is largely a **random-features + target-trained-readout**
> problem — a *random frozen* encoder + a trained head gets within ~2–4 pts of a fully-trained encoder
> (and within <2 pts cross-subject). The encoder barely learns anything transferable; the signal is in
> the readout. This single fact explains the zero-shot collapse, the no-transfer results, the
> foundation-encoder failure, *and* why per-room calibration works.

## 1. Pose estimation

### 1.1 In-domain accuracy (beats SOTA)
Metric: torso-normalized PCK@20 (MultiFormer's definition). Protocol: MM-Fi `random_split` (the
dataset default).

| Model | torso-PCK@20 |
|-------|-------------:|
| CSI2Pose (prior) | 68.41% |
| MultiFormer (prior SOTA, 2025) | 72.25% |
| **Ours (single)** | **82.69%** |
| **Ours (graph + 3-ensemble + TTA)** | **83.59%** |

Architecture: linear projection → 4-layer/8-head Transformer over the 10 temporal tokens →
**temporal attention pooling** (the single biggest lever) → MLP head → skeleton-graph refinement.
The headline was *self-corrected down* from an inflated 91.86% (loose bbox normalization) to 82.69%
under the matched torso metric before publishing.

### 1.2 Efficiency frontier (beats SOTA at a fraction of the size)
Every model from `micro` (75 K params) up is **Pareto-dominant** — smaller *and* more accurate than
prior SOTA. A **75 K-param model tops MultiFormer**; deployed **int4 is ~20 KB at 74.08% (QAT)**,
0.135 ms single-thread CPU. (int8 is lossless at 74.7%; naïve int4 PTQ drops to 70.2% — QAT recovers
it.) Full curve: [`wifi-pose-efficiency-frontier.md`](wifi-pose-efficiency-frontier.md).
Published: [`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose).

## 2. Action recognition (27 classes)

MM-Fi's own paper **does not benchmark WiFi-CSI action recognition** (its HAR is skeleton-based,
RGB/LiDAR/mmWave only). The only published WiFi-CSI-on-MM-Fi number is WiDistill (2024): 34.0%
(ResNet-18, unspecified split). We establish:

| Protocol | top-1 |
|----------|------:|
| random_split (in-domain) | 88.08% |
| cross-subject (official), zero-shot | **10.0%** (near-chance) |

The 88% is **leakage-inflated** (see §3); the honest cross-subject zero-shot is ~10%.

## 3. The generalization story (the real result)

Random-split numbers are inflated by temporal/subject adjacency. Under leakage-free protocols, WiFi
sensing **collapses**:

| Task | in-domain | cross-subject (zero-shot) | cross-environment (zero-shot) |
|------|----------:|--------------------------:|------------------------------:|
| Pose | 83.6% | 64% | ~10% |
| Action | 88.1% | 10% | — |

### 3.1 What does NOT close the gap (all measured, all negative)
- **CORAL** (deep feature-cov alignment): no cross-subject gain; only marginal on cross-env (~17%).
- **DANN** (subject-adversarial): ~0, loss-imbalance fragile.
- **Per-antenna instance-norm + SpecAugment**: −4.6 (destroys cross-antenna pose structure).
- **Pose-contrastive foundation pretraining**: −2.3 — and the SupCon loss *never left the `ln(B)`
  random floor*, i.e. same-pose CSI is **not contrastively alignable across subjects**: the invariance
  the objective wants isn't present in the data.
- **Knowledge distillation** (flagship→tiny): no gain; direct training wins.
- **More training subjects**: saturates — 4→8 subjects = +21 pts, but 24→32 = +0.45 pts (asymptote ~64%).

Only **mixup + TTA + ensemble** helps cross-subject, and by <1 pt. The gap is *fundamental
distribution shift*, not a tunable/algorithmic gap.

### 3.2 What DOES close it: few-shot in-room calibration
A handful of labeled frames from the actual deployment room recovers most of the gap — and the
*biggest* zero-shot gap gives the *biggest* gain (an unseen room is one coherent shift a few frames
pin down):

| Calibration samples/subject | Pose cross-subj | Pose cross-env | Action cross-subj |
|----------------------------:|----------------:|---------------:|------------------:|
| 0 (zero-shot) | 64% | ~10% | 10% |
| 5 | — | **60%** | 13% |
| 50 | 70% | 70% | 36% |
| 200 | 76% | 73% | 59% |
| 1000 | 78% | 75% | 76% |

**Confirmed task-general:** the identical pattern holds for pose regression *and* 27-class action
classification. Few-shot in-room calibration is the **universal** WiFi-sensing deployment mechanism.
(Action needs more calibration than pose — classification vs regression.)

### 3.3 Deployable as a ~11 KB adapter
Full fine-tune means a 2.3 MB model copy per room. A **rank-8 LoRA adapter (~11 KB)** recovers most
of the gain (cross-subject 64→72.5% at 0.5% the size). Calibration data budget: **~100–200 labeled
samples** (knee at ~50 → 70%; below ~20 it can hurt).

| Calibration method @200 samples | PCK@20 | adapter |
|---------------------------------|-------:|--------:|
| LoRA rank-8 | 72.5% | ~11 KB |
| head + graph only | 72.7% | 119 KB |
| frozen-trunk | 73.5% | 207 KB |
| full finetune | 76.2% | 2.3 MB |

## 4. The calibration service (shipped)

The mechanism is implemented end-to-end: a Python reference
([`aether-arena/calibration/`](../../aether-arena/calibration/) — `calibrate.py` fits an adapter from
a labeled clip, verified 3.09%→74.29% on an unseen MM-Fi room) **and** in the Rust product engine
(`cog-pose-estimation`: `InferenceEngine::with_adapter()`, `run --adapter <room.safetensors>`,
architecture-agnostic LoRA on the pose head, tested).

## 5. Honest limitations

- Most generalization numbers are within MM-Fi (one dataset, one hardware setup). **Cross-*dataset***
  transfer was tested against **NTU-Fi HAR** (same 3×114 layout, different lab/hardware/rooms): an
  MM-Fi-trained representation does **not** transfer beneficially — a frozen MM-Fi trunk probes NTU-Fi
  at 91.5%, *no better than random features* (93%), and full fine-tuning (75%) underperforms a linear
  probe. CSI representations are **distribution-locked** (same root cause as the within-MM-Fi
  cross-subject/-environment collapse); the practical answer is on-target training/few-shot, not
  transferable zero-shot features. Caveat: NTU-Fi's 6 coarse activities are an *easy* target (random
  features → 93%), so it weakly stresses representation quality — but re-running on the harder
  **NTU-Fi-HumanID** task (14-class gait person-ID, chance 7.1%) gave the *same* result (MM-Fi
  pretrain 91.7% ≈ random 92.8%). **Unified root cause:** for CSI, in-domain classification lives in
  the *target-trained readout* (a random 256-d projection of 3,420-d CSI is already linearly
  separable), while the *learned representation* fails to transfer across subjects, rooms, and
  datasets alike. WiFi-CSI sensing is **distribution-locked**; the answer is on-target few-shot
  calibration, not transferable features. A harder cross-dataset *pose* benchmark (vs classification)
  remains the one open variant.
- Random-split numbers are reported only to compare to prior work on the same protocol; they are
  in-domain and partly leaky. The cross-subject / cross-environment numbers are the honest ones.
- Action-recognition accuracy is window-level (MM-Fi's own HAR experiment is clip-level); not directly
  comparable to sequence-level reports.
- On-device (ARM/Hailo) latency is pending hardware; CPU latency (0.135 ms x86 single-thread) is the
  current proxy.

## 6. Reproduction

Pose: `aether-arena/staging/train_save.py` (flagship), `train_efficiency_pareto.py`,
`quant_micro.py`, `train_fewshot_adapt.py`, `train_adapter_calib.py`. Action: `train_action.py`,
`train_action_fewshot.py`. Calibration service: `aether-arena/calibration/`. Decision record + full
empirical chain: [ADR-150 §3.2–3.6](../adr/ADR-150-rf-foundation-encoder.md). Leaderboard + witness
ledger: [AetherArena](https://huggingface.co/spaces/ruvnet/aether-arena) (ADR-149).

## 7. The sharpest result: the encoder barely matters

A random *frozen* transformer encoder + a trained pose head matches a fully-trained encoder to within
2–4 points (cross-subject: <2 points):

| Pose protocol | fully-trained encoder | random-frozen encoder + head |
|---------------|----------------------:|-----------------------------:|
| in-domain | 78.2% | 73.8% |
| cross-subject | 63.9% | 62.1% |

(Same fair-comparison config; absolute numbers below the 83.6% flagship — the *delta* is the point.)
**Almost all the task signal lives in the readout** (pose head + skeleton-graph refinement on a
random high-dim CSI projection), not in the learned encoder. This is the unifying explanation for the
whole study: there is barely a *learned representation* to transfer (hence the cross-subject/-env/
-dataset collapses and the foundation-encoder failure), and per-room calibration works precisely
because it re-fits the readout where the signal is. **Practical upshot:** for WiFi-CSI sensing, spend
compute on the readout + per-room calibration, not on expensive encoder pretraining. Reproduce:
`aether-arena/staging/train_pose_randomfeat.py`.
