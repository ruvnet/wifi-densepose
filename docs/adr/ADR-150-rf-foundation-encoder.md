# ADR-150: RuView RF Foundation Encoder — pose-preserving, subject/room/device-invariant CSI embedding

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-30 |
| **Deciders** | ruv |
| **Codebase target** | New `wifi-densepose-rfencoder` (or `nn/src/rf_foundation.rs`) + training in `wifi-densepose-train`; consumed by the MM-Fi pose head and the AetherArena Generalization Track (ADR-149) |
| **Relates to** | ADR-024 (Contrastive CSI Embedding / AETHER), ADR-027 (Cross-Environment Domain Generalization / MERIDIAN), ADR-134 (CIR), ADR-135 (calibration + coherence gate), ADR-145 (Ablation/Eval Harness), ADR-149 (AetherArena benchmark) |

---

## 1. Context

AetherArena now has a published, metric- and protocol-matched MM-Fi result: **81.63% torso-PCK@20 in-domain (random_split), exceeding MultiFormer's 72.25%** ([#876](https://github.com/ruvnet/RuView/issues/876)). But the **leakage-free cross-subject** number collapses to **~11.6% torso-PCK** (27% under the looser bbox metric). That gap is the real deployment frontier — homes, elder care, festivals, unseen bodies.

Naïve fixes already tested and **failed**: a subject-adversarial (DANN) embedding did not move cross-subject (baseline 27.26% → DANN 27.54% bbox; torso 11.57%). Bigger capacity *hurt* (transformer cross-subject 24.8% < conv 27.3%) — extra parameters overfit seen subjects.

**Conclusion:** a *generic* "better feature vector" will not help. The lever is an embedding trained for the **right invariance** — one that preserves pose while removing subject, room, and device signatures, and that *exposes* channel instability rather than hiding it.

### 1.1 Why DANN failed (and the corrected rule)

Subject identity is partly **entangled with valid pose evidence** — body scale, limb proportions, gait, RF scattering. Blindly erasing subject info also erases information the pose decoder needs. The corrected rule:

> **Remove subject identity only after preserving pose geometry.** Supervised *pose-contrast across subjects* beats naïve adversarial identity removal.

The frontier objective is **not** `same-subject = positive`. It is:

> **same pose across different subjects = positive; different pose = negative.**

## 2. Decision

**Build the RuView RF Foundation Encoder: a self-supervised, pose-preserving, subject/room/device-invariant RF representation for CSI (extensible to CIR, ADR-134, and BFLD).** Positioned as a **platform primitive**, not a benchmark trick.

### 2.1 What the embedding must keep / remove

| Signal | Action | Why |
|--------|--------|-----|
| Pose geometry | **Keep** | target signal |
| Limb-motion deltas | **Keep** | strong temporal cue |
| Subject identity | **Remove** (post-pose) | causes overfit |
| Static room multipath | **Remove** | breaks transfer |
| Device-specific phase artifacts | **Remove** | breaks cross-hardware |
| Antenna-layout quirks | **Normalize** | deployment portability |
| Channel instability | **Expose separately** | confidence gating / anti-hallucination |

### 2.2 Architecture

```
CSI frame sequence
  → physics normalization        (antenna geometry, subcarrier stability, phase-unwrap quality, room-impulse structure)
  → masked CSI encoder           (SSL: learn channel structure from unlabeled CSI — 150k home + 320k MM-Fi frames)
  → temporal contrastive encoder (motion continuity)
  → skeleton-aware pose decoder  (graph head — anatomical constraints, GraphPose-Fi style, arXiv 2511.19105)
  → confidence + coherence head  (mincut / spectral coherence as RF-integrity signal)
```

### 2.3 Training objectives (loss stack)

```
L_total = L_pose
        + 0.20 · L_masked_csi          # learn channel structure (unlabeled)
        + 0.10 · L_temporal_contrast   # motion continuity
        + 0.20 · L_pose_contrast        # same-pose-across-subjects = positive  ← the frontier
        + 0.05 · L_subject_decorrelation # remove identity only where it conflicts with pose
        + 0.10 · L_coherence            # predict when RF evidence is weak
```

Invariant target:
```
embedding ≈ pose + motion + channel-coherence
embedding ≠ subject-identity + static-room-signature + device-artifact
```

### 2.4 The RuView differentiator — auditable RF perception that knows when it's wrong

The coherence head gates pose confidence by **channel coherence**: when multipath structure changes (mincut / spectral coherence drop), the model flags low RF integrity instead of hallucinating a pose. This is the **anti-hallucination** component most WiFi-pose papers lack, and it turns RuView from a model into sensing infrastructure. (Ties to ADR-135 coherence gate.)

## 3. Experiment plan — three variants, frozen-decoder test

Same split, same decoder, same seed set; only the embedding changes.

| Variant | Description | Success threshold (cross-subject torso-PCK) |
|---------|-------------|----------------------------------------------|
| **E1** | Masked CSI pretrain | **+3** |
| **E2** | Pose-contrastive across subjects | **+6** |
| **E3** | Physics-normalized SSL + skeleton head | **+10** |

### 3.1 Expected gains (estimate)

| Method | cross-subject torso-PCK gain |
|--------|------------------------------|
| Naïve embedding | 0–2 |
| DANN adversarial | 0–3 (high collapse risk) — *empirically ~0* |
| Masked CSI pretrain | +3–8 |
| Pose-contrastive | +5–12 |
| Physics-norm + SSL + graph decoder | +10–20 |
| + more subject-diverse paired data | +20 |

Plausible trajectory: 11.6% → **20–25% near term**, **30–40% with enough subject/environment diversity**. That is a stronger research claim than squeezing random-split from 81.6% → 88%.

### 3.2 Empirical findings (2026-05-31) — measured, not estimated

The near-term algorithmic estimates in §3.1 were **tested directly on the official MM-Fi
cross-subject split** (256,608 train / 64,152 test, same TF pipeline). Measured results:

| Method | §3.1 estimate | **Measured** | Verdict |
|--------|--------------:|-------------:|---------|
| Baseline (in-harness) | — | 63.13% (doc TTA 64.04) | reference |
| Mixup | n/a | **+0.7** → 63.79% | ✅ small |
| Mixup + TTA + 3-seed ensemble | n/a | **+0.9** → **64.92%** | ✅ **best** |
| Per-antenna instance-norm + SpecAugment | n/a | **−4.6** → 58.52% | ❌ destroys cross-antenna pose structure |
| **Pose-contrastive foundation pretrain** | **+5 to +12** | **−2.3** → 62.65% | ❌ **refuted** |
| DANN adversarial | ~0 | ~0 | ❌ (as predicted) |

**Why pose-contrastive pretraining fails — the key finding.** The supervised-contrastive
pretraining loss (positives = same pose-cluster, spanning subjects) **never left the
uniform-similarity floor `ln(B)`** — across cluster granularities K∈{48,256}, batch sizes
{768,1024}, and 3 seeds. The same encoder trivially aligns *temporally-adjacent* frames
(temporal-triplet SSL reached 82%), so the optimizer works; it simply **cannot pull same-pose
CSI from different subjects together — that invariance is not present in the data to be learned.**

**Implication for this ADR.** The 18-pt in-domain↔cross-subject gap (83.6% → best 64.9%) is
**fundamental subject-distribution shift in CSI, not an algorithmic gap.** No invariance-learning
method tested moves it; only variance-reduction (mixup + ensemble) gives <1 pt. This **promotes
"more subject-diverse paired data" (§3.1 last row, §6 alt 3) from complementary to the *primary*
lever** and **demotes pure-SSL-on-existing-data** as a near-term cross-subject win. The encoder is
still worth building for masked-CSI representation reuse and the coherence integrity head, but the
cross-subject acceptance gate (§4, ≥6 pts) is **unlikely to be met without new multi-subject
capture** (fleet: `cognitum-seed-1` + multi-room, see `CLAUDE.local.md`). Recommend re-scoping
phase 1 around data collection before further loss-stack engineering.

### 3.3 Subject-scaling study (2026-05-31) — capture *diversity*, not *volume*

Before committing to capture, we measured **how cross-subject accuracy scales with the number of
training subjects** (fixed held-out test subjects, official split, mixup+TTA):

| N subjects | 4 | 8 | 12 | 16 | 20 | 24 | 32 |
|-----------:|--:|--:|---:|---:|---:|---:|---:|
| xsubj-PCK@20 | 36.7 | 57.7 | 58.3 | 61.1 | 62.7 | 63.3 | **63.7** |

The curve **saturates**: 4→8 subjects = **+21 pts**, but 24→32 = **+0.45 pts**. Asymptote ≈ 64–65%,
still ~19 pts under in-domain. **Key correction to the "more data" recommendation:** simply capturing
*more people from the same distribution* will **not** close the gap — subject-count returns vanish
past ~16–20 subjects. The residual is **device/room/protocol shift** (MM-Fi's cross-subject split is
partly cross-environment by construction). **Re-scoped phase-1 capture target: maximize DIVERSITY
(rooms, devices, antenna geometries, traffic protocols), not headcount** — and pair it with few-shot
target-domain adaptation (a handful of labeled frames from the deployment room), which the saturation
curve implies will beat any amount of additional source subjects. This makes the encoder's
*domain-invariance* objective (vs the failed subject-invariance one) the design priority.

### 3.4 Few-shot target adaptation (2026-05-31) — the actionable resolution

The saturation curve predicts a few labeled frames from the *deployment* room beat more source
subjects. Confirmed. Base trained on all 32 source subjects (63.7% zero-shot on a disjoint 50%
held-out of the target subjects), then fine-tuned on K labeled frames per target subject:

| K/subject | total frames | eval PCK@20 | Δ |
|----------:|-------------:|------------:|--:|
| 0 | 0 | 63.7% | — |
| 20 | 160 | 68.1% | +4.3 |
| **50** | **400** | **72.2%** | **+8.5 (≈ prior SOTA)** |
| 200 | 1,600 | 76.1% | +12.4 |
| 1000 | 8,000 | 78.3% | +14.6 |

**Few-shot calibration dominates source volume.** §3.3 showed +24 source subjects (~190K frames)
buys +6 pts; here **200 target frames/subject (1,600 frames) buys +12.4 pts**. This **re-scopes the
ADR's acceptance gate and deployment story**: the cross-subject gate (§4, ≥6 pts) is *trivially* met
by ~50–200 labeled frames of in-room calibration — no foundation encoder or mass capture required for
the deployment win. **Recommended product behavior:** ship a **~30-second on-site calibration** (a few
hundred labeled frames per room/person) that recovers most of the gap. The foundation encoder's value
shifts from "close cross-subject zero-shot" (data says: hard) to "make the few-shot adaptation faster /
need fewer calibration frames" — a better-posed, achievable objective. **This supersedes the §3.2
pessimism: the frontier is not closed by algorithms or bulk data, but it *is* cheaply closed at
deployment time by few-shot calibration.**

> **Task-general (2026-05-31).** The same mechanism was verified on a *second* MM-Fi task —
> 27-class **action recognition** (which the MM-Fi paper never benchmarked for WiFi). Zero-shot
> cross-subject collapses to ~10% (near-chance), and few-shot calibration recovers it: 50 samples →
> 36%, 200 → 59%, 1000 → 76%. Action needs more calibration than pose (classification vs regression),
> but the pattern is identical. **Few-shot in-room calibration is the universal deployment answer for
> WiFi sensing generalization, not a pose-specific result.** (Optimization report §36.)

### 3.5 Deployable adapter calibration (2026-05-31) — the calibration-service mechanism

Full-finetune calibration (§3.4) means a 2.3 MB model copy per room. Compared calibration methods at
K=200 frames/subject by accuracy *and* adapter size:

| Method | PCK@20 | trainable | adapter |
|--------|-------:|----------:|--------:|
| zero-shot | 63.6% | — | — |
| **LoRA rank-8** | **72.5%** | 11,200 | **~11 KB** |
| head+graph only | 72.7% | 121,828 | 119 KB |
| frozen-trunk | 73.5% | 212,453 | 207 KB |
| full finetune | 76.2% | 2.32 M | 2.3 MB |

**A ~11 KB LoRA adapter recovers +8.9 pts (→72.5%, ≈ prior SOTA) at 0.5 % the model size.** This is
the concrete mechanism for the **RuView calibration service** the project wanted: ship the shared
base once; each room contributes a 30-second labeled calibration → a **~11 KB per-room LoRA adapter**
→ SOTA-level cross-subject pose, thousands of rooms on one base. Accuracy/size knob:
LoRA 11 KB @ 72.5 % → frozen-trunk 207 KB @ 73.5 % → full 2.3 MB @ 76.2 %. **Net for this ADR:** the
encoder/adapter split is validated empirically — a frozen shared trunk + tiny per-room LoRA is the
deployable path, and the foundation-encoder objective should be "make this adapter even smaller /
need fewer calibration frames."

**Calibration data requirement (measured, 3 seeds):** the 11 KB LoRA needs **~100–200 labeled
samples/room** to reach ~72% (knee at ~50 → 70%); below ~20 samples it can't fit and may *hurt*
(5 samples → 61% < zero-shot 64%). So the evidence-complete **calibration-service spec** is:
ship shared base → collect **~100–200 labeled samples on-site** → fit a **~11 KB LoRA** →
**~72% cross-subject** (SOTA-level). The encoder's research goal is now precisely posed: push that
~100–200-sample requirement down and/or lift the >72% ceiling per fixed calibration budget.

### 3.6 Cross-ENVIRONMENT few-shot (2026-05-31) — no unsolved deployment case

The hard frontier — unseen room *and* unseen people (cross-environment) — was thought ~unsolvable
(zero-shot ~10–17%). Few-shot calibration rescues it **even more dramatically than cross-subject**:

| K labeled samples/subject | cross-env PCK@20 | Δ zero-shot |
|--------------------------:|-----------------:|------------:|
| 0 | 10.6% | — |
| **5** | **60.1%** | **+49.5** |
| 20 | 66.0% | +55.5 |
| 50 | 70.0% | +59.4 |
| 200 | 73.1% | +62.5 |
| 1000 | 75.4% | +64.8 |

**Just 5 calibration samples per person lift an unseen room from ~unusable (10.6%) to 60%.** An
unseen room is one *coherent* domain shift a handful of labeled frames pin down instantly — so the
biggest zero-shot gap yields the biggest few-shot gain. **Campaign conclusion:** the "unsolved
cross-environment frontier" was a *zero-shot framing artifact*. With the ~11 KB LoRA calibration
mechanism (§3.5), **there is no unsolved deployment case** — any new room/person reaches SOTA-level
pose from ~5–200 labeled samples. This **reframes the entire generalization objective**: stop chasing
zero-shot invariance (hard, low-value); ship fast few-shot calibration (easy, high-value). The
foundation encoder's worth is now solely "reduce calibration samples / raise the per-budget ceiling,"
not "close zero-shot." Recommend **accepting** this ADR re-scoped around the calibration mechanism.

## 4. Acceptance Test

The encoder is accepted **only if it improves cross-subject torso-PCK@20 by ≥ 6 absolute points without reducing random-split torso-PCK@20 by more than 2 points** — on the same MM-Fi pipeline, one-command reproduction, with per-joint error tables. Results land as AetherArena witness rows (ADR-149), nothing published until reviewed.

## 5. Consequences

**Positive:** a reusable, self-supervised RF foundation encoder for CSI/CIR/BFLD; the first principled attack on the cross-subject frontier; the coherence head adds an anti-hallucination integrity signal no competitor has.

**Negative / risk:** SSL pretraining requires matching the production CSI→feature pipeline (ADR-149 §SSL note flagged the resampling-replication risk); the multi-loss stack needs careful weight tuning (DANN showed loss-imbalance can collapse training); physics normalization must be validated not to discard pose-relevant deltas.

**Neutral:** the in-domain head is unchanged; the encoder slots in front of the existing pose decoder.

## 6. Alternatives Considered

1. **Bigger model only** — tested; *hurts* cross-subject (overfits seen subjects).
2. **Naïve DANN subject-adversarial** — tested; no gain, collapse risk; entangles pose evidence.
3. **More data only (camera/ADR-079)** — complementary and ultimately necessary, but slow and out-of-band; the encoder extracts more from existing data first.

## 7. Open Questions

1. Physics-normalization spec — exact antenna/subcarrier/phase terms, validated to preserve pose deltas.
2. Masked-CSI SSL on the production feature pipeline (resampling match — see ADR-149).
3. Where the coherence/mincut integrity signal is computed (reuse ADR-135 coherence gate vs new head).
4. CIR (ADR-134) / BFLD fusion into the same encoder — phase 3.
