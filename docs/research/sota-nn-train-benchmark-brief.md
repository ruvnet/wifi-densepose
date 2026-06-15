# SOTA Evidence Brief — `wifi-densepose-nn` / `wifi-densepose-train` Benchmark ADR Seed

| Field | Value |
|-------|-------|
| **Date** | 2026-06-14 |
| **Author** | deep-research (Opus) |
| **Purpose** | Seed a future benchmark/optimization ADR for the NN-inference (`wifi-densepose-nn`) and training (`wifi-densepose-train`) crates |
| **Scope** | The DELTA beyond what ADR-152 / ADR-150 / ADR-015 already establish — current published WiFi-CSI pose SOTA, winning architectures, edge-quantization SOTA, and a defensible benchmark-suite design |
| **Ethos** | Every claim graded PEER-REVIEWED / PREPRINT / VENDOR-CLAIM / BLOG, with MEASURED-on-public-benchmark distinguished from marketing. Numbers that could not be verified are flagged. No fabricated citations. |

> **Citation discipline carried in from ADR-152 §2.2:** preprint accuracy numbers are CLAIMED until reproduced on our hardware. The project has already retracted its own "92.9% PCK@20" and "shipped-WiFlow-STD 97.25%" figures after measurement; this brief inherits that bar.

---

## 1. Executive summary

**Where the project stands vs the 2026 frontier.** The repo is, by the evidence already in-tree, *ahead of most academic groups on benchmark hygiene* and roughly *at parity on capability* — but the two are measured on incompatible yardsticks, which is the single biggest risk to any "beyond-SOTA" claim.

- The project's headline reproductions (`benchmarks/wiflow-std/RESULTS.md`) are MEASURED and rigorous: WiFlow-STD retrained to **96.09–96.61% PCK@20** on the authors' own 360k-window 2D dataset (RTX 5080), shipped checkpoint REFUTED, dataset/code defects documented. This is a genuinely strong, reproducible result.
- **But that number is not on a standard public benchmark.** WiFlow-STD's dataset is self-collected (5 subjects, 15 keypoints, 2D, in-domain random split, hardware unspecified). The academic frontier on the *standard* public 3D benchmark (MM-Fi) reports **PCK@20 ≈ 61% / MPJPE ≈ 161 mm random-split** (GraphPose-Fi, Nov 2025) — a *harder* metric (3D, mm-scale, standard PCK normalization). The project's own AetherArena MM-Fi number (**81.63% torso-PCK@20 in-domain**, ADR-150) uses a *torso-normalized PCK* that is looser than GraphPose-Fi's standard PCK, so the three numbers (96% / 81.6% / 61%) **cannot be lined up** without a unified harness. Making them comparable IS the highest-value work item.
- The deployment frontier — **cross-subject / cross-environment generalization** — is where everyone collapses, the project included (ADR-150: 81.63% in-domain → ~11.6% leakage-free cross-subject). GraphPose-Fi independently confirms the cliff (61.1% random → 12.9% cross-environment PCK@20). This is the real research target, not in-domain PCK.

**Top 3 highest-value optimization/benchmark targets:**

1. **A unified, metric-locked accuracy harness in `wifi-densepose-train`** that scores any model under *one* explicit PCK definition (normalization, keypoint convention, split) so WiFlow-STD-repro, AetherArena/MM-Fi, and GraphPose-Fi numbers become directly comparable. Without this, no "beyond-SOTA" claim survives the "prove it" bar — the project has already been burned twice by metric ambiguity (the retracted 92.9% used absolute, not torso-normalized, PCK).
2. **A QAT path for the WiFlow-STD-class edge model.** The in-tree edge work (`RESULTS.md`) has *fully characterized PTQ* (static QDQ conv-only is the int8 sweet spot; dynamic int8 is a no-op on this all-conv architecture) and found the **half model (843k params) strictly dominates the published 2.23M** and **tiny (56k, 295 KB ONNX fp32) holds 94.1% PCK@20**. The one untested lever is **quantization-aware training**, which the general literature says recovers most of the PTQ accuracy gap. That is the next defensible edge win.
3. **Criterion-backed regression benches wired into CI** for the real Candle/ONNX forward path. The benches *exist* (`wifi-densepose-nn/benches/{inference,onnx,native_conv}_bench.rs`, `wifi-densepose-train/benches/training_bench.rs`) and `benchmarks/edge-latency/RESULTS.md` shows the methodology is sound (host≠ESP32 caveat made explicit). The gap is turning point-in-time captures into committed regression baselines.

---

## 2. Findings per research question

### RQ1 — Latest WiFi-CSI pose SOTA (2024–2026): published PCK@20 / MPJPE on the standard public benchmarks

The crucial framing: **"WiFi pose SOTA" splits into two non-comparable tracks** — 3D pose on MM-Fi/Person-in-WiFi-3D (mm-scale MPJPE, standard PCK) vs 2D pose on self-collected sets (image-normalized PCK). The project's flagship reproduction lives in the second track; the academic frontier lives in the first.

| Method | Venue / Year | Benchmark + split | PCK@20 | MPJPE | Grade |
|---|---|---|---|---|---|
| **GraphPose-Fi** (arXiv [2511.19105](https://arxiv.org/abs/2511.19105)) | PREPRINT, Nov 2025 | MM-Fi P1, **random split** | **61.1%** | **160.6 mm** (PA-MPJPE 105.0) | numbers MEASURED-in-study (preprint); beats MetaFi++, HPE-Li, DT-Pose |
| GraphPose-Fi | same | MM-Fi P1, **cross-subject** | 44.2% | 210.5 mm | same |
| GraphPose-Fi | same | MM-Fi P1, **cross-environment** | 12.9% | 302.7 mm | same — the generalization cliff |
| **DT-Pose** (arXiv [2501.09411](https://arxiv.org/abs/2501.09411)) | PREPRINT (ICLR'25 OpenReview [aPnLQ6WfQQ](https://openreview.net/forum?id=aPnLQ6WfQQ)), Jan 2025; code [cseeyangchen/DT-Pose](https://github.com/cseeyangchen/DT-Pose) | MM-Fi (domain-gap + topology focus) | not cleanly extractable from abstract | reports MPJPE; self-supervised masked pretrain + topology decode | numbers NOT verified at exact-table level here — flagged |
| **Person-in-WiFi-3D** (CVPR 2024, [openaccess](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html)) | **PEER-REVIEWED**, CVPR 2024 | own 97k-frame multi-person set | — (multi-person, not single-PCK) | **91.7 mm (1p) / 108.1 (2p) / 125.3 (3p)** 3D joint error | MEASURED (peer-reviewed); own dataset, not MM-Fi |
| **WiFlow-STD** (arXiv [2602.08661](https://arxiv.org/abs/2602.08661), [DY2434 repo](https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling)) | PREPRINT, Apr 2026 | self-collected, 5-subj, **2D, in-domain random** | 97.25% (claimed) | 0.007 m (image-norm) | claimed CLAIMED; **project reproduced 96.09–96.61% (MEASURED, RTX 5080)** after repairing dataset/code |
| **PerceptAlign** (arXiv [2601.12252](https://arxiv.org/abs/2601.12252)) | PREPRINT + MobiCom'26 acceptance | own 7-layout cross-domain 3D set | — | 222.4 mm (Scene4) / 317.1 (Scene5), claims −54% cross-env vs SOTA | CLAIMED (preprint); failure mode corroborated |
| **Project AetherArena** (ADR-150, [issue #876](https://github.com/ruvnet/RuView/issues/876)) | internal | MM-Fi, **random split**, **torso-PCK** | **81.63% torso-PCK@20** | — | MEASURED-internal; **torso-PCK ≠ GraphPose-Fi standard PCK** |
| **Project WiFlow-STD repro** (`benchmarks/wiflow-std/RESULTS.md`) | internal | their data, their split | **96.09–96.61%** | 0.0094–0.0098 m | MEASURED-internal (RTX 5080) |

**How the project's ~96% compares to the frontier:** It is *not directly comparable*. The 96% is on an easier task (2D, in-domain, image-normalized PCK, single-environment, 5 subjects) than GraphPose-Fi's 61.1% (3D, standard PCK, mm-scale). The project's own MM-Fi-track number (81.63% torso-PCK@20) *appears* to beat GraphPose-Fi's 61.1%, **but only because torso-PCK is a looser normalization** — the project explicitly flags this (ADR-150 cites beating "MultiFormer's 72.25%" under the *same* torso metric, not GraphPose-Fi's). The honest statement: **the project is competitive on in-domain MM-Fi under its own torso metric, and collapses cross-subject exactly as the published frontier does.** No public number lets the project claim "beyond-SOTA" today.

### RQ2 — What's winning architecturally now (2025–2026)

The clear trend across the verified 2025–2026 papers:

- **Graph / skeleton-aware decoders are the current academic SOTA on MM-Fi.** GraphPose-Fi (PREPRINT, Nov 2025) wins by injecting anatomical graph structure into the decoder — exactly the `GraphPose-Fi-style skeleton-aware graph head` ADR-150 §2.2 already names as the planned decoder. *The project's architecture direction matches the frontier.*
- **Self-supervised masked pretraining (MAE) is the cross-domain lever, not capacity.** UNSW MAE study (arXiv [2511.18792](https://arxiv.org/abs/2511.18792), PREPRINT, Nov 2025): cross-domain gains scale **log-linearly with pretraining data, unsaturated at 1.3M samples**; ViT-Base adds only 0.4–0.9% over ViT-Small. Recipe: **80% masking, (30,3) small patches**. DT-Pose (arXiv 2501.09411) independently uses masked pretraining + topology constraints for the domain gap. *Caveat (MEASURED in ADR-152 §2.3): UNSW's downstream tasks are classification, not pose — pose transfer remains a hypothesis. The project's own measurement (b) found WiFlow-STD pretrained features give optimization transfer but NOT feature transfer to ESP32 CSI.*
- **Spatio-temporal decoupling is the efficiency lever.** WiFlow-STD's whole contribution is decoupling spatial and temporal CSI processing to hit 2.23M params. The project verified the params/FLOPs (MEASURED) and then **beat it**: the half-model (843k) matches accuracy with 0.38× params (`RESULTS.md` efficiency sweep).
- **Geometry/layout conditioning is the cross-layout lever.** PerceptAlign (MobiCom'26): fusing transceiver-position embeddings + two-checkerboard calibration, claimed −60% cross-domain. ADR-152 §2.1 already adopted this (`NodeGeometry`, geometry embeddings).
- **NOT winning / absent:** diffusion models for CSI pose did not surface in the verified frontier. Full DensePose-UV regression from commodity WiFi remains undemonstrated (ADR-152 F5, MEASURED by full-text screening). No 2025–2026 paper was found that *beats the project's current direction* — the project is tracking, not trailing, the architecture frontier.

**Verdict RQ2:** the winning stack (MAE pretrain → graph/skeleton decoder → geometry conditioning, ViT-Small-class capacity) is *already the planned ADR-150/152 stack*. The gain available is not a new architecture; it's (a) more heterogeneous pretraining data and (b) honest cross-domain measurement.

### RQ3 — Edge/quantized inference SOTA for small CSI pose models

The in-tree edge work (`benchmarks/wiflow-std/RESULTS.md` "Edge optimization" + "Static PTQ" + "Efficiency sweep") is already at or beyond what the public literature offers for this specific model class, and is MEASURED. Key findings to carry forward:

- **Dynamic INT8 is a trap on all-conv CSI models.** WiFlow-STD has **zero `nn.Linear` layers** (21 Conv1d + 22 Conv2d + BatchNorm). `torch.quantize_dynamic` quantizes 0% of params (dynamic int8 has no conv kernels). MEASURED.
- **Static QDQ conv-only PTQ is the int8 sweet spot.** PCK@20 96.60–96.63% (vs fp32 96.68%, dynamic 96.52%), 2.53 MB. All-ops QDQ is strictly worse (−1.4 pt). MEASURED.
- **ONNX Runtime fp32 is the real CPU latency win**: 3.2 ms/window batch-1 vs torch 11.0 ms (~3.4×) at parity (2.4e-7). int8 is ~2× *slower* than ONNX fp32 at batch-1 (ConvInteger kernels). MEASURED.
- **Smaller-than-published dominates.** half (843k) ≥ full on accuracy; **tiny (56k, 295 KB ONNX fp32, 0.66 ms/win, 94.1% PCK@20)** is the smallest deployable artifact. At tiny scale int8 is a *bad* trade (−1.43 pt for −47 KB). MEASURED.
- **General QAT-vs-PTQ context (BLOG/VENDOR):** [NVIDIA TensorRT QAT blog](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/), [Ultralytics QAT glossary](https://www.ultralytics.com/glossary/quantization-aware-training-qat), [ONNX Runtime quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html): QAT "almost always" recovers accuracy PTQ loses on sensitive models; ONNX Runtime does NOT retrain (QAT must happen in PyTorch, then export QDQ). The [Onboard Optimization survey, arXiv 2505.08793](https://arxiv.org/pdf/2505.08793) (PREPRINT) covers on-device optimization broadly. These are *general* claims, not CSI-pose-specific — grade accordingly.
- **Hailo / Pi target (CLAUDE.local.md):** the 4× Pi+Hailo cluster (Hailo-8 @ 26 TOPS / Hailo-10 @ 40 TOPS) needs a **HEF** compile path, which is its own toolchain (not ONNX/Candle). No in-tree HEF benchmark exists yet — this is a genuine gap for the edge-inference claim.

**Actionable for an inference-speed benchmark:** the honest comparand set is `{torch fp32, ONNX fp32, ONNX static-QDQ-conv-only int8, candle fp32}` × `{full, half, tiny}` on a fixed host, with the **host≠ESP32 / host≠Hailo caveat stated up front** (the `edge-latency/RESULTS.md` template already does this correctly). The one new datapoint worth producing: **QAT-int8 on the half model** to test whether QAT closes the PTQ −0.16 pt gap *and* keeps the size win.

### RQ4 — Rigorous, reproducible benchmark methodology

The repo already demonstrates the right methodology in three places — the ADR should codify it, not invent it:

- **`benchmarks/wiflow-std/RESULTS.md`** — the gold standard already in-tree: pinned upstream commit, seed-42 file-level split documented, corruption masks committed as ground truth, every forced deviation recorded, mean-pose honesty baseline, MEASURED-vs-CLAIMED grading.
- **`benchmarks/edge-latency/RESULTS.md`** — criterion 0.5, explicit host machine, low/median/high brackets, contention caveat, host≠ESP32 separation, steady-state-vs-cold-start distinction.
- **Rust micro-bench:** criterion benches already exist in both crates (`wifi-densepose-nn/benches/`, `wifi-densepose-train/benches/`).

What a credible "beyond-SOTA" claim requires (the bar that survives "prove it"):
1. **One locked accuracy definition** — PCK normalization (torso vs absolute vs bbox), keypoint convention (15 vs 17 COCO), and split (random / cross-subject / cross-environment) declared *before* the run. The retracted 92.9% died exactly because PCK normalization was unstated.
2. **A mean-pose / constant-output honesty baseline** on every split (already done in measurement (b) — a single-subject near-static set scored 95.9% torso-PCK@20 with a *constant* pose). Any claim must beat this.
3. **MEASURED-vs-CLAIMED grading** per number, with the exact command and raw-JSON path committed.
4. **Cross-domain, not just in-domain.** In-domain PCK is saturated and uninformative; the defensible claim is on cross-subject/cross-environment, where the frontier is 12–44% PCK@20.

---

## 3. Proposed benchmark-suite design

A two-part suite (`wifi-densepose-train` accuracy harness + `wifi-densepose-nn` latency harness), both committing raw JSON + a graded RESULTS.md.

### 3.1 Accuracy harness (`wifi-densepose-train`)

- **Metric module with one canonical PCK** (parameterized: `{torso, bbox, absolute}` normalization × threshold × keypoint-map), so a single function scores WiFlow-STD-repro, MM-Fi/AetherArena, and a GraphPose-Fi re-run identically. Lock the default to **torso-PCK@20 on 17-kp COCO** and *always* also print standard-PCK to expose the gap.
- **Fixed datasets/splits:** (i) WiFlow-STD cleaned 360k (their split, for repro parity), (ii) MM-Fi P1 random + cross-subject + cross-environment (to line up against GraphPose-Fi 61.1/44.2/12.9 and the project's 81.63), (iii) ESP32 paired eval set when ≥2k multi-subject windows exist.
- **Mandatory honesty baselines** emitted every run: mean-pose, constant-output, and (for cross-domain) source-only.
- **Output:** raw JSON + a RESULTS.md table with MEASURED/CLAIMED grades, mirroring `benchmarks/wiflow-std/RESULTS.md`.

### 3.2 Latency/size harness (`wifi-densepose-nn`)

- **Matrix:** `{torch fp32 (ref), ONNX fp32, ONNX static-QDQ-conv-only int8, candle fp32}` × `{full 2.23M, half 843k, tiny 56k}` × `{batch 1, 64}`, criterion-timed, host declared.
- **Report:** disk size, batch-1 + batch-64 ms/window (median + low/high), and PCK@20 on the locked 10k-window subset, so latency and accuracy never get cited apart.
- **Caveat block up front:** host ≠ ESP32-S3/WASM3, host ≠ Hailo HEF. No host number is presented as the edge number.
- **CI gate:** commit the current medians as regression baselines; fail PRs that regress latency >X% or accuracy >Y pt.

### 3.3 What counts as a defensible "beyond-SOTA" result

A claim is citable only if **all** hold: (1) scored under a pre-declared metric/split, (2) beats the relevant published frontier number *on the same metric definition* (e.g. >61.1% standard-PCK@20 on MM-Fi random, or >12.9% on cross-environment), (3) beats the mean-pose honesty baseline, (4) raw JSON + exact command committed, (5) graded MEASURED. The single most valuable "beyond-SOTA" target is **cross-environment MM-Fi**, where the published bar (12.9% PCK@20) is low enough that a real win is both achievable and unambiguous.

---

## 4. Gap table

| Capability | Project current (graded) | Published SOTA (graded) | Proposed target | Data / hardware needed |
|---|---|---|---|---|
| In-domain 2D PCK@20 (self-collected) | 96.09–96.61% (MEASURED, RTX 5080, WiFlow-STD repro) | 97.25% claimed (WiFlow-STD, CLAIMED) | match within noise + own architecture | cleaned 360k dataset (have); already met |
| In-domain MM-Fi PCK@20 (torso-norm) | 81.63% torso-PCK (MEASURED-internal) | GraphPose-Fi 61.1% *standard*-PCK (PREPRINT) — **not comparable** | re-score both under **one** PCK def | MM-Fi P1 (have); unified metric harness (gap) |
| **Cross-subject MM-Fi PCK@20** | ~11.6% torso (MEASURED, the cliff) | GraphPose-Fi 44.2% standard (PREPRINT) | close gap via MAE pretrain + graph decoder | 1.3M heterogeneous CSI corpus (ADR-150/152 §2.3), ViT-Small encoder |
| **Cross-environment MM-Fi PCK@20** | untested-internal | GraphPose-Fi 12.9% standard (PREPRINT) | **beat 12.9% → cleanest beyond-SOTA win** | MM-Fi cross-env split + geometry conditioning (ADR-152 §2.1) |
| ESP32 CSI→pose (17-kp) | no run beats mean-pose baseline (MEASURED, measurement b) | n/a (no public ESP32 pose benchmark) | beat mean-pose on temporal split | ≥2k multi-subject/multi-position paired windows (gap) |
| Edge int8 size/accuracy | static QDQ conv-only 96.61% @ 2.53 MB; tiny 94.1% @ 295 KB fp32 (MEASURED) | no model-matched public number | **QAT-int8 on half model** (untested lever) | PyTorch QAT + QDQ export; RTX 5080 (have) |
| Edge CPU latency | ONNX fp32 3.2 ms/win b1 host (MEASURED) | n/a (model-specific) | committed criterion regression baseline | host bench (have); ESP32/Hailo on-hardware (gap) |
| Hailo HEF edge inference | none in-tree (gap) | n/a | first MEASURED HEF latency | Hailo compile toolchain + Pi cluster (have hardware, CLAUDE.local.md) |
| Foundation encoder (MAE) | recipe adopted, untrained (ADR-152 §2.3) | UNSW: log-linear cross-domain scaling on *classification* (PREPRINT) | pose-transfer validation (hypothesis today) | 1.3M-sample corpus aggregation (priority per F3) |

---

## 5. Sources (graded)

| Source | Type | Grade | Used for |
|---|---|---|---|
| GraphPose-Fi, arXiv [2511.19105](https://arxiv.org/abs/2511.19105) | preprint | PREPRINT; table numbers MEASURED-in-study (fetched + quoted) | RQ1 MM-Fi frontier (61.1/44.2/12.9 PCK@20, 160.6/210.5/302.7 mm) |
| WiFlow-STD, arXiv [2602.08661](https://arxiv.org/abs/2602.08661) + [DY2434 repo](https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling) | preprint+code | numbers CLAIMED; artifacts MEASURED; **project repro 96% MEASURED** | RQ1/RQ2/RQ3 |
| PerceptAlign, arXiv [2601.12252](https://arxiv.org/abs/2601.12252) | preprint + MobiCom'26 acceptance | CLAIMED numbers; failure mode corroborated | RQ1/RQ2 geometry conditioning |
| UNSW MAE, arXiv [2511.18792](https://arxiv.org/abs/2511.18792) | preprint | ablations MEASURED-in-study; pose transfer = hypothesis | RQ2 MAE recipe |
| DT-Pose, arXiv [2501.09411](https://arxiv.org/abs/2501.09411), OpenReview [aPnLQ6WfQQ](https://openreview.net/forum?id=aPnLQ6WfQQ), [code](https://github.com/cseeyangchen/DT-Pose) | preprint+code (ICLR'25) | exact MPJPE table NOT verified here — flagged | RQ2 masked-pretrain + topology |
| Person-in-WiFi-3D, [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.html) | peer-reviewed | MEASURED (91.7/108.1/125.3 mm); own dataset | RQ1 3D multi-person frontier |
| ONNX Runtime quantization [docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) | vendor docs | VENDOR | RQ3 PTQ/QAT mechanics |
| NVIDIA TensorRT QAT [blog](https://developer.nvidia.com/blog/achieving-fp32-accuracy-for-int8-inference-using-quantization-aware-training-with-tensorrt/), [Ultralytics](https://www.ultralytics.com/glossary/quantization-aware-training-qat) | vendor/blog | BLOG/VENDOR; general, not CSI-specific | RQ3 QAT>PTQ context |
| Onboard Optimization survey, arXiv [2505.08793](https://arxiv.org/pdf/2505.08793) | preprint | PREPRINT | RQ3 on-device optimization landscape |
| In-tree `benchmarks/wiflow-std/RESULTS.md`, `benchmarks/edge-latency/RESULTS.md`, ADR-150, ADR-152, ADR-015 | internal MEASURED | MEASURED-internal | grounding, all RQs |

**Unverified / flagged:** DT-Pose exact MM-Fi MPJPE table not extracted at primary-source precision (abstract-level only). GraphPose-Fi parameter count not reported in the paper. WiFlow-STD/PerceptAlign accuracy numbers are author-self-reported preprints. No CSI-pose-specific QAT benchmark exists in the public literature — the QAT recommendation rests on general (non-CSI) vendor/blog evidence.
