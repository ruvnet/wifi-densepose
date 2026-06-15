# ADR-152: WiFi-Pose SOTA 2026 Intake — Geometry-Conditioned Calibration, External Benchmarks, and the Foundation-Encoder Training Recipe

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-06-10 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-calibration` (geometry conditioning, ADR-151 Stage 2), `wifi-densepose-train` (camera-supervised path, MAE recipe), `wifi-densepose-cli` (benchmark harness), docs |
| **Relates to** | ADR-151 (Per-Room Calibration), ADR-150 (RF Foundation Encoder), ADR-135 (Empty-Room Baseline), ADR-079 (Camera-Supervised Pose), ADR-027 (MERIDIAN), ADR-024 (AETHER), ADR-149 (AetherArena), ADR-029 (Multistatic) |
| **Research provenance** | Deep-research run 2026-06-10: 22 sources fetched, 110 claims extracted, 25 adversarially verified (3-vote), 24 confirmed / 1 refuted. Evidence grades per source below. |

---

## 1. Context

A structured survey of the 2025–2026 WiFi human-sensing state of the art was run on 2026-06-10 to answer: *what should RuView integrate next, and does anything published invalidate our current direction?* Every claim below was verified against the primary source by independent adversarial reviewers; **evidence grades distinguish what the papers measured from what they merely claim**. Almost all performance numbers are author-self-reported preprint results — treated here as CLAIMED until reproduced on our hardware.

### 1.1 The five verified findings

**(F1) "Coordinate overfitting" is a named, diagnosed failure mode of camera-supervised WiFi pose — and our ADR-079 pipeline has the exact shape of it.**
PerceptAlign (arXiv [2601.12252](https://arxiv.org/abs/2601.12252), accepted ACM MobiCom 2026) shows that models regressing CSI directly to camera-frame coordinates memorize the deployment-specific transceiver layout; SOTA baselines degrade to >600 mm MPJPE in unseen scenes. Their fix is cheap: a <5-minute calibration using two checkerboards and a few photos to align WiFi and vision in one shared 3D frame, plus **fusing transceiver-position embeddings with CSI features**. Claimed: −12.3% in-domain error, −60%+ cross-domain error. They release the claimed-largest cross-domain 3D WiFi pose dataset (21 subjects, 5 scenes, 18 actions, **7 device layouts**). *Evidence: improvements CLAIMED (preprint w/ MobiCom acceptance); the failure mode itself is corroborated across the cross-domain literature — and independently by our own ADR-150 data (81.63% in-domain vs ~11.6% leakage-free cross-subject torso-PCK).*

**(F2) An external model named "WiFlow" claims 97.25% PCK@20 with 2.23M params and ships everything.**
arXiv [2602.08661](https://arxiv.org/abs/2602.08661) (Apr 2026) — spatio-temporal-decoupled CSI pose, 97.25% PCK@20 / 99.48% PCK@50 / 0.007 m MPJPE, 2.23M parameters (~2.2 MB int8). Code, pretrained weights, and a 360k-sample CSI-pose dataset are public under Apache-2.0 ([repo](https://github.com/DY2434/WiFlow-WiFi-Pose-Estimation-with-Spatio-Temporal-Decoupling), Kaggle dataset). *Evidence: artifact availability MEASURED (verified by direct repo inspection); PCK numbers CLAIMED (5-subject, in-domain, self-collected dataset; hardware unspecified; 15 keypoints vs our 17).* ⚠️ **Name collision:** this is unrelated to RuView's internal WiFlow model. In all RuView docs the external model is referred to as **WiFlow-STD (DY2434)**.

**(F3) For CSI foundation encoders, data scale — not model capacity — is the bottleneck, and the tokenization recipe is now known.**
UNSW's MAE pretraining study (arXiv [2511.18792](https://arxiv.org/abs/2511.18792), Nov 2025) — the largest heterogeneous CSI pretraining run to date (1,320,892 samples, 14 public datasets incl. MM-Fi, Widar 3.0, Person-in-WiFi 3D; 4 devices; 2.4/5/6 GHz; 20–160 MHz) — reports zero-shot cross-domain gains of 2.2–15.7% over supervised baselines, with unseen-domain performance scaling **log-linearly with pretraining data, unsaturated at 1.3M samples**, while ViT-Base adds only 0.4–0.9% over ViT-Small. Optimal recipe: **80% masking ratio, small (30,3) patches** (+4.7% over (40,5) by preserving fine temporal dynamics). *Evidence: MEASURED within-study (ablations verified in body text) but preprint; downstream tasks are classification, NOT pose — pose transfer is a hypothesis. Independently corroborates ADR-150's finding that capacity hurts cross-subject.*

**(F4) Hardware/standards: 802.11bf is finished; Espressif ships official sensing; Wi-Fi 6 AP CSI is reachable.**
- **IEEE 802.11bf-2025** published **2025-09-26** (verified against the IEEE SA record) — sensing standardization is complete for both sub-7 GHz and >45 GHz, with formal sensing setup/feedback procedures. No ESP32 silicon implements it yet. *Evidence: MEASURED (standards-body record).*
- **Espressif `esp_wifi_sensing`** (Apache-2.0, v0.1.x, ESP Component Registry): official CSI presence/motion FSM; esp-csi actively maintained (commit 2026-04-22, verified), CSI confirmed across ESP32/S2/C3/S3/C5/C6/C61. *Evidence: MEASURED (vendor pages + commit log).* ⚠️ A stronger "drop-in compatible with RuView nodes" claim was **REFUTED 0-3** — WiFi-6 parts use a different CSI acquisition config struct.
- **ZTECSITool** (arXiv [2506.16957](https://arxiv.org/abs/2506.16957), [code](https://github.com/WiFiZTE2025/ZTE_WiFi_Sensing)): CSI from commercial Wi-Fi 6 APs at up to 160 MHz / 512 subcarriers (~5–10× ESP32 subcarrier count; the gain is aperture, not per-Hz granularity). Firmware is gated behind a ZTE serial-number approval. *Evidence: capability CLAIMED by the vendor-authored tool paper; code artifact MEASURED.*

**(F5) Nothing in 2025–2026 does full DensePose UV regression from commodity WiFi.** Keypoint pose remains the field's frontier. Three "wireless foundation model" papers were screened out by full-text inspection (HeterCSI = simulated cellular channels only; the NeurIPS-2025 FMCW pilot = mmWave radar, presence-only; arXiv 2509.15258 = survey, no artifacts). *Evidence: MEASURED (absence verified by full-text inspection of the candidates that surfaced; absence of evidence across the whole literature is necessarily weaker).*

### 1.2 What this means for the ADR-151 calibration system

ADR-151's enrollment protocol captures guided human anchors but does **not** record or condition on transceiver geometry. F1 says that omission is precisely the thing that makes camera-supervised (and, plausibly, anchor-supervised) heads layout-brittle. ADR-151's per-room thesis ("teach the room before you teach the model") is *strengthened* by F1 — PerceptAlign is independent evidence that layout must be modeled explicitly — and the fix composes naturally with our Stage-2 enrollment.

ADR-150's masked-CSI-encoder design is *validated* by F3, which also hands us the hyperparameters and the priority call: **collect/aggregate more heterogeneous CSI before scaling the encoder.**

## 2. Decision

Adopt four changes, ordered by effort-vs-gain:

### 2.1 Geometry-condition the calibration system (extends ADR-151 Stage 2) — ACCEPTED

1. **Record transceiver geometry at enrollment.** `EnrollmentProtocol` gains an optional `NodeGeometry` record per node (position estimate, antenna orientation, inter-node distances where known). Stored alongside the room baseline in the bank; schema-versioned so existing banks remain readable.
2. **Fuse geometry embeddings into specialist training.** Where a specialist head consumes the (future, ADR-150) backbone embedding, concatenate a small learned embedding of `NodeGeometry` — the PerceptAlign mechanism, transplanted to our per-room banks. Statistical specialists (current) ignore it; LoRA heads (ADR-151 P6) consume it.
3. **Adopt the two-checkerboard alignment for the camera-supervised path (ADR-079).** When MediaPipe supervision is used, calibrate camera↔WiFi into one shared 3D frame before regression (<5 min, two checkerboards, a few photos). This is the direct defense against F1 for our camera-supervised pipeline. ~~92.9%-PCK@20~~ — *that figure was retracted during measurement (b) (2026-06-10): the surviving holdout shows a constant-output model under an absolute (non-torso) threshold on 69 near-static frames; mean predictor scores 100% under the same protocol. The §2.2 no-citation rule now applies to it.*
4. **Evaluate on the PerceptAlign cross-domain dataset** (21 subjects / 7 layouts) as the MERIDIAN cross-layout benchmark — *gated on confirming its license and downloadability* (open question; repo per paper: github.com/Trymore-lab/PerceptAlign).
   > **Gate resolved (2026-06-10, MEASURED by repo inspection):** repo exists, **MIT license**, dataset downloadable from HuggingFace (5 per-scene repos, raw CSI + separate vision keypoints; Intel 5300, 1TX×3RX×3 ant, 57 subcarriers — same order as ESP32 subcarrier counts; Scene3 ships 3 distinct layouts). Code present, no pretrained weights. Benchmark adoption unblocked; dataset-side license terms inherit HF dataset terms (not separately stated — check at download time).

### 2.2 Benchmark against WiFlow-STD (DY2434) — ACCEPTED

Pull the Apache-2.0 weights + 360k-sample dataset; run three measurements: (a) their model on their data (reproduce 97.25% claim), (b) their model fine-tuned on our ESP32 17-keypoint eval set, (c) our internal WiFlow on their dataset (15-keypoint subset mapping). Until (a)–(c) are measured, **no RuView doc may cite 97.25% as a comparable number** — different dataset, subjects, keypoints.

> **Status (2026-06-10, measurement (a) complete — `benchmarks/wiflow-std/RESULTS.md`):** shipped checkpoint REFUTED (0.08% PCK@20 — wrong keypoint normalization, predates published code); released code does not run as published (6 defects, incl. broken package import and an unreachable test phase); released dataset's last 13 files are corrupted (9,072 windows: NaN + float32-max garbage, diverges fp16 training via BatchNorm poisoning). After repairing both, retraining with upstream defaults reproduced **96.09% PCK@20 full-test / 96.61% corruption-free / MPJPE 0.0094–0.0098** (published: 97.25% / 0.007) on an RTX 5080. Accuracy claims graded MEASURED-EQUIVALENT; params (2.23M) and FLOPs (~0.055G) verified. (b)/(c) remain open.

### 2.3 Apply the UNSW recipe to the ADR-150 encoder — ACCEPTED (amends ADR-150 §2.3)

- Pretraining corpus: start from the same 14 public datasets (1.3M samples) + our home/MM-Fi frames; data aggregation takes priority over architecture work.
- Tokenization: 80% masking, (30,3)-class small patches; encoder stays ViT-Small-class (~15M params) — F3 and our own DANN/transformer results agree that capacity does not pay.
- The published log-linear scaling (unsaturated) sets the expectation: more heterogeneous CSI in, better zero-shot out.

### 2.4 Hardware watch items — ACCEPTED (no code now)

- **802.11bf**: track silicon/certification; OTA binding remains deferred until commodity chipsets expose standardized sensing measurements. **Amended by ADR-153** (2026-06-10): implement a pure Rust forward-compatibility protocol layer now — typed procedure models, a deterministic session FSM, a transport abstraction, simulation tests, and an `OpportunisticCsiBridge` that maps today's ESP32 CSI batches into standardized sensing-report shape.
- **esp_wifi_sensing**: benchmark our presence pipeline against the vendor FSM (one afternoon; useful external baseline). Do **not** treat as drop-in (refuted claim).
- **ZTECSITool AP**: optional high-resolution anchor node for the ADR-029 multistatic mesh — procurement-gated; only pursue if a 160 MHz anchor materially helps tomography.

### 2.5 Explicitly NOT adopted

- No pivot toward "wireless foundation model" papers that don't ship WiFi-CSI artifacts (HeterCSI, FMCW pilot, surveys).
- No DensePose-UV work item: the field has not demonstrated UV regression from commodity WiFi; keypoints remain our supervised target (F5).

### 2.6 RuVector vendor sync + integration opportunities (added 2026-06-10)

**Vendor sync record.** `vendor/ruvector` moved from pin `e38347601` (2026-05-07) to `a083bd77f` (origin/main, 3 commits past tag `ruvector-v0.2.28`; vendored workspace version 2.2.3). 111 commits in the range, roughly half NAPI-binary/lint chores. Substantive: graph condensation + differentiable min-cut (#547), core HNSW correctness fixes v2.2.3 (#502), RUSTSEC/clippy hardening (#504), ONNX embedder API-contract fix (#523/#525 — npm/TypeScript package only), dead parallel-worker import removal (#532). *Evidence: MEASURED (git range + commit-stat inspection).*

**Opportunity table.** Workspace policy is crates.io versions only, so unpublished crates are WATCH by definition regardless of fit.

| Crate | What it offers | wifi-densepose target | crates.io | Verdict |
|---|---|---|---|---|
| `ruvector-graph-condense` (new, #547) | Training-free min-cut graph condensation + **differentiable normalized-cut loss** (`DiffCutCondenser`, analytic MinCutPool-style gradients, gradient-checked tests; provenance-retaining super-nodes) | `subcarrier_selection.rs` (condense 114 subcarriers into cut-preserving regions instead of raw min-cut); auxiliary clustering regularizer for `wifi-densepose-train`; `DynamicPersonMatcher` region structure | **Not published** | **WATCH** — strongest technical fit in the sync; adopt when published. README's "no published method uses graph-cut condensation" is CLAIMED; the diffcut implementation + tests are MEASURED |
| `ruvector-attention` 2.1.0 | #304 SOTA modules: MLA, KV-cache, SSM, sparse/MoE, hybrid search, Graph RAG (publish date 2026-03-27 matches the #304 commit — MEASURED) | Supersedes pinned 2.0.4 used by `model.rs` spatial attention + `bvp.rs`; SSM/MLA are candidate pure-Rust edge-inference primitives for the ADR-150 encoder | 2.1.0 (pinned **2.0.4**) | **ADOPT** (minor bump; API-compat check first) |
| `ruvector-gnn` 2.2.0 | panic→`Result` constructors, gradient clipping, MSE/CE/BCE losses, seeded-RNG layer init (#495 is post-2.2.0) | `wifi-densepose-train` GNN path (pinned 2.0.5, `default-features = false`) | 2.2.0 (pinned **2.0.5**) | **ADOPT** (bump) |
| `ruvector-mincut` / `ruvector-solver` 2.0.6 | Patch-level fixes (workspace republish 2026-03-25) | `metrics.rs` DynamicPersonMatcher, subcarrier interpolation, triangulation | 2.0.6 (pinned **2.0.4** each) | **ADOPT** (routine patch bump) |
| `ruvector-core` 2.2.3 (vendor) | HNSW correctness: k=0 guard, sorted results, flat-index fixes, cross-integration helpers (#502 — MEASURED, `index/hnsw.rs` + new integration tests) | `homecore-recorder` `RuvectorSemanticIndex` (real HNSW consumer); `sketch.rs` quantization unaffected | **2.2.0 = latest published**; 2.2.3 unpublished | **WATCH** — bump the moment 2.2.3 publishes |
| `ruvector-cnn` 2.0.6 | Pure-Rust SIMD conv kernels (AVX2/NEON/WASM), MobileNetV3, INT8 quantization, contrastive losses (InfoNCE/triplet, #252) | **Not** the WiFlow-STD training port — `wiflow_std/model.rs` is tch/libtorch (MEASURED). Relevant to the *edge inference* path of the trained ~2.2 MB int8 model, and InfoNCE/triplet overlaps AETHER (ADR-024) | 2.0.6 | **EVALUATE** — only if/when we commit to a no-libtorch edge runtime for WiFlow-STD-class models |
| `ruvector-acorn` (new-ish) | ACORN predicate-agnostic filtered HNSW (SIGMOD'24 algorithm; γ·M denser graphs for low-selectivity filters) | Metadata-filtered pattern search over ADR-151 calibration banks — speculative; bank sizes are far below where filtered-ANN recall collapse matters | **Not published** | **WATCH** |
| `ruvector-cluster` 2.0.6 | Distributed sharding, gossip discovery, DAG consensus | No current need; ADR-029 mesh coordination is ESP32-side, not vector-DB-side | 2.0.6 | **WATCH** |
| ONNX embedder fix (#523/#525) | API-contract + packaging fixes in `npm/packages/ruvector` (TypeScript) | None — `wifi-densepose-nn`'s ONNX backend is Rust (ort/tract), untouched by this change (MEASURED: commit touches npm/ only) | n/a | No action |
| `ruvector-perception` (new, #547) | "Physical perception substrate" (hypothesis/topology/witness modules) — agent-perception oriented, not RF | None identified | Not published | WATCH (name-overlap only) |

**Security note (RUSTSEC #504).** The substantive fixes target `ruvllm`, `ruvector-dag`, `prime-radiant`, `rvagent-*`, and the `ruvector-server` HTTP endpoint (NaN-safe `partial_cmp`, input-validation guards, env-allowlisted exec) — **none of which we pin**. The commit states `cargo audit` returns clean across the workspace. *Evidence: MEASURED (commit message + file list). Conclusion: no pinned version has an outstanding advisory; no urgent bump required.* The NaN-sort hardening is panic-robustness hygiene our pinned 2.0.4-era crates predate, which is one more reason for the routine bumps below.

**Version-bump recommendations (follow-up PR — no Cargo.toml change in this ADR):** `ruvector-mincut` 2.0.4→2.0.6, `ruvector-solver` 2.0.4→2.0.6, `ruvector-attention` 2.0.4→2.1.0, `ruvector-gnn` 2.0.5→2.2.0. Current: `ruvector-core` 2.2.0, `ruvector-attn-mincut` 2.0.4, `ruvector-temporal-tensor` 2.0.6, `ruvector-crv` 0.1.1 — all at latest published. Nothing in the sync changes §2.1.2 geometry conditioning (our `viewpoint/attention.rs` `GeometricBias` already implements the fusion mechanism) or the ADR-150 MAE recipe (training stays in tch).

## 3. Consequences

**Positive:** the calibration system gains the one mechanism (geometry conditioning) the 2026 literature identifies as the difference between layout-brittle and layout-robust supervised WiFi pose; ADR-150 gets a measured training recipe instead of a guessed one; we acquire two external benchmarks (WiFlow-STD, PerceptAlign dataset) to keep our claims honest.

**Negative / risks:** geometry records add schema surface to banks (mitigated: optional + versioned); every adopted number is preprint-grade until our own benchmark runs land (mitigated by §2.2's no-citation rule); PerceptAlign dataset license is unconfirmed (gated); name collision risk in docs (mitigated: "WiFlow-STD (DY2434)" naming rule).

**Re-check by 2026-12:** 802.11bf silicon, esp_wifi_sensing maturity (v0.1.x today), and the preprint field (newest source Apr 2026).

## 4. Open questions (carried from the research run)

1. Does WiFlow-STD retain accuracy when fine-tuned on ESP32-S3/C6 CSI (fewer subcarriers, lower SNR), scored on our 17-keypoint set? (§2.2 answers this.)
   > **Partial answer (MEASURED 2026-06-11, measurement (b) on 2,046 single-room windows — `benchmarks/wiflow-std/RESULTS.md`):** pretrained init shows strong *optimization* transfer (65% PCK@20 vs scratch's 0% collapse under the same budget) but **no feature transfer** (frozen-trunk + linear adapter ≈ 0%). And no run beat the mean-pose baseline (95.9% PCK@20 — single subject, near-static normalized coords), so no CSI→pose capability is citable from this data. A definitive answer needs multi-subject/multi-position data where the mean pose is weak.
2. Is the PerceptAlign dataset downloadable under a usable license, and does the two-checkerboard procedure work with ESP32 transceiver geometry? (§2.1.4 gate.)
3. Will esp_wifi_sensing evolve toward 802.11bf compliance, replacing opportunistic CSI extraction?

## 5. Source register (evidence-graded)

| Source | Type | Used for | Grade |
|---|---|---|---|
| arXiv 2601.12252 (PerceptAlign, MobiCom'26) | preprint+acceptance | F1, §2.1 | CLAIMED numbers; failure mode corroborated |
| arXiv 2602.08661 + DY2434 repo (WiFlow-STD) | preprint + code | F2, §2.2 | numbers CLAIMED; artifacts MEASURED |
| arXiv 2511.18792 (UNSW MAE) | preprint | F3, §2.3 | ablations MEASURED in-study; pose transfer hypothesis |
| IEEE SA 802.11bf-2025 record | standards body | F4, §2.4 | MEASURED |
| Espressif component registry + esp-csi repo | vendor | F4, §2.4 | MEASURED; "drop-in" REFUTED 0-3 |
| arXiv 2506.16957 + ZTE repo (ZTECSITool) | vendor preprint + code | F4, §2.4 | capability CLAIMED; code MEASURED |
| arXiv 2601.18200 (HeterCSI), OpenReview LMufK3vzE5 (FMCW pilot), arXiv 2509.15258 (survey) | preprints | F5, §2.5 (screened out) | MEASURED (full-text inspection) |
