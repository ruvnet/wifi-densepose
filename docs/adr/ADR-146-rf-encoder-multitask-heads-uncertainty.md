# ADR-146: RF Encoder Multi-Task Heads and Uncertainty Quantification

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-nn` (encoder/model), `wifi-densepose-train` (`ContrastiveBatcher`); AETHER (ADR-024) / MERIDIAN (ADR-027) context |
| **Relates to** | ADR-136 (RuView Streaming Engine, Frame Contracts, QualityScored), ADR-140 (Semantic State Record Schema & Agent Bridge), ADR-145 (Ablation Evaluation Harness), ADR-024 (AETHER Contrastive CSI Embedding), ADR-027 (MERIDIAN Cross-Environment Generalization), ADR-023 (Trained DensePose Model + RuVector Pipeline) |

---

## 1. Context

### 1.1 The Gap

The current Rust stack already owns a shared RF encoder backbone and one contrastive projection head, but it lacks the multi-head fan-out, the per-head uncertainty, and the formalized batcher that ADR-140's `SemanticStateRecord` and ADR-136's `QualityScored` trait will require as upstream producers. Three concrete observations from the real codebase establish the gap.

**A single backbone exists, but it feeds only two task heads.** `v2/crates/wifi-densepose-train/src/model.rs` defines `WiFiDensePoseModel` with a shared `translator` → `backbone` path producing `ModelOutput.features`, consumed by exactly two heads:

```rust
// v2/crates/wifi-densepose-train/src/model.rs
pub struct WiFiDensePoseModel {
    // translator → backbone (shared) ...
    kp_head: KeypointHead,   // line 70
    dp_head: DensePoseHead,  // line 71
}
```

`forward_impl()` (model.rs ~line 193) emits `ModelOutput { keypoints, part_logits, uv_coords, features }`. The `features` tensor is the shared representation, but it is consumed only by pose-regression heads. There is no presence head, count head, activity head, vitals head, gait head, or an exported identity-embedding head wired off the same backbone. Presence, count, activity, vitals, and gait are computed today by *separate* signal-processing modules (`ruvsense/`, `wifi-densepose-vitals`) that do not share the encoder representation, so they cannot benefit from contrastive pretraining (ADR-024) or cross-environment LoRA (ADR-027).

**A projection head and contrastive loss exist, but in the serving crate, not as a formal head taxonomy.** `v2/crates/wifi-densepose-sensing-server/src/embedding.rs` already implements:
- `ProjectionHead` (2-layer MLP, `d_model=64 → d_proj=128`, ReLU + L2-norm), with optional rank-4 LoRA adapters (`lora_1`, `lora_2`) for environment-specific fine-tuning (ADR-027) — all pure-Rust `Vec<f32>` (`forward(&self, x: &[f32]) -> Vec<f32>`, embedding.rs line 131).
- `info_nce_loss()` (embedding.rs line 476) and `CsiAugmenter::augment_pair()` (line 362).
- `EmbeddingExtractor`, the full backbone + projection pipeline.

This is the *seventh* head (identity-embedding) of the proposed taxonomy, already materialized — but as a one-off in the serving crate rather than as one branch among seven over a shared encoder. It is the proof that the pure-Rust `f32` ABI is viable; it is not yet a general multi-task head abstraction.

**Contrastive pair construction exists, but ad hoc.** `v2/crates/wifi-densepose-train/src/rapid_adapt.rs` (MERIDIAN Phase 5) defines `AdaptationLoss::ContrastiveTTT` whose doc-comment is literally *"positive = temporally adjacent, negative = random"* (rapid_adapt.rs line 9), and `contrastive_step()` (line 201) implements it. But this lives inside test-time adaptation, sampling within a single CSI stream. There is **no `ContrastiveBatcher`** anywhere in the workspace (`grep -rn "ContrastiveBatcher" v2/crates` returns nothing). The cross-environment positive/negative pair construction that ADR-027 §6.x requires — same activity / same person across *different rooms* as positives, different semantics as negatives — has no formal sampling contract. Batched iteration is provided generically by `DataLoader`/`DataLoaderIter` over `CsiSample` (`dataset.rs` line 150), with no notion of anchor/positive/negative tuples.

**Consequence.** ADR-140's `SemanticStateRecord` is meant to carry a `model_version` and trace every semantic field to its evidence. ADR-136's `QualityScored` trait is meant to attach confidence bounds to every stage output. Today, the only encoder-derived quantity that could populate such a record is pose; presence/count/activity/vitals/gait arrive from non-encoder modules with their own (incomparable) confidence conventions, and none of them emit calibrated uncertainty. This ADR closes that gap: a single shared RF encoder with seven typed heads, each emitting a `QualityScored` output with per-head uncertainty, trained with a formalized `ContrastiveBatcher` and a calibration-robustness loss that ties the encoder to ADR-135's `calibration_id`.

### 1.2 Scope Boundary

This ADR is about **the encoder and its head fan-out**, not about the downstream semantic record (ADR-140) or the streaming frame contracts (ADR-136). It defines:
- The seven-head taxonomy over the shared backbone in `wifi-densepose-nn`.
- The per-head uncertainty quantification layer and its mapping onto `QualityScored`.
- The calibration-robustness loss tying training to `calibration_id`.
- The `ContrastiveBatcher` sampling contract in `wifi-densepose-train`.
- The pure-Rust `f32` tensor ABI for deterministic, witnessable inference.
- The ablation hooks consumed by ADR-145.

It does **not** define the `SemanticStateRecord` wire schema (ADR-140) nor the stage abstraction (ADR-136); it defines the *producer* that feeds them.

### 1.3 Why `wifi-densepose-nn` and Not `wifi-densepose-train`

Training lives in `wifi-densepose-train` (libtorch / `tch`), which is GPU- and `Tensor`-bound. Inference must run on the Pi+Hailo cluster and in WASM. The current encoder *definition* lives in `wifi-densepose-train/src/model.rs` (a libtorch graph), while the *inference* projection head lives in `wifi-densepose-sensing-server/src/embedding.rs` (pure Rust `f32`). This split is the underlying disease: the head taxonomy and the ABI belong in `wifi-densepose-nn` (which already owns `Tensor` = `Array{1..4}D<f32>` in `tensor.rs`, and `densepose.rs`/`inference.rs`), so both the training crate and the serving crate depend on one definition. The new head-trait and uncertainty types are added to `wifi-densepose-nn`; `wifi-densepose-train` adds only the `ContrastiveBatcher` and the loss terms.

### 1.4 Pipeline Position

```
 CSI window (amplitude, phase)
   → [wifi-densepose-signal preprocessing + ADR-135 baseline subtract]
   → RfEncoder::encode()            (shared backbone → embedding z ∈ R^d_model)   [wifi-densepose-nn, NEW trait]
        ├── PoseHead          ─┐
        ├── PresenceHead       │
        ├── CountHead          │   each head: forward → (value, UncertaintyHead → bounds)
        ├── ActivityHead       ├─ → MultiTaskOutput { per-head QualityScored }     [NEW]
        ├── VitalsHead         │
        ├── GaitHead           │
        └── IdentityEmbedHead ─┘   (the existing ADR-024 ProjectionHead, relocated)
   → SemanticStateRecord assembly  (ADR-140; stamps model_version + calibration_id) 
   → Fusion engine quality scoring (ADR-136 QualityScored)
```

During training, the shared backbone receives gradients from all *enabled* heads (ADR-145 ablation matrix can disable any head), plus the contrastive term over `ContrastiveBatcher` tuples, plus the calibration-robustness term over `calibration_id` groups.

---

## 2. Decision

### 2.1 Seven Task-Specific Head Branches Over the Shared Encoder

Add a `RfEncoder` abstraction in `wifi-densepose-nn` that owns the shared backbone and produces a single embedding `z ∈ ℝ^{d_model}` (default `d_model = 64`, matching `embedding.rs` and `model.rs` today). Seven heads consume `z`. Each head is independently constructible, toggleable, and emits a `QualityScored` output.

| # | Head | Output value | Output type | Existing seed in repo |
|---|------|--------------|-------------|------------------------|
| 1 | `PoseHead` | 17 keypoints + DensePose UV | `PoseEstimate` | `KeypointHead`/`DensePoseHead` (model.rs) |
| 2 | `PresenceHead` | occupancy probability | `f32 ∈ [0,1]` | `ruvsense/coherence_gate.rs` (non-encoder) |
| 3 | `CountHead` | person count | `u8` (argmax over softmax) | none |
| 4 | `ActivityHead` | activity class | `ActivityClass` | `ruvsense/gesture.rs` (non-encoder) |
| 5 | `VitalsHead` | breathing/HR rate | `Vitals { br_hz, hr_hz }` | `wifi-densepose-vitals` (non-encoder) |
| 6 | `GaitHead` | gait signature | `GaitFeatures` | `ruvsense/longitudinal.rs` (non-encoder) |
| 7 | `IdentityEmbedHead` | 128-d unit embedding | `Embedding128` | `ProjectionHead` (embedding.rs) — **relocated** |

Head #7 is the existing ADR-024 `ProjectionHead`, moved from `wifi-densepose-sensing-server` into `wifi-densepose-nn` and re-exported (the serving crate re-imports it; no behavior change, identical Xavier seeds 2024/2025 preserved for determinism and existing RVF compatibility). Heads #2, #4, #5, #6 supersede the standalone signal modules *as encoder-derived alternatives*; the signal modules remain for the no-model fallback path and as ablation baselines (ADR-145).

```rust
// v2/crates/wifi-densepose-nn/src/encoder/mod.rs  (NEW module)
use crate::tensor::Tensor;        // Array{1..4}D<f32> — pure Rust, no libtorch at inference

/// Shared RF encoder backbone. Produces a fixed-width embedding from a CSI window.
pub trait RfEncoder {
    /// Encode a preprocessed CSI window into the shared embedding `z`.
    /// Input is amplitude+phase already baseline-subtracted (ADR-135).
    fn encode(&self, window: &EncoderInput) -> Embedding;
    /// Embedding width (`d_model`). Default deployment: 64.
    fn d_model(&self) -> usize;
    /// Identifier of the weights producing this embedding — flows into
    /// ADR-140 `SemanticStateRecord.model_version`.
    fn model_version(&self) -> &ModelVersion;
}

/// Owned set of task heads sharing one encoder.
pub struct MultiTaskRfModel<E: RfEncoder> {
    encoder: E,
    pose: Option<PoseHead>,
    presence: Option<PresenceHead>,
    count: Option<CountHead>,
    activity: Option<ActivityHead>,
    vitals: Option<VitalsHead>,
    gait: Option<GaitHead>,
    identity: Option<IdentityEmbedHead>,
    enabled: HeadMask,   // ablation control (§2.5)
}

/// One unified inference call. Only enabled heads are evaluated.
pub struct MultiTaskOutput {
    pub embedding: Embedding,
    pub pose:     Option<QualityScored<PoseEstimate>>,
    pub presence: Option<QualityScored<f32>>,
    pub count:    Option<QualityScored<u8>>,
    pub activity: Option<QualityScored<ActivityClass>>,
    pub vitals:   Option<QualityScored<Vitals>>,
    pub gait:     Option<QualityScored<GaitFeatures>>,
    pub identity: Option<Embedding128>,   // unit vector; quality is uniformity/alignment, not per-frame conf
    pub model_version: ModelVersion,
    pub calibration_id: Option<CalibrationId>,  // ADR-135; None ⇒ uncalibrated mode
}

impl<E: RfEncoder> MultiTaskRfModel<E> {
    pub fn forward(&self, input: &EncoderInput) -> MultiTaskOutput;
}
```

**Interface boundary.** `MultiTaskOutput` is the *only* thing the ADR-140 record assembler reads. Each `QualityScored<T>` carries the value, its uncertainty (§2.2), the `model_version`, and (if present) the `calibration_id` — satisfying the project rule that every semantic state traces to signal evidence + model version + calibration version + privacy decision (the privacy decision is stamped downstream by ADR-141, out of scope here).

### 2.2 Per-Head Uncertainty Quantification → `QualityScored`

Every head except the embedding head emits a calibrated uncertainty. The method differs by head type but all converge onto the ADR-136 `QualityScored` trait so the fusion engine can compare confidences across heads.

```rust
// re-exported from the ADR-136 contract; shown here for the producer side
pub trait QualityScored {
    fn quality(&self) -> QualityScore;   // ∈ [0,1], calibrated (ECE-checked, §2.6)
    fn evidence(&self) -> &EvidenceRef;   // points at the CSI window + calibration_id
}

pub struct QualityScore {
    pub confidence: f32,     // point confidence ∈ [0,1]
    pub bound: UncertaintyBound,
}

pub enum UncertaintyBound {
    /// Regression heads (vitals, pose coords): predictive ±σ per dimension.
    Gaussian { mean: Vec<f32>, sigma: Vec<f32> },
    /// Classification heads (presence, count, activity): full categorical posterior.
    Categorical { probs: Vec<f32>, entropy: f32 },
    /// Identity/gait: cosine-margin to the next-nearest cluster.
    Margin { top1: f32, margin: f32 },
}
```

Uncertainty mechanism per head:

| Head | UQ mechanism | Why this and not MC-dropout/ensembles |
|------|-------------|----------------------------------------|
| Pose | Per-keypoint predictive variance head (Gaussian NLL, learned σ) | Closed-form, single forward pass — required for 20 Hz real-time and for WASM/Hailo where dropout sampling is impractical |
| Presence | Categorical posterior + entropy | Binary; entropy near `ln 2` ⇒ abstain |
| Count | Categorical (softmax over {0..K_max}) + entropy | Discrete; entropy distinguishes "2 vs 3 people" ambiguity from confident calls |
| Activity | Categorical posterior + entropy | Same as count; entropy is the abstention signal |
| Vitals | Gaussian NLL (learned σ on br_hz, hr_hz) | Physiological rates need a continuous confidence band, not a class label |
| Gait | Cosine margin to enrolled-gait clusters | Gait is an open-set matching problem, like identity |
| Identity | Embedding uniformity/alignment (ADR-024 metrics) | Already defined in AETHER; no per-frame "confidence", quality is index-level |

**Decision: heteroscedastic single-pass UQ, not MC-dropout or deep ensembles.** Justified in §3 Alternatives. The learned-σ head is two extra linear layers per regression head and adds a Gaussian-NLL term to the loss; the categorical heads need no extra parameters (the softmax *is* the posterior). This keeps the pure-Rust `f32` inference path single-pass and deterministic.

**Calibration of the score itself.** `confidence` must be *calibrated* (a 0.8 confidence is right 80% of the time), enforced via post-hoc temperature scaling per head, with Expected Calibration Error (ECE) checked in the acceptance tests (§2.6). The temperature scalars are stored alongside weights and stamped into `model_version`.

### 2.3 Calibration-Robustness Loss Tied to ADR-135 `calibration_id`

The encoder must be **invariant to per-device baseline shifts** so that an embedding for "empty room, device A" and "empty room, device B" land in the same place and a person produces the same activity/pose regardless of which calibrated node observed them. ADR-135 produces a `BaselineCalibration` per device with a stable identity; this ADR introduces `CalibrationId` as a hashable key over `(device_id, tier, captured_at)` and uses it as a **domain label** in a calibration-robustness loss.

```
L_total = Σ_h w_h · L_head_h(enabled)
        + λ_con · L_contrastive          (NT-Xent over ContrastiveBatcher tuples, §2.4)
        + λ_cal · L_calib_robust         (NEW, this section)
        + λ_uq  · L_uncertainty          (Gaussian-NLL terms across regression heads)
```

`L_calib_robust` is a **calibration-adversarial / variance-penalty** term. Two equivalent formulations are supported (config-selectable):

1. **Group-variance penalty (default).** For a mini-batch, group embeddings by `calibration_id`. Penalize the *between-group* variance of the embedding conditioned on the *same* semantic label (same activity/presence), pulling cross-device representations of the same event together:
   `L_calib_robust = mean_over_labels( Var_{calib_id}( z | label ) )`.
2. **Gradient-reversal domain classifier (DANN-style).** A small `calibration_id` classifier behind a gradient-reversal layer; the encoder learns features the classifier *cannot* use to recover which calibrated device produced them.

The default is the group-variance penalty: it has no adversarial training instability, it requires `≥2` distinct `calibration_id`s per mini-batch (enforced by `ContrastiveBatcher`, §2.4), and it directly operationalizes "invariant to per-device baseline shift." When `calibration_id` is `None` (uncalibrated capture), the sample is excluded from `L_calib_robust` but still contributes to head losses.

**Interface boundary.** The training loop reads `CalibrationId` from each `CsiSample` (a new optional field populated from the capture's ADR-135 baseline). Inference stamps the *active* `calibration_id` into `MultiTaskOutput` so the semantic record traces to the calibration version — satisfying the project provenance rule.

### 2.4 The `ContrastiveBatcher` Sampling Contract

Formalize the ad-hoc rapid_adapt pairing (`positive = temporally adjacent, negative = random`) into a first-class, cross-environment sampler in `wifi-densepose-train`. It produces **anchor / positive / negative** tuples obeying ADR-027's cross-environment generalization requirement.

```rust
// v2/crates/wifi-densepose-train/src/dataset.rs  (NEW; alongside DataLoader)
pub struct ContrastiveBatcher<'a> {
    dataset: &'a dyn CsiDataset,
    batch_size: usize,
    strategy: PairStrategy,
    /// Minimum distinct calibration_ids per batch (≥2 to make L_calib_robust well-posed).
    min_calib_ids: usize,
    seed: u64,
}

pub enum PairStrategy {
    /// ADR-024 default: positive = augmented view of same window (CsiAugmenter),
    /// negative = other windows in the batch.
    SelfSupervised,
    /// ADR-027: positive = SAME semantic label in a DIFFERENT environment
    /// (different calibration_id / room); negative = different label.
    /// This is the contract that forces cross-environment invariance.
    CrossEnvironment { label_key: LabelKey },
    /// rapid_adapt parity: positive = temporally adjacent, negative = random.
    Temporal { window: usize },
}

pub struct ContrastiveBatch {
    pub anchors:   Vec<CsiSample>,
    pub positives: Vec<CsiSample>,   // aligned 1:1 with anchors
    pub negatives: Vec<Vec<CsiSample>>,  // per-anchor negative set (in-batch or sampled)
    pub calib_ids: Vec<Option<CalibrationId>>,  // aligned with anchors; ≥ min_calib_ids distinct
}

impl<'a> ContrastiveBatcher<'a> {
    pub fn new(dataset: &'a dyn CsiDataset, batch_size: usize,
               strategy: PairStrategy, seed: u64) -> Self;
    /// Deterministic given (seed, epoch). Reuses DataLoader's xorshift shuffle.
    pub fn iter(&self, epoch: u64) -> impl Iterator<Item = ContrastiveBatch> + '_;
}
```

**Contract guarantees** (tested in §2.6):
1. **Determinism**: `(seed, epoch)` fully determines the batch sequence — same xorshift RNG already used by `DataLoader`.
2. **Positive validity**: under `CrossEnvironment`, `positive.label == anchor.label` AND `positive.calibration_id != anchor.calibration_id` (when ≥2 environments exist; otherwise it degrades gracefully to `SelfSupervised` with a warning).
3. **Negative validity**: every negative differs from the anchor in the semantic label dimension being contrasted.
4. **Calibration coverage**: each batch contains ≥ `min_calib_ids` distinct `calibration_id`s so `L_calib_robust` (§2.3) is computable; if the dataset has fewer, the batcher errors at construction (fail fast, not silent degradation).

The existing `CsiAugmenter::augment_pair()` (embedding.rs line 362) provides the augmentation for `SelfSupervised`/`CrossEnvironment` positive views and is re-exported from `wifi-densepose-nn`. `info_nce_loss()` (embedding.rs line 476) consumes the batch unchanged.

### 2.5 Pure-Rust `f32` Tensor ABI for Deterministic, Witnessable Inference

**Decision: the inference ABI for the encoder and all heads is pure-Rust `f32` (`ndarray`), identical to the existing `wifi-densepose-nn::tensor::Tensor` enum (`Float1D..Float4D`, `tensor.rs`) and the `ProjectionHead::forward(&[f32]) -> Vec<f32>` convention already in `embedding.rs`.** No libtorch at inference time.

Rationale:
- **Witnessability (ADR-028).** A pure-`f32` forward pass with a fixed evaluation order is bit-reproducible. The same SHA-256 proof discipline applied to ADR-134/135 (`verify.py` + `expected_features.sha256`) extends to the multi-task forward: feed a fixed CSI window, hash `MultiTaskOutput` floats, assert stable. libtorch reductions are not bit-stable across builds/devices and cannot anchor a witness hash.
- **Deployment.** Hailo/WASM targets do not link libtorch. The serving path (`embedding.rs`) already proves pure-Rust inference works; this generalizes it to all seven heads.
- **Training/inference split.** Training stays in `wifi-densepose-train` (libtorch `tch`). A weight-export step converts trained head/encoder weights into the flat `Vec<f32>` layout already used by `ProjectionHead::flatten_into`/`unflatten_from` (embedding.rs lines 159/165). Each head defines `flatten_into`/`unflatten_from` for round-trip stability (the same pattern as the existing projection head and its LoRA `flatten_lora`/`unflatten_lora`).

**ABI specification (per head, little-endian f32, row-major):**
```
[u32 magic 0x52464548 "RFEH"][u16 schema=1][u16 d_model][u8 n_heads][u8 head_mask]
[ModelVersion: 32-byte content hash of all weights]
[per enabled head: u16 head_id, u32 param_len, f32 × param_len]
```
`ModelVersion` is the 32-byte hash that flows into `SemanticStateRecord.model_version` (ADR-140) — making the weights self-identifying so a record can never claim a model version it did not run.

### 2.6 Ablation Hooks for ADR-145

Each head is individually toggleable at *both* train and inference time via `HeadMask`, exactly the toggle ADR-145's ablation matrix needs.

```rust
pub struct HeadMask(u8);  // bit per head; bit 0 = pose ... bit 6 = identity
impl HeadMask {
    pub const ALL: HeadMask;
    pub fn with(self, h: HeadKind) -> Self;
    pub fn without(self, h: HeadKind) -> Self;
    pub fn is_enabled(&self, h: HeadKind) -> bool;
}
```

- **Inference**: a disabled head is not evaluated and its `MultiTaskOutput` field is `None` (zero CPU cost — this is what ADR-145 measures for latency-vs-head-count).
- **Training**: a disabled head contributes no loss term and no gradient (its `w_h = 0`), so the ablation harness can measure each head's *contribution to the shared backbone* and detect negative transfer between heads.
- **Privacy-leakage probe (ADR-145)**: the `IdentityEmbedHead` and `GaitHead` can be disabled to produce a privacy-reduced model; the harness measures how much identity information remains recoverable from the *remaining* heads' embedding `z`. The encoder exposes `z` directly so ADR-145 can run a linear-probe leakage test without re-running heads.

`MultiTaskRfModel::with_mask(mask)` returns a view enabling exactly the named heads; the ablation harness iterates the `2^7` (or a curated subset of) masks.

### 2.7 Proof / Witness

Per ADR-028, add witness rows to `docs/WITNESS-LOG-028.md`:

| Row | Capability | Evidence | Hash |
|-----|-----------|----------|------|
| W-39 | Multi-task forward determinism (pure-Rust f32, fixed window) | `cargo test -p wifi-densepose-nn encoder::tests::forward_determinism` | SHA-256 of `MultiTaskOutput` floats |
| W-40 | `ContrastiveBatcher` determinism + positive/negative validity | `cargo test -p wifi-densepose-train dataset::tests::contrastive_contract` | SHA-256 of batch index sequence |
| W-41 | Per-head ECE within bound after temperature scaling | `cargo test -p wifi-densepose-nn encoder::tests::ece_calibrated` | recorded ECE values |
| W-42 | Weight ABI round-trip (flatten → unflatten bit-identical) | `cargo test -p wifi-densepose-nn encoder::tests::abi_round_trip` | SHA-256 of serialized weights |

`source-hashes.txt` gains `SHA-256(encoder/mod.rs)` and `SHA-256(dataset.rs ContrastiveBatcher region)`.

---

## 3. Consequences

### 3.1 Positive

- **One representation, seven tasks.** Presence/count/activity/vitals/gait now benefit from ADR-024 contrastive pretraining and ADR-027 cross-environment LoRA, instead of each signal module learning in isolation. Multi-task co-regularization typically improves data efficiency for the weaker heads (count, gait) by sharing the backbone with the data-rich heads (pose, presence).
- **Comparable, calibrated confidences.** Every head emits `QualityScored` with ECE-checked confidence, so ADR-136's fusion engine can weight pose-confidence against vitals-confidence on a common scale, and ADR-140's record carries calibrated uncertainty per field.
- **Cross-device invariance.** `L_calib_robust` keyed on ADR-135 `calibration_id` means a model trained across the fleet (ESP32-S3, C6, cognitum-seed-1) does not learn device-specific shortcuts; embeddings are comparable across nodes — directly enabling multistatic fusion (ADR-029) on encoder embeddings, not just raw CSI.
- **Witnessable inference.** Pure-Rust `f32` ABI extends the ADR-028 proof chain to the full model and ships to Hailo/WASM without libtorch.
- **Ablation-ready.** ADR-145 gets its head toggle for free; the `z`-exposure enables the privacy-leakage probe without bespoke hooks.

### 3.2 Negative

- **Weight-export step required.** Training (libtorch) and inference (pure-Rust) now have a mandatory, tested conversion. A bug in `flatten/unflatten` silently degrades inference; W-42 guards it.
- **Loss has more knobs.** `w_h` (seven), `λ_con`, `λ_cal`, `λ_uq` — more hyperparameters to tune; negative transfer between heads is possible and must be monitored via the ablation harness.
- **Relocating `ProjectionHead`** from `wifi-densepose-sensing-server` to `wifi-densepose-nn` touches the serving crate's imports and any RVF segment that referenced the old path. Seeds and layout are preserved so existing RVF embedding indices remain valid, but the move is a real refactor.
- **`ContrastiveBatcher` needs multi-environment data.** `CrossEnvironment` strategy is only meaningful with ≥2 calibrated rooms; with one room it degrades to self-supervised. Until multi-room paired capture exists (CLAUDE.local.md: cognitum-seed-1 + the COM9 node are the two provisioned environments), cross-environment training is data-limited.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Heteroscedastic σ collapses to a constant (head ignores input, learns global noise) | Medium | UQ is uninformative; ECE looks fine but bounds are useless | β-NLL / σ-floor regularization; W-41 ECE test plus a per-input σ-variance assertion |
| Negative transfer: adding count/gait heads degrades pose | Medium | Headline pose metric regresses | ADR-145 ablation matrix quantifies each head's effect on every other; gate head inclusion on no-regression |
| `calibration_id` group too small in a batch → `L_calib_robust` noisy | Medium | Cross-device invariance under-trained | `ContrastiveBatcher` enforces `min_calib_ids ≥ 2` at construction (fail fast) |
| Pure-Rust forward diverges from libtorch training graph (op mismatch) | Low-Med | Inference accuracy ≠ training accuracy | Golden-output parity test: same weights, same input, assert pure-Rust output within tolerance of libtorch reference; part of W-39 |
| Identity/gait heads enabled by default leak biometric data | Medium | Privacy regression | Heads default-off behind `HeadMask`; ADR-141 privacy mode must explicitly enable them; ADR-145 leakage probe verifies residual leakage with them off |

---

## 4. Alternatives Considered

### 4.1 Separate Models Per Task (status quo)

Keep pose in the encoder and leave presence/count/activity/vitals/gait as independent signal modules. **Rejected**: no shared representation means no contrastive/cross-environment benefit for the weaker tasks, incomparable confidences (each module invents its own), and every task re-pays the feature-extraction cost. The status quo is precisely the gap §1.1 documents.

### 4.2 MC-Dropout or Deep Ensembles for Uncertainty

Sample N stochastic forward passes (MC-dropout) or average M models (ensemble) for predictive uncertainty. **Rejected for the inference path**: N× or M× compute breaks the 20 Hz real-time budget on Pi/Hailo and is impractical in WASM; ensembles also multiply the weight-export and witness-hash surface by M. Heteroscedastic single-pass UQ gives a calibrated band in one deterministic pass. (Deep ensembles remain available as an *offline* evaluation oracle in the ADR-145 harness, not as the shipped UQ.)

### 4.3 One Multi-Output Head (single MLP emitting everything)

A single wide head producing all task outputs from `z`. **Rejected**: prevents per-head ablation (§2.6) — you cannot disable count without disabling pose — and forces one loss-weighting compromise. Independent heads are the only structure that satisfies ADR-145's toggle requirement and lets each head own its UQ mechanism (§2.2).

### 4.4 Keep the ABI as libtorch Tensors End-to-End

Use `tch::Tensor` for inference too. **Rejected**: not witnessable (non-bit-stable reductions), not deployable to Hailo/WASM, and contradicts the already-shipping pure-Rust `embedding.rs` inference path. The training/inference split with a tested weight-export is the cost of determinism and edge deployment.

### 4.5 Sample Contrastive Pairs Within a Single Stream (rapid_adapt parity only)

Reuse only the `Temporal` strategy from `rapid_adapt.rs`. **Rejected as the default**: temporally adjacent positives teach *temporal* smoothness, not *environment* invariance. ADR-027's whole premise is cross-room generalization, which requires `CrossEnvironment` positives spanning `calibration_id`s. `Temporal` is retained as a strategy variant for test-time adaptation parity, not as the training default.

---

## 5. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-024 (AETHER Contrastive Embedding) | **Extended**: the `ProjectionHead`/`info_nce_loss`/`CsiAugmenter` become head #7 and the `SelfSupervised` strategy; relocated into `wifi-densepose-nn` |
| ADR-027 (MERIDIAN Cross-Environment) | **Operationalized**: `ContrastiveBatcher::CrossEnvironment` + `L_calib_robust` formalize cross-room invariance; `rapid_adapt.rs` LoRA path consumes the same head taxonomy |
| ADR-023 (Trained DensePose + RuVector) | **Built on**: `WiFiDensePoseModel`'s shared backbone and `kp_head`/`dp_head` become the `RfEncoder` + `PoseHead` |
| ADR-135 (Empty-Room Baseline Calibration) | **Consumed**: `CalibrationId` keys `L_calib_robust`; baseline-subtracted frames are the encoder input; `calibration_id` stamped into every output |
| ADR-136 (Streaming Engine / QualityScored) | **Producer for**: each head's `QualityScored` output is what the fusion engine and frame contracts read |
| ADR-140 (Semantic State Record) | **Producer for**: `MultiTaskOutput` populates the record; `model_version` (self-identifying weight hash) and `calibration_id` satisfy the provenance rule |
| ADR-141 (BFLD Privacy Control Plane) | **Gated by**: identity/gait heads default-off; privacy mode decides which heads run; the privacy decision completes the four-part provenance (evidence + model + calibration + privacy) |
| ADR-145 (Ablation Eval Harness) | **Consumer**: `HeadMask` and exposed `z` provide the toggle + leakage-probe surface |
| ADR-028 (ESP32 Capability Audit / Witness) | **Witness extended**: rows W-39…W-42; `encoder/mod.rs` + `ContrastiveBatcher` hashes added to `source-hashes.txt` |

---

## 6. References

### Production Code (verified to exist)

- `v2/crates/wifi-densepose-train/src/model.rs` — `WiFiDensePoseModel`, shared backbone, `ModelOutput.features`, `KeypointHead`/`DensePoseHead` (becomes `RfEncoder` + `PoseHead`)
- `v2/crates/wifi-densepose-sensing-server/src/embedding.rs` — `ProjectionHead`, `EmbeddingExtractor`, `CsiAugmenter::augment_pair`, `info_nce_loss`, LoRA + `flatten/unflatten` (head #7, relocated; pure-Rust f32 ABI proof)
- `v2/crates/wifi-densepose-train/src/rapid_adapt.rs` — `AdaptationLoss::ContrastiveTTT` ("positive = temporally adjacent, negative = random"), `contrastive_step` (formalized into `ContrastiveBatcher::Temporal`)
- `v2/crates/wifi-densepose-train/src/dataset.rs` — `DataLoader`/`DataLoaderIter`, `CsiSample`, `CsiDataset` (new `ContrastiveBatcher` added alongside)
- `v2/crates/wifi-densepose-nn/src/tensor.rs` — `Tensor` enum (`Float1D..FloatND`, pure-Rust `ndarray` f32 ABI)
- `v2/crates/wifi-densepose-nn/src/{densepose.rs,inference.rs,lib.rs}` — inference crate where `encoder/` module is added
- `docs/adr/ADR-024-contrastive-csi-embedding-model.md` — AETHER backbone, projection head, L_AETHER loss
- `docs/adr/ADR-027-cross-environment-domain-generalization.md` — MERIDIAN RapidAdaptation, calibration-frame fine-tuning
- `docs/adr/ADR-135-empty-room-baseline-calibration.md` — `BaselineCalibration`, source of `CalibrationId`

### External Papers

- Kendall, A. & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS*. — Heteroscedastic aleatoric uncertainty via learned σ and Gaussian NLL; basis for the single-pass regression-head UQ in §2.2.
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks." *ICML*. — Temperature scaling and Expected Calibration Error; basis for the per-head score calibration and the W-41 ECE acceptance test.
- Ganin, Y. et al. (2016). "Domain-Adversarial Training of Neural Networks (DANN)." *JMLR*. — Gradient-reversal domain classifier; the alternative `L_calib_robust` formulation in §2.3, with `calibration_id` as the domain label.
- Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)." *ICML*. — NT-Xent / projection-head design reused by ADR-024 and the `ContrastiveBatcher` self-supervised strategy.
- Bardes, A. et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." *ICLR*. — Variance/covariance regularization (the invariance term motivates the group-variance form of `L_calib_robust`).
- IdentiFi (2025) / WhoFi (2025) — WiFi CSI contrastive identity embedding (cited in ADR-024); motivate head #7 and the gait/identity margin-based UQ.


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `f18b096f2`, issue #850): `RfEmbedding` (pure-Rust f32 ABI), the 7 task heads with per-head uncertainty, the calibration-robustness and triplet losses, and the deterministic `ContrastiveBatcher`. 7 tests.

**Integration glue -- not yet on the live path (this is the model-training phase):** training the shared encoder backbone on real data via Burn/Candle/libtorch; populating `FrameMeta.model_id` / `model_version` from a head registry once models are versioned for deployment.

**Trust contribution:** each head reports *how sure it is*, and the encoder is trained to give the same answer across rooms and calibrations -- honesty about confidence plus cross-environment robustness.
