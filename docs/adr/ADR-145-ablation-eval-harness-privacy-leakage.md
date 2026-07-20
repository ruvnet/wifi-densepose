# ADR-145: Ablation Evaluation Harness with Privacy-Leakage and Latency Metrics

| Field | Value |
|-------|-------|
| **Status** | Accepted — partial (built + tested building block; integration glue pending — see Implementation Status, commit `0f336b7d3`) |
| **Date** | 2026-05-28 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-train` (`src/eval.rs`, `src/metrics.rs`, `src/ruview_metrics.rs`, `src/proof.rs`); `wifi-densepose-signal` (`src/bin/*_proof_runner.rs`); `wifi-densepose-cli` |
| **Relates to** | ADR-011 (Deterministic Proof Harness), ADR-014 (SOTA Signal Processing), ADR-027 (Cross-Environment Domain Generalization / MERIDIAN), ADR-031 (RuView Sensing-First RF Mode), ADR-120 (BFLD Privacy Class & Hash Rotation), ADR-136 (RuView Rust Streaming Engine), ADR-141 (BFLD Privacy Control Plane), ADR-144 (UWB Range-Constraint Fusion) |

---

## 1. Context

### 1.1 The Gap

The repository has two independent, well-formed evaluation surfaces that have never been wired together into a single ablation matrix:

1. **`wifi-densepose-train/src/ruview_metrics.rs`** implements the ADR-031 three-metric acceptance test — `evaluate_joint_error()` (PCK@0.2 / OKS / torso jitter / p95 error), `evaluate_tracking()` (MOTA / ID-switches / fragmentation), `evaluate_vital_signs()` (breathing/heartbeat BPM error and SNR) — and rolls them into `RuViewAcceptanceResult` with a `RuViewTier` (`Fail` / `Bronze` / `Silver` / `Gold`) via `determine_tier()`. The threshold structs (`JointErrorThresholds`, `TrackingThresholds`, `VitalSignThresholds`) carry the `Default` impls that encode the deployment gates.

2. **`wifi-densepose-train/src/eval.rs`** implements the ADR-027 MERIDIAN cross-environment evaluator — `CrossDomainEvaluator::evaluate()` returns `CrossDomainMetrics { in_domain_mpjpe, cross_domain_mpjpe, few_shot_mpjpe, cross_hardware_mpjpe, domain_gap_ratio, adaptation_speedup }`. Domain `0` is in-domain; non-zero domain IDs are cross-domain. It reports a single scalar `domain_gap_ratio = cross / in_domain`.

These two surfaces share **no common driver**. There is:

- **No feature-ablation concept anywhere.** A workspace-wide search for `ablation` / `Ablation` across `v2/crates` returns zero matches. There is no struct that says "run the acceptance test with CIR disabled" or "with Doppler enabled," and no way to attribute a tier change to a specific feature branch (CSI-only vs CSI+CIR vs +Doppler).
- **No privacy-leakage metric in the eval path.** Privacy is enforced *structurally* in `wifi-densepose-bfld` — `signature_hasher.rs` implements the ADR-120 BLAKE3-keyed per-site, daily-rotated `rf_signature_hash` (invariant I3), and `embedding.rs` keeps `IdentityEmbedding` in-RAM-only (invariant I1/I2). But there is no *measured* leakage scalar: nothing runs a membership-inference attack against the hash-rotation pipeline and reports a number in `[0, 1]`. The acceptance test cannot fail a model for leaking identity.
- **No latency profile in the acceptance result.** `RuViewAcceptanceResult` reports accuracy and tracking but carries no `p50`/`p95`/`p99` inference-latency fields. The ADR-031 mode says nothing about timing budgets (a grep of `ADR-031` for `latency`/`p95` returns nothing), so a model that passes Gold at 800 ms/frame is indistinguishable from one at 40 ms/frame.
- **No per-variant determinism binding.** The proof harness exists and is mature: `wifi-densepose-train/src/proof.rs` runs `N_PROOF_STEPS = 50` under `PROOF_SEED = 42` / `MODEL_SEED = 0` and SHA-256-hashes the model weights (`hash_model_weights()`), comparing against `expected_proof.sha256`. The signal side mirrors this — `src/bin/calibration_proof_runner.rs` (ADR-135) and `src/bin/cir_proof_runner.rs` (ADR-134) hash deterministic synthetic outputs against `archive/v1/data/proof/expected_calibration_features.sha256` and `expected_cir_features.sha256`. But **no proof artifact pins an ablation report**: there is no `expected_ablation_*.sha256`, so re-running the matrix on a fixed seed could silently produce a different tier and CI would not notice.

The cost of the gap is concrete. When ADR-134 (CIR) and ADR-135 (calibration) landed, the only way to know whether CIR *helped* presence/localization was to read the commit message — there was no harness that ran the acceptance test with and without CIR and emitted a side-by-side delta. As ADR-144 (UWB fusion) and the BFLD privacy modes (ADR-141) come online, the number of feature combinations grows combinatorially, and "does turning on feature X regress tier or leak identity?" becomes unanswerable without a deterministic ablation matrix.

### 1.2 What "Ablation" Means Here

An **ablation** is one acceptance-test run over a fixed evaluation set with a named subset of signal features enabled. The matrix is the set of those runs plus the pairwise deltas between them. Each ablation produces:

- A `RuViewAcceptanceResult` (the existing struct, unchanged) → tier, PCK, OKS, MOTA, breathing error.
- New scalar metrics this ADR adds: presence accuracy, localization error, activity accuracy, FP/FN rates, latency p50/p95/p99, **privacy-leakage score** ∈ `[0, 1]`, and cross-room degradation.
- A determinism record: the SHA-256 of the variant's witness-replay output, which must match the per-variant expected hash or CI fails.

An ablation is **not** a hyperparameter sweep or a training run. It evaluates a *fixed, already-trained* `model.bin` snapshot under different *inference-time feature gates*. Training is out of scope — this ADR consumes the model the way `proof.rs` consumes a fixed-seed model.

### 1.3 Hardware Constraints on the Feature Set

The ablation feature combinations are bounded by what RuView hardware can actually produce, per the project hardware table and ADR-136's streaming engine:

| Tier | Feature | Source | Available today? |
|------|---------|--------|------------------|
| F0 | CSI amplitude/phase | ESP32-S3 (20 MHz, 52 active subcarriers, HT20) | Yes (COM9) |
| F1 | CIR (delay taps) | ADR-134 `CirEstimator` over the same CSI | Yes |
| F2 | Doppler / micro-motion | ADR-014 spectrogram over a frame window | Yes |
| F3 | BFLD beamforming-feedback features | ADR-118/120 `wifi-densepose-bfld` (802.11ac/ax BFI) | Yes (gated) |
| F4 | UWB range constraint | ADR-144 fusion with WorldGraph anchors | **No — hardware not landed** |

The 6-node TDM mesh and the 20 MHz ESP32-S3 bandwidth cap the realistic combinations. UWB (F4) is **deferred**: ADR-144 specifies the fusion contract but the ranging hardware is not in the fleet, so the `+UWB` ablation is a *defined-but-skipped* variant (it appears in the matrix as `Skipped { reason }`, not silently absent — same pattern as the unprovisioned seeds in ADR-135 §2.8).

### 1.4 Pipeline Position

```
model.bin snapshot (fixed)
  + witness-bundle CSI replay (PROOF_SEED=42, fixed salt)
        │
        ▼
  AblationHarness::run_matrix()      ← NEW (wifi-densepose-train)
        │  for each AblationVariant (F-mask):
        │    feature-gate the signal stages (ADR-136 streaming engine)
        │      → eval.rs CrossDomainEvaluator  (cross-room degradation)
        │      → ruview_metrics.rs acceptance   (tier, PCK/OKS/MOTA/vitals)
        │      → SpecMetrics                     (presence/loc/activity/FP-FN)
        │      → LatencyProfile (criterion p50/p95/p99)
        │      → PrivacyLeakage (MIA on ADR-120 hash-rotation pipeline)
        │      → SHA-256(variant canonical bytes) vs expected_ablation_*.sha256
        ▼
  AblationReport  →  markdown auto-report + summary.json
```

The harness sits *above* the streaming engine: it does not re-derive features, it toggles which ADR-136 stages are active and re-reads the existing `eval.rs` / `ruview_metrics.rs` outputs. Determinism is inherited from the proof harness substrate (ADR-011).

---

## 2. Decision

### 2.1 The Six Ablation Variants

We define exactly six feature combinations, of which five run today and one is deferred:

```rust
// New module: wifi-densepose-train/src/ablation.rs

/// One feature combination to evaluate. Bitflags over the signal stages
/// that ADR-136's streaming engine can gate on or off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AblationVariant {
    /// F0 only: raw CSI amplitude + phase. The floor baseline.
    CsiOnly,
    /// F1 only: CIR delay-tap features (ADR-134), CSI not fed to the head.
    CirOnly,
    /// F0 + F1: amplitude/phase plus CIR taps. The current production default.
    CsiPlusCir,
    /// F0 + F1 + F2: adds Doppler / micro-motion spectrogram (ADR-014).
    PlusDoppler,
    /// F0 + F1 + F2 + F3: adds BFLD beamforming-feedback features (ADR-118/120).
    PlusBfld,
    /// F0..F4: adds UWB range constraint (ADR-144). HARDWARE-DEFERRED.
    PlusUwb,
}

impl AblationVariant {
    /// The full deterministic matrix, in canonical (stable) order.
    pub const MATRIX: [AblationVariant; 6] = [
        AblationVariant::CsiOnly,
        AblationVariant::CirOnly,
        AblationVariant::CsiPlusCir,
        AblationVariant::PlusDoppler,
        AblationVariant::PlusBfld,
        AblationVariant::PlusUwb,
    ];

    /// Whether the variant's required hardware is present in the current fleet.
    /// `PlusUwb` returns `false` until ADR-144 ranging hardware lands.
    pub fn is_runnable(&self) -> bool {
        !matches!(self, AblationVariant::PlusUwb)
    }

    /// Stable string slug used in report tables, JSON keys, and proof-hash names.
    pub fn slug(&self) -> &'static str { /* "csi_only", "cir_only", ... */ }
}
```

**Interface boundary.** `AblationVariant` does not know how to *compute* features. It is a pure descriptor. The harness translates each variant into a `StageMask` consumed by the ADR-136 streaming engine; the streaming engine remains the single owner of feature extraction. This keeps the train crate free of any `unsafe`/FFI signal code (consistent with `lib.rs`'s note that only `tch` brings `unsafe`).

The order in `MATRIX` is **load-bearing**: it is the iteration order used by the proof hash and by the report, so it must never be re-sorted (same discipline as `proof.rs::hash_model_weights()` sorting variables by name for stable order).

### 2.2 Spec Metrics Bound to Existing Interfaces

The new metrics are added as additive structs that *compose with*, not replace, `RuViewAcceptanceResult` and `CrossDomainMetrics`. We deliberately do not widen the existing public structs (they are consumed by checked-in tests and by the `summary()` formatters), per the rule "prefer editing an existing file but do not break a stable public API."

```rust
/// Detection-mode spec metrics that the ADR-031 acceptance test does not
/// currently capture. Every field traces to a documented evaluation protocol.
#[derive(Debug, Clone)]
pub struct SpecMetrics {
    /// Presence detection accuracy (TP+TN)/N over the labelled set.
    pub presence_accuracy: f32,
    /// Localization error in metres (mean Euclidean, occupied frames only).
    pub localization_err_m: f32,
    /// Activity classification accuracy (multi-class, balanced).
    pub activity_accuracy: f32,
    /// Breathing-rate error in BPM (mirrors VitalSignResult.breathing_error_bpm).
    pub breathing_err_bpm: f32,
    /// False-positive rate: P(predict occupied | truly empty).
    pub false_positive_rate: f32,
    /// False-negative rate: P(predict empty | truly occupied).
    pub false_negative_rate: f32,
}

/// Inference-latency profile measured with `criterion`-style sampling over
/// the replay set. Wall-clock, single-frame end-to-end through the gated
/// streaming pipeline for this variant.
#[derive(Debug, Clone)]
pub struct LatencyProfile {
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    /// Number of timed frames (sample count).
    pub n_samples: usize,
}

/// Cross-room degradation, extending ADR-027 MERIDIAN reporting (§2.5).
#[derive(Debug, Clone)]
pub struct CrossRoomDegradation {
    /// room_A accuracy − room_B accuracy (signed; positive = B is worse).
    pub accuracy_delta: f32,
    /// Underlying cross-domain metrics from eval.rs (unchanged struct).
    pub cross_domain: crate::eval::CrossDomainMetrics,
    /// Per-joint degradation heatmap: 17 entries, room_A−room_B PCK per joint.
    pub per_joint_pck_delta: [f32; 17],
}
```

The acceptance thresholds for the new spec metrics extend the existing `Default`-carrying threshold structs by **adding a sibling**, not by mutating `JointErrorThresholds` etc.:

```rust
#[derive(Debug, Clone)]
pub struct SpecThresholds {
    pub min_presence_accuracy: f32, // default 0.90
    pub max_localization_err_m: f32, // default 0.50
    pub min_activity_accuracy: f32, // default 0.70
    pub max_false_positive_rate: f32, // default 0.05
    pub max_false_negative_rate: f32, // default 0.10
    pub max_p95_latency_ms: f32, // default 100.0  (ADR-136 streaming budget)
    pub max_privacy_leakage: f32, // default 0.05  (see §2.3)
}
```

`max_p95_latency_ms = 100.0` is the streaming-engine real-time budget implied by ADR-136 (20 Hz sensing → 50 ms/frame headroom with margin). `max_privacy_leakage = 0.05` is justified in §2.3.

### 2.3 Privacy-Leakage via Membership Inference

The privacy-leakage scalar measures how much an adversary holding the model's outputs can recover identity that the ADR-120 hash-rotation pipeline is supposed to destroy. We measure it as a **membership-inference (MIA) attack success above chance**, normalized to `[0, 1]`.

**Setup.** The ADR-120 pipeline maps identity features → `rf_signature_hash = BLAKE3-keyed(site_salt, day_epoch || features)` (`signature_hasher.rs`). The structural guarantee (invariant I3) is that two sites, or two days, produce uncorrelated hashes. The *measured* question is different: given the model's emitted per-frame outputs for a known set of enrolled identities (members) and an equal set of held-out identities (non-members), can a simple attacker classifier decide membership better than a coin flip?

```rust
/// Privacy-leakage measurement against the ADR-120 hash-rotation pipeline.
#[derive(Debug, Clone)]
pub struct PrivacyLeakage {
    /// MIA attacker AUC ∈ [0.5, 1.0]; 0.5 = no leakage, 1.0 = full recovery.
    pub mia_auc: f32,
    /// Normalized leakage score ∈ [0,1]: 2*(mia_auc − 0.5), clamped.
    pub leakage_score: f32,
    /// Fisher-information trace of identity-feature gradients (diagnostic).
    /// Higher trace = identity is more recoverable from model sensitivity.
    pub fisher_trace: f32,
    /// Number of (member, non-member) pairs probed.
    pub n_probes: usize,
}
```

Two estimators are reported; the harness uses the MIA estimator for the pass/fail gate and the Fisher trace as a diagnostic:

1. **MIA simulator** (gate). Train a lightweight shadow classifier on the variant's emitted outputs to predict member/non-member, evaluate its AUC on a disjoint split. `leakage_score = clamp(2·(AUC − 0.5), 0, 1)`. An AUC of 0.5 → `leakage_score = 0` (the model leaks nothing the hash rotation has not already destroyed); AUC of 1.0 → `leakage_score = 1.0`.
2. **Fisher-information trace** (diagnostic). The trace of the Fisher information matrix of the model's outputs with respect to the (pre-hash) identity features. This is a closed-form sensitivity measure: a model whose outputs are invariant to identity features has near-zero trace. It is reported but not gated, because its scale is not normalized across variants.

**Why MIA and not just trusting the structural invariant.** The BLAKE3 hash rotation guarantees that the *stored signature* cannot be cross-correlated. It says nothing about whether the *pose/presence outputs themselves* carry a usable identity fingerprint (gait, body geometry). A model can pass every ADR-120 structural test and still leak identity through its keypoint trajectories. MIA measures exactly that residual channel. The pass gate is `leakage_score ≤ 0.05`, i.e. attacker AUC ≤ 0.525 — within sampling noise of chance for the probe count used.

**Determinism.** The shadow classifier is trained with a fixed seed derived from `PROOF_SEED = 42` and a fixed split, so the AUC is reproducible. The Fisher trace is computed on the fixed replay set. Both feed the per-variant proof hash (§2.6) at coarse quantization, following the cross-platform lesson documented in `calibration_proof_runner.rs` (lines 1–13): quantize to 1e-3 in natural order, no sort, no libm-sensitive comparison.

### 2.4 `ruview-cli --ablation mode=auto`

A new CLI surface drives the matrix. It is added as a `Commands::Ablation(AblationArgs)` variant alongside the existing `Commands::Calibrate` / `Commands::Mat` / `Commands::Version` in `wifi-densepose-cli/src/lib.rs` (the same `clap` `Subcommand` enum that already hosts `Calibrate(calibrate::CalibrateArgs)`).

```
wifi-densepose ablation [OPTIONS]

OPTIONS:
    --mode <MODE>         auto | single   [default: auto]
                          auto: run the full 6-variant matrix.
                          single: run one --variant.
    --variant <SLUG>      csi_only | cir_only | csi_plus_cir |
                          plus_doppler | plus_bfld | plus_uwb
                          (required when --mode=single)
    --model <PATH>        Path to the frozen model.bin snapshot to evaluate.
    --replay <PATH>       Witness-bundle CSI replay file
                          [default: archive/v1/data/proof/sample_csi_data.json]
    --seed <N>            Proof seed [default: 42]
    --salt <HEX>          Fixed 32-byte site salt for the BLAKE3 hasher
                          (deterministic privacy probe). [default: fixed test salt]
    --out <PATH>          Markdown report path [default: ablation_report.md]
    --check-hash          Compare each variant's canonical bytes against
                          archive/v1/data/proof/expected_ablation_<slug>.sha256
                          and exit non-zero on any mismatch (CI mode).
    --generate-hash       Write/refresh the per-variant expected hashes.
```

**Auto mode flow** (mirrors `proof.rs::run_proof` discipline):

1. Snapshot the model: load `--model`, freeze weights, record `SHA-256(model.bin)` as the model-version stamp.
2. For each `AblationVariant::MATRIX` entry where `is_runnable()`:
   a. Set the streaming `StageMask`; replay the CSI under `PROOF_SEED=42` + fixed salt.
   b. Compute `RuViewAcceptanceResult`, `SpecMetrics`, `LatencyProfile`, `PrivacyLeakage`, `CrossRoomDegradation`.
   c. Serialise the variant's canonical metric bytes (coarse-quantized, natural order) and SHA-256 it. Compare to `expected_ablation_<slug>.sha256`; fail CI on mismatch in `--check-hash` mode.
3. For `PlusUwb`: emit `VariantOutcome::Skipped { reason: "ADR-144 UWB hardware not present" }`.
4. Emit the markdown report and `summary.json`.

The exit-code convention matches `proof.rs`: `0 = PASS`, `1 = FAIL` (hash mismatch or threshold breach), `2 = SKIP` (no expected hash file). This lets the ablation step drop into the existing ADR-011 / ADR-028 witness chain without a new CI grammar.

**Why `criterion` for latency.** The `criterion` crate gives a sampled distribution with percentile extraction rather than a single timing. We run a fixed warmup + sample budget so p50/p95/p99 are stable; the percentiles are quantized to 0.1 ms before hashing so wall-clock jitter does not break the proof hash (the metric is gated on `p95 ≤ threshold`, the *hash* only pins the quantized accuracy/privacy fields, not raw latency — latency is environment-dependent and therefore reported but excluded from the determinism hash, exactly as runtime wall-clock is excluded from `proof.rs`'s weight hash).

### 2.5 Cross-Room Degradation (MERIDIAN Extension)

ADR-027's `CrossDomainEvaluator` already partitions predictions by domain ID and computes `domain_gap_ratio`. This ADR extends the *reporting*, not the evaluator: it consumes the existing `evaluate()` output and adds room_A − room_B deltas plus a per-joint heatmap.

```rust
/// Extend eval.rs reporting with a two-room A/B split and a per-joint heatmap.
/// `room_a_preds`/`room_b_preds` are (pred, gt) pairs as in CrossDomainEvaluator.
pub fn cross_room_degradation(
    evaluator: &crate::eval::CrossDomainEvaluator,
    room_a: &[(Vec<f32>, Vec<f32>)],
    room_b: &[(Vec<f32>, Vec<f32>)],
) -> CrossRoomDegradation;
```

The per-joint heatmap is the 17-entry vector of `PCK_room_A[j] − PCK_room_B[j]`, indexed by COCO joint (the same 17-joint convention used in `ruview_metrics.rs::COCO_SIGMAS` and `metrics.rs::COCO_KP_SIGMAS`). The multi-room test set reuses the domain-label convention: room A is domain `0` (in-domain), room B is a non-zero domain ID. This is a pure consumer of `eval.rs` — no change to `CrossDomainEvaluator` or `CrossDomainMetrics`.

### 2.6 Determinism Binding to the Proof Harness

Each runnable variant produces a canonical byte payload hashed with SHA-256, following the established signal-proof pattern (`calibration_proof_runner.rs`, `cir_proof_runner.rs`). A new binary `src/bin/ablation_proof_runner.rs` in `wifi-densepose-signal` (alongside the two existing `*_proof_runner.rs`) regenerates the matrix on the fixed seed/salt/replay and asserts the hashes match `archive/v1/data/proof/expected_ablation_<slug>.sha256`.

**Canonical payload per variant** (coarse quantization, natural field order, no sort — the libm-portability rule from `calibration_proof_runner.rs` lines 1–13):

```
[0]  variant slug bytes (length-prefixed, like proof.rs param names)
[1]  model.bin SHA-256 (32 bytes)              ← model version
[2]  calibration version tag (from ADR-135 baseline meta)
[3]  privacy decision tag (BFLD mode, ADR-141)
[4]  pck_all        (× 1e3 round) u16
[5]  oks            (× 1e3 round) u16
[6]  mota           (× 1e3 round) u16
[7]  presence_acc   (× 1e3 round) u16
[8]  localization   (× 1e3 round, metres) u16
[9]  activity_acc   (× 1e3 round) u16
[10] fp_rate        (× 1e3 round) u16
[11] fn_rate        (× 1e3 round) u16
[12] leakage_score  (× 1e3 round) u16
[13] tier byte (0=Fail,1=Bronze,2=Silver,3=Gold)
```

Latency fields are **excluded** from the hash (wall-clock is non-deterministic across machines, exactly as `proof.rs` excludes timing). Fields `[1]`–`[3]` make the evidence-traceability rule structural: the proof hash *cannot match* unless the model version, calibration version, and privacy decision are the ones that were pinned — so every reported semantic metric traces to a specific model + calibration + privacy decision, by construction.

### 2.7 Evidence Traceability

Per the project rule that every semantic state record traces to signal evidence + model version + calibration version + privacy decision, the `AblationReport` carries these four provenance fields per variant and binds them into the proof hash (§2.6 fields `[1]`–`[3]`, plus the replay file SHA as signal evidence):

```rust
#[derive(Debug, Clone)]
pub struct VariantProvenance {
    /// Signal evidence: SHA-256 of the witness-replay CSI file.
    pub replay_sha256: String,
    /// Model version: SHA-256 of the frozen model.bin.
    pub model_sha256: String,
    /// Calibration version: ADR-135 baseline schema_version + captured_at.
    pub calibration_version: String,
    /// Privacy decision: the BFLD mode (ADR-141) under which features were gated.
    pub privacy_mode: String,
}
```

A variant whose provenance cannot be fully populated (e.g. no calibration baseline loaded) is reported as `Degraded`, never as a passing tier — the report refuses to claim a Gold tier without a calibration version, the same way ADR-135 refuses `subtract()` on a tier mismatch.

### 2.8 Output: Auto-Report and Summary JSON

The markdown report has one row per variant, columns: variant slug · tier · PCK · OKS · MOTA · presence · localization · activity · FP · FN · **leakage** · p50/p95/p99 ms · runnable?. A delta block lists pairwise deltas of interest (e.g. `csi_plus_cir − csi_only` to show CIR's contribution; `plus_bfld − csi_plus_cir` to show whether BFLD features regress privacy). `summary.json` carries the same data machine-readably plus per-variant `VariantProvenance`, for the cognitum-v0 dashboard and the ADR-141 privacy control plane to ingest.

---

## 3. Consequences

### 3.1 Positive

- **Feature contribution becomes measurable.** The `csi_plus_cir − csi_only` delta answers "did CIR (ADR-134) help?" with a number, not a commit message. Every future signal ADR can be justified or rejected against the matrix.
- **Privacy regression becomes a CI gate.** A model that leaks identity through pose trajectories — invisible to the structural ADR-120 tests — now fails `leakage_score ≤ 0.05`. This closes the residual-channel gap between *hash* privacy and *output* privacy.
- **Latency budget is enforced.** `p95 ≤ 100 ms` makes the ADR-136 real-time claim falsifiable. A Gold-accuracy model that misses the streaming budget no longer passes silently.
- **Deterministic and CI-friendly.** Reusing `PROOF_SEED=42` + fixed salt + witness replay + per-variant SHA-256 plugs directly into the ADR-011/ADR-028 witness chain. No new CI grammar; same `0/1/2` exit codes as `proof.rs`.
- **Additive, non-breaking.** `RuViewAcceptanceResult`, `CrossDomainMetrics`, and the threshold `Default` impls are untouched. The harness composes them; existing tests keep passing.
- **UWB is forward-declared.** `PlusUwb` is in the matrix as `Skipped`, so when ADR-144 hardware lands the only change is flipping `is_runnable()` and generating its expected hash.

### 3.2 Negative

- **Evaluation set must be curated.** The matrix is only as meaningful as the labelled multi-room replay set. Building a paired room_A/room_B set with presence/localization/activity labels is real work and is a prerequisite, not delivered by this ADR.
- **MIA is an estimate, not a proof.** A `leakage_score = 0` means *this* attacker found nothing; a stronger attacker might. The metric is a regression tripwire, not a cryptographic guarantee — the cryptographic guarantee remains ADR-120's structural invariant.
- **Six variants × full metric suite is slow.** The matrix runs the acceptance test, MERIDIAN eval, MIA shadow-classifier training, and criterion latency sampling per variant. This is a minutes-scale CI job, not seconds — it belongs in a nightly/witness job, not the per-commit fast path.
- **Latency excluded from the hash means latency can drift unnoticed.** We gate on `p95 ≤ threshold` but cannot pin it deterministically; a slow regression below the threshold is invisible. Mitigated by trending p95 in `summary.json` over time.

### 3.3 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MIA shadow classifier under-trained → false "no leakage" | Medium | A leaky model passes the privacy gate | Fix shadow-classifier capacity and probe count; report `n_probes`; require AUC CI confidence in `summary.json`; treat the gate as a tripwire, keep ADR-120 structural tests as the primary guarantee |
| Per-variant hash too sensitive → flaky CI across libm | Medium | Spurious FAIL on macOS vs Linux | Coarse u16 quantization at 1e-3, natural order, no sort — exactly the documented fix in `calibration_proof_runner.rs` lines 1–13; latency excluded from hash |
| Curated multi-room set leaks into training | Low | Inflated cross-room numbers | Evaluation replay set is frozen and SHA-pinned as `replay_sha256`; never used by `trainer.rs` |
| `PlusBfld` privacy probe needs a real `site_salt` | Low | Non-deterministic privacy hash | `--salt` defaults to a fixed test salt; the proof runner always uses the fixed salt so the hash is reproducible |
| Streaming `StageMask` toggles interact (e.g. CIR depends on calibration) | Medium | A variant silently runs uncalibrated | `VariantProvenance` requires a `calibration_version`; missing → `Degraded`, never a passing tier (§2.7) |

---

## 4. Alternatives Considered

### 4.1 Widen `RuViewAcceptanceResult` Instead of Adding Sibling Structs

Rejected. `RuViewAcceptanceResult` and its `summary()` are consumed by checked-in tests in `ruview_metrics.rs` (e.g. `tier_determination_gold`) and likely by downstream callers. Adding `presence_accuracy`, `leakage_score`, etc. as fields would churn those tests and the `summary()` format string. The additive `SpecMetrics` / `LatencyProfile` / `PrivacyLeakage` siblings compose cleanly and leave the ADR-031 contract intact.

### 4.2 A Hyperparameter Sweep Framework Instead of a Fixed Matrix

Rejected. A general sweep (Optuna-style) optimizes *training*; this ADR evaluates a *frozen* model under inference-time feature gates. Conflating them would couple the harness to `trainer.rs` and break the proof-determinism story (a sweep is, by design, exploratory and non-deterministic). The fixed six-variant matrix is the minimum that answers "what does each feature contribute?" deterministically.

### 4.3 Differential Privacy Accounting Instead of MIA

Rejected for this scope. DP (ε-accounting) is a *training-time* mechanism; it would require instrumenting the training loop with noise and a privacy ledger. The deployed model is already trained, and the question here is empirical output leakage on a fixed snapshot — MIA answers that directly with no training-time change. DP remains a valid future ADR for the training pipeline, but it does not measure residual leakage of an already-shipped model.

### 4.4 Skip Latency Entirely (Accuracy-Only Ablation)

Rejected. ADR-136 makes a real-time streaming claim with no enforcement. Without a `p95` gate, a feature that doubles accuracy but triples latency would "win" the ablation and ship, breaking the 20 Hz budget. Latency is reported and gated even though it is excluded from the determinism hash.

### 4.5 Define `+UWB` as Absent Rather Than `Skipped`

Rejected. Silently omitting `PlusUwb` until hardware lands would mean the matrix shape changes when hardware arrives, breaking report diffs and the per-variant hash set. The `Skipped { reason }` outcome keeps the matrix shape stable and self-documenting — the same discipline ADR-135 §2.8 uses for unprovisioned seed nodes.

---

## 5. Testing and Acceptance

### 5.1 Acceptance Criteria

| ID | Criterion | Evidence |
|----|-----------|----------|
| AC1 | `AblationVariant::MATRIX` has exactly 6 entries in canonical order; `PlusUwb.is_runnable() == false`, all others `true`. | `ablation::tests::matrix_shape` |
| AC2 | `cross_room_degradation()` returns a 17-entry `per_joint_pck_delta` and a signed `accuracy_delta`; perfect-equal rooms → all-zero heatmap and `accuracy_delta == 0`. | `ablation::tests::cross_room_zero_when_identical` |
| AC3 | `PrivacyLeakage` on an identity-invariant model → `leakage_score < 0.05` (AUC ≈ 0.5); on an identity-encoding model → `leakage_score > 0.5`. | `ablation::tests::mia_separates_leaky_model` |
| AC4 | `SpecThresholds::default()` gates: `presence ≥ 0.90`, `loc ≤ 0.50 m`, `activity ≥ 0.70`, `FP ≤ 0.05`, `FN ≤ 0.10`, `p95 ≤ 100 ms`, `leakage ≤ 0.05`. | `ablation::tests::spec_thresholds_default` |
| AC5 | A variant with missing `calibration_version` is reported `Degraded`, never a passing tier. | `ablation::tests::no_calibration_is_degraded` |
| AC6 | Re-running the matrix under `PROOF_SEED=42` + fixed salt + fixed replay produces byte-identical canonical payloads (per-variant hash stable across two runs). | `ablation::tests::canonical_bytes_deterministic` |
| AC7 | `ablation_proof_runner` exits `0` when all runnable variants match `expected_ablation_<slug>.sha256`, `1` on any mismatch, `2` on placeholder hashes. | `cargo run -p wifi-densepose-signal --bin ablation_proof_runner --release --no-default-features` |
| AC8 | The proof hash changes if the model SHA, calibration version, or privacy mode changes (provenance is bound into the hash). | `ablation::tests::provenance_affects_hash` |

### 5.2 Test Tiers

**Tier 1 — Matrix and metric unit tests (CI).** `matrix_shape`, `spec_thresholds_default`, `cross_room_zero_when_identical`, and the MIA separation test with two synthetic models (one identity-invariant, one that copies an identity feature into its output). These run without `tch` and without hardware.

**Tier 2 — Determinism proof (CI, extends ADR-011/ADR-028).** `ablation_proof_runner` regenerates each runnable variant's canonical bytes on `PROOF_SEED=42` + fixed salt + `sample_csi_data.json` replay and hashes them. Expected hashes live at `archive/v1/data/proof/expected_ablation_<slug>.sha256`. Until the harness lands, each file holds a `PLACEHOLDER` token and the runner exits `2` (the same bootstrap pattern as `calibration_proof_runner.rs`).

**Tier 3 — Full auto-report integration (nightly).** `wifi-densepose ablation --mode=auto --model <frozen.bin> --check-hash` runs the complete matrix, emits `ablation_report.md` + `summary.json`, and asserts every runnable variant's hash matches. `PlusUwb` is asserted `Skipped`.

**Tier 4 — Real-hardware sanity (gated, not CI).** Behind `#[cfg(feature = "hardware-test")]`: replay a live 30 s capture from the ESP32-S3 on COM9 through `csi_only` and `csi_plus_cir`, assert `csi_plus_cir` does not regress presence accuracy and that p95 latency stays under the 100 ms budget on the ruvzen box.

### 5.3 Witness / Proof Rows

Per ADR-028, three rows are added to `docs/WITNESS-LOG-028.md`:

| Row | Capability | Evidence | Hash |
|-----|-----------|----------|------|
| W-39 | Ablation matrix deterministic over 5 runnable variants | `ablation_proof_runner` exits 0 | SHA-256 of `csi_plus_cir` canonical bytes |
| W-40 | Privacy-leakage MIA separates leaky vs invariant model | `cargo test ablation::tests::mia_separates_leaky_model` | SHA-256 of test binary |
| W-41 | Provenance binds model+calibration+privacy into the proof hash | `cargo test ablation::tests::provenance_affects_hash` | SHA-256 of two distinct-provenance payloads |

`source-hashes.txt` in the witness bundle gains `SHA-256(wifi-densepose-train/src/ablation.rs)` and `SHA-256(wifi-densepose-signal/src/bin/ablation_proof_runner.rs)`.

---

## 6. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-011 (Deterministic Proof Harness) | **Substrate**: per-variant SHA-256 + `PROOF_SEED=42` + `0/1/2` exit codes reuse the `proof.rs` discipline directly |
| ADR-014 (SOTA Signal Processing) | **Source**: the Doppler/spectrogram feature gated by the `PlusDoppler` variant |
| ADR-027 (MERIDIAN Cross-Environment) | **Extended (reporting)**: `cross_room_degradation()` consumes `eval.rs::CrossDomainEvaluator` and adds A/B deltas + per-joint heatmap; the evaluator itself is unchanged |
| ADR-031 (RuView Sensing-First RF Mode) | **Extended**: the ablation harness drives the `ruview_metrics.rs` acceptance test (`RuViewTier`, `JointErrorThresholds`, …) per variant, adding presence/localization/activity/FP-FN/latency/privacy metrics it did not previously capture |
| ADR-120 (BFLD Privacy Class & Hash Rotation) | **Measured**: the MIA probe attacks the `signature_hasher.rs` hash-rotation pipeline's residual output leakage; the structural invariants remain the primary guarantee |
| ADR-136 (RuView Streaming Engine) | **Consumer/owner of features**: the harness toggles ADR-136 `StageMask`; the streaming engine remains the sole feature-extraction owner; the `p95 ≤ 100 ms` gate enforces ADR-136's real-time claim |
| ADR-141 (BFLD Privacy Control Plane) | **Provenance source/consumer**: the `privacy_mode` in `VariantProvenance` is an ADR-141 named mode; `summary.json` feeds the control plane |
| ADR-144 (UWB Range-Constraint Fusion) | **Forward-declared**: `PlusUwb` is a defined-but-`Skipped` variant until ADR-144 ranging hardware lands |

---

## 7. References

### Production Code

- `v2/crates/wifi-densepose-train/src/ruview_metrics.rs` — ADR-031 acceptance test; `RuViewAcceptanceResult`, `RuViewTier`, `JointErrorThresholds`, `determine_tier()` reused unchanged
- `v2/crates/wifi-densepose-train/src/eval.rs` — `CrossDomainEvaluator`, `CrossDomainMetrics`; consumed by `cross_room_degradation()`
- `v2/crates/wifi-densepose-train/src/metrics.rs` — `MetricsResult`, `COCO_KP_SIGMAS`; 17-joint convention for the per-joint heatmap
- `v2/crates/wifi-densepose-train/src/proof.rs` — `run_proof`, `PROOF_SEED`, `hash_model_weights`, `0/1/2` exit-code convention reused as the harness substrate
- `v2/crates/wifi-densepose-train/src/ablation.rs` — **new**: `AblationVariant`, `AblationHarness`, `SpecMetrics`, `LatencyProfile`, `PrivacyLeakage`, `CrossRoomDegradation`, `VariantProvenance`
- `v2/crates/wifi-densepose-signal/src/bin/calibration_proof_runner.rs` — canonical-bytes / coarse-quantization / libm-portability pattern (lines 1–13) reused
- `v2/crates/wifi-densepose-signal/src/bin/cir_proof_runner.rs` — sibling proof-runner pattern
- `v2/crates/wifi-densepose-signal/src/bin/ablation_proof_runner.rs` — **new**: regenerates the matrix hashes
- `v2/crates/wifi-densepose-bfld/src/signature_hasher.rs` — ADR-120 BLAKE3 hash-rotation pipeline; MIA target
- `v2/crates/wifi-densepose-bfld/src/embedding.rs` — `IdentityEmbedding` (in-RAM-only); identity-feature source for the Fisher trace
- `v2/crates/wifi-densepose-cli/src/lib.rs` — `Commands` enum; new `Commands::Ablation(AblationArgs)` variant beside `Calibrate`
- `archive/v1/data/proof/expected_ablation_<slug>.sha256` — **new**: per-variant expected hashes
- `archive/v1/data/proof/sample_csi_data.json` — default witness replay set
- `archive/v1/data/proof/verify.py` — proof chain; gains an `ablation_matrix_check()` extension
- `docs/WITNESS-LOG-028.md` — rows W-39 through W-41

### External References

- Shokri, R. et al. (2017). "Membership Inference Attacks Against Machine Learning Models." *IEEE S&P*. — Shadow-classifier MIA methodology underlying the `mia_auc` estimator.
- Carlini, N. et al. (2022). "Membership Inference Attacks From First Principles." *IEEE S&P*. — AUC-based leakage normalization and the "attacker AUC above 0.5" framing used for `leakage_score`.
- COCO Keypoint Evaluation. — PCK / OKS definitions and the 17-joint sigmas mirrored from `ruview_metrics.rs` and `metrics.rs`.
- Bernstein, J.-P. (BLAKE3 team) (2020). *BLAKE3 specification*. — Keyed-hash mode used by `signature_hasher.rs`, the ADR-120 pipeline under privacy test.


---

## Implementation Status & Integration (2026-05-29)
*Part of the ADR-136 streaming-engine series -- skeleton/scaffolding, trust-first, mostly not yet on the live 20 Hz path. See ADR-136 (Implementation Status) for the series framing.*

**Built -- tested building block** (commit `0f336b7d3`, issue #849): the 6-variant `FeatureSet` matrix and `AblationMetrics` (FP/FN, latency p50/p95, membership-inference privacy leakage, cross-room degradation) with a deterministic markdown report and the `csi_cir_beats_csi_only` acceptance check. 5 tests.

**Integration glue -- not yet on the live path:** the `ruview-cli --ablation mode=auto` subcommand that snapshots the model and runs the 6 variants under `PROOF_SEED=42` witness-bundle replay (also where ADR-136 AC6 lands); the `+UWB` variant once ADR-144 hardware exists.

**Trust contribution:** makes every pipeline change *measurable* -- including how much a model leaks about its training data -- so improvements are proven, not asserted. The scorecard behind every other claim in the series.
