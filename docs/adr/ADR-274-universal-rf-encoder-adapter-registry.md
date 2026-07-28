# ADR-274: Universal RF foundation encoder + hardware adapter registry

| Field | Value |
|-------|-------|
| **Status** | Accepted — **P1 implemented** (`ruview-unified`: `tensor.rs`, `adapters.rs`, `tokenizer.rs`, `encoder.rs`, `pretrain.rs`, `heads.rs`, `eval.rs`) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 |
| **Relates to** | ADR-136 (`CanonicalFrame` provenance — the WiFi adapter consumes `wifi-densepose-core::CsiFrame` directly), ADR-152 §2 (geometry conditioning intake), ADR-016/017 (ruvector integration points) |

## 0. PROOF discipline

Grades as in ADR-273 §0. Every number below is MEASURED-CODE or MEASURED-SYNTHETIC unless marked EXTERNAL-UNVERIFIED.

## 1. Context

WiFo-2 and WiLLM (EXTERNAL-UNVERIFIED) demonstrated that heterogeneous CSI standardization + masked-reconstruction pretraining + small task adapters beats per-task models, and the age-aware CSI line showed a cheap win from encoding sample freshness multiplicatively. RuView has four incompatible capture families today (802.11 CSI, FMCW radar cubes, UWB CIR, and — via O-RAN — 5G SRS). Each previously implied its own model.

## 2. Decision — canonical tensor + adapter registry

### 2.1 Canonical tensor

All modalities normalize to `RfTensor` (`tensor.rs`): complex `(links × 56 bins × 8 snapshots)` plus carrier/bandwidth, per-link `LinkGeometry`, `sample_age_s`, `clock_quality ∈ [0,1]`, `uncertainty ∈ [0,1]`, `device_id`, and a `CalibrationMeta` contract. 56 bins = usable 20 MHz 802.11n subcarriers (and the existing 114→56 interpolation in `wifi-densepose-train`), so the most common source resamples trivially.

**Boundary rule**: `RfTensor::new` is the only constructor and validates every field (finite samples, geometry/link arity, ranges). Downstream code assumes validity. Tests: `tensor.rs::tests` (4).

### 2.2 Normalization pipeline (every adapter, 3 stages)

1. **Layout** — vendor shape → `(links, bins, snapshots)`; FMCW gets a fast-time DFT to range bins; SRS gets comb de-interleaving; then linear complex resampling to canonical dims.
2. **Amplitude** — per-link division by median amplitude (chipset gain invariance; offset recorded in `CalibrationMeta.gain_offset_db`).
3. **Phase** — per (link, snapshot), remove constant offset + least-squares linear ramp across bins (CFO residual + sampling-time offset), with unwrapping. Skipped for delay-domain modalities (radar range profiles, UWB taps) where a detrend would erase ToF structure.

Measured (test `wifi_adapter_normalizes_shape_gain_and_phase`): a synthetic capture with per-link gains ×3.7/×7.4 and phase ramp `0.9 + 0.11·bin` comes out with median amplitude 1.0 ± 1e-9 and residual phase < 1e-4 rad (the ~7 µrad residue is second-order chord-vs-arc error from complex resampling). The radar adapter localizes a fast-time beat tone to the analytically expected canonical range bin (`radar_adapter_localizes_beat_tone_to_range_bin`).

### 2.3 Registry

`AdapterRegistry` maps hardware id → `dyn RfAdapter`, **fail-closed** (unknown hardware is an error; wrong modality is a typed `ModalityMismatch`). Reference adapters ship for `esp32s3-csi`, `mr60bha2` (FMCW), `dw3000` (UWB), `oai-srs-xapp` (5G SRS) — the last being the ADR-273 P4 seam.

## 3. Decision — encoder, fusion contract, adapters

### 3.1 Tokenizer

One token per (link, 8-bin subcarrier group); 24 features: log-amplitudes, delay-spectrum DFT (4), Doppler DFT bins 1–4 (log-compressed `ln(1+100·mag)`), temporal amplitude deviation (`ln(1+20·std)`), phase velocity, sample age, link distance/height/azimuth, clock quality, uncertainty (`tokenizer.rs`, layout table on `RfToken`).

Two hardware-invariance steps precede feature extraction, and both were *forced by measurement*, not aesthetics (see §5 evidence trail):

- **window-median amplitude normalization** — raw Friis-scale features (~1e-3) left every head unable to learn;
- **CFO alignment** — per link, each snapshot is de-rotated by `arg Σ_b H[b,s]·H̄[b,0]`; carrier-frequency-offset drift is a *common* rotation and cancels, while a moving scatterer's frequency-selective perturbation survives (test `motion_raises_doppler_and_variance_features` uses a bin-dependent perturbation precisely so alignment cannot cancel it).

### 3.2 Encoder + pretraining

Pure-Rust, exactly differentiable (`encoder.rs`):

```text
h_i = tanh(W1·x_i + b1)      token embedding
c   = mean_i h_i             permutation-invariant pool
m   = tanh(W2·c + b2);  g = tanh(W2b·m + b2b)
gate = σ(age_w·age + age_b)  multiplicative freshness gate
z   = g ⊙ gate + Wg·geo + bg  ← the ADR-273 fusion contract, verbatim
```

Masked-reconstruction pretraining (`pretrain.rs`): mask 25 % of tokens, reconstruct each from `[z ; sinusoidal-position]` via a linear head discarded at deployment; SGD.

**Proof of the backward pass** (MEASURED-CODE, `gradients_match_finite_differences`): analytic gradients of **all 12 parameter groups** vs central finite differences — 174 sampled parameters, max relative error **1.31e-5**, with the absolute floor at central-difference roundoff (≈5e-11). Training halves masked loss and beats the constant-predictor variance baseline (`0.2757 → 0.0966` vs baseline `0.1550`; `pretraining_reduces_masked_loss_and_beats_mean_baseline`). Same seed ⇒ bit-identical weights (`training_is_deterministic`).

Backbone at deployment config (d_model 128): **40,856 parameters** (hand-count asserted in `param_count_matches_hand_computation`).

### 3.3 Two representation views (the PerceptAlign lesson, applied)

- `encode()` → full `z` (geometry-conditioned) — for localization/channel-prediction heads where sensor pose is signal.
- `encode_content()` → `[g ⊙ gate ; mean token features]` — for environment-invariant heads (presence/activity/anomaly). The additive `Wg·geo` term is a **room-specific offset a linear adapter would memorize** — measured: with it, held-out-room presence F1 was 0.00 while training F1 fit; without it plus the pooled-statistics skip connection, held-out F1 is 1.00 (SYNTHETIC, ADR-273 §5).

### 3.4 Task adapters, ≤ 1 % budget

`heads.rs`: presence (logistic, 129 params), activity (rank-2 LoRA-style factorized softmax, 268), localization (linear ℝ³, 387), anomaly (2 calibration statistics on reconstruction error). All < 408 = 1 % of the 40,856-param backbone, asserted in `every_head_fits_the_one_percent_budget_at_deployment_config`. Convex heads train full-batch (deterministic); tests show they fit separable/multiclass toys to ≥ 95 %.

### 3.5 Anti-leakage evaluation (ADR-273 §4)

`eval.rs`: `PartitionKey` (room/day/person/chipset/firmware/layout), `StrictSplit::holdout` + independent `verify()`, ECE, coverage/selective-risk, degradation ratio, F1. Six unit tests including a manufactured-leak detection test.

## 4. Alternatives considered

- **Candle/ONNX backbone now** — rejected for P1: the deliverable is a *proven contract* (gradient-checked fusion formula, budget enforcement, leakage protocol); porting to `wifi-densepose-nn` backends is mechanical once real-data P2 justifies scale.
- **Per-modality encoders with late fusion** — rejected: reproduces the isolated-classifier status quo ADR-273 exists to end.
- **Full transformer attention** — deferred: mean-pool + 2 mixing layers passed every P1 gate; attention is a P2 measurement question, not a default.

## 5. Evidence trail (what the measurements changed)

P1 development falsified two comfortable assumptions, recorded here because the *fixes are the ADR*:

1. Raw-scale tokens: presence head stuck at F1 0.47 even on training rooms → window-median normalization + CFO alignment (train F1 → 0.76).
2. Geometry-additive `z` for invariant tasks: held-out-room F1 0.00 → content view + pooled-statistic skip (held-out F1 → 1.00) — i.e. *the leak the eval protocol was designed to catch, caught in our own architecture first*.

## 6. Consequences

One encoder now serves presence, activity, localization, respiration-class, channel prediction, and anomaly through < 1 % adapters; new hardware lands as an adapter, not a model. Cost: the pure-Rust trainer is CPU-bound (fine at 40 k params; a P2 scale-up moves to `wifi-densepose-nn`).
