# ADR-154: Signal/DSP Beyond-SOTA Sweep — Milestone 0 (Correctness, Provable Perf, and the SOTA Landscape)

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-06-11 |
| **Deciders** | ruv |
| **Codebase target** | `wifi-densepose-signal` (`ruvsense/`, `features.rs`, `csi_processor.rs`, `spectrogram.rs`, `bvp.rs`), benches, docs |
| **Relates to** | ADR-134 (CIR sparse recovery), ADR-135 (Empty-Room Baseline), ADR-029/030/032 (Multistatic mesh + security), ADR-152 (WiFi-Pose SOTA 2026 intake), ADR-153 (802.11bf forward-compat) |
| **Scope** | Milestone 0 of the beyond-SOTA signal/DSP sweep: high-leverage **correctness/security fixes**, two **measured** perf wins, the per-module SOTA landscape with evidence grades, and a prioritized roadmap. **45 review findings are explicitly deferred** (§7 backlog) — nothing is silently dropped. |

---

## 0. PROOF discipline (this ADR's contract)

This project has been publicly accused of "AI slop." This ADR answers that with **evidence, not adjectives**:

- Every claimed code improvement ships with a **committed regression test** (correctness) or a **committed criterion bench** (performance).
- Every perf number below is **MEASURED before/after** with the exact reproduce command. A perf claim without a measured before/after is **UNPROVEN** and is not made here.
- Every external SOTA reference is graded **MEASURED** / **CLAIMED** / **THEORETICAL**, distinguishing what a paper *measured* from what it *asserts* and from what is merely *plausible*.
- The headline finding — a **dead CIR coherence gate that silently fell back in production for every canonical frame** — is disclosed in full (§2), not buried.

Test machine for the perf numbers: Windows 11, `cargo bench --release`, criterion 0.5. Numbers are wall-clock medians on this box; they are about **ratios** (before/after), which are stable across machines, not absolute ns.

---

## 1. Context

The RuvSense signal stack (16 `ruvsense/` modules + the classic `features.rs`/`csi_processor.rs`/`spectrogram.rs`/`bvp.rs` pipeline) grew quickly across ADR-014/029/030/134/135. A beyond-SOTA review surfaced ~50 findings ranging from two **critical correctness/security defects** to micro-optimizations and SOTA-gap research items. Milestone 0 closes the **provable, high-leverage subset**: the two criticals, a divide-by-zero trio, two measured perf wins, and the research landscape. The remaining ~45 are catalogued in §7 so the backlog is explicit and auditable.

---

## 2. The headline finding — the ADR-134 CIR coherence gate was DEAD in production (CRITICAL, FIXED)

### 2.1 What was wrong

`MultistaticFuser` fuses **canonical CSI frames**: `hardware_norm.rs` resamples every chipset onto a uniform **56-tone canonical grid** before fusion (`HardwareNormalizer`, default `canonical_subcarriers = 56`). The ADR-134 CIR coherence gate (`cir_gate_coherence`, multistatic.rs) is supposed to blend a CIR dominant-tap ratio into the cross-node coherence — `coherence = 0.7·freq + 0.3·dominant_tap_ratio`.

But the gate was wired to `CirEstimator::new(CirConfig::ht20())` (`with_cir_ht20`), and `ht20()` expects **64 FFT bins or 52 active tones**. A canonical-56 frame matches *neither*, so every call returned `CirError::SubcarrierMismatch` and `cir_gate_coherence` hit its **silent `Err(_) => freq_coherence` fallback** (multistatic.rs). Net effect: **the CIR gate never ran on a single production frame** — `use_cir_gate = true` was indistinguishable from `false`. This is the exact shape of "AI slop": a feature that compiles, has tests on the *estimator*, and is dead at the *integration seam*.

### 2.2 The fix (the gate now actually runs)

- New `CirConfig::canonical56()` (cir.rs): 64-bin HT20 framing, **56 active tones**, 168 delay taps, Φ built over a contiguous −28..+28 active-tone grid (also the native Atheros-56 layout). `bandwidth_hz`/`tap_spacing` stay physically correct for a 20 MHz HT20 channel; only the active-tone count differs from `ht20()`.
- New `MultistaticFuser::with_cir_canonical56()` — the **correct default** for the RuvSense pipeline. `with_cir_ht20()` is retained for genuine raw-64/52 feeds and now carries a loud doc-warning.
- `active_indices()` handles `(64, 56)` explicitly and the fallback now selects the slice whose length matches `num_active` (so Φ's column count is always self-consistent — no silent fall-through to the 52-index slice).
- The remaining silent fallback is made **LOUD**: a `SubcarrierMismatch` inside `cir_gate_coherence` now fires a `debug_assert!` naming the misconfiguration ("CIR gate DEAD … build it with `CirConfig::canonical56()`"). A *config* error can no longer hide as a graceful runtime degrade.
- `cir_estimate_first()` exposes the raw `estimate()` verdict so a test can **count Ok vs Err** on a canonical-56 stream.

### 2.3 The PROOF (committed regression tests, `ruvsense::multistatic::tests`)

| Test | Asserts | Result |
|------|---------|--------|
| `cir_gate_ht20_is_dead_on_canonical56` | old ht20 estimator on 8 canonical-56 frames → **0 Ok, 8 `SubcarrierMismatch`** | the dead gate, measured |
| `cir_gate_canonical56_is_alive` | new canonical56 estimator on the same 8 frames → **8 Ok, 0 Err** | the gate runs |
| `cir_gate_on_changes_coherence_vs_off` | `coherence(gate on)` ≠ `coherence(gate off)` (\|Δ\| > 1e-6) | the CIR term is actually applied |
| `cir_gate_dead_ht20_equals_gate_off` (release-only) | dead-ht20 coherence == gate-off coherence (\|Δ\| < 1e-9) | confirms the silent degradation the fix removes |

**Reproduce:**
```bash
cd v2 && cargo test -p wifi-densepose-signal --no-default-features --lib \
  ruvsense::multistatic::tests::cir
# 3 passed (the 4th is #[cfg(not(debug_assertions))], add --release to run it)
```

**Resolution: FIXED** (not merely loud-fail-documented). The gate now decodes 100% of canonical-56 frames where it previously decoded 0%.

---

## 3. The second critical — NaN/inf adversarial-detector bypass (CRITICAL, FIXED)

### 3.1 What was wrong

`AdversarialDetector::check` (adversarial.rs) takes per-link `link_energies: &[f64]`. A single **NaN/inf** entry bypassed the whole detector: every `e > threshold` test is `false` on NaN, the Gini sort used `partial_cmp().unwrap_or(Equal)`, and the final `anomaly_score.clamp(0,1)` returns NaN on a NaN input. A real RF link can never have NaN/inf energy, so a non-finite input is *itself* the strongest possible spoof — yet it could slip through as "clean."

### 3.2 The fix

Finite-validate at the boundary: the first non-finite `link_energies` entry now **short-circuits to a definite anomaly** (`anomaly_detected = true`, `anomaly_score = 1.0`, `affected_links = [bad_idx]`, `FieldModelViolation`), and the poisoned frame is **not** seeded into the temporal-continuity state.

### 3.3 The PROOF

| Test | Asserts |
|------|---------|
| `nan_link_energy_flags_anomaly` | a NaN link energy → `anomaly_detected`, score 1.0, affected link reported, `anomaly_count == 1` |
| `inf_link_energy_flags_anomaly` | both `+inf` and `−inf` → anomaly, score 1.0 |

```bash
cd v2 && cargo test -p wifi-densepose-signal --no-default-features --lib \
  ruvsense::adversarial::tests::nan_link ruvsense::adversarial::tests::inf_link
```

---

## 4. Divide-by-(n−1) window trio (CORRECTNESS, FIXED)

Three windowing helpers divided by `(n − 1)` with no small-`n` guard:

| Site | Bug | Fix |
|------|-----|-----|
| `csi_processor.rs` `CsiPreprocessor::hamming_window(n)` | `n=0` underflowed `0usize − 1`; `n=1` divided by 0 → all-NaN window | `match n { 0 => [], 1 => [1.0], _ => … }` |
| `bvp.rs` Hann window | `window_size=1` divided by 0 → NaN BVP | length-1 guard → constant `[1.0]` |
| `spectrogram.rs` `make_window` | `size=1` divided by 0 for Hann/Hamming/Blackman | `size <= 1` short-circuit → `vec![1.0; size]` |

The standard convention for a length-1 window is the constant `1.0`; length-0 is empty.

**PROOF:** `test_hamming_window_degenerate_sizes` (csi_processor), `bvp_window_size_one_is_finite` (bvp), `make_window_size_0_and_1_are_safe` (spectrogram) — each asserts finiteness at sizes 0/1/2.

The Python deterministic proof (`archive/v1/data/proof/verify.py`) still prints **VERDICT: PASS** with the **same** pipeline hash `f8e76f21…46f7a` — the reference path uses `n ≥ 2`, so the guard is bit-transparent there.

---

## 5. Measured performance wins (MEASURED before/after; benches committed)

Both changes are **bit-equivalent** (asserted by a committed test) — they only remove wasted work. New criterion benches in `benches/features_bench.rs` (registered in `Cargo.toml`).

**Reproduce both:**
```bash
cd v2 && cargo bench -p wifi-densepose-signal --no-default-features --bench features_bench
# compile-only: append --no-run
```

### 5.1 FFT-planner caching for PSD (features.rs)

`PowerSpectralDensity::from_csi_data` constructed a fresh `FftPlanner` and re-planned the FFT **on every frame** — and `FeatureExtractor::extract` calls it per frame on the hot path. New `from_csi_data_with_fft(csi, fft_size, &Arc<dyn Fft>)` reuses a plan cached in `FeatureExtractor` (built once in `new()`). Output is **bit-identical** (`psd_cached_fft_bit_identical_to_fresh` compares `f64::to_bits` of values + all summary stats across 6 FFT sizes).

Bench group `psd_fft_planner` — `fresh_planner` (before) vs `cached_planner` (after), per frame:

| fft_size | before (fresh plan), median | after (cached), median | speedup |
|----------|------------------------------|-------------------------|---------|
| 64  | 5.84 µs/frame | 1.89 µs/frame | **3.09×** |
| 128 | 9.31 µs/frame | 3.61 µs/frame | **2.58×** |
| 256 | 13.77 µs/frame | 6.73 µs/frame | **2.04×** |

Medians from criterion (warm-up 1 s, 20 samples). Raw three-point estimates (low/median/high), per frame:
`fresh/64 [5.27, 5.84, 6.34] µs` vs `cached/64 [1.76, 1.89, 2.03] µs`;
`fresh/256 [13.29, 13.77, 14.32] µs` vs `cached/256 [6.26, 6.73, 7.43] µs`.
The win is the re-planned `FftPlanner` construction the cache hoists out of the per-frame loop; it grows in *relative* terms at small FFTs (planning is a larger fraction of a cheap transform) and stays a flat ~2× at 256.

### 5.2 DTW Sakoe-Chiba band honored (gesture.rs)

`dtw_distance` computed the band bounds `j_start/j_end` but still iterated the **full** `1..=m` row, `continue`-ing on out-of-band cells — so the band constrained the *path* but not the *work* (still O(n·m)). The fix iterates only `j_start..=j_end` (O(n·band)), resetting just the two boundary-guard cells the recurrence can read, and computes the endpoint reachability (`|n−m| ≤ band`) at the return site. Result is **bit-identical** to the full-row version across 12 shapes × 8 band widths (`dtw_banded_bit_identical_to_fullrow`).

Bench group `dtw_sakoe_chiba` — `full_row` (before) vs `banded` (after):

| case | before (full row), median | after (banded), median | speedup |
|------|-----------------------------|--------------------------|---------|
| n=m=100, band=5  | 33.45 µs | 13.77 µs | **2.43×** |
| n=m=200, band=5  | 122.32 µs | 29.55 µs | **4.14×** |
| n=m=200, band=10 | 159.98 µs | 60.19 µs | **2.66×** |

Medians from criterion (warm-up 1 s, 20 samples). Raw (low/median/high):
`full_row n200_band5 [107.6, 122.3, 146.5] µs` vs `banded n200_band5 [26.4, 29.5, 33.1] µs`.
The speedup tracks the inner-loop cell-count ratio `m / (2·band+1)` — n=m=200, band=5 → 200/11 ≈ 18× fewer cells, but euclidean-distance cost and loop overhead dominate at these sizes so the wall-clock win is ~4× (still the **largest at the longest sequence / narrowest band**, exactly as the algorithm predicts). It shrinks toward 1× as the band widens to cover the whole matrix (band=10 → 2.66×), and grows with sequence length (band=5: 2.43× at n=100 → 4.14× at n=200).

> **Note on the other re-plan sites.** `spectrogram.rs`/`bvp.rs` plan their FFT **once per call** and reuse it across all frames/subcarriers (already amortized), so caching there is marginal — deferred (§7). The PSD site was the only one re-planning *per frame*.

---

## 6. Per-module SOTA landscape (evidence-graded)

Grades: **MEASURED** (the source measured it, ideally with public method/code), **CLAIMED** (asserted, no reproducible artifact), **THEORETICAL** (plausible, no published target).

### 6.1 CSI → CIR (cir.rs — our ISTA/L1 sparse recovery)

- **Deep-unfolded ISTA / LISTA for CSI→CIR — MEASURED.** Learned ISTA unrolling reports ~**3 dB NMSE** improvement over classical OMP/FISTA for channel/CIR estimation (arXiv [2211.15440](https://arxiv.org/abs/2211.15440); survey [2502.05952](https://arxiv.org/abs/2502.05952)). Public methods; numbers measured in-paper. **This is our #1 future item (§7) — our `cir.rs` already builds the sub-DFT Φ that LISTA would make trainable.**
- **Diffusion CIR prior — MEASURED (artifact).** [github.com/benediktfesl/Diffusion_channel_est](https://github.com/benediktfesl/Diffusion_channel_est) ships **public weights** for a diffusion-model channel-estimation prior. Heavier than our edge budget; tracked, not adopted.
- **Coherence gating (the §2 gate) — THEORETICAL.** Our 0.7/0.3 freq/CIR blend is an engineering heuristic with no published accuracy target; now that it *runs*, it can finally be A/B-measured.

### 6.2 Adversarial robustness (adversarial.rs)

- **Adversarial-robustness eval for WiFi sensing — MEASURED.** arXiv [2511.20456](https://arxiv.org/abs/2511.20456) + the **Wi-Spoof** benchmark provide a measured evaluation protocol for spoofed/injected CSI. Our detector's physical-plausibility checks (consistency/Gini/temporal/energy) are in the same spirit; adopting Wi-Spoof as an external benchmark is a §7 item. (The §3 NaN fix is a precondition: a detector that NaN-bypasses can't be benchmarked honestly.)

### 6.3 Multi-AP / multistatic fusion (multistatic.rs)

- **Bayesian multi-AP fusion — CLAIMED.** arXiv [2512.02462](https://arxiv.org/abs/2512.02462) proposes a Bayesian fusion across APs; **no code released**, numbers self-reported. Our attention-weighted fusion is a different (cheaper) mechanism; tracked as a comparison target, not adopted.

### 6.4 RF intention-lead / pre-movement (intention.rs) — THEORETICAL

The 200–500 ms pre-movement "lead signal" framing has **no published commodity-WiFi target** we can grade. Honestly THEORETICAL; no work item.

---

## 7. Decision, roadmap, and the deferred-findings backlog

### 7.1 Accepted now (this milestone)

The §2–§5 fixes are **ACCEPTED and committed**: dead CIR gate fixed, NaN bypass fixed, window trio fixed, calibration dead-branch de-misled, two measured perf wins. All `cargo test -p wifi-densepose-signal --no-default-features` (and `--features cir`) green; Python proof PASS.

### 7.2 Top accepted-future item — LISTA-for-CIR (NOT implemented here)

**Unroll the existing ISTA in `cir.rs` into trainable layers (LISTA).** Effort: **M**. The sensing matrix Φ and the ISTA recurrence already exist; LISTA replaces the fixed step size / threshold with per-layer learned parameters over a fixed unroll depth. Measured target to beat: **~3 dB NMSE over OMP/FISTA** (arXiv 2211.15440 — MEASURED). Proposed, not built in Milestone 0.

### 7.3 Other graded-future items

- Adopt **Wi-Spoof** (arXiv 2511.20456, MEASURED) as the external adversarial benchmark for `adversarial.rs`.
- Evaluate the **diffusion CIR prior** (public weights, MEASURED) as an offline quality ceiling — *not* an edge target.
- Bayesian multi-AP fusion (2512.02462, CLAIMED) — comparison only, pending released code.

### 7.4 Deferred Milestone-0 review findings (explicit backlog)

Catalogued so nothing is silently dropped. Priority: **P1** correctness-adjacent, **P2** perf, **P3** clarity/style.

**Milestone-1 update (2026-06-13):** the **four P1 backlog items** (#1, #9, #10, #13) are now cleared — #1 and #10 **RESOLVED (MEASURED)**, #9 and #13 **RESOLVED-PARTIAL (DATA-GATED:** de-magicked + boundary-tested, operating values unchanged**)**. Each fix is pinned by a regression test that fails on the old behaviour (commits `fd32f094a`, `4a9f2bcf4`, `d672fa602`, `5193f6369`); workspace `--no-default-features` green, Python proof unchanged (bit-exact).

**Milestone-2 update (2026-06-13):** the **bench-first P2 perf subset** (#5, #6, #7, #8, #20) and the **three missing boundary tests** (#14, #16, #19) are now cleared — ~36 P2/P3 items remain deferred. PROOF discipline (§0): every perf item was **benched before being touched** — committed in `benches/dsp_perf_bench.rs` (criterion, this Windows box). Only **#20** proved hot and was optimized; **#5/#6/#7** are committed **MEASURED-NULLs** (benched, not hot, left as-is for clarity — exactly the §5.1 "already amortized" pattern); **#8** is **MEASUREMENT-ONLY** but its `eigenvalue`/BLAS backend won't build on this Windows host, so its µs cost must come from a Linux/BLAS box (recorded, not fabricated). Commits `e839fa8f1` (#20 fix), `02e5dd13a` (#14/#16/#19 tests), `aad9464f0` (benches). Workspace `--no-default-features` green; Python proof unchanged (#20 is bit-identical, off the proof path).

| # | Module | Finding | Pri | Why deferred |
|---|--------|---------|-----|--------------|
| 1 | cir.rs ~937 | `phase_variance` uses **linear** variance on **wrapped** angles (doc says "variance of phase angles") — spuriously inflates near ±π | P1 | **RESOLVED (`fd32f094a`) — metric MEASURED, threshold DATA-GATED.** Replaced with Mardia's circular variance V = 1 − R̄ ∈ **[0,1]**, invariant to the cluster's position on the circle (branch-cut artefact gone). Guard re-derived against the bounded metric via named const `GHOST_TAP_CIRCULAR_VARIANCE_MAX = 0.99` (fires only when R̄ ≤ 0.01 — essentially uniform phase). The **threshold value is DATA-GATED**: a clean single-path ramp also sweeps the circle, so V alone can't separate clean from unsanitized without labelled frames — the default is deliberately conservative (strictly more permissive at the wrap boundary than the buggy linear guard). Fails-on-old: `phase_variance_circular_not_fooled_by_branch_cut` (old linear variance > TAU on wrap-straddling phases while circular V≈0, guard no longer trips), `phase_variance_circular_is_bounded_and_extremal`. |
| 2 | calibration.rs ~311 | `subtract_in_place` had a vacuous `if active_input {ki} else {ki}` branch implying a full-FFT→bin remap that didn't exist | P3 | **Resolved here** (branch removed, sequential-convention documented to match the sibling `extract_first_stream`). Listed for visibility — behavior unchanged. |
| 3 | spectrogram.rs / bvp.rs | FFT planner built once-per-call (already amortized across frames) | P2 | Marginal vs the per-frame PSD site; cache if these become hot. |
| 4 | features.rs ~347 | Doppler FFT planner planned once per call, reused across subcarriers | P2 | Already amortized within the call. |
| 5 | multistatic.rs | `node_attention_weights` recomputes consensus/softmax each call; no SIMD | P2 | **MEASURED-NULL (`aad9464f0`) — benched, not hot, left as-is.** `multistatic_attention/weights`: **181 ns** (2 nodes) … **848 ns** (8 nodes) @ 56 subcarriers — sub-µs, no hot-path allocation. A precompute/SIMD rewrite buys nothing measurable at the realistic 2–8 node fan-in; the cosine/softmax cost is dwarfed by the surrounding fusion + per-frame FFT. Bench `multistatic_attention` in `dsp_perf_bench.rs`. |
| 6 | tomography.rs | ISTA L1 solver re-allocates voxel buffers per solve | P2 | **MEASURED-NULL (`aad9464f0`) — benched, not hot, left as-is.** A full 50-iteration `reconstruct` (256 voxels): **47.5 µs** (16 links) / **60.4 µs** (32 links). The two voxel buffers (`x`, `gradient`; ~4 KB) are already allocated *once* per `reconstruct()` and `.fill`-reused across iterations — the per-solve alloc is a negligible fraction of the O(iters·links·voxels) inner product. Reusing scratch across *calls* would force `reconstruct(&self)`→`&mut self` (API break) for no measurable gain. Bench `tomography_reconstruct`. |
| 7 | pose_tracker.rs | Kalman gain matrices reallocated per update | P2 | **MEASURED-NULL (`aad9464f0`) — benched, not hot, left as-is.** A Kalman predict+update cycle: **150 ns** (17 keypoints) / **2.82 µs** (170). The "gain matrices" (`s:[f32;3]`, `k:[[f32;3];6]`) are fixed-size **stack** arrays, *not* heap — there is no per-update allocation to reuse; the compiler keeps them in registers/stack. Bench `pose_kalman_update`. |
| 8 | field_model.rs | SVD recomputed on every perturbation extract | P2 | **MEASUREMENT-ONLY (`aad9464f0`) — BLAS-gated, not measurable on this host.** Correction: `extract_perturbation` does **not** recompute the SVD — it projects against the cached `modes` from `finalize_calibration`. The real per-call eigendecomposition is in the `eigenvalue`-feature `estimate_occupancy` (`cov.eigh()` on a 56×56 covariance, an O(n³)≈175k-flop symmetric eigensolve + O(n²·frames) covariance build, run per call). The bench (`dsp_perf_bench`'s `eig` module) is committed, but `openblas-src` **fails to build on this Windows box** ("Non-vcpkg builds are not supported on Windows" — the very reason the project gate runs `--no-default-features`), so a measured µs number must come from a Linux/BLAS host; **not estimated/fabricated here.** Incremental SVD remains a sized future project, not a micro-fix. |
| 9 | coherence.rs / coherence_gate.rs | Z-score thresholds are magic constants, untested at boundaries | P1 | **RESOLVED-PARTIAL (`5193f6369`) — DATA-GATED.** De-magicked `classify_drift` (`DRIFT_STABLE_SCORE=0.85`, `DRIFT_STEP_CHANGE_MAX_STALE=10`) and the `coherence_gate.rs` defaults (`DEFAULT_ACCEPT_THRESHOLD`/`…REJECT…`/`…MAX_STALE_FRAMES`/`…PREDICT_ONLY_NOISE`) into named, documented consts marked EMPIRICAL DEFAULT; added at/just-below/just-above boundary tests (`classify_drift_*_boundary`) + `*_consts_unchanged_from_literals`. **Operating values explicitly NOT changed** — defensible values still need labelled stable/drifting traces. The gate already exposed these via `GatePolicyConfig` (config seam). |
| 10 | longitudinal.rs | Welford update not numerically guarded for n=0 | P1 | **RESOLVED (`4a9f2bcf4`) — MEASURED.** The shared `WelfordStats` (`field_model.rs`, consumed by longitudinal.rs) `count < 2` guards already prevent the n=0 NaN / n=1 div0 / `(count−1)` underflow, but the boundary was untested. Added `welford_finite_at_n0_and_n1` (finite + documented 0.0 sentinel at n=0/n=1). Fails-on-old proof: removing the `sample_variance` guard makes the test panic with "attempt to subtract with overflow" at the `(count − 1)` underflow. |
| 11 | cross_room.rs | Fingerprint hash collisions unhandled | P2 | Low collision prob; needs design. |
| 12 | gesture.rs | `euclidean_distance` no length-mismatch guard | P3 | Caller-enforced; add `debug_assert`. |
| 13 | adversarial.rs | Gini/consistency thresholds are magic constants | P1 | **RESOLVED-PARTIAL (`d672fa602`) — DATA-GATED.** Lifted the bare literals in `check`/`check_consistency` (`FIELD_MODEL_GINI_VIOLATION=0.8`, `ENERGY_RATIO_HIGH_VIOLATION=2.0`, `ENERGY_RATIO_LOW_VIOLATION=0.1`, `CONSISTENCY_ACTIVE_FRACTION_OF_MEAN=0.1`, `SCORE_W_*`) into named, documented consts marked EMPIRICAL DEFAULT; added at/just-below/just-above boundary tests (`energy_ratio_high_boundary`, `energy_ratio_low_boundary`, `field_model_gini_boundary`, `consistency_active_fraction_boundary`) + `tuning_consts_unchanged_from_literals`. **Operating values explicitly NOT changed** — defensible values still need labelled spoofed/clean CSI (Wi-Spoof, §6.2/§7.3). Bumping a const fails a boundary test (verified). |
| 14 | cir.rs | `fft_operator` path changes the witness hash (documented) — no test that it's *numerically close* to dense | P2 | **RESOLVED (`02e5dd13a`) — tolerance test added.** `fft_operator_within_tolerance_of_dense_canonical56` pins the **full `Cir` output** of the FFT path within a *documented* relative tolerance of the dense path on the production **canonical-56** config across τ ∈ {20,50,90} ns: every tap within `1e-2·|dominant|`, identical `dominant_tap_idx`, `active_tap_count`, `ranging_valid`, `dominant_tap_ratio` within `1e-2`, `rms_delay_spread` within `1e-2` rel. A regression that lets the FFT path drift (scaling/Φ-column bug) now fails here instead of silently corrupting a downstream witness. Extends the existing HT20/single-τ `fft_estimate_matches_dense_dominant_tap`. |
| 15 | multistatic.rs | `cir_gate_coherence` only estimates the **first** node/channel; multi-node CIR consensus unused | P2 | Design item (which node's CIR is authoritative?). |
| 16 | phase_align.rs | Iterative LO offset estimation has no convergence cap test | P2 | **RESOLVED (`02e5dd13a`) — cap test added.** `refinement_terminates_at_iteration_cap_when_not_converging` forces non-convergence (`tolerance = 0.0`, unreachable since `max_update ≥ 0`) and asserts the loop runs **exactly `max_iterations`** then returns — proving the cap (not convergence) bounds the loop, so a non-converging input can never spin forever. Companion `refinement_converges_before_cap_on_easy_input` proves the cap is an upper bound, not the only exit. Internal-only refactor: `estimate_phase_offsets` still returns the identical offset vector; a `…_counted` core surfaces the iteration count for the test. |
| 17 | hampel.rs | Window edge handling at series boundaries | P3 | Cosmetic. |
| 18 | motion.rs | Threshold constants undocumented | P3 | Doc-only. |
| 19 | csi_ratio.rs | Division guard relies on `1e-12` epsilon; no test | P2 | **RESOLVED (`02e5dd13a`) — boundary test added.** Finding clarification: `csi_ratio.rs` implements the CSI *ratio model* as the **conjugate product** `H_i·conj(H_j)` (SpotFi/IndoTrack) — there is **no division**, hence no literal `1e-12` epsilon; the classic `H_i/H_j` ratio (which a `1e-12` guard protects) is deliberately avoided. `ratio_finite_at_and_below_1e_12_epsilon` pins the property the finding cares about: at and below the `1e-12` target magnitude (and at exact zero — where a division ratio is ±inf/NaN) the conjugate-product output is **finite**, exactly the conjugate product (bit-exact), collapses toward zero (the physically correct "no path" answer), and stays finite through `ratio_to_amplitude_phase`. |
| 20 | spectrogram.rs | `compute_multi_subcarrier_spectrogram` re-plans per subcarrier via `compute_spectrogram` | P2 | **MEASURED-HOT (`e839fa8f1`) — optimized, bit-identical.** Hoisted the FFT plan + window out of the per-subcarrier loop (new `compute_spectrogram_with_plan` core). **56-subcarrier** multi-spectrogram: **467.88 µs → 254.75 µs = 1.84×** (window 128); **627.27 µs → 448.39 µs = 1.40×** (window 256). The removed cost is the per-subcarrier `FftPlanner` re-plan (~1.86 µs/plan @ w128 × 56). Bit-identical (`multi_subcarrier_hoisted_plan_bit_identical`, `f64::to_bits` across all 4 windows × {power,magnitude}). The most likely real win predicted by the §7.4 intro — confirmed. (Relates to #3, which stays deferred: `spectrogram.rs`/`bvp.rs` single-signal callers already plan once-per-call.) |
| 21–45 | (assorted) | Remaining clarity/doc/magic-constant/missing-boundary-test findings across `ruvsense/*`, `features.rs`, `motion.rs` | P3 | Bulk-addressable in a dedicated "test-the-boundaries + de-magic-constant" follow-up; not high-leverage individually. |

> **Horizon-ledger one-liner.** Milestone-0 DONE: dead CIR gate (FIXED+proved), NaN/inf adversarial bypass (FIXED+proved), divide-by-(n−1) window trio (FIXED+proved), calibration dead-branch (FIXED), PSD FFT-planner cache (MEASURED), DTW band (MEASURED). **Milestone-1 DONE (2026-06-13): all four P1 backlog items cleared — circular phase variance #1 (RESOLVED/MEASURED metric, DATA-GATED threshold), Welford n=0 guard #10 (RESOLVED/MEASURED), threshold magic-constants #9 & #13 (RESOLVED-PARTIAL/DATA-GATED — de-magicked + boundary-tested, values unchanged).** **Milestone-2 DONE (2026-06-13): bench-first P2 perf subset + missing boundary tests cleared — spectrogram per-subcarrier FFT re-plan #20 (MEASURED-HOT, 1.40–1.84×, bit-identical); attention/tomography/Kalman #5/#6/#7 (MEASURED-NULL — benched, not hot, left as-is); field_model eigendecompose #8 (MEASUREMENT-ONLY, BLAS un-buildable on this Windows host, number deferred to a BLAS box, NOT fabricated); fft_operator tolerance #14, phase-align convergence-cap #16, csi-ratio epsilon #19 (RESOLVED, tests added).** DEFERRED to follow-up: the ~36 remaining P2/P3 findings in §7.4 — none silently dropped.

---

## 8. Consequences

- **Positive:** the ADR-134 CIR gate is alive for the first time in production; the adversarial detector can no longer be NaN-bypassed; three latent divide-by-zero NaN sources are gone; the per-frame PSD path and gesture DTW are measurably faster with bit-identical output; the SOTA landscape and a concrete LISTA-for-CIR roadmap are graded and recorded.
- **Negative / honest limits:** `canonical56()` models the canonical grid as a contiguous 56-tone band — a reasonable physical interpretation of a *resampled* grid, but not a literal hardware tone map; the CIR gate still uses only the first node's CIR (#15). The `phase_variance` **metric** is now correct (Mardia circular variance, Milestone-1 #1), so the branch-cut false-trip is gone — but its ghost-tap **threshold** (`GHOST_TAP_CIRCULAR_VARIANCE_MAX = 0.99`) is a conservative DATA-GATED default, not a calibrated operating point, and still awaits labelled sanitized/unsanitized frames to tune. Likewise the de-magicked coherence/adversarial thresholds (#9/#13) keep their pre-existing empirical values pending labelled calibration.
- **Neutral:** no public API removed; `with_cir_ht20()` kept (warned); files stay scoped; new bench is additive.
