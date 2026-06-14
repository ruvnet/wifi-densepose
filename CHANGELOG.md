# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- **ADR-157 Milestone-1 B4 - constant-time HMAC sync-beacon tag compare (`wifi-densepose-hardware`).** `AuthenticatedBeacon::verify` compared the 8-byte HMAC-SHA256 tag with `self.hmac_tag == expected`, which short-circuits on the first differing byte and leaks, through verification latency, how many leading bytes an attacker's forged tag matched - a byte-by-byte tag-recovery oracle (~256*N trials instead of 256^N). Replaced with a hand-rolled branch-free `constant_time_tag_eq` (XOR-accumulate every byte difference into a single `u8`, no early exit, `#[inline(never)]` + `core::hint::black_box` to stop the optimizer reintroducing a short-circuit or a non-constant-time `memcmp`). **No new dependency** - ADR-157 had deferred this only to avoid adding the `subtle` crate; a fixed 8-byte compare needs none. Grade MEASURED (constant-time *construction*; micro-timing on a noisy host is a smoke check only, gated `#[ignore]`). Pinned by `tag_compare_is_constant_time_shape` (equal/first-differ/last-differ/all-differ/length-mismatch + an end-to-end `verify()` last-byte tamper), proven to fail on a last-byte-skipping constant-time bug. ADR-157 §8 B4 -> RESOLVED.
- **ADR-080 open HIGH findings closed on the Rust `wifi-densepose-sensing-server` boundary (ADR-164 G11).** The QE sweep's three HIGH findings — XFF-spoofing bypass, leaked stack traces, JWT-in-URL (CWE-598) — were logged against the Python v1 API and never re-verified against the shipped Rust sensing-server; the HOMECORE/M7 sweep (ADR-161) covered `homecore-server`, not this crate.
  - **#2 leaked internal errors (the one live exposure) — FIXED.** Six handlers in `main.rs` serialized the internal error `Display` straight into the JSON response body: `edge_registry_endpoint` returned a panicked `spawn_blocking` `JoinError` (`"task … panicked"`) in a `500`, plus the raw upstream error in a `503`; `delete_model`/`delete_recording`/`start_recording` returned `std::io::Error` strings (OS detail / path); `calibration_start`/`calibration_stop` returned the `FieldModel` error chain. New `error_response` module logs the full detail **server-side only** (with a correlation id) and returns a generic body (`{"error":"internal_error","correlation_id":…}`) — no `panicked`, no file paths, no Debug chain. 5 module tests (a leak-substring guard proven to fail on the reverted old body) + the existing handler suite.
  - **#1 XFF-spoofing bypass — VERIFIED ABSENT, regression-pinned.** The sensing-server has no XFF-trusting control to bypass: there is no IP-based rate-limiter or IP-allowlist, and neither `bearer_auth` (token-only) nor `host_validation` (Host-header only) reads `X-Forwarded-For`/`X-Forwarded-Host` (no `forwarded`/`peer_addr`/`client_ip` anywhere in the crate). Added regression tests proving a spoofed `X-Forwarded-For` never flips an auth decision and a spoofed `X-Forwarded-Host` never bypasses the Host allowlist.
  - **#3 JWT-in-URL (CWE-598) — VERIFIED ABSENT, regression-pinned.** `require_bearer` reads the token only from the `Authorization` header; the WebSocket handlers take no token query param and the sole `Query` extractor (`EdgeRegistryParams`) is a non-secret `refresh` flag. Added a regression proving `?token=`/`?access_token=` in the URL never authenticates while the header path still does.

### Fixed
- **ADR-155 Milestone-1b — metric-definition unification, the §8 backlog subset (Goals A/B/C).** Closed the two §8 metric-integrity items; every change pinned by a test, graded MEASURED. The audit (Goal A) also surfaced findings the §1 table under-counted — recorded honestly in ADR-155 §8.1, not hidden. Workspace stays green; Python proof unchanged (metrics are not on the deterministic proof's signal path).
  - **Goal B — `test_metrics.rs` now validates the production metric, not a reimplementation.** The integration test previously asserted properties of its OWN local `compute_pck`/`compute_oks` (a test that can't catch a canonical-impl bug — both could be wrong the same way). Hoisted the canonical core (`pck_canonical`/`oks_canonical`/`canonical_torso_size`/sigmas/`bounding_box_diagonal`) into a new **un-gated** `metrics_core` module so the single definition is reachable under `cargo test --no-default-features` (the `metrics` module is `tch-backend`-gated); `metrics` re-exports it → still exactly ONE implementation. Rewrote the test to assert the production `pck_canonical`/`oks_canonical` equal **hand-computed** fixtures (`canonical_pck_matches_hand_computed_fixture` = 3/4 correct ⇒ 0.75; hip↔hip normalizer pin; zero-visible⇒0.0; OKS perfect⇒1.0; fake-Gold pin) plus a differential cross-check (`test_kernel_agrees_with_canonical`: an independent raw-threshold kernel must AGREE with canonical where torso==1.0). `wifi-densepose-train --no-default-features`: test_metrics **10→12**, 0 failed.
  - **Goal C — divergent live-server PCK/OKS relabelled so they're never conflated with canonical.** Goal C named `training_api.rs:804` (torso-HEIGHT PCK); the audit found that file is an **orphan (not `mod`-declared, does not compile)** and the **real** live `best_pck`/`best_oks` come from `trainer.rs` — a **raw, unnormalized** `pck_at_threshold` and an **`area=1.0` fake-Gold** `oks_map` (both MISSED by ADR-155 §1, both on the claim-inflating side, both serialized as bare "PCK@0.2"/"OKS"). Torso-height/raw math is load-bearing (pixel-space, different scale axis, no `ndarray`/train dep), so the honest fix is **relabel, not force-unify**: `training_api.rs` `compute_pck` → `compute_pck_torso_height` + field/log docs; `trainer.rs` kernels documented raw/fake-Gold; `main.rs` prints `pck_raw@0.2` / `oks_map(area=1.0 proxy)`. No wire-format field or `pub`-fn renames (no silent API break). Pinned by `torso_pck_is_labelled_distinctly_from_canonical` + `pck_at_threshold_is_raw_unnormalized_not_canonical`. `wifi-densepose-sensing-server --no-default-features`: lib **450→451**, 0 failed. True unification onto `pck_canonical`/`oks_canonical` remains a tracked ADR-155 §8 item.
- **Pre-existing `SketchBank::topk` heap inversion returned the FARTHEST sketches (found during ADR-156 §8 Pass-2 work).** The `n > k` partial-sort path in `wifi-densepose-ruvector/src/sketch.rs` used `BinaryHeap<Reverse<(dist,id)>>` (a min-heap) but its eviction logic treated the peek as the max, so it kept the k *farthest* sketches and returned them as "nearest." The shipped unit tests only exercised the `n ≤ k` fast path (≤ 3 entries), so the inversion shipped silently in ADR-084. Fixed to a plain max-heap. Pinned by `topk_heap_path_returns_nearest` (farthest-first insertion exposes it) and `tight_clusters_give_high_coverage_with_overfetch` (**measured 0.072 coverage on the old code** — effectively random — vs >0.99 fixed). Every ADR-084 top-K coverage number depends on the fixed path. MEASURED, not a no-op.
- **ADR-154 Milestone-1 — cleared the P1 deferred backlog in `wifi-densepose-signal` (§7.4 #1, #10; partial #9, #13).** Each fix pinned by a regression test that fails on the old behaviour; every claim graded MEASURED / DATA-GATED; no fabricated thresholds. Python proof unchanged (`f8e76f21…46f7a`, bit-exact — the CIR ghost-tap guard is not on the deterministic proof path).
  - **#1 (MEASURED metric / DATA-GATED threshold): circular phase variance.** `cir.rs::phase_variance` computed a *linear* sample variance over phase angles that wrap at ±π, so a tightly-clustered set straddling the branch cut reported spuriously HIGH dispersion — false-tripping the `> TAU` ghost-tap **guard** on real, tightly-clustered CIR taps. Replaced with Mardia's **circular variance** V = 1 − R̄, bounded **[0,1]** and invariant to where the cluster sits on the circle. The old TAU-scaled threshold is meaningless on [0,1]; re-derived against a named const `GHOST_TAP_CIRCULAR_VARIANCE_MAX = 0.99` (fires only when R̄ ≤ 0.01 — essentially uniform phase). The **metric is MEASURED**; the **threshold value is DATA-GATED** (a clean single-path ramp also sweeps the circle, so V alone can't separate clean from unsanitized without labelled frames — the default is deliberately conservative, strictly more permissive at the wrap boundary than the buggy linear guard). Fails-on-old: `phase_variance_circular_not_fooled_by_branch_cut` (old linear variance > TAU on wrap-straddling phases while circular V≈0, guard no longer trips) + `phase_variance_circular_is_bounded_and_extremal` (V∈[0,1], V≈0 identical, V≈1 uniform).
  - **#10 (MEASURED): Welford n=0/n=1 finiteness guard pinned.** The shared `WelfordStats` (`field_model.rs`) `count < 2` guards keep `variance`/`sample_variance`/`std_dev`/`z_score` finite at the boundaries, but the n=0 case was untested (same family as the §4 divide-by-(n−1) trio). Added `welford_finite_at_n0_and_n1` — finite + documented-sentinel (0.0) at n=0/n=1. Fails-on-old proof: removing the `sample_variance` guard makes the test panic with "attempt to subtract with overflow" at the `(count − 1)` underflow (guard restored).
  - **#9, #13 (DATA-GATED): de-magicked thresholds + boundary tests (values UNCHANGED).** Lifted the bare detection literals in `adversarial.rs` (`check`/`check_consistency`: Gini 0.8, energy ratios 2.0/0.1, consistency 0.1·mean, score weights), `coherence.rs::classify_drift` (0.85, 10) and `coherence_gate.rs` defaults (0.85/0.5/200/3.0) into named, documented consts marked EMPIRICAL DEFAULT pending labelled calibration. Added characterization/boundary tests pinning each decision at/just-below/just-above its threshold (`energy_ratio_high_boundary`, `energy_ratio_low_boundary`, `field_model_gini_boundary`, `consistency_active_fraction_boundary`, `classify_drift_*_boundary`, `*_consts_unchanged_from_literals`) so a future labelled-data retune is a visible, tested change. The operating **values were not changed**; the de-magicking + tests are MEASURED, the values stay DATA-GATED.
- **Multistatic fusion guard was too tight for real TDM hardware (#1031).** `MultistaticConfig::default().guard_interval_us` was 5,000 µs (5 ms) with a comment claiming "well within the 50 ms TDMA cycle" — but on a real N-slot TDM schedule node `k` transmits in slot `k`, so two nodes are separated by the *slot offset*, not clock jitter. A real 2-node mesh (slots 0/1) measured an **18,194 µs** spread, so every real frame set exceeded the 5 ms guard and `fuse()` silently fell back to per-node sum/dedup — multistatic fusion never actually ran on hardware. Raised the default hard guard to **60 ms** (a full 50 ms TDMA cycle + 20% jitter headroom, derived from the slot model and documented in the field doc) and the soft guard to **20 ms** (just above the observed 18.2 ms 2-slot spread, so a normal cycle fuses cleanly with no privacy demotion). Added `MultistaticConfig::for_tdm_schedule(total_slots, slot_duration_us)` to derive the guard from a deployment's exact schedule, and a `WDP_TDM_SLOTS`+`WDP_TDM_SLOT_US` env seam in sensing-server. The honest per-node fallback remains for genuinely-mismatched frames — now the exception, not the default. Pinned by `fuse_real_tdm_spread_18194us_fuses_with_default_guard` (fails on the old 5 ms default) + `configurable_guard_rejects_too_large_spread` (guard still rejects a spread beyond one cycle).
- **Published HuggingFace model was unloadable — RVF format mismatch (#894).** The `ProgressiveLoader` rejected the published `ruvnet/wifi-densepose-pretrained` model with the opaque `invalid magic at offset 0: expected 0x52564653 (RVFS), got 0x77455735`, then silently fell back to signal heuristics (the "10 persons for 1" garbage reporters saw). The HF repo ships `model.safetensors`, `model-q{2,4,8}.bin` (magic `0x77455735` = "5WEw"), and `model.rvf.jsonl` — none carry the binary-RVF magic. New `model_format` module **auto-detects** RVFS / safetensors / HF-quant-bin / JSONL by magic+name, returns a **typed actionable** `ModelLoadError` (lists accepted formats + the one-command convert path — never the opaque magic), and **converts** `model.safetensors` / `model.rvf.jsonl` → RVF in-memory so the published full-precision model now loads via `--model`. A `--convert-model <in> --convert-out <out>` CLI subcommand gives a one-command offline path; the silent heuristics fallback is now a loud, actionable error. **Honest scope:** the converter wires the format/load path (safetensors F32 tensors → RVF weight segment, manifest written, Layer A/B/C all succeed, weights round-trip) — it does **not** claim end-to-end pose accuracy, since the HF pose-decoder architecture differs from this crate's inference head (still data-gated in #894). Quantized `.bin` blobs are rejected with a typed error pointing at the safetensors path. Pinned by `safetensors_converts_and_loads` + `hf_quant_classifies_to_actionable_error` (both fail on the old opaque-magic path).

### Changed
- **ADR-157 Milestone-1 §5 #4 - native `wlanapi.dll` multi-BSSID throughput MEASURED on real hardware (`wifi-densepose-wifiscan`).** The ADR's prior status ("asserted but NOT implemented; live scanner is the ~2 Hz netsh shim") is now stale: `wlanapi_native.rs` already implements the real `WlanOpenHandle` -> `WlanEnumInterfaces` -> `WlanGetNetworkBssList` -> `WlanFreeMemory`/`WlanCloseHandle` FFI and `WlanApiScanner` already wires it native-first with a netsh fallback. This milestone **measured it on this box** (Intel Wi-Fi 7 BE201 320MHz, 2026-06-13): a new `benchmark_backend(backend, window)` drives each backend over the same fixed 10 s wall-clock window so netsh is timed independently (the prior `benchmark()` picked native-first and never measured netsh on a Windows box where native works). **MEASURED: native 21.42 Hz vs netsh 3.84 Hz = 5.57x** (mean 5.0 BSSIDs/scan, both paths); a separate native-only run measured 18.0 Hz. Native genuinely beats netsh - this is a real positive result, not a fabricated "10x". 50 back-to-back native scans completed 50/50 with no handle leak/degradation. Live-WLAN tests (`measure_native_vs_netsh_throughput`, `native_scans_dont_leak_handles`, `measure_native_scan_rate`) are `#[ignore]` for CI but were RUN here; `native_scan_runs_real_ffi_on_windows` is a non-ignored schema-valid pin. ADR-157 §5 #4 + §8 -> MEASURED (was ACCEPTED-FUTURE / CLAIMED-unmeasured).
- **Mesh partition risk now demotes the privacy class and is witnessed (ADR-032).** The dynamic min-cut guard's `at_risk` signal was advisory-only (it fed the recalibration advisor). It now also contributes to the ADR-141 privacy demotion alongside fusion- and array-level contradictions: a mesh close to partitioning makes the fused belief less trustworthy, so the cycle emits at a more restricted class (monotonic — information only removed). Because `effective_class` feeds the BLAKE3 witness, a fragmenting array now shifts the witness — partition risk is auditable, not just logged. The mesh computation moved ahead of the demotion step in `process_cycle`; new `mesh_guard_mut()` exposes risk-threshold tuning. Test proves a forced-risk 3-node cycle demotes PrivateHome Anonymous→Restricted and shifts the witness vs a clean *same-topology* baseline (the only delta between the two cycles is the forced risk).

### Added
- **ADR-155 Milestone-2 — cleared the host-verifiable subset of the §8 P3 backlog in `wifi-densepose-train` (+ the pure-Rust `rf_encoder.rs`/`densepose.rs` the §3/§4 items named).** Mirrors the ADR-154 M3 cleanup discipline. **Honest enumeration first (grep, not the ADR's "~40" estimate):** the actual non-tch train/nn surface is smaller — **7 de-magicked (const + `*_consts_unchanged_from_literals` pin == prior literal), 9 boundary/characterization tests, 1 added input guard (`rf_encoder::LinearHead::try_new`) + test, 2 doc-only fixes, 1 perf item bench-first → MEASURED-INCONCLUSIVE (not shipped)**. **This is cleanup — no operating value or behaviour changed:** each lifted literal is bit-identical to its prior value, each boundary test pins CURRENT behaviour. De-magicked: `metrics_core.rs` (`VISIBILITY_THRESHOLD`/`MIN_REFERENCE_EXTENT`/`OKS_FALLBACK_SIGMA`), `ruview_metrics.rs` (`NUM_KEYPOINTS`/`VISIBILITY_THRESHOLD`/`PCK_THRESHOLD`/`MIN_BBOX_DIAG`/`MIN_DURATION_MINUTES`), `subcarrier.rs` (6 `SPARSE_*` consts), `eval.rs` (`MIN_POSITIVE_MPJPE`), `domain.rs` (`LAYER_NORM_EPS`), `virtual_aug.rs` (`BOX_MULLER_U1_FLOOR`/`MIN_ROOM_SCALE`), `rf_encoder.rs` (`SOFTPLUS_LINEAR_THRESHOLD`). **§3 `rf_encoder.rs`:** added a pure-Rust fallible `LinearHead::try_new` → typed `RfHeadError` so untrusted/deserialized checkpoint weights can be shape-validated without the `new()` panic (`new` unchanged; additive). **§4 native-conv:** `densepose.rs::apply_conv_layer` (pure-Rust naive loop) was benched (committed `benches/native_conv_bench.rs`); a bit-identical range-clamped rewrite measured ~35% faster on padding-heavy small-channel maps but ~3% *slower* on channel-heavy maps, all inside a ±20% host-noise floor — **MEASURED-INCONCLUSIVE, so NOT shipped** (no fabricated number), characterized by `native_conv_matches_reference` and honestly deferred. **Skipped honestly (not-real / already-handled):** `ablation.rs` (NaN-sort + boundaries already fixed/tested in M1), `signal_features.rs` (consts already named, n=0 tested), `mae.rs` (no bare guard literals). `wifi-densepose-train --no-default-features`: **303 passed** (was 288, +15), 0 failed; `wifi-densepose-nn --no-default-features` lib: **38** (was 35, +3). Workspace `--no-default-features`: GREEN (single clean run). Python proof **VERDICT: PASS**, hash **`f8e76f21…46f7a` UNCHANGED, bit-exact** (asserted — the metrics path is off the deterministic signal proof path). **Remaining §8 backlog stays deferred-not-dropped:** GraphPose-Fi / ONNX-INT4 / CSI-JEPA (data/model-gated), ONNX read-lock (upstream `ort`-gated), tch-gated panic sites in `proof.rs`/`trainer.rs`/`model.rs` + `metrics.rs` `*_v2` dead-code (tch-gated — need a libtorch host). **The non-tch-verifiable subset of §8 is now cleared.**
- **ADR-154 Milestone-3 — cleared the §7.4 row #21–45 P3 backlog in `wifi-densepose-signal` (the lumped "remaining clarity/doc/magic-constant/missing-boundary-test findings across `ruvsense/*`, `features.rs`, `motion.rs`").** Honest enumeration first (grep, not the ADR's estimate): the lumped row was **~25 findings → 22 real, de-magicked across 11 modules; 6 boundary/characterization tests added; ~4 doc-only; the rest were already-handled or not-real and are reported as such** (the "row #21–45" count was an estimate — there were not 25 *distinct* magic constants left after M0–M2). **This is cleanup — no operating value or behaviour changed:** every de-magicked literal becomes a named, documented EMPIRICAL-DEFAULT const that **equals the prior literal exactly** (each module ships a `*_consts_unchanged_from_literals` pin test), and every boundary test pins **current** behaviour so a future retune is a visible, tested change. Modules touched: `motion.rs` (#18, fusion weights/normalization/adaptive-threshold consts + 5 tests), `gesture.rs` (#12, `euclidean_distance` length-mismatch `debug_assert` documenting the silent-truncation contract + DTW n=0/m=0 boundary), `longitudinal.rs` (drift thresholds 7-day/2σ/3-day/7-day/EMA + day-6/7 + zero-vector cosine), `cross_room.rs`/`multiband.rs`/`intention.rs`/`hampel.rs` (division-guard epsilons + zero-norm/zero-variance/zero-MAD boundary + `half_window==0` error path), `rf_slam.rs` (`NS_PER_DAY` + fixed-map defaults + zero-span guard), `attractor_drift.rs` (buffer/recent-window consts + documented the implicit `recent.len()≥1` divide-safety + `min_observations` off-by-one boundary), `coherence.rs` (#9 completion — variance-floor + default-decay), `calibration.rs` (#2 — `DEFAULT_MIN_FRAMES` deduped across 4 tier constructors + motion/subtract thresholds), `fusion_quality.rs` (contradiction penalty/bounds + n=0 identity), `temporal_gesture.rs` (confidence epsilon + quantization scale). **A "magic" the agents flagged that was NOT real:** an `attractor_drift.rs:301` "divide-by-zero" is unreachable (the `count < min_observations` guard guarantees `recent.len()≥1`) — documented + boundary-tested rather than guarded, per the no-behaviour-change rule. Signal crate lib `--no-default-features`: **476 passed, 0 failed, 1 ignored**; `--no-default-features --features cir`: **476 passed, 0 failed** (plain `--features cir` is unbuildable on this Windows host — the default `eigenvalue` feature pulls `openblas-src`, the same BLAS gate documented in M2 #8). Workspace `--no-default-features`: **3,275 / 0 failed** (single clean run). Python proof **VERDICT: PASS**, hash **`f8e76f21…46f7a` UNCHANGED, bit-exact** (asserted explicitly — these modules are off the deterministic PSD/Doppler proof path, and the de-magicked consts are bit-identical regardless). **This clears ADR-154's §7.4 deferred backlog to zero across M0–M3.**
- **ADR-154 Milestone-2 — bench-first P2 perf subset + missing boundary tests (`wifi-densepose-signal`, §7.4 #5/#6/#7/#8/#14/#16/#19/#20).** PROOF discipline (ADR-154 §0): every perf item was **benched before being touched** (new committed `benches/dsp_perf_bench.rs`, criterion, this Windows box); only the one item the bench proved hot was optimized, the rest are committed MEASURED-NULLs — a benched null is the proof the micro-opt was unnecessary, the §5.1 "already amortized" pattern. Every behaviour-changing edit is pinned bit-identical (or documented-tolerance). Signal crate lib `--no-default-features`: **447 passed, 0 failed, 1 ignored**; `--features cir`: **447 passed, 0 failed**.
  - **#20 MEASURED-HOT, optimized (bit-identical).** `compute_multi_subcarrier_spectrogram` re-planned a fresh `FftPlanner` for *every* subcarrier (via `compute_spectrogram`). Hoisted the plan + window out of the per-subcarrier loop (new `compute_spectrogram_with_plan` core; `compute_spectrogram` delegates, unchanged). **56-subcarrier: 467.88 µs → 254.75 µs = 1.84×** (window 128); **627.27 µs → 448.39 µs = 1.40×** (window 256). Bit-identical via `multi_subcarrier_hoisted_plan_bit_identical` (`f64::to_bits` of every value across all 4 window functions × {power,magnitude}). The §7.4 intro's predicted "most likely real win" — confirmed.
  - **#5 / #6 / #7 MEASURED-NULL, left as-is.** `node_attention_weights` 181 ns (2 nodes)…848 ns (8) — sub-µs, no hot-path alloc. `tomography reconstruct` (full 50-iter ISTA, 256 voxels) 47.5 µs (16 links) / 60.4 µs (32) — the 2 voxel buffers are already alloc-once + `.fill`-reused, negligible vs O(iters·links·voxels). `pose_tracker` Kalman cycle 150 ns (17 keypoints) / 2.82 µs (170) — the "gain matrices" are fixed-size **stack** arrays, zero heap to reuse. No rewrite shipped; the committed benches prove each is not hot.
  - **#8 MEASUREMENT-ONLY, BLAS-gated (number deferred, not fabricated).** Correction to the finding: `extract_perturbation` does **not** recompute the SVD (it projects against cached `finalize_calibration` modes); the real per-call eigendecomposition is the `eigenvalue`-feature `estimate_occupancy` (`cov.eigh()` on a 56×56 covariance). The `eig` bench is committed but `openblas-src` won't build on this Windows host ("Non-vcpkg builds are not supported on Windows" — the exact reason the project gate runs `--no-default-features`), so its µs cost must come from a Linux/BLAS box. Recorded, not estimated. Incremental SVD stays a sized future item.
  - **#14 / #16 / #19 RESOLVED — tests added (no behaviour change).** `fft_operator_within_tolerance_of_dense_canonical56` pins the full `Cir` output of the opt-in FFT path within a documented relative tolerance of the dense path on the production canonical-56 config (τ ∈ {20,50,90} ns) — it changes the witness hash, so it must be provably *close*, not silently divergent. `refinement_terminates_at_iteration_cap_when_not_converging` (+ convergent companion) proves the LO-offset refinement terminates at exactly `max_iterations` on a non-converging input (cap, not convergence, bounds the loop; internal `…_counted` refactor returns the identical offsets). `ratio_finite_at_and_below_1e_12_epsilon` pins that the conjugate-product CSI-ratio (no division → no `1e-12` divide-guard needed) is finite + bit-exact at/below the epsilon boundary and at exact zero (where a naive `H_i/H_j` ratio is ±inf/NaN).
- **ADR-156 §11 Milestone-2: RaBitQ unbiased distance estimator — IMPLEMENTED & MEASURED (RESOLVED-NEGATIVE on the strict-K bar).** Closes the §10.5 / §8 backlog "full RaBitQ residual-distance estimator (not just a uniform scalar code)" item — the **real** Gao & Long (SIGMOD 2024) contribution, not just sign bits. New `wifi-densepose-ruvector/src/estimator.rs`: `EstimatorSketch` carries the Pass-2 sign code (over the padded FHT length `D = next_pow2(dim)`) **plus 8 B/vec side info** (`residual_norm` + `x_dot_o = ⟨x̄, o'⟩`, 2× f32); `DistanceEstimator` computes the **unbiased** estimate `⟨o',q'⟩ ≈ ⟨x̄,q'⟩ / x_dot_o` (the random rotation makes the 1-bit code's quantization error orthogonal-in-expectation to the query, paper `O(1/√D)` bound); `EstimatorBank::topk_estimated_cosine` reranks the candidate set by the estimate instead of raw Hamming. **Zero-centroid simplification (`c = 0`) stated honestly** — the paper-faithful per-cluster centroid path (`from_embedding_centred` / `EstimatorBank::with_centroid`) is also built so the simplification is a measured choice (no centroid coverage number is reported against the cosine ground truth, because cosine-of-residual ≠ cosine-of-raw would be a metric mismatch). **Purely additive + backward-compatible** — new types only; Pass-1 `Sketch` / Pass-2 `SketchBank` / `WireSketch` wire format unchanged; all external callers (`event_log.rs`, `signal/longitudinal.rs`, `sensing-server`) use Pass-1 and are unaffected. **MEASURED strict-K coverage** (same fixture/seeds as §10: dim=128 N=2048 K=8, 64 clusters, noise=0.35, 128 queries, cosine ground truth): the estimator lifts the strict `candidate_k=K` bar **46.39% (Pass-2 sign) → 49.71% (estimator, cosine rerank)** — a real **+3.3 pp** lift, **still ~40 pp short of the ADR-084 ≥90% strict bar.** At over-fetch the estimator beats sign (candidate_k=24: **95.12%** vs 91.60%). **Honest verdict — RESOLVED-NEGATIVE: the unbiased estimator does NOT clear the strict-K 90% bar on this distribution** (the binding constraint is the 1-bit code's information ceiling, not estimator variance); the bar is still met only via the over-fetch "candidate set" pattern ADR-084 specifies, though the estimator **reduces the over-fetch factor** needed. A published negative, reported as such — no benchmark tuned to manufacture a pass. Unbiasedness pinned by `estimator_unbiased_on_fixture` (Monte-Carlo mean over 4000 rotation seeds → true inner product within tolerance); not-worse-than-sign pinned by `estimator_rerank_not_worse_than_sign`; determinism by `estimator_is_deterministic`. +12 tests in the crate (119→131). Workspace **3,228 / 0 failed** (`cargo test --workspace --no-default-features`, 162 test binaries, single clean run), Python proof **VERDICT: PASS** (`f8e76f21…46f7a`, unchanged — estimator is not on the proof's signal path). Full numbers + reproduce commands in ADR-156 §11 / ADR-084 "Pass 2b".
- **ADR-156 §8 Milestone-1: RaBitQ Pass-2 randomized rotation + multi-bit experiment — IMPLEMENTED & MEASURED (RESOLVED-PARTIAL).** Closes the §8 "Multi-bit / Extended RaBitQ" backlog item. New `wifi-densepose-ruvector/src/rotation.rs`: a deterministic randomized orthogonal rotation `R = H·D` — **Fast Hadamard Transform** (`O(d log d)`, in-place, `1/√m`-normalized so norm-preserving) + seeded ±1 sign flips (SplitMix64 from a stored `u64` seed; identical at index + query time). Chosen over a dense `d×d` matrix (`O(d²)`, infeasible at the 65,535-d the wire format provisions for); pads to `next_pow2(d)`. Additive, backward-compatible API (`Sketch::from_embedding_rotated`, `SketchBank::with_rotation` + `insert_embedding`/`topk_embedding`/`novelty_embedding`); Pass-1 and the wire format are byte-for-byte unchanged. New `coverage.rs` single-source-of-truth top-K coverage harness (anisotropic planted-cluster fixture, cosine ground truth) backs both a `#[test]` report and the `sketch_bench` coverage table. **MEASURED (dim=128 N=2048 K=8, 64 clusters, noise=0.35, 128 queries, seeded):** at the strict `candidate_k=K` bar, rotation lifts coverage **36.13% → 46.39%**; Pass-2 reaches the **ADR-084 ≥90% bar at candidate_k=24 (~3× over-fetch)**; multi-bit Pass-3 reaches 54%/67%/74% at 2/3/4-bit (strict bar). **Honest verdict: neither rotation nor ≤4-bit multi-bit clears the strict-K 90% bar on this distribution — the bar is met only via the over-fetch "candidate set" pattern ADR-084 specifies.** No benchmark was tuned to manufacture a pass; the strict-bar gap is documented (ADR-156 §10, ADR-084 "Pass 2" section). +19 tests in the crate (100→119), workspace **3,225 / 0 failed**, Python proof VERDICT: PASS (`f8e76f21…`, unchanged — sketch is not on the proof's signal path).
- **Beyond-SOTA `v2/crates/` sweep (ADR-154–158) + full stub-implementation push — every claim MEASURED or graded.** A 5-milestone review/optimize/secure/benchmark/validate sweep, then a verified-audit-driven push to replace every production stub with real, tested logic (no labels, no placeholders). Each fix is pinned by a test that fails on the old code; every number ships with a reproduce command. Workspace: **3,122 tests / 0 failed** (`cargo test --workspace --no-default-features`), Python proof **VERDICT: PASS** (bit-exact).
  - **ADR-154 Signal/DSP** — revived a dead ADR-134 CIR coherence gate (canonical-56 vs ht20 mismatch meant it never ran in production: 8/8 Err → 8/8 Ok); NaN-bypass + window div0 guards; PSD FFT-planner cache (**2.0–3.1×**) + honored DTW band (**2.4–4.1×**).
  - **ADR-155 NN/Training** — unified 7 divergent PCK/OKS metric definitions into one canonical torso-normalized source (fixed two claim-inflating bugs: zero-visible PCK 1.0→0.0, OKS fake-Gold); leak-free subject-disjoint MM-Fi split + injected-leak detector; rapid_adapt replaced fake gradients with real finite-difference; proof.rs gained a min-decrease margin + committed-hash requirement; zero-copy ORT input (**1.48×**).
  - **ADR-156 RuVector/Fusion** — closed crafted-input DoS panics (triangulation/heartbeat); honest dimensionless GDOP = √(trace(G⁻¹)) replacing an RMSE mislabel; canonical wrapped angular distance; fuse() double-clone removed (**~2.17×** marshalling). SOTA graded: SymphonyQG (CLAIMED), multi-bit RaBitQ (near-term), GraphPose-Fi (data-gated).
  - **ADR-157 Hardware/Sensing** — `Vec::remove(0)` O(n²) sliding windows → `VecDeque`; breathing partial-weight renormalization; IIR low-sample-rate divergence clamp. Centerpiece: a MEASURED **negative-results** audit showing the layer (802.11bf model, parsers, calibration) was already hardened — cited file:line, NO-ACTION.
  - **ADR-158 MAT/world-model** — **unified two divergent triage engines** (the confidence-gated result was computed then discarded; gate==record now); **killed survivor count-inflation** (real RSSI localization + vitals-signature dedup, MEASURED 3→1); real ESP32/UDP/PCAP CSI ingest with honest typed `HardwareUnavailable`/`UnsupportedAdapter` errors for hardware-gated adapters (Intel5300/Atheros/PicoScenes — never fabricated CSI); real parabolic peak interpolation; real GDOP.
  - **Soul Signature §3.6 matcher made real (`wifi-densepose-bfld`, issue #1021).** An external audit correctly found person-identification was spec-only behind a no-op `NullOracle`. Now a real per-channel weighted-cosine matcher + `EnrolledMatcher: SoulMatchOracle` (364 tests). MEASURED: same-person 1.0000 vs cross-person 0.8088; and the audit's own claim proven — on WiFi-only cardiac+respiratory channels alone two people are **not separable** (gap 0.0005). Named identity is honestly **data-gated** on the AETHER/body-resonance channel being fed by a real enrollment; no working-named-identity claim is made.
  - **OccWorld real forward pass** — replaced `Tensor::randn` encoder/decoder stubs (which emitted trajectory priors from pure noise) with a real deterministic conv VQ-VAE forward pass (input-dependent, proven by tests that fail on the old randn) + a `weights_trained` honesty flag (false until a real checkpoint loads); pointcloud `to_gaussian_splats` 9→2 passes (**1.24×** MEASURED).
  - **Native multi-BSSID `wlanapi.dll` FFI** (`wifi-densepose-wifiscan`) — real `WlanOpenHandle`/`WlanEnumInterfaces`/`WlanGetNetworkBssList`, **MEASURED 9.74 Hz** on Windows (vs netsh ~2 Hz; no fabricated "10×"), typed `Unsupported` off-Windows. Real Matter 1.3 manual-pairing-code field-packing (canonical 34970112332, lossless decode) replacing a lossy-modulo placeholder.
  - **HOMECORE assistant** — real `LocalRunner` response path, real semantic intent recognizer (exact in-memory cosine k-NN; MEASURED 0.855 match / 0.106 no-match), real SQL state text-search — three always-empty stubs removed.
- **ADR-152 WiFi-Pose SOTA 2026 intake — verified external benchmark + four Rust integrations.** A 22-source adversarially-verified survey of the 2025–2026 WiFi-sensing SOTA, with every adopted number reproduced or graded before integration:
  - **WiFlow-STD (DY2434) reproduction (`benchmarks/wiflow-std/`)** — the external "97.25% PCK@20, 2.23M params" claim audited end-to-end: the **shipped checkpoint is REFUTED** (0.08% PCK@20 — wrong keypoint normalization, predates the published code), the released code does not run as published (6 documented defects, incl. an import that fails and an unreachable test phase), and the released dataset's final 13 files are corrupted (9,072 windows of NaN + float32-max garbage that NaN-poisons fp16 BatchNorm training). After repairing both, retraining with upstream defaults on an RTX 5080 reproduced **96.09% PCK@20 (full test) / 96.61% (corruption-free)** — claims graded MEASURED-EQUIVALENT; params (2,225,042) and FLOPs (~0.055 G) verified exactly. Full forensics in `benchmarks/wiflow-std/RESULTS.md`.
  - **`GeometryEmbedding` (ADR-152 §2.1.2, `wifi-densepose-calibration`)** — 32-slot permutation-invariant, NaN-proof featurization of the §2.1.1 `NodeGeometry` records (centroid/spread, measured-first pairwise distances, circular azimuth stats, covariance-eigenvalue geometric diversity, per-node flags), schema-versioned for the ADR-151 P6 LoRA heads; derived `SpecialistBank::geometry_embedding()` accessor. The PerceptAlign "coordinate overfitting" defense, transplanted to per-room banks.
  - **MAE pretraining recipe (ADR-152 §2.3, `wifi-densepose-train/src/mae.rs`)** — `MaePretrainConfig` pinning the UNSW-measured recipe (80% masking, (30,3) patches) with pure-Rust patchify/random-mask (exact counts, seed-deterministic, error-not-truncate divisibility, NaN rejection), property-tested; the consumption seam for the future ADR-150 ViT-Small encoder.
  - **`WiFlowStdModel` Rust port (`wifi-densepose-train/src/wiflow_std/`)** — tch-gated idiomatic port of the verified spatio-temporal-decoupled architecture (grouped causal TCN → asymmetric conv stack → dual axial attention); ungated param formula asserted equal to the reference 2,225,042; 15/17-keypoint variants share weights (enables the ADR-152 §2.2(b) ESP32 fine-tune).
  - **RuVector vendor sync + §2.6 opportunity survey** — vendor at `a083bd77f`; graded ADOPT/EVALUATE/WATCH table; crates.io bumps applied (mincut/solver 2.0.6, attention 2.1.0, gnn 2.2.0; RUSTSEC #504 audit: no pinned crate affected); top WATCH: unpublished `ruvector-graph-condense` differentiable min-cut for trainable subcarrier grouping.
- **ADR-153 IEEE 802.11bf-2025 forward-compatibility protocol model (`wifi-densepose-hardware/src/ieee80211bf/`)** — typed WLAN-sensing procedures (measurement setup/instance/report, SBP, termination) with `SpecProfile` version gates, `SensingCapabilities` negotiation, and **required** `ConsentMode` governance metadata on every setup; deterministic session FSM with rejection/timeout paths; `SensingTransport` seam with `SimTransport` and an `OpportunisticCsiBridge` mapping live ESP32 CSI batches into standardized report shape (a future chipset adapter replaces the bridge without touching RuvSense consumers). Not a certified implementation — simulation-tested protocol surface; OTA binding lands when silicon does. 19 acceptance tests.
- **Dynamic min-cut mesh partition guard in the streaming engine (`mesh_guard`).** Maintains a `ruvector-mincut` exact min-cut over the live mesh coupling graph (nodes = sensing nodes, coupling = product of fusion attention weights), surfacing per cycle: the global **cut value** (how close the array is to splitting — a structural measure per-node heuristics miss), the **weak side** (which specific nodes would partition: failure/jamming triage feeding ADR-032 posture), and an **at-risk flag** that counts as a structural event for the drift→recalibration advisor. Surfaced as `TrustedOutput::mesh`. **Measured cost policy** (criterion, 12-node mesh): weights are quantized (1/64; a *nonzero* coupling below one quantum saturates to quantum 1 so quantization never erases a live coupling — without the floor, balanced meshes of ≥ 65 nodes had every ~1/n coupling erased and sat permanently "at risk") and updates change-gated, so the steady-state cycle does zero graph work (~7.3 µs, ~23× cheaper than building); on any real change a full exact rebuild (~171 µs) is used because one `DynamicMinCut` delete+insert measured ~240 µs — the incremental machinery's overhead targets much larger graphs, so rebuild-on-change is the measured optimum at mesh scale (one-edge case −28% after the policy switch). Degenerate cases fail toward risk: a node with zero coupling is reported as already partitioned (cut 0). 9 mesh-guard tests + an engine-level wiring test; full `process_cycle` with the guard: ~33 µs for 4 nodes (50 ms budget).
- **Opt-in FFT operator for the CIR ISTA solver (8–14× measured).** Φ is a sub-DFT, so each ISTA mat-vec can run as one length-G FFT (O(G log G)) instead of a dense O(K·G) product. New `CirConfig::fft_operator` (default **false** — the dense path stays the bit-exact witness default; the FFT evaluates the same sums in a different order, so enabling it shifts float results and requires regenerating any pinned witness). `FftOperator` (rustfft, planned once at construction, scratch reused across the ISTA loop) dispatches inside `ista_solve`; warm-start/Lipschitz stay dense at construction. Measured (criterion, same run): ht20 2.22 ms → 265 µs (**8.4×**), ht40 10.26 ms → 717 µs (**14.3×**); the real HE40 grid (K=484, G=1452) scales further. 3 new tests: FFT↔dense matvec equivalence to float tolerance (ht20 + he40 grids), end-to-end dominant-tap agreement on a single-path frame, and all default configs keep FFT off. New `cir_estimate_fft` bench group.
- **Per-room adapter provenance + drift→recalibration advisor in the streaming engine.** Closes the trust-chain gap where an ~11 KB per-room LoRA adapter (ADR-150 §3.4) could silently change inference without the witness noticing. `StreamingEngine::set_room_adapter(AdapterInfo)` pins the adapter's content-derived id into provenance `model_version` (`rfenc-v1+adapter:<id>`) — and therefore into the BLAKE3 witness — so swapping or clearing adapter weights always shifts the witness (engine test proves base → adapter → other-adapter → cleared all witness differently, and cleared == base). New `RecalibrationAdvisor` recommends re-running the ADR-135 baseline / refitting the adapter on sustained low fusion coherence (streak threshold, default 60 cycles ≈ 3 s at 20 Hz) or an ADR-142 change-point; surfaced as `TrustedOutput::recalibration_recommended` and recorded on the sensing-server's `EngineBridge` alongside the witness. Bridge plumbing: `EngineBridge::{set_room_adapter, clear_room_adapter}` + live-path test that the adapter id flows into the live witness. *Scope note: this is the deployable provenance/trigger half of the "retrained model" roadmap item — fitting the adapter itself runs in the existing external calibration service (`aether-arena/calibration/`), and a trained RF-encoder checkpoint still does not exist in-tree.*
- **RuView beyond-SOTA research series** (`docs/research/ruview-beyond-sota/`, 6 docs) — research-swarm output defining the beyond-SOTA bar and the path to it: system capability audit (role→crate maturity matrix, gap analysis, risk register), web-verified 2026 SOTA landscape per capability axis (incl. ratified IEEE 802.11bf-2025), 8-pillar target architecture on the ADR-136 contract spine (no rewrite), 6-layer benchmark/validation methodology (all 15 criterion bench targets inventoried; ADR-171 statistical protocol), and a determinism-safe optimization roadmap. Includes session validation evidence: 2,797 workspace tests / 0 failed, Python proof PASS (bit-exact), paired pre/post criterion runs.

### Performance
- **CIR estimator warm-start precompute** — the diagonal Tikhonov preconditioner `diag(Φ^H Φ)+λI` and its CSR matrix were rebuilt every frame although they depend only on Φ and λ (fixed at `CirEstimator::new`); now precomputed at construction (`ruvsense/cir.rs`). Bit-identical floats (summation order unchanged, witness chain unaffected). Measured: `cir_estimate/he40` −3.9% (p<0.01), multiband groups −1.2/−1.4%; smaller configs within container noise.
- **RF tomography solver hoisting** — ISTA gradient buffer no longer allocated inside the 100-iteration loop, and the Frobenius Lipschitz bound moved from per-`reconstruct` to construction (`ruvsense/tomography.rs`). Bit-identical results.

### Added
- **Falsifiable occupancy benchmark (`wifi-densepose-train::occupancy_bench`).** Makes the presence/person-count "beyond SOTA" claim falsifiable in code instead of aspirational (the unfalsifiability gap from the beyond-SOTA system review). Grades predictions vs ground truth and gates a SOTA claim behind one `claim_allowed` invariant requiring all of: `DataProvenance::Measured` (synthetic/mock is scorable but **never claimable** — anti-mock-contamination per the CLAUDE.md Kconfig-bug lesson), a leak-free `EvalSplit` (refuses any split where a subject *or* environment id appears in both train and test — subject leakage / per-environment overfitting), `n_test ≥ min`, a **non-degenerate test set** (both truth classes represented: present-rate ≥ `min_positive_rate` and ≥ 1 absent sample — an all-absent set plus an always-absent predictor cannot release a claim; vacuous F1 scores 0.0, never 1.0), presence-F1 **bootstrap-CI lower bound** (deterministic seeded splitmix64) clearing the threshold, and count MAE within threshold. The claim string is unreadable except through the gate (`NO_CLAIM` otherwise). What remains is data, not method: a frozen, SHA-pinned, subject/environment-disjoint measured replay set turns the claim into a passing/failing test. 12 tests cover each refusal path, including the point-above/CI-below case (claim withheld on the CI lower bound even when the point estimate clears the threshold).
- **Live trust path: sensing-server routes real frames through the governed `StreamingEngine` (parallel governed path with partial output gating).** Previously the live server ran only the *bare* `MultistaticFuser` (fused amplitudes, no trust control plane), while the privacy/provenance/witness engine (ADR-135..146) ran only on synthetic in-test frames — the gap called out in ADR-136 §8 and the beyond-SOTA system review. New `engine_bridge` module drives `StreamingEngine::process_cycle` from the server's live `NodeState` map (reusing the existing `NodeState → MultiBandCsiFrame` conversion), lazily wiring each node as a WorldGraph sensor and bounding belief growth via the retention cap; every *governed belief* carries evidence + model + calibration + privacy decision and a deterministic witness. **Honest scope:** the engine runs alongside (not instead of) the bare fusion path that feeds the live `SensingUpdate`. What its decision gates on the wire today: a cycle emitted at class `Restricted` (base mode or contradiction/mesh-risk demotion) suppresses the per-node raw amplitude vectors from the live publish — the same field mapping `wifi-densepose-bfld`'s privacy gate applies at `Restricted`; gating the remaining derived outputs (person count, classification, signal field) is tracked as a follow-up. Trust state is no longer write-only: the latest witness, effective privacy class, demotion flag, recalibration recommendation, and an engine-error counter are readable on `GET /api/v1/status`, and engine errors are counted + rate-limit logged instead of silently swallowed (`EngineBridge::observe_cycle`). Adds `wifi-densepose-engine/-worldgraph/-bfld/-geo` deps. Bridge tests cover witnessed belief with provenance, determinism, idempotent node registration, retention bound, privacy-mode propagation, trust-state recording, the error-counter path, and Restricted-class raw-output suppression.

### Fixed
- **Real HE20 CSI no longer silently dropped or replaced with simulated data (fixes #1009, #1004).** Two ingest bugs caused real ESP32-C6 HE20 frames to be discarded or never received — the exact "real data silently lost" failure class the project fights. Each fix is pinned by a test that fails on the old code.
  - **#1009 §1b — HE20 baseline recorder trimmed 256 → 242 bins by sequential index (`wifi-densepose-signal/src/ruvsense/calibration.rs`).** ESP-IDF v5.5.2 delivers all 256 FFT bins for an HE20 frame; `CalibrationConfig::he20()` carried `num_active: 242`, so the recorder (which has no HE20 tone map — `extract_first_stream` takes the first `num_active` columns *sequentially*) kept bins 0..242 of the 256-bin grid. Those are the lower guard band + DC, **not** the 242 active tones, silently corrupting the empty-room baseline. Now `num_active: 256` records every delivered bin, staying aligned 1:1 with the live `deviation()` path. The exact-242 tone map deliberately stays only in `cir.rs` (`HE20_ACTIVE`), where the Φ sensing matrix genuinely needs it. Test `he20_records_all_256_bins_not_trimmed_to_242` asserts the finalized baseline covers all 256 bins (was 242). HE20 synthetic/bench fixtures updated to feed 256-bin frames (the real wire format).
  - **#1009 §1a/§1c — already-fixed u8→u16 `n_subcarriers` truncation, now regression-pinned.** The ADR-018 wire format carries `n_subcarriers` as u16 LE at bytes 6–7. A 256-bin HE20 frame (byte6=0x00, byte7=0x01) read as a single byte decodes to **0 subcarriers** → every frame skipped (invisible until HE20: ESP32-S3's ≤192 bins fit in one byte). The CLI parser (`wifi-densepose-cli/calibrate.rs`) and the sensing-server template parser (`wifi-densepose-sensing-server` `parse_esp32_frame`) were already corrected to u16 under #1005/ADR-110; added regression tests (`parse_esp32_frame_he20_256_bins_not_truncated`, CLI `test_parse_csi_packet_he_su_256_bins`) that fail on the old single-byte read so the truncation cannot silently return.
  - **#1004 — `--source auto` latched on `simulate` forever, never binding UDP :5005 (`wifi-densepose-sensing-server/src/main.rs`).** A one-shot boot probe resolved the source once; with no CSI flowing at boot (the normal firmware/server startup race) it served simulated poses for the whole process and ignored real CSI that arrived seconds later (the prior #937 fix hard-exited instead — equally wrong, the server could never pick up late-starting CSI). New `plan_source()` state machine: in `auto` mode **always bind the UDP receiver** and serve simulated data only until the first real frame, at which point `udp_receiver_task` promotes `source` → `esp32` (mirroring the existing `esp32 → esp32:offline` reversion in `effective_source()`); `simulated_data_task` self-suspends once promoted so it never clobbers live CSI. Explicit `--source simulated` stays a hard, UDP-free override for offline demos. 6 unit tests pin the resolution/promotion machine (`auto_with_no_boot_source_still_binds_udp_and_simulates`, etc.); the auto-binds-UDP assertion fails on the old behavior.
- **`wifi-densepose-mat` standalone `--no-default-features` build (101 errors → 0).** `pub mod api` was unconditional while its only dependency, serde, is optional behind the `api` feature — so any build without default features failed with unresolved serde imports (masked in `--workspace` runs by feature unification). The `api` module and its `create_router`/`AppState` re-export are now `#[cfg(feature = "api")]`-gated (with docsrs annotations). All feature combos compile: bare `--no-default-features`, `--no-default-features --features api`, and full default (177 tests pass).
- **WorldGraph no longer grows unboundedly under the live loop.** `StreamingEngine::process_cycle` appended one `SemanticState` belief per cycle with no eviction — ~1.7M nodes/day at 20 Hz (identified in `docs/research/ruview-beyond-sota/04-optimization-roadmap.md`). Added `WorldGraph::prune_semantic_states(max)` — deterministic eviction of the oldest beliefs by `(valid_from_unix_ms, id)`, structural nodes (rooms/zones/sensors/anchors/tracks/events) never eligible — and wired it into the engine after each belief append (`StreamingEngine::DEFAULT_SEMANTIC_RETENTION` = 7,200 ≈ 6 min at 20 Hz; tunable via `set_semantic_retention`). The WorldGraph holds *current* beliefs; durable history is the recorder's job, so no audit data is lost. 3 new tests (bounded growth end-to-end, oldest-only eviction, deterministic tie-break).
- **ESP32 edge heart rate no longer stuck at ~45 BPM / dropping wildly — #987.** The on-device HR estimator (`edge_processing.c`, `0xC5110002`) reported ~45 BPM regardless of true heart rate (Apple-Watch ground truth 87 BPM read as ~45) and swung frame-to-frame. Two root causes: (1) a hardcoded `sample_rate = 10.0f` that became wrong after #985's self-ping raised the CSI callback rate to a variable ~13–19 Hz — BPM scales as `assumed/actual × true`, so 87 read ~45 and the reading swung as CSI yield fluctuated; (2) the zero-crossing estimator locked onto a breathing harmonic (a 0.25 Hz breathing fundamental puts its 3rd harmonic at ~0.74 Hz ≈ 44 BPM inside the HR band). Fix: measure the real sample rate from inter-frame timestamps (used for BPM conversion + biquad re-tuning on >15% drift); replace the HR zero-crossing with an autocorrelation estimator that rejects breathing harmonics (driven by a robust autocorr breathing period); median-13 smooth the output. Hardware A/B (fixed vs unmodified control board, both `edge_tier=2`): control pegged 40–49 BPM; fixed reaches the true 88–91 BPM (vs 87 GT) and holds a stable physiological value (spread 59→0 for a steady subject). Known limitation: heavy subject motion still degrades the estimate (motion gating is a follow-up).
- **Person count no longer leaks up to 10 in heuristic mode — addresses #894.** `field_bridge::occupancy_or_fallback` returned the eigenvalue-based `FieldModel::estimate_occupancy` count **unbounded** (its internal ceiling is 10), while the sibling estimators on the same single-link data — the perturbation-energy fallback right below it and `score_to_person_count` — both cap at 3 ("1-3 for single ESP32"). On noisy / under-calibrated CSI the eigenvalue count inflated, producing the "10 persons reported when 1 present" symptom (seen when `--model` fails to load and the server runs on heuristics). Bounded the eigenvalue path to the shared `MAX_SINGLE_LINK_OCCUPANCY` (3) so every estimator on one link agrees; genuine higher counts come from the multistatic fusion path, not a single-link covariance estimate.
- **MQTT multi-node deployments now create one Home-Assistant device per node — closes #898.** After the #872 MQTT wiring landed, the JSON→`VitalsSnapshot` bridge hard-coded a single `node_id` (the MQTT client id) and the publisher used a single `OwnedDiscoveryBuilder`, so every physical node collapsed into one device (`identifiers:["wifi_densepose_wifi-densepose-1"]`), contradicting the "one device per node" docs. The bridge now emits one snapshot per node in the sensing update's `nodes[]` (each with its own `node_id` + RSSI, falling back to a single aggregate snapshot for wifi/simulate sources), and the publisher derives a per-node builder (`OwnedDiscoveryBuilder::for_node`) that publishes discovery + availability lazily on first sight of each `node_id` and routes state to per-node topics — yielding N distinct HA devices with per-node availability/LWT. Unit-tested (distinct nodes → distinct `wifi_densepose_<node>` identifiers); 71 MQTT tests pass.
- **Person count no longer pinned to 1 — addresses #803.** The aggregate occupancy reported by the sensing server was derived from `smoothed_person_score`, an EMA-smoothed *activity* score (amplitude variance / motion / spectral energy). That score saturates near a single occupant — one moving person maxes it out — so it cannot discriminate occupancy *count* and stayed clamped at 1 across S3/C6 and the Python/Docker/Rust servers. Meanwhile the count-aware per-node estimates the ESP32 paths already compute (firmware `n_persons`, and the DynamicMinCut `corr_persons`) were stashed in `NodeState::prev_person_count` and then **discarded** by the aggregator (same dead-wiring class as #872). The aggregator now takes `max(activity_count, node_max)` via a unit-tested `aggregate_person_count` helper, so a node positively estimating 2–3 occupants is surfaced instead of overwritten. The fix can only ever *raise* the count when a node reports more people, so the single-occupant case is provably never inflated (regression-guarded by test). **Second half:** the pure-CSI per-node path itself clamped its own estimate — the DynamicMinCut occupancy (`estimate_persons_from_correlation`, 0–3) was mapped to a score via `corr_persons / 3.0`, putting 2 people at 0.667, *just under* the 0.70 up-threshold of `score_to_person_count`, so the per-node count never climbed past 1 (so `node_max` was also stuck at 1 for CSI-only nodes). Replaced it with a threshold-aligned `corr_persons_to_score` mapping (1→0.40, 2→0.74, 3→0.96) whose steady state round-trips back to the same count through the EMA + hysteresis, while still gating transient noise. A convergence test replays the exact EMA loop to prove min-cut=2 now reports 2 (and documents that the old `/3.0` mapping reported 1). Full multi-person accuracy still depends on the underlying estimator quality; this removes the two server-side clamps that masked it. 586 sensing-server tests pass.
- **MQTT publisher now actually runs (`--mqtt`) — closes #872.** The `--mqtt*` flags were defined only in `cli::Args` (dead code, referenced nowhere) while the binary parses a *separate* `main::Args` with no mqtt fields, and `main.rs` never started the `mqtt::` publisher — so MQTT/Home-Assistant integration was completely unwired (`--mqtt` errored as an unexpected argument, and even with the Docker image's `--features mqtt` build the publisher never ran). Earlier attempts chased a Docker *rebuild*; the real cause was disconnected *code*. Extracted the flags into a shared `cli::MqttArgs` (`#[command(flatten)]` into both structs), spawn the publisher on `--mqtt`, and bridge the JSON sensing broadcast into the typed `VitalsSnapshot` stream with a defensive `serde_json::Value` mapping. Verified end-to-end against `mosquitto`: 20 HA auto-discovery entities + live state (presence/person-count/…). 577 (default) / 580 (`--features mqtt`) tests pass.
- **Mass Casualty triage never reports a survivor with a heartbeat as Deceased (safety) — PR #926.** Both triage paths in `wifi-densepose-mat` — `TriageCalculator::calculate` (`combine_assessments(Absent, None) ⇒ Deceased`) and the detection path `EnsembleClassifier::determine_triage` (`!has_breathing && !has_movement ⇒ Deceased`) — ignored the `heartbeat` field. A survivor with a detectable **pulse** but no sensed breathing/movement (respiratory arrest — the most time-critical *savable* state, Immediate/Red) was therefore reported **Deceased (Black)** and deprioritized for rescue. The domain path was in fact only reachable *because* a heartbeat made `has_vitals()` true, so every "Deceased" was a live person. Both paths now escalate to **Immediate** when a heartbeat is present; total absence of breathing, movement *and* heartbeat is unchanged (domain → `Unknown`, ensemble → `Deceased`). 2 safety regression tests; full MAT suite (177) green.
- **Per-node Home-Assistant devices now report each node's *own* presence/motion — PR #918.** After the one-device-per-node fan-out landed, the MQTT bridge still applied the *room-level aggregate* `classification` to every node, so in a multi-node deployment a node watching an empty corner inherited another node's "present" (and `motion_level: "absent"` was mis-mapped to full motion). Each node in the broadcast `nodes[]` already carries its own `classification`; the bridge now reads it per node (extracted into a testable `vitals_snapshots_from_sensing_json`), keeping vitals + person count room-level. 4 unit tests.
- **`--model` gives an actionable diagnostic instead of a cryptic magic error — PR #919 (refs #894).** Passing a HuggingFace `ruvnet/wifi-densepose-pretrained` file (`model.safetensors` / `model-q4.bin` / `model.rvf.jsonl`) to `--model` produced `invalid magic at offset 0: … got 0x77455735`, then a silent fall back to heuristics. The load-failure path now detects the format (safetensors / quantized blob / JSONL manifest) and explains that those files are a different format **and** encoder architecture than the RVF binary container the progressive loader expects, pointing to #894. Pure `diagnose_model_load_error` + 4 tests.
- **`--export-rvf` no longer silently produces a placeholder model — PR #920.** The `--export-rvf` handler ran *before* `--train`/`--pretrain` and unconditionally wrote placeholder sine-wave weights, so the documented `--train … --export-rvf <path>` workflow short-circuited to a fake model and never trained (while printing "exported successfully"). It now emits the placeholder **container-format demo** only standalone (with a clear warning), and falls through to real training when `--train`/`--pretrain` is set; docs point to `--save-rvf` for the real model. 3 guard tests.

### Added
- **ADR-151 per-room calibration & specialist training — full `baseline → enroll → extract → train` pipeline (new `wifi-densepose-calibration` crate).** "Teach the room before you teach the model": a local-first pipeline that turns a few minutes of clean human anchors — layered on the ADR-135 empty-room baseline — into a versioned bank of small, room-calibrated specialists for **presence, posture, breathing, heartbeat, restlessness, and anomaly**. Stages: guided enrollment with an adaptive quality gate (event-sourced `EnrollmentSession`, re-prompts bad anchors); feature extraction (autocorrelation periodicity in breathing/HR bands + variance/motion); six small specialists (learned threshold / nearest-prototype / band-limited periodicity / novelty); a `SpecialistBank` with baseline-drift **STALE** invalidation; and a `MixtureOfSpecialists` runtime with presence short-circuit + anomaly veto + confidence gating. Specialists are statistical heads today (runnable + hardware-validated); the frozen ADR-150 HF RF Foundation Encoder backbone is the documented upgrade path.
  - **CLI:** `enroll` / `train-room` / `room-status` / `room-watch`, plus the Stage-1 `calibrate-serve` HTTP API (CORS-enabled: `POST /start`, `GET /status`, `POST /stop`, `GET /result`, `GET /baselines`, `GET /health`) and a firewall-free `scripts/csi-udp-relay.py` for local Windows ESP32 testing without admin.
  - **Multistatic fusion (ADR-029):** `MultiNodeMixture` fuses several co-located nodes (each with its own room-calibrated bank) into one room state — presence OR'd across nodes, posture/breathing/heartbeat from the highest-confidence node, a single implausible node vetoes the room's vitals. Driven via `room-watch --node-bank N:path` (repeatable), which groups live frames by `node_id` and fuses. Same-room only; cross-room is federation (ADR-105).
  - **Validated on live ESP32-S3 (COM8, `edge_tier=0` raw CSI):** baseline capture (120 frames → 52-subcarrier baseline); the real parser → feature-extraction → mixture runtime detecting breathing (~16–31 BPM); and the multistatic ingest grouping/fusing by node-id end-to-end. Full multi-anchor enrollment accuracy requires the operator to perform the poses; true 2-node fusion + phase-based breathing + RVF/HNSW storage are noted follow-ups. 54 tests pass (35 calibration + 19 CLI).
- **WiFi-CSI pose: efficiency frontier + per-room calibration service** (ADR-150 §3.2–3.6). Two beyond-SOTA results on the MM-Fi benchmark, plus the deployment mechanism that resolves real-world generalization:
  - **Efficiency frontier** — a **75 K-param model beats published SOTA** (74.3% vs MultiFormer 72.25% torso-PCK@20); every config from `micro` up is Pareto-dominant (smaller *and* more accurate than prior work). Shipped a deployable **int4 edge model (~20 KB, verified 74.08%, 0.135 ms single-thread CPU)** — published at [`ruvnet/wifi-densepose-mmfi-pose/edge`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose). See [`docs/benchmarks/wifi-pose-efficiency-frontier.md`](docs/benchmarks/wifi-pose-efficiency-frontier.md).
  - **Generalization solved by few-shot calibration** — zero-shot cross-subject (~64%) and cross-environment (~10%) are *not* closeable by algorithms (CORAL, DANN, instance-norm, contrastive foundation-pretraining all tested, all failed) or by more training subjects (saturates ~64%). But **~100–200 labeled in-room samples recover SOTA-level pose**: cross-subject 64→76%, **cross-environment 10→73% (60% from just 5 samples)** — deployable as a **~11 KB per-room LoRA adapter** on a frozen shared base. Full empirical chain in ADR-150 §3.2–3.6.
  - **Calibration service (complete, both model paths, cross-language verified)** — `aether-arena/calibration/`: `calibrate.py` (transformer model, `.npz` adapter) + `infer.py` (verified 3.09%→74.29% on an unseen MM-Fi room), **and `cog_calibrate.py`** which fits a `fc1.a/fc1.b/fc2.a/fc2.b` **safetensors** adapter for the deployed cog conv+MLP model (`pose_v1.safetensors`). Consumed by the Rust product engine: `InferenceEngine::with_adapter()` + `cog-pose-estimation run --config <cfg> --adapter <room.safetensors>`. Self-contained regression tests for both Python producers (`test_calibration.py`, `test_cog_calibration.py`) **plus a cross-language Rust integration test** that loads a real `cog_calibrate.py`-generated adapter fixture and asserts it activates + changes engine output. All green.
- **Windows workspace build + test now green** (cross-platform fixes). `wifi-densepose-worldmodel` imported `tokio::net::UnixStream` unconditionally, so `cargo build/test --workspace` failed to compile on Windows (E0432) — now the OccWorld Unix-socket bridge is `#[cfg(unix)]`-gated with a clear non-unix fallback. And `wifi-densepose-bfld`'s `readme_quickstart_uses_canonical_public_api` test checked a multi-line `pipeline\n    .process` needle that never matched on a CRLF checkout — now normalizes line endings. Result: **2,682 workspace tests pass / 0 fail on Windows** (the pre-merge gate was previously unrunnable there).
- **`ruview-swarm` crate (ADR-148)** — drone swarm control system with hierarchical-mesh topology, Raft consensus, MAPPO multi-agent reinforcement learning, and CSI sensing integration. 14 modules: topology (Raft/Gossip/Mesh), formation control (virtual-structure/leader-follower/Reynolds flocking), RRT-APF path planning, auction+FNN task allocation, MARL actor + PPO training loop, security (MAVLink v2 HMAC-SHA256 signing, UWB anti-spoofing, geofencing, Remote ID, FHSS anti-jamming), 10-state fail-safe machine, and SwarmOrchestrator. ITAR-gated coordination features (USML Category VIII(h)(12)) behind `itar-unrestricted` feature.
- **Ruflo integration for `ruview-swarm`** — feature-gated (`ruflo`) AI-agent capability layer connecting to the claude-flow daemon: AgentDB mission memory (`memory_store`/`memory_search`), HNSW pattern learning (`agentdb_pattern-store`/`-search`), AIDefence MAVLink message scanning, and SONA intelligence trajectory hooks. `RufloBackend` trait with `HttpRufloBackend` (JSON-RPC 2.0) and `MockRufloBackend` implementations.

### Performance
- `ruview-swarm` benchmarks (criterion, release): MARL actor inference 3.3 µs, RRT-APF planning 0.043 ms, multi-view CSI fusion 58.5 ns, 3-view localization 1.732 m (beats Wi2SAR 5 m SOTA baseline), 4-drone SAR coverage 223 s for 400×400 m (under 240 s target).

### Added
- **ADR-147 — OccWorld world model integration** (`wifi-densepose-worldmodel` v0.3.0 published to crates.io). 15-frame trajectory prediction at 209 ms / 3.37 GB VRAM on RTX 5080. Phase 3 domain adapter `scripts/ruview_occ_dataset.py` (`RuViewOccDataset`) converts WorldGraph snapshots to OccWorld tensors with indoor class remapping + zero ego-poses (validated). Phase 5 retraining pipeline `scripts/occworld_retrain.py` — VQVAE + transformer fine-tuning on RuView occupancy snapshots. See [ADR-147](docs/adr/ADR-147-nvidia-cosmos-world-foundation-model-integration.md) · [benchmark proof](docs/adr/ADR-168-benchmark-proof.md).

### Added
- **ADR-125 (APPLE-FABRIC) — RuView ↔ Apple Home native HAP bridge proposal + reference impl** (issue #796). New ADR-125 lays out a three-phase plan to expose RuView as a discoverable HomeKit accessory on the LAN so a HomePod (as Home Hub) sees presence / vitals / BFLD-derived events natively — zero Home-Assistant intermediary. Two architectural decisions resolved in the ADR per design review: (1) **one HAP bridge with N child accessories** (single pairing, matches Hue/Eve pattern), and (2) **identity-risk mapping is semantic, not probabilistic** — `identity_risk_score` and Soul-Signature match probability never cross the HAP boundary; instead three thresholded events are exposed (`Unknown Presence`, `Unexpected Occupancy`, `Unrecognized Activity Pattern`) so RuView reads as calm-tech ambient awareness, not surveillance UX. ADR-125 §2.1.a reference impl ships now: `scripts/hap-test-sensor.py` (HAP-1.1 bridge advertised over mDNS, paired with operator's iPhone) + `scripts/c6-presence-watcher.py` (parses ESP32 `RV_FEATURE_STATE_MAGIC = 0xC5110006` UDP packets with IEEE CRC32 validation, hysteresis, and a Python port of `wifi-densepose-bfld::PrivacyClass` that enforces ADR-125 §2.1.d invariant I1 at the HomeKit edge — only `Anonymous` (2) and `Restricted` (3) frames may cross; `Raw`/`Derived` are refused with exit code 2 and the cited ADR clause). Validated end-to-end on real hardware (no mocks): ESP32-C6 on `ruv.net` → UDP/5005 → mac-mini watcher → BFLD gate → HAP bridge → iPhone Home app shows `Unknown Presence` live characteristic flip. **Empirical**: 50-51 valid CRC-passing feature_state packets per 10 s window from the live C6; zero CRC errors. P2 (Rust-native HAP via the `hap` crate, replaces the Python sidecar) and P3 (Matter Controller once `matter-rs` stabilizes) follow.

### Security
- **ESP32 OTA upload now fails closed when no PSK is provisioned** (#596 audit finding — critical, **breaking change for unprovisioned nodes**). `ota_check_auth()` previously returned `true` when `s_ota_psk[0] == '\0'`, so a freshly-flashed node would accept attacker-controlled firmware over plain HTTP on port 8032 from any host on the WiFi. No Secure Boot V2, no signed-image verification — a single LAN call could brick or backdoor a node. The fix rejects every OTA upload until a PSK is written to NVS (the OTA HTTP server still starts so operators can run `provision.py --ota-psk <hex>` over USB-CDC without reflashing). **Operators affected**: any deployment that relied on the unauthenticated OTA endpoint working out of the box now needs to provision a PSK before subsequent OTA pushes will succeed. Boot-time `ESP_LOGW` makes the new posture visible.
- **Bearer-token auth accepts the scheme case-insensitively (RFC 6750) — PR #929.** `require_bearer` parsed the `Authorization` header with a case-sensitive `strip_prefix("Bearer ")`, so a *correct* `RUVIEW_API_TOKEN` sent as `Authorization: bearer <token>` (or `BEARER`, or with extra whitespace) was rejected with a confusing 401 — needless friction when enabling auth. The scheme is now matched with `eq_ignore_ascii_case` (per RFC 6750 §2.1 / RFC 7235 §2.1); the token compare is unchanged — still exact and constant-time (`ct_eq`) — so a wrong token or a non-Bearer scheme (`Basic …`) still returns 401. Audited the surrounding code while here: `ct_eq` correctly rejects length mismatch (no prefix-auth bypass) and the middleware fails closed. New `accepts_case_insensitive_bearer_scheme` test.
- **Path-traversal vulnerabilities patched in five sensing-server endpoints** (closes #615 — critical). New `wifi_densepose_sensing_server::path_safety::safe_id()` enforces `[A-Za-z0-9._-]` only (no leading `.`, max 64 chars) before any user-controlled identifier reaches a `format!()` building a filesystem path. Applied at:
  - `POST /api/v1/recording/start` (`recording.rs` — `session_name`)
  - `GET /api/v1/recording/download/:id` (`recording.rs` — `id`)
  - `DELETE /api/v1/recording/delete/:id` (`recording.rs` — `id`)
  - `POST /api/v1/models/load` (`model_manager.rs` — `model_id`)
  - `training_api.rs` `load_recording_frames` (`dataset_id`s)

  Pre-fix, unauthenticated callers could read `../../etc/passwd`-style paths, write arbitrary JSONL files, load attacker-controlled `.rvf` model files, or delete arbitrary files the server process could touch. 9 unit tests in `path_safety::tests` exercise the rejection envelope (empty, too-long, path separators, parent-dir traversal, null byte, whitespace/specials, non-ASCII).

### Fixed
- **WebSocket `/ws/sensing` now reports `esp32:offline` when ESP32 hardware goes stale** (closes #618). `broadcast_tick_task` was re-emitting the cached `latest_update` with a frozen `source: "esp32"` field forever after the hardware lost power or network. The REST `/health` endpoint already called `effective_source()` (which returns `"esp32:offline"` after `ESP32_OFFLINE_TIMEOUT` = 5 s with no UDP frames), but the WS broadcast path was the one consumer that didn't. Result: the UI's "LIVE — ESP32 HARDWARE Connected" banner stayed green long after the hardware went away, and `vital_signs`/`features`/`classification` re-broadcasted the last-seen values indefinitely. Fix: clone the cached `latest_update` per tick, overwrite `source` with `s.effective_source()`, then serialize and broadcast. UI can now switch to an offline state on the same 5-second budget the REST surface uses.
- **Proof replay (`archive/v1/data/proof/verify.py`) is now cross-platform deterministic** (closes #560). Three changes together: (1) `features_to_bytes()` now `np.round(.., HASH_QUANTIZATION_DECIMALS=6)`s each feature array before packing as little-endian f64, collapsing ULP-level drift from scipy.fft pocketfft SIMD reordering; (2) the `Verify Pipeline Determinism` workflow pins `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` — multi-threaded BLAS reductions were a deeper source of non-determinism than SIMD reordering, and 6-decimal quantization alone wasn't enough across Azure VM microarchitectures; (3) `expected_features.sha256` regenerated under the new conditions. CI now passes the determinism check (same hash across consecutive runs on canonical Linux x86_64 CI runner: `667eb054c44ac510342665bf9c93d608868a8ead948ae8774b2796ebce6f8fe7`). `scripts/probe-fft-platform.py` updated to mirror `HASH_QUANTIZATION_DECIMALS=6` for cross-machine spot-checks.
- **`archive/v1/src/services/pose_service.py:223` calls the right method on `PhaseSanitizer`** (closes #612). The call was `self.phase_sanitizer.sanitize(phase_data)`, but `PhaseSanitizer`'s full-pipeline entry point is named `sanitize_phase()` (`unwrap_phase` + `remove_outliers` + `smooth_phase` chained, see `archive/v1/src/core/phase_sanitizer.py:266`). The shorter `sanitize` name doesn't exist on the class, so any path that reached this branch raised `AttributeError` and crashed the pose service mid-frame.
- **`adaptive_classifier.rs:94` no longer panics on NaN feature values** (closes #611).
  `sorted.sort_by(|a, b| a.partial_cmp(b).unwrap())` returned `None` and panicked
  whenever a single `NaN` reached the classifier from real ESP32 hardware (silent
  DSP div-by-zero, empty buffer). One bad frame killed the entire sensing-server
  process. Swapped for `unwrap_or(Ordering::Equal)`, matching the pattern the
  same file already used at lines 149-150 and 155. Per-frame hot path; this was
  a real production crash vector.
- **Completed the #611 NaN-panic audit across the sensing-server crate** (follow-up
  to #613). The original audit grepped for the literal `partial_cmp(b).unwrap()`
  and missed seven additional production sites that use comparator variants
  (`partial_cmp(b.1).unwrap()`, `partial_cmp(&variances[b]).unwrap()`). All share
  the same crash class — a single `NaN` in CSI-derived state panics the whole
  sensing-server. Fixed:
  - `adaptive_classifier.rs:205` — `AdaptiveModel::classify()` argmax over softmax
    probs. **Same per-frame hot path as #611**; NaN flows through normalise →
    logits → softmax and still reaches this site even after the #613 IQR fix.
  - `adaptive_classifier.rs:480, 500` — training-loop argmax in `train()`
    (training/per-class accuracy reporting).
  - `main.rs:2446, 2449` and `csi.rs:602, 605` — variance-based source/sink
    selection in `count_persons_mincut`. The outer `unwrap_or((0, &0))` only
    catches an empty iterator; it cannot rescue a comparator panic.

  Remaining `partial_cmp(...).unwrap()` sites in the workspace are all inside
  `#[cfg(test)]` / `#[test]` blocks (`spectrogram.rs:269`, `depth.rs:234`,
  `connectivity.rs:477`, `vital_signs.rs:737`) where inputs are controlled.
- **`ui/utils/pose-renderer.js` no longer divides by zero** when two render frames land in the same `performance.now()` tick (issue #519 Bug 2). `deltaTime` is now `Math.max(currentTime - lastFrameTime, 1)` before the `1000 / deltaTime` division, capping displayed FPS at 1000 — far above any real render rate, but finite so the EMA `averageFps = averageFps * 0.9 + fps * 0.1` no longer poisons itself to `Infinity` on a single zero-dt tick.

### Removed
- **Stub crates `wifi-densepose-api`, `wifi-densepose-db`, `wifi-densepose-config`** (closes #578).
  Each was a single-line doc-comment placeholder with an empty `[dependencies]`
  section and zero references from any source file or `Cargo.toml`. The names
  were reserved early for an envisioned REST/database/config split that never
  materialised; the functionality they would provide is covered today by
  `wifi-densepose-sensing-server` (Axum REST/WS), per-crate config + CLI args,
  and the project's real-time-only (no-persistent-state) posture. Removing them
  from the workspace prevents `cargo` from listing dead crates and shipping
  empty published artifacts. If any of these names is needed in the future,
  they can be reintroduced with a real implementation.

### Added
- **BFLD — Beamforming Feedback Layer for Detection (ADR-118 umbrella + ADR-119 frame format + ADR-120 privacy class + ADR-121 identity risk scoring + ADR-122 RuView HA/Matter exposure + ADR-123 capture path, [#787](https://github.com/ruvnet/RuView/issues/787)).** New crate `wifi-densepose-bfld` (`v2/crates/wifi-densepose-bfld/`) — the privacy-gated WiFi sensing layer that detects when RF data crosses from "ambient sensing" into "identity record" and **structurally prevents** identity-correlated data from leaving the node. Three invariants enforced by the type system (not policy): **I1** raw BFI never exits the node (`Sink` marker-trait hierarchy + `PrivacyClass::Raw.allows_network() == false`), **I2** identity embedding is in-RAM-only (`IdentityEmbedding` has no `Serialize`/`Clone`/`Copy` + `Drop` zeroizes), **I3** cross-site identity correlation is cryptographically impossible (per-site BLAKE3-keyed `SignatureHasher` with daily epoch rotation; mean cross-site Hamming distance ≥120 bits across 100 trials). Ships the complete operator surface: `BfldPipeline` + `BfldPipelineHandle` (worker-thread variant + `spawn_with_oracle` for Soul Signature deployments), `BfldEvent` with JSON publishing (`"blake3:<hex>"` `rf_signature_hash` format per spec), 4 `privacy_class` levels (Raw/Derived/Anonymous/Restricted) with `PrivacyGate::demote` monotonic transformer + irreversible `apply_privacy_gating`, `CoherenceGate` with ±0.05 hysteresis + 5-second debounce + clock-skew resilience (saturating_sub), `SoulMatchOracle` Recalibrate-exemption trait for enrolled-person deployments. **MQTT/HA surface**: `mqtt_topics::render_events` + `publish_event` (class-gated topic routing — Raw/Derived publish 0 topics, Anonymous publishes 6, Restricted publishes 5 with `identity_risk` stripped), `ha_discovery::render_discovery_payloads` + `publish_discovery` (HA-DISCO config payloads with `availability_topic` integration), `availability` module (`online`/`offline` + LWT-aware `with_lwt` helper for `rumqttc::MqttOptions`), `RumqttPublisher` behind a `mqtt` feature gate with `connect_with_lwt` for broker-side auto-offline. **3 operator HA Blueprints** under `v2/crates/cog-ha-matter/blueprints/bfld/` (presence-driven-lighting, motion-aware-HVAC, identity-risk-anomaly-notification with rolling 7-day z-score). **Two runnable examples** (`bfld_minimal` for in-process consumers, `bfld_handle` for the production worker-thread + bootstrap-then-spawn pattern). **GitHub Actions CI workflow** (`.github/workflows/bfld-mqtt-integration.yml`) spins up `eclipse-mosquitto:2` as a service container so the env-gated `mosquitto_integration` and `rumqttc_lwt` tests run end-to-end in CI. **Performance**: `BfldFrame::to_bytes()` measured at **320,255 frames/sec** debug (6.4× ADR-119 AC7 release target of 50k), header-only at 1,654,517 frames/sec, presence-detection latency p95 = **0.9µs** (~1,000,000× under ADR-119 AC2's 1s target), 9.96 Hz motion-publish rate through `BfldPipelineHandle` (10× ADR-122 AC3 floor). **Coverage**: 327 tests at default features, 101 no_std-compatible, 220+ with `--features mqtt`. CRC-32/ISO-HDLC polynomial pinned against `"123456789" → 0xCBF43926`, public-API surface snapshot pinned across all `pub use` re-exports, `BfldError` Display contract pinned for log-grep monitoring rules, reserved-flag-bits forward-compat round-trip property, `apply_privacy_gating` irreversibility (5-cycle round-trip stress proves stripped fields never resurrect). Companion research dossier in `docs/research/BFLD/` (11 files, 13,544 words). 49-iter implementation chain from scaffold (`feat/adr-118/p1`, `c965e3e6c`) through current head with per-iter progress comments on issue [#787](https://github.com/ruvnet/RuView/issues/787). Try it: `cargo run -p wifi-densepose-bfld --example bfld_handle`.
- **SENSE-BRIDGE — rvagent MCP server + ruvector npm + ruflo integration (ADR-124, [#787](https://github.com/ruvnet/RuView/issues/787)).** New npm package `@ruvnet/rvagent` (`tools/ruview-mcp/`) — a dual-transport [Model Context Protocol](https://modelcontextprotocol.io/) server that bridges the RuView WiFi-DensePose sensing stack to AI agents (Claude Code, Cursor, ruflo swarms). **6 of 20 ADR-124 §4.1 tools wired** in this initial release: `ruview.presence.now` (occupancy), `ruview.vitals.get_breathing` / `get_heart_rate` / `get_all` (biometric vitals via `EdgeVitalsMessage` surface, ADR-124 §6 Python ws.py:74-88 parity), `ruview.bfld.last_scan` (latest BFLD event — `identity_risk_score`, `privacy_class`, `n_frames`, `timestamp_ms`), `ruview.bfld.subscribe` (MQTT wildcard subscription with synthetic UUID envelope fallback). **Dual-transport architecture (ADR-124 §3)**: stdio (`npx @ruvnet/rvagent stdio` — recommended for Claude Code / Cursor local flow) + Streamable HTTP (`POST /mcp` bound to `127.0.0.1:3001` by default — for remote ruflo swarms across the Tailscale fleet). **Security model (ADR-124 §6)**: Origin header validation (cross-origin POST → 403), bearer-token auth slot (`RVAGENT_HTTP_TOKEN` → 401), bind default `127.0.0.1` per MCP spec requirement. **Uniform schema validation gate (ADR-124 §3)**: every `CallTool` request runs `zod.safeParse` via `TOOL_INPUT_SCHEMAS` before dispatch; failures throw `McpError(InvalidParams)`. **Full Zod schema barrel (ADR-124 §4.1 + §4.1a)**: `src/schemas/tools.ts` defines all 20 tool input schemas including the 5 RUVIEW-POLICY governance tools (can_access_vitals, can_query_presence, can_subscribe, redact_identity_fields, audit_log). **Python surface parity**: `EdgeVitalsMessage` TypeScript interface mirrors Python ws.py:74-88; ADR-124 §6 parity table drives the field names. **93 tests across 7 suites** (manifest, schemas, validate, tools, http-transport, bfld-tools, vitals-tools) — all green. Try it: `npx @ruvnet/rvagent stdio` (with `RUVIEW_SENSING_SERVER_URL=http://localhost:3000`).
- **Home Assistant + Matter integration (ADR-115).** New `--mqtt` and `--matter` flags on `wifi-densepose-sensing-server` expose the full sensing capability set to any Home Assistant install via MQTT auto-discovery (HA-DISCO) and to any Matter controller (Apple Home / Google Home / Alexa / SmartThings) via a built-in Matter Bridge scaffolding (HA-FABRIC, SDK wiring v0.7.1). Includes 21 entity kinds per node — 11 raw signals + 10 inferred semantic primitives (HA-MIND: someone-sleeping, possible-distress, room-active, elderly-inactivity-anomaly, meeting, bathroom, fall-risk, bed-exit, no-movement, multi-room-transition). The semantic primitives run server-side so `--privacy-mode` strips HR/BR/pose values from the wire while still publishing the inferred *states* — the architectural win for healthcare and AAL deployments. Ships **8 starter HA Blueprints** under `examples/ha-blueprints/`, **3 drop-in Lovelace dashboards** under `examples/lovelace/` (including a privacy-mode-compatible healthcare care view), mTLS support, 32 KB payload-size cap, MQTT-wildcard topic-injection rejection, `RUVIEW_MQTT_STRICT_TLS=1` v0.8.0 upgrade path. **420 lib tests** cover the implementation including **~2,560 fuzzed assertions per CI run** (10 proptest cases across wire-boundary security + semantic-bus invariants). Plus mosquitto-backed integration tests in `.github/workflows/mqtt-integration.yml`, criterion benchmarks beating every ADR target by 1.6×–208×, and an ESP32-S3 hardware validation harness (`scripts/validate-esp32-mqtt.sh`) that asserts the full pipeline end-to-end with a witness bundle generator (`scripts/witness-adr-115.sh`) that self-verifies. See [`docs/releases/v0.7.0-mqtt-matter.md`](docs/releases/v0.7.0-mqtt-matter.md), [`docs/integrations/home-assistant.md`](docs/integrations/home-assistant.md), [`docs/integrations/semantic-primitives-metrics.md`](docs/integrations/semantic-primitives-metrics.md), [`docs/integrations/benchmarks.md`](docs/integrations/benchmarks.md), [`docs/adr/ADR-115-home-assistant-integration.md`](docs/adr/ADR-115-home-assistant-integration.md), tracking issue [#776](https://github.com/ruvnet/RuView/issues/776), PR [#778](https://github.com/ruvnet/RuView/pull/778). Matter SDK wiring (P8b) and CSA-certification path (P10) deferred to v0.7.1+ per ADR §9.10. Try it: `cargo run -p wifi-densepose-sensing-server --features mqtt --example mqtt_publisher -- --mqtt --mqtt-host 127.0.0.1`.
- **ESP32-C6 firmware target with Wi-Fi 6 / 802.15.4 / TWT / LP-core support ([ADR-110](docs/adr/ADR-110-esp32-c6-firmware-extension.md), #762).** `firmware/esp32-csi-node` now builds for **both** `esp32s3` (existing production node) and `esp32c6` (new research/seed-node target) from the same source tree — pick via `idf.py set-target esp32c6` and ESP-IDF auto-applies the new `sdkconfig.defaults.esp32c6` overlay. Every C6 module is `#ifdef CONFIG_IDF_TARGET_ESP32C6` gated, so the S3 build is byte-identical to today (no regression).
  - **Wi-Fi 6 HE-LTF subcarrier tagging** — `csi_collector.c` now reads `rx_ctrl.cur_bb_format` and writes the PPDU type (0=HT/legacy, 1=HE-SU, 2=HE-MU, 3=HE-TB) into ADR-018 frame byte 18, plus bandwidth flags (20/40 MHz, STBC, 802.15.4-sync-valid) into byte 19. Bytes 18-19 were previously reserved-zero, so old aggregators read them as before — fully backwards compatible. Magic stays `0xC5110001`. Default on via `CONFIG_CSI_FRAME_HE_TAGGING`. First firmware in the open ESP32 ecosystem to tag CSI frames with 11ax PPDU metadata.
  - **802.15.4 mesh time-sync** — new `c6_timesync.{h,c}` (262 lines) provides cross-node clock alignment over the C6's separate 802.15.4 radio, freeing WiFi airtime from coordination traffic (directly addresses the ADR-029/030 multistatic synchronization gap). Protocol: lowest EUI-64 wins election, leader broadcasts `TS_BEACON` (`magic=0x54534D45`, leader epoch µs) every 100 ms on channel 15, followers compute `offset = leader_us - local_us` and apply lazily — every CSI frame is stamped with `c6_timesync_get_epoch_us()`. Target alignment ±100 µs. Default on via `CONFIG_C6_TIMESYNC_ENABLE`. Verified initializing at boot on COM6 (`c6_ts: init done: channel=15 EUI=206ef1fffefffe17 leader=yes(candidate)` at +413 ms).
  - **TWT (Target Wake Time)** — new `c6_twt.{h,c}` (223 lines) wraps `esp_wifi_sta_itwt_setup` from `esp_wifi_he.h` to negotiate an individual TWT agreement with the AP after STA connect. Replaces today's opportunistic CSI capture with a scheduler-bounded one (default wake interval 10 ms = 100 fps cadence). Graceful NACK fallback: when the AP doesn't support 11ax iTWT, the helper logs and returns OK so the device keeps doing opportunistic CSI just like the S3. Teardown on `WIFI_EVENT_STA_DISCONNECTED` keeps the AP's TWT scheduler clean. Gated on `SOC_WIFI_HE_SUPPORT` (auto-set on C6/C5 chips).
  - **LP-core wake-on-motion hibernation** — new `c6_lp_core.{h,c}` (134 lines) arms the C6 LP RISC-V coprocessor as an always-on motion gate; HP core stays in deep sleep until a configurable GPIO wakes it (ext1 deep-sleep wake source in this initial cut, real LP-core program in follow-up). Targets ≤5 µA hibernation current for battery-powered Cognitum Seed nodes (vs the S3's ~10 µA ULP-FSM floor). Opt-in via `CONFIG_C6_LP_CORE_ENABLE` (default off — only enabled on nodes flashed for battery-powered seed duty).
  - **Build matrix**: S3 stays `partitions_display.csv` (8 MB + display + WASM), C6 uses `partitions_4mb.csv` (4 MB single OTA, no display, no WASM3, no LCD). C6 final binary 1003 KB (46% partition slack), 9 % smaller than S3 production. Free heap 310 KiB at boot, app_main reached in 343 ms, 802.15.4 stack up in another 70 ms.
  - **Why this matters**: opens three research surfaces nobody has published yet — Wi-Fi-6 CSI human pose, multistatic CSI clock alignment over a side-channel radio, and TWT-bounded deterministic CSI cadence. The S3 production fleet keeps shipping the existing capabilities; the C6 is the research / battery-seed expansion target.
  - **Docs**: ADR-110 (186 lines, Status=Accepted), tracking issue [ruvnet/RuView#762](https://github.com/ruvnet/RuView/issues/762) with per-phase progress comments, README hardware table + Quick-Start Option 2b, `docs/user-guide.md` full ESP32-C6 section (build, flash, provision, multi-room time-sync, battery seed mode), full empirical record in [`docs/WITNESS-LOG-110.md`](docs/WITNESS-LOG-110.md) with verified / claimed / bugs-fixed / bugs-found sections.
  - **Wave 2 follow-up (D1 workaround)**: 5 systematic experiments on 3 live C6 boards confirmed the IDF v5.4 802.15.4 RX path is unfixable from user code (TX works 100 %, RX delivers 0 frames; coex/channel/OpenThread/manual-rearm all ruled out). Pivoted to ESP-NOW for the cross-node sync transport — `main/c6_sync_espnow.{h,c}` is the same TS_BEACON protocol over WiFi peer-to-peer, same `get_epoch_us / is_valid / is_leader` API surface. **120 s single-board soak: 1151 transmits, 0 failures (0.00 %), 9.6 tx/s sustained, no crash or reset.** The 802.15.4 path stays in source as documented-broken (D1) for when the IDF driver gets fixed.
  - **Host-side dual-pipeline decoder for ADR-018 byte 18-19** (ADR-110 protocol closure):
    - **Rust** (`v2/crates/wifi-densepose-hardware`): new `PpduType` enum (HtLegacy/HeSu/HeMu/HeTb/Unknown) and `Adr018Flags` struct (bw40/stbc/ldpc/ieee802154_sync_valid) on `CsiMetadata`. 6 new deterministic unit tests; **122/122 hardware-crate tests pass**.
    - **Python** (`archive/v1/src/hardware/csi_extractor.py`): `HEADER_FMT` extended from `<IBBHIIBB2x` to `<IBBHIIBBBB`; new metadata fields (`ppdu_type`, `he_capable`, `bw40`, `stbc`, `ldpc`, `ieee802154_sync_valid`). 5 new `TestAdr110ByteEncoding` cases; **11/11 parser tests pass**.
    - Both decoders match the firmware encoder bit-for-bit. Pre-ADR-110 firmware sends zeros that round-trip as `HtLegacy` + default flags — fully backwards compatible.
  - **Security fix** (`scripts/redact-secrets.py` + `generate-witness-bundle.sh`): the Python proof step was echoing `.env` contents into the bundled `verification-output.log` via Pydantic validation errors. Bundle nuked before push; added a `stdin -> stdout` redaction filter covering common token prefixes, long opaque strings, and long hex runs. Verified zero leaks on rebuild.
  - **Wave 3 — firmware v0.6.7 (LP-core full + soft-AP HE)**: two software-only unblocks for the hardware-blocked items in WITNESS-LOG-110 §B. (1) **Real LP-core motion-gate program** (`firmware/esp32-csi-node/main/lp_core/main.c` + integration in `c6_lp_core.c`). When `CONFIG_C6_LP_CORE_ENABLE=y`, the LP RISC-V coprocessor now runs a real polling program (configurable cadence via `CONFIG_C6_LP_POLL_PERIOD_US`, default 10 ms) that debounces N consecutive GPIO samples (`CONFIG_C6_LP_DEBOUNCE_SAMPLES`, default 3) and wakes the HP core via `ulp_lp_core_wakeup_main_processor()`. HP entry uses `esp_sleep_enable_ulp_wakeup` + `ESP_SLEEP_WAKEUP_ULP`. Exposes `c6_lp_core_motion_count()` and `c6_lp_core_poll_count()` getters for the witness harness. **Replaces** the v0.6.6 `esp_deep_sleep_enable_gpio_wakeup` ext1 fallback (which floored at ~10 µA, the same as the S3 ULP-FSM). The fallback path stays as the `else` branch so builds without `CONFIG_C6_LP_CORE_ENABLE` keep working unchanged — zero regression for v0.6.6-era fleets. Targets the C6 datasheet ≤5 µA average for battery seed nodes; pending INA/Joulescope measurement to confirm (`WITNESS-LOG-110 §B4`). (2) **Wi-Fi 6 soft-AP with TWT Responder=1** (`c6_softap_he.{h,c}` + `main.c` AP+STA mode switch). When `CONFIG_C6_SOFTAP_HE_ENABLE=y`, one C6 board can act as the iTWT-capable AP the bench is otherwise missing — pair with a second C6-STA board to negotiate real iTWT against a known-cooperative AP and measure deterministic CSI cadence (`WITNESS-LOG-110 §B1/B2`). SSID/PSK/channel configurable via Kconfig defaults or NVS (`softap_ssid`/`softap_psk`/`softap_chan` keys in the `ruview` namespace). Default off so existing nodes are unaffected. **Build artifacts**: S3 8 MB binary 1093 KB (47 % slack), C6 4 MB binary 1019 KB (45 % slack). Tag: `v0.6.7-esp32`.
  - **Wave 4 — firmware v0.6.8 (ESP-NOW mesh offset smoother)**: `c6_sync_espnow.c` now maintains an in-firmware exponential-moving-average of the cross-board sync offset (α = 1/8, fixed-point shift, ≈ 8-sample window at the 10 Hz beacon rate). New getter `c6_sync_espnow_get_offset_us_smoothed()`. `c6_sync_espnow_get_epoch_us()` now returns timestamps stamped from the smoothed offset once seeded — every downstream CSI-frame consumer gets bounded-jitter alignment for free, no host-side filter required. **Measured on the bench**: 5-min two-board soak (WITNESS-LOG-110 §A0.10) drops raw offset stdev 411.5 µs → smoothed 104.1 µs (**3.95× suppression** on stdev, 4.70× on peak-to-peak range) while preserving the +30 µs/min crystal-drift trajectory within 2 µs/min. **The ADR-110 §2.4 ≤100 µs multistatic alignment target that v0.6.6 designed is now empirically measured, not just stated.** Cross-board beacon match rate 99.56% over 5 min, 0 TX failures. Binary cost: +32 bytes (one int64, one bool, one getter). Diag log adds `smoothed=…` field. Tag: `v0.6.8-esp32`. **Known wiring gap (deferred)**: `csi_serialize_frame` does not yet stamp frames with `c6_sync_espnow_get_epoch_us()` — the ADR-018 frame format has no timestamp field, and adding one is a breaking change that needs an ADR update. Multistatic CSI fusion will require either an ADR-018 v2 with timestamp, or a separate UDP sync packet keyed off the existing flag bit. Tracked in WITNESS-LOG-110 §A0.11.
  - **Wave 5 — firmware v0.6.9 + v0.7.0 + host wiring (loop iter 8 → iter 26)**: closes the §A0.11 gap and lights up the substrate end-to-end across firmware → host → JSON broadcast. **Firmware**: (a) **v0.6.9-esp32** — `csi_collector.c` emits a 32-byte UDP sync packet (magic `0xC511A110`, distinct from CSI frame magic `0xC5110001`) every `CONFIG_C6_SYNC_EVERY_N_FRAMES` (default 20) CSI frames, carrying `node_id`, `local_us`, mesh-aligned `epoch_us` (from the Wave 4 smoothed offset), and the CSI sequence high-water for host-side pairing. Same UDP socket as CSI; host dispatches by leading magic. Operator-tunable cadence via the new Kconfig knob — N=1 (10 Hz) for tight multistatic, N=200 (~20 s) for low-power seeds. Live-verified on COM9+COM12 (§A0.12): follower reports `local − epoch = 1 163 565 µs`, matches the §A0.10 boot-delta measurement within 285 µs of WiFi MAC TX jitter. (b) **v0.7.0-esp32** — `csi_collector.c:221` ADR-018 byte 19 bit 4 ("cross-node sync valid") now ORs in `c6_sync_espnow_is_valid()` so frames from sync'd ESP-NOW nodes correctly advertise sync (previously only sourced from the broken 802.15.4 path — false-negative bug, §A0.13). Side effect: S3 boards now also set the bit since `c6_sync_espnow` is cross-target. **Host decoders + 25 unit tests**: Python `SyncPacketParser` + `SyncPacket` dataclass with `apply_to_local` / `mesh_aligned_us_for_sequence` / `local_minus_epoch_us` (10 tests in `TestSyncPacketParser`); Rust `wifi_densepose_hardware::SyncPacket` + `SyncPacketFlags` + `SYNC_PACKET_MAGIC` re-exported from the crate root with identical API surface (15 tests in `sync_packet::tests`). **Cross-language conformance gate** (loop iter 21): the same 32-byte canonical hex `10a111c509010600f26db70100000000c5aca501000000001400000000000000` is pinned in both test suites; if either decoder drifts from the wire, exactly one named test fires and points at the moved side. **Sensing-server wiring**: `udp_receiver_task` magic-dispatches `0xC511A110` and stores per-node `latest_sync: Option<SyncPacket>` + `latest_sync_at: Option<Instant>` on `NodeState`. New helpers: `NodeState::mesh_aligned_us(local_us)`, `NodeState::mesh_aligned_us_for_csi_frame(sequence)` (uses the per-node measured fps EMA with 5-sample warmup + 9 s staleness gate), `NodeState::observe_csi_frame_arrival(now)` (feeds `update_csi_fps_ema` α=1/8, called once per accepted CSI frame). 4 fps-EMA tests + 3 NodeSyncSnapshot serialization tests on the binary target. **Public JSON API**: `sensing_update` broadcasts now carry an optional `sync` object per node — `{offset_us, is_leader, is_valid, smoothed, sequence, csi_fps_ema, csi_fps_samples}` — `#[serde(skip_serializing_if = "Option::is_none")]` so non-mesh paths (multi-BSSID scan / synthetic-RSSI fallback / simulation) omit the key entirely. Existing pre-v0.7.0 UI clients ignore it cleanly. Documented in `docs/user-guide.md` "Per-node mesh sync (ADR-110)" section with field table, UI rendering rules, and the timestamp-recovery recipe. **Branch-coordination**: `docs/ADR-110-BRANCH-STATE.md` maps which files each of `adr-110-esp32c6` vs `feat/adr-115-ha-mqtt-matter` touches (regions are disjoint, merges should be clean line-merges). **Verification baselines**: full v2 cargo workspace at **1437 tests passing** (no regression across 17 crate batches), full `wifi-densepose-hardware` crate at **137 tests**. ADR-110 §B substrate is now end-to-end visible to UI clients and ready for ADR-029/030 multistatic CSI fusion consumption.
- **Real-time CSI introspection / low-latency tap on `wifi-densepose-sensing-server` (ADR-099).**
  New `wifi_densepose_sensing_server::introspection` module wires
  [midstream](https://github.com/ruvnet/midstream)'s `temporal-attractor` (Lyapunov +
  regime classification) and `temporal-compare` (DTW pattern matching) as a
  **parallel tap** alongside RuView's existing event pipeline — no replacement,
  no behaviour change to the existing `/ws/sensing` fan-out or `wifi-densepose-signal`
  DSP. Two new endpoints (off by default, enabled via `--introspection`):
  - `GET /ws/introspection` — newline-delimited JSON snapshots streamed at the CSI
    frame rate. Each snapshot carries `frame_count`, `regime` (Idle / Periodic /
    Transient / Chaotic / Unknown), `lyapunov_exponent`, `attractor_dim`,
    `attractor_confidence`, `regime_changed` (boolean — flips on the first frame
    after a regime transition), and `top_k_similarity[]` (highest-scoring
    signature matches against a per-deployment library).
  - `GET /api/v1/introspection/snapshot` — single-shot JSON snapshot, auth-gated
    when `RUVIEW_API_TOKEN` is set.
  Per-frame `update()` budget measured at **0.041 ms p99** on the I5 bench
  (~24× under ADR-099 D4's 1 ms target). Shape-match latency on a 1-D
  mean-amplitude L1 stand-in: **5 frames** (3.20× ratio vs the 16-frame event-path
  floor). ADR-099 D8 honestly amended — the aspirational 10× bar is contingent on
  ADR-208 Phase 2 multi-dim NPU embeddings; this release ships the tap off-by-default
  while the foundation lands. 8 lib tests + 5 latency/regression tests (`tests/introspection_latency.rs`,
  including a 200-frame noise warm-up → 10-frame motion-ramp signature benchmark).
- **Opt-in bearer-token auth on `wifi-densepose-sensing-server`'s `/api/v1/*` HTTP surface (closes #443).**
  New `wifi_densepose_sensing_server::bearer_auth` module: when the
  `RUVIEW_API_TOKEN` env var is set, every request whose path begins with
  `/api/v1/` must carry an `Authorization: Bearer <token>` header (constant-time
  compared) or the server responds `401 Unauthorized`. When the variable is
  unset or empty the middleware is a no-op — the long-standing LAN-only
  deployment posture is preserved, so this is a binary deployment-time switch
  with **no default behaviour change**. `/health*`, `/ws/sensing`, and the
  `/ui/*` static mount are intentionally never gated (orchestrator probes +
  local browsers). Startup logs which mode is active and warns when auth is on
  with a `0.0.0.0` bind. 8 unit tests on the middleware (lib test count 191 → 199).
  Resolves the security audit raised in #443.

### Changed
- **Docker image: build-time guard for the UI assets, plus a CI workflow that
  rebuilds and pushes on every change (closes #520, #514).** `docker/Dockerfile.rust`
  now `RUN`s a guard after `COPY ui/` that fails the build if any of
  `index.html` / `observatory.html` / `pose-fusion.html` / `viz.html` / the
  `observatory/` / `pose-fusion/` / `components/` / `services/` directories are
  missing, so a stale image can never be silently produced again. New
  `.github/workflows/sensing-server-docker.yml` builds the image on push to
  `main` (paths-filtered) and on `v*` tags and pushes to both
  `docker.io/ruvnet/wifi-densepose` and `ghcr.io/ruvnet/wifi-densepose` with
  `latest` + `vX.Y.Z` + `sha-<short>` tags, then smoke-tests the published
  artifact: `/health`, `/api/v1/info`, the observatory + pose-fusion UI assets,
  and the `RUVIEW_API_TOKEN` auth path (no token → 401, wrong → 401, correct
  → 200). Uses `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets for the
  Docker Hub push; ghcr.io uses the workflow's `GITHUB_TOKEN`.
- **rvCSI moved to its own repo and is now vendored as a submodule.** The 9 `rvcsi-*`
  crates (`rvcsi-core`/`-dsp`/`-events`/`-adapter-file`/`-adapter-nexmon`/`-ruvector`/
  `-runtime`/`-node`/`-cli` — added inline in #542) now live in
  [`github.com/ruvnet/rvcsi`](https://github.com/ruvnet/rvcsi): published to crates.io
  as `rvcsi-* 0.3.x`, to npm as `@ruv/rvcsi`, with a Claude Code plugin marketplace and
  a RuView-style README. RuView vendors it under `vendor/rvcsi` (alongside
  `vendor/ruvector` / `vendor/midstream` / `vendor/sublinear-time-solver`) and no longer
  carries inline copies in `v2/crates/`; consumers depend on the published crates (or the
  submodule's `crates/rvcsi-*` paths). `v2/Cargo.toml`, `CLAUDE.md`, and the README docs
  table updated accordingly. The ADRs (ADR-095, ADR-096), PRD, and DDD model stay in
  `docs/` here as the design record of the incubation.

### Fixed
- **README: corrected the camera-supervised pose-accuracy claim.** The README stated
  "92.9% PCK@20" for camera-supervised training; that figure does not appear in
  ADR-079 and is ~2.6× the ADR's own success target (>35% PCK@20). ADR-079 phases
  P7 (data collection), P8 (training + evaluation on real paired data) and P9
  (cross-room LoRA) are still `Pending`, so no measured camera-supervised PCK@20 has
  been published. README now states the proxy-supervised baseline (≈2.5%) and the
  ADR-079 target (35%+), and notes the eval phases are pending. Surfaced by the
  PowerPlatePulse training-pipeline audit (2026-05-11); 6 remaining audit findings
  tracked in the PR.
- **rvCSI `BaselineDriftDetector`: drift thresholds are now scale-relative, not absolute.**
  The detector compared `mean_amplitude` against its EWMA baseline with absolute
  thresholds (`anomaly_threshold = 1.0`, `drift_threshold = 0.15`) — fine for the
  synthetic unit tests (amplitudes ≈ 1.0), but raw ESP32 CSI is `int8` I/Q with
  amplitudes up to ~128, so the window-to-window RMS distance is routinely 5–50 ≫ 1.0
  and `AnomalyDetected` fired on ~96 % of windows (319/331 on a real node-1 capture).
  Drift is now `‖current − baseline‖₂ / ‖baseline‖₂` (a fraction, with an `eps` floor
  for a degenerate near-zero baseline), so one tuning works across raw-`int8` ESP32,
  `int16`-scaled Nexmon, and baseline-subtracted streams alike — `AnomalyDetected`
  drops to 40/331 on the same data, the existing detector tests still pass, and a
  `baseline_drift_is_scale_invariant_no_anomaly_storm` regression test was added.
  ADR-095 D13 / ADR-096 §2.1, §5 updated. Surfaced by an end-to-end test against
  real ESP32 CSI (a 7,000-frame node-1 capture; transcoder at
  `scripts/esp32_jsonl_to_rvcsi.py`).

### Added
- **rvCSI — edge RF sensing runtime (design + first implementation).** New subsystem **rvCSI**: a Rust-first / TypeScript-accessible / hardware-abstracted edge RF sensing runtime that normalizes WiFi CSI from Nexmon, ESP32, Intel, Atheros, file and replay sources into one validated `CsiFrame` schema, runs reusable DSP, emits typed confidence-scored events, and bridges to RuVector RF memory, an MCP tool server and a TS SDK.
  - **Design docs:** `docs/prd/rvcsi-platform-prd.md` (purpose, users, success criteria, FR1–FR10, NFRs, system architecture, data model); `docs/adr/ADR-095-rvcsi-edge-rf-sensing-platform.md` (the 15 architectural decisions: Rust core, C-at-the-boundary, TS SDK via napi-rs, normalized schema, validate-before-FFI, CSI-as-temporal-delta, RuVector as RF memory, replayability, detection≠decision, local-first, read-first/write-gated MCP, mandatory quality scoring, versioned calibration, plugin adapters); `docs/adr/ADR-096-rvcsi-ffi-crate-layout.md` (crate topology, the napi-c shim record format & contract, the napi-rs Node surface, build/test invariants); `docs/ddd/rvcsi-domain-model.md` (7 bounded contexts: Capture, Validation, Signal, Calibration, Event, Memory, Agent — with aggregates, invariants, context map and domain services). Indexed in `docs/adr/README.md` and `docs/ddd/README.md`.
  - **Crates** (9 new `v2/crates/rvcsi-*` workspace members): `rvcsi-core` (normalized `CsiFrame`/`CsiWindow`/`CsiEvent` schema, `AdapterProfile`, `CsiSource` plugin trait, id newtypes + `IdGenerator`, `RvcsiError`, the `validate_frame` pipeline + quality scoring; `forbid(unsafe_code)`); `rvcsi-adapter-nexmon` — the **napi-c** seam: `native/rvcsi_nexmon_shim.{c,h}` (the only C in the runtime — allocation-free, bounds-checked, ABI `1.1`), compiled via `build.rs`+`cc`, handling **two byte formats** — the compact self-describing "rvCSI Nexmon record", and the **real nexmon_csi UDP payload** (the 18-byte `magic 0x1111 · rssi · fctl · src_mac · seq · core/stream · chanspec · chip_ver` header + `nsub` int16 I/Q samples, the modern BCM43455c0/4358/4366c0 export read by CSIKit/`csireader.py`), with a Broadcom d11ac **chanspec decoder** (channel/bandwidth/band) — plus a pure-Rust **libpcap reader** (classic `.pcap`, all byte-order/timestamp-resolution magics, Ethernet/raw-IPv4/Linux-SLL link types) and a **Nexmon-chip / Raspberry-Pi-model registry** (`NexmonChip` / `RaspberryPiModel` — including the **Raspberry Pi 5** (CYW43455/BCM43455c0, same wireless as the Pi 4 — 20/40/80 MHz, 2.4+5 GHz, 64/128/256 subcarriers), the Pi 3B+/4/400, and the Pi Zero 2 W (BCM43436b0); `nexmon_adapter_profile` / `raspberry_pi_profile` build the per-chip `AdapterProfile`; `chip_ver` words auto-resolve to a chip). Wrapped by a documented `ffi` module and two `CsiSource`s: `NexmonAdapter` (record buffers) and `NexmonPcapAdapter` (real nexmon_csi UDP inside a `tcpdump -i wlan0 dst port 5500 -w csi.pcap` capture — the pcap timestamp stamps each frame; the chip is auto-detected from `chip_ver`, overridable via `.with_pi_model(Pi5)` / `.with_chip(...)`). `rvcsi-dsp` (DC removal, phase unwrap, smoothing, Hampel/MAD filter, sliding variance, baseline subtraction, motion-energy/presence/confidence features, heuristic breathing-band estimate, non-destructive `SignalPipeline`); `rvcsi-events` (`WindowBuffer`, the `EventDetector` trait + presence/motion/quality/baseline-drift state machines, `EventPipeline`; the baseline-drift detector uses **scale-relative** thresholds — drift as a fraction of the baseline's RMS magnitude — so one tuning works across raw-`int8` ESP32, `int16`-scaled Nexmon, and baseline-subtracted streams alike); `rvcsi-adapter-file` (the `.rvcsi` JSONL capture format, `FileRecorder`, `FileReplayAdapter` deterministic replay); `rvcsi-ruvector` (deterministic window/event embeddings, `cosine_similarity`, the `RfMemoryStore` trait, `InMemoryRfMemory` + `JsonlRfMemory` — a standin until the production RuVector binding); `rvcsi-runtime` (the no-FFI composition layer: `CaptureRuntime` = `CsiSource` + `validate_frame` + `SignalPipeline` + `EventPipeline`, plus one-shot helpers `summarize_capture`/`decode_nexmon_records`/`decode_nexmon_pcap`/`summarize_nexmon_pcap`/`events_from_capture`/`export_capture_to_rf_memory`); `rvcsi-node` — the **napi-rs** seam (a `["cdylib","rlib"]` Node addon, `build.rs` runs `napi_build::setup()`; thin `#[napi]` wrappers over `rvcsi-runtime` — `nexmonDecodeRecords`/`nexmonDecodePcap` (with optional `chip`)/`inspectNexmonPcap`/`decodeChanspec`/`nexmonChipName`/`nexmonProfile`/`nexmonChips`/`inspectCaptureFile`/`eventsFromCaptureFile`/`exportCaptureToRfMemory` + an `RvcsiRuntime` streaming class; everything that crosses to JS is a validated/normalized struct serialized to JSON); `rvcsi-cli` (the `rvcsi` binary: `record` (Nexmon-dump *or* `--source nexmon-pcap [--chip pi5]` → `.rvcsi`), `inspect`, `inspect-nexmon`, `nexmon-chips`, `decode-chanspec`, `replay`, `stream`, `events`, `health`, `calibrate` v0-baseline, `export ruvector`). Plus the `@ruv/rvcsi` npm package (`package.json`/`index.js`/`index.d.ts`/`README`/`__test__`) alongside `rvcsi-node` — a curated JS surface that parses the addon's JSON into plain `CsiFrame`/`CsiWindow`/`CsiEvent`/`SourceHealth`/`CaptureSummary`/`NexmonPcapSummary`/`DecodedChanspec` objects, with a lazy native-addon load.
  - **Tests:** 169 across the rvcsi crates (core 29, dsp 28, events 19 — incl. a baseline-drift scale-invariance regression, adapter-file 20 + 1 doctest, adapter-nexmon 28 — round-tripping through the C shim and synthetic libpcap files, incl. Pi 5 / chip-detection, ruvector 20 + 1 doctest, runtime 13, cli 10), 0 failures; all rvcsi crates build together and are clippy-clean (`rvcsi-node` under `deny(clippy::all)`); `forbid(unsafe_code)` everywhere except `rvcsi-adapter-nexmon` (FFI, every `unsafe` block documented). Also exercised end-to-end against a real 7,000-frame ESP32 node-1 capture (transcoded with `scripts/esp32_jsonl_to_rvcsi.py` — the stand-in for the not-yet-shipped `record --source esp32-jsonl`): `rvcsi inspect`/`replay`/`calibrate`/`events` all run on real hardware data. Not yet wired in: live radio capture, `rvcsi-adapter-esp32` (live serial/UDP ESP32 source), the WebSocket daemon (`rvcsi-daemon`), the MCP tool server (`rvcsi-mcp`), and the legacy nexmon *packed-float* CSI export — follow-ups on top of these crates.
- **`wifi-densepose-train`: `signal_features` module — wires `wifi-densepose-signal` into the training pipeline.** `wifi-densepose-signal` was previously a phantom dependency of `wifi-densepose-train` (listed in `Cargo.toml`, never imported). New `wifi_densepose_train::signal_features::extract_signal_features` (and `CsiSample::signal_features()`) run a windowed CSI observation's centre frame through `wifi_densepose_signal::features::FeatureExtractor`, producing a fixed-length (`FEATURE_LEN = 12`) amplitude/phase/PSD feature vector — the hook for a future vitals / multi-task supervision head (breathing- and heart-rate-band power are read off the PSD summary). The vector is produced on demand and not yet fed back into the loss. Surfaced by the 2026-05-11 training-pipeline audit (findings #1 "vitals features absent from training" and #2 "`wifi-densepose-signal` ghost dep").
- **`wifi-densepose-train`: `TrainingConfig` subcarrier-layout presets + a real-loader integration test.** New `TrainingConfig::for_subcarriers(native, target)` plus named presets `ht40_192()` (≈192-sc ESP32 HT40 → 56) and `multiband_168()` (168-sc ADR-078 multi-band mesh → 56), so non-MM-Fi CSI shapes are first-class instead of requiring manual `native_subcarriers`/`num_subcarriers` overrides; field docs now list the supported source counts and the multi-NIC mapping. New `tests/test_real_loader.rs` round-trips synthetic CSI through `.npy` files → `MmFiDataset::discover`/`get` (including the subcarrier-interpolation branch and the empty-root case) — exercising the on-disk loader path the deterministic `verify-training` proof intentionally bypasses. Addresses training-pipeline audit findings #6 (56-sc/1-NIC config default) and #7 (multi-band mesh not in config); the #4 concern ("proof uses synthetic data") is reframed — the proof *should* use a reproducible source, and this test covers the real loader it skips.

### Fixed
- **HuggingFace `MODEL_CARD.md`: marked the PIR/BME280 environmental-sensor ground-truth path as planned, not implemented** (training-pipeline audit finding #3) — the card presented PIR/BME280 weak-label fine-tuning as a current capability; there is no env-sensor ingestion in the training pipeline today.
- **README: corrected the camera-supervised pose-accuracy claim** (audit finding #5; see PR #535) — "92.9% PCK@20" → the ADR-079 target (35%+; proxy baseline 35.3%), noting P7/P8/P9 are pending.

### Added
- **`RollingP95` adaptive feature normalizer** (`v2/crates/wifi-densepose-sensing-server`) —
  Streaming P95 estimator (600-sample / ~30 s sliding window) that self-calibrates
  feature normalization to whatever distribution the deployment produces. Replaces
  fixed-scale denominators (`variance/300`, `motion/250`, `spectral/500`) which saturated
  when live ESP32 values exceeded those limits, collapsing dynamic range to zero.
  Cold-start (<60 samples) falls back to the legacy denominators so day-0 behaviour
  is preserved. Deployment-neutral: no hardcoded values. (ADR-044 §5.2)

- **`dedup_factor` runtime configuration API** (`v2/crates/wifi-densepose-sensing-server`) —
  Exposes the multi-node person-count deduplication divisor at runtime via REST:
  - `GET  /api/v1/config/dedup-factor` — read current value.
  - `POST /api/v1/config/dedup-factor` — set value (clamped 1.0–10.0, persisted).
  - `POST /api/v1/config/ground-truth` — auto-tunes `dedup_factor` from a known
    person count (`{"count": N}`); derives optimal divisor from current node-sum.
  Config is persisted to `data/config.json` and reloaded on restart. (ADR-044 §5.3)

- **`nvsim` crate — deterministic NV-diamond magnetometer pipeline simulator** (ADR-089) —
  New standalone leaf crate at `v2/crates/nvsim` modeling a forward-only
  magnetic sensing path: scene → source synthesis (Biot–Savart, dipole,
  current loop, ferrous induced moment) → material attenuation
  (Air/Drywall/Brick/Concrete/Reinforced/SteelSheet) → NV ensemble
  (4 〈111〉 axes, ODMR linear-readout proxy, shot-noise floor per
  Wolf 2015 / Barry 2020) → 16-bit ADC + lock-in demodulation →
  fixed-layout `MagFrame` records → SHA-256 witness. Six-pass build
  per `docs/research/quantum-sensing/15-nvsim-implementation-plan.md`.
  50 tests, ~4.5 M samples/s on x86_64 (4500× the Cortex-A53 1 kHz
  acceptance gate), pinned reference witness
  `cc8de9b01b0ff5bd97a6c17848a3f156c174ea7589d0888164a441584ec593b4`
  for byte-equivalence regression. WASM-ready by construction
  (zero `std::time/fs/env/process/thread`); builds cleanly for
  `wasm32-unknown-unknown`. ADR-090 (Proposed, conditional) tracks the
  optional Lindblad/Hamiltonian extension if AC magnetometry, MW power
  saturation, hyperfine spectroscopy, or pulsed protocols become required.

### Fixed
- **WebSocket broadcast handler now handles Lagged events gracefully and sends periodic ping keepalives to prevent dashboard disconnects** —
  `handle_ws_client` and `handle_ws_pose_client` in `wifi-densepose-sensing-server`
  were treating `RecvError::Lagged` as a fatal error, causing instant disconnect
  when clients fell behind the 256-frame broadcast buffer at 10 Hz ingest.
  Clients would reconnect, immediately lag again, and rapid-cycle every 2–4 s.
  `Lagged` now continues (drops missed frames, logs debug) rather than breaking.
  Added 30 s ping keepalive on the sensing handler to prevent proxy idle timeouts.
- **Ghost skeletons in live UI with multi-node ESP32 setups** (#420, ADR-082) —
  `tracker_bridge::tracker_to_person_detections` documented itself as filtering
  to `is_alive()` tracks but in fact passed every non-Terminated track to the
  WebSocket stream. `Lost` tracks — kept inside `reid_window` for
  re-identification but not currently observed — were rendering as phantom
  skeletons, accumulating to 22-24 with 3 nodes × 10 Hz CSI while
  `estimated_persons` correctly reported 1. Added
  `PoseTracker::confirmed_tracks()` (Tentative + Active only) and rewired the
  bridge to use it. Lost tracks remain in the tracker for re-ID; they just
  no longer ship to the UI. Regression test:
  `test_lost_tracks_excluded_from_bridge_output`.
- **Rust workspace build with `--no-default-features` on Windows** (#366, #415) —
  `wifi-densepose-mat`, `wifi-densepose-sensing-server`, and `wifi-densepose-train`
  all depended on `wifi-densepose-signal` with default features enabled, which
  pulled `ndarray-linalg` → `openblas-src` → vcpkg/system-BLAS through the entire
  workspace. `--no-default-features` at the workspace root then could not opt out
  of BLAS, breaking `cargo build` / `cargo test` on Windows without vcpkg. All
  three consumers now declare `wifi-densepose-signal = { ..., default-features = false }`,
  so `cargo test --workspace --no-default-features` builds cleanly without
  vcpkg/openblas. Validated: 1,538 tests pass, 0 fail, 8 ignored.
- **`signal` test `test_estimate_occupancy_noise_only` failed without `eigenvalue`** —
  The test unwrapped the `NotCalibrated` stub returned when the BLAS-backed
  `estimate_occupancy` is compiled out. Gated with `#[cfg(feature = "eigenvalue")]`
  so it only runs when the real implementation is available.

## [v0.6.2-esp32] — 2026-04-20

Firmware release cutting ADR-081 and the Timer Svc stack fix discovered during
on-hardware validation. Cut from `main` at commit pointing to this entry.
Tested on ESP32-S3 (QFN56 rev v0.2, MAC `3c:0f:02:e9:b5:f8`), 30 s continuous
run: no crashes, 149 `rv_feature_state_t` emissions (~5 Hz), medium/slow ticks
firing cleanly, HEALTH mesh packets sent.

### Fixed
- **Firmware: Timer Svc stack overflow on ADR-081 fast loop** — `emit_feature_state()` runs inside the FreeRTOS Timer Svc task via the fast-loop callback; it calls `stream_sender` network I/O which pushes past the ESP-IDF 2 KiB default timer stack and panics ~1 s after boot. Bumped `CONFIG_FREERTOS_TIMER_TASK_STACK_DEPTH` to 8 KiB in `sdkconfig.defaults`, `sdkconfig.defaults.template`, and `sdkconfig.defaults.4mb`. Follow-up (tracked separately): move heavy work out of the timer daemon into a dedicated worker task.
- **Firmware: `adaptive_controller.c` implicit declaration** (#404) — `fast_loop_cb` called `emit_feature_state()` before its static definition, triggering `-Werror=implicit-function-declaration`. Added a forward declaration above the first use.

### Changed
- **CI: firmware build matrix (8MB + 4MB)** — `firmware-ci.yml` now matrix-builds both the default 8MB (`sdkconfig.defaults`) and 4MB SuperMini (`sdkconfig.defaults.4mb`) variants, uploading distinct artifacts and producing variant-named release binaries (`esp32-csi-node.bin` / `esp32-csi-node-4mb.bin`, `partition-table.bin` / `partition-table-4mb.bin`).

### Added
- **ADR-081: Adaptive CSI Mesh Firmware Kernel** — New 5-layer architecture
  (Radio Abstraction Layer / Adaptive Controller / Mesh Sensing Plane /
  On-device Feature Extraction / Rust handoff) that reframes the existing
  ESP32 firmware modules as components of a chipset-agnostic kernel. ADR
  in `docs/adr/ADR-081-adaptive-csi-mesh-firmware-kernel.md`. Goal: swap
  one radio family for another without changing the Rust signal /
  ruvector / train / mat crates.
- **Firmware: radio abstraction vtable (`rv_radio_ops_t`)** — New
  `firmware/esp32-csi-node/main/rv_radio_ops.{h}` defines the
  chipset-agnostic ops (init, set_channel, set_mode, set_csi_enabled,
  set_capture_profile, get_health), profile enum
  (`RV_PROFILE_PASSIVE_LOW_RATE` / `ACTIVE_PROBE` / `RESP_HIGH_SENS` /
  `FAST_MOTION` / `CALIBRATION`), and health snapshot struct.
  `rv_radio_ops_esp32.c` provides the ESP32 binding wrapping
  `csi_collector` + `esp_wifi_*`. A second binding (mock or alternate
  chipset) is the portability acceptance test for ADR-081.
- **Firmware: `rv_feature_state_t` packet (magic `0xC5110006`)** — New
  60-byte compact per-node sensing state (packed, verified by
  `_Static_assert`) in `firmware/esp32-csi-node/main/rv_feature_state.h`:
  motion, presence, respiration BPM/conf, heartbeat BPM/conf, anomaly
  score, env-shift score, node coherence, quality flags, IEEE CRC32.
  Replaces raw ADR-018 CSI as the default upstream stream (~99.7%
  bandwidth reduction: 300 B/s at 5 Hz vs. ~100 KB/s raw).
- **Firmware: mock radio ops binding for QEMU** — New
  `firmware/esp32-csi-node/main/rv_radio_ops_mock.c`, compiled only when
  `CONFIG_CSI_MOCK_ENABLED`. Satisfies ADR-081's portability acceptance
  test: a second `rv_radio_ops_t` binding compiles and runs against the
  same controller + mesh-plane code as the ESP32 binding.
- **Firmware: feature-state emitter wired into controller fast loop** —
  `adaptive_controller.c` now emits one 60-byte `rv_feature_state_t` per
  fast tick (default 200 ms → 5 Hz), pulling from the latest edge vitals
  and controller observation. This is the first end-to-end Layer 4/5
  path for ADR-081.
- **Firmware: `csi_collector_get_pkt_yield_per_sec()` /
  `_get_send_fail_count()` accessors** — Expose the CSI callback rate
  and UDP send-failure counter so the ESP32 radio ops binding can
  populate `rv_radio_health_t.pkt_yield_per_sec` and `.send_fail_count`,
  closing the adaptive controller's observation loop.
- **Firmware: host-side unit test suite for ADR-081 pure logic** — New
  `firmware/esp32-csi-node/tests/host/` (Makefile + 2 test files + shim
  `esp_err.h`). Exercises `adaptive_controller_decide()` (9 test cases:
  degraded gate on pkt-yield collapse + coherence loss, anomaly > motion,
  motion → SENSE_ACTIVE, aggressive cadence, stable presence →
  RESP_HIGH_SENS, empty-room default, hysteresis, NULL safety) and
  `rv_feature_state_*` helpers (size assertion, IEEE CRC32 known
  vectors, determinism, receiver-side verification). 33/33 assertions
  pass. Benchmarks: decide() 3.2 ns/call, CRC32(56 B) 614 ns/pkt
  (87 MB/s), full finalize() 616 ns/call. Pure function
  `adaptive_controller_decide()` extracted to
  `adaptive_controller_decide.c` so the firmware build and the host
  tests share a single source-of-truth implementation.
- **Scripts: `validate_qemu_output.py` ADR-081 checks** — Validator
  (invoked by ADR-061 `scripts/qemu-esp32s3-test.sh` in CI) gains three
  checks for adaptive controller boot line, mock radio ops
  registration, and slow-loop heartbeat, so QEMU runs regression-gate
  Layer 1/2 presence.
- **Firmware: ADR-081 Layer 3 mesh sensing plane** — New
  `firmware/esp32-csi-node/main/rv_mesh.{h,c}` defines 4 node roles
  (Anchor / Observer / Fusion relay / Coordinator), 7 on-wire message
  types (TIME_SYNC, ROLE_ASSIGN, CHANNEL_PLAN, CALIBRATION_START,
  FEATURE_DELTA, HEALTH, ANOMALY_ALERT), 3 authorization classes
  (None / HMAC-SHA256-session / Ed25519-batch), `rv_node_status_t`
  (28 B), `rv_anomaly_alert_t` (28 B), `rv_time_sync_t`,
  `rv_role_assign_t`, `rv_channel_plan_t`, `rv_calibration_start_t`.
  Pure-C encoder/decoder (`rv_mesh_encode()` / `rv_mesh_decode()`) with
  16-byte envelope + payload + IEEE CRC32 trailer; convenience encoders
  for each message type. Controller now emits `HEALTH` every slow-loop
  tick (30 s default) and `ANOMALY_ALERT` on state transitions to ALERT
  or DEGRADED. Host tests: `test_rv_mesh` exercises 27 assertions
  covering roundtrip, bad magic, truncation, CRC flipping, oversize
  payload rejection, and encode+decode throughput (1.0 μs/roundtrip
  on host).
- **Rust: ADR-081 Layer 1/3 mirror module** — New
  `crates/wifi-densepose-hardware/src/radio_ops.rs` mirrors the
  firmware-side `rv_radio_ops_t` vtable as the Rust `RadioOps` trait
  (init, set_channel, set_mode, set_csi_enabled, set_capture_profile,
  get_health) and provides `MockRadio` for offline testing.
  Also mirrors the `rv_mesh.h` types (`MeshHeader`, `NodeStatus`,
  `AnomalyAlert`, `MeshRole`, `MeshMsgType`, `AuthClass`) and ships
  byte-identical `crc32_ieee()`, `decode_mesh()`, `decode_node_status()`,
  `decode_anomaly_alert()`, and `encode_health()`. Exported from
  `lib.rs`. 8 unit tests pass; `crc32_matches_firmware_vectors`
  verifies parity with the firmware-side test vectors
  (`0xCBF43926` for `"123456789"`, `0xD202EF8D` for single-byte zero),
  and `mesh_constants_match_firmware` asserts `MESH_MAGIC`,
  `MESH_VERSION`, `MESH_HEADER_SIZE`, and `MESH_MAX_PAYLOAD` match
  `rv_mesh.h` byte-for-byte. Satisfies ADR-081's portability
  acceptance test: signal/ruvector/train/mat crates are untouched.
- **Firmware: adaptive controller** — New
  `firmware/esp32-csi-node/main/adaptive_controller.{c,h}` implements
  the three-loop closed-loop control specified by ADR-081: fast
  (~200 ms) for cadence and active probing, medium (~1 s) for channel
  selection and role transitions, slow (~30 s) for baseline
  recalibration. Pure `adaptive_controller_decide()` policy function is
  exposed in the header for offline unit testing. Default policy is
  conservative (`enable_channel_switch` and `enable_role_change` off);
  Kconfig surface added under "Adaptive Controller (ADR-081)".

### Fixed
- **Firmware: SPI flash cache crash under high CSI callback pressure** (RuView#396, #397) — ESP32-S3 nodes crashed in `cache_ll_l1_resume_icache` / `wDev_ProcessFiq` after ~2400 callbacks when the promiscuous filter admitted DATA frames at 100–500 Hz. Fixed by narrowing the filter mask to `WIFI_PROMIS_FILTER_MASK_MGMT` (~10 Hz beacons), adding a 50 Hz early callback rate gate (`CSI_MIN_PROCESS_INTERVAL_US`) that drops excess callbacks before any processing work, and enabling `CONFIG_ESP_WIFI_EXTRA_IRAM_OPT=y` as defense-in-depth. Stability validated with a 4-min-per-node soak.
- **Firmware: `filter_mac` / `node_id` clobber by WiFi driver init** (#232, #375, #385, #386, #390, #397) — `g_nvs_config` can be corrupted during `wifi_init_sta()` on some devices (confirmed on `80:b5:4e:c1:be:b8`), reverting `node_id` to the Kconfig default and producing garbage MAC-filter reads in the CSI callback (100–500 Hz). New `csi_collector_set_node_id()` API called from `app_main()` **before** `wifi_init_sta()` captures both fields into module-local statics (`s_node_id`, `s_filter_mac`, `s_filter_mac_set`). `csi_collector_init()` now runs a canary that distinguishes "early≠g_nvs_config" (corruption confirmed) from a no-op match. All CSI runtime paths use the defensive copies exclusively.
- **Firmware: `edge_processing` sample rate mismatch** (#397) — `estimate_bpm_zero_crossing()` was called with a hard-coded `sample_rate = 20.0f`, but MGMT-only promiscuous delivers ~10 Hz. Breathing and heart-rate reports were 2× too high. Corrected to `10.0f` with an explicit comment tying it to the callback rate.
- **`provision.py` esptool command form** (#391, #397) — ESP-IDF v5.4 bundles `esptool 4.10.0`, which only accepts `write_flash` (underscore). Standalone `pip install esptool` v5.x accepts both forms but prefers `write-flash`. #391 switched to `write-flash` which broke the documented ESP-IDF Python venv flow; #397 reverts to `write_flash` (works with both esptool 4.x and 5.x) with an inline comment warning future maintainers not to "re-fix" it.
- **`provision.py` esptool v5 dry-run hint** (#391) — Stale `write_flash` (underscore) syntax in the dry-run manual-flash hint now uses `write-flash` (hyphenated) for esptool >= 5.x. The primary flash command was already correct.
- **`provision.py` silent NVS wipe** (#391) — The script replaces the entire `csi_cfg` NVS namespace on every run, so partial invocations were silently erasing WiFi credentials and causing `Retrying WiFi connection (10/10)` in the field. Now refuses to run without `--ssid`, `--password`, and `--target-ip` unless `--force-partial` is passed. `--force-partial` prints a warning listing which keys will be wiped.
- **Firmware: defensive `node_id` capture** (#232, #375, #385, #386, #390) — Users on multi-node deployments reported `node_id` reverting to the Kconfig default (`1`) in UDP frames and in the `csi_collector` init log, despite NVS loading the correct value. The root cause (memory corruption of `g_nvs_config`) has not been definitively isolated, but the UDP frame header is now tamper-proof: `csi_collector_init()` captures `g_nvs_config.node_id` into a module-local `s_node_id` once, and `csi_serialize_frame()` plus all other consumers (`edge_processing.c`, `wasm_runtime.c`, `display_ui.c`, `swarm_bridge_init`) read it via the new `csi_collector_get_node_id()` accessor. A canary logs `WARN` if `g_nvs_config.node_id` diverges from `s_node_id` at end-of-init, helping isolate the upstream corruption path. Validated on attached ESP32-S3 (COM8): NVS `node_id=2` propagates through boot log, capture log, init log, and byte[4] of every UDP frame.

### Docs
- **CHANGELOG catch-up** (#367) — Added missing entries for v0.5.5, v0.6.0, and v0.7.0 releases.

## [v0.7.0] — 2026-04-06

Model release (no new firmware binary). Firmware remains at v0.6.0-esp32.

### Added
- **Camera ground-truth training pipeline (ADR-079)** — End-to-end supervised WiFlow pose training using MediaPipe + real ESP32 CSI.
  - `scripts/collect-ground-truth.py` — MediaPipe PoseLandmarker webcam capture (17 COCO keypoints, 30fps), synchronized with CSI recording over nanosecond timestamps.
  - `scripts/align-ground-truth.js` — Time-aligns camera keypoints with 20-frame CSI windows by binary search, confidence-weighted averaging.
  - `scripts/train-wiflow-supervised.js` — 3-phase curriculum training (contrastive → supervised SmoothL1 → bone/temporal refinement) with 4 scale presets (lite/small/medium/full).
  - `scripts/eval-wiflow.js` — PCK@10/20/50, MPJPE, per-joint breakdown, baseline proxy mode.
  - `scripts/record-csi-udp.py` — Lightweight ESP32 CSI UDP recorder (no Rust build required).
- **ruvector optimizations (O6-O10)** — Subcarrier selection (70→35, 50% reduction), attention-weighted subcarriers, Stoer-Wagner min-cut person separation, multi-SPSA gradient estimation, Mac M4 Pro training via Tailscale.
- **Scalable WiFlow presets** — `lite` (189K params, ~19 min) through `full` (7.7M params, ~8 hrs) to match dataset size.
- **Pre-trained WiFlow v1 model** — 92.9% PCK@20, 974 KB, 186,946 params. Published to [HuggingFace](https://huggingface.co/ruv/ruview) under `wiflow-v1/`.

### Validated
- **92.9% PCK@20** pose accuracy from a 5-minute data collection session with one $9 ESP32-S3 and one laptop webcam.
- Training pipeline validated on real paired data: 345 samples, 19 min training, eval loss 0.082, bone constraint 0.008.

## [v0.6.0-esp32] — 2026-04-03

### Added
- **Pre-trained CSI sensing weights published** — First official pre-trained models on [HuggingFace](https://huggingface.co/ruv/ruview). `model.safetensors` (48 KB), `model-q4.bin` (8 KB 4-bit), `model-q2.bin` (4 KB), `presence-head.json`, per-node LoRA adapters.
- **17 sensing applications** — Sleep monitor, apnea detector, stress monitor, gait analyzer, RF tomography, passive radar, material classifier, through-wall detector, device fingerprint, and more. Each as a standalone `scripts/*.js`.
- **ADRs 069-078** — 10 new architecture decisions covering Cognitum Seed integration, self-supervised pretraining, ruvllm pipeline, WiFlow architecture, channel hopping, SNN, MinCut person separation, CNN spectrograms, novel RF applications, multi-frequency mesh.
- **Kalman tracker** (PR #341 by @taylorjdawson) — temporal smoothing of pose keypoints.

### Fixed
- Security fix merged via PR #310.

### Performance
- Presence detection: 100% accuracy on 60,630 overnight samples. *(Retracted — that recording was single-class (one sleeping person, 6,062/6,063 frames "present"), so a constant "yes" scores ~99.98%. Superseded by the honest 82.3% held-out temporal-triplet metric; see [#882](https://github.com/ruvnet/RuView/issues/882). Kept here as the in-place public record.)*
- Inference: 0.008 ms per sample, 164K embeddings/sec.
- Contrastive self-supervised training: 51.6% improvement over baseline.

## [v0.5.5-esp32] — 2026-04-03

### Added
- **WiFlow SOTA architecture (ADR-072)** — TCN + axial attention pose decoder, 1.8M params, 881 KB at 4-bit. 17 COCO keypoints from CSI amplitude only (no phase).
- **Multi-frequency mesh scanning (ADR-073)** — ESP32 nodes hop across channels 1/3/5/6/9/11 at 200ms dwell. Neighbor WiFi networks used as passive radar illuminators. Null subcarriers reduced from 19% to 16%.
- **Spiking neural network (ADR-074)** — STDP online learning, adapts to new rooms in <30s with no labels, 16-160x less compute than batch training.
- **MinCut person counting (ADR-075)** — Stoer-Wagner min-cut on subcarrier correlation graph. Fixes #348 (was always reporting 4 people).
- **CNN spectrogram embeddings (ADR-076)** — Treat 64×20 CSI as an image, produce 128-dim environment fingerprints (0.95+ same-room similarity).
- **Graph transformer fusion** — Multi-node CSI fusion via GATv2 attention (replaces naive averaging).
- **Camera-free pose training pipeline** — Trains 17-keypoint model from 10 sensor signals with no camera required.

### Fixed
- **#348 person counting** — MinCut correctly counts 1-4 people (24/24 validation windows).

## [v0.5.4-esp32] — 2026-04-02

### Added
- **ADR-069: ESP32 CSI → Cognitum Seed RVF ingest pipeline** — Live-validated pipeline connecting ESP32-S3 CSI sensing to Cognitum Seed (Pi Zero 2 W) edge intelligence appliance. 339 vectors ingested, 100% kNN validation, SHA-256 witness chain verified.
- **Feature vector packet (magic 0xC5110003)** — New 48-byte packet with 8 normalized dimensions (presence, motion, breathing, heart rate, phase variance, person count, fall, RSSI) sent at 1 Hz alongside vitals.
- **`scripts/seed_csi_bridge.py`** — Python bridge: UDP listener → HTTPS ingest with bearer token auth, `--validate` (kNN + PIR ground truth), `--stats`, `--compact` modes, hash-based vector IDs, NaN/inf rejection, source IP filtering, retry logic.
- **Arena Physica research** — 26 research documents in `docs/research/` covering Maxwell's equations in WiFi sensing, Arena Physica Studio analysis, SOTA WiFi sensing 2025-2026, GOAP implementation plan for ESP32 + Pi Zero.
- **Cognitum Seed MCP integration** — 114-tool MCP proxy enables AI assistants to query sensing state, vectors, witness chain, and device status directly.

### Fixed
- **Compressed frame magic collision** — Reassigned compressed frame magic from `0xC5110003` to `0xC5110005` to free `0xC5110003` for feature vectors.
- **Uninitialized `s_top_k[0]` read** — Guarded variance computation against `s_top_k_count == 0` in `send_feature_vector()`.
- **Presence score normalization** — Bridge now divides by 15.0 instead of clamping, preserving dynamic range for raw values 1.41-14.92.
- **Stale magic references** — Updated ADR-039, DDD model to reflect `0xC5110005` for compressed frames.

### Security
- **Credential exposure remediation** — Removed hardcoded WiFi passwords and bearer tokens from source files. Added NVS binary/CSV patterns to `.gitignore`. Environment variable fallback for bearer token.
- **NaN/Inf injection prevention** — Bridge validates all feature dimensions are finite before Seed ingest.
- **UDP source filtering** — `--allowed-sources` argument restricts packet acceptance to known ESP32 IPs.

### Changed
- Wire format table now includes 6 magic numbers: `0xC5110001` (raw), `0xC5110002` (vitals), `0xC5110003` (features), `0xC5110004` (WASM events), `0xC5110005` (compressed), `0xC5110006` (fused vitals).

## [v0.5.3-esp32] — 2026-03-30

### Added
- **Cross-node RSSI-weighted feature fusion** — Multiple ESP32 nodes fuse CSI features using RSSI-based weighting. Closer node gets higher weight. Reduces variance noise by 29%, keypoint jitter by 72%.
- **DynamicMinCut person separation** — Uses `ruvector_mincut::DynamicMinCut` on the subcarrier temporal correlation graph to detect independent motion clusters. Replaces variance-based heuristic for multi-person counting.
- **RSSI-based position tracking** — Skeleton position driven by RSSI differential between nodes. Walk between ESP32s and the skeleton follows you.
- **Per-node state pipeline (ADR-068)** — Each ESP32 node gets independent `HashMap<u8, NodeState>` with frame history, classification, vitals, and person count. Fixes #249 (the #1 user-reported issue).
- **RuVector Phase 1-3 integration** — Subcarrier importance weighting, temporal keypoint smoothing (EMA), coherence gating, skeleton kinematic constraints (Jakobsen relaxation), compressed pose history.
- **Client-side lerp smoothing** — UI keypoints interpolate between frames (alpha=0.15) for fluid skeleton movement.
- **Multi-node mesh tests** — 8 integration tests covering 1-255 node configurations.
- **`wifi_densepose` Python package** — `from wifi_densepose import WiFiDensePose` now works (#314).

### Fixed
- **Watchdog crash on busy LANs (#321)** — Batch-limited edge_dsp to 4 frames before 20ms yield. Fixed idle-path busy-spin (`pdMS_TO_TICKS(5)==0`).
- **No detection from edge vitals (#323)** — Server now generates `sensing_update` from Tier 2+ vitals packets.
- **RSSI byte offset mismatch (#332)** — Server parsed RSSI from wrong byte (was reading sequence counter).
- **Stack overflow risk** — Moved 4KB of BPM scratch buffers from stack to static storage.
- **Stale node memory leak** — `node_states` HashMap evicts nodes inactive >60s.
- **Unsafe raw pointer removed** — Replaced with safe `.clone()` for adaptive model borrow.
- **Firmware CI** — Upgraded to IDF v5.4, replaced `xxd` with `od` (#327).
- **Person count double-counting** — Multi-node aggregation changed from `sum` to `max`.
- **Skeleton jitter** — Removed tick-based noise, dampened procedural animation, recalibrated feature scaling for real ESP32 data.

### Changed
- Motion-responsive skeleton: arm swing (0-80px) driven by CSI variance, leg kick (0-50px) by motion_band_power, vertical bob when walking.
- Person count thresholds recalibrated for real ESP32 hardware (1→2 at 0.70, EMA alpha 0.04).
- Vital sign filtering: larger median window (31), faster EMA (0.05), looser HR jump filter (15 BPM).
- Vendored ruvector updated to v2.1.0-40 (316 commits ahead).

### Benchmarks (2-node mesh, COM6 + COM9, 30s)
| Metric | Baseline | v0.5.3 | Improvement |
|--------|----------|--------|-------------|
| Variance noise | 109.4 | 77.6 | **-29%** |
| Feature stability | std=154.1 | std=105.4 | **-32%** |
| Keypoint jitter | std=4.5px | std=1.3px | **-72%** |
| Confidence | 0.643 | 0.686 | **+7%** |
| Presence accuracy | 93.4% | 94.6% | **+1.3pp** |

### Verified
- Real hardware: COM6 (node 1) + COM9 (node 2) on ruv.net WiFi
- All 284 Rust tests pass, 352 signal crate tests pass
- Firmware builds clean at 843 KB
- QEMU CI: 11/11 jobs green

## [v0.5.2-esp32] — 2026-03-28

### Fixed
- RSSI byte offset in frame parser (#332)
- Per-node state pipeline for multi-node sensing (#249)
- Firmware CI upgraded to IDF v5.4 (#327)

## [v0.5.1-esp32] — 2026-03-27

### Fixed
- Watchdog crash on busy LANs (#321)
- No detection from edge vitals (#323)
- `wifi_densepose` Python package import (#314)
- Pre-compiled firmware binaries added to release

## [v0.5.0-esp32] — 2026-03-15

### Added
- **60 GHz mmWave sensor fusion (ADR-063)** — Auto-detects Seeed MR60BHA2 (60 GHz, HR/BR/presence) and HLK-LD2410 (24 GHz, presence/distance) on UART at boot. Probes 115200 then 256000 baud, registers device capabilities, starts background parser.
- **48-byte fused vitals packet** (magic `0xC5110004`) — Kalman-style fusion: mmWave 80% + CSI 20% when both available. Automatic fallback to standard 32-byte CSI-only packet.
- **Server-side fusion bridge** (`scripts/mmwave_fusion_bridge.py`) — Reads two serial ports simultaneously for dual-sensor setups where mmWave runs on a separate ESP32.
- **Multimodal ambient intelligence roadmap (ADR-064)** — 25+ applications from fall detection to sleep monitoring to RF tomography.

### Verified
- Real hardware: ESP32-S3 (COM7) WiFi CSI + ESP32-C6/MR60BHA2 (COM4) 60 GHz mmWave running concurrently. HR=75 bpm, BR=25/min at 52 cm range. All 11 QEMU CI jobs green.

## [v0.4.3-esp32] — 2026-03-15

### Fixed
- **Fall detection false positives (#263)** — Default threshold raised from 2.0 to 15.0 rad/s²; normal walking (2-5 rad/s²) no longer triggers alerts. Added 3-consecutive-frame debounce and 5-second cooldown between alerts. Verified on real ESP32-S3 hardware: 0 false alerts in 60s / 1,300+ live WiFi CSI frames.
- **Kconfig default mismatch** — `CONFIG_EDGE_FALL_THRESH` Kconfig default was still 2000 (=2.0) while `nvs_config.c` fallback was updated to 15.0. Fixed Kconfig to 15000. Caught by real hardware testing — mock data did not reproduce.
- **provision.py NVS generator API change** — `esp_idf_nvs_partition_gen` package changed its `generate()` signature; switched to subprocess-first invocation for cross-version compatibility.
- **QEMU CI pipeline (11 jobs)** — Fixed all failures: fuzz test `esp_timer` stubs, QEMU `libgcrypt` dependency, NVS matrix generator, IDF container `pip` path, flash image padding, validation WARN handling, swarm `ip`/`cargo` missing.

### Added
- **4MB flash support (#265)** — `partitions_4mb.csv` and `sdkconfig.defaults.4mb` for ESP32-S3 boards with 4MB flash (e.g. SuperMini). Dual OTA slots, 1.856 MB each. Thanks to @sebbu for the community workaround that confirmed feasibility.
- **`--strict` flag** for `validate_qemu_output.py` — WARNs now pass by default in CI (no real WiFi in QEMU); use `--strict` to fail on warnings.

## [Unreleased]

### Added
- **QEMU ESP32-S3 testing platform (ADR-061)** — 9-layer firmware testing without hardware
  - Mock CSI generator with 10 physics-based scenarios (empty room, walking, fall, multi-person, etc.)
  - Single-node QEMU runner with 16-check UART validation
  - Multi-node TDM mesh simulation (TAP networking, 2-6 nodes)
  - GDB remote debugging with VS Code integration
  - Code coverage via gcov/lcov + apptrace
  - Fuzz testing (3 libFuzzer targets + ASAN/UBSAN)
  - NVS provisioning matrix (14 configs)
  - Snapshot-based regression testing (sub-second VM restore)
  - Chaos testing with fault injection + health monitoring
- **QEMU Swarm Configurator (ADR-062)** — YAML-driven multi-ESP32 test orchestration
  - 4 topologies: star, mesh, line, ring
  - 3 node roles: sensor, coordinator, gateway
  - 9 swarm-level assertions (boot, crashes, TDM, frame rate, fall detection, etc.)
  - 7 presets: smoke (2n/15s), standard (3n/60s), ci-matrix, large-mesh, line-relay, ring-fault, heterogeneous
  - Health oracle with cross-node validation
- **QEMU installer** (`install-qemu.sh`) — auto-detects OS, installs deps, builds Espressif QEMU fork
- **Unified QEMU CLI** (`qemu-cli.sh`) — single entry point for all 11 QEMU test commands
- CI: `firmware-qemu.yml` workflow with QEMU test matrix, fuzz testing, NVS validation, and swarm test jobs
- User guide: QEMU testing and swarm configurator section with plain-language walkthrough

### Fixed
- Firmware now boots in QEMU: WiFi/UDP/OTA/display guards for mock CSI mode
- 9 bugs in mock_csi.c (LFSR bias, MAC filter init, scenario loop, overflow burst timing)
- 23 bugs from ADR-061 deep review (inject_fault.py writes, CI cache, snapshot log corruption, etc.)
- 16 bugs from ADR-062 deep review (log filename mismatch, SLIRP port collision, heap false positives, etc.)
- All scripts: `--help` flags, prerequisite checks with install hints, standardized exit codes

- **Sensing server UI API completion (ADR-043)** — 14 fully-functional REST endpoints for model management, CSI recording, and training control
  - Model CRUD: `GET /api/v1/models`, `GET /api/v1/models/active`, `POST /api/v1/models/load`, `POST /api/v1/models/unload`, `DELETE /api/v1/models/:id`, `GET /api/v1/models/lora/profiles`, `POST /api/v1/models/lora/activate`
  - CSI recording: `GET /api/v1/recording/list`, `POST /api/v1/recording/start`, `POST /api/v1/recording/stop`, `DELETE /api/v1/recording/:id`
  - Training control: `GET /api/v1/train/status`, `POST /api/v1/train/start`, `POST /api/v1/train/stop`
  - Recording writes CSI frames to `.jsonl` files via tokio background task
  - Model/recording directories scanned at startup, state managed via `Arc<RwLock<AppStateInner>>`
- **ADR-044: Provisioning tool enhancements** — 5-phase plan for complete NVS coverage (7 missing keys), JSON config files, mesh presets, read-back/verify, and auto-detect
- **25 real mobile tests** replacing `it.todo()` placeholders — 205 assertions covering components, services, stores, hooks, screens, and utils
- **Project MERIDIAN (ADR-027)** — Cross-environment domain generalization for WiFi pose estimation (1,858 lines, 72 tests)
  - `HardwareNormalizer` — Catmull-Rom cubic interpolation resamples any hardware CSI to canonical 56 subcarriers; z-score + phase sanitization
  - `DomainFactorizer` + `GradientReversalLayer` — adversarial disentanglement of pose-relevant vs environment-specific features
  - `GeometryEncoder` + `FilmLayer` — Fourier positional encoding + DeepSets + FiLM for zero-shot deployment given AP positions
  - `VirtualDomainAugmentor` — synthetic environment diversity (room scale, wall material, scatterers, noise) for 4x training augmentation
  - `RapidAdaptation` — 10-second unsupervised calibration via contrastive test-time training + LoRA adapters
  - `CrossDomainEvaluator` — 6-metric evaluation protocol (MPJPE in-domain/cross-domain/few-shot/cross-hardware, domain gap ratio, adaptation speedup)
- ADR-027: Cross-Environment Domain Generalization — 10 SOTA citations (PerceptAlign, X-Fi ICLR 2025, AM-FM, DGSense, CVPR 2024)
- **Cross-platform RSSI adapters** — macOS CoreWLAN (`MacosCoreWlanScanner`) and Linux `iw` (`LinuxIwScanner`) Rust adapters with `#[cfg(target_os)]` gating
- macOS CoreWLAN Python sensing adapter with Swift helper (`mac_wifi.swift`)
- macOS synthetic BSSID generation (FNV-1a hash) for Sonoma 14.4+ BSSID redaction
- Linux `iw dev <iface> scan` parser with freq-to-channel conversion and `scan dump` (no-root) mode
- ADR-025: macOS CoreWLAN WiFi Sensing (ORCA)

### Fixed
- **sendto ENOMEM crash (Issue #127)** — CSI callbacks in promiscuous mode exhaust lwIP pbuf pool causing guru meditation crash. Fixed with 50 Hz rate limiter in `csi_collector.c` and 100 ms ENOMEM backoff in `stream_sender.c`. Hardware-verified on ESP32-S3 (200+ callbacks, zero crashes)
- **Provisioning script missing TDM/edge flags (Issue #130)** — Added `--tdm-slot`, `--tdm-total`, `--edge-tier`, `--pres-thresh`, `--fall-thresh`, `--vital-win`, `--vital-int`, `--subk-count` to `provision.py`
- **WebSocket "RECONNECTING" on Dashboard/Live Demo** — `sensingService.start()` now called on app init in `app.js` so WebSocket connects immediately instead of waiting for Sensing tab visit
- **Mobile WebSocket port** — `ws.service.ts` `buildWsUrl()` uses same-origin port instead of hardcoded port 3001
- **Mobile Jest config** — `testPathIgnorePatterns` no longer silently ignores the entire test directory
- Removed synthetic byte counters from Python `MacosWifiCollector` — now reports `tx_bytes=0, rx_bytes=0` instead of fake incrementing values

---

## [3.0.0] - 2026-03-01

Major release: AETHER contrastive embedding model, Docker Hub images, and comprehensive UI overhaul.

### Added — AETHER Contrastive Embedding Model (ADR-024)
- **Project AETHER** — self-supervised contrastive learning for WiFi CSI fingerprinting, similarity search, and anomaly detection (`9bbe956`)
- `embedding.rs` module: `ProjectionHead`, `InfoNceLoss`, `CsiAugmenter`, `FingerprintIndex`, `PoseEncoder`, `EmbeddingExtractor` (909 lines, zero external ML dependencies)
- SimCLR-style pretraining with 5 physically-motivated augmentations (temporal jitter, subcarrier masking, Gaussian noise, phase rotation, amplitude scaling)
- CLI flags: `--pretrain`, `--pretrain-epochs`, `--embed`, `--build-index <type>`
- Four HNSW-compatible fingerprint index types: `env_fingerprint`, `activity_pattern`, `temporal_baseline`, `person_track`
- Cross-modal `PoseEncoder` for WiFi-to-camera embedding alignment
- VICReg regularization for embedding collapse prevention
- 53K total parameters (55 KB at INT8) — fits on ESP32

### Added — Docker & Deployment
- Published Docker Hub images: `ruvnet/wifi-densepose:latest` (132 MB Rust) and `ruvnet/wifi-densepose:python` (569 MB) (`add9f19`)
- Multi-stage Dockerfile for Rust sensing server with RuVector crates
- `docker-compose.yml` orchestrating both Rust and Python services
- RVF model export via `--export-rvf` and load via `--load-rvf` CLI flags

### Added — Documentation
- 33 use cases across 4 vertical tiers: Everyday, Specialized, Robotics & Industrial, Extreme (`0afd9c5`)
- "Why WiFi Wins" comparison table (WiFi vs camera vs LIDAR vs wearable vs PIR)
- Mermaid architecture diagrams: end-to-end pipeline, signal processing detail, deployment topology (`50f0fc9`)
- Models & Training section with RuVector crate links (GitHub + crates.io), SONA component table (`965a1cc`)
- RVF container section with deployment targets table (ESP32 0.7 MB to server 50+ MB)
- Collapsible README sections for improved navigation (`478d964`, `99ec980`, `0ebd6be`)
- Installation and Quick Start moved above Table of Contents (`50acbf7`)
- CSI hardware requirement notice (`528b394`)

### Fixed
- **UI auto-detects server port from page origin** — no more hardcoded `localhost:8080`; works on any port (Docker :3000, native :8080, custom) (`3b72f35`, closes #55)
- **Docker port mismatch** — server now binds 3000/3001 inside container as documented (`44b9c30`)
- Added `/ws/sensing` WebSocket route to the HTTP server so UI only needs one port
- Fixed README API endpoint references: `/api/v1/health` → `/health`, `/api/v1/sensing` → `/api/v1/sensing/latest`
- Multi-person tracking limit corrected: configurable default 10, no hard software cap (`e2ce250`)

---

## [2.0.0] - 2026-02-28

Major release: complete Rust sensing server, full DensePose training pipeline, RuVector v2.0.4 integration, ESP32-S3 firmware, and 6 security hardening patches.

### Added — Rust Sensing Server
- **Full DensePose-compatible REST API** served by Axum (`d956c30`)
  - `GET /health` — server health
  - `GET /api/v1/sensing/latest` — live CSI sensing data
  - `GET /api/v1/vital-signs` — breathing rate (6-30 BPM) and heartbeat (40-120 BPM)
  - `GET /api/v1/pose/current` — 17 COCO keypoints derived from WiFi signal field
  - `GET /api/v1/info` — server build and feature info
  - `GET /api/v1/model/info` — RVF model container metadata
  - `ws://host/ws/sensing` — real-time WebSocket stream
- Three data sources: `--source esp32` (UDP CSI), `--source windows` (netsh RSSI), `--source simulated` (deterministic reference)
- Auto-detection: server probes ESP32 UDP and Windows WiFi, falls back to simulated
- Three.js visualization UI with 3D body skeleton, signal heatmap, phase plot, Doppler bars, vital signs panel
- Static UI serving via `--ui-path` flag
- Throughput: 9,520–11,665 frames/sec (release build)

### Added — ADR-021: Vital Sign Detection
- `VitalSignDetector` with breathing (6-30 BPM) and heartbeat (40-120 BPM) extraction from CSI fluctuations (`1192de9`)
- FFT-based spectral analysis with configurable band-pass filters
- Confidence scoring based on spectral peak prominence
- REST endpoint `/api/v1/vital-signs` with real-time JSON output

### Added — ADR-023: DensePose Training Pipeline (Phases 1-8)
- `wifi-densepose-train` crate with complete 8-phase pipeline (`fc409df`, `ec98e40`, `fce1271`)
  - Phase 1: `DataPipeline` with MM-Fi and Wi-Pose dataset loaders
  - Phase 2: `CsiToPoseTransformer` — 4-head cross-attention + 2-layer GCN on COCO skeleton
  - Phase 3: 6-term composite loss (MSE, bone length, symmetry, joint angle, temporal, confidence)
  - Phase 4: `DynamicPersonMatcher` via ruvector-mincut (O(n^1.5 log n) Hungarian assignment)
  - Phase 5: `SonaAdapter` — MicroLoRA rank-4 with EWC++ memory preservation
  - Phase 6: `SparseInference` — progressive 3-layer model loading (A: essential, B: refinement, C: full)
  - Phase 7: `RvfContainer` — single-file model packaging with segment-based binary format
  - Phase 8: End-to-end training with cosine-annealing LR, early stopping, checkpoint saving
- CLI: `--train`, `--dataset`, `--epochs`, `--save-rvf`, `--load-rvf`, `--export-rvf`
- Benchmark: ~11,665 fps inference, 229 tests passing

### Added — ADR-016: RuVector Training Integration (all 5 crates)
- `ruvector-mincut` → `DynamicPersonMatcher` in `metrics.rs` + subcarrier selection (`81ad09d`, `a7dd31c`)
- `ruvector-attn-mincut` → antenna attention in `model.rs` + noise-gated spectrogram
- `ruvector-temporal-tensor` → `CompressedCsiBuffer` in `dataset.rs` + compressed breathing/heartbeat
- `ruvector-solver` → sparse subcarrier interpolation (114→56) + Fresnel triangulation
- `ruvector-attention` → spatial attention in `model.rs` + attention-weighted BVP
- Vendored all 11 RuVector crates under `vendor/ruvector/` (`d803bfe`)

### Added — ADR-017: RuVector Signal & MAT Integration (7 integration points)
- `gate_spectrogram()` — attention-gated noise suppression (`18170d7`)
- `attention_weighted_bvp()` — sensitivity-weighted velocity profiles
- `mincut_subcarrier_partition()` — dynamic sensitive/insensitive subcarrier split
- `solve_fresnel_geometry()` — TX-body-RX distance estimation
- `CompressedBreathingBuffer` + `CompressedHeartbeatSpectrogram`
- `BreathingDetector` + `HeartbeatDetector` (MAT crate, real FFT + micro-Doppler)
- Feature-gated behind `cfg(feature = "ruvector")` (`ab2453e`)

### Added — ADR-018: ESP32-S3 Firmware & Live CSI Pipeline
- ESP32-S3 firmware with FreeRTOS CSI extraction (`92a5182`)
- ADR-018 binary frame format: `[0xAD, 0x18, len_hi, len_lo, payload]`
- Rust `Esp32Aggregator` receiving UDP frames on port 5005
- `bridge.rs` converting I/Q pairs to amplitude/phase vectors
- NVS provisioning for WiFi credentials
- Pre-built binary quick start documentation (`696a726`)

### Added — ADR-014: SOTA Signal Processing
- 6 algorithms, 83 tests (`fcb93cc`)
  - Hampel filter (median + MAD, resistant to 50% contamination)
  - Conjugate multiplication (reference-antenna ratio, cancels common-mode noise)
  - Phase sanitization (unwrap + linear detrend, removes CFO/SFO)
  - Fresnel zone geometry (TX-body-RX distance from first-principles physics)
  - Body Velocity Profile (micro-Doppler extraction, 5.7x speedup)
  - Attention-gated spectrogram (learned noise suppression)

### Added — ADR-015: Public Dataset Training Strategy
- MM-Fi and Wi-Pose dataset specifications with download links (`4babb32`, `5dc2f66`)
- Verified dataset dimensions, sampling rates, and annotation formats
- Cross-dataset evaluation protocol

### Added — WiFi-Mat Disaster Detection Module
- Multi-AP triangulation for through-wall survivor detection (`a17b630`, `6b20ff0`)
- Triage classification (breathing, heartbeat, motion)
- Domain events: `survivor_detected`, `survivor_updated`, `alert_created`
- WebSocket broadcast at `/ws/mat/stream`

### Added — Infrastructure
- Guided 7-step interactive installer with 8 hardware profiles (`8583f3e`)
- Comprehensive build guide for Linux, macOS, Windows, Docker, ESP32 (`45f8a0d`)
- 12 Architecture Decision Records (ADR-001 through ADR-012) (`337dd96`)

### Added — UI & Visualization
- Sensing-only UI mode with Gaussian splat visualization (`b7e0f07`)
- Three.js 3D body model (17 joints, 16 limbs) with signal-viz components
- Tabs: Dashboard, Hardware, Live Demo, Sensing, Architecture, Performance, Applications
- WebSocket client with automatic reconnection and exponential backoff

### Added — Rust Signal Processing Crate
- Complete Rust port of WiFi-DensePose with modular workspace (`6ed69a3`)
  - `wifi-densepose-signal` — CSI processing, phase sanitization, feature extraction
  - `wifi-densepose-core` — shared types and configuration
  - `wifi-densepose-nn` — neural network inference (DensePose head, RCNN)
  - `wifi-densepose-hardware` — ESP32 aggregator, hardware interfaces
  - `wifi-densepose-config` — configuration management
- Comprehensive benchmarks and validation tests (`3ccb301`)

### Added — Python Sensing Pipeline
- `WindowsWifiCollector` — RSSI collection via `netsh wlan show networks`
- `RssiFeatureExtractor` — variance, spectral bands (motion 0.5-4 Hz, breathing 0.1-0.5 Hz), change points
- `PresenceClassifier` — rule-based 3-state classification (ABSENT / PRESENT_STILL / ACTIVE)
- Cross-receiver agreement scoring for multi-AP confidence boosting
- WebSocket sensing server (`ws_server.py`) broadcasting JSON at 2 Hz
- Deterministic CSI proof bundles for reproducible verification (`archive/v1/data/proof/`)
- Commodity sensing unit tests (`b391638`)

### Changed
- Rust hardware adapters now return explicit errors instead of silent empty data (`6e0e539`)

### Fixed
- Review fixes for end-to-end training pipeline (`45f0304`)
- Dockerfile paths updated from `src/` to `archive/v1/src/` (`7872987`)
- IoT profile installer instructions updated for aggregator CLI (`f460097`)
- `process.env` reference removed from browser ES module (`e320bc9`)

### Performance
- 5.7x Doppler extraction speedup via optimized FFT windowing (`32c75c8`)
- Single 2.1 MB static binary, zero Python dependencies for Rust server

### Security
- Fix SQL injection in status command and migrations (`f9d125d`)
- Fix XSS vulnerabilities in UI components (`5db55fd`)
- Fix command injection in statusline.cjs (`4cb01fd`)
- Fix path traversal vulnerabilities (`896c4fc`)
- Fix insecure WebSocket connections — enforce wss:// on non-localhost (`ac094d4`)
- Fix GitHub Actions shell injection (`ab2e7b4`)
- Fix 10 additional vulnerabilities, remove 12 dead code instances (`7afdad0`)

---

## [1.1.0] - 2025-06-07

### Added
- Complete Python WiFi-DensePose system with CSI data extraction and router interface
- CSI processing and phase sanitization modules
- Batch processing for CSI data in `CSIProcessor` and `PhaseSanitizer`
- Hardware, pose, and stream services for WiFi-DensePose API
- Comprehensive CSS styles for UI components and dark mode support
- API and Deployment documentation

### Fixed
- Badge links for PyPI and Docker in README
- Async engine creation poolclass specification

---

## [1.0.0] - 2024-12-01

### Added
- Initial release of WiFi-DensePose
- Real-time WiFi-based human pose estimation using Channel State Information (CSI)
- DensePose neural network integration for body surface mapping
- RESTful API with comprehensive endpoint coverage
- WebSocket streaming for real-time pose data
- Multi-person tracking with configurable capacity (default 10, up to 50+)
- Fall detection and activity recognition
- Domain configurations: healthcare, fitness, smart home, security
- CLI interface for server management and configuration
- Hardware abstraction layer for multiple WiFi chipsets
- Phase sanitization and signal processing pipeline
- Authentication and rate limiting
- Background task management
- Cross-platform support (Linux, macOS, Windows)

### Documentation
- User guide and API reference
- Deployment and troubleshooting guides
- Hardware setup and calibration instructions
- Performance benchmarks
- Contributing guidelines

[Unreleased]: https://github.com/ruvnet/wifi-densepose/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/ruvnet/wifi-densepose/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/ruvnet/wifi-densepose/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/ruvnet/wifi-densepose/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ruvnet/wifi-densepose/releases/tag/v1.0.0
