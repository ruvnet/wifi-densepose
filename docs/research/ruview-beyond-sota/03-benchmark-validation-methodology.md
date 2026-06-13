# Beyond-SOTA Validation, Test & Benchmark Methodology

**Series:** `docs/research/ruview-beyond-sota/` · Document 03
**Date:** 2026-06-09
**Scope:** How RuView proves (and gates) beyond-SOTA claims using the verification
infrastructure that already exists in this repository. Every number below is sourced
from a cited file in this repo; nothing is invented.

---

## 1. The Layered Validation Pyramid

Six layers, cheapest/most-deterministic at the bottom, most expensive/most-credible at
the top. A beyond-SOTA claim must survive **every layer below it** before it may be
published from the layer it lives at.

| Layer | What it proves | Tooling | Frequency | Determinism |
|-------|----------------|---------|-----------|-------------|
| **L0** Unit/integration tests | Code correctness | `cargo test --workspace --no-default-features` + pytest | per commit | exact |
| **L1** Deterministic proof + witness bundle | Pipeline is real, unchanged, reproducible | `archive/v1/data/proof/verify.py`, `scripts/generate-witness-bundle.sh` | per merge / release | exact (SHA-256) |
| **L2** Criterion micro-benchmarks | Compute latency only — never quality (ADR-171 §2) | 15 bench targets across `v2/crates/*/benches/` | nightly / pre-release | statistical |
| **L3** Dataset-level accuracy eval | Pose/presence/vitals quality vs published SOTA | MM-Fi / Wi-Pose (ADR-015), `ruview_metrics.rs` tiers, ADR-145 ablation harness | per model release | seeded |
| **L4** Hardware-in-loop | Real CSI on real ESP32, no mocks | COM9 (S3) / COM12 (C6) protocol, witness firmware hashes | per firmware release | A/B controlled |
| **L5** Field trials / live capture | End-to-end behavior in a real room | live-session captures (e.g. `benchmark_baseline.json`) | campaign | statistical |

### 1.1 L0 — Workspace tests (current counts)

- ADR-028 audit (2026-03-01): **1,031 passed, 0 failed, 8 ignored** for
  `cargo test --workspace --no-default-features`
  (`docs/adr/ADR-028-esp32-capability-audit.md` §2).
- Current `CHANGELOG.md` (Unreleased, cross-platform fix entry): **2,682 workspace
  tests pass / 0 fail on Windows** — the suite has more than doubled since the audit.
- `CLAUDE.md` pre-merge gate still cites "1,031+ passed, 0 failed" as the floor.

**Rule:** the post-change test count may never be lower than the pre-change count, and
failures must be 0. The witness bundle records the full log
(`test-results/rust-workspace-tests.log`) and an aggregated `summary.txt`
(`scripts/generate-witness-bundle.sh` step 3).

### 1.2 L1 — Deterministic proof ("Trust Kill Switch") + witness bundle

`archive/v1/data/proof/verify.py` (header comment): feeds 1,000 synthetic CSI frames
(seed=42, `sample_csi_data.json`) through the **production** `CSIProcessor`
(`src/core/csi_processor.py`), hashes the first 100 frames' feature output
(`VERIFICATION_FRAME_COUNT = 100`), and compares against
`archive/v1/data/proof/expected_features.sha256`.

- **Current published hash (file contents, verified during this investigation):**
  `f8e76f21a0f9852b70b6d9dd5318239f6b20cbcb4cdd995863263cecdc446f7a`
- The hash is **environment-coupled** and has been legitimately regenerated before:
  ADR-028 §5.3 recorded `8c0680d7…` under numpy 2.4.2/scipy 1.17.1; `CHANGELOG.md`
  (#560 fix) recorded `667eb054…` after 6-decimal quantization + single-thread BLAS
  pinning (`OMP_NUM_THREADS=1` etc.). Each regeneration must follow the documented
  procedure: `python verify.py --generate-hash` then `python verify.py` → `VERDICT: PASS`.

`scripts/generate-witness-bundle.sh` packages: witness log + ADR-028, the Python proof
(verify.py + expected hash + reference-signal metadata), full Rust test log + summary,
the ADR-134 CIR proof, firmware source/binary SHA-256s, crate version manifest, npm
tarball SHA-256, and a recipient-side `VERIFY.sh`.

**Accuracy note on check counts:** `CLAUDE.md` describes the recipient verification as
"7/7 PASS"; the current `VERIFY.sh` embedded in the script performs **10** `check()`
assertions (witness log, ADR, proof-hash file, tests, firmware hashes, crate manifest,
npm manifest, Python proof, CIR proof, CIR hash file) but prints a hardcoded
`"ALL CHECKS PASSED (8/8)"` string (`generate-witness-bundle.sh` line 293). The
hardcoded count is stale relative to the actual check list — fix it to print
`${PASS_COUNT}/${PASS_COUNT+FAIL_COUNT}` so the verdict can never silently desynchronize
from the check inventory.

### 1.3 L2 — Criterion micro-benchmark inventory (all 15 targets)

All bench sources read directly. Per ADR-171 §2 these are **latency regression gates
only, never quality evidence**.

| Bench target | Crate | Benchmark functions / groups | What it measures | Recorded value or in-source target (citation) |
|---|---|---|---|---|
| `engine_cycle.rs` | wifi-densepose-engine | `process_cycle_4nodes_56sc` | One full `StreamingEngine::process_cycle` (fuse + quality + calibration provenance + privacy gate + WorldGraph node), 4-node/56-subcarrier ESP32-S3 HT20 mesh | Budget: **50 ms** (20 Hz) — bench header |
| `signal_bench.rs` | wifi-densepose-signal | `CSI Preprocessing`, `Phase Sanitization`, `Feature Extraction`, `Motion Detection`, `Full Pipeline` | SOTA signal stages (ADR-014) at varying frame sizes | no recorded baseline |
| `cir_bench.rs` | wifi-densepose-signal | `cir_estimate` (HT20/HT40/HE20/HE40), `cir_estimate_12link`, `cir_estimator_new` | ADR-134 `CirEstimator::estimate()` per tier; 12-link multistatic amortization; cold-start | no recorded baseline |
| `calibration_bench.rs` | wifi-densepose-signal | `bench_recorder_record`, `bench_recorder_finalize`, `bench_deviation`, `bench_record_600`, `bench_to_bytes` (K=52/114/242/484) | ADR-135 empty-room baseline recorder + deviation scoring | no recorded baseline |
| `aether_prefilter_bench.rs` | wifi-densepose-signal | `aether_search_d…_n…_k…` (search vs prefilter) | ADR-084 Pass-2: `EmbeddingHistory::search_prefilter` vs brute force, prefilter_factor=8 | Pass: **≥4× at n=1024** — bench header |
| `sketch_bench.rs` | wifi-densepose-ruvector | `compare_d128/256/512` × `float_l2`/`float_cosine`/`sketch_hamming` | ADR-084 sketch-vs-float per-pair compare cost (AETHER 128-d, spectrogram 256-d) | Pass: **sketch ≥8× faster** at every dim (ADR-084 threshold 8×–30×) — bench header |
| `crv_bench.rs` | wifi-densepose-ruvector | `gestalt_classify_single/batch_100`, `sensory_encode_single`, `pipeline_full_session`, `convergence_two_sessions`, `crv_session_create`, `crv_embedding_dimension_scaling` (32/128/384), `crv_stage_vi_partition` | CRV integration throughput | no recorded baseline |
| `inference_bench.rs` | wifi-densepose-nn | `tensor_ops` (relu/sigmoid/tanh), `densepose_inference`, `translator_inference`, `mock_inference`, `batch_inference` | NN forward-pass cost by input/batch size | no recorded baseline; **`mock_inference` group must never be quoted as a pipeline number** (§6) |
| `training_bench.rs` | wifi-densepose-train | `interp_114_to_56_batch32`, `interp_scaling`, `compute_interp_weights_114_56`, `synthetic_dataset_get`, `synthetic_epoch`, `config_validate`, PCK over 100 samples | Training preprocessing + metrics hot paths; fixtures fully deterministic (no `rand`) — header | no recorded baseline |
| `detection_bench.rs` | wifi-densepose-mat | `breathing_detection`, `heartbeat_detection`, `movement_classification`, `detection_pipeline`, localization (triangulation/depth), alert generation | MAT survivor-detection algorithms at varying signal lengths / noise | no recorded baseline |
| `transport_bench.rs` | wifi-densepose-hardware | `beacon_serialize_16byte/28byte_auth/quic_framed`, `auth_beacon_verify`, `replay_window`, `framed_message` encode/decode, `secure_tdm_cycle` (manual vs QUIC) | TDM beacon crypto + transport | no recorded baseline |
| `mqtt_throughput.rs` | wifi-densepose-sensing-server | `discovery::build_*`, `state::*`, `rate_limiter::allow_*`, `privacy::decide_*`, `semantic::bus_tick_all_10_primitives` | ADR-115 MQTT hot path | Targets (header): discovery **<5 µs**, state encode **<2 µs**, rate limit **<100 ns**, privacy **<50 ns**, bus tick **<10 µs** |
| `swarm_bench.rs` | ruview-swarm | `marl_actor_inference`, `rrt_apf_100iter`, `multiview_fusion_3drones`, `demo_coverage_estimate`, `ppo_update_64transitions` | ADR-148 swarm control-loop compute | Measured: **3.3 µs / 43 µs / 54–58.5 ns / 100 ps / 248 µs** (ADR-171 §4.3; `CHANGELOG.md` Performance section) |
| `pipeline_throughput.rs` | nvsim | `pipeline_run` (sample-count sweep), `witness::run` vs `run_with_witness` | NV-diamond sim throughput + witness overhead | Acceptance: **≥1 kHz** simulated samples/s on Cortex-A53-class CPU — bench header |
| `state_machine.rs` | homecore | `set` first/warm/no-op, `get` hit/miss, `all_snapshot`, `all_by_domain_light_20_of_100`, `broadcast_fan_out` | HOMECORE state-machine hot paths | no recorded baseline |

**Honest gap — `benchmark_baseline.json` is not a criterion baseline.** The repo-root
`benchmark_baseline.json` (369.9 KB) contains **1,566 live-capture samples** from a
2-node session (fields: `tick`, `n_nodes`, `variance`, `motion`, `presence`,
`confidence`, `est_persons`, `n_persons_rendered`, `kp_spread`, `rssi`) plus a summary
block — it records **field-trial telemetry (L5)**, not micro-benchmark latencies.
No file in the repo references it (`grep -rn benchmark_baseline` → 0 hits outside the
file itself); its producer must be identified and committed (§5.3). Summary values
(all from the file's `summary` object):

| Metric | Baseline value |
|---|---:|
| `total_frames` | 1,566 |
| `presence_ratio` | 0.9336 (1,462/1,566 frames presence-true) |
| `confidence_mean` | 0.6433 |
| `variance_mean` / `variance_std` | 109.36 / 154.13 |
| `kp_spread_mean` / `kp_spread_std` | 86.73 / 4.52 |
| `person_count_changes` | 10 |

Criterion latencies that *have* been recorded live in ADR documents instead
(ADR-168-benchmark-proof.md, ADR-171 §4.3, CHANGELOG Performance) — §5 below defines
how to consolidate them into a real machine-readable criterion baseline.

### 1.4 L3 — Dataset-level accuracy evaluation

- **Datasets (ADR-015):** primary **MM-Fi** (40 subjects × 27 actions × ~320K frames,
  1TX×3RX, 114 subcarriers @100 Hz, 17-keypoint COCO + DensePose UV, CC BY-NC 4.0);
  secondary **Wi-Pose** (12 volunteers × 12 actions × 166,600 packets, 3×3, 30
  subcarriers). 114→56 subcarrier interpolation via `subcarrier.rs`; validation split =
  subjects 33–40 held out (ADR-015 Phase 1).
- **Acceptance tiers:** `wifi-densepose-train/src/ruview_metrics.rs` —
  PCK@0.2 / OKS / MOTA / vitals rolled into `RuViewTier`
  (Fail/Bronze/Silver/Gold) (ADR-145 §1.1).
- **Ablation harness (ADR-145):** 6-variant matrix (`csi_only`, `cir_only`,
  `csi_plus_cir`, `plus_doppler`, `plus_bfld`, `plus_uwb`-skipped), each variant
  producing acceptance tier + `SpecMetrics` (presence ≥0.90, localization ≤0.50 m,
  activity ≥0.70, FP ≤0.05, FN ≤0.10), `LatencyProfile` (p95 ≤100 ms), and
  `PrivacyLeakage` (MIA `leakage_score` ≤0.05), SHA-256-pinned per variant under
  `PROOF_SEED=42` (ADR-145 §2.2–2.6). Built at commit `0f336b7d3` (ADR-145
  implementation status); CLI auto-mode wiring is pending.
- **Cross-environment:** ADR-027 MERIDIAN `CrossDomainEvaluator`
  (`wifi-densepose-train/src/eval.rs`) — `domain_gap_ratio`, extended by ADR-145
  `cross_room_degradation()` with a 17-joint PCK-delta heatmap.

### 1.5 L4 — Hardware-in-loop

- Real CSI nodes: ESP32-S3 on **COM9**, ESP32-C6 + MR60BHA2 on **COM12** (`CLAUDE.md`
  hardware table). ADR-018 binary frame protocol over UDP:5005 (ADR-028 §3.2/§3.4).
- ADR-145 Tier-4 test (gated, `#[cfg(feature = "hardware-test")]`): replay a live 30 s
  COM9 capture through `csi_only` and `csi_plus_cir`; assert no presence regression and
  p95 < 100 ms.
- A/B board protocol precedent (`CHANGELOG.md` #987): fixed vs unmodified control board
  against Apple-Watch ground truth (control pegged 40–49 BPM; fixed 88–91 vs 87 GT) —
  this fixed-board/control-board + external ground-truth pattern is the required design
  for all hardware vital-sign claims.
- Witness bundle pins firmware: per-file SHA-256 of all sources + release binaries
  (`generate-witness-bundle.sh` step 5).

### 1.6 L5 — Field trials

Live multi-node sessions captured as JSONL/JSON with summary statistics —
`benchmark_baseline.json` (§1.3) is the existing exemplar. ADR-171 §6 adds the seeded
`evals/` episode harness (Stage 1 kinematic full-matrix, Stage 2 Gazebo/PX4 SITL on the
3 median seeds) for the swarm domain.

---

## 2. Beyond-SOTA Acceptance Criteria per Capability Axis

A claim is "beyond SOTA" only with: a named external baseline, an exact metric and
protocol match, the dataset/split named, the threshold pre-registered, and the
statistical procedure of §3 followed. Current axes with measured status:

| Axis | Metric (exact) | Dataset / protocol | SOTA baseline | Beyond-SOTA threshold | Measured status (cited) |
|---|---|---|---|---|---|
| In-domain pose accuracy | torso-PCK@20: `‖pred−gt‖ ≤ 0.2·‖R-shoulder−L-hip‖` | MM-Fi `random_split` (ratio 0.8, seed 0) | MultiFormer **72.25%** (Table VII); CSI2Pose 68.41% | > 72.25% with 95% CI lower bound above it | Flagship **83.59%**; micro (75,237 params) **74.30%** (`docs/benchmarks/wifi-pose-efficiency-frontier.md`) |
| Edge efficiency frontier | torso-PCK@20 at deployed precision + params + batch-1 latency | same | MultiFormer 72.25% at full size | Pareto-dominance: smaller **and** above 72.25% at the deployed precision | int8 73.5 KB **74.70%**; int4-QAT 36.7 KB **74.46%**; shipped int4 verified **74.08%**, 0.135 ms 1-thread x86 (same file) |
| Cross-subject generalization | torso-PCK@20, official MM-Fi cross-subject split (256,608 train / 64,152 test) | leakage-free split | own zero-shot baseline 63.99% | ADR-150 §4 gate: **+≥6 pts cross-subject without losing >2 pts random-split** | Best zero-shot **64.92%** (mixup+TTA+3-seed); gate judged unreachable without new capture (ADR-150 §3.2) |
| Few-shot calibration (deployment) | PCK@20 after K labeled in-room samples; adapter size | MM-Fi cross-subject & cross-environment splits | zero-shot (64% / 10.6%) | SOTA-level (≳72%) from ≤200 samples with ≤~11 KB per-room adapter | cross-subject ~**72%** @100–200 samples (3 seeds); cross-env **10.6→73.1%** @200, 60.1% @5 (ADR-150 §3.5–3.6) |
| Swarm SAR localization | CEP50/CEP95 (m), GDOP-stratified | seeded episode distribution (ADR-171 §6), not single geometry | Wi2SAR **5 m** (arxiv 2604.09115, paper-to-paper) | CEP50 < 5 m, IQM over ≥10 seeds, 95% CI excluding 5 m | 1.732 m single synthetic geometry — graded **Low–Medium**, not yet claimable (ADR-171 §7) |
| Swarm coverage | coverage-rate@240 s; time-to-95% | episode rollouts | Wi2SAR 160k m²/13.5 min | rollout (not analytic) mean+CI beating baseline | 223 s is an analytic estimate — graded **Low** (ADR-171 §7) |
| Control-loop latency | criterion wall-clock | local hardware, named | 10 ms / 100 Hz budget | all stages ≪ budget | 3.3 µs MARL / 43 µs RRT-APF / 54 ns fusion / 248 µs PPO (ADR-171 §4.3) |
| World-model trajectory | MDE (m) at 5-frame horizon | RuView CSI-derived occupancy | pre-fine-tune random-weight baseline 9.49 m MDE | **≤1.0 m (2.0 vox)** at 5-frame horizon (ADR-147 §5 target, cited in benchmark-proof §4) | 9.49 m / FDE 16.23 m random weights; 208.45 ms median latency on real CSI (ADR-168-benchmark-proof §4, §7) |
| Privacy leakage | MIA `leakage_score = 2·(AUC−0.5)` | fixed replay, fixed-seed shadow classifier | chance (0) | ≤ **0.05** (attacker AUC ≤ 0.525) | gate defined, harness built (ADR-145 §2.3) |
| Vitals (hardware) | BPM error vs wearable ground truth | live A/B board protocol | control board behavior | within physiological agreement of ground truth, stable spread | 88–91 BPM vs 87 GT, spread 59→0 (CHANGELOG #987) |

### Claim-language discipline (from ADR-171 §7 grading)

| Evidence | Permitted language |
|---|---|
| Single run / single geometry / analytic estimate | "directional", never "beats SOTA" |
| Seeded multi-run with CIs vs paper baseline | "exceeds the published X result paper-to-paper" |
| Same metric, same split, same protocol, CI excludes baseline | "beyond SOTA on <dataset>/<split>" |
| No public leaderboard exists (swarm CSI-SAR) | never claim "leaderboard standing" (ADR-171 §3) |

---

## 3. Statistical Procedure for Honest Claims

Adopted from ADR-171 §5 (Agarwal 2021 / Gorsane 2022 standard) and the practices
already used in ADR-150/efficiency-frontier measurements:

1. **Seeds.** ≥10 independent seeds for RL/episodic claims (ADR-171 §5); ≥3 seeds
   minimum for supervised dataset evals (ADR-150 §3.5 used 3 seeds; report all).
   Training seeds, eval seeds, and split files are versioned and committed.
2. **Aggregate.** IQM (not mean/median) for episodic metrics + performance profiles;
   for dataset accuracy report mean across seeds with each seed's value listed.
3. **Confidence intervals.** 95% stratified bootstrap, 1,000 resamples (ADR-171 §5;
   reference impl: `rliable`).
4. **Paired comparisons.** When comparing model A vs B (e.g. `csi_plus_cir` vs
   `csi_only`, or ours vs a reproduced baseline), evaluate both on the **identical
   frozen test frames** and use a paired bootstrap over per-sample correctness
   (PCK hit/miss is per-joint binary — pair at the joint-sample level). For
   paper-to-paper comparisons where the baseline cannot be re-run, state so
   explicitly ("paper-to-paper", ADR-171 §2) and require the CI lower bound to clear
   the published point value.
5. **Pre-registration.** The threshold lives in an ADR **before** the run
   (precedent: ADR-150 §4 gate written before §3.2 measurements; the measurements
   honestly reported the gate as not met).
6. **Negative results are recorded.** ADR-150 §1/§3.2 keeps DANN-failed,
   capacity-hurts, and KD-didn't-help results in the record — required practice.
7. **Eval episodes (swarm):** 50 fixed, versioned episodes per policy
   (10 victim layouts × 5 CSI-noise levels), ≥3 baselines (random walk,
   boustrophedon+triangulation, IPPO) (ADR-171 §5).
8. **GDOP stratification** for any localization claim, so geometry artifacts cannot
   produce the headline (ADR-171 §6.3).

---

## 4. Regression-Gate Design (CI Enforcement)

### 4.1 Three gate classes, three tolerances

| Gate class | Source of truth | Tolerance | On breach |
|---|---|---|---|
| Determinism hashes | `expected_features.sha256`, `expected_cir_features.sha256`, `expected_calibration_features.sha256`, future `expected_ablation_<slug>.sha256` | **exact (0%)** | exit 1 = FAIL; exit 2 = SKIP only for placeholder hashes (proof.rs `0/1/2` convention, ADR-145 §2.4) |
| Accuracy / quality metrics | per-variant canonical bytes, quantized 1e-3 (ADR-145 §2.6) | exact after quantization | FAIL CI; tier change requires ADR amendment |
| Latency / throughput | criterion estimates JSON | **% tolerance per scale** (below) | FAIL on regression beyond tolerance; trend everything |

### 4.2 Criterion baseline file (replaces the current gap)

Today criterion numbers live in prose (ADR-168-benchmark-proof, ADR-171 §4.3,
CHANGELOG). Formalize:

1. `cargo bench --workspace -- --save-baseline main` on a **named, fixed runner**
   (ADR-147 used RTX 5080 / specific host; record host + toolchain in the file).
2. Export `target/criterion/*/estimates.json` point estimates into a committed
   `v2/benchmarks/criterion-baseline.json`: `{bench_id, crate, p50_ns, host, commit}`.
3. CI compares new runs against it with scale-aware tolerance — wall-clock noise is
   proportionally larger at small magnitudes:

| Magnitude | Tolerance | Rationale |
|---|---|---|
| < 1 µs (e.g. fusion 54 ns, privacy decide <50 ns target) | ±25% | timer/jitter dominated |
| 1 µs – 1 ms (MARL 3.3 µs, RRT-APF 43 µs, PPO 248 µs) | ±15% | criterion CI typically <5%, leave CI-runner headroom |
| > 1 ms (engine cycle vs 50 ms budget, OccWorld ~209 ms) | ±10% **and** absolute budget (50 ms / 500 ms ADR-147 §6) | budgets are the contract |

4. Hard in-source acceptance thresholds remain authoritative regardless of baseline:
   sketch ≥8× (`sketch_bench.rs`), prefilter ≥4× (`aether_prefilter_bench.rs`),
   nvsim ≥1 kHz (`pipeline_throughput.rs`), MQTT header targets, ADR-145 p95 ≤100 ms.
5. Latency stays **out of determinism hashes** (ADR-145 §2.6) but **in** the trended
   `summary.json`, so sub-threshold drift is visible (ADR-145 §3.2 mitigation).

### 4.3 Live-capture baseline gate (`benchmark_baseline.json`)

Adopt the file as the L5 regression anchor with documented provenance, then gate a
re-capture of the same scenario (same 2-node placement, same room class) against the
summary block:

| Field | Baseline | Suggested gate |
|---|---:|---|
| `presence_ratio` | 0.9336 | ≥ 0.90 for an occupied-room session |
| `confidence_mean` | 0.6433 | within ±0.10 |
| `kp_spread_std` | 4.52 | ≤ 2× baseline (skeleton stability) |
| `person_count_changes` | 10 / 1,566 frames | ≤ 2× baseline (count flapping — see CHANGELOG #803/#894 clamp bugs this metric would have caught) |

Field-trial gates are **soft** (warn + require human sign-off), never auto-merge
blockers — environments differ; the gate exists to force an explanation.

### 4.4 Wiring

Pre-merge (`CLAUDE.md` checklist): L0 + L1. Nightly: L2 criterion + ADR-145 Tier-3
ablation matrix (minutes-scale, ADR-145 §3.2). Release: full witness bundle +
`VERIFY.sh` + L4 on real COM-port hardware (`CLAUDE.md` firmware rule 6/7).

---

## 5. Reproducibility & External-Witness Requirements

Anyone outside the project must be able to re-run every claimed result:

1. **One command per layer.** `cargo test --workspace --no-default-features`;
   `python archive/v1/data/proof/verify.py`; `bash scripts/generate-witness-bundle.sh`
   then `bash VERIFY.sh` inside the bundle; per ADR-150 §4 every accuracy result needs
   "one-command reproduction" (efficiency frontier publishes its exact command:
   `python aether-arena/staging/train_efficiency_pareto.py npy/X.npy npy/Y.npy npy/split_random.npy`).
2. **Pinned numerical environment.** The Python proof requires single-threaded BLAS
   (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
   `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`) and 6-decimal quantization
   (`HASH_QUANTIZATION_DECIMALS=6`) — the #560 fix in `CHANGELOG.md`; Rust proof
   runners use coarse u16 quantization at 1e-3 in natural order
   (`calibration_proof_runner.rs` pattern, ADR-145 §2.6) for libm portability.
3. **Seeds are constants, committed:** `PROOF_SEED=42`, `MODEL_SEED=0`
   (`proof.rs`, ADR-015 Phase 5); dataset splits committed as `.npy`
   (`split_random.npy`); swarm configs as versioned YAML with all seeds (ADR-171 §5).
4. **Artifacts carry hashes.** Published model artifacts include SHA-256 (HuggingFace
   `pose_micro_int4.npz`, sha256 `c03eeb…` — efficiency-frontier doc); witness bundle
   has a `MANIFEST.sha256` over every file; provenance fields
   (`replay_sha256`, `model_sha256`, `calibration_version`, `privacy_mode`) are bound
   into ablation proof hashes (ADR-145 §2.7) so a metric cannot be quoted without its
   exact model + calibration + privacy decision.
5. **Hardware claims name the hardware.** ADR-147 records RTX 5080 / CUDA 12.8 /
   PyTorch 2.10.0; nvsim states the Cortex-A53 scaling caveat in the bench header;
   efficiency-frontier flags ARM validation as pending. Copy this discipline.
6. **Witness rows.** Every new proof gains rows in `docs/WITNESS-LOG-028.md`
   (ADR-145 §5.3 adds W-39…W-41) and the bundle's `source-hashes.txt`.
7. **Secret hygiene in evidence.** Bundle logs pass through
   `scripts/redact-secrets.py` (ADR-110 wave-5 incident note in
   `generate-witness-bundle.sh` step 4) — external evidence must never embed `.env`.

---

## 6. Known Measurement Pitfalls (WiFi-sensing specific)

| # | Pitfall | Repo evidence | Mitigation in this methodology |
|---|---|---|---|
| 1 | **Subject leakage / split optimism.** In-domain `random_split` has temporal/subject-adjacency effects; the same model family scores 83.6% random-split but ~11.6% torso-PCK on the leakage-free cross-subject split | efficiency-frontier "Controlled claim" footnote; ADR-150 §1, §3.2 | Always report the split name; publish random-split and cross-subject numbers side by side; cross-subject claims only on the official split |
| 2 | **Per-environment overfitting.** Zero-shot cross-environment collapses to 10.6%; subject-scaling saturates ~63.7% past 16–20 subjects because the residual is room/device shift | ADR-150 §3.3, §3.6 | Cross-room degradation + 17-joint heatmap in every ablation (ADR-145 §2.5); claim deployment accuracy only with the calibration protocol stated (K samples, adapter size) |
| 3 | **Mock-mode contamination.** Mock firmware missed a real Kconfig threshold bug; the nn crate ships a `mock_inference` criterion group that must never be quoted as pipeline performance | `CLAUDE.md` firmware rule 7; `inference_bench.rs` `bench_mock_inference` | L4 mandatory before firmware release ("Always test with real WiFi CSI, not mock mode"); label mock benches in reports; ADR-147 §7 re-ran the benchmark on real CSI explicitly "no mocks" |
| 4 | **Single-run point estimates.** 1.732 m localization from one synthetic geometry; 223 s coverage from an analytic formula | ADR-171 §1, §7 | §3 seed/CI protocol; evidence-grade table before publication |
| 5 | **Random-weight / untrained baselines read as results.** OccWorld MDE 9.49 m is a pre-fine-tuning random-weight reading | ADR-168-benchmark-proof §4 | Label baseline-vs-target explicitly; never aggregate untrained-model numbers into capability claims |
| 6 | **Latency conflated with quality.** Criterion µs numbers prove no compute bottleneck, nothing about accuracy | ADR-171 §2, §4.3 | L2 is gate-only; quality claims live in L3+ |
| 7 | **Floating-point nondeterminism breaking proofs.** SciPy FFT SIMD reordering + multithreaded BLAS produced different hashes across CI microarchitectures | CHANGELOG #560; `calibration_proof_runner.rs` lines 1–13 (cited in ADR-145 §2.3) | Quantize before hashing; pin thread env vars; exclude wall-clock from hashes |
| 8 | **Hash churn without procedure.** Three distinct historical values of the proof hash exist (`8c0680d7…` ADR-028, `667eb054…` CHANGELOG #560, `f8e76f21…` current file) | cited files | Every regeneration via `--generate-hash` + re-verify + CHANGELOG entry + witness bundle refresh |
| 9 | **Aggregation bugs masking accuracy.** Person count clamped to 1 by EMA mapping; eigenvalue path leaking counts up to 10; both invisible to unit tests for months | CHANGELOG #803, #894 | L5 summary gates on `person_count_changes`/count distributions; convergence tests replaying the live loop |
| 10 | **Stale verification claims.** `VERIFY.sh` prints hardcoded "(8/8)" over 10 actual checks; `CLAUDE.md` says "7/7" | `generate-witness-bundle.sh` line 293; `CLAUDE.md` | Compute the verdict count; audit doc claims against scripts each release |
| 11 | **Licensing limits on the eval set.** MM-Fi is CC BY-NC — weights trained solely on it cannot back commercial claims | ADR-015 Consequences | Track dataset license alongside every published number |

---

## 7. Gap List (what must be built to fully execute this methodology)

| Gap | Owner layer | Source |
|---|---|---|
| Machine-readable criterion baseline (`v2/benchmarks/criterion-baseline.json`) + CI comparison job | L2 | §4.2 (numbers currently only in ADR prose) |
| Provenance + producer script for `benchmark_baseline.json`; soft-gate job | L5 | §1.3, §4.3 (zero code references today) |
| `ruview-cli --ablation mode=auto` wiring + `expected_ablation_<slug>.sha256` (currently placeholders → exit 2) | L3 | ADR-145 implementation status |
| Seeded swarm `evals/` harness + `evals/RESULTS.md` internal leaderboard | L3/L5 | ADR-171 §6, §8 open issues |
| Fix `VERIFY.sh` hardcoded verdict count; reconcile `CLAUDE.md` "7/7" | L1 | §1.2 |
| Curated paired room-A/room-B labeled replay set (frozen, SHA-pinned, never trained on) | L3 | ADR-145 §3.2 |
| ARM/edge on-device latency validation for the int4 model (x86-only today) | L4 | efficiency-frontier doc ("Pi fleet pending") |
| Bench validation of the antenna-placement matrix on real hardware | L4 | PRODUCTION-ROADMAP.md Tier 2.3 |

---

## Update — falsifiable occupancy benchmark implemented

`wifi-densepose-train::occupancy_bench` (added this branch) makes the
presence/person-count claim **falsifiable in code**, directly enforcing the L3
discipline above. It grades predictions vs ground truth and gates a SOTA claim
behind a single `claim_allowed` invariant that requires **all** of:

1. `DataProvenance::Measured` — synthetic/mock data is scorable for regression
   but **never claimable** (anti-mock-contamination; the CLAUDE.md Kconfig-bug
   lesson made structural).
2. A leak-free `EvalSplit` — `validate()` refuses any split where a subject *or*
   environment id appears in both train and test (subject leakage / per-env
   overfitting).
3. `n_test ≥ min_test_samples` (small-N guard).
4. Presence F1 whose **bootstrap-CI lower bound** (deterministic splitmix64,
   seeded) clears the threshold — not the point estimate.
5. Count MAE within threshold.

The claim string is unreadable except through the gate (returns `NO_CLAIM`
otherwise) — same discipline as the `ruview-gamma` acceptance gate. 10 tests
cover each refusal path. What remains is *data*, not *method*: feed it a frozen,
SHA-pinned, subject/environment-disjoint **measured** replay set (the curated
room-A/room-B item above) and the "beyond SOTA" claim becomes a passing or
failing test, not a slogan.

---

*All values cited from: `benchmark_baseline.json`, `v2/crates/*/benches/*.rs` (15
files), `docs/adr/ADR-168-benchmark-proof.md`,
`docs/adr/ADR-171-swarm-benchmarking-evaluation-methodology.md`,
`docs/adr/ADR-145-ablation-eval-harness-privacy-leakage.md`,
`docs/adr/ADR-028-esp32-capability-audit.md`,
`docs/adr/ADR-015-public-dataset-training-strategy.md`,
`docs/adr/ADR-150-rf-foundation-encoder.md`,
`docs/benchmarks/wifi-pose-efficiency-frontier.md`,
`scripts/generate-witness-bundle.sh`, `archive/v1/data/proof/verify.py`,
`archive/v1/data/proof/expected_features.sha256`, `CHANGELOG.md`, `CLAUDE.md`,
`docs/research/sota-2026-05-22/PRODUCTION-ROADMAP.md`.*
