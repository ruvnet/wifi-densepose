# RuForecast benchmark protocol

## Evidence status

No RuForecast runtime, accuracy, calibration, memory, or operational result is
recorded here yet. Every threshold below is an ADR-348 target, not a measured
claim. The first accepted row must identify a clean commit, the `Cargo.lock`
digest, the exact Rust toolchain, host, backend, model/configuration digest,
fixture or corpus digest, command, and evidence label.

## Benchmark boundaries

RuForecast uses two different forms of evidence:

1. Deterministic correctness evidence comes from Rust unit, property, replay,
   split-isolation, and artifact-tamper tests. A fixed input and artifact must
   produce the same output within its declared platform class.
2. Runtime evidence comes from Criterion on a named host. Criterion inputs are
   generated from fixed checked-in code and seeds, but elapsed time is not
   deterministic. Timing from shared GitHub runners is informational only.

The benchmark implementation must not download a model or dataset, read raw
CSI, use a hosted model output, or enable CUDA implicitly. CPU benchmarks use
the explicit `cpu` feature and the Burn ndarray backend. CUDA validation belongs
to a separately governed Linux or hosted-accelerator receipt.

## Required benchmark targets

| Package | Target | Purpose | CI authority |
|---|---|---|---|
| `ruview-forecast-model` | `forecast_inference` | Fixed-seed forward pass, batch and shape scaling, ordered-quantile output | Compile gate; shared-runner timing is informational |
| `ruview-forecast-train` | `data_pipeline` | Fixed generated records through validation, windowing, masking and batching | Compile gate; shared-runner timing is informational |

Both targets must use code-generated synthetic inputs, fixed seeds, bounded
allocations, `criterion::black_box`, and `required-features = ["cpu"]`. Setup,
artifact construction, and dataset generation stay outside the timed region
unless a benchmark name explicitly says they are included.

The model implementation owns structural parameter-count assertions. The
currently reviewed design values are 35,700 parameters for the tiny CI preset
and 20,285,108 for the large preset. These are design invariants, not benchmark
results, and must be derived by a test from the actual module graph before they
are quoted in a model card.

## Local Linux reproducer

Run from a clean checkout after installing Rust 1.92.0:

```bash
RUFORECAST_CPUSET=0-7 \
RUFORECAST_THREADS=8 \
scripts/run-ruforecast-benchmarks.sh
```

The runner executes the focused contract/model/training tests, one real
optimizer step over a local hash-addressed synthetic JSONL shard, and the
idempotent synthetic CLI smoke. It then compile-checks both Criterion targets,
runs the targets, captures the CPU and toolchain metadata, and hashes every
output. A failed run retains its partial logs with `status=FAILED` and its exit
code rather than looking like a complete report. Results go under
`target/ruforecast-evidence/`, which is excluded from source control.

For a conservative single-thread reproducibility check, omit both environment
variables. To run against an uncommitted tree for diagnosis only, set
`RUFORECAST_ALLOW_DIRTY=1`; the resulting metadata is labelled `SYNTHETIC`
with scope `DIRTY_WORKTREE_DIAGNOSTIC_ONLY` and cannot support a release claim.

A clean run labels its host timing `MEASURED` and its input class `SYNTHETIC`,
but remains `UNREVIEWED`. Only a maintainer may append it to the accepted ledger
after checking the digests, shape, command, Criterion report and host scope.

The runner intentionally has no CUDA option and does not parse Criterion output
into a pass/fail performance verdict. This prevents a noisy host result from
silently acquiring release authority.

To compile the two benchmark targets without measuring them:

```bash
cd v2
cargo +1.92.0 bench --locked -p ruview-forecast-model \
  --no-default-features --features cpu --bench forecast_inference --no-run
cargo +1.92.0 bench --locked -p ruview-forecast-train \
  --no-default-features --features cpu --bench data_pipeline --no-run
```

For a short informational run, use the same targets without `--no-run`:

```bash
cargo +1.92.0 bench --locked -p ruview-forecast-model \
  --no-default-features --features cpu --bench forecast_inference -- \
  --warm-up-time 1 --measurement-time 2 --sample-size 10
cargo +1.92.0 bench --locked -p ruview-forecast-train \
  --no-default-features --features cpu --bench data_pipeline -- \
  --warm-up-time 1 --measurement-time 2 --sample-size 10
```

Running these commands does not add a ledger row automatically. Preserve the
raw report and environment metadata, then have a maintainer assign its evidence
scope before publishing a number.

The inference bench runs only `tiny_ci` by default so a routine CI trend step
cannot accidentally start the very expensive large CPU probe. Set
`RUFORECAST_BENCH_LARGE=1` only on a controlled host when intentionally
measuring the fixed deployment shape:

```bash
RUFORECAST_BENCH_LARGE=1 scripts/run-ruforecast-benchmarks.sh
```

## Deployment measurement shape

The initial CPU deployment probe is batch 1, context 1,024, 32 declared feature
streams, the fixed `large_linux` horizon of 300, and all seven declared
quantiles. Record at least 20 warmup
iterations and 200 measured iterations for a release candidate. Report p50,
p95, p99 or maximum, throughput, and the process peak resident set size.

ADR-348 G5 currently targets p95 at or below one second for 32 declared streams
and peak process memory at or below 4 GiB. A Criterion result alone cannot close
the memory gate because Criterion and Cargo are not the production inference
process. G5 remains open until a standalone inference probe reports its own peak
resident set size.

## Accuracy and calibration protocol

Runtime speed never substitutes for forecasting quality. The frozen evaluation
manifest must report identical examples for:

1. Last-value and seasonal-naive baselines.
2. RuForecast without RuVector retrieval.
3. RuForecast with split-scoped RuVector retrieval.

Required report fields include weighted quantile loss by horizon, nominal 80%
interval coverage, missingness, abstention coverage, selective risk, site and
device slices, interference regime, and retrieval ablation. ADR-348 G3 targets
weighted quantile loss at least 10% better than seasonal naive and 80% interval
coverage between 75% and 85%. Those targets remain unmeasured until a frozen,
leakage-free report is attached.

### Informal HPO exploration note (unaccepted, not a release claim)

**2026-09-01.** An exploratory session ran the accuracy protocol above end to
end against a governed 24-window **synthetic** dataset (`tiny_ci` profile,
context 64 / horizon 12, temporal train/test split, not entity-holdout —
only one synthetic generator was used, so entity holdout does not apply) and
a small `OptimizerSpec` hyperparameter search (learning rate, weight decay,
gradient clip norm, batch size, epochs) using a new Darwin Mode numeric-genome
evolution engine (upstream: `ruvnet/metaharness` PR #260, not yet merged).
This is **exploratory evidence only** — not a frozen, leakage-free,
maintainer-reviewed report, and not eligible for the ledger below until one
is produced.

Prior to this exploration, a **single real household window** (76 real
1&nbsp;Hz vital-signs samples, one physical ESP32 sensor, temporal not entity
holdout) scored **worse than both baselines** (WQL 0.537 vs. last-value
0.106 and seasonal-naive 0.123) — consistent with a single training window
overfitting rather than generalizing.

With a larger (still synthetic, still `tiny_ci`) 24-window training set and
three rounds of hyperparameter search, weighted quantile loss on the held-out
synthetic split improved and stayed ahead of both baselines throughout:

| Round | learning_rate | weight_decay | grad_clip | batch | epochs | WQL (model) | WQL (last-value) | WQL (seasonal-naive) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default config | 0.0010000 | 1.00e-4 | 1.000 | 8 | 60 | 0.257 | 0.277 | 0.514 |
| Search round 1 | 0.0002356 | 3.05e-6 | 0.100 | 27 | 195 | 0.161 | 0.277 | 0.514 |
| Search round 2 | 0.0000298 | 4.09e-11 | 4.746 | 26 | 356 | **0.153** | 0.277 | 0.514 |

Round-2 gain over round 1 (−0.008) was much smaller than round-1's gain over
the default (−0.096) — a diminishing-returns signal consistent with a local
optimum for this model size and dataset, not a converged global result.
`gradient_clip_norm` landed at opposite bound extremes across rounds
(0.1 then 4.7), so no directional recommendation on that parameter should be
drawn from this exploration alone.

**Explicit scope limits — do not generalize beyond these:**
- `tiny_ci` only. Nothing here has been run against `large_linux`; its far
  larger parameter count and different compute profile mean these
  hyperparameters are not a starting point for it without their own search.
- Synthetic dataset only (24 windows, one generator/seed family). Not
  validated against any real corpus at this scale.
- Self-signed, evaluation-only model activation (a throwaway local Ed25519
  key, not a release signature) was used to run inference for scoring.
- No security/provenance/maintainer-approval gate has passed — the Darwin
  Mode promotion rule correctly refused to promote any candidate here.

Reproducer: `harness/ruview/flywheel/ruforecast/` (genome, gate, evaluator,
dry-run/`--confirm` driver) in the `ruvnet/RuView` repo, paired with
`ruvnet/metaharness` PR #260 (`evolve-numeric`) linked locally via
`npm link`. Neither the genome defaults here nor any repo default config
were changed by this note — it is a record of exploratory evidence, not a
committed recommendation.

### Amendment (2026-09-01, later same day): the round-2 result above did not generalize -- retracted

**The "Search round 2" row above (WQL 0.153, learning_rate=0.0000298 etc.) is
RETRACTED as a claim of improvement.** It is kept in the table (append-only,
never silently edit a prior measurement) but must be read together with this
amendment: independent verification, run the same day using a new
regression-candidate promotion path (`ruvnet/autogenous` PR-in-progress,
branch `feat/regression-candidate-kind`, not yet merged/pushed --
`v2/crates/ruforecast-autogenous-bridge` in this worktree) against TWO FRESH
synthetic corpora that were never part of that search (seeds 1000/1097, vs.
the search's single fixed seed 0), showed that "winner" genome performing
**WORSE than the baseline on both**:

| Judge corpus | Candidate WQL | Baseline WQL | Candidate beats baseline by |
|---|---:|---:|---:|
| seed 1000 | 0.155 | **0.099** | -0.056 (worse) |
| seed 1097 | 0.219 | **0.109** | -0.110 (worse) |

**Root cause**: every evaluation in the three search rounds above (default,
round 1, round 2) trained and scored every candidate against the exact same
fixed synthetic corpus (`prepare-synthetic-dataset`'s implicit `--seed 0`
default). The search had learned to exploit that one corpus's specific
random windows -- textbook overfitting -- not found a genuinely better
hyperparameter configuration. This is a real, measured failure mode, not a
hypothetical caveat.

**Fix applied**: `harness/ruview/flywheel/ruforecast/gate.mjs`'s
`evaluateGenome` now trains and scores every candidate against THREE
independent synthetic corpora (`DEFAULT_SEARCH_SEEDS = [11, 23, 47]`, none
of which overlap the retracted search's seed 0 or the verification seeds
1000/1097) and takes the WORST CASE `primary` across them, not an average --
a candidate only counts as a win if it beats both baselines on every corpus.
See `harness/ruview/flywheel/ruforecast/README.md`'s "Multi-seed fitness"
section for the full account.

**Re-running the search with the fix, still no verified improvement.** A
fresh search under the corrected multi-seed fitness (2 generations x 2
children, cleared `.metaharness-numeric` archive so no stale pre-fix state
leaked in) found a genuine winner ON ITS OWN THREE SEARCH SEEDS --
`learning_rate=0.008012, weight_decay=0.000159, gradient_clip_norm=0.267,
batch_size=24, epochs=20`, beating the baseline on all three (primary
0.099 / 0.556 / 0.576, worst-case 0.099). Independent verification against
the same fresh seeds 1000/1097 (never part of this candidate's own search)
again showed it losing to the baseline on both:

| Judge corpus | Candidate WQL | Baseline WQL | Candidate beats baseline by |
|---|---:|---:|---:|
| seed 1000 | 0.178 | **0.099** | -0.079 (worse) |
| seed 1097 | 0.314 | **0.109** | -0.205 (worse) |

**Honest conclusion**: two independent search rounds (pre- and post- the
multi-seed fitness fix) both produced a genome that looked like a real
improvement on its own search data and both failed independent out-of-sample
verification. This converges on a different, deeper explanation than "the
search methodology was broken" (that part IS fixed): at this dataset scale
(24 synthetic training windows), held-out WQL varies enormously by which
corpus is drawn REGARDLESS of hyperparameters -- the baseline genome itself,
evaluated with the corrected multi-seed fitness function, swings from
primary 0.83 (a strong win) to a full regression (primary 0, WQL worse than
last-value) purely from which of the three fixed search seeds was used, with
identical hyperparameters throughout. The corpus-noise floor at n=24
windows appears to dominate any real hyperparameter effect. No RuForecast
hyperparameter configuration has been shown, by this exploration, to
reliably beat the trivial baselines out-of-sample at this scale. A larger
training corpus (more windows, ideally real governed data under the
`large_linux` profile) is the more promising next lever than further
hyperparameter search on this fixture.

A related implementation-only fix: the promotion verifier
(`envelope::regression::verify_regression_promotion` in the Autogenous
branch above) initially required all judges' receipts to share one
`corpus_id`, inherited unreviewed from a same-evidence review model that
does not fit this kind's intentionally cross-corpus judge design. This was
corrected (`ReceiptCorpusMismatch` is no longer produced by that function);
it did not change either REJECT verdict above, both of which were already
correctly driven by the real `NotBetterThanParent` signal.

Reproducer for both verification runs above: same as below, plus
`v2/crates/ruforecast-autogenous-bridge` (`cargo +1.92.0 run --manifest-path
crates/ruforecast-autogenous-bridge/Cargo.toml -- --ruforecast-bin
./target/debug/ruforecast --candidate-genome <genome>.json --parent-genome
<baseline>.json --judges 2 --work-dir <scratch>`) against the unpushed
`ruvnet/autogenous` branch `feat/regression-candidate-kind`.

### Real-household-data result (2026-09-01, MEASURED, unaccepted, not a release claim)

Following the informal HPO exploration above, this session also collected
**6,390 real 1&nbsp;Hz vital-signs samples** (heart rate, breathing rate,
signal quality) from a real, live, ESP32-sourced household sensing
deployment over a continuous 2-hour window (88.75% real sample coverage;
gaps handled honestly via `observed_mask=0`, never fabricated
interpolation) — the "more real training data" lever flagged as the
credible next step in the note above. This directly answers that open
question.

Two genuinely independent temporal splits of the same real corpus (not
synthetic seeds — real data has no seed to vary, so independence here
means two different train/test boundary choices on the same timeline,
each with its own 90s embargo gap) were trained (default, untuned
`OptimizerSpec`: `lr=0.001, weight_decay=0.0001, gradient_clip_norm=1.0,
batch_size=8, epochs=60`) and scored via the real `evaluate` CLI, then
independently, cryptographically verified through
`ruforecast-autogenous-bridge`'s real signed regression-candidate
promotion path (`ruvnet/autogenous`, `envelope::regression`):

| Judge | Split | Real test windows | Model WQL | Best trivial baseline WQL | Model beats baseline by |
|---|---|---:|---:|---:|---:|
| 1 | 70% train / 90s embargo / 30% test | 27 | 0.0514 | 0.0563 (last-value) | +0.0049 |
| 2 | 50% train / 90s embargo / 50% test | 46 | 0.0670 | 0.0543 (seasonal-naive) | −0.0128 |

**Signed verdict: REJECT.** Judge 1's nominal win (+0.0049) is below the
0.01 non-inferiority margin, so it doesn't clear the promotion bar even
on its own; Judge 2 lost outright. Both rejections are recorded as
`NotBetterThanParent` in the signed promotion envelope.

**Honest conclusion:** even with a real household corpus (n=6,390 real
samples, not synthetic), the result is exactly the same shape as every
synthetic search this session — a result that looks like a win on one
split does not hold up on an independently verified second split. This
is not evidence that real data can't help; it is evidence that this
scale of real data (6,390 samples, one household, one physical sensor)
is not yet enough to distinguish a genuine effect from split-dependent
noise. The credible next lever remains more real data — more households,
longer collection windows, or the `large_linux` profile — not further
hyperparameter search on any fixture this small, synthetic or real.

Reproducer: `v2/crates/ruforecast-autogenous-bridge/examples/real_data_verify.rs`
and `v2/crates/ruforecast/crates/ruforecast-train/examples/real_data_windows.rs`
in this worktree (`train/ruforecast-rust` branch, commits `130f547` in the
`ruforecast` submodule and `788401c5` in this repo — both local, not yet
pushed). Raw real vitals data never left the collecting/training hosts and
was never written to any git-tracked or pushed path.

## Append-only evidence ledger

Never replace a prior measurement. Append a row and retain the failed or stale
row when code, model, configuration, corpus, hardware, or methodology changes.

| Date | Commit | Lock SHA-256 | Host/toolchain | Backend/config | Shape | Samples | p50 | p95 | p99/max | Peak RSS | Evidence | Reproducer |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|

No rows have been accepted.

### Real public dataset: BIDMC PPG/Respiration (2026-09-02, cross-entity holdout)

Every real-data test up to this point used a single household/entity with only
a **temporal** holdout (same physical sensor, different time windows). This
test is the first with a genuine **cross-entity** holdout: 53 real ICU
patients from the BIDMC PPG and Respiration Dataset (PhysioNet, Open Data
Commons Attribution License v1.0, public and openly licensed — no
credentialing, https://physionet.org/content/bidmc/1.0.0/), splitting by
*patient*, not by time, so the held-out test set contains real people the
model never saw during training.

25,546 real 1 Hz rows across 53 recordings (~8 minutes each), heart rate +
respiratory rate + SpO2. 3 of 53 patients' windows were excluded honestly
(genuine sensor-dropout `NaN` values in the source recordings, not
fabricated/interpolated). Two independent, disjoint patient-partition splits:

| Judge | Train patients | Test patients | Test windows | Model WQL | Best baseline WQL | Result |
|---|---:|---:|---:|---:|---:|---|
| A (contiguous split) | 34 | 16 | 16 | 0.01964 | 0.01159 (last-value) | worse, +69% |
| B (interleaved split) | 24 | 26 | 26 | 0.07022 | 0.01002 (last-value) | worse, +601% |

**Same conclusion as every prior test this session**, now on real, public,
multi-subject clinical data with genuine cross-entity generalization: the
model does not beat trivial forecasting baselines. The margin is decisive on
both independent splits, not a near-miss — the earlier hypothesis that a
larger, genuinely diverse real dataset (many different people, not one
household) might change the picture does not hold at this scale/model
configuration either.

**Honest scope note on verification**: prior real-data tests in this document
were independently checked through Autogenous's signed regression-candidate
promotion path. That additional cryptographic-signing step was **not** run
for this test — the result is reported directly from the `evaluate` CLI's
real output on two genuinely disjoint, real patient-holdout splits, which is
itself real, independent, out-of-sample evidence, but it does not carry a
signed promotion-gate verdict the way the earlier entries do. Flagging this
explicitly rather than presenting it with the same evidentiary weight.

Reproducer: `v2/crates/ruforecast/crates/ruforecast-train/examples/bidmc_prepare.rs`
(untracked scratch example, worktree `train/ruforecast-rust`) — downloads
`bidmc_NN_Numerics.csv` for patients 01-53 directly from PhysioNet, builds
one 76-row (context 64 + horizon 12) window per eligible patient, and writes
governed `train.jsonl`/`test.jsonl`/`train-local.toml` per judge split. Raw
data cached at `/tmp/bidmc-raw/` on the training host only.
