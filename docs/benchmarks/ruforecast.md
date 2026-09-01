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

## Append-only evidence ledger

Never replace a prior measurement. Append a row and retain the failed or stale
row when code, model, configuration, corpus, hardware, or methodology changes.

| Date | Commit | Lock SHA-256 | Host/toolchain | Backend/config | Shape | Samples | p50 | p95 | p99/max | Peak RSS | Evidence | Reproducer |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|

No rows have been accepted.
