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

## Append-only evidence ledger

Never replace a prior measurement. Append a row and retain the failed or stale
row when code, model, configuration, corpus, hardware, or methodology changes.

| Date | Commit | Lock SHA-256 | Host/toolchain | Backend/config | Shape | Samples | p50 | p95 | p99/max | Peak RSS | Evidence | Reproducer |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|

No rows have been accepted.
