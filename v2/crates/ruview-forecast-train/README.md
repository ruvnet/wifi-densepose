# RuView Forecast training

This crate is an independent Rust implementation. It has not inspected or
used Google TimesFM code, configuration, tests, weights, or outputs.

## Linux CPU smoke (Rust 1.92)

```bash
cd v2
rustup toolchain install 1.92.0
cargo +1.92.0 run --locked -p ruview-forecast-train \
  --no-default-features --features cpu,cli --bin ruforecast -- \
  smoke --job-id smoke-001 --windows 4 --output ./artifacts
```

This performs actual Burn autodiff/AdamW updates and atomically writes an
unsigned candidate, checkpoint, manifest, and training receipt beneath
`./artifacts/smoke-001/`. The candidate is untrusted and cannot activate the
runtime until a separate local release authority verifies and signs it.

Validate an unsigned candidate without granting trust:

```bash
cargo +1.92.0 run --locked -p ruview-forecast-train \
  --no-default-features --features cli --bin ruforecast -- \
  verify-candidate --candidate ./artifacts/smoke-001/model.mpk
```

## Local real-data JSONL

Real/customer RuView data is local-only. It is never accepted by the fal DTO.
The request file points to a relative, SHA-256/size-bound shard under a
capability root. Each JSONL line is one bounded `JsonlWindow` with `version`,
`series_key`, `context_start_ms`, `variates`, row-major `values`, binary
`observed_mask`, variate-major `targets`, and binary `target_mask`. The reader
opens every path component with no-follow semantics, verifies the already-open
file handle before training, re-hashes the exact bytes consumed during every
epoch, caps each line at 8 MiB, checks every window against the training split,
and uses a reservation-derived shuffle buffer capped at 64 windows. An
in-place shard mutation therefore fails before candidate publication even
when the pathname and inode remain unchanged.

The TrainSpec `dataset_digest` is:

```text
CanonicalDigest::of_bytes(b"ruview-jsonl-window-shard-v1", raw_sha256_bytes)
```

Generate a complete local-only specimen first. This creates a new private
directory and refuses to overwrite it:

```bash
cd v2
cargo +1.92.0 run --locked -p ruview-forecast-train \
  --no-default-features --features cli --bin ruforecast -- \
  prepare-local-example --directory ./local-example
```

The generated policy and values are synthetic demonstration data. Before a
real run, replace the JSONL, dataset size/SHA-256, feature-schema and dataset
digests, split membership, privacy classification, governance receipt,
retention, job ID, device, and budgets with reviewed values. The retention
timestamp is an execution admission/deadline check; this crate does not delete
the local dataset or published artifacts. Put real runs under an operator-owned
retention/deletion service and verify deletion across restart before treating
the local path as a production service.

Run:

```bash
cd v2
cargo +1.92.0 run --locked -p ruview-forecast-train \
  --no-default-features --features cpu,cli --bin ruforecast -- \
  train-local --request ./local-example/train-local.toml \
  --dataset-root ./local-example --output /srv/ruview-artifacts
```

For an NVIDIA run, build the same command with `--features cuda,cli` and set
the request's typed device to `kind = "cuda"` with `ordinal = 0`. The local
runner then uses the same model, optimizer, receipt, and artifact path as CPU;
only the explicit backend differs. A complete programmatic request and JSONL
record specimen lives in
[`tests/local_jsonl_smoke.rs`](tests/local_jsonl_smoke.rs) and executes one
real optimizer update in CI.

Use a small batch first. The validator applies both the model's 64M-forward-
cell ceiling and a conservative autodiff memory estimate. On a 128 GiB host,
set `max_memory_bytes` no higher than 96 GiB so the OS and filesystem cache
retain headroom. Policy retention must extend beyond the complete declared
wall-time budget; it is checked during streaming, optimization, checkpointing,
and each durable publication boundary. JSONL v1 supports
`normalization = "NONE"`; a request for
train-only standardization fails closed until the two-pass adapter is added.

## fal Direct Server (synthetic only)

Fal v1 sends a deterministic synthetic recipe, fixed model/optimizer profiles,
opaque request/job digests, build identities, and operator-approved time/spend
caps. It cannot encode paths, dataset bytes, policy objects, tenants, sites,
rooms, devices, sessions, or split identities. `X-Fal-No-Retry: 1` is always
sent. Bounded synthetic request/result metadata is retained by the provider so
the typed result route can work; customer data is never present. Submission is
one HTTP send; ambiguity is
reported as `RemoteUnknown` and must be reconciled rather than resubmitted.

Never run `fal run` or `fal deploy` against the current worktree. The required
wrapper fails when forecast sources are dirty, builds a temporary context from
`git archive HEAD`, admits only the three crate manifests and Rust source trees,
the lockfile/build script, and the exact Fal launcher/Dockerfile inputs, records
the Git tree and Cargo.lock SHA-256, and deletes the context afterward. From the
repository root:

```bash
python -m venv .venv-fal
. .venv-fal/bin/activate
pip install "fal==1.80.0"
fal auth login
fal auth whoami
python v2/crates/ruview-forecast-train/deploy/fal/deploy.py self-test
python v2/crates/ruview-forecast-train/deploy/fal/deploy.py run \
  --receipt /secure/operator/ruforecast-fal-run.json
python v2/crates/ruview-forecast-train/deploy/fal/deploy.py deploy \
  --receipt /secure/operator/ruforecast-fal-deploy.json
```

Keep the secret only in the process environment; never put it in a request,
config, git, logs, or a command argument:

```bash
read -rsp 'FAL_KEY: ' FAL_KEY && export FAL_KEY && echo
export RUVIEW_FAL_APP=OWNER/ruforecast
export RUVIEW_WORKER_BUILD_ID="$(jq -r .worker_build_id /secure/operator/ruforecast-fal-deploy.json)"
export RUVIEW_BUILD_MANIFEST_SHA256="$(jq -r .build_manifest_sha256 /secure/operator/ruforecast-fal-deploy.json)"
cargo +1.92.0 build --locked --release --manifest-path v2/Cargo.toml \
  -p ruview-forecast-train --no-default-features --features cli,fal-client \
  --bin ruforecast
RUF=./v2/target/release/ruforecast
"$RUF" fal submit \
  --windows 1024 --variates 8 --seed 7 --max-micro-usd 5000000 \
  --ack-unenforced-provider-cost --expires-in-seconds 6300
```

`max_micro_usd` is an operator reservation recorded in the digest-bound
request; fal does not enforce it. The acknowledgement flag is therefore
mandatory, and production billing enforcement remains an open gate. For a
substantive typed run, add (for example)
`--model-profile large-linux --epochs 4 --batch-size 2`, plus explicit
`--learning-rate`, `--weight-decay`, `--gradient-clip-norm`,
`--max-wall-time-seconds`, `--max-billable-seconds`, and
`--max-memory-bytes` reservations. The current A100 deployment caps wall and
billable reservations at 3300 seconds, leaving 300 seconds for response handoff
and cleanup before the 3600-second Direct Server timeout. It rejects a model/batch whose conservative
activation, parameter, gradient, optimizer, and serialization estimate exceeds
the memory reservation.

The client generates a fresh random hosted namespace; local job text never
affects the wire. Preserve the entire returned handle together. In addition to
the three digests/IDs below, it contains the expected worker build, build
manifest, and `artifacts_expire_at_ms`. Follow-up CLI commands obtain the build
identities from the deployment receipt environment and require both the expiry
and immutable cumulative `max_artifact_bytes` value from that handle:

```bash
RUF=./v2/target/release/ruforecast
"$RUF" fal status --request-id REQUEST_ID \
  --request-digest REQUEST_DIGEST --job-digest JOB_DIGEST \
  --artifacts-expire-at-ms EXPIRES_AT_MS \
  --max-artifact-bytes MAX_ARTIFACT_BYTES
"$RUF" fal result --request-id REQUEST_ID \
  --request-digest REQUEST_DIGEST --job-digest JOB_DIGEST \
  --artifacts-expire-at-ms EXPIRES_AT_MS \
  --max-artifact-bytes MAX_ARTIFACT_BYTES
"$RUF" fal download --request-id REQUEST_ID \
  --request-digest REQUEST_DIGEST --job-digest JOB_DIGEST \
  --artifacts-expire-at-ms EXPIRES_AT_MS \
  --max-artifact-bytes MAX_ARTIFACT_BYTES \
  --quarantine /secure/quarantine/ruforecast
"$RUF" fal cancel --request-id REQUEST_ID \
  --request-digest REQUEST_DIGEST --job-digest JOB_DIGEST \
  --artifacts-expire-at-ms EXPIRES_AT_MS \
  --max-artifact-bytes MAX_ARTIFACT_BYTES
```

Fal output paths are relative to `/data`; for example the worker file
`/data/ruview-forecast/artifacts/JOB/model.mpk` is retrieved through
`/v1/serverless/files/file/ruview-forecast/artifacts/JOB/model.mpk`. The Rust
client accepts only a strict result bound to the request/job/build/expiry identities,
exactly one model, manifest, receipt, and checkpoint descriptor, and
`production_signed=false`. Before the first artifact request it rejects a
checked four-descriptor byte sum above the submitted cumulative cap. It then
downloads all four into a capability-confined quarantine store and rechecks
byte counts and SHA-256 values.

The image compiles Rust 1.92 with the `cuda` backend on CUDA 12.8 and the Direct
Server chooses CUDA whenever that feature exists. Build identifiers are
compiled into the worker and must exactly match the digest-bound request. The
wire never accepts a free-form command or image.

The server admits one training execution at a time. Its expiry includes a
30-minute queue-start allowance, the approved billable interval, and a
15-minute result/quarantine grace period. Cleanup is scheduled before training,
so success, cancellation, and failure all receive the same best-effort expiry
cleanup. The cancellation checkpoint in v1 is export-only model weights; it
does not contain optimizer/window state and cannot resume training.

Provider handoff is experimental. Candidate files are namespaced by the random
hosted digest and a worker thread makes a best-effort deletion at the on-wire
expiry, but a worker crash can prevent that cleanup. Use a dedicated private
fal app/account with provider-side expiry/reconciliation and never treat `/data`
as durable or production-ready storage. Provider queue metadata remains until
its separately configured provider lifecycle removes it. Complete a real
private-app acceptance test before relying on `fal result` or `fal download`.
