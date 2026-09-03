# ADR-349: Governed local and fal.ai forecast training

- **Status**: Proposed
- **Date**: 2026-09-01
- **Deciders**: ruv
- **Owners**: RuView forecast training, security, and release maintainers
- **Tags**: forecasting, training, fal, rust, receipts, idempotency, cost-control
- **Parent**: ADR-348
- **Extends**: ADR-010, ADR-145, ADR-298, ADR-319, ADR-348
- **Supersedes**: None

## Decision

RuView Forecast will use one Rust training engine behind two deliberately
different request boundaries. The local Linux request may bind a governed,
hash-addressed RuView JSONL shard. Hosted v1 accepts only a bounded,
deterministic synthetic generator recipe. It cannot serialize a local
`TrainSpec`, `DataPolicy`, path, dataset bytes, tenant/account/workspace/site/
device/session identity, split receipt, or RuVector namespace.

The fal.ai path is a bounded execution adapter, not a remote shell. It may
train the fixed synthetic profile and export fixed artifact kinds, but it may
not choose an arbitrary command, executable, container, URL, environment
variable, output path, or release action. A later real-data hosted path needs a
new accepted ADR, provider/privacy review, and an operator-signed export grant;
it is intentionally not implemented here.

Every hosted result is untrusted and quarantined until it is exported with a
complete receipt, downloaded through the authenticated provider channel,
re-hashed locally, scanned, and bound to the original request. Only the local
release boundary may sign or activate it. Signing keys never enter fal.ai.

This ADR delegates dataset rights, clean-room evidence, model-card, patent,
trademark, and production gates to [ADR-348](./ADR-348-independent-rust-multivariate-forecasting.md)
and the [clean-room protocol](../security/ruview-forecast-clean-room.md).

**Evidence status:** typed local and hosted configuration, root-confined file
verification, atomic fixed-kind artifact publication, and cooperative
cancellation source surfaces exist. End-to-end local training and mock hosted
transport are software-testable in this change. Real fal.ai execution,
provider cancellation, cost reconciliation, quarantine promotion, and local
signature verification remain **UNMEASURED and unapproved** until their named
acceptance evidence exists.

## Context

The same training engine needs to run on a 128 GiB Linux workstation and on an
optional hosted accelerator without creating two architecture lineages. A
generic remote-command interface would allow request data to become executable
authority, make cost difficult to bound, and weaken cancellation. Sending a
general local request to fal would also expose governance and customer
identifiers that the hosted smoke path does not need.

The initial `ruview-forecast-train` crate already separates three relevant
modules:

| Module | Current contract |
|---|---|
| [`config.rs`](../../v2/crates/ruview-forecast-train/src/config.rs) | `TrainingRequest`, `ValidatedTrainingRequest`, bounded `JobId`, root-relative dataset path, exact size/SHA-256, named model profiles, typed CPU/CUDA choice, and bounded optimizer values; unknown fields are denied |
| [`artifact.rs`](../../v2/crates/ruview-forecast-train/src/artifact.rs) | verified open dataset handle, root confinement, fixed `Model`/`Manifest`/`Receipt`/`Checkpoint` kinds, atomic writes, content descriptors, and conflict-on-different-bytes idempotency |
| [`cancel.rs`](../../v2/crates/ruview-forecast-train/src/cancel.rs) | `Cancellation`, `CancelToken`, and cooperative cancellation checks for batch/checkpoint boundaries |

The training crate features keep `cli`, `server`, `fal-client`, `cpu`, and
`cuda` explicit, with default features empty. WGPU remains a model-crate
inference backend and is not advertised as a training runner. The `ruforecast`
binary requires the `cli` feature. Source presence is not execution evidence.

## Typed execution contracts

The non-serializable trusted `TrainingRequest` is built locally from a
size-bounded `LocalTrainingRequestWire`. It contains:

- stable job identity;
- a constructor-validated local split/horizon/normalization `TrainSpec`;
- either a root-relative dataset identity with exact byte count and SHA-256,
  or a deterministic synthetic recipe;
- one allowlisted model capacity profile and compiled-in local backend;
- bounded optimizer, checkpoint cadence, seed, memory, step, time, checkpoint,
  and artifact budgets.

Local JSON and TOML deny unknown fields. Dataset paths resolve below an
operator-configured root, and the exact regular file remains open after
size/hash verification to avoid replacement between validation and use. JSONL
windows are decoded incrementally with per-line and per-window caps rather
than loading the full shard into memory.

The distinct `HostedSyntheticRequestV1` contains only a schema version,
idempotency/request digests, a fixed model profile, deterministic generator
parameters and seed, bounded optimizer/resource caps, and public build
identities. Its constructor needs a non-serializable core governance authority
proving that the synthetic recipe is the approved hosted source. Hosted v1 has
no external dataset field.

Neither request contains a shell string, argument vector, executable path,
container/image selector, package installer, arbitrary URL, callback URL,
environment map, secret, source fragment, or unrestricted output name.

The local client maps `HostedSyntheticRequestV1` to one allowlisted,
process-configured fal app and pre-deployed content-addressed worker image. The
caller cannot override the app, machine, worker entry point, or artifact
destination. The endpoint returns typed status or artifact descriptors only.
Provider paths are derived from validated job identity and fixed artifact
kinds, then downloaded through the authenticated fal Platform Files API; they
are not accepted as remote URLs or local paths.

The deploy wrapper builds from an allowlisted `git archive`, selects the exact
`fal_app.py::run_server` symbol, and passes `--auth private` for both `fal run`
and `fal deploy`; ephemeral Fal runs otherwise default to public. Hosted v1
retains only bounded synthetic request/result metadata so its typed queue result
can be reconciled. It sets `X-Fal-No-Retry: 1` and a bounded timeout rather than
claiming that `X-Fal-Store-IO: 0` or an unverified lifecycle header protects
the result. A live unauthenticated 401/403 probe and provider retention/deletion
reconciliation remain blocking operational evidence.

## Idempotency and cost authority

The local effective idempotency identity is:

```text
job_id + canonical request digest + source commit + lockfile digest
+ container digest + dataset/split digest + initial-weight digest
```

Hosted v1 replaces `dataset/split digest` with the canonical synthetic-recipe
digest. No local governance identifier is part of, or derivable from, the
hosted payload.

Repeating an identical completed request returns the existing verified receipt
and descriptors. Reusing `job_id` with any different governed input fails as a
conflict. A retry after an ambiguous network failure must first query the same
provider job; it must not silently create a second billable run.

Before hosted submission, the caller sets explicit maximum steps, wall-clock
time, memory, checkpoint count, export bytes, and billable units. Submission
also requires an explicit local spend-approval record bound to the request
digest and maximum units. The adapter rejects a request when its configured
cap exceeds that approval. The final receipt records estimated and actual
provider units/cost when the provider supplies them, the price source/time,
and whether the amount is final or provider-reported. A provider estimate is
`CLAIMED`, never `MEASURED`, until reconciled against a bill.

No automatic retry may increase the approved budget. Budget changes create a
new signed authorization record. The initial wire carries operator ceilings
for wall time, billable seconds, and micro-USD, but it does not yet bind a
provider price quote or reconcile a final provider ledger. Monetary enforcement
therefore remains an operator/Fal-account control and the production hosted
spend gate stays open.

## Cancellation and checkpoints

Cancellation is cooperative and idempotent:

1. Local signal or authenticated hosted cancel marks one `job_id` cancelled.
2. Training checks `Cancellation` at every batch boundary and before/after each
   checkpoint/export boundary.
3. A cancellation checkpoint is committed atomically when safe and within the
   artifact budget; a partial file never becomes a valid artifact.
4. The terminal receipt records `cancelled`, the last completed epoch/step,
   checkpoint digest, provider state, elapsed resource units, and final known
   cost.
5. Repeating cancel returns the same terminal state.

Cancellation does not promise immediate GPU termination. Cancellation latency,
checkpoint durability, and residual provider billing are **UNMEASURED** until a
real fal.ai run supplies a receipt. A timeout or lost cancellation response
keeps the job and any output quarantined.

## Export, quarantine, and local signing

The worker may emit only the fixed artifact set from `ArtifactKind`:

| Kind | Required purpose |
|---|---|
| `Model` | Burn model record bytes |
| `Manifest` | architecture, data/split, seed, build, and clean-room identity |
| `Receipt` | request, environment, metrics, cost, status, and export lineage |
| `Checkpoint` | final or cancellation-time model weights; v1 does not claim optimizer/cursor resume |

Every descriptor includes kind, size, and SHA-256. Export acceptance requires
all mandatory descriptors, bounded lengths, exact hashes, the expected job and
request digest, no duplicate kind, and a terminal status consistent with the
artifact set.

Downloaded files enter a non-executable quarantine outside the model search
path. Local verification repeats envelope/schema/size/hash checks, SBOM and
malware/policy scans, clean-room declarations, dataset/split lineage, metric
labels, and parent-checkpoint validation. No provider output may update a
symlink, current-model pointer, RuVector index, sensing server, or release tag.

After quarantine passes, an authorized local signer binds the exact export
receipt and artifact digests to an `ArtifactReceipt` and release manifest. The
signature records algorithm, public key ID, signer capability, and time. Fal.ai
receives no private signing material and cannot promote its own output. The
current core supplies canonical digests and receipts, not a complete release
signature implementation; signature-backed activation remains open.

## Requirements and acceptance

| ID | Requirement | ADR-348 gate | Acceptance evidence | Current state |
|---|---|---|---|---|
| FT-001 | Local and hosted adapters invoke the same model/trainer implementation while using intentionally separate local-data and hosted-synthetic request schemas. | G1 | Shared-engine tests plus hosted DTO exclusion and recipe-digest binding tests | **OPEN / UNMEASURED** |
| FT-002 | No request field or endpoint can select arbitrary code, command, image, URL, environment, or output path. | G2 | Schema negatives, endpoint capability test, dependency review | **OPEN / UNMEASURED** |
| FT-003 | Job/request idempotency prevents duplicate execution and conflicts on changed governed input. | G1, G5 | Concurrent/retry/lost-response tests with provider job query | **OPEN / UNMEASURED** |
| FT-004 | Hosted execution has explicit non-escalating resource and monetary caps with estimated/actual cost receipts. | G5 | Over-budget rejection and reconciled provider-bill fixture plus real-run receipt | **OPEN / UNMEASURED** |
| FT-005 | Cancellation is authenticated, cooperative, idempotent, checkpoint-safe, and terminally receipted. | G2, G5 | Local property tests and a real hosted cancellation drill | **OPEN / UNMEASURED** |
| FT-006 | Export is fixed-kind, bounded, complete, hash-verified, and bound to request/job/environment. | G1, G2 | Missing/duplicate/truncated/tampered export tests | **OPEN / UNMEASURED** |
| FT-007 | Hosted outputs remain quarantined until local verification and local-only signing succeed. | G1, G5 | Promotion-denial tests, signing-key absence check, rollback drill | **OPEN / UNMEASURED** |
| FT-008 | Hosted v1 receives synthetic inputs only; provider credentials, logs, retention, region, output reuse, and deletion satisfy ADR-348 privacy/security approval before any real run. | G1, G5 | Hosted DTO exclusion tests, provider review, and deletion/export receipts | **OPEN / UNMEASURED** |

No successful unit test closes a hosted gate. G5 requires one bounded real
fal.ai run, one cancellation drill, one ambiguous-retry/idempotency drill, cost
reconciliation, quarantine rejection of a tampered export, and local signature
verification over the accepted artifact. Until then fal.ai support is a typed
software surface, not an operational capability claim.

## Consequences

The design gives local and hosted training one auditable lineage and keeps
provider output below local release authority. It adds receipt, budget,
quarantine, and signing work, and cooperative cancellation may still incur
provider cost. If a hosted provider cannot support bounded idempotent status,
export, cancellation, and deletion, the Linux path remains the only approved
training environment.

## References

- [ADR-348](./ADR-348-independent-rust-multivariate-forecasting.md)
- [RuView Forecast clean-room protocol](../security/ruview-forecast-clean-room.md)
- [`ruview-forecast-train` manifest](../../v2/crates/ruview-forecast-train/Cargo.toml)
- [`ruview-forecast-core` receipts](../../v2/crates/ruview-forecast-core/src/receipt.rs)
- [`ruview-forecast-model` public artifact boundary](../../v2/crates/ruview-forecast-model/src/lib.rs)
