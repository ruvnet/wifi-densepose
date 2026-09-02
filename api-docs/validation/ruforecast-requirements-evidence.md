# RuForecast requirement to evidence matrix

## Authority and current state

This matrix operationalizes ADR-348 and its ADR-349/350 children without
changing them. The ADRs own the requirements and gates. Rust tests, immutable
reports, signed receipts, and explicit human approvals supply evidence. A
checked box or green build is not permission to deploy, publish a checkpoint,
upload customer-derived data, or advance the rollout mode.

Current state: all release and operational gates are open. No benchmark result,
trained checkpoint, external training receipt, or production authority is
recorded by this document.

## Focused verification commands

The feature-off contract remains on the workspace Rust 1.89 line:

```bash
cd v2
cargo +1.89.0 test --locked -p ruview-forecast-core --no-default-features --lib --tests
cargo +1.89.0 test --locked -p ruview-forecast-model --no-default-features --lib --tests
cargo +1.89.0 check --locked -p ruview-forecast-model -p ruview-forecast-train --no-default-features --all-targets
cargo +1.89.0 test --locked -p ruview-forecast-model --no-default-features --features ruvector --lib --tests
cargo +1.89.0 clippy --locked -p ruview-forecast-core -p ruview-forecast-model -p ruview-forecast-train --no-default-features --all-targets -- -D warnings
cargo +1.89.0 clippy --locked -p ruview-forecast-model --no-default-features --features ruvector --all-targets -- -D warnings
```

Burn 0.21 CPU activation uses the explicitly separate Rust 1.92 line:

```bash
cd v2
cargo +1.92.0 test --locked -p ruview-forecast-model --no-default-features --features cpu --lib --tests
cargo +1.92.0 test --locked -p ruview-forecast-train --no-default-features --features cpu,cli --lib --bins
cargo +1.92.0 test --locked -p ruview-forecast-train --no-default-features --features cpu --test local_jsonl_smoke -- --exact local_hash_addressed_jsonl_executes_one_real_optimizer_step
cargo +1.92.0 test --locked -p ruview-forecast-train --no-default-features --features cpu,cli --test cli_smoke -- --exact cli_smoke_trains_and_writes_the_complete_candidate_set
cargo +1.92.0 test --locked -p ruview-forecast-train --no-default-features --features cpu,cli,server,fal-client --lib --bins
cargo +1.92.0 clippy --locked -p ruview-forecast-model --no-default-features --features cpu --all-targets -- -D warnings
cargo +1.92.0 clippy --locked -p ruview-forecast-train --no-default-features --features cpu,cli,server,fal-client --all-targets -- -D warnings
```

The CUDA line is compile-only. It proves that the explicitly gated types build
on Rust 1.92; it is not a GPU execution, training, latency, or compatibility
claim:

```bash
cd v2
cargo +1.92.0 check --locked -p ruview-forecast-model --no-default-features --features cuda --lib
cargo +1.92.0 check --locked -p ruview-forecast-train --no-default-features --features cuda,cli,server --lib --bins
```

The hosted boundary stays backend-free and is exercised on Rust 1.89 with no
provider credential. Tests must use local mocks; compiling these features does
not authorize network use or a hosted training run:

```bash
cd v2
cargo +1.89.0 tree --locked -e normal,build -p ruview-forecast-train --no-default-features --features cli,server,fal-client > forecast-hosted-feature-tree.txt
! grep -Eiq '(^|[[:space:]])(burn|cubecl)(-|[[:space:]])' forecast-hosted-feature-tree.txt
FAL_KEY='' cargo +1.89.0 test --locked -p ruview-forecast-train --no-default-features --features cli,server,fal-client --lib --bins --tests
FAL_KEY='' cargo +1.89.0 test --locked -p ruview-forecast-train --no-default-features --features cli,server,fal-client privacy_external_dataset_payload_is_denied
FAL_KEY='' cargo +1.89.0 clippy --locked -p ruview-forecast-train --no-default-features --features cli,server,fal-client --all-targets -- -D warnings
```

The broader repository gate remains:

```bash
cd v2
cargo +1.89.0 test --locked --workspace --no-default-features
cargo +1.89.0 bench --locked --workspace --no-default-features --no-run
```

No CUDA runtime result is implied by these commands. CUDA execution needs a
separately identified GPU, driver, toolkit, source commit, container digest,
and signed training or inference receipt.

The repository-wide security workflow separately hard-gates the checked-in
lockfile with `cargo audit --file v2/Cargo.lock --json`. The forecast workflow
also runs both modes of `scripts/csi-data-policy-check.sh`, rejects forbidden
clean-room source/import names and model blobs, checks the hosted DTO field
surface, and blocks common literal private-key, FAL-key, JWT, cloud-key and
presigned-URL patterns. These focused checks do not close dependency-license or
complete secret-scanning evidence: the repository has no committed
`v2/deny.toml`, inherited advisory/yanked-version debt needs an
owner/date/expiry baseline, and the existing general secret scanners are
non-blocking. Until dedicated retained reports exist, those parts of G0, G1,
and G5 remain open.

### Supply-chain and secret release prerequisites

| Check | Reproducer | Present authority |
|---|---|---|
| Advisory database | `cargo audit --file v2/Cargo.lock --json` | Existing blocking workflow; report retained |
| Advisory warnings | `cargo audit --file v2/Cargo.lock --deny warnings` | Not green by assertion; inherited warning debt needs an expiring reviewed baseline |
| Rust sources/licenses/bans | `cargo deny --manifest-path v2/Cargo.toml --config v2/deny.toml check advisories bans licenses sources` | Blocked until a reviewed `v2/deny.toml` is committed and CI pins `cargo-deny` |
| Forecast-path secrets | `gitleaks detect --no-banner --redact --source .` | Focused CI blocks common literal patterns; the broader GitLeaks job/report is still non-blocking |
| Tracked sensing data | `bash scripts/csi-data-policy-check.sh --self-test && bash scripts/csi-data-policy-check.sh --tracked` | Blocking in the focused hosted-boundary job |

`cargo vet check` becomes a release prerequisite only if the project commits a
vet policy and review process; listing it without that policy would be a false
supply-chain claim.

### Existing baseline debt and scope

- [`../benchmarks/physics-pose-refinement.md`](../benchmarks/physics-pose-refinement.md)
  records unrelated workspace rustfmt/warning debt, an incomplete full-workspace
  test run on that authoring host, existing RustSec debt, and yanked
  `spin 0.9.8` in the optional Burn graph. Forecast CI therefore uses focused
  warnings-denied gates and does not describe the whole workspace as green.
- [`.github/workflows/bench-regression.yml`](../../.github/workflows/bench-regression.yml)
  records an upstream stable-Rust failure in optional `ruvector-crv`; the
  focused forecast benchmarks neither enable nor inherit a CRV claim.
- The current ADR corpus contains pre-existing duplicate ADR-263 and ADR-264
  numbers. The focused contract rejects a duplicate ADR-348, ADR-349 or ADR-350
  without falsely asserting that the historic corpus already passes a global
  uniqueness lint.
- Shared GitHub-hosted timing is noisy by repository policy. Only benchmark
  compilation gates the PR; timing logs are informational until repeated on a
  named, controlled host.

## Requirements

| Requirement | Machine evidence | Human or external evidence | Acceptance authority | Current state |
|---|---|---|---|---|
| RF-001 | Forecast-path source and artifact scan; forbidden model/blob fixture scan; clean-room manifest schema and tamper tests | Current contributor attestations and clean-room custodian adjudication | ADR-348 G0 | OPEN |
| RF-002 | Canonical receipt round trip; digest/signature tamper negatives; source, config, dataset and checkpoint identifiers required | Reviewer verifies every dependency, reference, job and checkpoint lineage | ADR-348 G0 and G1 | OPEN |
| RF-003 | Dataset manifest rejects missing license, incompatible use, unknown privacy class and unresolved bytes | Data steward and legal approval for every immutable dataset digest | ADR-348 G1 | OPEN |
| RF-004 | Feature-off dependency boundary excludes Burn/CubeCL; fixed artifact/input CPU replay; offline load tests; bounded property and fuzz corpus | Named platform-class review and 24-hour accelerated replay | ADR-348 G2 | OPEN |
| RF-005 | Shape/product overflow, finite-value, timestamp, mask, schema, horizon, quantile and resource-limit tests | Review of production input caps and abstention thresholds | ADR-348 G2 | OPEN |
| RF-006 | Forecast evidence label is immutable; observation hash round trips; confidence cannot increase; derived values cannot overwrite observations | Evidence-engine and downstream schema review | ADR-348 G2 and G4 | OPEN |
| RF-007 | Frozen split manifest; train-only normalization/calibration; quantile order; interval coverage and loss by horizon/domain | Independent evaluation report on untouched site/session/device holdouts | ADR-348 G3 | OPEN |
| RF-008 | Per-split RuVector index isolation; no overlapping horizon or holdout neighbour; paired retrieval-off/retrieval-on report | Reviewer verifies index manifest and ablation comparability | ADR-348 G3 | OPEN |
| RF-009 | Capability tests prove the forecast and RuVLLM explanation have no actuator, spending, access-control, emergency or model-promotion authority | Downstream policy owner approves any advisory consumer | ADR-348 G4 | OPEN |
| RF-010 | Local and hosted receipts compare source, lock, container, data, config and initial-weight digests; provider result is untrusted until verified | Provider retention, log, credential and data-processing review | ADR-348 G1 and G5 | OPEN |
| RF-011 | Benchmark report binds commit, lock, toolchain, host, backend, config, corpus, command and evidence label | Maintainer adjudicates claim label and model card language | ADR-348 G3 through G5 | OPEN |
| RF-012 | Mode transition and rollback state-machine tests; failed activation retains the prior artifact; raw sensing continues | Signed rollback drill and authorized mode-transition record | ADR-348 G5 | OPEN |

## Governed training requirements

These rows trace ADR-349. A mock-backed test can close a software subcondition,
but cannot close the hosted operational evidence named in the ADR.

| Requirement | Machine evidence | Human or external evidence | Acceptance authority | Current state |
|---|---|---|---|---|
| FT-001 | Synthetic hosted DTO and reconstructed local request bind the same generator, model, optimizer and build identities; external manifests are rejected | Reviewer compares immutable local and provider receipts | ADR-349 and G1 | OPEN |
| FT-002 | Unknown-field and arbitrary command/image/URL/environment/path negatives | Endpoint capability and worker-image review | ADR-349 and G2 | OPEN |
| FT-003 | Concurrent/retry/lost-response idempotency state-machine tests | Real provider ambiguous-retry drill and cost reconciliation | ADR-349 and G1/G5 | OPEN |
| FT-004 | Budget boundary and over-budget rejection tests | Provider price/bill fixture review and bounded real-run receipt | ADR-349 and G5 | OPEN |
| FT-005 | Authenticated cancellation, checkpoint and terminal-state property tests | Real hosted cancellation and orphan-job drill | ADR-349 and G2/G5 | OPEN |
| FT-006 | Missing/duplicate/truncated/tampered fixed-kind export matrix | Quarantine/export receipt review | ADR-349 and G1/G2 | OPEN |
| FT-007 | Hosted-signing denial, local verification and atomic promotion/rollback tests | Release-key isolation approval and rollback drill | ADR-349 and G1/G5 | OPEN |
| FT-008 | Captured egress and log-redaction tests for credentials/privacy/retention fields | Provider DPA, region, retention, reuse and deletion review | ADR-349 and G1/G5 | OPEN |

## Predictive-memory and explanation requirements

These rows trace ADR-350. This PR creates no sensing-server, RuVector or RuVLLM
runtime authority; the rows remain open until a separately reviewed bridge
provides the evidence.

| Requirement | Machine evidence | Human or external evidence | Acceptance authority | Current state |
|---|---|---|---|---|
| PM-001 | Cross-tenant/split/version/time-scope query negatives | Signed index-manifest and tenancy review | ADR-350 and G3/G5 | OPEN |
| PM-002 | Property tests reject holdout identities and overlapping contexts/horizons | Frozen leakage-report review | ADR-350 and G3 | OPEN |
| PM-003 | Bounded zero/error/success retrieval receipts and tamper tests | Retrieval receipt schema approval | ADR-350 and G2/G3 | OPEN |
| PM-004 | Same-example, same-artifact paired retrieval-off/on evaluator | Frozen paired ablation and overhead report | ADR-350 and G3 | OPEN |
| PM-005 | Envelope signature mutation, expiry, tenant, key and replay negatives | Local signing/key-rotation receipt | ADR-350 and G2/G5 | OPEN |
| PM-006 | Evidence-monotonicity property tests reject measured-observation promotion | Evidence-engine owner review | ADR-350 and G2/G4 | OPEN |
| PM-007 | Numeric/unit/provenance mutation corpus fails closed | RuVLLM adapter review | ADR-350 and G4 | OPEN |
| PM-008 | Default-deny capability and attempted-escalation tests | Downstream policy/capability review | ADR-350 and G4/G5 | OPEN |
| PM-009 | Tenant deletion, retention and access-audit tests | Privacy and membership/extraction-risk approval | ADR-350 and G5 | OPEN |

## Threat-model contract coverage

The IDs below are defined by
[`../security/ruview-forecast-threat-model.md`](../security/ruview-forecast-threat-model.md).
The feature job compiles and runs the current mock-backed suite, but an ID stays
open until the named negative test and its retained CI report exist.

| Contract IDs | Required machine evidence | Current state |
|---|---|---|
| PRIV-001 through PRIV-009 | Egress-denial, tenant-isolation, log-redaction, nonce-isolation and tracked-data-policy negatives | OPEN |
| AUTH-001 through AUTH-005; STATE-001 through STATE-004 | Fail-closed identity/scope/service-auth tests and property-tested idempotent transition machine | OPEN |
| BOUND-001 through BOUND-007 | Boundary/property cases, parser fuzzing, cumulative resource limits and bounded accelerated replay | OPEN |
| RVEC-001 through RVEC-006 | Scope-authority construction denial, identifier/privacy bounds, isolation, cancellation and non-persistence evidence | OPEN |
| FAL-001 through FAL-012; SSRF-001; PATH-001 through PATH-003 | Mock request capture, exact-body webhook/replay cases, app/build/expiry binding, URL/DNS/redirect denial and filesystem escape negatives | OPEN |
| ART-001 through ART-005; DE-001; EVID-001 through EVID-002 | Artifact tamper/rollback/crash matrix, trusted-type construction denial and evidence-authority invariants | OPEN |

## Gate evidence bundles

| Gate | Minimum bundle before review | Status |
|---|---|---|
| G0 | Clean-room manifest, contributor attestations, repository/artifact scan report, adjudication log | OPEN |
| G1 | Dataset rights manifests, byte-level lineage report, random-initialization receipt, local/hosted digest comparison | OPEN |
| G2 | Focused tests, property/fuzz reports, deterministic replay hash, dependency-boundary tree, 24-hour replay report | OPEN |
| G3 | Frozen split manifest, baseline/model/retrieval rows, weighted quantile loss, interval calibration, reproducer and report digests | OPEN |
| G4 | Fourteen-day shadow report, empty-room/occupied-room confusion matrices, drift/abstention slices, authority audit | OPEN |
| G5 | Dedicated-host latency and process-RSS report, signed model card, SBOM/provenance, rollback drill, security/privacy/legal approvals | OPEN |

## Evidence record template

Copy one row per immutable evidence artifact. Never replace a failed result.

| Evidence ID | Requirement/gate | Commit | Lock/config/data digest | Environment | Command | Result | Evidence label | Artifact digest | Reviewer/date |
|---|---|---|---|---|---|---|---|---|---|

No evidence artifacts have been accepted.
