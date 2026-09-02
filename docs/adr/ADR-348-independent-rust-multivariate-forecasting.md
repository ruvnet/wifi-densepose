# ADR-348: Independent Rust multivariate forecasting for RuView

- **Status**: Proposed
- **Date**: 2026-09-01
- **Deciders**: ruv
- **Owners**: RuView perception, model, security, and data-governance maintainers
- **Tags**: forecasting, rust, ruvector, temporal, uncertainty, clean-room, provenance
- **Extends**: ADR-016, ADR-020, ADR-145, ADR-273, ADR-282, ADR-295,
  ADR-298, ADR-302, ADR-304, ADR-317, ADR-318, ADR-319
- **Supersedes**: None

## Executive decision

RuView will develop an independently specified and independently trained Rust
multivariate forecasting subsystem. It will forecast bounded, versioned
temporal feature streams, publish calibrated quantiles and an explicit
abstention state, optionally use a split-safe RuVector analogue index, and
remain advisory until all release gates in this ADR pass.

This is not a port, compatibility layer, distillation, or behavioural clone of
Google TimesFM. The implementation API is derived from RuView requirements.
Its weights must descend from an approved random initialization and approved
training data only.

**Evidence status at proposal:** the forecasting accuracy, false-alert
reduction, calibration, CPU latency, memory use, cross-building generalization,
and operational value described below are all **UNMEASURED targets**. This ADR
authorizes implementation and evaluation; it makes no `MEASURED` capability
claim and approves no production model.

## Context

RuView currently observes present and recent RF state. Forecasting could add a
separate answer to questions such as whether a feature trajectory is consistent
with real occupancy, whether motion is likely to transition between zones, or
whether a radio link is degrading. A forecast is not a sensor observation. It
is derived evidence with uncertainty and must never overwrite the immutable
observation that produced it.

TimesFM 3 is useful research context because its public description discusses
multivariate targets, past covariates, known-future covariates, probabilistic
outputs, temporal patching, and cross-series attention. Its licensing creates
two distinct surfaces:

- the Google TimesFM source repository states that source code is Apache-2.0;
- the TimesFM 3 pretrained weights are distributed under a separate
  non-commercial, non-production license that also restricts commercial
  training, fine-tuning, and distillation from the model.

An Apache-compliant Rust port would be a permissible but attributed derivative
source path, subject to its exact license obligations. It would not support the
strongest independent-development claim. Using the restricted TimesFM 3 model
or its outputs to create a RuView commercial model is outside this ADR. The
project therefore chooses the stricter clean-room protocol in
[`../security/ruview-forecast-clean-room.md`](../security/ruview-forecast-clean-room.md).

Clean-room controls reduce copyright, contract, and provenance risk. They do
not establish freedom to operate against patents, clear trademarks, approve a
dataset, or replace legal review in a launch jurisdiction.

## Options considered

1. **Use the TimesFM 3 checkpoint in RuView.** Rejected for commercial and
   production use under the current checkpoint license.
2. **Translate the Apache-licensed implementation into Rust.** Not selected.
   This can be evaluated under a separate ADR, with attribution and license
   compliance, but must not be mixed with the independent-development path.
3. **Build an independent Rust forecasting specialist for RuView.** Chosen.
   This narrows model scope, keeps inference native, controls provenance, and
   makes RF-specific evaluation the source of authority.
4. **Do not add forecasting.** Retained as the deployment baseline. Forecasting
   must beat simpler deterministic and statistical baselines before acquiring
   runtime authority.

## Requirements

| ID | Requirement | Verification authority |
|---|---|---|
| RF-001 | No TimesFM 3 code, configuration, weights, parameters, outputs, activations, or derivatives enter specification, implementation, training, evaluation feedback, or release artifacts. | Clean-room manifest, contributor attestations, repository and artifact scans |
| RF-002 | Every contributor, reference, dependency, dataset, transform, training job, and checkpoint has immutable provenance. | Signed provenance manifests and review receipts |
| RF-003 | Every training byte has documented rights for commercial machine-learning use and applicable privacy approval. | Dataset licensing gate and data-steward approval |
| RF-004 | The default runtime is native Rust, bounded, offline, deterministic for a fixed artifact/input within the declared platform class, and free of runtime model download. | Unit, property, fuzz, replay, and dependency tests |
| RF-005 | Inputs and outputs are versioned, finite, bounded, timestamped, mask-aware, and provenance-labelled. Invalid, stale, insufficient, or OOD inputs abstain. | Contract tests and malformed-input corpus |
| RF-006 | Forecasts remain derived evidence. They cannot rewrite observations, silently increase observation confidence, or be relabelled `MEASURED`. | Schema invariants and evidence-engine tests |
| RF-007 | Quantile ordering, interval calibration, missing-data behaviour, and abstention are evaluated on leakage-free site/session/device holdouts. | Frozen evaluation manifest and reproducible report |
| RF-008 | RuVector retrieval is optional, split-scoped, provenance-recorded, and evaluated against the same model without retrieval. | Retrieval isolation tests and paired ablation |
| RF-009 | RuVLLM may explain or summarize a signed forecast but cannot alter numeric forecasts, bypass policy, spend money, or trigger medical, emergency, access-control, or life-safety action. | Capability tests and downstream policy review |
| RF-010 | Training on local Linux or a hosted accelerator uses identical signed code, container, data, and configuration identities; the provider is an untrusted processor. | Training receipts and digest comparison |
| RF-011 | Accuracy, latency, memory, power, and operational claims remain `UNMEASURED` or `CLAIMED` until a named reproducer satisfies the repository evidence contract. | Documentation and release review |
| RF-012 | Rollout is reversible and cannot advance from offline evaluation to shadow or advisory operation without the mapped gates below. | Signed mode transition and rollback drill |

## Architecture boundary

The initial implementation is split across three crates:

| Crate | Owned responsibility | Dependency/runtime rule |
|---|---|---|
| `ruview-forecast-core` | Backend-neutral schemas, invariants, metrics, forecast receipts, and the `Forecaster` trait | No Burn, CUDA, WGPU, provider SDK, sensing-server, or network dependency |
| `ruview-forecast-model` | Independent Burn 0.21 patch-mixer architecture and artifact execution | CPU, CUDA, and WGPU are explicit optional features; every backend feature is off by default |
| `ruview-forecast-train` | Dataset manifests/splits, trainer, evaluator, the `ruforecast` CLI, and Linux/fal.ai training assets | Training-only authority; no production activation or sensing-server mutation |

This PR does not connect the forecaster to the sensing server. A shadow bridge
requires a follow-up change after the core contract and evidence receipt have
stabilized. Landing crates or passing synthetic tests therefore creates no
live RuView forecasting capability.

Version one activates only the exact reviewed `tiny_ci` and `large_linux`
architecture profiles. The model and training boundaries enforce checked
dimension, parameter, activation-cell, input-cell, and forward multiply-add
limits per batch. Adding or changing a profile is therefore a reviewed code and
artifact-schema decision, not untrusted request configuration.

The proposed logical pipeline is:

```text
authenticated RF observations
        |
        v
versioned one-second feature windows
        |
        +----> split-scoped RuVector analogue retrieval (optional)
        |
        v
independent Rust temporal model
        |
        v
point forecast + ordered quantiles + validity mask + abstention
        |
        v
signed derived-evidence record
        |
        +----> RuView policy/evidence engine
        +----> RuVLLM explanation with no numeric mutation authority
```

The initial feature contract should favour compact one-second summaries rather
than feeding an unbounded raw CSI stream. Candidate fields include motion
energy, Doppler summary, amplitude/phase dispersion, coherence, RSSI, packet
loss, channel utilization, current detector confidence, device temperature,
and source-validity masks. Each field needs a schema version, physical unit,
aggregation rule, validity semantics, and provenance. Adding a field is a
schema change, not an implicit positional extension.

The model may independently combine normalization, temporal patch encoding,
temporal mixing, cross-stream fusion, and quantile heads. This is a functional
design space, not permission to reproduce protected source expression,
checkpoint dimensions, constants, tests, diagrams, or API choices. Exact
architecture and parameter count remain implementation decisions recorded in
the model card and training receipt.

## Forecast contract

For each request the runtime returns one typed result, including abstention and
error paths. The result must identify:

- input schema, feature-window, model, configuration, and calibration digests;
- source time range, requested horizon, cadence, target streams, and masks;
- point estimate and declared quantiles for each target/horizon cell;
- whether quantiles were corrected for crossing and which method was used;
- OOD, insufficient-context, stale-input, and non-finite dispositions;
- optional RuVector index version, neighbour identifiers, distances, and
  leakage-scope receipt;
- runtime platform class, duration, and peak-memory measurement when enabled;
- evidence label `DERIVED_FORECAST`, never `MEASURED_OBSERVATION`;
- deterministic content hash and optional RVF signature.

The runtime rejects non-finite values, duplicate or non-monotonic timestamps,
unsupported schema versions, unbounded dimensions, invalid quantile requests,
future covariates without an allowed source, and payloads above configured
limits. It abstains rather than imputing an authoritative state when history or
validity coverage is below the model-card threshold.

## RuVector integration

RuVector supplies temporal memory, not ground truth. During training and
evaluation, each split has a separate index built only from records permitted
for that split. A test query may retrieve training analogues, but it may never
retrieve a window from the same subject/session/site holdout, an overlapping
target horizon, or any test record. Fitted normalizers and retrieval thresholds
come from training data only.

Every release reports three comparable rows on identical examples:

1. deterministic/statistical baseline;
2. forecasting model without retrieval;
3. forecasting model with RuVector retrieval.

This prevents retrieval from hiding a weak model or leaking future state.

## RuVLLM and action authority

RuVLLM consumes an immutable, signed forecast record and may produce an
explanation referencing its uncertainty and provenance. Its prose is not the
forecast and receives no higher evidence grade. Numeric values exposed to APIs
come from the forecasting record, not regenerated text.

No forecast or LLM explanation independently triggers a health alert, emergency
response, access decision, actuator, firmware change, model promotion, or
commercial spend. Such actions require a separate governed policy, explicit
capability, and appropriate observed evidence.

## Training and artifact boundary

Training jobs run from a pinned container or reproducible local environment.
The receipt binds the source commit, lockfile, compiler, container, datasets,
transforms, architecture, hyperparameters, seeds, hardware, provider job ID,
start/end time, and every emitted checkpoint. Hosted workers receive only the
least data and credentials necessary. Provider caches, logs, retention, and
reuse rights must pass security and data review before customer-derived data is
uploaded.

A release artifact is immutable and hash addressed. It is loaded locally after
signature, schema, size, and compatibility verification. The inference process
does not fetch models, execute embedded code, accept arbitrary operators, or
contact the training provider.

## Leakage and evaluation protocol

- Split by deployment/site first, then subject, session, device, and contiguous
  time block as applicable. No overlapping raw sequence or derived window may
  cross a split.
- Freeze test manifests before tuning. A failed test result does not become a
  new training target; material architecture changes require a new untouched
  holdout.
- Fit normalization, calibration, feature selection, thresholds, and RuVector
  index parameters on training/validation data only.
- Report missingness, abstention coverage, selective risk, per-horizon errors,
  interval coverage, quantile loss, and results by site/device/interference
  regime. Pooled performance cannot hide a failed domain.
- Compare against last value, seasonal naive, and at least one small classical
  or recurrent baseline. TimesFM 3 output is not an acceptance oracle.
- Forecast evaluation does not prove presence, pose, fall, respiration, or
  medical accuracy. Any downstream claim requires its own labelled protocol.

## Acceptance gates and requirement mapping

Targets below are release criteria, not current results.

| Gate | Requirements | Pass condition | Current state |
|---|---|---|---|
| G0 independent-authoring boundary | RF-001, RF-002 | Approved source allowlist; 100% current contributor attestations; zero restricted artifacts or outputs; all similarity findings adjudicated by the clean-room custodian. | **OPEN / UNMEASURED** |
| G1 data and lineage | RF-002, RF-003, RF-010 | 100% of training/evaluation bytes resolve to approved manifests; zero unknown, noncommercial, research-only, no-ML, or no-derivatives sources; checkpoint lineage reaches approved random initialization; local/hosted receipts bind identical governed inputs. | **OPEN / UNMEASURED** |
| G2 bounded Rust contract | RF-004, RF-005, RF-006 | Unit/property/fuzz tests cover dimensions, non-finite values, timestamp order, masks, quantile order, stale/OOD/insufficient context, deterministic hashes, offline loading, evidence labels, and resource caps; 24-hour accelerated replay has no panic or unbounded growth. | **OPEN / UNMEASURED** |
| G3 leakage-free model evidence | RF-007, RF-008, RF-011 | Frozen site/session/device-disjoint report; all baseline and ablation rows present; nominal 80% interval coverage target is 75%-85%; weighted quantile loss target is at least 10% better than seasonal naive; every metric has a reproducer. | **OPEN / TARGETS UNMEASURED** |
| G4 RuView shadow value | RF-006, RF-007, RF-008, RF-009 | At least 14 days shadow-only; empty-room false-alert target is at least 50% relative reduction without more than 2 percentage points occupied-room recall loss; no safety-critical action authority; drift and abstention reported by deployment. | **OPEN / TARGETS UNMEASURED** |
| G5 deployment fitness | RF-004, RF-009, RF-010, RF-011, RF-012 | CPU p95 target is at most 1 second for 32 declared streams with peak process memory at most 4 GiB; signed model card/SBOM/provenance; privacy, security, trademark, dataset, and patent reviews; rollback drill; no open severity 1/2 issue. | **OPEN / TARGETS UNMEASURED** |

Failure at any gate keeps the model offline or shadow-only. Passing a software
gate is not real-hardware evidence and does not convert an accuracy target into
a measurement.

## Rollout and rollback

The only permitted progression is:

```text
OFF -> OFFLINE_EVAL -> SHADOW -> ADVISORY
```

This ADR does not authorize autonomous action. Each transition records actor,
old/new mode, artifact and configuration digests, evidence report, and reason.
Regression, provenance failure, calibration drift, security incident, or
licensing uncertainty returns immediately to `OFF` or `SHADOW` while observed
RuView sensing continues unchanged.

Rollback deactivates the forecast artifact atomically, clears ephemeral model
and retrieval state, preserves signed aggregate evidence and incident records,
and never downgrades or rewrites the raw observation stream.

## Security and privacy

Forecasting can infer routines from occupancy and movement even without raw
CSI. Feature windows, neighbour identifiers, forecasts, and explanations are
therefore deployment data subject to purpose limitation, tenant isolation,
retention, deletion, access control, and audit.

Model artifacts, feature schemas, calibration, indexes, and manifests are
untrusted until their signature and digest verify. Cardinality, dimensions,
horizon, context, allocations, execution time, and concurrent requests are
bounded. Metrics use allowlisted aggregate values and exclude raw CSI, precise
room coordinates, persistent person identifiers, and unrestricted feature
payloads.

The detailed authoring, data, AI-tool, incident, trademark, and patent controls
are normative in
[`../security/ruview-forecast-clean-room.md`](../security/ruview-forecast-clean-room.md).

## Consequences

### Positive

- RuView gains a provider-neutral predictive evidence primitive with explicit
  uncertainty and abstention.
- Rust-native inference can be evaluated on CPU without requiring a Python or
  cloud runtime in production.
- RuVector retrieval becomes a measurable ablation rather than an implicit
  memory claim.
- Signed provenance supports reproducible Linux and hosted training.

### Negative

- Independent training needs substantial diverse, correctly licensed temporal
  data and untouched deployment holdouts.
- Strict source separation and manifests increase contributor and review cost.
- A plausible forecast can make weak sensing look more authoritative unless
  downstream evidence labels remain intact.
- Clean-room development does not remove patent, trademark, privacy, or dataset
  risk.

### Neutral

- This ADR does not repair an inaccurate upstream presence or pose estimator.
- It does not approve TimesFM 3 for RuView use.
- It does not select a final model size, training budget, or hosted provider.
- No production checkpoint or measured benchmark is created by this decision.

## References

- [TimesFM 3 public research article](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/)
- [Google TimesFM repository and source-license notice](https://github.com/google-research/timesfm)
- [TimesFM 3 checkpoint license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [17 U.S.C. section 102](https://www.copyright.gov/title17/92chap1.html)
- [USPTO Patent Public Search](https://www.uspto.gov/patents/search/patent-public-search)
- [USPTO comprehensive trademark clearance](https://www.uspto.gov/trademarks/search/comprehensive-clearance-search-similar-trademarks)
- [RuView Forecast clean-room protocol](../security/ruview-forecast-clean-room.md)
- [RuView Forecast model-card template](../huggingface/RUVIEW_FORECAST_MODEL_CARD_TEMPLATE.md)
- [ADR-145](./ADR-145-ablation-eval-harness-privacy-leakage.md)
- [ADR-282](./ADR-282-ruview-ecosystem-positioning.md)
- [ADR-295](./ADR-295-source-provenance-state-machine.md)
- [ADR-298](./ADR-298-model-release-sanity-gates.md)
- [ADR-302](./ADR-302-out-of-distribution-detection.md)
- [ADR-304](./ADR-304-evidence-engine.md)
- [ADR-317](./ADR-317-benchmark-multi-domain-scorecard.md)
- [ADR-318](./ADR-318-capability-certificates.md)
- [ADR-319](./ADR-319-witness-chain.md)
