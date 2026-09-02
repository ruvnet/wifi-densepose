# RuView Forecast model card template

> **Template only. Do not publish or treat this file as a model release.** Copy
> it for one immutable candidate, replace every `<REQUIRED>` field, and retain
> every unresolved item as an explicit blocker. Deleting a placeholder does not
> satisfy it.

This template implements the model-card requirements in
[ADR-348](../adr/ADR-348-independent-rust-multivariate-forecasting.md) and the
[RuView Forecast clean-room protocol](../security/ruview-forecast-clean-room.md).
It does not imply that source, training, evaluation, clean-room, security,
privacy, trademark, patent, or production gates have passed.

## Completion rules

1. Create one card per exact weight digest. Do not reuse a card across retrains.
2. Label every numeric claim `MEASURED`, `SYNTHETIC`, `CLAIMED`, or
   `UNMEASURED` under the repository evidence policy.
3. A `MEASURED` value needs an immutable dataset/split, artifact digest,
   configuration, hardware/environment, metric definition, and reproducer.
4. Use `UNMEASURED` when evidence does not exist. Do not substitute an estimate.
5. State source-code, weight, dataset, and service terms separately.
6. Do not publish raw CSI, customer data, precise room coordinates, identities,
   secrets, private contracts, or private contributor records in this card.
7. A completed card documents evidence; it cannot waive an ADR-348 gate.

## Suggested Hugging Face metadata

Copy and complete this front matter in the release card:

```yaml
---
license: <REQUIRED exact weight-license identifier>
tags:
  - time-series-forecasting
  - multivariate-forecasting
  - wifi-sensing
  - rust
  - ruvector
  - probabilistic-forecasting
  - edge-ai
language:
  - en
library_name: burn
pipeline_tag: time-series-forecasting
---
```

`license: other` is acceptable only when the card links the complete exact
license. Do not infer the weight license from the Rust source license.

## Candidate identity

Replace the template title with `<REQUIRED model name and version>` in the
release copy.

## Release decision and evidence status

| Field | Required value |
|---|---|
| Release candidate ID | `<REQUIRED>` |
| Weight SHA-256 | `<REQUIRED>` |
| Git commit | `<REQUIRED>` |
| Model-card SHA-256 | `<REQUIRED after finalization>` |
| Proposed mode | `OFFLINE_EVAL`, `SHADOW`, or `ADVISORY`; `<REQUIRED>` |
| ADR-348 highest gate passed | `G0` through `G5`, or `NONE`; `<REQUIRED>` |
| Accuracy status | `UNMEASURED` until a qualifying report exists; `<REQUIRED>` |
| Calibration status | `UNMEASURED` until a qualifying report exists; `<REQUIRED>` |
| CPU latency status | `UNMEASURED` until a named CPU reproducer exists; `<REQUIRED>` |
| Memory status | `UNMEASURED` until a named runtime reproducer exists; `<REQUIRED>` |
| Cross-site generalization status | `UNMEASURED` until untouched site holdouts exist; `<REQUIRED>` |
| Clean-room status | `OPEN`, `PASS`, or `FAIL`; `<REQUIRED>` |
| Production approval | `NOT APPROVED` unless all G5 receipts are signed; `<REQUIRED>` |

### Release summary

`<REQUIRED: State what this exact artifact does, what evidence exists, which
mode is requested, and the most important unresolved limitation in no more
than 150 words. Do not copy competitor marketing language.>`

## Model details

| Field | Value |
|---|---|
| Developer/owner | `<REQUIRED>` |
| Model family | `RuView Forecast` |
| Model version | `<REQUIRED immutable version>` |
| Artifact format | `<REQUIRED, including RVF/safetensors/version>` |
| Parameter count | `<REQUIRED or UNMEASURED>` |
| Numeric precision | `<REQUIRED>` |
| Architecture/config digest | `<REQUIRED>` |
| Input schema version | `<REQUIRED>` |
| Output schema version | `<REQUIRED>` |
| Maximum context | `<REQUIRED with cadence and units>` |
| Supported horizons | `<REQUIRED with cadence and units>` |
| Declared quantiles | `<REQUIRED>` |
| Training framework | `<REQUIRED exact Rust crate/features/version>` |
| Inference runtime | `<REQUIRED exact Rust crate/features/version>` |
| Source license | `<REQUIRED SPDX expression>` |
| Weight license | `<REQUIRED exact license and URL>` |

### Rust implementation boundary

Describe the exact code surfaces used by this candidate:

| Crate or binary | Version/digest | Responsibility | Enabled features |
|---|---|---|---|
| `ruview-forecast-core` | `<REQUIRED>` | Backend-neutral schemas, invariants, metrics, receipts, and `Forecaster` trait | `<REQUIRED; expected default only>` |
| `ruview-forecast-model` | `<REQUIRED>` | Independent Burn 0.21 patch mixer and artifact execution | `<REQUIRED; CPU/CUDA/WGPU are opt-in and default off>` |
| `ruview-forecast-train` / `ruforecast` | `<REQUIRED>` | Dataset splits, trainer, evaluator, training receipt, and Linux/fal.ai assets | `<REQUIRED>` |
| `<optional service/runtime>` | `<REQUIRED or NONE>` | `<REQUIRED>` | `<REQUIRED>` |

State whether inference is offline and whether any runtime network capability
exists. The expected production answer is no runtime model download and no
network requirement:

`<REQUIRED>`

State whether a sensing-server bridge exists. For the initial ADR-348 PR the
required answer is `NO: contracts/training only; a separately reviewed shadow
bridge is deferred`:

`<REQUIRED>`

## Intended use

### Approved candidate use

`<REQUIRED: identify the exact RuView feature streams, deployment class,
forecast horizons, mode, users, and decision support purpose.>`

### Out-of-scope and prohibited use

This artifact must not be used unless a later approved card explicitly changes
the boundary:

- as a sensor observation or ground-truth label;
- as the sole source for medical, emergency, fall-response, industrial-safety,
  access-control, policing, insurance, employment, or autonomous-actuation
  decisions;
- to infer identity or protected characteristics;
- outside the feature schema, cadence, hardware, site, population, and horizon
  validated by this card;
- after calibration, feature schema, source provenance, or OOD checks fail;
- to train another model unless the weight, dataset, and output licenses and a
  new provenance review explicitly permit it;
- to claim TimesFM compatibility, affiliation, endorsement, or equivalence.

Add candidate-specific exclusions:

`<REQUIRED>`

## Functional architecture

Describe independently implemented components without reproducing external
source expression, diagrams, identifiers, or constants:

```text
<REQUIRED: compact original diagram of feature windows, optional split-safe
retrieval, normalization, temporal/cross-stream model, quantile heads,
abstention, receipt, and policy boundary>
```

| Component | Candidate implementation | Security/resource bound |
|---|---|---|
| Feature validation | `<REQUIRED>` | `<REQUIRED>` |
| Normalization | `<REQUIRED>` | `<REQUIRED>` |
| Temporal encoding/mixing | `<REQUIRED>` | `<REQUIRED>` |
| Cross-stream fusion | `<REQUIRED>` | `<REQUIRED>` |
| Missing-data handling | `<REQUIRED>` | `<REQUIRED>` |
| Point/quantile head | `<REQUIRED>` | `<REQUIRED>` |
| Quantile crossing policy | `<REQUIRED>` | `<REQUIRED>` |
| OOD/abstention | `<REQUIRED>` | `<REQUIRED>` |
| RuVector retrieval | `<REQUIRED or NONE>` | `<REQUIRED>` |
| Artifact verification | `<REQUIRED>` | `<REQUIRED>` |

## Input contract

### Feature schema

| Field | Unit | Cadence/aggregation | Range | Missing/invalid semantics | Source provenance |
|---|---|---|---|---|---|
| `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

State:

- minimum and maximum context length;
- maximum targets, covariates, horizon, and batch;
- timestamp monotonicity and duplicate policy;
- whether any future-known covariate is permitted and how its authority is
  verified;
- finite/range checks and allocation limits;
- minimum validity coverage before abstention;
- treatment of cadence gaps, resets, device changes, and calibration changes;
- OOD input behaviour.

`<REQUIRED>`

### Data not accepted

`<REQUIRED: include raw/unbounded payloads, unknown schema, non-finite values,
untrusted future covariates, stale data, unsupported cadence, and candidate
specific exclusions.>`

## Output contract

| Output | Shape/unit | Meaning | Evidence class |
|---|---|---|---|
| Point forecast | `<REQUIRED>` | `<REQUIRED>` | `DERIVED_FORECAST` |
| Quantile forecast | `<REQUIRED>` | `<REQUIRED>` | `DERIVED_FORECAST` |
| Validity mask | `<REQUIRED>` | `<REQUIRED>` | `DERIVED_METADATA` |
| Abstention/disposition | `<REQUIRED>` | `<REQUIRED>` | `DERIVED_METADATA` |
| OOD score/state | `<REQUIRED>` | `<REQUIRED>` | `DERIVED_METADATA` |
| Receipt/provenance | `<REQUIRED>` | `<REQUIRED>` | `SIGNED_METADATA` if signed |

The output must bind model, configuration, input schema, calibration, source
time range, optional retrieval index, and content hashes. Explain whether and
how quantile crossing is corrected:

`<REQUIRED>`

## RuVector retrieval

| Field | Value |
|---|---|
| Enabled | `<REQUIRED true/false>` |
| Index artifact/version/digest | `<REQUIRED or NONE>` |
| Allowed retrieval corpus | `<REQUIRED>` |
| Split-isolation receipt | `<REQUIRED>` |
| Neighbour exclusion rules | `<REQUIRED>` |
| Maximum neighbours/search budget | `<REQUIRED>` |
| Behaviour when index is absent/stale | `<REQUIRED>` |

Report identical-dataset ablations for no retrieval and retrieval. If these do
not exist, state `UNMEASURED`.

## RuVLLM integration

State whether RuVLLM consumes this artifact's signed forecast records. It may
explain results but must not rewrite point/quantile values or acquire action
authority from prose.

| Control | Evidence |
|---|---|
| Numeric outputs originate only from signed forecast record | `<REQUIRED or NOT INTEGRATED>` |
| Explanation cites artifact and forecast receipt | `<REQUIRED or NOT INTEGRATED>` |
| No model promotion, spending, actuation, or safety-critical authority | `<REQUIRED or NOT INTEGRATED>` |
| Prompt/data retention and tenant policy | `<REQUIRED or NOT INTEGRATED>` |

## Independent-development record

| Record | Digest/approval |
|---|---|
| Approved source allowlist | `<REQUIRED>` |
| Specification digest | `<REQUIRED>` |
| Contributor exposure manifest | `<REQUIRED>` |
| Contributor attestation bundle | `<REQUIRED>` |
| AI-tool prompt/session manifest | `<REQUIRED>` |
| Prohibited-artifact scan | `<REQUIRED>` |
| Source-similarity review | `<REQUIRED>` |
| Clean-room custodian approval | `<REQUIRED>` |
| Legal release approval | `<REQUIRED for production>` |

Declaration:

`<REQUIRED: State exactly what was independently authored and trained. Do not
state that clean room eliminates patent, trademark, dataset, privacy, or
jurisdiction risk.>`

Prior exposure disclosures and dispositions, without private details:

`<REQUIRED or NONE>`

## Training data

Do not list a dataset until its licensing/privacy gate passes.

| Dataset ID/version | Role | Origin/owner | License or contract ID | Commercial ML | Privacy class | Records/windows | Source/transform/split digests |
|---|---|---|---|---|---|---:|---|
| `<REQUIRED>` | train/validation/calibration | `<REQUIRED>` | `<REQUIRED>` | `YES` required | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

### Excluded data

Explicitly state that TimesFM 3 weights, outputs, activations, generated labels,
and derivatives were excluded, then list all other exclusions:

`<REQUIRED>`

### Collection, consent, minimization, and retention

`<REQUIRED: collection authority, purpose, consent/contract, controller and
processor, sensitive inferences, pseudonymization, fields removed, geographic
scope, retention, deletion, access, and incident contact.>`

### Synthetic data

| Generator/version | Seed-data rights | Provider/output terms | Seeds/config digest | Proportion | Use |
|---|---|---|---|---:|---|
| `<REQUIRED or NONE>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

## Split and leakage protocol

| Split | Sites | Subjects | Sessions | Devices | Time blocks | Windows | Manifest digest |
|---|---:|---:|---:|---:|---:|---:|---|
| Train | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Validation | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Calibration | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Test | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

Required leakage checks:

- parent raw record and derived-window overlap;
- contiguous sequence/session overlap;
- subject/site/device/calibration leakage;
- target-horizon overlap;
- fitted preprocessing or threshold leakage;
- RuVector neighbour/index leakage;
- duplicate and near-duplicate leakage;
- test-informed architecture or hyperparameter changes.

Report and digest:

`<REQUIRED>`

## Training procedure

| Field | Value |
|---|---|
| Random initialization | `true` required for this model family |
| Parent checkpoint | `NONE` or approved RuView artifact and digest |
| Source commit | `<REQUIRED>` |
| Cargo.lock digest | `<REQUIRED>` |
| Rust toolchain | `<REQUIRED exact>` |
| Build/training container | `<REQUIRED digest>` |
| Training config digest | `<REQUIRED>` |
| Seeds | `<REQUIRED>` |
| Optimizer/schedule/loss | `<REQUIRED>` |
| Batch/epochs/stopping | `<REQUIRED>` |
| Hardware | `<REQUIRED exact CPU/GPU/RAM>` |
| Provider | `<REQUIRED local Linux or approved provider>` |
| Provider job ID | `<REQUIRED non-secret identifier>` |
| Wall time | `<REQUIRED and evidence label>` |
| Peak host/GPU memory | `<REQUIRED and evidence label>` |
| Estimated/actual cost | `<REQUIRED and evidence label>` |
| Emitted checkpoint digests | `<REQUIRED>` |
| Training receipt digest | `<REQUIRED>` |

If local and hosted runs are compared, list numerical-determinism differences
and prove that governed code, data, config, and parent identities match:

`<REQUIRED or NOT APPLICABLE>`

## Evaluation

### Baselines and ablations

Every row uses the same frozen examples and metric implementation.

| Model | Artifact/version | Retrieval | Parameters | Evidence label | Notes |
|---|---|---|---:|---|---|
| Last value | `<REQUIRED>` | no | n/a | `<REQUIRED>` | deterministic baseline |
| Seasonal naive | `<REQUIRED>` | no | n/a | `<REQUIRED>` | `<REQUIRED period>` |
| Small classical/recurrent baseline | `<REQUIRED>` | no | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| RuView Forecast | `<REQUIRED>` | no | `<REQUIRED>` | `<REQUIRED>` | required ablation |
| RuView Forecast | `<REQUIRED>` | yes | `<REQUIRED>` | `<REQUIRED>` | required if retrieval is enabled |

Do not use TimesFM 3 as an implementation oracle, label source, tuning signal,
or required acceptance baseline.

### Forecast metrics

Report per horizon and per site/device/interference regime, plus pooled values.

| Metric/domain/horizon | Result | 95% interval | Evidence label | Reproducer |
|---|---:|---:|---|---|
| Weighted quantile loss | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| MAE or scaled error | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Nominal 80% interval coverage | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Interval width | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Quantile crossing before/after policy | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Abstention coverage/selective risk | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

ADR-348 G3 targets are at least 10% weighted-quantile-loss improvement over
seasonal naive and 75%-85% measured coverage for a nominal 80% interval. These
remain `UNMEASURED targets` until populated by qualifying evidence.

### RuView shadow outcomes

| Metric | Baseline | Candidate | Delta/CI | Evidence label | Reproducer |
|---|---:|---:|---:|---|---|
| Empty-room false alerts | `<REQUIRED or UNMEASURED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Occupied-room recall | `<REQUIRED or UNMEASURED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Abstention by deployment | `<REQUIRED or UNMEASURED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Drift/OOD rate | `<REQUIRED or UNMEASURED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

ADR-348 G4 targets at least 50% relative empty-room false-alert reduction with
no more than 2 percentage points occupied-room recall loss over at least 14
shadow days. These are `UNMEASURED targets`, not current capabilities.

### Runtime measurements

| Platform | Build/features | Batch/streams | Context/horizon | p50/p95/p99 | Peak RSS | Evidence label | Reproducer |
|---|---|---:|---|---|---:|---|---|
| `<REQUIRED named CPU>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |
| `<optional GPU>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |

ADR-348 G5 CPU targets are at most 1 second p95 for 32 declared streams and at
most 4 GiB peak process memory. They are `UNMEASURED targets` until this table
contains a qualifying named-platform reproducer.

## Calibration, OOD, and abstention

| Control | Fit data | Frozen parameters/digest | Test result | Evidence label |
|---|---|---|---|---|
| Input normalization | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Quantile calibration | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| OOD threshold | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Minimum validity/context | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Drift threshold | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

List all typed abstention reasons and demonstrate that unknown schema, missing
calibration, stale data, insufficient history, invalid masks, non-finite input,
OOD state, and artifact failure do not produce an authoritative forecast:

`<REQUIRED>`

## Robustness and failure analysis

Report at minimum:

- unseen site and unseen device;
- empty room and low-motion occupancy;
- burst loss, cadence change, clock reset, and sensor restart;
- interference, channel change, and calibration drift;
- long missing runs and adversarial non-finite/range inputs;
- corrupted/truncated/oversized model and input artifacts;
- absent or stale RuVector index;
- concurrent-load resource caps and timeout;
- false confidence from narrow intervals during distribution shift.

| Scenario | Expected safe behaviour | Observed result | Evidence label | Open risk/owner |
|---|---|---|---|---|
| `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |

## Security and privacy

### Threat model summary

| Threat | Control | Evidence | Residual risk/owner |
|---|---|---|---|
| Malicious/corrupt model artifact | Signature, digest, schema and size verification; no embedded code | `<REQUIRED>` | `<REQUIRED>` |
| Oversized/malformed request | Dimension, allocation, horizon, context, batch and deadline caps | `<REQUIRED>` | `<REQUIRED>` |
| Poisoned data or split leakage | Approved manifests, immutable transforms, lineage and overlap checks | `<REQUIRED>` | `<REQUIRED>` |
| Cross-tenant analogue retrieval | Tenant and split-scoped RuVector indexes | `<REQUIRED>` | `<REQUIRED>` |
| Routine/location inference | Minimization, purpose, retention, deletion, access and audit | `<REQUIRED>` | `<REQUIRED>` |
| Model extraction/membership inference | Rate/access control and privacy evaluation | `<REQUIRED>` | `<REQUIRED>` |
| Hosted-worker data retention | Approved provider terms, least data, deletion receipt | `<REQUIRED>` | `<REQUIRED>` |
| LLM numeric mutation or overclaim | Signed numeric record and capability policy | `<REQUIRED or NOT INTEGRATED>` | `<REQUIRED>` |

### Data handling

`<REQUIRED: tenant isolation, encryption, keys, access roles, logs, metrics
allowlist, retention, deletion, export, incident handling, and whether feature
or forecast persistence is enabled.>`

## Deployment

| Field | Value |
|---|---|
| Supported mode | `<REQUIRED>` |
| Supported platform/CPU/GPU | `<REQUIRED>` |
| Minimum RAM/storage | `<REQUIRED and evidence-labelled>` |
| Offline artifact source | `<REQUIRED>` |
| Signature/trust root | `<REQUIRED public key ID>` |
| Configuration/calibration artifact | `<REQUIRED digest>` |
| RuVector index requirement | `<REQUIRED>` |
| Startup self-test | `<REQUIRED>` |
| Health/drift metrics | `<REQUIRED allowlisted fields>` |
| Timeout/backpressure policy | `<REQUIRED>` |
| Rollback artifact/mode | `<REQUIRED>` |

No deployment may fetch an unpinned model at runtime. Describe exact startup
failure and fail-closed behaviour:

`<REQUIRED>`

## Monitoring, rollback, and retirement

| Signal | Threshold | Window | Response | Owner |
|---|---:|---|---|---|
| Calibration/coverage drift | `<REQUIRED>` | `<REQUIRED>` | shadow/off | `<REQUIRED>` |
| OOD/abstention rate | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Latency/resource regression | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Empty/occupied outcome regression | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Provenance/signature failure | any | immediate | disable artifact | `<REQUIRED>` |
| Security/privacy incident | any material incident | immediate | isolate, preserve evidence, disable | `<REQUIRED>` |

Rollback drill receipt and result:

`<REQUIRED>`

Retirement and data/index/checkpoint deletion policy:

`<REQUIRED>`

## Limitations and largest uncertainty

`<REQUIRED: List observed and unmeasured limitations. The default largest
uncertainty is whether training diversity supports unseen-room and unseen-device
generalization without suppressing legitimate occupied states. State the
specific data/evaluation fix path.>`

## Cost, energy, and operational burden

| Item | Value | Evidence label | Method |
|---|---:|---|---|
| Training accelerator hours | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |
| Training cost | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |
| Training energy/emissions | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |
| CPU inference cost/energy | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |
| Storage/index overhead | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |
| Operator/calibration burden | `<REQUIRED or UNMEASURED>` | `<REQUIRED>` | `<REQUIRED>` |

## Licenses and notices

| Surface | Exact license/terms | URL/file | Obligations | Approval |
|---|---|---|---|---|
| Rust source | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Weights | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Each dataset | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Dependencies/SBOM | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Training provider | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |

Trademark clearance ID and scope:

`<REQUIRED for public production release>`

Patent freedom-to-operate disposition and launch jurisdictions:

`<REQUIRED for production; keep privileged analysis outside this public card>`

## Provenance and reproducibility

| Artifact | Digest/signature/location |
|---|---|
| Source allowlist | `<REQUIRED>` |
| Specification | `<REQUIRED>` |
| Source commit | `<REQUIRED>` |
| Cargo.lock | `<REQUIRED>` |
| SBOM | `<REQUIRED>` |
| Dataset manifest | `<REQUIRED>` |
| Split/leakage report | `<REQUIRED>` |
| Training config | `<REQUIRED>` |
| Training receipt | `<REQUIRED>` |
| Evaluation report | `<REQUIRED>` |
| Clean-room report | `<REQUIRED>` |
| Model weights | `<REQUIRED>` |
| RVF signature | `<REQUIRED or NONE with blocker>` |
| Reproducer | `<REQUIRED>` |

Reproduction commands must use pinned artifacts and must not contain secrets or
download restricted material:

```bash
<REQUIRED safe, exact commands>
```

## ADR-348 gate traceability

| Gate | Requirements | Evidence in this card/bundle | Result |
|---|---|---|---|
| G0 independent-authoring boundary | RF-001, RF-002 | allowlist, exposure, attestation, scans, similarity review | `<OPEN, PASS, or FAIL>` |
| G1 data and lineage | RF-002, RF-003, RF-010 | dataset approvals, lineage, local/hosted receipts | `<OPEN, PASS, or FAIL>` |
| G2 bounded Rust contract | RF-004, RF-005, RF-006 | contract/fuzz/replay/resource/dependency reports | `<OPEN, PASS, or FAIL>` |
| G3 leakage-free model evidence | RF-007, RF-008, RF-011 | frozen split, baselines, ablations, calibration, reproducers | `<OPEN, PASS, or FAIL>` |
| G4 RuView shadow value | RF-006, RF-007, RF-008, RF-009 | 14-day outcomes, drift, abstention, no action authority | `<OPEN, PASS, or FAIL>` |
| G5 deployment fitness | RF-004, RF-009, RF-010, RF-011, RF-012 | runtime, approvals, signed bundle, rollback | `<OPEN, PASS, or FAIL>` |

Highest permitted deployment mode from these results:

`<REQUIRED>`

## Approvals

Signatures cover this exact model-card digest and weight digest.

| Role | Approver/receipt | Decision | Timestamp |
|---|---|---|---|
| Model owner | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Clean-room custodian | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Data steward/privacy | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Security owner | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Runtime owner | `<REQUIRED>` | `<REQUIRED>` | `<REQUIRED>` |
| Legal release owner | `<REQUIRED for production>` | `<REQUIRED>` | `<REQUIRED>` |

## Citation

Use the final public model name, exact version, weight digest, repository commit,
and model-card URL:

```bibtex
@software{<REQUIRED citation key>,
  title        = {<REQUIRED>},
  author       = {<REQUIRED>},
  year         = {<REQUIRED>},
  version      = {<REQUIRED>},
  url          = {<REQUIRED>},
  note         = {Weights SHA-256: <REQUIRED>; source commit: <REQUIRED>}
}
```

## Change history

| Card version | Weight digest | Change | Gate impact | Date |
|---|---|---|---|---|
| `<REQUIRED>` | `<REQUIRED>` | Initial candidate | All gates evaluated independently | `<REQUIRED>` |
