# ADR-350: RuVector predictive memory and RuVLLM authority boundary

- **Status**: Proposed
- **Date**: 2026-09-01
- **Deciders**: ruv
- **Owners**: RuView forecast, RuVector, evidence, and RuVLLM integration maintainers
- **Tags**: forecasting, ruvector, ruvllm, retrieval, memory, receipts, authority
- **Parent**: ADR-348
- **Extends**: ADR-004, ADR-010, ADR-016, ADR-145, ADR-261, ADR-295,
  ADR-304, ADR-319, ADR-348
- **Supersedes**: None

## Decision

RuVector may augment RuView Forecast by retrieving analogous historical
feature/forecast states from a tenant- and split-scoped predictive-memory
index. Retrieval is optional context, never ground truth. Every release must
report the same frozen examples with retrieval disabled and enabled so its
incremental value and leakage risk remain visible.

Each forecast is published in an immutable, locally signed envelope binding the
model artifact, request, output, source evidence, and retrieval receipt. RuVLLM
may read that envelope to explain uncertainty and analogous history. It may not
rewrite numeric forecasts, create a stronger evidence class, select a model,
promote an artifact, spend money, invoke an actuator, or make medical,
emergency, access-control, or life-safety decisions.

This child ADR delegates forecasting, clean-room, data, and general production
gates to [ADR-348](./ADR-348-independent-rust-multivariate-forecasting.md). It
does not authorize a sensing-server bridge.

**Evidence status:** canonical forecast/receipt types and an indexable latent
state exist in source. A RuVector adapter, retrieval receipt, signature wrapper,
outcome reconciler, RuVLLM explanation adapter, paired retrieval evaluation,
latency, storage cost, accuracy lift, and operational value are all
**UNMEASURED and unapproved**.

## Existing implementation boundary

The current crates establish only the preconditions:

| Module | Implemented surface used by this ADR |
|---|---|
| [`ruview-forecast-core/src/forecast.rs`](../../v2/crates/ruview-forecast-core/src/forecast.rs) | validated `ForecastRequest`, backend-neutral `Forecaster`, ordered finite `Forecast`, canonical output digest, and receipt verification |
| [`ruview-forecast-core/src/receipt.rs`](../../v2/crates/ruview-forecast-core/src/receipt.rs) | `SourceState`, `ArtifactReceipt`, and `ForecastReceipt`; derived forecasts cannot retain `MEASURED` merely because measured input existed |
| [`ruview-forecast-core/src/series.rs`](../../v2/crates/ruview-forecast-core/src/series.rs) | bounded feature schema, masked time series, canonical series digest, and training-only scaler fit |
| [`ruview-forecast-model/src/network.rs`](../../v2/crates/ruview-forecast-model/src/network.rs) | `ForecastModelOutput.state` with shape `[batch, variates, d_model]`, intended as a bounded representation candidate for indexing |

`ForecastReceipt` is content-addressed but not itself a digital signature.
`ForecastModelOutput.state` is indexable but not automatically safe, private,
stable across model versions, or useful. Those distinctions remain release
gates.

## Predictive-memory record

RuVector stores a versioned `PredictiveMemoryRecord` containing:

- tenant and index namespace;
- model/artifact/config/feature-schema digests;
- source series and forecast-request digests;
- split, site, subject-class, session, device, calibration, and bounded time
  scope using pseudonymous identifiers;
- bounded latent-state or engineered-feature digest plus the approved vector;
- forecast envelope digest and horizon;
- observed validity mask and evidence class;
- optional reconciled outcome digest added only after the forecast horizon;
- retention/deletion class and creation/expiry time.

Raw CSI, unrestricted feature windows, precise room coordinates, persistent
person identity, RuVLLM prompts, and explanation prose are excluded by default.
The vector is still potentially sensitive because it can encode routines or
location. Tenant isolation, encryption, access control, retention, deletion,
and membership-inference review apply.

An outcome is appended through a new immutable record linked to the original;
the historical forecast is never rewritten. Outcome reconciliation can measure
forecast quality but cannot retroactively turn the forecast into a measured
observation.

## Split-scoped analogue retrieval

Every index is bound to an immutable corpus manifest, model version, feature
schema, preprocessing configuration, and split policy. These rules are
mandatory:

1. Tenants never share an index or query result without a separately authorized
   privacy-preserving federation protocol.
2. Training, validation, calibration, and test records have distinct
   namespaces. A test query may retrieve approved training analogues only; it
   may not retrieve test examples or validation/calibration records used to set
   thresholds.
3. The same site, subject, session, device, calibration episode, overlapping
   context, or overlapping target horizon is excluded when that dimension is a
   holdout.
4. Index construction, distance metric, normalization, filter policy, `k`, and
   score threshold are fitted on training data and frozen before test.
5. Missing, stale, mismatched, unauthorized, or unverifiable indexes disable
   retrieval and return the no-retrieval forecast or abstention according to
   the model card. They never trigger an implicit global-index fallback.
6. Every query returns a bounded `RetrievalReceipt` even when zero neighbours
   qualify.

The retrieval receipt binds query digest, index/corpus/policy digests,
namespace, exclusion filter, neighbour record IDs and distances, `k`, latency,
and disposition. It excludes raw neighbour payloads from ordinary logs.

## Paired ablation

Retrieval has no deployment authority without a paired evaluation on identical
frozen requests:

| Row | Model and examples | Retrieval |
|---|---|---|
| A | exact candidate artifact and frozen examples | disabled |
| B | exact candidate artifact and frozen examples | enabled with frozen index/policy |

The report includes weighted quantile loss, interval coverage/width,
abstention, per-horizon error, per-site/device/interference slices, retrieval
hit/filtered/empty rates, latency, and memory/storage overhead. It reports both
aggregate and paired deltas with uncertainty. A gain on pooled error cannot
hide a failed deployment domain, calibration regression, or neighbour leakage.

The no-retrieval row remains a supported fallback. Retrieval is removed when
its lower confidence bound does not show useful improvement, when it weakens
calibration beyond the ADR-348 gate, or when its privacy/latency/storage cost
exceeds its measured value. No uplift is claimed today.

## Immutable signed forecast envelope

The signed envelope covers:

```text
envelope schema/version
+ ArtifactReceipt canonical digest
+ ForecastRequest canonical digest
+ Forecast payload/output digest
+ ForecastReceipt canonical digest
+ RetrievalReceipt digest or explicit retrieval-disabled marker
+ policy/calibration/index digests
+ creation/expiry time and tenant namespace
```

Signing occurs at the trusted local boundary after artifact and retrieval
verification. The wrapper records algorithm, public key ID, signer capability,
and signature. Private keys never enter model artifacts, RuVector, RuVLLM, or a
hosted training worker.

Consumers verify signature, expiry, tenant, schemas, every nested digest, and
the `Forecast::verify_receipt` invariant before using numeric values. Any
failure yields unavailable/abstain. Unsigned content-addressed receipts remain
useful for local tests but are not described as signed and cannot cross the
production trust boundary.

## RuVLLM explanation boundary

RuVLLM receives a read-only projection of the verified envelope:

- exact point and quantile values as structured fields;
- units, horizons, calibration/OOD/abstention state;
- bounded analogue summaries and retrieval receipt references;
- evidence labels, model/version, and envelope digest;
- approved explanation policy and audience.

Explanation prose is stored separately and linked to the envelope digest. It
is `CLAIMED_EXPLANATION`, not a replacement forecast. Numeric API fields are
copied from the verified envelope after generation, never parsed back from LLM
text. If prose contradicts a structured number, evidence state, unit, or
disposition, the response fails validation and the numeric envelope remains
authoritative.

The explanation capability exposes no tools for artifact activation, model
promotion, retraining, spending, messaging, emergency dispatch, access control,
or actuation. A separate downstream policy may consume a forecast only under
its own ADR, capabilities, approvals, and observed-evidence requirements. The
LLM cannot grant itself that authority or lower an approval threshold.

## Requirements and acceptance

| ID | Requirement | ADR-348 gate | Acceptance evidence | Current state |
|---|---|---|---|---|
| PM-001 | Every predictive-memory query is tenant-, model-, schema-, corpus-, split-, and time-scope bound. | G3, G5 | Cross-tenant/split/version negative tests and signed index manifest | **OPEN / UNMEASURED** |
| PM-002 | Holdout identities, overlapping contexts, and target horizons cannot appear as neighbours. | G3 | Property tests plus frozen leakage report | **OPEN / UNMEASURED** |
| PM-003 | Every query, including disabled/empty/error, produces a bounded retrieval receipt. | G2, G3 | Receipt round-trip, tamper, bound, and zero-result tests | **OPEN / UNMEASURED** |
| PM-004 | Retrieval-disabled and retrieval-enabled rows use identical artifact/examples and report paired metrics and overhead. | G3 | Frozen paired ablation with reproducer | **OPEN / UNMEASURED** |
| PM-005 | Forecast envelope signatures bind all nested forecast/retrieval identities and fail closed on mutation, expiry, tenant, or key error. | G2, G5 | Signature/tamper/replay/cross-tenant tests and local signing receipt | **OPEN / UNMEASURED** |
| PM-006 | Derived forecast and explanation can never become `MEASURED_OBSERVATION` or exceed source evidence. | G2, G4 | Evidence-monotonicity schema/property tests | **OPEN / UNMEASURED** |
| PM-007 | RuVLLM cannot mutate structured numbers; contradictions fail validation and explanations remain separately labelled. | G4 | Numeric/unit/evidence mutation corpus and fail-closed integration tests | **OPEN / UNMEASURED** |
| PM-008 | RuVLLM explanation has no spending, promotion, messaging, safety, access-control, or actuator capability. | G4, G5 | Default-deny capability and attempted-escalation tests | **OPEN / UNMEASURED** |
| PM-009 | Vector/explanation privacy, retention, deletion, extraction, and membership risks are approved and operationally tested. | G5 | Privacy review, tenant deletion drill, and access audit | **OPEN / UNMEASURED** |

G3 additionally requires the paired retrieval ablation to satisfy ADR-348's
forecast and calibration gates. G4 requires shadow evidence that retrieval and
explanation improve operator comprehension or forecast value without changing
numeric/action authority. G5 requires signed-envelope verification, rollback,
tenant deletion, and key-rotation drills. None has passed.

## Consequences

RuVector can evolve from retrospective search into outcome-linked predictive
memory while preserving a measurable no-retrieval baseline. RuVLLM can make
forecasts understandable without becoming the numeric or action authority. The
cost is additional index isolation, storage, signature, deletion, evaluation,
and policy complexity. If paired evidence does not justify that cost, the
correct release keeps retrieval and explanation disabled.

## References

- [ADR-348](./ADR-348-independent-rust-multivariate-forecasting.md)
- [ADR-349](./ADR-349-governed-local-and-fal-forecast-training.md)
- [RuView Forecast clean-room protocol](../security/ruview-forecast-clean-room.md)
- [RuView Forecast model-card template](../huggingface/RUVIEW_FORECAST_MODEL_CARD_TEMPLATE.md)
- [RuForecast benchmark protocol](../benchmarks/ruforecast.md)
- [ADR-016](./ADR-016-ruvector-integration.md)
- [ADR-145](./ADR-145-ablation-eval-harness-privacy-leakage.md)
- [ADR-261](./ADR-261-ruvector-graph-ann-index.md)
- [ADR-304](./ADR-304-evidence-engine.md)
- [ADR-319](./ADR-319-witness-chain.md)
