# RuView Forecast clean-room, data, and release protocol

**Status:** Proposed and mandatory for any artifact claiming independent
development under ADR-348.

**Evidence status:** This document defines controls. It does not assert that the
controls have passed, that a trained model exists, or that any forecasting
capability is measured. Every ADR-348 gate remains **OPEN / UNMEASURED** until a
signed evidence bundle proves otherwise.

## 1. Purpose and authority

This protocol governs the specification, implementation, training, evaluation,
hosting, and release of the independent Rust multivariate forecaster defined by
[ADR-348](../adr/ADR-348-independent-rust-multivariate-forecasting.md).
It applies to source, issue and review text, prompts, dependencies, datasets,
intermediate tensors, checkpoints, containers, CI caches, RuVector indexes,
fal.ai jobs, benchmark results, model cards, and published artifacts.

The protocol is intentionally stricter than the minimum conditions for using
Apache-licensed source. Its purpose is to preserve evidence that the RuView
implementation and weights were independently created. It is not a legal
opinion. The legal release owner must approve the current licenses, contributor
exposure record, patent review, trademark review, datasets, and intended launch
jurisdictions before production.

## 2. License boundary as of 2026-09-01

The authoritative public materials state two different license surfaces:

1. The [Google TimesFM repository](https://github.com/google-research/timesfm)
   identifies its source code as Apache-2.0.
2. The [TimesFM 3 checkpoint license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
   permits non-commercial, non-production use and restricts commercial use,
   distribution, and use to train, fine-tune, or distill another commercial
   model.

The checkpoint license defines a derivative broadly enough that relying on its
logic, parameters, or model-generated material creates avoidable contractual
risk. No contributor may accept gated model terms on behalf of the project or
its owner without written authorization.

The [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) permits
source reproduction and derivative works subject to its conditions, including
license delivery, change notices, retention of applicable notices, and NOTICE
handling. Apache also provides a limited contributor patent grant and no general
trademark grant. If maintainers ever choose an Apache-derived port, it must use
a separate branch, attribution path, artifact name, and superseding ADR. It
must not be described as the clean-room implementation.

Publicly described ideas, procedures, processes, systems, and methods are not
the same as copied expression. United States copyright law states the
idea-expression boundary in [17 U.S.C. section 102(b)](https://www.copyright.gov/title17/92chap1.html).
That does not settle contract, patent, trademark, database-right, trade-secret,
or non-U.S. law questions.

## 3. Non-mixing rule

Only one of these paths may govern a source tree:

| Path | Source basis | Required representation |
|---|---|---|
| Independent implementation | Approved abstract specification, public papers/articles, general literature, RuView requirements, approved data | "Independently developed and independently trained" after all gates pass |
| Apache-derived port | Apache-licensed Google source with full license and notice compliance | "Apache-derived Rust port" with attribution |

There is no hybrid path. A permissive source license makes porting possible; it
does not make a port independent. If exposure cannot be bounded, maintainers
must either quarantine and rewrite the affected component or reclassify the
whole affected component under the Apache-derived path.

## 4. Roles and separation

| Role | May access | Must not access or do |
|---|---|---|
| Clean-room custodian | Exposure records, allowlist, scan reports, quarantined evidence | Contribute core model implementation after reviewing prohibited source fragments |
| Specification reviewer | Frozen public-paper/article allowlist, RuView requirements, generic literature | Google TimesFM implementation, tests, configs, checkpoint files, model outputs; contribute implementation |
| Rust implementer | Approved independent specification, RuView schemas, approved generic dependencies | TimesFM source, tests, configs, weights, outputs, line-by-line/API translation, reference differential testing |
| Data steward | Dataset source, contracts, consent, privacy and transformation records | Approve unknown or incompatible rights; infer that a software license covers data |
| Training operator | Signed code/container/config and approved dataset manifests | Change code/data/config outside a new receipt; add an unapproved parent checkpoint |
| Independent evaluator | Frozen candidate, frozen evaluation manifests, approved baselines | Return reference predictions or implementation hints to developers; tune against the test set |
| Legal release owner | Complete evidence bundle and private legal analysis | Treat clean-room checks as patent clearance or dataset approval by implication |

One person may hold multiple roles only where the forbidden-access rules remain
true. The strongest separation uses different people for public specification
and implementation. A specification reviewer who has inspected prohibited
source cannot become a core implementer for the independent path.

## 5. Source classification

### 5.1 Allowed

- The public TimesFM research articles and peer-reviewed papers, frozen by URL,
  retrieval date, and SHA-256 digest.
- General time-series, transformer, probabilistic forecasting, normalization,
  missing-data, and calibration literature.
- RuView requirements, independently written feature schemas, and existing
  RuView contracts whose provenance already satisfies repository policy.
- Generic Rust dependencies whose license, source, and purpose are recorded and
  approved.
- Public benchmark specifications and datasets after the dataset gate passes.
- Published aggregate competitor results used only as contextual background.

### 5.2 Conditional

- Code implementing generic algorithms may be used only after license review
  and must be declared as a dependency or attributed source. It cannot be used
  to claim that every line was independently authored.
- TimesFM versions up to 2.5 may have permissively licensed code or weights, but
  they remain outside the independent implementation and training path. An
  evaluator may use an approved permissive baseline only after the candidate
  is frozen, and its output cannot become training data or implementation
  feedback.
- A contributor with prior exposure to TimesFM code must disclose the exact
  material and date. The custodian and legal owner decide whether the person is
  limited to non-implementation work or the affected component must be
  reclassified.
- Synthetic data is allowed only when source data, generator code/model,
  provider terms, output rights, and intended commercial use all pass review.

### 5.3 Prohibited

- TimesFM 3 weights, parameters, checkpoints, tensor dumps, checkpoint
  configuration, hidden activations, embeddings, gradients, or derived
  fingerprints.
- TimesFM 3 forecasts, quantiles, scores, probabilities, recommendations, or
  synthetic labels in development, training, tuning, or release evidence.
- Fine-tuning, distillation, imitation, teacher-student learning, differential
  testing, behavioural cloning, or hyperparameter search against TimesFM 3.
- Manual, mechanical, or AI-assisted translation of Google source, tests,
  comments, constants, APIs, file layout, or diagrams.
- Third-party code, datasets, or checkpoints whose origin may contain copied or
  distilled TimesFM material without documented permission.
- Model, dataset, or service terms labelled non-commercial, research-only,
  evaluation-only, no-derivatives, no-ML, no-production, or unknown.
- Confidential third-party information or material obtained under an agreement
  that does not authorize this implementation and commercial use.

## 6. Exposure intake and contributor attestation

Before receiving implementation access, every contributor completes an
exposure intake covering:

- TimesFM repositories, model cards, gated checkpoint files, local caches,
  notebooks, issues, tutorials, and generated outputs viewed or downloaded;
- whether gated terms were accepted and for which individual or entity;
- source or output material placed in an LLM, IDE assistant, retrieval index,
  prompt, or code-generation session;
- prior work for Google, a competitor, customer, or other party involving
  confidential forecasting implementation;
- datasets, pretrained models, and external code expected in the contribution.

Each pull request also carries a signed attestation:

```text
RuView Forecast clean-room attestation v1

Contributor:
Employer or represented entity:
Role:
Covered commits:
Prior TimesFM exposure: NONE or complete disclosure
Approved sources used:
Datasets introduced:
AI tools used:
Prompt/session manifest digest:

I attest that the covered contribution was created from the approved
specification and listed sources. I did not access, copy, translate, decompile,
query, distill, or use TimesFM 3 implementation material, configuration,
weights, parameters, outputs, or internal representations. I did not place
prohibited material in an AI tool context. I disclosed all contrary facts
above. Every dependency and dataset I introduced has documented rights for its
stated use.

Signature:
Timestamp:
```

A DCO `Signed-off-by` line does not replace this attestation. The attestation
digest, not private identity material, is referenced from the public release
manifest.

## 7. Specification controls

The custodian freezes an allowlist before implementation begins. Every entry
records title, canonical URL or DOI, publication date, retrieval time, content
digest, license or access terms, reviewer, and the abstract requirement it
supports.

The specification may state mathematical functions, tensor roles, input/output
invariants, resource bounds, and RuView-specific requirements. It must not
include copied source, comments, tests, distinctive identifiers, constants,
checkpoint dimensions extracted from files, API compatibility requirements, or
Google diagrams. RuView API names must be derived from the local domain.

Changes to the specification after test evaluation receive a new version and
must not encode held-out answers. A material architecture revision consumes a
new untouched holdout or remains labelled exploratory.

## 8. AI coding-tool controls

An AI assistant can unintentionally defeat source separation by retrieving or
reproducing reference code. For sessions contributing implementation:

1. Disable web retrieval and repository indexing outside the approved local
   tree where the tool supports it.
2. If a user describes the goal as a port, clone, reproduction, or emulation,
   stop and restate the task as independently authored RuView functional
   requirements. Record that wording in the exposure manifest; do not use it
   as permission to retrieve reference implementation material or pursue
   behavioral equivalence.
3. Do not attach Google code, checkpoint metadata, outputs, screenshots, or
   detailed third-party implementation summaries.
4. State the independent specification and prohibited-source boundary in the
   session instructions.
5. Retain a bounded prompt and tool-source manifest digest without committing
   raw private transcripts.
6. Record the model/tool version and whether retrieval was enabled.
7. Treat unexplained code that resembles a prohibited implementation as an
   incident, not as a harmless generated suggestion.

AI-generated code receives the same authorship, license, security, and
similarity review as human-authored code.

## 9. Dataset licensing and privacy gate

No dataset enters preprocessing, a RuVector index, training, calibration, or
evaluation until a data steward records affirmative answers to every applicable
gate:

| Gate | Required evidence | Automatic rejection examples |
|---|---|---|
| Origin | Named owner/provider, acquisition method, immutable source digest | Scrape or file of unknown origin |
| Training rights | Written right to use for ML training and the intended commercial purpose | Research-only, NC, no-ML, evaluation-only |
| Derivatives | Right to transform, derive windows/features, and create model artifacts | No-derivatives or ambiguous custom terms |
| Redistribution | Whether raw, transformed, manifests, and weights may be redistributed separately | Assumption that public access means redistribution |
| Attribution | Exact attribution and notice obligations | Missing author/source/version |
| Database rights | Jurisdiction and database-right review where applicable | Unreviewed EU database extraction |
| Consent and contract | Collection authority, participant/customer consent, purpose, controller/processor roles | Customer telemetry without explicit model-training authority |
| Sensitive inference | Classification of occupancy, routines, location, health-adjacent and biometric implications | Unbounded identity or health inference |
| Minimization | Required fields only, pseudonymization, retention and deletion schedule | Raw CSI/person identifiers retained without need |
| Split integrity | Site/subject/session/device/time-block lineage and overlap report | Contiguous or derived-window leakage across splits |
| Synthetic lineage | Generator, seed data, service terms, output rights, seeds/config digest | Restricted teacher or prohibited source data |

Typical disposition guidance:

- Owned data, CC0/public-domain data, and data under an explicit commercial ML
  contract may pass after privacy and provenance review.
- CC BY data requires attribution plus database and downstream-weight review.
- MIT and Apache are software licenses and do not automatically license nearby
  data.
- NC, ND, research-only, evaluation-only, no-ML, no-production, and unknown
  terms fail unless a separate written commercial grant is obtained.

Every transform is content-addressed. Normalizers, feature selectors,
calibrators, thresholds, and RuVector indexes fit training data only. Raw
sequence overlap and derived-window overlap are both checked. A parent record
in one split makes all overlapping descendants ineligible for another split.

## 10. Hosted training boundary

The Linux machine and fal.ai are execution environments, not sources of
authority. A hosted job is accepted only when:

- the exact source commit, Cargo lockfile, compiler, container digest, data
  manifests, configuration, seeds, and parent checkpoint digest are signed
  before upload;
- provider terms, retention, region, subprocessors, logs, cache deletion,
  confidentiality, output ownership, and provider-training reuse have been
  reviewed for the data classification;
- credentials are short-lived, least-privilege, excluded from images and logs,
  and rotated after suspected exposure;
- customer-derived data is not uploaded before controller/processor and
  transfer requirements pass;
- emitted checkpoints and logs are hashed immediately and compared with the job
  receipt;
- the downloaded artifact is scanned before entering the trusted release
  boundary;
- a hosted provider cannot promote, sign, or activate a model.

Hosted and local runs are comparable only when their governed input identities
match. Numerical nondeterminism must be documented; it does not permit an
unrecorded dependency, data, or configuration change.

## 11. Provenance manifest

The release bundle contains a machine-readable manifest with at least:

```yaml
schema_version: <required>
artifact:
  id: <required>
  version: <required>
  git_commit: <sha>
  source_spec_digest: <sha256>
  cargo_lock_digest: <sha256>
  rust_toolchain: <exact>
  target_triple: <exact>
  build_container_digest: <digest>
  sbom_digest: <sha256>

contributors:
  - pseudonymous_id: <stable id>
    role: <role>
    exposure_class: <unexposed|disclosed-reviewed>
    attestation_digest: <sha256>
    signed_at: <rfc3339>

references:
  - title: <title>
    canonical_uri: <url-or-doi>
    publication_date: <date>
    retrieved_at: <rfc3339>
    content_digest: <sha256>
    license_or_terms: <identifier>
    allowed_use: <purpose>
    reviewer: <id>

datasets:
  - dataset_id: <id>
    version: <version>
    owner: <owner>
    origin: <uri-or-contract-id>
    license_or_contract: <identifier>
    commercial_ml_allowed: <true>
    redistribution_allowed: <true|false>
    consent_basis: <identifier>
    privacy_class: <class>
    jurisdictions: [<jurisdiction>]
    retention_policy: <policy-id>
    source_digest: <sha256-or-merkle-root>
    transform_digest: <sha256>
    split_digest: <sha256>
    record_count: <count>

training:
  random_initialization: true
  parent_checkpoint: null
  code_commit: <sha>
  config_digest: <sha256>
  rng_seeds: [<seed>]
  hardware: <inventory>
  provider: <local-linux|approved-provider>
  provider_job_id: <non-secret-id>
  environment_digest: <sha256>
  started_at: <rfc3339>
  ended_at: <rfc3339>
  forbidden_artifact_scan_digest: <sha256>
  leakage_scan_digest: <sha256>

evaluation:
  frozen_manifest_digest: <sha256>
  baseline_artifacts: [<id-and-digest>]
  metric_schema: <version>
  report_digest: <sha256>
  calibration_report_digest: <sha256>
  domain_holdouts: [<site|subject|session|device>]

release:
  weights_digest: <sha256>
  model_card_digest: <sha256>
  clean_room_report_digest: <sha256>
  signature_key_id: <public-key-id>
  rvf_signature: <signature>
  source_license: <spdx-expression>
  weights_license: <exact-identifier>
  data_steward_approval: <signed-receipt>
  security_approval: <signed-receipt>
  legal_approval: <signed-receipt>
```

Secrets, raw personal identifiers, contract text, raw customer data, and
private contributor identities stay outside the public manifest. The manifest
references controlled records by digest or approval ID.

## 12. Source, dependency, and artifact checks

The custodian records tool versions, rules, timestamps, and complete findings.
At minimum:

- search the working tree and Git history for prohibited model IDs, URLs,
  filenames, license strings, binary signatures, and unexpected large files;
- scan source similarity against the prohibited implementation in an isolated
  custodian environment; implementation contributors do not receive reference
  fragments from the report;
- manually adjudicate every material similarity match and record independent
  origin, generic necessity, rewrite, or reclassification;
- run dependency license, source, duplicate-version, and vulnerability policy
  checks and generate a CycloneDX or SPDX SBOM;
- inspect container layers, CI and provider caches, mounted volumes, model/data
  buckets, notebooks, logs, and local tool indexes for restricted artifacts;
- verify that every checkpoint parent is an approved RuView artifact or the
  declared random initialization;
- verify that model loading performs no network access and accepts no embedded
  executable operator or arbitrary path;
- verify that public packages, endpoints, docs, and metadata do not use Google
  trademarks as a product identity.

Keyword scans are a tripwire, not proof of independence. References in this
governance document are expected and must be path allowlisted. The final result
depends on provenance, exposure records, review, and the absence of prohibited
material in implementation/training paths.

## 13. Model card and claim gate

Every candidate copies and completes
[`../huggingface/RUVIEW_FORECAST_MODEL_CARD_TEMPLATE.md`](../huggingface/RUVIEW_FORECAST_MODEL_CARD_TEMPLATE.md).
No placeholder, `UNMEASURED`, unknown license, missing digest, or missing
approval may be silently deleted. It must instead be resolved or retained as an
explicit release blocker.

All accuracy, latency, memory, power, cost, and generalization values are
labelled `MEASURED`, `SYNTHETIC`, `CLAIMED`, or `UNMEASURED` under repository
policy. `MEASURED` requires a reproducer, immutable inputs, exact artifact, and
named hardware/environment. A model-card benchmark does not create authority
for a medical, emergency, security, or autonomous-action claim.

## 14. Naming, trademark, and comparative statements

Approved project identities include `RuView Forecast` and crate names derived
from the RuView domain. Do not use `TimesFM`, `Google`, or a confusingly similar
mark in crate names, binaries, model slugs, endpoints, logos, icons, or product
headlines. Do not copy diagrams or visual branding.

Comparative documentation may accurately identify an external model and its
version when necessary, with a statement that RuView Forecast is independently
developed and not affiliated with or endorsed by Google. Comparative claims
need identical datasets, metric definitions, permitted model use, and evidence
labels.

Before public naming or commercial release, complete a professional trademark
clearance that includes federal, state, common-law, domain, and relevant
international sources. The [USPTO clearance guidance](https://www.uspto.gov/trademarks/search/comprehensive-clearance-search-similar-trademarks)
is a starting point, not a complete legal opinion.

## 15. Residual patent and jurisdiction risk

Clean-room evidence does not prevent patent infringement. Before production,
patent counsel performs a claim-level freedom-to-operate review covering at
least temporal patching, multivariate/variate attention or mixing, masked
horizon prediction, known-future covariates, probabilistic heads, normalization,
retrieval augmentation, and relevant training procedures. Searches include
assignees, inventors, continuations, unpublished timing uncertainty, and launch
jurisdictions. The [USPTO Patent Public Search](https://www.uspto.gov/patents/search/patent-public-search)
supports preliminary searching but is not a freedom-to-operate opinion.

The Apache patent grant applies only within its stated scope. An independent
implementation must not assume it inherits that grant. Public papers may be
prior art yet still coexist with earlier, pending, territorial, or narrower
claims.

## 16. Contamination incident response

Treat any prohibited source, output, model, data, or unreviewed exposure as a
provenance incident:

1. Stop affected implementation, training, evaluation feedback, and release.
2. Preserve hashes, timestamps, actors, locations, access records, and affected
   lineage. Do not erase the audit trail.
3. Quarantine the material and revoke it from build, data, model, cache, and
   retrieval paths.
4. Identify every affected commit, specification revision, dataset transform,
   RuVector index, checkpoint, benchmark, and descendant artifact.
5. Rotate credentials if a provider, cache, or secret may be exposed.
6. Have the custodian and legal owner choose one disposition: proven
   non-impact, clean rewrite by unexposed contributors, full retraining from the
   last clean ancestor, or explicit Apache-derived reclassification.
7. Repeat all affected ADR-348 gates and record the incident and closure
   receipts.

If prohibited output becomes a label or tuning signal, all descendant weights
are affected. Code deletion alone cannot repair model lineage; retraining from
an approved clean initialization is required.

## 17. Requirement-mapped acceptance checklist

| Check | ADR-348 requirements | Gate | Acceptance condition |
|---|---|---|---|
| CR-01 source allowlist | RF-001, RF-002 | G0 | 100% of specification sources frozen, hashed, licensed/termed, and approved |
| CR-02 contributor exposure | RF-001, RF-002 | G0 | 100% of contributors have current intake and signed attestation; every disclosure has a disposition |
| CR-03 prohibited-artifact scan | RF-001 | G0 | Zero unresolved prohibited artifacts or outputs across source, history, caches, containers, jobs, data, indexes, and checkpoints |
| CR-04 similarity review | RF-001, RF-002 | G0 | Every material match adjudicated without exposing implementers to reference fragments |
| CR-05 dependency/SBOM | RF-002, RF-004 | G1, G2 | Zero unknown or denied dependency licenses; vulnerability policy passes; signed SBOM exists |
| CR-06 dataset rights | RF-003 | G1 | 100% of bytes map to approved manifests; zero NC, ND, research-only, no-ML, no-production, or unknown terms |
| CR-07 privacy | RF-003, RF-009 | G1, G5 | Consent/contract, minimization, tenant, retention, deletion, transfer, and sensitive-inference review approved |
| CR-08 checkpoint lineage | RF-001, RF-002, RF-010 | G1 | Every checkpoint reaches approved random initialization with no prohibited parent or label |
| CR-09 split isolation | RF-007, RF-008 | G3 | Zero raw/derived overlap across frozen site/subject/session/device/time splits; split-scoped RuVector indexes verified |
| CR-10 hosted parity | RF-010 | G1, G5 | Local/hosted receipts bind identical governed inputs; provider terms and deletion evidence approved |
| CR-11 model card | RF-002, RF-007, RF-011 | G3-G5 | Template complete; every value evidence-labelled and reproducible; all blockers explicit |
| CR-12 trademark and patent | RF-001, RF-011 | G5 | Naming clearance and claim-level freedom-to-operate disposition signed for launch jurisdictions |
| CR-13 rollback | RF-012 | G5 | Contamination and runtime rollback drills preserve observations and evidence while disabling forecast authority |
| CR-14 bounded forecast contract | RF-004, RF-005, RF-006 | G2 | Property/fuzz/replay evidence proves finite bounded inputs, abstention, offline execution, immutable observation linkage, and derived-only evidence labels |
| CR-15 downstream authority | RF-006, RF-009, RF-012 | G4, G5 | No sensing-server hook in the initial PR; any later bridge proves forecast/LLM output cannot mutate observations or acquire prohibited action authority |
| CR-16 claims and rollout | RF-011, RF-012 | G3-G5 | Every claim has an allowed evidence label and reproducer; mode transitions cannot outrun the highest passed gate |

Any failed or missing check blocks the mapped gate. A maintainer waiver cannot
convert a prohibited license into permission or an unmeasured capability into a
measured claim.

## 18. Release record

The legal, data, security, model, and runtime owners sign the same immutable
release digest. Their approval covers the exact source, data, configuration,
weights, model card, SBOM, intended use, deployment mode, and jurisdictions.
Changing any governed input creates a new candidate and invalidates inherited
approval.

The first production review must also answer one explicit question: does the
business value of the measured forecast exceed the added privacy, operational,
compute, and legal burden compared with deterministic baselines? If not, the
correct outcome is to keep forecasting offline.
