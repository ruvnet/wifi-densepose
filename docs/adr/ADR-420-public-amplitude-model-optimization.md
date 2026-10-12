# ADR-420: Public amplitude activity model optimization

Status: Proposed, candidate implementation

Date: 2026-10-12

## Context

ADR-419 reproduces the eight-feature temporal embedding contract. Public
ESP-Fi HAR samples contain 52 amplitude channels over 950 time points, not
those eight features. Reinterpreting subcarriers as engineered features would
invalidate the pretrained model's input semantics. Activity recognition is
also a different task from temporal retrieval or dense pose estimation.

## Decision

Implement a separate amplitude classifier experiment with its own artifacts.
The original model and production sensing remain the rollback baseline.

Pin the upstream revision and full archive SHA256. Retain the upstream CC BY
4.0 attribution, numeric activity identifiers and explicit provenance. The
upstream README and code disagree about activity names, so no candidate may
claim named activity or fall detection performance until that conflict is
resolved. Never commit captures or prepared signal arrays.

Separate rooms and participants before fitting any preprocessing. Exclude
crossed validation/test identities from the training set. Reject duplicates,
nonfinite arrays, unsafe archive paths, unsupported shapes and unverifiable
source bytes. Software fixture tests must retain synthetic evidence labels.

Use a bounded declared search over a statistical baseline and compact temporal
networks. Fit scaling on training only. Select architecture and checkpoint on
validation macro F1. Freeze that choice before final test predictions. Report
the fixed majority baseline, confusion matrix, per-class F1, artifact bytes,
parameter count, runtime and batch-one CPU latency scope.

Fold convolution batch normalization only after training, check validation
logit parity, then measure the actual inference result. A smaller file is not
proof of lower device RAM, a CPU timing is not an ESP32 timing, and an activity
score is not a pose score. Record exact code and artifact digests.

## Acceptance and deployment

Unit and integration tests cover partition isolation, provenance, malformed
data, real gradients, export reload and failed lifecycle rejection. A real
public data run must be reproduced separately from synthetic CI.

No candidate automatically replaces the published Hugging Face encoder or a
production model. Public benchmark performance does not establish transfer
to RuView field hardware. Deployment requires independent field validation,
confirmed label semantics, target runtime integration and explicit promotion.
Repeated inspection of the same final test invalidates an untouched test
claim for later tuning; retain an external confirmation set for that work.

## Sources

* https://github.com/AutoSmartGroup/ESP-Fi-HAR
* https://doi.org/10.1016/j.adhoc.2026.104192
* ADR-419 and ADR-299

## Development-only generalization extension

After the initial test was opened, further optimization must not treat it as
an untouched evaluation. A separate development loader validates the complete
archive-derived bundle but deserializes only training and validation arrays.
The original test is absent from the model search API.

Freeze four feature families and four ridge regularization strengths before
searching. Select using two inner folds entirely within the original training
partition: room 1 participants 1 through 4 to room 2 participants 5 and 6,
and the reverse room direction with the same participant separation. Fit every
scaler on the fitting side of its fold. Select by mean inner macro F1, then fit
all original training data. Compare once on existing validation against the
original frozen bundle and label this development evidence, not a new test.

Only a fresh independent confirmation dataset can authorize a stronger
generalization claim. Feature extraction, model latency and artifact integrity
must be measured together for the new artifact family.

## Field context research boundary

The RuField and WorldGraph exploration uses two explicitly separate paths.
The Python amplitude experiment may add temporal descriptors within a trial;
it cannot invent continuous chronology, verified empty room calibration,
positions or room topology. The standalone Rust integrity probe calls pinned
actual RuField and WorldGraph crates using authored synthetic events. Public
unlabeled replay is inspected for format only and never promoted to labeled
activity evidence or graph beliefs.

Root sensor evidence and derived graph beliefs remain distinct. The probe
bridge rejects derived input as new independent roots, duplicate event hashes,
stale observations and expired or untraceable beliefs. The bridge is not a
production graph cycle detector or an enrollment authority. Native WorldGraph
permits cycles, and native RuField simulation trust does not enforce wall clock
freshness. Neither behavior proves a production defect when used as documented.

The frozen temporal extension improved internal training folds but regressed
development accuracy and macro F1. Reject it and preserve the retained motion
ACF model. Evidence, executable commands and future field acceptance are in
[`espfi-field-context-research.md`](../espfi-field-context-research.md).
