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
