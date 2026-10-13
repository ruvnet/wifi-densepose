# ADR 424: Correct pooled representation collapse and test downstream utility

Status: Frozen research protocol before measurement. No promotion authority.

Date: 2026 10 13 UTC

## Observed failure

ADR 423's fixed experiment produced effective recording rank 1.4055 and a collapse
warning. Its masked latent MSE 0.9740 lost to a global constant baseline at 0.9413.
The penalty on token features did not establish diverse pooled representations.
Loss reduction alone is not a usable sensing improvement.

## Hypothesis and one bounded intervention

Keep the existing encoder, EMA teacher, masked objective and token variance
penalty. Add a penalty on the exact representation exported for downstream use:
the unmasked online encoder averaged over tokens for each recording. Use only
the current fitting minibatch. Penalize low standard deviation in each pooled
dimension and off diagonal covariance. This is inspired by VICReg, not a
reproduction of its complete method or a new algorithm claim.

Freeze the candidate before execution:

* Original token variance weight 0.1, as in ADR 423.
* Additional recording variance weight 1.0, target standard deviation 1.0,
  numerical epsilon 0.0001.
* Additional covariance weight 0.1, squared off diagonal sum divided by 32.
* Same initialization seed 42, architecture, normalization, batch size 8, five
  epochs, mask order, optimizer and EMA configuration for baseline and candidate.
* Only original training data: fit room 1 participants 1 through 4; evaluate room
  2 participants 5 and 6. These are reused development groups, not confirmation.

The baseline must reproduce the prior encoder's tensors or frozen embeddings
before candidate utility is interpreted. Extra deterministic forwards must not
change baseline training. Different EMA teachers define different latent target
spaces; cross model latent MSE is not the utility comparison.

## Downstream measurement

Freeze all encoders before fitting a linear activity probe. Evaluate baseline,
candidate, untrained encoder, raw statistical features and a majority baseline.
Fit scaling, ridge weights and intercept on fitting records only, with fixed
ridge regularization 1.0 and seven numeric classes. No validation alpha search,
checkpoint selection or adaptation. Room, participant and trial IDs cannot enter
features. Activity class names remain unresolved; no occupancy labels are implied.

The raw statistics comparator has 156 features versus 32 encoder features.
Report this asymmetry, per class F1, confusion matrices, accuracy and macro F1.
Deterministic fitted state hashes must remain unchanged when development labels
or development embeddings are changed. Label changes may change evaluation only.

## Predeclared development gate

The candidate must satisfy all conditions:

1. Macro F1 at least 0.02 above each of baseline, untrained and statistics probes.
2. Accuracy no more than 0.01 below the highest comparator accuracy.
3. No newly zero F1 class relative to the baseline.
4. Effective recording rank at least 2.0 on development embeddings.
5. Batch one encoder p95 at most 1.2 times baseline in every declared timing round.

Use repeated timings on identical inputs and the same CPU/thread settings.
Disclose normalization, loading and acquisition inclusions/exclusions. Compare
training duration and artifact sizes as costs. A numerical gate pass remains
development evidence and never authorizes promotion. A failure is preserved;
do not retune coefficients on these results or retry until the gate passes.

## Boundaries and rollback

Add isolated experiment modules; preserve ADR 423 runtime and receipts. Bound
input shapes, labels, fitting dimensions, resource settings and artifact reads.
Use safetensors with explicit versioned metadata, exact tensor checks and hashes.
No hardware control, autonomous installation, paid compute or model replacement.
Existing ruOS CPU may execute the bounded experiment. Remove the new modules to
roll back; production paths are not changed.

## Acceptance and source

Tests cover gradient direction and teacher isolation, pooled representation
definition, baseline reproduction, group isolation, frozen probe fitting,
malformed artifacts and finite outputs. A reproducible report must include every
comparator and every gate, even when the candidate fails.

VICReg primary source: https://arxiv.org/abs/2105.04906

The cited paper demonstrates the approach in its own setting. Applying recording
variance and covariance penalties to this CSI pipeline remains an experiment.
