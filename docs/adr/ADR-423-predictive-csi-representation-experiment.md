# ADR 423: Predictive CSI representation experiment

Status: Experimental research harness. Not a paper reproduction or replacement model.

Date: 2026 10 13 UTC

## Context and decision

CSI-JEPA and AM-FM motivate representation learning from unlabeled CSI. Their
paper results have not been reproduced here. Implement a small masked latent
prediction experiment with an exponential moving average teacher, stop gradient
targets, a visible context encoder and a masked target predictor. Record collapse
diagnostics and compare validation loss against an untrained and constant
prediction baseline using the same final teacher targets.

This is an amplitude representation with explicit time and subcarrier axes. It
is not the original eight engineered feature encoder, occupancy classifier,
dense pose model or a physical world simulator. Preserve the published model.
Train normalization only on the training partition. Hold out complete groups
before any fitting. The optional existing ESP-Fi path admits only the original
training partition, with its own inner development split; it does not reopen
the original validation or final test arrays. Its activity labels do not become
occupancy ground truth.

## Training and artifact boundaries

Bound shapes, samples, epochs and resource settings. Use deterministic seeds
and record exact dependency versions and source/artifact hashes. Save tensors
through safetensors, not executable pickle. Reject nonfinite inputs and outputs,
unsafe paths, incompatible normalization or tensor shapes, and malformed metadata.
An exported frozen encoder can feed a small downstream adapter; implementing an
adapter interface does not demonstrate downstream accuracy.

Mask the target information before student encoding. Targets receive no gradient;
the teacher updates only by the declared moving average. Normalization must not
use per example hidden targets to leak target information into visible context.
Loss decline alone can mean representation collapse. Report variance diagnostics
and do not automatically promote a checkpoint for lower pretraining loss.

## Alternatives and acceptance

A reconstruction model is simpler but tests a different objective. Scaling
training first would spend resources before establishing valid data and objective
contracts. This small CPU experiment establishes the software path and failure
diagnostics before authorized larger training.

Tests require real gradients, teacher isolation, masking behavior, train only
normalization, group separation, bounded safe reload and frozen encoder parity.
Timing must disclose preprocessing and observation window exclusions. Synthetic
smoke results remain labelled synthetic. Real transfer must be evaluated with
independent downstream labels and simple baselines on fresh groups. No training
run in this ADR authorizes model replacement or production deployment.

## Research sources

* CSI-JEPA: https://arxiv.org/abs/2605.14171
* AM-FM: https://arxiv.org/abs/2602.11200

Source checkpoint availability and licensing have not been verified. No upstream
weights or datasets are implicitly authorized by citing these papers.
