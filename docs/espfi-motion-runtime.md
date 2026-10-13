# Buffered ACF inference

Protocol frozen on 2026-10-13. Optimize execution of the retained compact
activity classifier without changing its weights, features, training or labels.
The compact reference and original explicit model remain rollback paths.

## Evidence and hypothesis

A diagnostic cProfile run over 1,000 real development windows attributed
0.490 of 0.533 seconds inside prediction to motion feature extraction, about
92%. Profiling adds overhead and those numbers are not latency benchmarks.
The classifier is a small part of this path. Reusing feature buffers and
avoiding temporary arrays is therefore the selected implementation experiment.

The existing compact export already removes reconstructable map storage.
Further file compression is not the objective here. Float64 feature arithmetic
remains mandatory because earlier float32 feature experiments failed nearly
constant signal tests despite matching ordinary development predictions.

## Architecture and boundaries

Use a standalone NumPy and safetensors runtime that loads the existing compact
artifact. Verify bounded status, metadata and model bytes; hash the same bytes
that are deserialized. Require the frozen candidate and exact random map recipe,
float64 tensor shapes, finite state and positive feature scales. Regenerated
map hashes must match before affine folding. Do not rewrite the artifact.

Reuse a fixed per-instance workspace for 52 by 190 amplitude windows, centered
log samples, products, carrier means and energies, 468 features and 128 hidden
values. Preserve the six autocorrelation and three absolute difference lags.
Compute the amplitude floor in float64 explicitly, including float32 inputs.
Keep the original global RMS floor 1e-6 and carrier energy floor 190e-12.

One lock protects each instance's workspace. Returned logits own their memory.
Use a separate instance per concurrent stream when parallel execution is needed.
Bound input batches at 4,096 windows and validate shape, dtype, nonnegativity and
finiteness. The inference import path must not load Torch, SciPy or training code.

## Frozen acceptance

1. Compare all 837 training and 70 reused development recordings against both
   the compact folded reference and the original explicit model. Require exact
   predicted classes and maximum absolute logit error at most 1e-10.
2. Verify mathematical edge cases, mixed floor precision, noncontiguous inputs,
   batch boundaries, output ownership, concurrent calls, corrupted artifacts
   and failed status rejection. Do not open final test arrays.
3. Run three rounds of 1,000 rotating alternating paired batch-one predictions
   after 100 warmups. Compare the actual public prediction APIs against the
   compact folded reference on identical inputs and one CPU thread.
4. Retain the runtime only if median and p95 are each at least 10% lower in
   every round and all correctness checks pass. Preserve all measured rounds,
   including failures; no timing cherry picking or lower post hoc threshold.

Timing includes input validation, feature extraction and classification. It
excludes loading, map regeneration, acquisition, pooling and I/O. Report retained
model and workspace bytes separately from total process memory. The original
test was opened once during the first pilot and is not reopened here. Existing
development data establishes numerical parity, not new accuracy evidence.

The biggest remaining deployment uncertainty is latency on the intended device.
Acceptance there requires the same prediction checks and measured device p50/p95;
shared x86 timings cannot establish ESP32, ARM or sensor-to-decision latency.
