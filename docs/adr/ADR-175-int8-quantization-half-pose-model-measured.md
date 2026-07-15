# ADR-175: int8 Quantization of the WiFlow-STD "half" Pose Model — MEASURED accuracy/size trade-off

| Field | Value |
|-------|-------|
| **Status** | Accepted — MEASURED, reproducible (honest negative) |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **EDGE-INT8** |
| **Sub-deliverable** | 8.2 of the benchmark/optimization milestone |
| **Metric lock** | ADR-173 (one declared PCK normalization for every reported number) |
| **Motivated by** | `docs/research/sota-nn-train-benchmark-brief.md` (§edge int8) |

## Context

The SOTA brief characterized the int8 edge story for the WiFlow-STD pose net as
"fully characterized" for PTQ on the **published 2.23M** model (static QDQ
conv-only = the sweet spot; dynamic int8 ≈ no-op on this all-conv net), and named
**QAT-int8 on the strictly-dominating 843,834-param "half" model** as "the one
untested edge lever." This ADR is the reading of that lever — a MEASURED
fp32-vs-int8 trade-off for the half model, not a claim.

The half model (`half_best.pth`, 843,834 params) is the efficiency-sweep winner
from ADR-152 (`run_sweep.py` VARIANTS[0]: `tcn=[270,220,170,120]`,
`conv=[4,8,16,32]`, `attn_groups=4`). Its fp32 accuracy was recorded in the sweep;
this ADR re-measures it under the locked normalization and quantizes it.

**The whole point of this deliverable is reproducibility.** Every number below was
produced by running `v2/crates/wifi-densepose-train/scripts/quantize_half_int8.py`
on host `ruvultra` (RTX 5080, torch 2.11.0+cu128) against the real checkpoint and
the real seed-42 test split. The script + the exact command + the recorded stdout
**is** the proof artifact. Nothing here is estimated.

## Decision

Quantize the half model to int8 with **both** levers and report both honestly:

1. **QAT (primary target)** — FX graph-mode quantization-aware training, fbgemm
   backend, 3 epochs of fake-quant fine-tuning from `half_best.pth` (AdamW lr 2e-5,
   the existing `PoseLoss`), then `convert_fx` to a true int8 graph.
2. **PTQ static QDQ (the brief's "sweet spot", measured as the honest fallback)** —
   FX graph-mode static PTQ, fbgemm, calibrated on 64 train batches.

### Locked normalization (ADR-173)

**Torso-diameter PCK** — neck (keypoint idx 2) → pelvis (idx 12) distance — the
standard MM-Fi/GraphPose-Fi convention. This is exactly the default
`use_torso_norm=True` path of the upstream harness's `utils/metrics.calculate_pck`.
The **same** `calculate_pck`/`calculate_mpjpe` that produced the sweep's fp32
numbers scores **both** fp32 and int8 here, so the comparison is metric-locked: no
normalization is mixed, and the fp32 baseline reproduces the sweep's recorded
`half` test numbers bit-for-bit (PCK@20 clean = 96.62%), confirming the harness is
the same one.

### Device note (why int8 is CPU)

PyTorch int8 quantized kernels execute on CPU (fbgemm/x86), not CUDA. So int8 eval
is CPU. To keep the accuracy delta device-matched (not confounding int8-vs-fp32
with CPU-vs-GPU), the script measures an **fp32-CPU** baseline too. fp32-CPU and
fp32-GPU agree to 4 decimals (PCK@20 clean 0.96623 vs 0.96623), so CPU/GPU
introduces no drift — the int8 deltas below are pure quantization effect.

## MEASURED results (clean test subset = 52,560 NaN-free windows; torso-PCK)

Source: stdout of the run below + `~/wiflow-std-bench/sweep/int8/int8_results.json`.

| model | quant | size (MB) | PCK@20 | PCK@50 | MPJPE | Δ PCK@20 | Δ PCK@50 | size win |
|-------|-------|-----------|--------|--------|-------|----------|----------|----------|
| **fp32** (cpu) | — | **3.351** | **96.62%** | **99.47%** | **0.008981** | — | — | 1.00× |
| int8 PTQ static | PTQ | 1.046 | 40.98% | 94.98% | 0.038262 | **−55.64 pp** | −4.49 pp | 3.20× smaller |
| int8 QAT (3 ep) | **QAT** | 1.043 | 67.48% | 98.69% | 0.026548 | **−29.15 pp** | −0.78 pp | 3.21× smaller |

Full-test-set (54,000 windows incl. NaN-zero-filled files 487–499) tracks the
clean subset: fp32 96.10% / int8-PTQ 41.11% / int8-QAT 67.48% PCK@20 — same shape,
recorded in the JSON.

### Verdict

**int8 is NOT a win for this model at the tight PCK@20 edge target — honest no.**

- **PTQ static collapses** (−55.64 pp PCK@20). Naive static QDQ destroys the half
  model. The "sweet spot" characterization from the brief does not transfer from
  the 2.23M model to this 843k model at the strict torso-PCK@20 threshold.
- **QAT recovers a large share of the relative gap** (PTQ 40.98% → QAT 67.48%) but
  still **loses 29.15 pp** at PCK@20 for a 3.21× size reduction. At the loose
  PCK@50 threshold QAT is nearly lossless (−0.78 pp), i.e. coarse-localization
  survives int8 but fine-localization does not.
- The size win is real and consistent (3.2× smaller, 3.351 MB → ~1.04 MB), but
  **3.2× compression at −29 pp PCK@20 is a bad trade** when the half model already
  fits comfortably in edge flash at fp32. Recommendation: **keep fp32 (or fp16)
  for the half model on the edge**; do not ship this int8 variant as-is.

### Observed fake-quant → int8 conversion gap (disclosed, not hidden)

During QAT the **fake-quant** model's val PCK@20 reached 83.45% (epoch 3), but the
**converted int8** model scores 67.48% on test. A ~16 pp drop on `convert_fx` is a
real effect — the fbgemm int8 kernels are not bit-identical to the fake-quant
simulation (per-tensor activation quant + the axial-attention `einsum`/softmax path
quantize worse than the straight-through estimate predicts). This gap is the honest
reason QAT did not close the loss, and it is exactly the kind of number that would
be invisible if one only reported the fake-quant proxy. We report the **converted
int8** number as the deliverable, not the fake-quant proxy.

## Reproduction

```bash
ssh ruvultra 'cd ~/wiflow-std-bench && source venv/bin/activate && \
  python ~/quantize_half_int8.py --mode both --qat-epochs 3 2>&1'
```

- Script (committed): `v2/crates/wifi-densepose-train/scripts/quantize_half_int8.py`
  (scp'd to `~/quantize_half_int8.py` on ruvultra for the run).
- Inputs (on ruvultra, unmodified): `~/wiflow-std-bench/sweep/half_best.pth`,
  `~/wiflow-std-bench/preprocessed_csi_data/` (seed-42 file-level 70/15/15 split),
  upstream `models`/`dataset`/`utils/metrics`/`losses` (DY2434/WiFlow @ 06899d29,
  Apache-2.0), and `sweep/model_compact.py` (the half-model definition).
- Outputs (written, non-destructive): `~/wiflow-std-bench/sweep/int8/` —
  `half_int8_qat.pth`, `half_int8_ptq_static.pth`, `int8_results.json`,
  `int8_run.log`. **No existing file under `~/wiflow-std-bench` was modified.**
- Run metadata: host `ruvultra`, GPU RTX 5080, torch `2.11.0+cu128`, fbgemm engine,
  `date_utc 2026-06-15T12:35:06Z`, QAT ≈ 97 s/epoch.

## What is MEASURED vs CLAIMED

- **MEASURED:** every PCK/MPJPE/size number in the table; the fp32 baseline (which
  reproduces the recorded sweep `half` numbers); the PTQ collapse; the QAT partial
  recovery; the fake-quant→int8 conversion gap; the 3.2× size reduction.
- **CLAIMED / not done here:** ONNX/TFLite export; on-real-edge (ESP32/Pi/Hailo)
  latency or energy (int8 here is measured on x86 fbgemm, the dev box, **not** an
  edge SoC — the size number transfers, a latency number does **not**); a
  per-layer mixed-precision search that might keep the attention block in fp32; QAT
  beyond 3 epochs or with learned-quant-range schedules. Those are the obvious next
  levers if int8 is revisited; none is asserted as a result.

## Honest scope / limitations

- **Single eval split** — one seed-42 file-level test partition; no cross-room /
  cross-environment generalization split (the GraphPose-Fi frontier from ADR-173 is
  a separate, harder split and is not what is measured here).
- **In-domain only** — these are in-distribution test numbers; they say nothing
  about the cross-environment robustness gap.
- **x86 int8, not edge-SoC int8** — accuracy and size transfer to an edge int8
  runtime; the runtime/latency does not (different kernels, different SoC). No
  latency claim is made.
- **QAT lightly tuned** — 3 epochs, single LR, default fbgemm qconfig. A longer /
  better-tuned QAT might narrow the −29 pp, but on the evidence here int8 does not
  reach fp32 at PCK@20, and that is the reportable result today.

## Consequences

### Positive
- The "one untested edge lever" (QAT-int8 on the half model) is now MEASURED. The
  edge int8 question for the half model is answered with reproducible numbers: at
  the strict PCK@20 target it loses, and we can say so with a committed script.
- Establishes a reusable, metric-locked quantization+eval harness
  (`quantize_half_int8.py`) for any future int8 attempt on these compact variants.

### Negative
- None to the codebase (additive script + ADR + CHANGELOG only; no production Rust
  or signal-pipeline change; Python deterministic proof hash
  `f8e76f21a0f9852b70b6d9dd5318239f6b20cbcb4cdd995863263cecdc446f7a` unchanged).

### Neutral
- The negative verdict means the half model stays fp32/fp16 on the edge for now.
  int8 for these compact pose nets is parked pending the next-lever work above.

## Links
- ADR-173 — metric-locked PCK/MPJPE harness (the locked normalization used here)
- ADR-152 — WiFi-Pose SOTA 2026 intake / WiFlow-STD benchmark / efficiency sweep
  (produced `half_best.pth`)
- `docs/research/sota-nn-train-benchmark-brief.md` — §edge int8 (the "one untested
  lever" this ADR measures)
- Script: `v2/crates/wifi-densepose-train/scripts/quantize_half_int8.py`
