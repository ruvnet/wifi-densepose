# RuView Calibration Service (reference implementation)

Turn a **shared WiFi-CSI pose base model** into a room-specific one with a **30-second labeled
calibration** and a **~11 KB per-room LoRA adapter**. This is the deployable resolution of the
cross-subject / cross-environment generalization problem (full study: [ADR-150 §3.3–3.6](../../docs/adr/ADR-150-rf-foundation-encoder.md)).

## Why

Zero-shot WiFi pose generalizes poorly to a **new room or new person** — an unseen room can drop a
strong model to near-random. But that gap is **not** algorithmically closeable (CORAL, DANN,
instance-norm, contrastive foundation-pretraining all failed) and **not** closeable by collecting
more subjects (saturates ~64%). It **is** closeable, cheaply, at deployment time: a handful of
labeled frames from the actual room pin down its multipath instantly.

| Deployment case | Zero-shot | + in-room calibration |
|-----------------|----------:|----------------------:|
| Same room, new person (cross-subject) | 64% | **76%** (200 samples) |
| **New room + new person (cross-environment)** | **~10%** | **60% @ 5 samples → 73% @ 200** |

**Verified demo (this code, source-only base on an unseen MM-Fi room E04):**
`zero-shot 3.09% → after 200-sample calibration 74.29%` (+71 pts).

## How it works

A frozen shared **base** (transformer + temporal attention pool + skeleton-graph head, the published
[`ruvnet/wifi-densepose-mmfi-pose`](https://huggingface.co/ruvnet/wifi-densepose-mmfi-pose)) plus a
tiny **LoRA adapter** (rank 8 on the input projection + pose head — **11,200 params ≈ 11 KB int8 /
22 KB fp16**) fitted per room. Thousands of room-adapters hang off one base.

## Usage

```bash
# 1) Capture a short labeled clip in the deployment room -> calib.npz {X:[N,3,114,10], Y:[N,17,2]}
#    (~100–200 samples recommended; below ~20 the adapter can underperform zero-shot)

# 2) Fit the per-room adapter (~11 KB):
python calibrate.py --base pose_mmfi_best.pt --data calib.npz --out room.adapter.npz

# 3) Run calibrated inference (base + room adapter):
python infer.py --base pose_mmfi_best.pt --adapter room.adapter.npz --data frames.npz --out kp.npy
#    omit --adapter to run the uncalibrated (zero-shot) base
```

`X` is CSI amplitude `[N, 3 antennas, 114 subcarriers, 10 frames]` (per-sample standardization is
applied internally). `Y` is `[N,17,2]` COCO keypoints in `[0,1]`.

## Calibration budget (measured, rank-8 LoRA, 3 seeds — ADR-150 §3.5)

| Labeled samples/room | cross-subject | cross-environment |
|---------------------:|--------------:|------------------:|
| 0 (zero-shot) | 64% | ~10% |
| 5 | — | 60% |
| 20 | 66% | 66% |
| 50 | 70% | 70% |
| 200 | 72% | 73% |

Knee at ~50 samples (~70%); **below ~20 samples the adapter can hurt** (too few to fit reliably).

## Two models, two producers (not interchangeable)

Adapters are **model-specific**. There are two calibration producers here:

| Producer | Target model | Input | Adapter format | Consumer |
|----------|--------------|-------|----------------|----------|
| `calibrate.py` | MM-Fi **transformer** (`pose_mmfi_best.pt`, 3×114×10) | `[N,3,114,10]` | `.npz` (`proj`/`head` LoRA) | this Python `infer.py` |
| `cog_calibrate.py` | cog **conv+MLP** (`pose_v1.safetensors`, 56×20) | `[N,56,20]` | `.safetensors` (`fc1.a`/`fc1.b`/`fc2.a`/`fc2.b`) | Rust `cog-pose-estimation run --adapter` |

```bash
# Produce a cog-format per-room adapter for the deployed Rust pose engine:
python cog_calibrate.py --base pose_v1.safetensors --data calib.npz --out room.safetensors
# then in the cog runtime:
cog-pose-estimation run --config <cfg> --adapter room.safetensors
```

Same LoRA *mechanism* (ADR-150 §3.5), different architecture and key layout — an adapter from one
producer will not load into the other model.

## Notes

- **Calibration only helps when the base hasn't already seen the room.** The published flagship was
  trained on MM-Fi `random_split`, so calibrating it on an MM-Fi subject is a near-no-op (it already
  saw them); for a genuinely new real-world room it is zero-shot and calibration applies. To
  *reproduce the demo* on a held-out MM-Fi room, train a source-only base (exclude the target
  environment) — see `ADR-150 §3.6` and the few-shot harness in `aether-arena/staging/`.
- Adapter is saved fp16 (~22 KB); quantize to int8 for the ~11 KB on-device form.
- Inference is real-time on CPU (the 75 K-param `micro` variant runs in 0.135 ms single-thread x86;
  see [`docs/benchmarks/wifi-pose-efficiency-frontier.md`](../../docs/benchmarks/wifi-pose-efficiency-frontier.md)).
