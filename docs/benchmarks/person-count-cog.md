# `cog-person-count` — Benchmark Log

Append-only log of every published count_v1 training run per ADR-103. New runs add a section; never overwrite history.

## v0.0.1 — first measured run (2026-05-21)

### Setup

| Component | Value |
|-----------|-------|
| Training host | `ruvultra` (Ubuntu, x86_64, RTX 5080) |
| Backend | PyTorch 2.12 + CUDA |
| Data | `data/paired/wiflow-p7-1779210883.paired.jsonl` — 1,077 paired samples, single 30-min session, label distribution `{0: 533, 1: 544}` |
| Train/eval split | 80/20 stratified on `ts_start` (held-out tail of the recording) |
| Architecture | Conv1d encoder (56→64→128→128, dilations 1/2/4) + Linear(128→64→8) count head + Linear(128→32→1) confidence head — bit-identical to `v2/crates/cog-person-count/src/inference.rs::CountNet` |
| Loss | `cross_entropy(count) + 0.3·BCE(conf) + 0.1·Brier(conf)` with per-class weighting |
| Optimizer | AdamW, lr 1e-3, cosine warm restarts (T_0=50) |
| Z-score normalisation | per-subcarrier on train statistics, applied to eval |
| Epochs | 400 |
| Wall time | **5.6 s** |

### Accuracy (held-out 215-sample tail of the 30-min recording)

| Metric | Value |
|--------|-------|
| Best eval accuracy | **65.1%** |
| Final eval accuracy | 65.1% |
| Within ±1 | **100%** (labels are all in `{0, 1}`, predictions trivially within ±1) |
| MAE | 0.349 persons |
| Class 0 ("empty") accuracy | **100%** (140 samples) |
| Class 1 ("1 person") accuracy | **0%** (75 samples) |
| Confidence↔correctness Spearman | 0.023 |

### Honest read

The model overfit hard. By epoch 100 train_acc reached 1.0 and eval_loss climbed from 0.67 → 7.8. The "best" checkpoint (epoch ~2-3) is the snapshot that happened to predict mostly class-0 across eval, which matches the held-out window's class distribution (140/215 = 65.1%) — i.e. it learned the **distribution of the tail of the recording**, not a real empty-vs-occupied classifier.

Why: the training data is one continuous 30-minute solo recording. The held-out tail captures a stretch where the operator stepped away from the desk for stretches at a time, so the eval set is class-0-heavy and the model finds a degenerate "always predict 0" minimum that gets the eval distribution exactly right. Class 1 accuracy = 0 is the smoking gun.

Same data-bound failure mode as `pose_v1` (#645). Same fix path: multi-room paired recordings.

### What v0.0.1 still validates

- **Pipeline correctness end-to-end.** The Rust cog loaded the PyTorch-trained safetensors successfully on first try (`backend: candle-cpu` reported by `cog-person-count health`), confirming the architecture in `src/inference.rs` is byte-compatible with `train-count.py`.
- **ONNX parity.** 16 KB ONNX, exports cleanly under opset 18 with dynamic batch axis.
- **Fast iteration loop.** 5.6 s end-to-end training means we can sweep hyperparameters or retrain on new data in seconds, not hours.
- **Cog binary size.** Same 2.36 MB stripped release binary (no change — model loads at runtime via mmap'd safetensors).

### Comparison to ADR-103 v0.1.0 targets

| Gate | Target | Today | Status |
|------|--------|-------|--------|
| Day-0 same-room accuracy within ±1 | ≥ 80% | 100% (trivially — labels span {0,1}) | met |
| Cross-room accuracy within ±1 | ≥ 60% | Not measured (no cross-room data) | deferred to v0.2.0 |
| MAE | ≤ 0.6 | 0.349 | met |
| Per-frame confidence reflects accuracy (Spearman) | r ≥ 0.5 | 0.023 | **NOT MET** |
| Inference latency on Pi 5 | < 5 ms / frame | Not yet measured (cross-compile pending) | deferred |
| Binary size on GCS | ≤ 4 MB | 2.36 MB | met |

The accuracy ones look "met" only because the labels collapse to {0, 1} and "within ±1" with 8 classes is trivially satisfied. The **confidence calibration is the real failure** for v0.0.1 — Spearman 0.023 means the confidence head is essentially random noise. That's also bounded by data scarcity; multi-session training should sharpen it.

### Artifacts

- `v2/crates/cog-person-count/cog/artifacts/count_v1.safetensors` — 392 KB
- `v2/crates/cog-person-count/cog/artifacts/count_v1.onnx` — 16 KB
- `v2/crates/cog-person-count/cog/artifacts/count_train_results.json` — full per-epoch loss curve + hyperparameters + per-class breakdown

### Reproducibility

```bash
# On any host with PyTorch + CUDA (cargo path not needed for training):
scp data/paired/wiflow-p7-1779210883.paired.jsonl <host>:/tmp/
scp scripts/train-count.py <host>:/tmp/
ssh <host> "cd /tmp && python3 train-count.py --paired wiflow-p7-1779210883.paired.jsonl --epochs 400"
```

Loads in the Rust cog with no translation step (safetensors layout matches `cog-person-count::inference::CountNet` exactly):

```bash
cp count_v1.safetensors v2/crates/cog-person-count/cog/artifacts/
cargo run -p cog-person-count --release -- health
# → {"backend":"candle-cpu", "synthetic_count": <int>, "synthetic_confidence": <float>, ...}
```
