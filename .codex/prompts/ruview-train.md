# /ruview-train — train a RuView model

Train / evaluate / publish a RuView model. Track: `$ARGUMENTS` (one of `camera-free`, `camera-supervised`, `embeddings`, `domain-gen`, `snn`, `gpu`; if empty, ask).

- **camera-free** (WiFlow pose, no labels): `cd v2 && cargo run -p wifi-densepose-sensing-server -- --pretrain --dataset data/csi/ --pretrain-epochs 50`, then `-- --train --dataset data/mmfi/ --epochs 100 --save-rvf model.rvf`. ~84 s on M4 Pro, modest accuracy. Bench `node scripts/benchmark-wiflow.js`, eval `node scripts/eval-wiflow.js`.
- **camera-supervised** (ADR-079, 92.9% PCK@20, ~19 min): `python scripts/collect-ground-truth.py` (MediaPipe landmarks; needs `data/pose_landmarker_lite.task`), `python scripts/collect-training-data.py` (CSI capture), `node scripts/align-ground-truth.js` (timestamp align), then `cd v2 && cargo run -p wifi-densepose-sensing-server -- --train --dataset data/paired/ --epochs <N> --save-rvf model.rvf`, eval `node scripts/eval-wiflow.js` (reports PCK@20).
- **embeddings** (AETHER ADR-024 / spectrogram ADR-076): `wifi-densepose-train` + `wifi-densepose-ruvector`; `-- --model model.rvf --embed`, `-- --model model.rvf --build-index env`. 171K emb/s on M4 Pro.
- **domain-gen** (MERIDIAN ADR-027): domain-generalization options in the training pipeline + `ruview_metrics`.
- **snn** (local env adaptation, <30 s): `node scripts/snn-csi-processor.js --port 5006`; `docs/tutorials/cognitum-seed-pretraining.md`; ADR-084/085 (RaBitQ), ADR-086 (novelty gate).
- **gpu**: `gcloud auth login && gcloud config set project cognitum-20260110`, then `bash scripts/gcloud-train.sh --dry-run` (smoke), `bash scripts/gcloud-train.sh --gpu l4 --hours 2` (proto, ~$0.80/hr), `bash scripts/gcloud-train.sh --gpu a100 --config scripts/training-config-sweep.json` (~$3.60/hr), `bash scripts/gcloud-train.sh --sweep` (full sweep). VM auto-deletes unless `--keep-vm`. Local Mac: `bash scripts/mac-mini-train.sh`. Bench: `python scripts/benchmark-model.py`.

Data: `data/recordings/` raw CSI · `data/csi/` pretrain · `data/mmfi/` MM-Fi · `data/paired/` camera↔CSI · `data/ground-truth/` MediaPipe · `models/` artifacts. Record more: `python scripts/record-csi-udp.py`.

After training: `cd v2 && cargo test --workspace --no-default-features`, `cd .. && python archive/v1/data/proof/verify.py` (VERDICT: PASS). Publish: `python scripts/publish-huggingface.py` (or `.sh`; `docs/huggingface/`). Then run `/ruview-verify`.
