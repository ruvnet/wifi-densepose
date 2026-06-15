---
title: AetherArena — Spatial-Intelligence Benchmark
emoji: 📡
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
python_version: "3.12"
app_file: app.py
pinned: true
license: cc-by-nc-4.0
tags:
  - benchmark
  - leaderboard
  - wifi-sensing
  - spatial-intelligence
  - pose-estimation
---

# AetherArena ("AA") — The Official Spatial-Intelligence Benchmark

> Public leaderboard. Private evaluation split. Open scorer. Signed results.

The field's standard yardstick for camera-free **spatial intelligence** (pose, presence,
occupancy, tracking, vitals) from RF/WiFi and, over time, mmWave / UWB / multimodal.

- **Project-agnostic** — any team, framework, or modality enters; RuView donated the seed
  scorer and is scored like everyone else.
- **Benchmark-first** — the board starts empty; every row is a real scoring-pipeline
  **witness** (`inputs_sha256` + `proof_sha256` + `harness_version`) in an append-only,
  hash-chained, tamper-evident ledger.
- **Reproducible** — the scorer is open; reproduce any proof hash + repeatability locally.

Spec: [ADR-149](https://github.com/ruvnet/RuView/blob/main/docs/adr/ADR-149-public-community-leaderboard-huggingface.md).
Source + open scorer: https://github.com/ruvnet/RuView/tree/main/aether-arena

Non-commercial (CC BY-NC 4.0): the v0 eval split derives from MM-Fi (CC BY-NC); AA is operated non-commercially.
