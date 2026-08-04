# wifi-densepose-sensing-server

[![Crates.io](https://img.shields.io/crates/v/wifi-densepose-sensing-server.svg)](https://crates.io/crates/wifi-densepose-sensing-server)
[![Documentation](https://docs.rs/wifi-densepose-sensing-server/badge.svg)](https://docs.rs/wifi-densepose-sensing-server)
[![License](https://img.shields.io/crates/l/wifi-densepose-sensing-server.svg)](LICENSE)

Lightweight Axum server for real-time WiFi sensing with RuVector signal processing.

## Overview

`wifi-densepose-sensing-server` is the operational backend for WiFi-DensePose. It receives raw CSI
frames from ESP32 hardware over UDP, runs them through the RuVector-powered signal processing
pipeline, and broadcasts processed sensing updates to browser clients via WebSocket. A built-in
static file server hosts the sensing UI on the same port.

The crate ships both a library (`wifi_densepose_sensing_server`) exposing the training and inference
modules, and a binary (`sensing-server`) that starts the full server stack.

Integrates [wifi-densepose-wifiscan](../wifi-densepose-wifiscan) for multi-BSSID WiFi scanning
per ADR-022 Phase 3.

## Features

- **UDP CSI ingestion** -- Receives ESP32 CSI frames on port 5005 and parses them into the internal
  `CsiFrame` representation.
- **Vital sign detection** -- Pure-Rust FFT-based breathing rate (0.1--0.5 Hz) and heart rate
  (0.67--2.0 Hz) estimation from CSI amplitude time series (ADR-021).
- **RVF container** -- Standalone binary container format for packaging model weights, metadata, and
  configuration into a single `.rvf` file with 64-byte aligned segments.
- **RVF pipeline** -- Progressive model loading with streaming segment decoding.
- **Graph Transformer** -- Cross-attention bottleneck between antenna-space CSI features and the
  COCO 17-keypoint body graph, followed by GCN message passing (ADR-023 Phase 2). Pure `std`, no ML
  dependencies.
- **SONA adaptation** -- LoRA + EWC++ online adaptation for environment drift without catastrophic
  forgetting (ADR-023 Phase 5).
- **Contrastive CSI embeddings** -- Self-supervised SimCLR-style pretraining with InfoNCE loss,
  projection head, fingerprint indexing, and cross-modal pose alignment (ADR-024).
- **Sparse inference** -- Activation profiling, sparse matrix-vector multiply, INT8/FP16
  quantization, and a full sparse inference engine for edge deployment (ADR-023 Phase 6).
- **Dataset pipeline** -- Training dataset loading and batching.
- **Multi-BSSID scanning** -- Windows `netsh` integration for BSSID discovery via
  `wifi-densepose-wifiscan` (ADR-022).
- **WebSocket broadcast** -- Real-time sensing updates pushed to all connected clients at
  `ws://localhost:8765/ws/sensing`.
- **Static file serving** -- Hosts the sensing UI on port 8080 with CORS headers.

## Modules

| Module | Description |
|--------|-------------|
| `vital_signs` | Breathing and heart rate extraction via FFT spectral analysis |
| `rvf_container` | RVF binary format builder and reader |
| `rvf_pipeline` | Progressive model loading from RVF containers |
| `graph_transformer` | Graph Transformer + GCN for CSI-to-pose estimation |
| `trainer` | Training loop orchestration |
| `dataset` | Training data loading and batching |
| `sona` | LoRA adapters and EWC++ continual learning |
| `sparse_inference` | Neuron profiling, sparse matmul, INT8/FP16 quantization |
| `embedding` | Contrastive CSI embedding model and fingerprint index |

## Quick Start

```bash
# Build the server
cargo build -p wifi-densepose-sensing-server

# Run with default settings (HTTP :8080, UDP :5005, WS :8765)
cargo run -p wifi-densepose-sensing-server

# Run with custom ports
cargo run -p wifi-densepose-sensing-server -- \
    --http-port 9000 \
    --udp-port 5005 \
    --static-dir ./ui
```

### Using as a library

```rust
use wifi_densepose_sensing_server::vital_signs::VitalSignDetector;

// Create a detector with 20 Hz sample rate
let mut detector = VitalSignDetector::new(20.0);

// Feed CSI amplitude samples
for amplitude in csi_amplitudes.iter() {
    detector.push_sample(*amplitude);
}

// Extract vital signs
if let Some(vitals) = detector.detect() {
    println!("Breathing: {:.1} BPM", vitals.breathing_rate_bpm);
    println!("Heart rate: {:.0} BPM", vitals.heart_rate_bpm);
}
```

## Architecture

```text
ESP32 ──UDP:5005──> [ CSI Receiver ]
                          |
                    [ Signal Pipeline ]
                    (vital_signs, graph_transformer, sona)
                          |
                    [ WebSocket Broadcast ]
                          |
Browser <──WS:8765── [ Axum Server :8080 ] ──> Static UI files
```

## Live snapshot and spatial semantics

`GET /api/v1/nodes` is the current host-side node registry. A node is reported
`active` while its last sensing frame is at most five seconds old. Each entry
includes additive lifecycle diagnostics such as source address, host receive
time, frame sequence, reconstructed mesh timestamp, measured CSI rate, sync
validity, staleness, and the current fusion eligibility decision.

`GET /api/v1/sensing/latest` and `/ws/sensing` carry the most recently produced
`SensingUpdate`. They are cycle snapshots, not joins against the current node
registry. The WebSocket may repeat the last snapshot between CSI arrivals.

The current `signal_field` is a heuristic visualization derived from measured
subcarrier variance, motion and signal quality. It is not geometric
multistatic localization or a measured point cloud. Person positions derived
from its strongest cells inherit that limitation.

## Persistent Docker deployment

Runtime configuration, completed field-model calibration
(`field-model-v1.json`), models and recordings live below `/app/data`. Build a
repository-local image from the repository root and mount that directory:

```bash
docker build -f docker/Dockerfile.rust -t ruvnet/wifi-densepose:local .
docker run --network host --restart unless-stopped \
  --env-file /path/to/existing.env \
  -v /opt/ruview/data:/app/data \
  ruvnet/wifi-densepose:local \
  --node-positions '0.05,1.72,0.85;3.37,2.67,0.76;2.40,4.72,0.72'
```

On restart the server loads `config.json`, `field-model-v1.json`, models below
`models/`, recordings below `recordings/`, and the adaptive classifier when
present. Invalid calibration snapshots are logged and ignored rather than
preventing startup.

## Related Crates

| Crate | Role |
|-------|------|
| [`wifi-densepose-wifiscan`](../wifi-densepose-wifiscan) | Multi-BSSID WiFi scanning (ADR-022) |
| [`wifi-densepose-core`](../wifi-densepose-core) | Shared types and traits |
| [`wifi-densepose-signal`](../wifi-densepose-signal) | CSI signal processing algorithms |
| [`wifi-densepose-hardware`](../wifi-densepose-hardware) | ESP32 hardware interfaces |
| [`wifi-densepose-wasm`](../wifi-densepose-wasm) | Browser WASM bindings for the sensing UI |
| [`wifi-densepose-train`](../wifi-densepose-train) | Full training pipeline with ruvector |
| [`wifi-densepose-mat`](../wifi-densepose-mat) | Disaster detection module |

## License

MIT OR Apache-2.0
