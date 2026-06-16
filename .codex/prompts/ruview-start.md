# /ruview-start — onboard onto RuView

Help me get started with RuView (WiFi-DensePose). Path: `$ARGUMENTS` (one of `docker`, `build`, `hardware`; if empty, ask which hardware I have).

- **docker** (no hardware): `docker pull ruvnet/wifi-densepose:latest && docker run -p 3000:3000 ruvnet/wifi-densepose:latest`, then open http://localhost:3000 (simulated CSI, full UI).
- **build** (from source): `cd v2 && cargo test --workspace --no-default-features`, then `cd .. && python archive/v1/data/proof/verify.py` (expect `VERDICT: PASS`). Single-crate sanity: `cargo check -p wifi-densepose-train --no-default-features`.
- **hardware** (ESP32-S3/C6): use `/ruview-flash` then `/ruview-provision`, then `cd v2 && cargo run -p wifi-densepose-sensing-server` to consume the UDP CSI stream. Also: `node scripts/rf-scan.js --port 5006`, `node scripts/snn-csi-processor.js --port 5006`.

Warn me about: ESP32-C3 / original ESP32 are unsupported (single-core); one node = limited spatial resolution (use 2+ or add a Cognitum Seed); camera-free pose is modest — camera-supervised training reaches 92.9% PCK@20 (ADR-079); no cloud/cameras/internet needed.

Then point me at next steps: `/ruview-app`, `/ruview-train`, `/ruview-verify`, and the configuration workflow (sdkconfig variants, NVS provisioning, edge modules, mesh, Cognitum Seed). Reference `README.md`, `docs/user-guide.md`, `docs/build-guide.md`, `docs/TROUBLESHOOTING.md`, `examples/`.
