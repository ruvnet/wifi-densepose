# /ruview-app — run a RuView sensing application

Run a RuView application. Which one: `$ARGUMENTS` (one of `presence`, `vitals`, `pose`, `sleep`, `environment`, `mat`, `pointcloud`, or a novel-RF app name; if empty, show the catalogue and ask).

- **presence / vitals / pose / environment** → `cd v2 && cargo run -p wifi-densepose-sensing-server` against a live ESP32 sink, or the Docker demo (`docker run -p 3000:3000 ruvnet/wifi-densepose:latest`) for simulated CSI. For environment also `-- --model model.rvf --build-index env`. Vitals: breathing 6–30 BPM (bandpass 0.1–0.5 Hz), heart rate 40–120 BPM (bandpass 0.8–2.0 Hz), `wifi-densepose-vitals` crate (ADR-021). Pose: 17 COCO keypoints via WiFlow (ADR-059 live pipeline) — train for accuracy (`/ruview-train`).
- **sleep** → `examples/sleep/` + `node scripts/apnea-detector.js` (sleep-stage classification, apnea screening).
- **mat** (Mass Casualty Assessment — disaster survivor detection) → `wifi-densepose-mat` crate, `docs/wifi-mat-user-guide.md`.
- **pointcloud** → `python scripts/mmwave_fusion_bridge.py` (camera depth via MiDaS + WiFi CSI + mmWave radar → unified spatial model, ~22 ms, 19K+ pts/frame; ADR-094).
- **novel RF** → `scripts/passive-radar.js`, `material-classifier.js`, `device-fingerprint.js`, `mincut-person-counter.js`, `gait-analyzer.js` (ADR-077/078).

No hardware? Fall back to the Docker demo or `python examples/ruview_live.py`. Visualisers: `node scripts/csi-spectrogram.js`, `node scripts/csi-graph-visualizer.js`.

Help me pick: through-wall → presence/activity (≤5 m depth); stationary subject → vitals/sleep; need skeletons → pose (train it); search & rescue → MAT; best spatial accuracy → 2+ ESP32 nodes + cross-viewpoint fusion (`v2/crates/wifi-densepose-ruvector/src/viewpoint/`), optionally + Cognitum Seed. Examples: `examples/{environment,medical,sleep,stress,happiness-vector}/`.
