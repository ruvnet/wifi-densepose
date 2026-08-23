# RuView Consumer NLOS

`ruview-nlos` is the RuView Labs G0 optical-transient scaffold for consumer time-of-flight sensors. It preserves zone-level photon timing histograms, consumes externally estimated sensor pose, evaluates an approximate canonical rigid-object likelihood with a particle filter, applies a CSI prior only in synthetic regression, and emits one bounded hidden-target posterior for RuVector, RuField, WorldGraph, Swift, and browser consumers.

This is not an ARKit depth map adapter. A depth map has already discarded the delayed multipath timing signal required for around the corner inversion. `TransientFrame::validate` rejects `depth_only` as live NLOS evidence.

## Reproduction boundary

The first physical research path is the public MIT consumer NLOS implementation at commit `15314de422a765a2d1b72ea7037dfafb2f908d7c`, used with the ST P NUCLEO 53L8A1 kit and VL53L8CH histogram output. RuView independently implements the documented STM32 row framing and 13-value configuration packet. The Rust preprocessing/scorer is a bounded synthetic architecture approximation, not numerical equivalence to the upstream 128-bin O'Toole resampling and calibration pipeline. Physical reproduction must run the pinned upstream path and the preregistered witness protocol.

The software path has four evidence classes:

| Input | Output ceiling | Meaning |
|---|---:|---|
| Deterministic generator | `l0_synthetic` | Software and performance regression only |
| Raw live histogram | `l1_measured` | Sensor bytes received, calibration not yet witnessed |
| Raw histogram plus bound empty room calibration | `l2_calibrated` | Measured optical posterior with calibration digest |

The v1 wire contract deliberately rejects `l3_corroborated`: it cannot retain both modality lineages. Measured CSI fusion is unavailable in v1; only a scope-bound synthetic L0 prior is accepted for architecture tests. A future contract must carry authenticated optical/RF lineage and coordinate bindings before measured fusion can be enabled.

Only the physical protocol in `docs/research/consumer-nlos-acceptance-protocol.md` can establish the hardware reproduction and fusion acceptance gates.

## Pipeline

1. `StAsciiDecoder` reads explicit USB serial rows and keeps all 8 to 128 timing bins for at most 64 zones.
2. `Calibration` averages an empty room, finds each direct wall peak, binds the result with SHA 256, subtracts background, masks the direct return, and maps time to uniform squared distance bins.
3. `MotionApertureTracker` retains up to 32 pose-tagged frames, estimates a bounded translational velocity in metres per second from monotonic frame time, back-warps moving hypotheses across the aperture, and evaluates 64 to 20,000 particles against a bounded canonical point cloud.
4. `CsiSpatialPrior` contributes a coarse Gaussian prior only when calibrated, finite, and fresh. Stale priors fail closed.
5. `TemporalFeatureMemory`, `RuFieldObservation`, and `WorldGraphUpdate` remove raw histograms and preserve confidence, evidence, calibration, and expiry.
6. `NlosHub` publishes an authenticated read-only HTTP/WebSocket surface. Production TLS is terminated by the required trusted reverse proxy. Native clients use a bearer header. Browsers exchange the bearer token for a 30 second, single-use, origin-bound ticket.

## Commands

```bash
cd v2

# Pure core, including deterministic acceptance tests
cargo test -p ruview-nlos --no-default-features

# Authenticated server and WebSocket tests
cargo test -p ruview-nlos

# Direct ST serial adapter compile and tests
cargo test -p ruview-nlos --all-features

# L0 architecture benchmark. This never sets the hardware gate to true.
cargo run -p ruview-nlos --release -- benchmark --frames 300 --particles 1000

# Track a bounded transient JSONL recording. Its first 60 frames must be empty room.
cargo run -p ruview-nlos -- track-jsonl capture.jsonl --background-frames 60

# Read the public STM32 firmware at 2,250,000 baud and emit raw transient JSONL.
cargo run -p ruview-nlos --features hardware -- capture-st \
  --port /dev/ttyACM0 --session lab-run-001 --frames 300 \
  --sensor-id st-kit-001 --sensor-model VL53L8CH \
  --firmware-version 15314de --pose-jsonl synchronized-poses.jsonl

# Run a loopback synthetic server for UI validation.
RUVIEW_NLOS_TOKEN="$(openssl rand -hex 32)" \
  cargo run -p ruview-nlos -- serve --synthetic \
  --allowed-origin http://127.0.0.1:8081
```

Non loopback bind requires `--behind-tls-proxy`; the flag is an operator assertion, not a TLS implementation. The proxy must terminate trusted TLS, strip untrusted forwarded headers, and apply network policy. Browser CORS is disabled unless one exact `--allowed-origin` is supplied; wildcard origins and cleartext non-loopback origins are rejected. The bearer value is hashed immediately and is never stored or logged.

## Performance model

The reference configuration is 16 zones by 48 bins by 1,000 particles. The optimized scorer does not materialize a full predicted histogram per particle. It projects only the three nonzero kernel samples around each canonical return, reducing the point target case from roughly 768,000 to 48,000 predicted sample operations per frame and bounding the aperture at eight frames by default.

`--fixed-sensor` is available only for bounded capture/transport diagnostics. Target motion does not substitute for the sensor-motion-induced aperture and a fixed-sensor capture cannot satisfy the physical MAS reproduction gate. The current pose JSONL is index-paired G0 offline scaffolding; live promotion needs timestamped pose/capture identity and clock synchronization.

CI requires at least 30 combined tracker updates per second and at least 25 percent synthetic lost track reduction from the CSI prior. Those are software regression gates. The published MIT result and the RuView field acceptance criteria remain separate hardware claims.

## Privacy and safety

Raw histograms stay in the local capture and calibration plane. Public track envelopes contain session local identifiers only and expire in at most five seconds. UNKNOWN tracks are first class. Neither client nor server can actuate a device. The threat model is in `docs/security/consumer-nlos-threat-model.md`.
