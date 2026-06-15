# RuView Streaming Engine v0.3.0 — Auditable Environmental Intelligence

## What this is

Most WiFi-sensing stacks emit a number and hope you trust it. **RuView's streaming
engine is built so you don't have to.** Every conclusion it reaches — "someone is
in the living room," "fall risk elevated," "the room layout changed" — carries a
full evidence trail: which sensors saw it, how much they agreed, which calibration
and model produced it, and what privacy policy it was emitted under.

The throughline is **trust**. If you ask *"why should I believe this when it says a
person fell?"*, the engine answers with signal evidence, sensor agreement,
calibration provenance, and an auditable privacy posture — not just a confidence
score.

This release lands the ADR-135→146 series: the data contracts, the
trust/privacy/audit machinery, and the algorithms — all real, tested, and
composed into one end-to-end pipeline cycle.

## The two layers that make it auditable

- **WorldGraph (`wifi-densepose-worldgraph`)** — the *where & why* graph. A typed
  graph of rooms, sensors, RF links, person tracks, object anchors, events, and
  beliefs, connected by typed edges: `observes`, `located_in`, `derived_from`,
  `contradicts`, `privacy_limited_by`. The privacy posture is *visible in the
  persisted graph* — an auditor can read exactly what was suppressed and why.
- **Trusted semantic records** — the *what we believe right now* record. Every
  semantic state carries model version, calibration version, evidence refs,
  confidence, expiry, and privacy action. High-stakes actions (caregiver
  escalation) require **multi-signal agreement**, not a single noisy primitive.

## What's new in v0.3.0

| Area | Capability |
|------|-----------|
| Frame contracts (ADR-136) | `ComplexSample` (LE-canonical), provenance fields on every frame, `CanonicalFrame` BLAKE3 witness, `Stage`/`Versioned`/`QualityScored` traits |
| Calibration (ADR-135) | `BaselineCalibration::apply()` stamps a deterministic `calibration_id` onto each frame |
| Fusion quality (ADR-137) | `QualityScore` with per-node weights, evidence refs, and contradiction flags; calibration-mismatch detection |
| Array coordination (ADR-138) | clock-quality + geometry gating; degraded nodes go "watch-only" |
| WorldGraph (ADR-139) | the typed digital twin + privacy rollup + deterministic persistence |
| Semantic records (ADR-140) | auditable state records + multi-signal agent routing |
| Privacy control plane (ADR-141) | named modes + actions + a BLAKE3 hash-chained, tamper-evident attestation |
| Evolution + VoxelMap (ADR-142) | cross-link "the room changed" detection + Bayesian occupancy, privacy-gated to a histogram |
| RF-SLAM (ADR-143) | persistent reflector discovery → learned static anchors |
| UWB fusion (ADR-144) | range-constraint refinement with outlier rejection (forward-looking) |
| Ablation harness (ADR-145) | feature-matrix metrics incl. membership-inference privacy leakage |
| RF encoder (ADR-146) | multi-task heads with per-head uncertainty + contrastive batcher (forward-looking) |
| **Engine (`wifi-densepose-engine`)** | the composition root: one `process_cycle()` runs the whole trust pipeline |

## Quick start

```rust
use wifi_densepose_engine::StreamingEngine;
use wifi_densepose_bfld::PrivacyMode;
use wifi_densepose_geo::types::GeoRegistration;
use wifi_densepose_signal::ruvsense::fusion_quality::CalibrationId;

// 1. Build the engine with a privacy posture + model version.
let mut engine = StreamingEngine::new(PrivacyMode::PrivateHome, 1, GeoRegistration::default());

// 2. Describe the space (rooms + sensors are WorldGraph nodes).
let room = engine.add_room("living_room", "Living Room");
let sensor = engine.add_sensor("esp32-com9", room);
engine.register_node_geometry(0, 1.0, 0.0, 0.0);   // ADR-138 array geometry (optional)

// 3. Each 50 ms cycle: feed per-node CSI frames + the calibration epoch.
let out = engine.process_cycle(&node_frames, CalibrationId(0xABCD), room, now_ms)?;

// 4. The result is a *trusted* belief — fully traceable.
println!("class={:?} demoted={} evidence={:?}",
         out.effective_class, out.demoted, out.provenance.evidence);
assert_eq!(out.quality.calibration_id, Some(CalibrationId(0xABCD)));

// 5. Persist the world model; reload reproduces the same query results.
let snapshot = engine.snapshot_json()?;        // RVF payload — never raw RF frames
```

Per-node calibration (mismatch demotes privacy automatically):

```rust
let out = engine.process_cycle_calibrated(
    &node_frames,
    &[Some(CalibrationId(1)), Some(CalibrationId(2))], // disagree → CalibrationIdMismatch
    room, now_ms)?;
assert!(out.demoted);                          // privacy class demoted to Restricted
assert_eq!(out.quality.calibration_id, None);  // no single calibration epoch
```

## Validated (acceptance tests that prove the architecture)

- **ADR-137** `two calibrated frames → calibration mismatch → QualityScore contradiction → Restricted → calibration_id None → witness stable`
- **ADR-139** `live_frame → fusion → worldgraph_update → privacy_rollup → persist → reload → same_contents` (no raw RF persisted)
- **ADR-140** `raw snapshot → semantic primitive → SemanticStateRecord → agreement rule → expired record rejected`
- **ADR-142** `3 links drift 30 frames → ChangePoint → VoxelMap accumulates → low-confidence suppressed → VoxelGate Restricted histogram → ADR-137 contradiction`

## Performance & safety

- **~6.35 µs per full cycle** (4 nodes / 56 subcarriers) — ~7,800× under the 50 ms / 20 Hz budget (criterion: `cargo bench -p wifi-densepose-engine`).
- New crates are `#![forbid(unsafe_code)]`; no hardcoded secrets; input validated at boundaries; privacy demotion is monotonic; mode changes are hash-chain attested.
- `wifi-densepose-core` and `wifi-densepose-bfld` build `#![no_std]` for the ESP32-S3 on-device path.

## Build & test

```bash
cd v2
cargo build --release --workspace --no-default-features    # optimized build
cargo test --workspace --no-default-features                # full suite
cargo test -p wifi-densepose-engine                         # 13 integration tests
cargo bench -p wifi-densepose-engine                        # per-cycle latency
```

## Status (honest)

Integrated and validated end-to-end: ADR-135/136/137/138/139/141/142/143 via the
`wifi-densepose-engine` composition root. Forward-looking / pending: live 20 Hz
sensing-server loop wiring, UWB hardware (ADR-144), and RF-encoder model training
(ADR-146). Each GitHub issue (#840–#850) lists what is *Built* vs *Integration glue*.
