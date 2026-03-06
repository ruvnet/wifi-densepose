# ADR-051: Sensing Server Decomposition — main.rs God Object Breakup

| Field | Value |
|-------|-------|
| Status | Proposed |
| Date | 2026-03-06 |
| Deciders | ruv |
| Depends on | ADR-050 (Quality Engineering — Sprint 2) |
| Issue | [#174](https://github.com/ruvnet/RuView/issues/174) |

## Context

`sensing-server/src/main.rs` is 3,765 lines with cyclomatic complexity ~65. It contains 12 structs, 60+ functions, 10 constants, and a 37-field `AppStateInner` god object. This violates the project's 500-line file limit (CLAUDE.md) and makes unit testing individual components impossible.

The file mixes concerns:
- CLI argument parsing and server bootstrap
- HTTP route handlers (health, models, recordings, training, pose, vitals)
- WebSocket upgrade and client management
- UDP CSI frame receiver and parser
- Signal processing pipeline (feature extraction, classification, smoothing)
- Simulated data generator
- Windows WiFi scanning integration
- Pose estimation from WiFi signals
- Vital sign smoothing and filtering
- Model/recording file management

## Decision

Decompose `main.rs` into 14 focused modules. Each module owns its types, constants, and functions. `main.rs` retains only CLI parsing, state initialization, router construction, and server startup (~250 lines).

### Module Extraction Plan

| Module | Source Lines | Contents | Target Size |
|--------|-------------|----------|-------------|
| `cli.rs` | 59-152 | `Args` struct, CLI parsing | ~100 |
| `state.rs` | 154-370 | `AppStateInner`, all DTOs (`Esp32Frame`, `SensingUpdate`, `NodeInfo`, etc.), `SharedState` type alias | ~220 |
| `signal.rs` | 542-890 | `generate_signal_field()`, `estimate_breathing_rate_hz()`, `compute_subcarrier_variances()`, `extract_features_from_frame()`, `raw_classify()` | ~350 |
| `smoothing.rs` | 886-1060 | Classification smoothing, vital sign smoothing, `trimmed_mean()`, constants | ~180 |
| `routes_health.rs` | 1660-2005 | `/health/*`, `/api/v1/info` endpoints | ~350 |
| `routes_model.rs` | 2058-2230 | `/api/v1/models/*`, LoRA profiles, `scan_model_files()` | ~180 |
| `routes_recording.rs` | 2233-2440 | `/api/v1/recording/*`, `scan_recording_files()` | ~210 |
| `routes_training.rs` | 2443-2560 | `/api/v1/train/*`, `/api/v1/adaptive/*` | ~120 |
| `routes_sensing.rs` | 2562-2710 | Vital signs, edge vitals, WASM events, model info, SONA endpoints | ~150 |
| `routes_pose.rs` | 1701-1930, 2007-2055 | Pose estimation, `derive_single_person_pose()`, pose/stats/zones endpoints | ~280 |
| `websocket.rs` | 1492-1660 | WS upgrade handlers, `handle_ws_client()`, `handle_ws_pose_client()` | ~170 |
| `udp_receiver.rs` | 2725-2890 | UDP CSI frame receiver task, frame parsing | ~170 |
| `data_sources.rs` | 1063-1465, 2888-3020 | Windows WiFi task, simulated data task, `probe_windows_wifi()`, `parse_netsh_interfaces_output()` | ~400 |
| `router.rs` | (new) | `build_router()` function assembling all routes | ~80 |

### Extraction Order (6 Phases)

1. **Phase 1**: `cli.rs` + `state.rs` — Zero behavioral change, just move types
2. **Phase 2**: `signal.rs` + `smoothing.rs` — Pure functions, easy to test
3. **Phase 3**: `routes_health.rs` + `routes_model.rs` + `routes_recording.rs` — Stateless-ish handlers
4. **Phase 4**: `routes_training.rs` + `routes_sensing.rs` + `routes_pose.rs` — Remaining HTTP handlers
5. **Phase 5**: `websocket.rs` + `udp_receiver.rs` + `data_sources.rs` — Async tasks
6. **Phase 6**: `router.rs` — Assemble all routes, slim `main.rs` to ~250 lines

### State Refactoring

`AppStateInner` (37 fields) will be split into domain-specific sub-states:

```rust
pub struct AppStateInner {
    pub config: ServerConfig,        // CLI args, ports, paths
    pub sensing: SensingState,       // CSI frames, features, classification
    pub vitals: VitalsState,         // Vital sign buffers, smoothing state
    pub models: ModelState,          // Active model, discovered models, LoRA
    pub recording: RecordingState,   // Active recording, file handles
    pub training: TrainingState,     // Training status, adaptive model
    pub pose: PoseState,             // Person detections, pose history
    pub broadcast_tx: broadcast::Sender<SensingUpdate>,
}
```

## Consequences

### Positive

- Each module is independently unit-testable
- No file exceeds 500 lines
- Domain boundaries are explicit (state sub-structs)
- New developers can find code by domain
- Merge conflict surface reduced (parallel module edits)

### Negative

- Large refactor with ~3,700 lines touched — high merge conflict risk
- `pub(crate)` visibility needed for cross-module state access
- Some functions share mutable state, requiring careful `&mut` threading

### Neutral

- No behavioral change — all endpoints, WebSocket, UDP behavior stays identical
- Existing integration tests (if any) continue to pass unchanged

## Implementation Notes

1. Each phase is a separate commit for easy revert
2. Run `cargo test` and `cargo check` after each phase
3. Use `pub(crate)` for internal types, keep public API surface minimal
4. Add `#[cfg(test)] mod tests` to each new module with at least smoke tests
5. Consider adding `tower` middleware for auth (Sprint 1 remaining item) during Phase 3

## References

- ADR-050: Quality Engineering Response (Sprint 2 plan)
- Issue #170: Quality Engineering Analysis
- CLAUDE.md: 500-line file limit rule
