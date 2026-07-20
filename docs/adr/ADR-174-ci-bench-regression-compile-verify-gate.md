# ADR-174: CI Bench-Regression Gate (Compile-Verify)

| Field | Value |
|-------|-------|
| **Status** | Accepted — implemented, caught one real bit-rotted bench |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **BENCH-GATE** |
| **Milestone** | benchmark/optimization re-balance — sub-deliverable 8.3 |
| **Motivated by** | `docs/research/sota-nn-train-benchmark-brief.md` (target 3: criterion benches as CI regression baselines) |

## Context

The v2/ workspace ships **26 criterion benches across 18 crates** (e.g.
`nvsim/pipeline_throughput`, `wifi-densepose-ruvector/{ann,sketch,fusion}_bench`,
`wifi-densepose-signal/{signal,dsp_perf,features,calibration,cir,…}_bench`,
`wifi-densepose-mat/detection_bench`, `wifi-densepose-nn/{inference,native_conv}_bench`,
`wifi-densepose-engine/engine_cycle`, …). Because **benches are not part of
`cargo test`**, nothing in CI compiled them — so they bit-rot silently the moment
a public API they call changes, and the rot is invisible until someone manually
runs `cargo bench` months later.

The SOTA brief named "wire existing criterion benches into CI as regression
baselines" as a concrete benchmark-hygiene target. The honest difficulty: true
*timing*-regression gating on shared GitHub runners is unreliable — wall-clock
varies 2–3× run-to-run (a captured 10-sample run showed `float_l2/512` ranging
307–444 ns), so a hard threshold or a cross-runner `criterion --baseline` compare
(baseline and PR land on different physical machines) would manufacture false
regressions. A gate that cries wolf gets disabled.

## Decision

Add `.github/workflows/bench-regression.yml` with **two jobs of explicitly
different authority** — and do NOT pretend to gate on timing.

### `bench-compile` — HARD GATE (real regression detection)
`cargo bench --workspace --no-default-features --no-run` compiles + links every
default-feature bench (no measurement → fully deterministic), plus a
`--features cir` compile of the gated `cir_bench`. Benches aren't in `cargo test`,
so this is the genuine guard: **the build fails the moment a bench stops
compiling.**

### `bench-fast-run` — INFORMATIONAL (`continue-on-error: true`, never gates)
Runs a curated pure-CPU subset (`nvsim/pipeline_throughput`,
`ruvector/{sketch,fusion}_bench`) in criterion quick-mode (1 s warm-up / 2 s
measure / 10 samples), targeted per-`--bench`, and uploads logs as an artifact.
Every number it produces is **informational only** — explicitly stated in the
workflow header.

### What is NOT done, and why (honest scope)
No timing-regression gate, no committed baseline JSON. The workflow header
documents the exact condition under which true timing-gating becomes honest: a
frequency-pinned **self-hosted** runner with a generous (>2×) floor. A
cross-runner baseline would be dishonest, so none is committed.

### Proof it matters (MEASURED)
Running the new gate on the current tree immediately caught
`wifi-densepose-mat/detection_bench` failing to compile:
`error[E0063]: missing field last_rssi in initializer of SensorPosition` — the
struct gained a field; the bench was never updated. **Fixed** in the same change
(`last_rssi: None`, the simulated-zone convention) and re-verified
(`cargo bench -p wifi-densepose-mat --no-default-features --bench detection_bench --no-run`
→ `Finished`). The gate paid for itself on its first run.

### Exclusions (documented in-workflow)
- `ruvector/crv_bench` — its crates.io dep `ruvector-crv 0.1.1` fails to build on
  stable (upstream `E0308` in `stage_iii.rs`); excluded with a re-add condition.
- `onnx_bench` / `mqtt_throughput` — feature-gated (ort / mqtt), left to their
  crates' own workflows. `wasm-edge/process_frame_bench` — workspace-excluded.

Conventions mirror existing workflows: `submodules: recursive` (the workspace
path-deps `vendor/rufield`), Swatinem/rust-cache `workspaces: v2`, Tauri/GTK apt
deps (a `--workspace` bench link pulls the whole graph), path-filtered triggers.

## Validation

- **Bit-rot caught + fixed** (above), re-verified `--no-run`.
- **MEASURED locally** (`--no-default-features`, Windows): nvsim, ruvector
  (sketch/fusion/ann), signal/cir_bench, mat/detection_bench (post-fix),
  vitals, ruview-swarm/swarm_bench all compile; fast subset runs (`nvsim
  pipeline_run/d1/256` ≈ 55 µs; `ruvector sketch_hamming` ≈ 3–7 ns vs `float_l2`
  ≈ 63–371 ns).
- `cargo test -p wifi-densepose-mat --no-default-features` → 166/6/2 passed, 0 failed.
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash
  `f8e76f21…46f7a` unchanged.
- **Honest limitation:** the full `--workspace --no-run` could not be
  end-to-end validated on this Windows box (`desktop` needs GTK, `candle-core`
  fails on MSVC, `swarm_bench` LTO-links OOM under parallel pressure — all
  Windows-env artifacts; each affected bench compiles standalone here). **The
  first green Linux CI run on the PR is the authoritative proof of the
  `--workspace` step.**

## Consequences

### Positive
- Bench bit-rot is now a hard CI failure, not a silent surprise — the 26 benches
  stay compilable as the APIs they exercise evolve.
- The benchmark-infrastructure half of the DoD (step 5) is satisfied honestly,
  setting up the next sub-deliverable (QAT-int8 measurement) to be
  regression-protected.

### Negative / Neutral
- No automated timing-regression detection (deliberate — see scope). Revisit only
  with a frequency-pinned self-hosted runner.
- One bench (`crv_bench`) excluded pending an upstream dep fix.

## Links
- ADR-173 — metric-locked accuracy harness (sub-deliverable 8.1)
- `docs/research/sota-nn-train-benchmark-brief.md` — motivating target
- ADR-134 (CIR), ADR-135 (calibration), ADR-154 (signal DSP benches) — benched paths
