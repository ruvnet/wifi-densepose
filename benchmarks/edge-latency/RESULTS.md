# Edge-Latency Benchmark Results — ADR-163

Converting **CLAIMED** edge latency budgets into **MEASURED-on-host** numbers,
closing the measurement debt flagged by Milestones 5/6 (ADR-159 / ADR-160).
Benches + docs only — **no production-code behavior changed**.

## The honest caveat, up front (read before citing any number)

Two distinct gaps separate every number below from the figure it is converting:

1. **Host ≠ ESP32.** The wasm-edge skill modules document budgets *"on ESP32-S3
   WASM3"* (e.g. `exo_time_crystal`: "H (<10 ms)"). These benches run **native
   x86_64 on a development laptop**, not the Xtensa/WASM3 target. A native host
   median is an **upper bound on the algorithm's work**, not the ESP32 number.
   WASM3 interpretation on a ~240 MHz Xtensa core is typically 1–2 orders of
   magnitude slower than native `-O` host code, so a host median far under the
   budget **does NOT prove the ESP32 meets it.** *The ESP32 figure is NOT
   reproduced here — it needs hardware.*

2. **Bench ≠ the doc-claimed measurement.** For the cogs, the manifest cites a
   **cold-start** number (`cold_start_ms_avg`, weight-load included); these
   benches measure **steady-state** per-frame `infer` (warm, weights resident).
   Different measurements; we report both, labelled.

Grades (per `benchmarks/wiflow-std/RESULTS.md` / ADR-152 vocabulary):
- **MEASURED-on-host** — reproduced in this repo on the machine below, exact
  command recorded. NOT the ESP32 / NOT the cold-start figure.
- **CLAIMED (ESP32)** — the doc budget; UNMEASURED on hardware here.

## Machine

| | |
|---|---|
| Host | `ruvzen` (Windows 11, this dev box) |
| CPU | Intel Core Ultra 9 285H |
| Toolchain | `cargo 1.91.1`, `--release` (opt-level per crate profile) |
| Bench harness | criterion 0.5 (`time: [low **median** high]` reported below) |
| Date | 2026-06-12 |

Run-to-run spread on this box is non-trivial (criterion's low/high bracket the
median by a few %); the medians below are single-session captures with the smoke
settings `--warm-up-time 1 --measurement-time 2` (wasm-edge) / `3` (cogs). Re-run
for your own machine — the absolute numbers are host-specific.

---

## T1 — wasm-edge `process_frame` hot paths (ADR-160 deferred item → DONE host)

The crate is **excluded from the v2 workspace**; bench from the crate dir.

```bash
cd v2/crates/wifi-densepose-wasm-edge
cargo bench --features std -- --warm-up-time 1 --measurement-time 2
# med_seizure_detect is medical-experimental-gated:
cargo bench --features std,medical-experimental -- --warm-up-time 1 --measurement-time 2 med_seizure
```

| Hot path (M6-audit-named) | Bench id | Host median | Grade | Doc budget (CLAIMED, ESP32) |
|---|---|---|---|---|
| `exo_time_crystal` 256-pt × 128-lag autocorrelation (full buffer) | `exo_time_crystal::process_frame[autocorr_256x128]` | **17.3 µs** | MEASURED-on-host | "H (<10 ms) on ESP32-S3 WASM3" — **NOT reproduced here (needs hardware)** |
| `exo_ghost_hunter` empty-room periodicity + hidden-breathing | `exo_ghost_hunter::process_frame[empty_room_periodicity]` | **1.44 µs** | MEASURED-on-host | research/exotic; no firm ESP32 figure — host proxy only |
| `sec_weapon_detect` per-subcarrier Welford (MAX_SC=32) | `sec_weapon_detect::process_frame[per_sc_welford]` | **0.42 µs** (420 ns) | MEASURED-on-host | research-grade; calibration-gated — host proxy only |
| `med_seizure_detect` clonic-phase rhythm path (steady-state frame) | `med_seizure_detect::process_frame[clonic_rhythm]` | **0.10 µs** (105 ns) | MEASURED-on-host (feature-gated) | doc budget "S (<5 ms) on ESP32"; **NOT reproduced here** |

Reading these honestly:

- `exo_time_crystal` at **17.3 µs host** is the only one whose host cost is even
  in the same *thousandths* of its 10 ms ESP32 budget — it does the most work
  (~32K MACs/frame). 17.3 µs native says the algorithm is cheap; it says
  **nothing** about whether WASM3-on-Xtensa lands under 10 ms. A naïve
  host→ESP32 extrapolation (assume 100× interpreter+clock penalty) would put it
  near ~1.7 ms, comfortably under — **but that is an extrapolation, not a
  measurement**, and is recorded here only to show the host number is not
  obviously in tension with the budget. ESP32 figure: **UNMEASURED**.
- `med_seizure_detect`'s 105 ns is the **steady-state** per-frame cost; the
  expensive clonic autocorrelation only fires when the state machine is in the
  clonic phase, so this is a lower-bound on the heavy path, not the worst case.
  It is still a real, committed host datapoint.
- The pre-existing `tests/budget_compliance.rs` already asserts the L/S/H
  wall-clock tiers (25 passing tests); these criterion benches add the
  regression-grade, reproducible median that ADR-160 deferred.

---

## T2 — cog steady-state inference latency (ADR-159/160 deferred item → DONE)

Cog crates are normal workspace members; bench from `v2/`. Real weights
(`count_v1.safetensors` / `pose_v1.safetensors`) ship in-repo under each cog's
`cog/artifacts/`, so the bench measures the **real Candle CPU forward**, not the
stub (the bench `assert!`s `backend().starts_with("candle-")`).

```bash
cd v2
cargo bench -p cog-person-count  --no-default-features --bench infer_bench -- --warm-up-time 1 --measurement-time 3
cargo bench -p cog-pose-estimation --no-default-features --bench infer_bench -- --warm-up-time 1 --measurement-time 3
```

| Cog | Bench id | Host median (steady-state infer, CPU) | Grade | Manifest cold-start (CLAIMED, different measurement + machine) |
|---|---|---|---|---|
| cog-person-count | `cog_person_count::infer[cpu_real_weights_steady_state]` | **305 µs** (idle box) | MEASURED-on-host | — (person-count manifest carries comparable provenance) |
| cog-pose-estimation | `cog_pose_estimation::infer[cpu_real_weights_steady_state]` | **305 µs** (idle box) | MEASURED-on-host | `cold_start_ms_avg: 5.4` (30 invocations, **ruvultra/RTX 5080 host**, candle 0.9 cpu) — **cold-start, NOT steady-state; NOT this machine** |

> Spread caveat (observed, honest): both medians above were captured with the box
> otherwise idle. A re-run of the validate-form command *while a second cargo job
> was loading the same cores* gave 385 µs (person-count) / 973 µs (pose) —
> the criterion low/high bracket widens to ~0.34–1.18 ms under contention. The
> 305 µs figures are the idle-box datapoints; the absolute number is host- and
> load-dependent (the ~10× pose swing is core contention, not a code change).

Reading these honestly:

- **Steady-state ≠ cold-start.** The pose manifest's `5.4 ms` folds in one-time
  weight load / mmap / first-forward allocation. This bench warms the engine
  first and times only the recurring per-frame forward, on a *different
  machine*. The two numbers are not comparable and we do not claim this bench
  reproduces the 5.4 ms manifest figure.
- Both cogs share the same conv encoder; person-count adds a count head +
  confidence head, pose adds a 256-wide MLP head. The host steady-state cost is
  dominated by the three dilated Conv1d layers (56→64→128→128) shared by both —
  which is why both land at ~305 µs.
- **Empirical confirmation of the steady-state/cold-start gap:** pose
  steady-state (305 µs host) is ~18× *under* the manifest's 5.4 ms cold-start.
  Even accounting for the different machine, this is the expected shape — the
  bulk of cold-start is one-time setup, not the forward pass — and it is exactly
  why conflating the two would be dishonest.

---

## Status vs the deferred items

| Deferred item | Was | Now |
|---|---|---|
| ADR-160 "Criterion benches for `process_frame` budget claims" | ACCEPTED-FUTURE | **DONE (host)**; ESP32-on-hardware still **PENDING** (needs the wasm32 target + a flashed ESP32-S3) |
| ADR-159/160 cog inference latency (`cold_start_ms_avg` uncommitted-benched) | CLAIMED | **MEASURED-on-host (steady-state)**; cold-start-on-ruvultra remains the manifest's separate claim |

Nothing here changes runtime behavior — these are benches + this results file
only. No crate needs republishing.
