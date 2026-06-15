# ADR-163: Edge-Latency Measurement — CLAIMED budgets → MEASURED-on-host

- **Status**: accepted
- **Date**: 2026-06-12
- **Deciders**: ruv
- **Tags**: edge-latency, wasm-edge, esp32, cog-inference, criterion, prove-everything, measurement-debt
- **Amends**: ADR-160 (deferred "criterion benches for process_frame budget claims" line now DONE-on-host); ADR-159 (cog inference latency)

## Context — Milestone 9 of the beyond-SOTA sweep

Prior milestones (M5/M6, ADR-159/ADR-160) flagged **measurement debt**: edge
latency budgets asserted in doc-comments and manifests but **never reproduced by
a committed benchmark**. Specifically:

- Many `wifi-densepose-wasm-edge` skill modules document a timing budget *"on
  ESP32-S3 WASM3"* (e.g. `exo_time_crystal`: "H (heavy, <10 ms)"). These were
  **CLAIMED**, not benchmarked. ADR-160's deferred backlog named exactly this:
  *"Criterion benches for `process_frame` budget claims — ACCEPTED-FUTURE."*
- `cog-pose-estimation`'s manifest cites `cold_start_ms_avg: 5.4`, but neither
  cog had a `benches/` directory or any committed inference-latency number.

Under the project's **prove-everything / anti-"AI-slop"** directive, a CLAIMED
latency budget that a skeptic cannot reproduce is debt. M9 pays it down — benches
and docs only, **no production-code behavior change** (so nothing republishes).

## Headline

**Converted the CLAIMED edge-latency budgets into MEASURED-on-host numbers, with
the honest host-vs-ESP32 caveat stated everywhere.** Added committed criterion
benches over the heaviest hot paths and a results file a skeptic can re-run. The
ESP32-on-hardware figure remains explicitly **UNMEASURED** — this milestone does
not pretend a laptop reproduces an Xtensa/WASM3 budget.

## Decision — benches landed

### T1 — wasm-edge `process_frame` budget benches

`v2/crates/wifi-densepose-wasm-edge/benches/process_frame_bench.rs` (criterion,
`harness = false`, `required-features = ["std"]`). The crate is **excluded from
the v2 workspace**, so it runs from the crate dir. Benches the M6-audit-named
heaviest hot paths over a **fixed synthetic CSI frame**, each driven through the
public `process_frame` after warming the relevant ring/phase buffers so the
expensive path actually executes:

- `exo_time_crystal::process_frame` — full 256-pt × 128-lag autocorrelation.
- `exo_ghost_hunter::process_frame` — empty-room periodicity / hidden-breathing.
- `sec_weapon_detect::process_frame` — per-subcarrier (MAX_SC=32) Welford.
- `med_seizure_detect::process_frame` — clonic-rhythm path (`#[cfg(feature =
  "medical-experimental")]`, only built/run with that gate).

The lib's `bench = false` was set so the libtest harness does not intercept
criterion CLI flags; the `ghost_hunter` bin is already `standalone-bin`-gated and
not built under `--features std`.

**Measured host medians** (Intel Core Ultra 9 285H, native `--release`):
`exo_time_crystal` **17.3 µs** · `exo_ghost_hunter` **1.44 µs** ·
`sec_weapon_detect` **0.42 µs** · `med_seizure_detect` **0.10 µs**.

### T2 — cog inference latency benches

`v2/crates/cog-person-count/benches/infer_bench.rs` and
`v2/crates/cog-pose-estimation/benches/infer_bench.rs` (criterion,
`harness = false`). Each loads the **real** shipped weights from the in-repo
`cog/artifacts/`, asserts the Candle CPU backend (so the stub can never be
silently benched), warms one forward, then times steady-state
`InferenceEngine::infer` over a fixed CSI window on `Device::Cpu`.

**Measured host medians:** cog-person-count **305 µs** · cog-pose-estimation
**305 µs** (steady-state, CPU, real weights).

### T3 — results file

`benchmarks/edge-latency/RESULTS.md`, in the `benchmarks/wiflow-std/RESULTS.md`
style: each number with its exact reproduce command, the machine, the
MEASURED-on-host grade, and the honest caveat.

## The honest caveat (recorded, non-negotiable)

1. **Host ≠ ESP32.** The wasm-edge benches run native x86_64, not Xtensa/WASM3.
   A host median is an **upper bound on algorithm work**, not the ESP32 number;
   WASM3 interpretation on a ~240 MHz core is 1–2 orders of magnitude slower than
   native `-O`. A host median under budget does **not** prove the ESP32 meets it.
   **The ESP32 figure is NOT reproduced here — it needs hardware.**
2. **Bench ≠ the doc-claimed measurement.** The cogs' manifest cites a
   **cold-start** number (weight-load included); these benches measure
   **steady-state** per-frame `infer`. We report both, labelled, and do not
   conflate them. Empirically, pose steady-state (305 µs host) is ~18× under the
   5.4 ms cold-start — the expected shape, and exactly why conflating would lie.

## Deferred / still-pending (nothing dropped)

- **ESP32-on-hardware `process_frame` latency** — **PENDING (hardware)**. Needs
  the `wasm32-unknown-unknown` target built + flashed to an ESP32-S3 and timed
  under WASM3. The host bench is the algorithm-cost proxy until then.
- **Per-skill *accuracy*** remains **DATA-GATED** (unchanged from ADR-160) —
  this ADR measures latency only, never claims detection accuracy.

## Reproduction (MEASURED)

```bash
# T1 — wasm-edge (workspace-excluded → run from the crate dir)
cd v2/crates/wifi-densepose-wasm-edge
cargo bench --features std -- --warm-up-time 1 --measurement-time 2
cargo bench --features std,medical-experimental -- --warm-up-time 1 --measurement-time 2 med_seizure

# T2 — cogs (workspace members)
cd v2
cargo bench -p cog-person-count   --no-default-features --bench infer_bench
cargo bench -p cog-pose-estimation --no-default-features --bench infer_bench

# existing tests still green (behavior unchanged)
cargo test -p cog-person-count -p cog-pose-estimation --no-default-features
```

## Consequences

- ADR-160's deferred *"Criterion benches for `process_frame` budget claims"* line
  is now **DONE (host)**; the ESP32-on-hardware confirmation is explicitly the
  one remaining pending item.
- The cogs now ship committed, reproducible steady-state inference-latency
  numbers, cleanly distinguished from the manifest's cold-start claim.
- No runtime behavior changed; no crate republishes. `PROOF.md`'s performance
  table and `scripts/prove.sh`'s gated section reference the new benches.
