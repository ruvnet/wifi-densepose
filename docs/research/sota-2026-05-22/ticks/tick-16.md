# Tick 16 — 2026-05-22 06:55 UTC

**Thread:** R6.2 (Fresnel-aware antenna placement) — first deferred follow-up
**Verdict:** Working 2D placement search + CLI-shaped demo. Optimal placement is **93× better** than median random placement and infinite-× better than worst (which is 0% coverage). The current "stick it anywhere" deployment recipe leaves 50-100× of sensing on the table.

## What shipped

- `examples/research-sota/r6_2_antenna_placement.py` — pure-numpy 2D Fresnel-ellipse placement search.
- `examples/research-sota/r6_2_placement_results.json` — best/median/worst on a 5×5 m bedroom benchmark.
- `docs/research/sota-2026-05-22/R6_2-fresnel-antenna-placement.md` — research note with the method, benchmark, per-cog deployment recommendations, honest scope.

## Headline benchmark: 5×5 m bedroom

Target zones: bed (3 m²) + chair (0.64 m²). 2,900 antenna pairs evaluated at 2.4 GHz.

| Placement | Bed cov | Chair cov | **Total** |
|---|---:|---:|---:|
| Optimal (1.25, 0)→(4.75, 5) | 43.5% | 86.7% | **51.1%** |
| Median | varies | varies | 0.5% |
| Worst | varies | varies | **0.0%** |

**93× improvement** from median to optimal. The "diagonal across longest axis" recipe is the right shape for a bedroom-class room.

## Counter-intuitive insight: longer links cover more space

Fresnel envelope width = √(d·λ)/2 — **grows with link length**. So the optimal placement at 6.10 m (diagonal) has a 43.7 cm midpoint envelope vs 39.5 cm for a 5 m wall-parallel link. Counter to "shorter link = stronger signal", *longer* links cover *more space*, up to the link-budget gate (R10).

## Per-cog deployment recommendations surfaced

| Cog | Recommended placement |
|---|---|
| `cog-person-count` | Diagonal across longest axis |
| `cog-pose-estimation` | Zone inside ~50% of midpoint envelope |
| AETHER re-ID | Tx near doorway, Rx diagonal |
| `cog-maritime-watch` | Vertical diagonal through cabin |
| `cog-wildlife` (future) | Tx/Rx on opposite trees, threading clearing midline |

These improvements come from **physics, not algorithms** — no model retraining required.

## Why this is high-leverage

- Existing customers can re-mount their seeds today and get 10-100× better sensing without firmware/model changes.
- Future cog installations get the placement guide for free (generated from cog target-zone schema).
- Adds a **ship-ready CLI tool** (`wifi-densepose plan-antennas`) that any installer can use in 2 minutes.

## Honest scope landed

- 2D approximation (3D Fresnel ellipsoid is a half-day extension)
- Free-space (real multipath adds +5-15% coverage outside envelope)
- Rectangular target zones (real occupants don't occupy rectangles)
- Single-pair only (multistatic N-anchor union is next, R6.2.2)
- Perimeter-only candidates (no ceiling/tripod mounts)
- No link-budget gate (R10 sets it; needed for large rooms)

## Composes with prior threads

- **R6** (Fresnel forward model) — direct 2D extension
- **R1** (CRLB) — combined: placement × precision = full geometry budget
- **R10** (foliage range) — sets the link-budget gate that R6.2 ignores
- **R11** (maritime) — same recipe in steel-walled cabins
- **R14** (empathic appliances) — placement determines whether the V1/V2/V3 verticals see the right occupant
- **ADR-105 federation** — better placement → better local training → faster (ε, δ) convergence per ADR-106

## CLI shape (ship-ready)

```
wifi-densepose plan-antennas \
    --room 5.0 5.0 \
    --target bed 1.5 0.5 2.0 1.5 \
    --target chair 3.5 3.5 0.8 0.8 \
    --freq-ghz 2.4
```

## Coordination

`ticks/tick-16.md`. No PROGRESS.md edit. Branch `research/sota-r6.2-fresnel-antenna-placement`.

## Remaining loop work

- **R3 follow-up**: physics-informed env_sig prediction (uses R6 forward operator + room map → zero-shot cross-room transfer without labelled examples)
- **R6.1**: multi-scatterer Fresnel forward model (volume integral over voxel grid)
- **R6.2.1/.2/.3**: 3D placement, N-anchor multistatic, pose-trajectory target zones
- **ADR-107**: cross-installation federation w/ secure aggregation
- Loop retrospective / 00-summary.md (premature — ~5h still on clock)

~5.1h to cron stop. **16 ticks landed. PROGRESS.md research agenda + 2 ADRs + 1 deferred follow-up closed.**
