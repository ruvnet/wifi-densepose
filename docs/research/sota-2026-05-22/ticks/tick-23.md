# Tick 23 — 2026-05-22 08:33 UTC

**Thread:** R6.2.3 (chest-centric placement)
**Verdict:** Chest-centric targeting gains **+26.9 pp coverage** vs body-centric for vital-signs cogs. R6.2's CLI needs a `--target-mode=chest` flag.

## What shipped

- `examples/research-sota/r6_2_3_chest_centric.py` — pure-numpy chest-vs-body placement benchmark.
- `examples/research-sota/r6_2_3_chest_centric_results.json` — full benchmark.
- `docs/research/sota-2026-05-22/R6_2_3-chest-centric-placement.md` — research note.

## Headline

5×5 m bedroom, same antenna candidate grid, two zone definitions:

| Configuration | Coverage | Best placement |
|---|---:|---|
| Body-centric (R6.2 default) | 49.3% | (4.25, 0) ↔ (0, 3.25), 5.35 m |
| **Chest-centric (R6.2.3 new)** | **82.4%** | (2.0, 0) ↔ (4.5, 5), 5.59 m |

Cross-eval:
- Body-optimal applied to chest zones: 55.5%
- **Chest-targeting gain on chest zones: +26.9 pp**
- Chest-optimal applied to body zones: 40.3% (-9.0 pp)

The two strategies are **not equivalent**. Different cogs want different placements.

## Per-cog deployment recommendation surfaced

| `--target-mode` | Zones | Best cog use |
|---|---|---|
| `body` (default) | Full body footprint | cog-person-count, cog-pose-estimation, cog-presence |
| `chest` (new) | 40×40 cm chest patches | cog-vital-signs, cog-breathing, cog-heart-rate |
| `extremity` (future) | Hand/foot zones | Gesture detection (not in scope) |

Same engine, different zones. ~20 LOC change to R6.2 CLI.

## Why placements differ

- **Body-centric** threads across the room to compromise across 3 m² bed + chair + desk by gross-area centroids.
- **Chest-centric** threads more efficiently through the 3 small chest patches because targets fit inside the Fresnel envelope.

When target ≈ envelope width, the envelope can cover it entirely. When target >> envelope, placement is forced to compromise.

## R14 vertical-specific recommendation

- V1 stress-responsive lighting: needs breathing rate → `chest` mode
- V2 adaptive HVAC: presence + breathing → mixed (placement for chest, additional anchors for presence)
- V3 attention-respecting conversational: shallow-breathing recovery → `chest` mode

R6.2.3 surfaces a per-cog config that empathic-appliance products need at install time.

## Composes with prior threads

- **R6.1 motivated this tick**: chest = 27.6% of signal, limbs are confound
- **R6.2 / R6.2.1 / R6.2.2** — orthogonal: chest-centric works in 2D, 3D, N-anchor
- **R14 V1/V3** — should use chest mode
- **R12 PABS** — chest-centric placement improves body-position-detection scenarios

## Honest scope

- Chest positions approximated (humans don't sit/lie at fixed coords)
- 2D still; 3D chest-centric = R6.2.3.1 follow-up (~+3-5 pp expected)
- Single subject; multi-subject = union of chest envelopes
- Per-cog zone schema is deployment-time, not research-time

## Coordination

`ticks/tick-23.md`. No PROGRESS.md edit. Branch `research/sota-r6.2.3-chest-centric`.

## Remaining work

- R6.2.3.1: 3D chest-centric (R6.2.1 + R6.2.3 compose)
- R6.2.4: pose-trajectory-aware chest zones (needs AETHER + ADR-105 federation)
- R12.1: pose-PABS closed loop
- R3.2: embedding-level physics-informed env (from R3.1's corrected sketch)
- ADR-108: Kyber substitution

~3.4h to cron stop. **23 ticks landed.** Loop now has 13 research threads + 3 ADRs + 8 deferred follow-ups closed.
