# Tick 21 — 2026-05-22 08:10 UTC

**Thread:** R6.2.1 (3D antenna placement extension)
**Verdict:** Counter-intuitive finding — **ceiling-only mounting gives 0% coverage**. Mixed-height (one low, one high) gives the best result.

## What shipped

- `examples/research-sota/r6_2_1_3d_placement.py` — pure-numpy 3D Fresnel ellipsoid placement search.
- `examples/research-sota/r6_2_1_3d_results.json` — strategy comparison.
- `docs/research/sota-2026-05-22/R6_2_1-3d-placement.md` — research note.

## Headline strategy comparison

3D room (5×5×2.5 m), three 3D target zones (bed at z=0.3-0.6, chair at z=0.5-1.2, standing at z=1.0-1.7):

| Strategy | Coverage |
|---|---:|
| Desk-height (0.8 m walls) | 22.2% |
| Wall-mount (1.5 m walls) | 17.4% |
| **Ceiling-only (2.5 m grid)** | **0.0%** |
| **Mixed walls + ceiling** | **25.7%** ← best |

## The physics

Ceiling-only fails because both antennas at 2.5 m create a Fresnel ellipsoid sitting **at ceiling height** (2.1-2.9 m vertically). Target zones at 0.3-1.7 m are below the envelope by 0.4-2.0 m. The 39 cm transverse radius is symmetric around LOS, so a flat horizontal link at any height misses targets at any other height.

**This is the 3D version of R6.1's on-LOS-degeneracy finding.** A horizontal link at any single height has its envelope concentrated at that height.

## Why mixed wins

Best placement: Tx at (5.0, 4.0, 0.8) desk-height + Rx at (0.0, 4.0, 1.5) wall-mount. The **diagonal-in-z** link tilts the ellipsoid through multiple elevations. Covers chair AND standing AND bed simultaneously.

**Vertical link diversity is the 3D insight 2D analysis missed.**

## Installation-guide updates

| Use case | Recipe |
|---|---|
| Single Tx-Rx pair | One low (0.8 m), one high (1.5 m), opposite walls |
| 4-anchor R6.2.2 | 2× low corners + 2× high opposite corners |
| 5-anchor knee | Mix 0.8 / 1.5 / one ceiling (2.5) for top-down |
| Bed-only sleep monitoring | Both LOW (0.5-0.8 m), opposite sides of bed |
| Standing-only (gym, kitchen) | Both HIGH (1.5 m) |
| **NEVER** | Both ceiling without low anchor |

## Why coverage numbers are lower than R6.2's 51%

3D target zones are *volumes*, not 2D *areas*. A point must be inside the ellipsoid in all 3 axes. Volumetric coverage is inherently lower; the 22-26% range is honest 3D physics.

## Composes with prior threads

- **R6.2** (2D) — incomplete; height matters as much as horizontal
- **R6.2.2** (N-anchor) — N=5 knee should distribute across heights
- **R6.1** multi-scatterer — needs 3D body model (head/chest/legs at different z) for proper composition
- **R14** V1/V2/V3 — each vertical needs height-recipe specific to its sensing zone
- **ADR-029** — anchor placement is (x, y, z), not (x, y)
- **R12 PABS** — sensitivity to intruders inherits the coverage; mixed-height detects standing/sitting/lying intruders alike

## Honest scope

- 3-zone discrete approximation of continuous human occupancy
- Single-pair only; multi-anchor 3D = R6.2.2.1 (next)
- No furniture occlusion
- 0.1 m resolution
- Greedy single-pair search (brute-force feasible at this scale)

## Coordination

`ticks/tick-21.md`. No PROGRESS.md edit. Branch `research/sota-r6.2.1-3d-placement`.

## Remaining work

- **R6.2.2.1**: 3D N-anchor union coverage
- **R6.2.3**: chest-centric zones (per R6.1 chest = 27.6% of signal)
- **R12.1**: pose-PABS closed loop
- **ADR-107**: cross-installation federation

~3.8h to cron stop. **21 ticks landed.** Loop covered R1-R15 + 2 ADRs + 6 deferred follow-ups + 3 negative-result categorisations.
