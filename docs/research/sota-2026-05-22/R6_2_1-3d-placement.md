# R6.2.1 — 3D antenna placement: ceiling-only mounting is the WORST option

**Status:** 3D Fresnel ellipsoid + height-strategy benchmark · **2026-05-22**

## Counter-intuitive headline

| Strategy | Coverage of 3 zones |
|---|---:|
| Desk-height (0.8 m, walls) | 22.2% |
| Wall-mount (1.5 m, walls) | 17.4% |
| **Ceiling-only (2.5 m, full ceiling grid)** | **0.0%** |
| **Mixed (any height, walls + ceiling)** | **25.7%** ← best |

Ceiling-only mounting **completely fails** — the Fresnel envelope sits at ceiling height (2.1-2.9 m) and never reaches floor-level targets (bed 0.3-0.6 m, chair 0.5-1.2 m, standing 1.0-1.7 m).

## The physics

In 3D the first Fresnel zone is a prolate ellipsoid with foci at Tx and Rx. The transverse radius at the midpoint is `sqrt(d·λ)/2`. For a 5 m link at 2.4 GHz: **39 cm transverse**. This is a *symmetric envelope around the LOS line*.

A ceiling-mounted link (Tx at 2.5 m, Rx at 2.5 m, horizontal LOS) has its Fresnel envelope vertically centred at 2.5 m, extending from 2.1 m to 2.9 m. Targets at 0.3-1.7 m are **below the envelope by 0.4-2.0 m**. Completely missed.

This is the 3D extension of the **on-LOS-degeneracy** finding from R6.1 — except now the issue is on-CEILING degeneracy. A flat horizontal link at any height blocks sensing in the perpendicular dimension.

## Why mixed wins

The optimal mixed placement picks Tx at (5.0, 4.0, 0.8) — desk height — and Rx at (0.0, 4.0, 1.5) — wall-mount height. The link is **diagonal in z** as well as x. The Fresnel ellipsoid is tilted to thread multiple elevations: covers chair (z=0.5-1.2) AND standing zone (z=1.0-1.7) AND a portion of bed (z=0.3-0.6).

**Vertical link diversity is the key 3D insight that 2D analysis missed.**

## Recommendations

| Use case | 3D placement recipe |
|---|---|
| Single Tx-Rx pair | One low (desk height ~0.8m), one high (wall ~1.5m), opposite walls |
| 4-anchor multistatic (R6.2.2) | 2× low corners + 2× high opposite corners |
| 5-anchor (R6.2.2 knee) | Mix of 0.8 m / 1.5 m / one ceiling at 2.5 m for top-down coverage |
| Bed-only (sleep monitoring) | Both antennas low (0.5-0.8 m) and **opposite sides of bed** |
| Standing-only (gym, kitchen) | Both antennas high (1.5 m) |
| **NEVER** | Both antennas ceiling-mounted with no low-anchor |

## What this says about the installation guide

Current RuView installer instructions are 2D: "place seeds on opposite walls". The 3D scrutiny says:

1. **Heights matter as much as horizontal positions.** Mixed-height placement gives +15.8% coverage over desk-height-only.
2. **Ceiling-mount fails alone.** If using ceiling as part of a multi-anchor configuration, MUST also have at least one low-height anchor to bring the envelope down to floor-level targets.
3. **Bedside sensing wants low anchors.** A bed at 0.3-0.6 m can only be covered by low-height links. High-mounted antennas miss the bed entirely.

These should be added to the installer-guide as **height recipes**, alongside R6.2's horizontal-placement recipes.

## Composes with prior threads

- **R6.2** (2D placement) — 2D analysis hides height issues entirely; R6.2 alone gives wrong installer guidance.
- **R6.2.2** (N-anchor multistatic) — N=5 anchors should be distributed across heights, not all at one elevation.
- **R6.1** (multi-scatterer) — the multi-scatterer body model is 2D top-down; a 3D body model (head at z=1.7, chest at z=1.3, legs at z=0.5) would tighten the per-body-part contribution estimates per height.
- **R14** (empathic appliances) — V1 lighting (bedroom: detect sleeper) needs low anchors. V3 (cognitive load at desk) needs mid-height. The placement strategy depends on the empathic-appliance use case.
- **ADR-029** (multistatic) — anchor-count + placement-height are both required configuration parameters.

## Honest scope

- **Coverage numbers (22%, 17%, 26%) are lower than R6.2's 2D 51%** because targets are 3D *volumes* now, not 2D *areas*. Volumetric coverage is inherently lower; a 3D point must be inside the ellipsoid in all three axes.
- **3 zones at distinct heights.** Real rooms have continuous human occupancy distributions (people stand, sit, lie); the 3-zone setup is a discrete approximation.
- **Single-pair only.** Multi-anchor 3D (R6.2.2.1) would saturate much earlier than the 2D version because each anchor's ellipsoid is sparser in 3D.
- **No furniture occlusion** in 3D either.
- **0.1 m resolution.** Finer resolution would refine the numbers slightly.
- **Greedy single-pair search.** Global optimum may be slightly higher; brute-force is feasible at this candidate count.

## What this DOES enable

1. **Updates the installation-guide recipe** from "place on opposite walls" to "place at mixed heights on opposite walls".
2. **Quantifies why ceiling-only WiFi sensing doesn't work** — common mistake in DIY deployments.
3. **Provides height-strategy recommendations per use case** (sleep / sitting / standing).
4. **A 3D placement search** that can be added to `wifi-densepose plan-antennas` as a `--3d` flag.

## What this DOES NOT enable

- Continuous occupancy distribution modelling (would need pose-trajectory data, R6.2.3).
- Multi-pair 3D optimisation (R6.2.2.1 — composition with R6.2.2 in 3D).
- Furniture / wall occlusion modelling (would need a 3D ray-tracing extension).
- Per-empathic-appliance optimised placement (would need V1/V2/V3 task-specific zones).

## Next ticks (R6.2 family)

- **R6.2.2.1**: 3D multi-anchor union coverage — does the 5-anchor knee hold in 3D?
- **R6.2.3**: chest-centric target zones (R6.1 says chest is 27.6% of signal — placement should target chest specifically).
- **R6.2 productisation**: add `--3d` flag to the CLI tool.

## Connection back

- **R6** Fresnel forward model — direct 3D extension.
- **R6.1** multi-scatterer — needs a 3D body model to compose properly with R6.2.1.
- **R6.2** — 2D was incomplete; height matters as much as horizontal position.
- **R6.2.2** — N-anchor knee likely shifts in 3D; needs follow-up benchmark.
- **R14** V1/V2/V3 — each vertical needs its own height-recipe.
- **ADR-029** — anchor placement specification needs (x, y, z) per anchor, not (x, y).
- **R12 PABS** — PABS sensitivity to structural changes inherits R6.2.1's coverage; mixed-height placements detect intruders standing AND sitting AND lying.
