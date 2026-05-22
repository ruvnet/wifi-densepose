# R6.2.3 — Chest-centric placement: +27 pp coverage gain for vital-signs cogs

**Status:** chest-vs-body placement benchmark · **2026-05-22**

## Premise

R6.1 showed the chest contributes **27.6% of CSI energy** — 5× the per-limb value — and that limbs are *confound, not signal* for breathing-rate detection. R6.2 / R6.2.1 / R6.2.2 treated target zones as full body footprint (full bed, full chair, full standing zone). R6.2.3 asks: **does targeting the chest specifically change the optimal placement?**

If chest-centric and body-centric produce the same placement, the cog-time DSP work (limb masking in `vital_signs.rs`) suffices. If they differ, R6.2's CLI tool needs a `--cog vital-signs` flag that switches target-zone definitions.

## Method

Same 5×5 m bedroom search as R6.2, but with two zone definitions:

**Body-centric** (R6.2 default):
- bed: 1.5×0.5 → 3.5×2.0 m (3.00 m²)
- chair: 3.5×3.5 → 4.3×4.3 m (0.64 m²)
- desk: 0.2×2.5 → 1.2×3.1 m (0.60 m²)

**Chest-centric** (R6.2.3 new):
- bed_chest: 60×40 cm patch where the chest sits while lying (2.2-2.8, 0.8-1.2)
- chair_chest: 40×40 cm patch on the seat (3.7-4.1, 3.7-4.1)
- desk_chest: 40×20 cm patch above the desk (0.5-0.9, 2.7-2.9)

Same antenna candidate grid, same greedy search.

## Result

| Configuration | Coverage | Best Tx | Best Rx | Link |
|---|---:|---:|---:|---:|
| Body-centric (R6.2) | 49.3% | (4.25, 0) | (0, 3.25) | 5.35 m |
| **Chest-centric (R6.2.3)** | **82.4%** | (2.0, 0) | (4.5, 5) | 5.59 m |

Cross-evaluation:

| Apply to | Body-centric placement | Chest-centric placement |
|---|---:|---:|
| Body zones | 49.3% (its own optimum) | 40.3% (-9.0 pp) |
| Chest zones | 55.5% | **82.4%** (+26.9 pp) |

**Chest-targeting wins by +26.9 pp** on chest zones; body-targeting wins by +9.0 pp on body zones. The two strategies are not equivalent — chest-centric is a genuinely different deployment recipe.

## Why the placement differs

The optimal placements:
- **Body-centric**: corner-to-corner-ish (4.25, 0) → (0, 3.25). Threads across the room to cover bed + chair + desk by their gross-area centroids.
- **Chest-centric**: diagonal (2.0, 0) → (4.5, 5). Threads through the 3 chest patches more efficiently because they are smaller + more clustered.

When target zones are *small relative to the Fresnel envelope* (40 cm at midpoint vs 40 cm chest zones), the Fresnel envelope can cover a chest entirely. When targets are *large* (3 m² bed), full coverage by a 40 cm envelope is impossible — the placement must compromise across the body's spatial extent.

Different geometry → different optimum.

## Per-cog placement recommendation surfaced

R6.2.3 says R6.2's CLI tool should add a `--target-mode` flag:

| `--target-mode` | Zone definition | Best cog use |
|---|---|---|
| `body` (default) | Full body footprint (current R6.2) | `cog-person-count`, `cog-pose-estimation`, `cog-presence` |
| `chest` (new) | 40×40 cm chest patches | `cog-vital-signs`, `cog-breathing`, `cog-heart-rate` |
| `extremity` (future) | Hand / foot zones | Gesture detection cogs (out of scope for this loop) |

The placement-search engine is unchanged; only the target zones differ. ~20 LOC change to the existing R6.2 CLI.

## Composes with prior threads

- **R6.1** (multi-scatterer) — directly motivated this tick: chest = 27.6% of signal, limbs are confound.
- **R6.2 / R6.2.1 / R6.2.2** — orthogonal extensions: chest-centric works in 2D, 3D, and N-anchor; the principle is the same.
- **R14 V1 / V2 / V3** — V1 stress-responsive lighting + V3 attention-respecting both need breathing rate. **Both should use `--target-mode=chest`** at installation time. V2 HVAC uses presence + breathing → mixed mode (chest for breathing, body for presence). R6.2.3 says: configure the placement per cog deployed.
- **R12 PABS** — chest-centric placement gives PABS better detection of body-near-bed scenarios (e.g. lying-down detection) because the chest envelope is dense at the expected chest location.

## Honest scope

- **Chest position is approximated** — humans don't sit / lie at fixed coordinates. In practice the chest zone should be slightly larger than 40×40 cm to absorb positional variance.
- **Per-cog zone schema** is a deployment-time question, not a research one. The CLI option is the actionable output of this tick.
- **2D still** — chest height (z=1.0-1.5 m for standing, 0.5-0.8 m for sitting, 0.2-0.4 m for lying) was implicit. A 3D chest-centric search (composing R6.2.1 + R6.2.3) would refine the placements further. Estimated +3-5 pp.
- **Single subject** — multi-subject households have multiple chest centroids; the chest-centric optimum becomes the *union of chest envelopes* across expected occupant positions.

## What this DOES enable

1. **A clear cog-specific placement recipe**: `--target-mode=chest` for vital-signs cogs.
2. **Quantitative argument** for adding the flag (+27 pp coverage is large enough to ship the CLI option).
3. **Confirmation that R6.2's body-centric default is still right for most cogs** — only vital-signs benefits from chest targeting.

## What this DOES NOT enable

- Multi-subject chest unions (out of scope for this tick).
- 3D chest-centric (R6.2.1 + R6.2.3 composition, future).
- Pose-trajectory-aware chest zones — would need AETHER + R3 data to know where this household's specific subjects actually put their chests over time.

## Next ticks

- **R6.2.3.1**: 3D chest-centric placement (compose with R6.2.1).
- **R6.2.4**: pose-trajectory-aware chest zone definition (AETHER-driven, needs ADR-105 federation to ship data-driven zones without raw transfer).
- **R6.2 CLI productisation**: add `--target-mode={body,chest}` flag.

## Connection back

- **R5 / R6 / R6.1** — physical basis; R6.1's chest dominance directly motivates this tick.
- **R6.2 / R6.2.1 / R6.2.2** — orthogonal extensions; R6.2.3 is a cog-mode option that composes with all three.
- **R14** (V1 lighting / V3 attention) — both should use chest mode.
- **R12 PABS** — placement-driven detection sensitivity improves with chest-centric targeting for body-position-detection scenarios.
- **ADR-104 (ruview-mcp + ruview-cli)** — `--target-mode` is a new CLI arg + a new MCP tool argument.
