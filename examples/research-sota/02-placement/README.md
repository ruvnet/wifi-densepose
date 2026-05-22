# 02 — Antenna placement (R6.2 family, 7 sub-ticks)

The 9-tick R6.2 family productised R6's forward model into a working CLI-shaped placement search. Each script answers one axis of the placement question.

## Scripts (in development order)

| Script | Thread | Axis | Headline |
|---|---|---|---|
| `r6_2_antenna_placement.py` | R6.2 | 2D single-pair | Optimal placement is **93× better** than median random; corner-to-corner diagonal |
| `r6_2_1_3d_placement.py` | R6.2.1 | 3D single-pair | **Ceiling-only mounting fails (0% coverage)** — Fresnel envelope sits at ceiling, never reaches floor targets |
| `r6_2_2_multistatic_placement.py` | R6.2.2 | 2D N-anchor | Knee at **N=5** anchors for typical bedroom (96.8% body-zone coverage) |
| `r6_2_2_1_3d_multistatic.py` | R6.2.2.1 | 3D N-anchor | **2D knee disappears in 3D** — only 49% at N=5 with body zones |
| `r6_2_3_chest_centric.py` | R6.2.3 | 2D chest-centric | +27 pp coverage gain when targeting chest specifically (vital-signs cog) |
| `r6_2_4_3d_chest_multistatic.py` | R6.2.4 | 3D chest-centric | Recovers 3D shortfall — **77% at N=5, 82% at N=6 chest-centric** |
| `r6_2_5_multi_subject.py` | R6.2.5 | Multi-subject union | **100% coverage for 1-4 occupants at N=5** chest-centric |

## Decision matrix (final ADR-113 output)

| Cog category | Dimension | Zone mode | Occupants | N | Heights | Coverage |
|---|---|---|---:|---:|---|---:|
| Vital signs | 2D | chest | 1-4 | 5 | walls 0.8/1.5 m | **100%** |
| Vital signs | 3D | chest | 1-4 | 6 | walls 0.8/1.5 (NO ceiling) | 82% |
| Pose estimation | 2D | body | 1-2 | 5 | walls mixed | 97% |
| Pose estimation | 3D | body | 1-2 | 7-8 | mixed L/M/H | 65%+ |
| Person count | 2D | body | 1-4 | 4 | walls mixed | 86% |

## Counter-intuitive findings

1. **Longer links cover more space.** Fresnel envelope width = √(d·λ)/2 grows with d.
2. **Ceiling-only fails entirely** (R6.2.1) — both anchors at 2.5 m put envelope at ceiling height, target zones below are missed.
3. **2D N=5 knee doesn't hold in 3D** (R6.2.2.1) — 3D ellipsoids are thin slabs, not 2D rectangles.
4. **Anchor heights should match target zone heights** (R6.2.4) — chest-centric uses low+mid, NOT ceiling.
5. **Chest-centric beats body-centric for vital signs by +27 pp** (R6.2.3).

## See also

- Architectural decision: `docs/adr/ADR-113-multistatic-placement-strategy.md`
- Research notes: `docs/research/sota-2026-05-22/R6_2*.md`
- Composes with: `01-physics-floor/` (forward model), `06-structure-detection/` (PABS uses placement coverage)
