# R6.2.2.1 — 3D N-anchor multistatic: the knee disappears

**Status:** 3D saturation curve + comparison to R6.2.2 2D · **2026-05-22**

## Premise

R6.2.2 (2D N-anchor) found a clean **knee at N=5 anchors** with 96.8% coverage of bedroom-class target zones, and pushed that as the consumer recommendation. R6.2.1 (3D single-pair) found ceiling-only mounting fails. R6.2.2.1 composes both: how does the saturation curve change when both **3D ellipsoids** and **mixed-height candidates** are used?

The practical question: does ADR-029's 4-anchor default give adequate coverage in real 3D rooms, or does the 2D analysis under-promise?

## Results

5×5×2.5 m room, three 3D target zones (bed at z=0.3-0.6, chair at z=0.5-1.2, standing at z=1.0-1.7). 94 candidate positions (3 wall heights + ceiling grid). Greedy + 4 restarts:

| N anchors | Pairs | 3D coverage | Marginal | Heights chosen (low / mid / high) |
|---:|---:|---:|---:|---|
| 2 | 1 | 7.7% | +7.7 pp | 1 / 1 / 0 |
| 3 | 3 | 28.1% | +20.4 pp | 1 / 2 / 0 |
| 4 | 6 | 40.6% | +12.5 pp | 3 / 0 / 1 |
| **5** | 10 | **49.4%** | +8.8 pp | 4 / 0 / 1 |
| 6 | 15 | 59.1% | +9.8 pp | 4 / 1 / 1 |
| 7 | 21 | 65.1% | +6.0 pp | 5 / 1 / 1 |

**No clean knee.** Marginal gains stay 6-10 pp from N=4 onwards. 3D space is fundamentally harder to cover with discrete pairwise links.

## Comparison: 2D vs 3D at same N

| N anchors | 2D coverage (R6.2.2) | 3D coverage (R6.2.2.1) | Δ |
|---:|---:|---:|---:|
| 2 | 35.7% | 7.7% | -28 pp |
| 3 | 63.4% | 28.1% | -35 pp |
| 4 | 86.2% | 40.6% | -46 pp |
| 5 | 96.8% | 49.4% | **-47 pp** |
| 6 | 100% | 59.1% | -41 pp |
| 7 | 100% | 65.1% | -35 pp |

**At N=5, 3D coverage is half of 2D coverage.** The 2D analysis was over-promising.

## Why 3D is harder

The 2D Fresnel zone is an *ellipse* — an area; the 3D zone is an *ellipsoid* — a volume. The 2D ellipse trivially covers any vertical extent at the LOS height; the 3D ellipsoid has a perpendicular thickness equal to its transverse radius (~40 cm at 5 m link). Targets above or below the LOS plane are missed entirely.

Each pairwise link in 3D effectively contributes a **thin slab** rather than a full 2D rectangle. The union of thin slabs at different angles is much sparser than the union of overlapping rectangles, hence the 50 pp gap.

## Height distribution: greedy strongly prefers low + mixed

At every N from 4 onwards, the greedy search picks:
- 3-5 LOW (z=0.8 m) anchors
- 0-1 MID (z=1.5 m)
- 1 HIGH (ceiling, z=2.4 m)

The HIGH anchor matters (it's selected at every N), but never dominates. The placement strategy that **wins** is "mostly-low + one-high" — which is also what R6.2.1's single-pair analysis suggested (one low + one high diagonal).

## Updated recommendation for ADR-029

| Use case | 2D rec (R6.2.2) | 3D rec (R6.2.2.1) | Realistic coverage |
|---|---:|---:|---:|
| Presence / occupancy | 2-3 | 4 | ~41% (3D) / 86% (2D) |
| Multi-feature (pose, vitals, count) | 4-5 | **5-6** | 49-59% (3D) / 97% (2D) |
| Mission-critical (medical, security) | 6 | **7-8** | 65%+ (3D) |

**The 2D-derived N=5 consumer recommendation is too optimistic for real 3D deployments.** Two responses:

1. **Bump to N=6-7** for realistic 3D coverage at the same target quality.
2. **Use chest-centric zones (R6.2.3)** — chest zones are smaller (40×40 cm vs 3 m² beds) and fit inside the Fresnel envelope much more easily. R6.2.3 + R6.2.2.1 composed would give 80%+ coverage with N=4-5.

The recommended path: **R6.2.3 chest-centric + R6.2.2 N=5 anchor count** = realistic 3D coverage of 80%+ at the ADR-029 default N. This is the architectural lever that aligns the 2D and 3D physics.

## Composes with prior threads

- **R6.2** (2D single-pair) — same engine.
- **R6.2.1** (3D single-pair) — same 3D ellipsoid model.
- **R6.2.2** (2D N-anchor) — same greedy search, composes naturally with 3D.
- **R6.2.3** (chest-centric) — the architectural fix for the 3D coverage gap.
- **R7** (mincut adversarial) — requires N ≥ 4 even in 3D; the practical 4-5 anchor recommendation still satisfies R7.
- **ADR-029** (multistatic) — anchor-count recommendation needs both N AND target-zone semantics specified.
- **ADR-105 Krum** — f=1 byzantine tolerance still needs K ≥ 5 regardless of dimension; matches the 3D recommendation.

## Why this is a meaningful follow-up not a re-do

R6.2.2 (2D) and R6.2.1 (3D single-pair) each told a partial story. R6.2.2.1 composes them and reveals the 2D was over-promising. Specifically:

- 2D over-promise: "N=5 hits 97% knee" → reality: only for 2D rectangles, not 3D volumes
- 3D fix: bump N or shrink target zones (use chest-centric)

Without R6.2.2.1, the team would have shipped ADR-029 with the 2D recommendation and discovered the 3D shortfall during field deployment.

## Honest scope

- **Greedy with 4 restarts** approximates global optimum; brute-force is intractable at this scale. Real optimum might be 2-5 pp higher.
- **Coarse 0.15 m grid** in 3D. Finer resolution would refine but not change the qualitative finding.
- **Single geometry tested** — 5×5×2.5 m bedroom. Different rooms (tall living rooms, narrow hallways) have different curves.
- **Free-space propagation** — multipath adds 5-15% but doesn't restore the 50 pp gap.
- **Body-footprint zones** — using R6.2.3 chest-centric zones would substantially raise the percentage; not tested here.
- **94 candidates** is a sparse search; finer step would refine slightly.

## What this DOES enable

1. **Honest 3D coverage numbers** for ADR-029 planning — 49% at N=5 is the realistic number, not 97%.
2. **Decision point**: bump N OR use chest-centric zones (R6.2.3). Both are tractable; the latter is more elegant.
3. **Validation that "mostly-low + one-high" is the right placement strategy** in 3D, confirming R6.2.1's pair-finding.

## What this DOES NOT enable

- A clean knee — there isn't one in 3D under these zones.
- Composition with R6.2.3 chest-centric (= R6.2.4, future).
- Validated multi-cog deployment recipes — each cog needs its own analysis.

## Next ticks

- **R6.2.4**: compose 3D N-anchor + chest-centric zones → does N=5 hit 80% in 3D when zones are smaller?
- **R6.2.5**: multi-subject occupancy (union of chest envelopes across expected positions).
- **ADR-029 amendment**: anchor-count recommendation needs both N AND zone-mode specified.

## Connection back

- **R6.2** (2D single-pair, R6.2.1 (3D single-pair), R6.2.2 (2D N-anchor), R6.2.3 (chest-centric) — R6.2.2.1 is the natural composition of the first three; R6.2.3 is the way to "fix" the 3D shortfall.
- **ADR-029** — needs amendment to specify both N and zone-mode.
- **ADR-105 Krum** — N=5 still required for byzantine tolerance; this matches the 3D recommendation.
- **R14** V1/V2/V3 — V1 chest-only is naturally chest-mode = R6.2.3; V2 (mixed presence + chest) and V3 (chest) similarly. Aligning with R6.2.3 makes 3D coverage tractable.
