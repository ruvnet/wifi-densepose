# R6.2.4 — 3D chest-centric N-anchor: validates R6.2.2.1's architectural fix

**Status:** prediction validation + counter-finding on ceiling mounts · **2026-05-22**

## Premise

R6.2.2.1 (3D N-anchor on body-footprint zones) showed N=5 gives only 49% coverage in 3D vs 97% in 2D. It predicted: **switching to chest-centric zones (R6.2.3) should recover 80%+ at N=5 in 3D**. This tick tests that prediction.

## Result: 76.8% at N=5 (validation: partial)

| N anchors | Coverage | Marginal | Heights (L / M / H) |
|---:|---:|---:|---:|
| 2 | 11.3% | +11.3 pp | 1 / 1 / 0 |
| 3 | 60.3% | +49.0 pp | 1 / 2 / 0 |
| 4 | 76.1% | +15.8 pp | 2 / 2 / 0 |
| **5** | **76.8%** | +0.6 pp | 3 / 2 / 0 |
| 6 | 81.6% | +4.8 pp | 4 / 2 / 0 |

**R6.2.2.1's prediction of 80%+ at N=5 was off by 3.2 pp.** N=5 hits 76.8%; **N=6 hits 81.6%** — the 80%+ knee shifts one anchor higher than predicted.

## 4-way comparison at N=5

| Configuration | N=5 coverage |
|---|---:|
| R6.2.2 (2D body) | 96.8% |
| R6.2.3 (2D chest) | 82.4% |
| R6.2.2.1 (3D body) | 49.4% |
| **R6.2.4 (3D chest)** | **76.8%** |

3D chest-centric **recovers 27 pp** over 3D body-centric — most of the 47 pp gap that R6.2.2.1 surfaced. The architectural fix mostly works.

## Counter-finding: ceiling anchors are not selected

R6.2.1 recommended "one ceiling anchor + low + mid" as the winning 3D strategy. R6.2.4 finds something different: **at no N does greedy select a ceiling (z=2.4 m) anchor for chest-centric zones**. The heights are 100% low (0.8 m) + mid (1.5 m).

Why: chest zones live at z=0.3-1.5 m. Ceiling anchors (z=2.4 m) put their Fresnel ellipsoid envelopes at z≈2.4 m — well above the chest targets. The targets are at heights *matching the chosen anchor mid-points*, not *between anchor extremes*.

**Sharpened recommendation: anchor heights should match the target-zone heights.**

| Target | Best anchor heights |
|---|---|
| Bed-only (z=0.3-0.6) | Low (0.5-0.8 m) on opposite sides of bed |
| Chair / sitting (z=0.5-1.0) | Low + mid |
| Standing chest (z=1.2-1.5) | Mid (1.2-1.5 m) |
| Full body (z=0.3-1.7) | Mixed low / mid / high (per R6.2.1) |
| **Mixed chest (z=0.3-1.5)** | **Low + mid only — NO ceiling** |

R6.2.1's "include ceiling" recommendation was correct for **full-body** coverage, not for **chest-centric** coverage. The two regimes diverge.

## Saturation curve has a flat spot at N=4→5

The +0.6 pp marginal at N=4→5 is suspicious — likely a greedy local-optimum artefact. N=6 jumps +4.8 pp, suggesting the global optimum has a slightly different 5-anchor configuration than greedy found. With more restarts (8-16) the N=5 number might recover to ~80%.

This is honest scope on the greedy algorithm: it's an approximation, and the N=5 result is probably 2-4 pp shy of the true global optimum. Not a research finding worth fixing in this tick; documented for future productisation.

## Updated ADR-029 anchor-count recommendation

Replacing the simple "5 anchors hits the knee" rec from R6.2.2 with the dimension- and zone-aware version:

| Configuration | Recommended N | Realistic coverage |
|---|---:|---:|
| 2D body-centric | 5 | 97% (R6.2.2) |
| 2D chest-centric | 5 | 82% (R6.2.3) |
| 3D body-centric | 7-8 | 65%+ (R6.2.2.1) |
| **3D chest-centric** | **6** | **82%** (R6.2.4) |

**For vital-signs cogs in real 3D deployments: N=6 + chest-centric zones + low/mid anchor heights.** This is the strongest single recommendation the R6 family produces.

## Why this tick matters

It's the **fourth tick** in the R6 family + the **second self-corrective tick** in the loop. R6.2.2.1 made an explicit prediction; R6.2.4 verifies + corrects it. This is the right structure for research progress:

1. R6 → R6.2 (productisation of forward model)
2. R6.2 → R6.2.2 (multistatic generalisation, 2D)
3. R6.2.2 + R6.2.1 → R6.2.2.1 (3D composition, surfaces 2D over-promise)
4. R6.2.2.1 prediction → R6.2.4 verification (chest-centric mostly closes the gap)

Each tick has a clear hypothesis and a clear empirical result that either confirms or revises the previous.

## Composes with prior threads

- **R6.2.1 / R6.2.2 / R6.2.2.1**: same physics, different zones
- **R6.2.3 (2D chest)**: motivated this tick; 3D extension is now done
- **R7 mincut**: N=6 still satisfies N ≥ 4 byzantine-detection requirement
- **ADR-029 / ADR-105**: anchor-count recommendation now has 4 dimensions (2D/3D × body/chest) of specification
- **R14 V1/V2/V3**: chest-mode + N=6 is the empathic-appliance deployment recipe in 3D
- **R12 PABS**: 3D chest coverage of 77% means PABS detects intruders standing/sitting/lying inside chest zones at this fraction; gaps in coverage are blind spots

## Honest scope

- **Greedy + 4 restarts** approximates global optimum; N=5 likely 2-4 pp shy
- **0.1 m 3D grid** in target zones (finer than R6.2.2.1's 0.15 m)
- **Same 5×5×2.5 m geometry** — other rooms need separate benchmarks
- **Three chest zones** — real deployments would have one to many per occupant
- **R6.2.1's ceiling recommendation was for full-body, not chest** — the counter-finding here doesn't invalidate R6.2.1 but refines it

## What this DOES enable

1. **Validated the architectural fix**: 3D chest-centric at N=6 = 82% coverage, matching 2D chest-centric numbers at N=5.
2. **Sharpened anchor-height recommendation**: heights should match target-zone heights; chest-centric uses LOW+MID only, NOT ceiling.
3. **Final ADR-029 anchor-count table** with 4 axes (dimension × zone-mode).

## What this DOES NOT enable

- Closing the last ~15 pp gap (3D chest 82% vs 2D body 97%) — fundamental 3D thinness of Fresnel ellipsoid
- Multi-subject occupancy union (R6.2.5)
- Productisation as a CLI flag (already catalogued)

## Next ticks (R6 family complete?)

After R6, R6.1, R6.2, R6.2.1, R6.2.2, R6.2.2.1, R6.2.3, R6.2.4 — the R6 family has covered: forward model (R6), multi-scatterer (R6.1), 2D placement (R6.2), 3D placement (R6.2.1), N-anchor (R6.2.2), 3D N-anchor (R6.2.2.1), chest-centric (R6.2.3), 3D chest N-anchor (R6.2.4). The family is **substantively complete** for placement-strategy purposes.

Remaining R6 follow-ups (pose-trajectory-aware, multi-subject union) need empirical AETHER + R3 data — out of scope for synthetic-data ticks.

## Connection back

- **R6 / R6.1**: physical foundation
- **R6.2 / R6.2.3**: 2D variants
- **R6.2.1 / R6.2.2 / R6.2.2.1**: 3D and N-anchor variants
- **R7 / ADR-029 / ADR-105**: composition with adversarial defence and federation
- **R14**: empathic appliance deployment recipe finalised: N=6 + 3D chest-centric + low/mid anchor heights
