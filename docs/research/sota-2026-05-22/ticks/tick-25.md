# Tick 25 — 2026-05-22 09:01 UTC

**Thread:** R6.2.4 (3D chest-centric N-anchor multistatic — composes R6.2.2.1 + R6.2.3)
**Verdict:** R6.2.2.1's prediction of "80%+ at N=5 in 3D chest-centric" partially validated: **N=5 = 76.8%**, **N=6 = 81.6%**. Knee shifts one anchor higher than predicted. Plus a counter-finding: **no ceiling anchors selected** for chest-centric zones.

## What shipped

- `examples/research-sota/r6_2_4_3d_chest_multistatic.py`
- `examples/research-sota/r6_2_4_3d_chest_results.json`
- `docs/research/sota-2026-05-22/R6_2_4-3d-chest-multistatic.md`

## 4-way comparison at N=5

| Configuration | Coverage |
|---|---:|
| R6.2.2 (2D body) | 96.8% |
| R6.2.3 (2D chest) | 82.4% |
| R6.2.2.1 (3D body) | 49.4% |
| **R6.2.4 (3D chest)** | **76.8%** |

3D chest **recovers 27 pp** of the 47 pp gap that R6.2.2.1 surfaced. Most of the architectural fix works.

## Counter-finding: ceiling anchors not selected

At no N does greedy pick a ceiling (z=2.4 m) anchor for chest-centric zones. Heights are 100% low (0.8 m) + mid (1.5 m).

**Why**: chest zones at z=0.3-1.5 don't benefit from ceiling anchors whose envelope sits at z≈2.4. R6.2.1's "include ceiling" rec was correct for full-body coverage, not chest-centric.

**Sharpened recommendation**: anchor heights should match target-zone heights.

| Target | Best anchor heights |
|---|---|
| Bed-only (z=0.3-0.6) | Low only |
| Chair / sitting (z=0.5-1.0) | Low + mid |
| Standing chest (z=1.2-1.5) | Mid only |
| Mixed chest (z=0.3-1.5) | Low + mid (NO ceiling) |
| Full body (z=0.3-1.7) | Low + mid + high (per R6.2.1) |

## Final ADR-029 anchor-count table (4-axis)

| Configuration | N | Coverage |
|---|---:|---:|
| 2D body-centric | 5 | 97% |
| 2D chest-centric | 5 | 82% |
| 3D body-centric | 7-8 | 65%+ |
| **3D chest-centric** | **6** | **82%** |

**For vital-signs cogs in real 3D deployments: N=6 + chest-centric zones + low/mid anchor heights.**

## R6 family substantively complete

8 ticks in the R6 family:
- R6 (forward model)
- R6.1 (multi-scatterer)
- R6.2 (2D placement)
- R6.2.1 (3D placement)
- R6.2.2 (2D N-anchor)
- R6.2.2.1 (3D N-anchor)
- R6.2.3 (chest-centric)
- R6.2.4 (3D + chest) ← this tick

Covered: physics, body model, 2D/3D placement, N-anchor, chest-vs-body zones. Remaining items (pose-trajectory-aware, multi-subject union) need empirical AETHER + R3 data, out of scope for synthetic-data ticks.

## Second self-corrective tick

R6.2.2.1 predicted 80%; actual is 76.8%. Self-correction is documented (prediction was 3.2 pp optimistic, knee shifts to N=6). This is the integrity pattern the loop has been producing — explicit predictions, explicit corrections.

## Composes with prior threads

- R6.2.1 / R6.2.2 / R6.2.2.1: same physics, different zones
- R6.2.3 motivated this tick
- R7 / ADR-029 / ADR-105: N=6 still satisfies byzantine + Krum requirements
- R14 V1/V2/V3: chest-mode + N=6 is the empathic-appliance deployment recipe

## Honest scope

- Greedy + 4 restarts; N=5 likely 2-4 pp shy of true global
- 0.1 m 3D grid; single geometry
- Three chest zones (real deployments would have one to many per occupant)
- R6.2.1's ceiling rec was for full-body, not invalidated — just refined

## Coordination

`ticks/tick-25.md`. No PROGRESS.md edit. Branch `research/sota-r6.2.4-3d-chest-multistatic`.

## Remaining work

- R6.2.5: multi-subject occupancy union (needs AETHER + R3 data)
- R12.1: pose-PABS closed loop
- R3.2: embedding-level physics-informed env
- ADR-108: Kyber substitution

~3.0h to cron stop. **25 ticks landed.** Loop covered 13 research threads + 3 ADRs + 10 deferred follow-ups + 8-tick R6 family + 3 negative-result categories + 2 self-corrections.
