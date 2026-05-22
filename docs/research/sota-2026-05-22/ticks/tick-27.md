# Tick 27 — 2026-05-22 09:32 UTC

**Thread:** R6.2.5 (multi-subject occupancy union)
**Verdict:** Clean positive — **N=5 hits 100% coverage** for households of 1-4 occupants with chest-centric zones. N=4 knee returns. R6 family completes with this tick.

## What shipped

- `examples/research-sota/r6_2_5_multi_subject.py`
- `examples/research-sota/r6_2_5_multi_subject_results.json`
- `docs/research/sota-2026-05-22/R6_2_5-multi-subject-union.md`

## Headline

| Scenario | # zones | Coverage @ N=5 |
|---|---:|---:|
| 1 occupant | 1 | **100%** |
| 2 occupants | 2 | **100%** |
| 3 occupants | 3 | **100%** |
| 4 occupants | 4 | **100%** |

4-occupant saturation curve:

| N | Coverage |
|---:|---:|
| 2 | 14.5% |
| 3 | 72.9% |
| **4** | **99.0%** ← knee |
| 5 | 100% |

**Knee at N=4** even for 4 occupants. The chest-centric small-zone approach generalises trivially.

## Cross-eval: multi-subject optimisation matters

| Placement | Coverage on 4 zones |
|---|---:|
| Single-subject-optimised | 70.6% |
| **Multi-subject-optimised** | **100%** |
| **Gain** | **+29.4 pp** |

CLI must accept multiple `--target` args and compute union.

## R6 family complete (9 ticks)

| Tick | Config | Result |
|---|---|---:|
| R6.2 | 2D body, single | 51% N=5 |
| R6.2.1 | 3D body, single | 26% N=2 |
| R6.2.2 | 2D body, N-anchor | 97% N=5 |
| R6.2.2.1 | 3D body, N-anchor | 49% N=5 |
| R6.2.3 | 2D chest, single | 82% N=5 |
| R6.2.4 | 3D chest, N-anchor | 77/82% N=5/6 |
| **R6.2.5** | **2D chest, multi-subject** | **100% N=5** |

**R6 family's ship recipe**: 2D chest-centric + multi-subject + N=5 = 100% coverage.

## Why N=4 knee returns for multi-subject

Each chest zone is 40×40 cm and fits inside one Fresnel ellipsoid (~40 cm wide at midpoint of 5 m link). N=4 anchors → 6 pairwise links → enough to cover 4 disjoint chest zones without much waste. Beyond N=4 the marginal gain drops to <1 pp.

**Chest-centric multi-subject is the sweet spot for the Fresnel envelope geometry.**

## Final R6.2 CLI surface (productisation spec)

```
wifi-densepose plan-antennas
    --room W H [Z]                       # 2D or 3D
    --target NAME X Y W H [DX DY DZ]    # repeatable
    --target-mode {body, chest}          # R6.2.3
    --freq-ghz F                         # 2.4, 5.0, 6.0
    --n-anchors N                        # auto-saturation if omitted
    --restarts K                         # 4 default
```

~50 LOC over the original R6.2.

## Composes with prior threads

- R6.2 / R6.2.3: direct extension (single → multi)
- R6.2.2 / R6.2.4: same saturation behaviour
- R14: V1/V2/V3 in households of 2-4 use this recipe
- R3 / ADR-024: per-subject identity + multi-subject placement = full empathic-appliance stack
- ADR-105/106/107: federation orthogonal to placement
- R12 PABS: multi-subject coverage = multi-subject intrusion detection

## Honest scope

- 2D only (3D multi-subject is mechanical extension)
- Static positions (real movement = conservative union)
- Single 5×5 m geometry
- Greedy + 4 restarts
- 4 occupants; beyond may degrade

## Coordination

`ticks/tick-27.md`. No PROGRESS.md edit. Branch `research/sota-r6.2.5-multi-subject`.

## Remaining loop work

- R12.1: pose-PABS closed loop (needs Rust integration, out of scope for synthetic ticks)
- ADR-108: Kyber substitution (quantum-resistant)
- Loop retrospective / 00-summary.md (still ~2.5h until cron stop)

~2.5h to cron stop. **27 ticks landed.** R6 family + R3 arc both substantively complete.
