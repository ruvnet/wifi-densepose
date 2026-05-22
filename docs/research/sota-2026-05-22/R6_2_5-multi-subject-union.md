# R6.2.5 — Multi-subject occupancy union: N=5 hits 100% for 4 occupants

**Status:** clean positive result · **2026-05-22**

## Premise

R6.2 / R6.2.3 picked one chest position per zone. Real households have 2-4 occupants who can be in different positions simultaneously. R6.2.5 extends to **union of chest envelopes** across all expected occupant positions. The practical question: does coverage degrade gracefully as occupant count grows?

## Result: graceful saturation at N=5

| Scenario | # zones | Total area | Coverage @ N=5 |
|---|---:|---:|---:|
| 1 occupant (chair) | 1 | 0.16 m² | **100%** |
| 2 occupants (chair + bed) | 2 | 0.40 m² | **100%** |
| 3 occupants (chair + bed + desk) | 3 | 0.48 m² | **100%** |
| 4 occupants (+ 2nd chair) | 4 | 0.64 m² | **100%** |

**N=5 hits 100% coverage for all configurations up to 4 occupants.** The chest-centric small-zone approach (R6.2.3) generalises trivially to multi-subject.

## 4-occupant saturation curve

| N | Coverage | Marginal |
|---:|---:|---:|
| 2 | 14.5% | +14.5 pp |
| 3 | 72.9% | +58.4 pp |
| **4** | **99.0%** | **+26.1 pp** |
| 5 | 100% | +1.0 pp |
| 6 | 100% | +0 pp |
| 7 | 100% | +0 pp |

**Knee returns to N=4** — even for 4 occupants, 4 anchors get us to 99%. This is the **2D chest-centric multi-subject** regime, which is the most demanding 2D configuration tested in the R6 family — and it still hits the knee at N=4.

## Cross-eval: single-subject placement is bad for multi-subject

| Placement | Coverage on 4-zone target |
|---|---:|
| Single-subject-optimised | 70.6% |
| Multi-subject-optimised | **100%** |
| **Gain from multi-subject optimisation** | **+29.4 pp** |

The CLI must accept multiple `--target` arguments and optimise for their **union** — not pick a representative zone and hope.

## Updated CLI recommendation

```bash
wifi-densepose plan-antennas \
    --room 5 5 \
    --target chair_chest 3.7 3.7 0.4 0.4 \
    --target bed_chest   2.2 0.8 0.6 0.4 \
    --target desk_chest  0.5 2.7 0.4 0.2 \
    --target chair2_chest 1.0 4.2 0.4 0.4 \
    --freq-ghz 2.4
```

Output: N=5 anchors hitting 100% coverage of the union.

## R6 family summary (8 ticks + this)

| Tick | Configuration | Headline number |
|---|---|---:|
| R6.2 | 2D body, single-subject | 51% N=5 |
| R6.2.1 | 3D body, single-subject | 26% N=2 (mixed-height) |
| R6.2.2 | 2D body, N-anchor | 97% N=5 |
| R6.2.2.1 | 3D body, N-anchor | 49% N=5 |
| R6.2.3 | 2D chest, single-subject | 82% N=5 |
| R6.2.4 | 3D chest, N-anchor | 77% N=5 / 82% N=6 |
| **R6.2.5 (this)** | **2D chest, multi-subject (1-4)** | **100% N=5** |

The R6 family's headline finding: **2D chest-centric + multi-subject + N=5 = 100% coverage**. This is the placement recipe to ship.

## Composes with prior threads

- **R6.2 / R6.2.3**: directly extends — single-subject → multi-subject union
- **R6.2.2 / R6.2.4**: same saturation behaviour at the multi-subject level
- **R14 (empathic appliances)**: V1 lighting / V2 HVAC / V3 attention in households of 2-4 occupants → use multi-subject placement
- **R3 / ADR-024**: per-subject identity (AETHER) + multi-subject placement = full empathic-appliance stack
- **ADR-105 / ADR-106 / ADR-107**: federation operates on the same model across occupant counts; placement is orthogonal
- **R12 PABS**: works per-subject within the union; multi-subject coverage = multi-subject intrusion detection

## Why N=4 knee returns for multi-subject

Each chest zone is small (40×40 cm) and fits inside a single Fresnel ellipsoid (which is ~40 cm wide at midpoint of a 5 m link). With N=4 anchors, we get 6 pairwise links — enough Fresnel ellipsoids to cover 4 disjoint 40×40 cm zones without much waste. Beyond N=4 the marginal gain drops to <1 pp.

This is *more saturated* than the single-subject R6.2 setup (which used 3 m² bed footprint and couldn't be covered fully even at N=8 with body-centric zones). **Chest-centric multi-subject is the sweet spot for the Fresnel envelope geometry.**

## Honest scope

- **2D only** — multi-subject 3D not benchmarked (extension is mechanical; expect N=6 to retain the chest-centric N=5 advantage).
- **Static positions** — real occupants move; the union should be conservative (larger than any instantaneous configuration).
- **Single 5×5 m geometry** — larger or oddly-shaped rooms need separate benchmarks.
- **Greedy + 4 restarts** — global optimum may be 1-2 pp higher.
- **4 occupants** — beyond 4-5 the coverage may degrade. Extreme density (e.g. classroom with 20 people) is a different regime.

## What this DOES enable

1. **A clean cap on the placement complexity story**: 4-occupant households are fully sensable at N=5 with multi-subject-aware placement.
2. **A required CLI feature**: support multiple `--target` arguments.
3. **An updated installer recipe**: for households of 1-4, the same N=5 chest-centric placement works.
4. **R6 family closes with a positive result** that ships directly.

## What this DOES NOT enable

- Beyond 4-5 occupants — separate regime, not tested.
- Time-varying occupancy (people moving between zones) — would benefit from pose-trajectory data (out of scope).
- 3D multi-subject — mechanical extension, not done here.

## Final R6.2 CLI surface

After this tick, the productisation of R6.2 should support:

```
wifi-densepose plan-antennas
    --room W H [Z]                       # 2D or 3D
    --target NAME X Y W H [DX DY DZ]    # repeatable
    --target-mode {body, chest}          # R6.2.3
    --freq-ghz F                         # 2.4, 5.0, 6.0
    --n-anchors N                        # auto-saturation if omitted
    --restarts K                         # 4 default
```

This covers the R6.2 / R6.2.1 / R6.2.2 / R6.2.2.1 / R6.2.3 / R6.2.4 / R6.2.5 use cases in a single CLI tool. ~50 LOC over the original R6.2.

## Connection back

- **R6 / R6.1**: physical foundation
- **R6.2 / R6.2.3**: single-subject body / chest
- **R6.2.1 / R6.2.2 / R6.2.2.1 / R6.2.4**: 3D / N-anchor / composition
- **R6.2.5 (this)**: multi-subject completes the matrix
- **R14**: empathic-appliance deployment recipe is now: N=5 + chest-centric + multi-subject-union targets, with mixed-height anchors for full-body coverage when needed
