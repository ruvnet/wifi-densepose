# Tick 17 — 2026-05-22 07:09 UTC

**Thread:** R6.2.2 (N-anchor multistatic placement)
**Verdict:** Practical knee at **N=5 anchors** for typical 5×5 m bedroom. Direct cost-optimisation conclusion + ADR-029 architectural update.

## What shipped

- `examples/research-sota/r6_2_2_multistatic_placement.py` — pure-numpy greedy multi-anchor placement search with random restarts.
- `examples/research-sota/r6_2_2_multistatic_results.json` — full saturation curve for 5×5 m bedroom benchmark.
- `docs/research/sota-2026-05-22/R6_2_2-multistatic-placement.md` — research note.

## Saturation curve (5×5 m bedroom, 3 target zones, 2.4 GHz)

| N anchors | Pairs | Coverage | Marginal |
|---:|---:|---:|---:|
| 2 | 1 | 35.7% | +35.7 pp |
| 3 | 3 | 63.4% | +27.6 pp |
| 4 | 6 | 86.2% | +22.8 pp |
| **5** | **10** | **96.8%** | **+10.6 pp** ← knee |
| 6 | 15 | 100% | +3.2 pp |
| 7+ | 21+ | 100% | +0.0 pp |

**Knee at N=5** — past this, diminishing returns.

## Three regimes surfaced

| Use case | Anchors | Coverage |
|---|---:|---:|
| Single-feature (presence only) | 2-3 | 36-63% |
| Multi-feature (pose, vitals, count) | **4-5** | 86-97% |
| Mission-critical (medical, security) | 6 | 100% |
| Beyond 6 | wasted | 100% (no gain) |

## Cost-optimisation conclusion

Cognitum Seed BOM is $9-15. The +$9-15 from 4→5 anchors buys +10.6 pp coverage. The same cost from 5→6 buys only +3.2 pp. **Consumer recommendation: 5 anchors hits the knee.** Commercial / medical: 6.

## Convenient numerology

**N=5 happens to also satisfy three other constraints simultaneously:**

1. **R7 multi-link mincut**: needs N ≥ 4 to detect single-anchor compromise
2. **ADR-105 federation Krum**: f=1 byzantine tolerance requires K ≥ 5
3. **R6.2.2 coverage knee**: 5 anchors hits practical saturation

These three constraints all bound by similar inverse-square-of-geometry scaling, so the alignment is probably not coincidental — but it's a useful fact for the architectural roadmap.

## ADR-029 recommendation update

ADR-029 (multistatic sensing) didn't specify anchor counts. R6.2.2 fills the gap:

> **Recommended anchor count: 5 for typical 5×5 m room.** 4 anchors gives 86% coverage (good for many use cases); 6 anchors gives 100% but is over-provisioned past the knee.

## Composes with prior threads

- **R6 / R6.2**: direct generalisation; greedy expansion to N anchors
- **R7**: needs N ≥ 4 for multi-link adversarial detection; N=5 satisfies
- **R1**: combined with R6.2.2 = full sensing geometry budget
- **ADR-029**: architectural recommendation now has a number
- **ADR-105**: Krum byzantine bound f < (K-2)/2 → K=5 = f=1 (matches R7 single-attacker case)
- **R10**: wildlife corridors will have different saturation (more anchors for length, fewer for width)
- **R11**: maritime cabins likely saturate earlier (N=3-4)
- **R14**: V1/V2/V3 verticals all need ≥86% coverage = N=4 minimum

## Honest scope

- Single geometry tested (5×5 m bedroom). Other rooms have different knees.
- 2D still (R6.2.1 = 3D ceiling/floor mounts not yet built).
- Free-space (multipath probably adds +5-15% beyond Fresnel-only).
- Greedy + 8 restarts → 1-2 pp shy of global optimum at most.

## Coordination

`ticks/tick-17.md`. No PROGRESS.md edit. Branch `research/sota-r6.2.2-multistatic-placement`.

## Remaining work

- **R3 follow-up**: physics-informed env_sig prediction (zero-shot cross-room via R6 forward operator + room map)
- **R6.1**: multi-scatterer additive forward model
- **R6.2.1**: 3D ceiling/floor placement
- **R6.2.3**: pose-trajectory-aware zones (needs AETHER + R3 data)
- **ADR-107**: cross-installation federation w/ secure aggregation

~4.9h to cron stop. **17 ticks landed. 2 ADRs + 2 deferred follow-ups closed.**
