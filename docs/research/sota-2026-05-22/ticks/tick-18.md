# Tick 18 — 2026-05-22 07:24 UTC

**Thread:** R6.1 (multi-scatterer additive Fresnel forward model)
**Verdict:** Working 6-scatterer body model. Discovers a **4.7 dB multi-scatterer penalty** that matches R13's 5-dB-shortfall finding — gives R13 a physical origin and unblocks R12's PABS revision path.

## What shipped

- `examples/research-sota/r6_1_multiscatterer.py` — pure-numpy multi-scatterer Fresnel forward model with 6 body-part scatterers + breathing motion.
- `examples/research-sota/r6_1_multiscatterer_results.json` — machine-readable predictions.
- `docs/research/sota-2026-05-22/R6_1-multiscatterer-forward-model.md` — research note.

## Headline finding

5 m link, 2.4 GHz, subject 25 cm off LOS, 30-second breathing time-series:

| Configuration | Breathing SNR (best subcarrier) |
|---|---:|
| Single-scatterer ideal (R6) | +23.7 dB |
| Multi-scatterer realistic (R6.1, 6 parts) | **+19.0 dB** |
| **Multi-scatterer penalty** | **+4.7 dB** |

This 4.7 dB penalty is the gap between R6's idealised physics and realistic deployment — and **it matches R13's 5 dB shortfall to within 0.3 dB**, suggesting R13's "we are 5 dB short of pulse-contour recovery" finding has a **physical origin** in the static body parts, not just measurement noise.

## Per-body-part energy contribution

- **Chest**: 27.6% of total CSI energy (highest reflectivity, 5× per-limb value)
- Each limb / head: 1.1% each
- The chest IS the breathing signal; limbs are confound, not signal

## Architectural implications

1. **Chest-centric placement targeting** (R6.2.3) — current R6.2 treats body as single point; should target chest specifically.
2. **Mask limbs in vital_signs pipeline** — pose pipeline (ADR-079, ADR-101) already extracts limb positions; vital_signs just doesn't use them.
3. **R14 V3 re-scope** — attention-respecting conversational appliance needs +25 dB pulse-contour recovery, which R6.1 says is unachievable. V3 should depend only on breathing *rate* stability, not pattern *shape*.

## R12's PABS revision unblocked

R12 (NEGATIVE eigenshift) suggested **PABS over Fresnel basis** as the revision. R6.1 IS the explicit A(voxel) forward operator that PABS needs. R12 + R6.1 = tractable structure-detection implementation.

## Why this is a satisfying integration

- R6 = bound (idealised single-scatterer)
- R6.1 = floor (realistic multi-scatterer)
- R13 = the actual failure mode (5 dB short)

The three threads now have a coherent physics story: pulse-contour recovery is bound below by what R6.1 leaves achievable, which is 4.7 dB worse than the R6 idealised limit, which is enough to make R13's contour recovery infeasible.

## On-LOS placement is degenerate

First simulation run had subject at y=0 (exactly on LOS), giving SNR of -60 dB (essentially undetectable). Path-delta is 2nd-order in offset for on-LOS scatterers, so breathing in y direction barely changes path. **Lesson surfaced**: real installations need subject OFF the LOS line, not on it. The off-LOS placement (25 cm) gives the +19 dB number.

This is a non-obvious deployment requirement that R6.2 placement search should respect — don't place antennas such that the *primary* target zone sits on the LOS line.

## Composes with prior threads

- **R5**: subcarrier selection prefers reliable, not high-SNR
- **R6**: provides the per-scatterer building block
- **R6.2 / R6.2.2 / R6.2.3 (future)**: chest-centric placement
- **R7**: residual-against-forward-model gives tighter adversarial detection
- **R12 NEGATIVE**: PABS A operator now unblocked
- **R13 NEGATIVE**: 5-dB gap has physical origin
- **R14**: V3 needs rescope to rate-only

## Honest scope

- 6 scatterers is 1st-order; 50-100 voxel body would be better
- Reflectivity ratios are guesses (RCS measurements at 2.4 GHz on real humans would refine)
- Static body assumption (limbs do micro-move during breathing)
- 2D top-down (3D would add vertical structure)
- No multipath (room reflections add scatterers; model is general enough to include them)

## Coordination

`ticks/tick-18.md`. No PROGRESS.md edit. Branch `research/sota-r6.1-multiscatterer-fresnel`.

## Remaining work

- **R3 follow-up**: physics-informed env_sig prediction (uses R6 + room map → zero-shot cross-room)
- **R6.2.1**: 3D ceiling/floor placement
- **R6.2.3**: chest-centric / pose-trajectory-aware target zones (now strongly motivated by R6.1)
- **R12 PABS implementation**: forward operator now available
- **ADR-107**: cross-installation federation w/ secure aggregation

~4.6h to cron stop. **18 ticks landed.** Loop has covered R1-R15 + 2 ADRs + 3 deferred follow-ups (R6.2, R6.2.2, R6.1).
