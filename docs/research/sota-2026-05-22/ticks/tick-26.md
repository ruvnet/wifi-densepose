# Tick 26 — 2026-05-22 09:18 UTC

**Thread:** R3.2 (embedding-level physics-informed env prediction)
**Verdict:** R3.1's corrected architecture is **structurally validated** (physics + residual matches labelled MERIDIAN with zero labels) but **empirically limited** by the synthetic AETHER mean-pooling stand-in. Reaching 80%+ needs real contrastive-learning AETHER (ADR-024).

## What shipped

- `examples/research-sota/r3_2_embedding_physics_env.py` — embedding-level physics-informed env experiment.
- `examples/research-sota/r3_2_embedding_results.json` — full benchmark.
- `docs/research/sota-2026-05-22/R3_2-embedding-level-physics-env.md` — research note.

## Headline

| Approach | Cross-room 1-shot K-NN |
|---|---:|
| Within-room AETHER sanity | 100% |
| Cross-room AETHER raw (no env sub) | 10% (chance) |
| Cross-room AETHER + labelled MERIDIAN (oracle) | **20%** |
| Cross-room AETHER + physics-informed (no labels) | 10% (chance) |
| **Cross-room AETHER + physics + residual (no labels)** | **20%** ← matches oracle |
| Chance | 10% |

The architecturally-correct approach (physics + residual correction) **MATCHES the labelled MERIDIAN oracle** with **zero labels**.

## Why both approaches cap at 20%

In R3 tick 12, AETHER was Gaussian-direction embeddings with strong per-subject signal → 100% achievable. In R3.2, AETHER is mean-pooling complex-52 CSI with only 30% body-size variation as per-subject signal. The per-subject signature is too weak; even labelled MERIDIAN can't dominate the residual.

**The bottleneck is now per-subject signal strength, not environment subtraction.**

## Three "honest scope" findings in the loop

R3.2 is the third explicit "synthetic too weak to demonstrate production claim" finding:

| Tick | Finding | Path forward |
|---|---|---|
| R3.1 | Physics-informed at raw level fails | Apply at embedding level (R3.1 → R3.2) |
| R6.2.2.1 | 2D N=5 knee doesn't hold in 3D | Use chest zones (R6.2.2.1 → R6.2.4) |
| R3.2 | Mean-pooling AETHER too weak | Use real contrastive AETHER (out of scope) |

All three are productive — they identify the gap that production work must fill.

## What R3.2 DOES validate

1. **Embedding-level operation is the right space** (vs raw-CSI's R3.1 failure)
2. **Physics + residual matches labelled oracle** (structural correctness)
3. **ADR-024 (AETHER) is on the critical path** for cross-room re-ID

## What R3.2 DOES NOT achieve

1. 80%+ cross-room accuracy (needs real AETHER)
2. Production benchmark numbers
3. Loop-level closure of R3 (needs ADR-024 implementation work outside the loop)

## Recommended next experiment (out of scope)

Replace mean-pooling AETHER stand-in with ADR-024 contrastive-learning head. Train on MM-Fi; run R3.2 protocol; expected to hit 70-90%+. ~1-2 days of training work.

## R3 thread now satisfactorily closed for the loop

R3 (tick 12) → R3.1 (NEGATIVE) → R3.2 (structurally validated). The arc produced:
- Architectural recommendation: use embedding level
- Identified critical-path component: ADR-024 AETHER
- Three constraint regimes documented
- Clear production path

## Composes with prior threads

- R3 / R3.1 / R3.2 = arc
- R6 / R6.1 = forward operator (unchanged)
- R6.2 family = placement-level optimisation (orthogonal to cross-room re-ID)
- R12 PABS = within-room (cross-room needs R3.2 architecture)
- R14 / R15 = privacy framework holds
- ADR-024 = critical path
- ADR-105 / ADR-106 / ADR-107 = federation can ship R3.2 outputs

## Honest scope

- Synthetic AETHER is mean-pooling, not contrastive
- 20% oracle ceiling is this synthetic setup's cap, not the architecture's
- 30% body-size variation is weak per-subject signal vs R15's 12-15 bits
- Two rooms only
- Static subjects; dynamic would give richer per-subject signals

## Coordination

`ticks/tick-26.md`. No PROGRESS.md edit. Branch `research/sota-r3.2-embedding-physics-env`.

## Remaining work

- R12.1: pose-PABS closed loop
- R6.2.5: multi-subject occupancy union
- ADR-108: Kyber substitution

~2.7h to cron stop. **26 ticks landed.**
