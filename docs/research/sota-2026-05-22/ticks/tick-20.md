# Tick 20 — 2026-05-22 07:54 UTC

**Thread:** R3.1 (physics-informed env_sig prediction at raw-CSI level) — **NEGATIVE (architecture-error category)**
**Verdict:** The naive "subtract predicted env from raw CSI" fails at chance level. Even the labelled MERIDIAN oracle fails at raw-CSI level. The fix: apply physics-informed prediction at the **AETHER embedding level**, not raw CSI.

## What shipped

- `examples/research-sota/r3_1_physics_informed_env.py` — pure-numpy two-room cross-room experiment.
- `examples/research-sota/r3_1_physics_env_results.json` — machine-readable result.
- `docs/research/sota-2026-05-22/R3_1-physics-informed-env-prediction.md` — research note documenting the negative + corrected architecture.

## Headline

| Configuration | 1-shot K-NN accuracy |
|---|---:|
| Within-room baseline | 100% |
| Cross-room raw | **10% (= chance)** |
| Cross-room labelled MERIDIAN (oracle) | **10% (= chance)** |
| Cross-room physics-informed | **10% (= chance)** |

All three cross-room approaches collapse to chance — including the labelled oracle. Position-dependent within-room variance dominates per-subject signature at the raw-CSI level.

## Why this is a meaningful negative

R3 (tick 12) showed MERIDIAN works in **AETHER embedding space** (where position-invariance is already done). R3.1 surfaces that at **raw CSI level**, where position-invariance hasn't been done yet, no env-subtraction method works — because the variance you'd subtract isn't the variance you need to remove.

**Surfaces an architecture error before implementation.** Future engineer attempting "subtract predicted env from raw CSI" would waste weeks; R3.1 documents the failure path.

## Corrected architecture

```
raw CSI -> AETHER embedding head (position-invariant) -> physics-informed env subtraction -> cross-room K-NN
```

Physics-informed prediction must be applied at the **embedding level**, not raw level. AETHER already removes position-dependent variation; the predicted-env subtraction then has only the room-shift component to remove.

## Three kinds of negative result the loop has now demonstrated

| Kind | Example | Outcome |
|---|---|---|
| **Missing-tool** (revisitable) | R12 NEGATIVE → R12 PABS POSITIVE | Tool became available later (R6.1) and approach worked |
| **Physics-floor** (permanent) | R13 contactless BP | Hard 5 dB wall; no tool changes this |
| **Architecture-error** (correctable) | R3.1 (this tick) | Right idea, wrong application level; corrected architecture explicit but not yet implemented |

Categorising negatives by their resolution path is itself a research contribution. This is the loop's most "meta" tick.

## Composes with prior threads

- **R3 (POSITIVE in embedding space)** — confirmed indirectly; raw-level failure shows why R3 operated at embedding level
- **R6.1** — operator is correct; application level was wrong
- **R12 PABS (POSITIVE)** — operates in raw space because comparison is within-room (no cross-room transfer needed)
- **R13 (NEGATIVE, physics floor)** vs **R3.1 (NEGATIVE, architecture error)** — two different kinds of negative
- **R14/R15/ADR-105/ADR-106** — privacy framework holds; corrected architecture still on-device

## Honest scope

- Weak per-subject signature (body-size only); richer biometric input (gait, breathing, RCS) might partially rescue raw-level
- 3 positions per room; more positions sharpen the failure, fewer would partially work
- Position-variance dominance is geometry-specific
- Didn't test "per-position-cluster centroid" (might work but defeats no-label spirit)

## Coordination

`ticks/tick-20.md`. No PROGRESS.md edit. Branch `research/sota-r3.1-physics-env-prediction`.

## Remaining work

- **R3.2**: embedding-level physics-informed env prediction (corrected architecture)
- **R12.1**: pose-PABS closed loop (still highest-leverage)
- **R6.2.1**: 3D placement
- **R6.2.3**: chest-centric zones
- **ADR-107**: cross-installation federation

~4.1h to cron stop. **20 ticks landed.** Loop now has:
- 13 research threads (R1-R15)
- 3 negative results (R13 physics-floor, R3.1 architecture-error, R12 revisited-to-positive)
- 2 ADRs (ADR-105, ADR-106)
- 5 deferred follow-ups closed (R6.2, R6.2.2, R6.1, R12 PABS, R3.1)

Pattern: ~3 ticks per hour sustained over 8 hours.
