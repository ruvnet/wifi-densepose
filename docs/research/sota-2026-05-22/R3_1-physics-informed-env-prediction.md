# R3.1 — Physics-informed env_sig prediction at raw-CSI level: NEGATIVE (with a clear path forward)

**Status:** experimental result + scope correction · **2026-05-22**

## The plan

R3 (tick 12) showed MERIDIAN env-centroid subtraction recovers cross-room re-ID accuracy in the **AETHER embedding space**, but requires labelled examples *in the new room*. R3's "next research lever":

> Use R6.1 forward operator + a coarse room map to PREDICT the env_sig without labelled examples — zero-shot transfer.

R6.1 (tick 18) shipped the multi-scatterer Fresnel forward operator. This tick implements the predicted-env approach at the **raw CSI level** (not the embedding level) and benchmarks it against R3's labelled MERIDIAN oracle.

## Result

Two synthetic rooms (5×5 m diagonal link vs 4×6 m different link), 10 subjects with 0.85-1.15× body-size variation, 3 positions per room:

| Configuration | 1-shot K-NN accuracy |
|---|---:|
| Within-room 1 baseline | **100%** |
| Within-room 2 baseline | **100%** |
| Cross-room raw (no env subtraction) | 10% (= chance) |
| Cross-room **labelled MERIDIAN** (oracle) | **10% (= chance)** |
| Cross-room physics-informed env prediction | 10% (= chance) |

**All three cross-room approaches collapse to chance.** Not just the physics-informed one — even the labelled MERIDIAN oracle fails. This is meaningfully different from R3's tick-12 result where labelled MERIDIAN reached 100%.

## Why R3 worked but R3.1 doesn't

R3 was simulated on a **128-dim AETHER-style embedding space** where:
- person_signature, environment_signature, and noise were in independent random directions
- env_sig was a single fixed vector per room (no within-room positional variance)
- cosine normalisation partially absorbed the env shift

R3.1 is at the **raw CSI level (52-dim complex)** where:
- Subjects move to 3 positions per room — each position has its own complex CSI signature
- Per-position variance within a room can exceed per-subject variance between rooms
- Subtracting a single per-room centroid removes the *mean* position but not the *variance*

The headline gap: **AETHER embedding space invariantises over within-room position**; raw CSI does not. **The cross-room problem at raw-CSI level is fundamentally harder than at the embedding level.**

## The honest takeaway

| What R3 showed | What R3.1 shows |
|---|---|
| Cross-room re-ID works in embedding space with MERIDIAN | Cross-room re-ID **doesn't** work at raw-CSI level |
| Labelled centroid subtraction is enough | Labelled centroid subtraction is **not** enough at raw CSI |
| Physics-informed prediction is a worthwhile next step | Physics-informed prediction at raw-CSI level is **also not enough** |

This is a **third honest negative result** for the loop (alongside R13 contactless BP and R12 NEGATIVE pre-PABS). The negative pattern: any cross-room method at raw-CSI level fails because position-variance is the dominant source of within-room CSI variation.

## The path forward

The physics-informed env prediction approach is *not dead* — it just needs to be **applied at the embedding level, not the raw-CSI level**. The corrected architecture:

```
raw CSI → AETHER embedding head (position-invariant) → physics-informed env subtraction → cross-room K-NN
```

Or equivalently: subtract the physics-predicted env_sig **from the AETHER head's output**, not from the raw input. AETHER already does the heavy lifting of invariantising over position; the physics-informed prediction then has only the room-shift component to remove.

This requires AETHER (ADR-024) to be trained or fine-tuned, which is out of scope for this loop. **The implementation roadmap is now clear:**

1. AETHER head fine-tuned per-installation (ADR-024 baseline)
2. Physics-informed env_sig from R6.1 forward operator + room map
3. Subtract (2) from (1)'s output → invariantised embedding
4. K-NN matching across rooms with no labels in the new room

R3.1 says: the **physics-informed prediction must be applied in the right space**. The raw-CSI experiment exposes that the wrong space gives no lift.

## Composes with prior threads

- **R3** (cross-room re-ID) — R3.1 confirms R3's MERIDIAN-in-embedding-space result by showing the *raw-CSI* version fails. R3's choice to operate in embedding space was correct.
- **R6.1** (multi-scatterer Fresnel) — provides the forward operator. R3.1 used it; the operator is correct; the application level was wrong.
- **R12 PABS** (POSITIVE) — operates on raw CSI directly *but doesn't compare across rooms*. PABS detects structural changes *within* a room; cross-room transfer needs an additional invariance layer (= AETHER).
- **R14 / R15 / ADR-105** — the privacy framework still holds; AETHER + physics-env-prediction stays on-device per ADR-106.

## Why this negative result is still useful

1. **Surfaces an architecture error before implementation.** Without this tick, a future engineer might attempt the obvious "subtract predicted env from raw CSI" approach and waste weeks. R3.1 documents that this fails.
2. **Tightens the R3 implementation roadmap.** The corrected architecture is now explicit.
3. **Demonstrates the difference between embedding-space and raw-space approaches.** This generalises beyond R3 — it informs every "subtract a learned/predicted nuisance" pattern in the codebase.

## Honest scope

- 10 subjects with 0.85-1.15× body-size variation is a deliberately weak per-subject signature. Stronger biometric primitives (gait, breathing, RCS from R15) would give larger per-subject contrasts. The "raw CSI level fails" finding might be sensitive to this scale; with richer biometric input the raw-level approach might recover.
- The simulation uses 3 positions per room. With more positions (5-10), the failure would be sharper. With fewer (1), it would partially work.
- Position-variance dominance is geometry-specific. Long-narrow rooms vs square rooms have different ratios; this is one geometry.
- We didn't test "labelled MERIDIAN per-position-cluster" (cluster positions within a room, subtract per-cluster centroid). That might work for the labelled oracle; physics-informed equivalent would need a position-clustering layer.

## What this DOES enable

- **A negative result** that prevents wasted implementation effort.
- **A corrected architecture sketch**: physics-informed env prediction at the embedding level (not raw level).
- **A reference benchmark** showing that the cross-room problem at raw-CSI level is genuinely hard, contextualising R3's embedding-level result.

## What this DOES NOT enable

- The originally hoped-for zero-shot cross-room re-ID. That still needs the embedding-level implementation (R3.2, future).
- Any improvement to the existing within-room re-ID (which already works).
- Cross-installation re-ID — still prohibited by R3 + R14 + R15 + ADR-106.

## What's next

- **R3.2**: embedding-level physics-informed env prediction (corrected architecture). Requires AETHER + R6.1 integration; out of scope for this loop.
- **R12.1 (pose-PABS closed loop)** — still the highest-leverage next implementation.
- **ADR-107 (cross-installation federation)** — still deferred.

## Connection back

- **R3 (POSITIVE in embedding space)** — confirmed indirectly; raw-level failure shows why R3 operated at the embedding level.
- **R6.1** — operator is correct; application level was wrong.
- **R12 PABS (POSITIVE)** — operates in raw space for *structure detection* (no cross-room transfer needed). PABS works at raw level because the comparison is within-room.
- **R13 (NEGATIVE, physics floor)** + **R3.1 (NEGATIVE, architecture error)** — two different kinds of negative result: one is a physics wall (R13), the other is a fixable design choice (R3.1).

## Three kinds of negative result this loop has produced

This tick is the third honest negative — and the loop now has examples of all three categories:

1. **R12 NEGATIVE → POSITIVE** (revisited): missing tool (forward operator) blocked the right approach; tool became available later, approach worked.
2. **R13 NEGATIVE → permanent**: physics floor (5 dB shortfall) cannot be overcome by any tool; the negative is final.
3. **R3.1 NEGATIVE → architecture-error**: right idea, wrong application level; corrected architecture is now explicit but not yet implemented.

Knowing which category a negative result falls into is itself a research contribution. R3.1 sits in category 3.
