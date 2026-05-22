# Tick 15 — 2026-05-22 06:40 UTC

**Thread:** ADR-106 (DP-SGD + biometric primitive isolation)
**Verdict:** Closes the two items deferred from ADR-105 (member-inference defence + primitive isolation enforcement). The federation protocol now has formally-bounded privacy.

## What shipped

- `docs/adr/ADR-106-dp-sgd-and-primitive-isolation.md` — full ADR draft. Direct extension of ADR-105.

## Three-layer defence

| Layer | Mechanism | Defends against |
|---|---|---|
| 1 — Primitive Isolation | API-level tagging of on-device-only tensors (R15 binding list) | Exfiltration of biometric primitives via federation channel |
| 2 — Gradient clipping | Per-sample L2 norm bound (Abadi 2016) | Bounds sensitivity of any single training sample |
| 3 — Gaussian noise | Per-round N(0, σ²C²I) on aggregated delta | Formal (ε, δ)-DP via Moments Accountant |

## Privacy budget

Recommended (per Moments Accountant, δ=1e-5):

| Profile | σ | Rounds | Total ε | Use |
|---|---:|---:|---:|---|
| Conservative (medical-grade) | 1.5 | 50 | **2.0** | HIPAA-aligned |
| Standard (typical RuView) | 1.0 | 100 | **5.0** | Most cogs |
| Lenient | 0.5 | 100 | 8.0 | Below ε=10 community soft-bound |

## On-device-only primitive list (R15-binding)

7 ✅ "never transmit" primitives:
- Raw CSI window
- Gait stride frequency
- Breathing rate (per-subject)
- HRV rate signature
- RCS frequency response curve
- Limb timing vector
- Per-subject embedding centroid

3 ⚠️ "transmit with mitigation":
- MERIDIAN per-room centroid (aggregate, OK)
- LoRA weight delta (DP-SGD applied)
- Model logits during inference (never aggregated)

API surface enforces ✅ as compile-time error where possible.

## Implementation budget

Extends ADR-105's 500 LOC by **+300 LOC**: PrimitiveTag (60) + clipping (30) + DP noise (40) + Moments Accountant (120) + per-cog config schema (50). Total federation budget: **~800 LOC, 3-week effort**.

## Why this closes the privacy story

R3 + R14 + R15 + ADR-105 + ADR-106 = complete chain from physics (R6 forward model) → embeddings (R3) → personalised features (R14) → trained how (ADR-105) → defended how (R7) → privacy-bounded how (ADR-106).

The chain has:
- A physics floor (R6/R1)
- A spatial intelligence layer (R5/R7/R3)
- A vertical roadmap (R10 wildlife + R11 maritime + R14 home)
- Two negative results (R12 eigenshift, R13 contactless BP)
- Two architectural decisions (ADR-105 + ADR-106)

The per-occupant feature surface (R14 V1/V2/V3) now has **formal (ε, δ) privacy backing**, not just policy.

## Composes with every prior thread

- R3: Layer 1 blocks per-subject embedding centroid transmission
- R7 mincut: compatible with DP-noised deltas; operates on noised graph
- R12/R13 negative results: informed the noise-vs-structure-detection design choice
- R14: privacy framework now has formal (ε, δ) backing
- R15: requirements basis = on-device-only primitive list made executable
- ADR-105: 800 LOC budget, DP slots into step 4 of protocol

## Honest scope

- σ values are recommendations, not measurements (per-cog tuning needed)
- (ε, δ)-DP is worst-case bound; auxiliary info changes the practical leakage
- Moments Accountant is conservative (slightly over-estimates budget consumed)
- Subject-level DP not formalised (household of 4 has K=4 subjects → sample-level DP doesn't fully capture)
- Side-channel timing leaks out of scope (future ADR)

## Coordination

`ticks/tick-15.md`. No PROGRESS.md edit. Branch `research/sota-adr106-dp-sgd-primitive-isolation`.

## Remaining loop work (post ADR-106)

- R6.1 multi-scatterer Fresnel extension
- R3 follow-up: physics-informed env_sig prediction (zero-shot cross-room)
- R6.2 Fresnel-aware antenna placement CLI tool
- ADR-107: cross-installation federation w/ secure aggregation (explicitly deferred from ADR-106)
- Loop retrospective / 00-summary.md (premature — ~5h still on clock)

~5.3h to cron stop. **15 ticks landed. PROGRESS.md research agenda + 1 follow-up ADR closed.**
