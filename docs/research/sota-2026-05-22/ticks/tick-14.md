# Tick 14 — 2026-05-22 06:32 UTC

**Thread:** R15 (RF biometric across rooms)
**Verdict:** Catalogues 5 environment-invariant biometric primitives in CSI with quantified discriminability + strengthens R14/R3/ADR-105 privacy framework. Closes the last unaddressed research-loop thread.

## What shipped

- `docs/research/sota-2026-05-22/R15-rf-biometric-primitives.md` — synthesis pulling from R5, R6, R8, R10, R13, R3, R14, ADR-105.

## Five biometric primitives inventoried

| Primitive | Bits/person | Cross-room invariance | Status |
|---|---:|:---:|---|
| Gait stride frequency | 5 | HIGH | shipped (R10 DSP) |
| Breathing rate + envelope | 5 | HIGH | shipped (vital_signs) |
| HRV (rate-level only) | 4 | HIGH at rate, LOW at contour | partial (R13 negative on contour) |
| Body-size RCS frequency response | 4 | MEDIUM (needs calibration target) | not built |
| Walking dynamics (limb timing) | 7 | HIGH (if pose works cross-room) | pose pipeline shipped, cross-room unmeasured |

**Composite biometric strength**: ~12-15 bits realistic (vs 25-bit independence upper bound). Enough for household + building-scale ID; insufficient for forensic / city-scale.

## Privacy framework strengthened

R15 makes a sharper point than R14/R3: **RF biometric is physical, not learned, so the same identification primitive that enables empathic appliances is also a surveillance primitive that's harder to opt out of than visual ID.**

| R3/ADR-105 baseline | R15-strengthened |
|---|---|
| No cross-installation linkage | Hardware-isolated, cryptographically proven |
| Embedding storage opt-in | Storage of any biometric primitive opt-in (not just embeddings) |
| Cryptographically verifiable forgetting | Forget raw primitives, not just outputs |
| No re-ID across legal entities | No sharing of any RF biometric primitive (including aggregate / derived) |

## ADR-105 amendment surfaced

Adds a constraint to ADR-105 federation:

> The federation aggregator MUST NOT receive any raw per-subject biometric primitive (gait frequency, breath rate, RCS curve, limb timing). It MAY receive aggregated, MERIDIAN-normalised model deltas. Per-subject primitives stay on-device.

This becomes the requirements basis for **ADR-106 (deferred DP-SGD ADR from ADR-105)**.

## Why R15 closes the loop

R15 is the last unaddressed PROGRESS.md thread. After R15:
- **Closed**: "what RF biometrics exist and how do they invariantise" has a worked answer
- **Open**: ADR-106, R6.1 multi-scatterer, R3 follow-up (physics-informed env_sig prediction), R6.2 antenna placement

The per-occupant feature surface (R14 V1/V2/V3) is now fully grounded in physics + constraints; remaining work is implementation, not research.

## Composes with every prior thread

- R5 saliency → primitive-specific saliency maps
- R6 Fresnel → physical basis for RCS frequency-response invariance
- R7 mincut → defends primitive-level poisoning
- R10 per-species gait taxonomy → transfers to per-individual gait biometric
- R13 NEGATIVE → 5-dB-short wall also rules out contour-level HRV
- R3 → embedding space combines the 5 primitives
- R14 → all 3 verticals (V1/V2/V3) work with the rate-level subset, no contour recovery
- ADR-105 → needs ADR-106 to formalise on-device-only primitive measurement

## Honest scope landed

- Bit counts are upper bounds; realistic 30-50% loss to noise/multipath/sensor variance
- Contour-level HRV not achievable (R13 wall)
- Walking-dynamics 7-bit assumes pose-from-CSI works cross-room (unmeasured)
- Body-size RCS needs calibration target in new room → ratio-only gives 3-4 bits not 5

## Coordination

`ticks/tick-14.md`. No PROGRESS.md edit. Branch `research/sota-r15-rf-biometric`.

## Remaining work (deferred to post-loop)

- **ADR-106**: on-device DP-SGD + primitive isolation requirements from R15
- **R6.1**: multi-scatterer additive Fresnel forward model
- **R3 follow-up**: physics-informed env_sig prediction (zero-shot cross-room)
- **R6.2**: Fresnel-aware antenna placement CLI tool

~5.4h to cron stop. **14 threads landed. PROGRESS.md research agenda exhausted.**

## Next-tick plan

Could either:
1. Pick up one of the deferred follow-ups (ADR-106 or R6.1 are the strongest)
2. Start consolidating into 00-summary.md (premature; loop has ~5h left)
3. Add a meta-analysis / loop retrospective tick

Recommend (1) on next tick — ADR-106 has clear requirements from R15 + ADR-105.
