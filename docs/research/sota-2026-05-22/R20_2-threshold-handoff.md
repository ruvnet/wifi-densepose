# R20.2 — Threshold-based hand-off: mixed result reveals production gap

**Status:** implementation of R20.1's catalogued refinement; mixed result reveals harmonic-rejection requirement · **2026-05-22**

## What R20.2 set out to fix

R20.1's naive precision-weighted Bayesian gave 84 BPM for HR when classical (105 BPM, 38% conf) disagreed with NV @ 1 m (72 BPM, 64% conf). The fix specified: when NV confidence > 60% AND amplitude > 3 pT, trust NV entirely.

## Result (5 distances)

| Distance | NV amp | NV rate | NV conf | Naive | Smart | Error (smart) | Regime |
|---:|---:|---:|---:|---:|---:|---:|---|
| **0.5 m** | 50.00 pT | 72.00 ✓ | 84% | 82.3 | **72.0** | **+0.0** ✓ | nv_drives |
| 1.0 m | 6.25 pT | 144.00 ✗ harmonic | 67% | 129.9 | **144.0** | **+72.0 ✗** | nv_drives |
| 1.5 m | 1.85 pT | 72.00 ✓ | 39% | 88.3 | 88.3 | +16.3 | weighted_fallback |
| 2.0 m | 0.78 pT | 77.00 | 36% | 91.5 | 91.5 | +19.5 | weighted_fallback |
| 3.0 m | 0.23 pT | 78.00 | 38% | 91.5 | 91.5 | +19.5 | weighted_fallback |

## What this reveals

- **At 0.5 m**: threshold hand-off works perfectly (+0.0 error, NV trusted, breathing+HR correct)
- **At 1 m**: smart hand-off **loses** to naive because the simple FFT picked a 2× harmonic of the true HR (144 vs 72)
- **At 1.5-3 m**: falls back to weighted (NV below confidence threshold), same as naive

## The production lesson

The threshold-based policy is **correct in spirit** (trust NV when good) but **incorrect with simple FFT** (which picks harmonics for narrow-band signals). Production needs:

1. **Harmonic rejection** in the rate estimator (e.g. autocorrelation-based, or Pan-Tompkins QRS for cardiac signals)
2. **Cross-check with classical breathing rate band** (true HR is rarely > 2× breathing rate × 6; the 144 result violates this and could be rejected)
3. **Per-frame plausibility window** (a healthy adult won't transition from 72 to 144 BPM in 1 second)

R20.1's note already flagged "production needs Pan-Tompkins QRS detection". R20.2 confirms this is **binding, not nice-to-have** for the threshold hand-off to be safe.

## What R20.2 DOES enable

1. **Empirical confirmation** that the smart hand-off works at 0.5 m bedside (target deployment scenario per ADR-114).
2. **Identification of a critical production gap**: harmonic rejection in the rate estimator is mandatory before threshold hand-off can ship.
3. **Refined ADR-114 implementation budget**: add ~30-50 LOC for Pan-Tompkins QRS detection.

## What R20.2 DOES NOT enable

- A clean win across all distances — the 1 m harmonic shows real-world robustness needs more work.
- Validation on real cardiac signals (synthetic Gaussian-pulse-train; real ECG/cardiac-B has different harmonic structure).
- Multi-subject hand-off (single subject only).

## Honest scope

This is a **mixed result, honestly reported**. The smart hand-off is right in principle; the FFT rate estimator beneath it is the weak link. Production fix is well-understood (Pan-Tompkins or autocorrelation), but the demo as written doesn't include it.

## Composes with

- R20.1 (this is the catalogued refinement)
- ADR-114 (production implementation needs Pan-Tompkins per R20.2)
- R13 NEGATIVE (this confirms classical HR is unusable, which is why we need NV at all)
- Doc 16 (cube-of-distance: at 3 m NV is below threshold and we fall back to weighted)

## Honest meta-observation

R20.2 is the **5-minute follow-up** to R20.1. The catalogue-then-revisit pattern works: R20.1 flagged production gap; R20.2 attempted the fix; the attempt surfaced a deeper gap (harmonic rejection). Three layers of refinement in one quantum integration arc.

## Connection back

R20 (vision, tick 37) → Doc 17 (bridge, tick 38) → ADR-114 (spec, tick 39) → R20.1 (working demo, tick 40) → **R20.2 (threshold refinement, this tick)**.

Five-step quantum integration arc. Production ADR-114 cog now has all known refinements catalogued before any Rust code is written.
