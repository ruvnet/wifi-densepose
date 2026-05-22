# Tick 11 — 2026-05-22 06:01 UTC

**Thread:** R13 (contactless BP) — **NEGATIVE RESULT**
**Verdict:** Don't pursue contactless BP from CSI as a primary product feature. The physics floors make it provably worse than a $20 arm cuff at every dimension.

## What shipped

- `examples/research-sota/r13_bp_physics_floor.py` — pure-numpy quantification of four physics floors that defeat the published CSI-BP approach.
- `examples/research-sota/r13_bp_results.json` — machine-readable predictions.
- `docs/research/sota-2026-05-22/R13-contactless-bp-negative.md` — explicit negative-result scrutiny note.

## Four floors quantified

| Floor | Need | Have | Gap |
|---|---|---|---|
| PTT temporal resolution | 0.5 ms (for 1 mmHg) | 10 ms typical, 1 ms max | typical ESP32 deployment cannot do <20 mmHg |
| Spatial separation of two body sites | 55 cm | 40 cm Fresnel at 5 m link | sites CANNOT be resolved by single link |
| Pulse-contour SNR | +25 dB | +20 dB after bandpass | **5 dB short** |
| Vs $20 arm cuff | ±2 mmHg | best published ±10 mmHg | **5× worse** |

The cleanest result: pulse signal motion at the chest is **0.3 mm**, breathing is **8 mm** — 27× larger. After bandpass we recover rate (we already ship this) but cannot recover waveform shape, which is what BP estimation needs.

## Why this is the most valuable kind of tick

A research loop that only publishes successes biases toward overclaiming. Two negative results this loop:

1. **R12 eigenshift** — naive SVD-spectrum approach fails because signal doesn't dominate drift floor
2. **R13 contactless BP** — published approaches require unrealistic SNR and spatial resolution

Both follow the same pattern: a plausible-sounding ML approach fails because the underlying signal doesn't dominate the noise. Both have explicit follow-up paths if anyone wants to revisit (R12 → PABS over Fresnel basis from R6; R13 → bed-instrumented `cog-bedside` niche, multistatic PWV with 6+ anchors).

## Confirms R14's design choice

R14 (empathic appliances) explicitly assumed BP would *not* be available — its V1/V2/V3 sketches depend only on breathing + HR rate + motion intensity. R13 confirms that assumption is right.

## What's still open in the negative space

Three niche scenarios where BP-from-CSI *might* close some day:
1. Single-subject **trend** monitoring (relative not absolute)
2. Bed-instrumented controlled-still subject (25+ dB SNR achievable)
3. Multistatic PWV with 6+ anchors + per-installation calibration

The general "BP from a $9 ESP32 in the corner" claim does not close.

## Composes with prior threads

- **R1** (CRLB) — confirms temporal-resolution floor for PTT
- **R6** (Fresnel) — provides the spatial floor that defeats two-site PTT
- **R5** (saliency) — band-spread occupancy explains why the whole chest is observed but the 0.3 mm pulse isn't
- **R12** — loop's other negative result; same failure pattern

## Coordination

`ticks/tick-11.md`. No PROGRESS.md edit. Branch `research/sota-r13-contactless-bp-negative`.

## Remaining threads

R3 (cross-room re-ID), R4 (federated learning), R15 (RF biometric across rooms).

~6.0h to cron stop. 11 threads landed (2 explicit negative results).
