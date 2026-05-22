# 07 — Negative results (R13 contactless BP)

**Productive failure**: empirical / physics-based scrutiny of widely-claimed but un-validated CSI capabilities.

## Scripts

| Script | Thread | Verdict |
|---|---|---|
| `r13_bp_physics_floor.py` | R13 | **Don't ship contactless BP from CSI as a primary RuView feature.** Four physics floors make it provably worse than a $20 arm cuff. |

## The four floors (R13)

| Floor | Need | Have | Gap |
|---|---|---|---|
| PTT temporal resolution | 0.5 ms (for 1 mmHg) | 10 ms typical, 1 ms max ESP32 | typical ESP32 deployment cannot do <20 mmHg |
| Spatial separation of two body sites | 55 cm | 40 cm Fresnel envelope at 5 m | sites NOT resolvable by single link |
| Pulse-contour SNR | +25 dB | +20 dB after bandpass | **5 dB short** (matches R6.1's 4.7 dB penalty) |
| Vs $20 arm cuff baseline | ±2 mmHg | best published ±10 mmHg | **5× worse** + needs per-subject calibration |

## Why R13 is sensor-bound, not physics-bound-period

R20 (tick 37) + doc 17 + ADR-114 establish that **the 5 dB shortfall is the multi-scatterer penalty** (R6.1). It's sensor-bound: a different sensor (NV-diamond magnetometer at bedside) recovers what CSI cannot.

| Sensor | Can detect HRV contour? | Can detect BP? |
|---|---:|---:|
| CSI alone (R13 NEGATIVE) | ❌ 5 dB short | ❌ same physics |
| **NV-diamond at 1 m bedside** (ADR-114) | ✅ SDNN 119 ms | ✅ via mm-PWV |
| Arm cuff (gold standard) | n/a | ✅ ±2 mmHg |

## R13's value in the loop

Categorising R13 as a **permanent physics-floor negative** initially saved engineering effort. Then R20 + doc 17 + ADR-114 recategorised it as **sensor-bound, recoverable**. This is the **research-loop pattern at its best**: explicit failure modes that survive scrutiny but get reclassified when new tools arrive.

R20.1 (quantum-fusion demo) is the concrete demonstration that R13's recovery works.

## Three niche scenarios where BP-from-CSI might close

1. Single-subject **trend** monitoring (relative not absolute)
2. Bed-instrumented controlled-still subject (25+ dB SNR achievable)
3. Multistatic PWV with 6+ anchors + per-installation calibration

The general "BP from a $9 ESP32 in the corner" claim does **not** close.

## See also

- Research notes: `docs/research/sota-2026-05-22/R13-contactless-bp-negative.md`
- Recovery path: `docs/research/sota-2026-05-22/R20-*.md`, doc 17, ADR-114
- The other 2 negative result categories: R12 (missing-tool, revisitable) in `06-structure-detection/`, R3.1 (architecture-error) in `05-cross-room-reid/`
