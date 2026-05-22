# 01 — Physics-floor primitives

Bedrock physics that bounds everything else in the loop. Three primitives:

## Scripts

| Script | Thread | Headline |
|---|---|---|
| `r1_toa_crlb.py` | R1 | 20 MHz HT20 @ 20 dB SNR ToA CRLB: 41 cm single-shot, 4 cm with 100× averaging. Phase vs ToA: 238× advantage with cycle-slip resolution. |
| `r6_fresnel_zone.py` | R6 | First-Fresnel envelope at 5 m link, 2.4 GHz: 40 cm wide ellipsoid at midpoint. Per-subcarrier phase predictions for 4 canonical scatterer scenarios. |
| `r6_1_multiscatterer.py` | R6.1 | 6-scatterer human body model. Multi-scatterer penalty: **+4.7 dB** worse than idealised single-scatterer (matches R13's 5-dB shortfall to 0.3 dB). |

## Why this folder bounds the rest

- **R1 CRLB** sets the temporal-resolution floor for any localisation feature.
- **R6 Fresnel** gives the spatial envelope of CSI sensitivity (~40 cm wide at 5 m link).
- **R6.1 multi-scatterer** extends R6 from point-scatterer to realistic distributed body; quantifies the gap between idealised and real physics.

Together: physics floors that bound R6.2 family (placement), R12 family (structure detection), R14 (vitals), R20 (quantum integration).

## Sample output

```
=== R6 first Fresnel radii (m) ===
 freq   lambda   link  p=0.10  p=0.25  p=0.50  p=0.75  p=0.90
  2.4 124.9mm  5.0m   0.237   0.342   0.395   0.342   0.237

=== R6.1 multi-scatterer penalty ===
  Single-scatterer ideal:  +23.7 dB
  Multi-scatterer (6 body parts): +19.0 dB
  Penalty: +4.7 dB
```

## Honest scope

- All numbers are best-case physics; real CSI has additional noise channels.
- Body model is 6 point-scatterers; real body is distributed continuous RCS.
- 2D (top-down) approximations; 3D extensions live in `02-placement/`.

## See also

- Loop research notes: `docs/research/sota-2026-05-22/R{1,6,6_1}-*.md`
- Used by: `02-placement/`, `03-spatial-intelligence/`, `06-structure-detection/`, `09-quantum-fusion/`
