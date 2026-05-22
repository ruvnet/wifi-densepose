# 08 — Exotic vertical physics

Concrete physics for two vertical applications: wildlife (through-foliage) and maritime (through-bulkhead). Other verticals (R14 empathic, R16 healthcare, R17 industrial, R18 disaster, R19 livestock, R20 quantum) are vision/spec only — only R10 and R11 have computable physics demos.

## Scripts

| Script | Thread | Vertical | Headline |
|---|---|---|---|
| `r10_foliage_attenuation.py` | R10 | Wildlife sensing through foliage | ITU-R P.833-9 attenuation. ESP32 sparse-foliage range: **~100 m at 2.4 GHz** (later corrected to ~70 m by R6's Fresnel-clearance consideration). Per-species gait taxonomy (8 species). |
| `r11_maritime_propagation.py` | R11 | Maritime / ship-cabin sensing | Steel skin depth 3.25 µm at 2.4 GHz → through-bulkhead impossible. But **slot diffraction works**: cabin door with 2 mm gasket gap = +31 dB SNR margin. **Through-seam, not through-bulkhead.** |

## What R10 (wildlife) enables

- Solar-powered ESP32 nodes at forest edges
- Through-foliage detection at ~70-100 m range
- Per-species gait classification:

| Species | Stride frequency |
|---|---|
| Bear / sloth / wild boar | 0.5-1.5 Hz |
| Human walking | 1.2-2.5 Hz |
| Deer / fox | 1.8-4.5 Hz |
| Squirrel / mouse / songbird | 4.0-15.0 Hz |

The gait taxonomy directly extends to R19 livestock (cattle 0.6-1.2 Hz, pig 1.0-2.0 Hz, etc.).

## What R11 (maritime) enables

- Man-overboard surface detection at ~200 m
- Through-seam crew vitals (lone-watch monitoring)
- Container tamper detection via 30 mm vent slots
- Hatch-seal integrity audit (predictive maintenance)

**What R11 rules out**:
- Through-hull submarine sensing (steel impassable)
- Underwater sensing at WiFi bands (saltwater 853 dB/m)

## Composition with privacy + federation chain

Both R10 and R11 use:
- `cog-wildlife` or `cog-maritime-watch` packaging (ADR-100)
- Cross-installation federation (ADR-107 + ADR-108 PQC)
- Per-installation embedding spaces (R3 + R14 privacy framework)

For wildlife the privacy framework is largely moot (animals don't consent); for maritime the crew-consent framework applies.

## See also

- Research notes: `docs/research/sota-2026-05-22/R10-*.md`, `R11-*.md`, plus R16-R20 verticals (vision-only)
- Composes with: `01-physics-floor/` (forward model), `06-structure-detection/` (PABS for predator-detection / container-tamper)
