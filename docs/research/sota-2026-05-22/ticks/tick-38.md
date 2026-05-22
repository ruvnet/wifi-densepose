# Tick 38 — 2026-05-22 11:20 UTC

**Thread:** Quantum-sensing series doc 17 (honest classical-quantum fusion)
**Verdict:** Bridges the existing 6-doc quantum-sensing series (docs 11-16) with this loop's 37+ ticks. Inherits doc 16's sober "no 40-mile cardiac magnetometry" posture.

## What shipped

- `docs/research/quantum-sensing/17-honest-classical-quantum-fusion.md` — synthesis document in the quantum-sensing series.

## Why this tick (user signal)

User opened `docs/research/quantum-sensing/11-quantum-level-sensors.md` **twice** in consecutive ticks. Strong repeat signal toward quantum integration. Inspecting the folder revealed a 6-doc series (11-16) that R20 (tick 37) didn't yet acknowledge. Doc 17 explicitly bridges the two work streams.

## The two reality-checks composing

1. **R13 NEGATIVE (loop tick 11)**: ruled out classical CSI BP/HRV-contour due to 5 dB shortfall
2. **Doc 16 Ghost Murmur (2026-04-26)**: ruled out 40-mile NV cardiac magnetometry due to cube-of-distance physics

Combined: **honest fusion adds NV-diamond cardiac magnetometry at 1-2 m bedside ranges** (where cube law gives ~1 pT/√Hz SNR), NOT 40 miles. The loop's classical primitives carry geometry; quantum carries fidelity.

## Five-cog fusion roadmap

| Cog | Series-anchor doc | Loop primitives | Timeline |
|---|---|---|---|
| cog-quantum-vitals (NV + CSI) | docs 13/14/15 (nvsim) | R14 V1 + R15 + NV HRV contour | 5y |
| cog-rydberg-anchor (calibrated multistatic) | doc 11.4 | R1 + R6.2.2 + Rydberg | 7-10y |
| cog-mm-position (atomic clock) | doc 11 | R1 + R3.2 + atomic clock | 10y |
| cog-deep-rubble-survivor (NV drone) | docs 13, 16 | R18 + NV-via-drone | 15y |
| cog-ICU-meg (room-temp SQUID) | doc 11.2.2 | R14 V3 + SQUID array | 20y |

## Cross-reference index

Every loop output mapped to a quantum-series doc:
- R13 NEGATIVE → doc 13 recovers HRV via NV
- R14 V3 → doc 13 + doc 11.2.2 SQUID for MEG
- R6.1 4.7 dB penalty → doc 11.3.3 quantum illumination (+6 dB)
- R1 CRLB → doc 11.4 Rydberg+atomic clock (~10 cm)
- R18 disaster → doc 13 NV cardiac at 5+ m rubble depth

Lets a reader navigate: "I'm interested in X loop finding; here's the quantum context that extends it."

## nvsim (ADR-089) integration concretised

Doc 17 specifies the code path from `nvsim` (currently a standalone leaf crate, WASM-ready) into production via the loop's primitives:

```
nvsim_output -> R14 V1 fusion / R12 PABS / R7 mincut / R6.1 residual basis
                                                       ↓
                                                cog-quantum-vitals
```

~150 LOC of glue. **This makes `nvsim` actually useful** beyond simulator scope.

## What this DOES enable

1. Clear integration between existing 6-doc series and SOTA loop
2. Five honest-scope fusion-cog roadmap items
3. "What we are NOT building" list (no 40-mile cardiac, no through-walls quantum)
4. Bridge for journalists / researchers / contributors

## What this DOES NOT enable

- 40-mile cardiac magnetometry (doc 16 stands)
- Through-multiple-walls quantum (1/r³ falloff persists)
- Replacement of medical devices without FDA/CE approval
- Quantum-enhanced WiFi protocol changes (Layer 1 stays classical)

## Composes with every loop output

R1, R3, R5-R15, R12.1, R13 NEGATIVE (recovered via NV), R14 V1/V3, R15, R16-R20 verticals, ADR-105-109, ADR-113. Plus all 6 quantum-sensing docs (11-16).

## Doc 17 special status

- First doc to bridge the SOTA loop (2026-05-22) with the quantum-sensing series (2026-03-08 onwards)
- Adopts doc 16's sober reality-check posture
- Identifies which loop NEGATIVE results are conditionally recoverable via quantum (R13)
- Concretises the `nvsim` → cog integration path

## Coordination

`ticks/tick-38.md`. No PROGRESS.md edit. Branch `research/sota-quantum-doc17-fusion`.

## Loop status (38 ticks, ~40 minutes to cron stop)

- 18 research threads (R1, R3, R5-R15, R16-R20)
- 8 exotic verticals + this cross-series synthesis
- 6 loop ADRs + 3 existing + 3 referenced from quantum series
- 3 negative result categories (R13 conditionally recovered via R20+doc 17)
- Production roadmap + quantum-classical fusion roadmap both shipped

00-summary.md to follow at 12:00 UTC stop.
