# Tick 33 — 2026-05-22 10:31 UTC

**Thread:** R17 (industrial safety) — second new exotic vertical
**Verdict:** Industrial vertical composes the same loop primitives as R16 healthcare, with different ADR-113 matrix rows (presence + vital-signs at coarser resolution) and R7 mincut **becomes binding** rather than nice-to-have due to hostile industrial RF.

## What shipped

- `docs/research/sota-2026-05-22/R17-industrial-safety.md` — full vertical sketch + R16 parallel comparison.

## Three deployment scenarios

| Scenario | Timeline | Cost vs status quo |
|---|---|---|
| Warehouse zone (100 m²) | 5y | $80/zone vs $500-2000 camera + monitoring |
| Construction site | 10y | per-project federation |
| Refinery / chemical plant | 15y | adds CSI to existing gas + cam + badge infrastructure |

## R17 vs R16 parallel

| | R16 healthcare | R17 industrial |
|---|---|---|
| Subjects | patients | workers |
| Mobility | stationary | mobile |
| Coverage | 30 m² ward | 100-1000 m² zone |
| ADR-113 row | vital-signs (chest, N=5) | presence (body, N=3-4) |
| Privacy regime | HIPAA / FDA | OSHA / employment |
| **R7 mincut** | nice-to-have | **binding** |
| Failure cost | missed clinical event | missed safety event |

**Same architecture, different parameter regime.** Loop's primitives form a **vertical-agnostic infrastructure layer**.

## Five specialised cog roadmap items

| Cog | Timeline | Primitive |
|---|---|---|
| cog-fall-detection | 5y | R12.1 + PPE-tuning |
| cog-zone-occupancy | 5y | R12 PABS + R6.2.5 |
| cog-lone-worker-vitals | 5y | R14 V1 (rate-only per R13) |
| cog-worker-fatigue | 10y | R10 gait + R15 |
| cog-multi-zone-orchestrator | 5y | R6.2.5 + ADR-105 fed |

## Why R7 mincut becomes binding

Industrial RF environment has legitimate noise (cell, BLE tools, walkie-talkies) that must be disambiguated from sensor compromise. R7 Stoer-Wagner mincut on N ≥ 4 anchors is the only defence; ADR-113 already requires N ≥ 4 for multi-feature cogs, which conveniently satisfies the industrial requirement.

## PPE-specific body model needed (R6.1 follow-up)

Construction PPE (hard hat, high-vis vest, safety harness, tool belt, steel-toed boots) changes per-part reflectivity by ~5-15%. ~1-2 weeks of labelled-data work for `cog-industrial-pose`.

## R10 gait + worker fatigue (10y mid-term)

R10's gait taxonomy extends within humans:
- Walking 1.2-2.5 Hz
- Fatigued walking 0.8-1.5 Hz (slower + asymmetric)
- Impaired walking: asymmetry > 25%

OSHA-aligned: pre-incident detection of worker fatigue via gait drift over a shift.

## Honest scope

- Synthetic data only; bench validation required for OSHA-grade claims
- PPE-specific body model unbuilt (R6.1 body model is bare-clothed)
- Outdoor / weather effects partly transfer from R10 foliage model
- Worker consent operational, not architectural
- Liability + insurance for missed-event failures outside this scope
- Audit trail integration with SAP / Maximo / etc. is per-customer

## R17 closes the parallel-vertical demonstration

After R17, the loop has demonstrated **vertical-agnostic infrastructure**: same primitives → R10 wildlife / R11 maritime / R14 home empathic appliances / R16 healthcare / **R17 industrial**. Outputs that generalise beyond original problems is the mark of well-factored research.

## Composes with every loop thread

- R1, R5, R6/R6.1, R6.2.5, R7 (binding here), R10, R12/R12.1, R13 NEGATIVE, R14, R15
- ADR-113 (placement matrix), ADR-105-109 (full privacy + PQC chain)
- R16 (parallel pattern)

## Coordination

`ticks/tick-33.md`. No PROGRESS.md edit. Branch `research/sota-r17-industrial-safety`.

## Loop summary update

Five exotic verticals + cross-thread identity work:
1. R10 wildlife (animal conservation)
2. R11 maritime (vessel safety + crew monitoring)
3. R14 empathic appliances (home)
4. R16 healthcare ward
5. **R17 industrial safety**

~1.4h to cron stop.
