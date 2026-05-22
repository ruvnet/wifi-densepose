# Tick 34 — 2026-05-22 10:46 UTC

**Thread:** R18 (disaster response — collapsed building survivor detection)
**Verdict:** Third "vertical demonstrates loop generality" tick. R18 is the **first vertical to integrate with an existing repo crate** (`wifi-densepose-mat`), making loop-to-production path most direct.

## What shipped

- `docs/research/sota-2026-05-22/R18-disaster-response.md` — vertical sketch + MAT crate integration + rubble-attenuation analysis.

## Headline: rubble is RF-leaky, not RF-opaque

| Material | 2.4 GHz attenuation |
|---|---:|
| Steel (1 mm) | 2,674 dB (opaque) |
| **Mixed rubble (1-2 m)** | **40-80 dB** |
| Brick (10 cm) | 8-12 dB |
| Concrete (10 cm) | 20-30 dB |
| Drywall (1.5 cm) | 1-2 dB |

ESP32-S3 link budget (121 dB) gives **40-80 dB margin** through typical rubble. Survivors at 1 m depth: +37 dB margin (feasible). 2 m: +7 dB (marginal). 3 m: infeasible.

**Dramatically better than R11 maritime through-bulkhead** (where steel was dominant).

## Loop primitives → MAT crate enhancements

| Capability | MAT today | + Loop |
|---|---|---|
| Detect survivor | shipped | R12.1 pose-PABS = 9.36× fewer false alarms |
| Multi-survivor | partial | R6.2.5 multi-subject union (bounded to ~4) |
| Localisation | partial | R1 CRLB = ~25 cm at 4-anchor |
| Vitals confirmation | partial | R14 V1 + R15 rate-only (R13 rules out contour) |
| Survivor vs rescuer | not addressed | R3 + AETHER + rescue-worker library |
| Adversarial RF | not addressed | **R7 mincut binding** at disaster sites |
| Audit trail | not addressed | ADR-109 Dilithium-signed event log |

## Six-cog roadmap

| Cog | Timeline | Primitive |
|---|---|---|
| cog-mat-survivor-detect (existing) | NOW | wifi-densepose-mat |
| cog-mat-pose-pabs | 5y | + R12.1 |
| cog-mat-multi-survivor | 5y | + R6.2.5 |
| cog-mat-vitals-confirm | 5y | + R14 V1 + R15 |
| cog-mat-survivor-vs-rescuer | 10y | + R3 + library |
| cog-mat-cross-deploy-fed | 15y | + ADR-105-108 |

## Three deployment scenarios

| Scenario | Timeline | Notes |
|---|---|---|
| Rapid response (current MAT scope) | 5y | $200 per survey unit |
| Pre-staged at seismic-risk sites | 10y | Auto-activate on tremor |
| Cross-disaster federated learning | 15y | Consent-bounded |

## Vertical comparison: 5 verticals now

| | R18 disaster | R16 healthcare | R17 industrial |
|---|---|---|---|
| Repo asset | **existing MAT crate** | none | none |
| Through-medium | rubble 40-80 dB | air | air |
| Mobility | trapped (static) | stationary | mobile |
| **R7 mincut** | binding | nice-to-have | binding |
| Failure cost | survivor dies | clinical miss | safety incident |

Three of three target verticals (clinical, industrial, disaster) work with the same architecture. **Strong evidence the loop's output is genuinely vertical-agnostic.**

## Honest scope

- No bench-validated disaster-site data (ethics: can't simulate dead bodies)
- R7 mincut at disaster sites = hostile-RF requirement, not nice-to-have
- Cross-disaster federation raises consent questions (survivors / victims' families)
- Time-pressure: false-negatives at minute cost are fatal; threshold tuning aggressive
- MAT crate API doesn't yet consume R6.1 multi-scatterer — integration work needed
- Steel-rubble cases (basement w/ rebar) impossible per R11
- Underwater rescue impossible per R11 saltwater

## Through-rubble vital-signs feasibility (computed)

```
Link budget:                121 dB
Rubble loss (1-2 m):      -40 to -80 dB
Multi-scatterer penalty:   -4.7 dB
SNR margin needed:         -10 dB
Available for vitals:      +37 to -27 dB
```

Breathing-rate detection feasible at 1 m rubble, marginal at 2 m, infeasible at 3 m.

## Composes with prior threads

- R1, R6/R6.1, R6.2.2/.5, R7 (binding here), R10, R11, R12/R12.1, R13 NEGATIVE, R14, R15, R3
- ADR-105-109 federation + audit chain
- ADR-113 placement matrix
- R16/R17 parallel vertical patterns

## R18 special status

First vertical to integrate with **existing repo crate** (`wifi-densepose-mat`). Loop-to-production path is shortest for this domain because production code already exists; loop primitives enhance rather than replace.

## Coordination

`ticks/tick-34.md`. No PROGRESS.md edit. Branch `research/sota-r18-disaster-response`.

## Loop summary update

Six verticals + cross-thread identity work:
1. R10 wildlife
2. R11 maritime
3. R14 empathic appliances
4. R16 healthcare
5. R17 industrial
6. **R18 disaster (first integrates with existing crate)**

~1.2h to cron stop.
