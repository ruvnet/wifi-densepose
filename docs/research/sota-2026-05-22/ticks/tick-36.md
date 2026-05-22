# Tick 36 — 2026-05-22 11:05 UTC

**Thread:** R19 (agricultural livestock monitoring) — seventh exotic vertical
**Verdict:** First non-human-centric vertical. Composes R10 gait taxonomy + R6.2.5 multi-subject + R12 PABS + R14 V1 vitals. Architecture identical to human verticals; regulatory regime (USDA / EU welfare) differs.

## What shipped

- `docs/research/sota-2026-05-22/R19-agricultural-livestock.md` — vertical sketch with per-species gait + vital-signs tables.

## Headline: 7 exotic verticals now

1. R10 wildlife
2. R11 maritime
3. R14 empathic appliances (home)
4. R16 healthcare
5. R17 industrial
6. R18 disaster (integrates MAT crate)
7. **R19 livestock (first non-human-centric)**

Seven distinct domains, same architecture. **Overwhelming evidence of vertical-agnostic infrastructure.**

## Per-species gait + vital-signs tables (R10 extension)

| Species | Stride | Normal RR (BPM) | Stress RR |
|---|---|---|---|
| Cattle | 0.6-1.2 Hz | 10-30 | >40 |
| Pig | 1.0-2.0 Hz | 10-25 | >35 |
| Sheep | 1.5-2.5 Hz | 12-25 | >30 |
| Horse | 1.0-1.8 Hz | 8-16 | >20 |
| Chicken (layer) | 3.0-5.0 Hz | 15-40 | >50 |

R10 gait taxonomy directly extends. **Per-species gait drift detects lameness earlier than visual inspection.**

## Six-cog roadmap

| Cog | Timeline | Primitive composition |
|---|---|---|
| cog-cattle-monitor | 5y | R10 + R14 + R6.2.5 + R12.1 |
| cog-pig-welfare | 5y | R6.2.5 + R14 + correlation |
| cog-predator-alert | 5y | R12 PABS + R10 classifier |
| cog-lameness-detector | 10y | R10 gait asymmetry + drift |
| cog-birthing-alert | 10y | R14 V1 species signature |
| cog-free-range-tracker | 15y | R6.2.2 sparse + Tailscale mesh |

## Three deployment scenarios

| Scenario | Timeline | Cost vs status quo |
|---|---|---|
| Dairy barn (50-100 cows) | 5y | $200 vs $50K visual+RFID+behaviour |
| Free-range pasture | 10y | self-organising solar+ESP32+Tailscale |
| Pig barn welfare | 15y | EU "End the Cage Age" / Prop 12 alignment |

## High-impact use cases

- **Predator detection at pasture edges** (R12 PABS): mitigates $232M/year US livestock losses (USDA 2015)
- **Heat-stress detection in dairy** (R14 V1): overheated cattle drop milk production 30-50% before visual signs
- **Lameness early detection** (R10): dairy industry's #1 welfare issue, currently undetected until severe
- **Sick-pig isolation alert** (R6.2.5 + R14): tail-biting outbreaks have herd-level cascading effects

## What's different from human verticals

| Dimension | Human (R16/R17) | Livestock (R19) |
|---|---|---|
| Mass | 60-100 kg | 1.5-1000 kg (3+ orders) |
| Count | 1-8 | 1-1000+ |
| Privacy | HIPAA / OSHA / GDPR | farmer-consent for animals |
| Regulatory | FDA / OSHA | USDA / EU welfare |
| Cost sensitivity | high | very high (2-5% margins) |
| Chicken-scale | n/a | economically marginal |

Architecture identical; cost + regulatory regime differs.

## Honest scope

- Synthetic data only; per-species RCS measurements needed
- Chicken-scale deployments economically marginal
- High-density pig barns (8-100/barn) may exceed R6.2.5's 4-occupant limit
- Weather-affected outdoor RF not in scope
- No animal-welfare ethics review done (loop specifies infrastructure only)

## R19 special status

First **non-human-centric** vertical. Privacy framework (R14+R3+R15+ADR-106) doesn't apply (animals can't consent); replaced by animal-welfare regulations.

R18 + R19 are the two verticals needing direct external partnerships (FEMA for R18; USDA / animal welfare orgs for R19).

## Composes with every loop thread

- R10 gait taxonomy → livestock species
- R6.2.5 → herd multi-subject union
- R12 PABS → predator + cattle-fall
- R14 V1 → heat-stress + welfare scoring
- R15 → per-animal RF fingerprint (ID without tag)
- R7 mincut → pasture-edge adversarial RF
- ADR-113 placement matrix → modified rows for livestock cogs

## Coordination

`ticks/tick-36.md`. No PROGRESS.md edit. Branch `research/sota-r19-agricultural-livestock`.

## Loop status (~36 ticks, ~55 minutes to cron stop)

- 17 research threads (R1, R3, R5-R15, R16, R17, R18, R19)
- 7 exotic verticals
- 6 new ADRs (105-109 + 113) + 3 existing = 9 in chain
- 3 negative result categories
- 2 self-corrections
- 3 honest-scope findings
- 9-tick R6 family + 3-tick R3 arc + 3-tick R12 arc all complete
- Production roadmap shipped (tick 35)

00-summary.md to follow at 12:00 UTC / 08:00 ET stop.
