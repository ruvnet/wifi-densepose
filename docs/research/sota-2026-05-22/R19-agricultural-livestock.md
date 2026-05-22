# R19 — Agricultural livestock monitoring: barns + free-range + welfare

**Status:** seventh exotic vertical · **2026-05-22**

## Premise

Livestock farming is enormous (~80B animals/year globally) and undermonitored. Current welfare-monitoring is mostly visual + walk-throughs, which catch <5% of distress events before they escalate. Cameras don't work well in barns (dust, low light, fly poop) and wearables don't work on animals (chewing, mud, broken collars).

CSI sensing has the right modality fit:
- **Continuous** (24/7, no shift change)
- **Dust/dirt tolerant** (RF goes through filth)
- **No animal cooperation needed** (no wearable to chew)
- **Through-stall** (concrete walls of typical dairy barns are 8-12 dB attenuation)
- **Privacy** (animals don't care about consent; farmers are the consenting party)

R10's per-species gait taxonomy already extends to livestock; R6.2.5's multi-subject union already covers dense populations; R12 PABS provides predator-detection capability. R19 catalogues how the loop's primitives compose into agricultural deployments.

## Animal categories + loop primitive match

| Species | Adult mass | Stride freq | RCS scale | Best loop primitive |
|---|---:|---|---|---|
| Dairy cow | 600 kg | 0.6-1.2 Hz | high | R10 gait + R12.1 fall detection |
| Beef cattle | 700-1000 kg | 0.5-1.0 Hz | very high | R10 gait + R6.2.5 herd count |
| Pig (sow) | 200-300 kg | 1.0-2.0 Hz | medium | R10 + R14 V1 breathing (stress) |
| Pig (piglet) | 5-20 kg | 2.0-3.5 Hz | low | R6.2.5 multi-subject count |
| Sheep | 60-80 kg | 1.5-2.5 Hz | medium | R10 gait + R12 PABS predator |
| Chicken (layer) | 1.5-2.5 kg | 3.0-5.0 Hz | very low | R6.2.5 (density)/R12 PABS only |
| Goat | 50-90 kg | 1.8-3.0 Hz | medium | R10 + R14 V1 |
| Horse | 400-600 kg | 1.0-1.8 Hz | high | R10 + R12.1 (welfare colic detection) |

R6.1's chest-dominant signal scales with body mass; cattle and horses are easier targets than chickens.

## Three deployment scenarios

### Scenario A: Dairy parlour + barn monitoring (5y)

Single barn, ~50-100 cows. Continuous monitoring of:
- **Herd presence + count** (R6.2.5 multi-subject union)
- **Individual cow ID** (R3 + AETHER per-installation embedding library)
- **Welfare anomalies** (R14 V1 breathing rate at large; calving stress detection)
- **Lameness early detection** (R10 gait asymmetry — clinically meaningful but currently undetected until severe)
- **Fall / down-cow detection** (R12.1 pose-PABS) — critical for cattle that can't right themselves
- **Predator intrusion** (R12 PABS — coyotes, wolves, mountain lions, dogs)
- **Heat / cooling stress** (R14 V1 breathing rate elevated)

Cost per dairy barn: ~$200 (12-20 anchors per ~500 m² barn). Compares to ~$50K for visual + RFID + behaviour-tracking systems.

### Scenario B: Free-range pasture monitoring (10y)

Larger spatial scale (~100-1000 hectares). ESP32 + solar + LiPo + Tailscale mesh = self-organising sensor network across a pasture. Detect:
- **Herd location** (R1 ToA + R6.2.2 N-anchor multistatic with sparse anchors)
- **Strays + lost animals** (R3 + AETHER)
- **Predator approach** (R12 PABS at field edges)
- **Birthing event** (R14 V1 breathing rate signature — cow about to calve)

Closer to wildlife sensing (R10) than barn monitoring. The 100 m sparse-foliage range from R10 directly maps.

### Scenario C: Pig barn density management (15y)

Pig housing has the highest density per square meter and the most ethical concerns (cramped housing → distress + disease). R19's most ethically valuable application:
- **Welfare scoring per stall** — breathing rate + motion intensity gives a per-pig stress index
- **Aggression detection** — multi-subject motion correlation (R6.2.5 + R12 PABS)
- **Sick-pig isolation alert** — stationary + elevated breathing + temperature drift
- **Tail-biting outbreak warning** — gait + close-contact patterns

Industrial-scale impact: enables welfare-aligned husbandry without manual rounds. Aligns with EU "End the Cage Age" policy and California Prop 12.

## What's different from human verticals (R16/R17/R18)?

| Dimension | Human verticals | R19 livestock |
|---|---|---|
| Subject mass | 60-100 kg | 1.5-1000 kg (3+ orders of magnitude) |
| Subject count per room | 1-8 | 1-1000+ |
| Subject behaviour | upright + bipedal | varies by species |
| Privacy | HIPAA / OSHA / employment | farmer-consents-for-animals |
| Regulatory | FDA / OSHA / GDPR | USDA / EU welfare regs |
| Cost sensitivity | high | very high (livestock margins are 2-5%) |
| Failure cost | clinical / safety event | welfare violation + lost animal value |

The cost sensitivity is the critical constraint. A $15/anchor BOM for cattle is fine; for chickens it's marginal (200 layers at $5 each = $1,000 of birds, ~$200 sensor system = 20% of inventory value is unacceptable).

## R10 gait taxonomy extension for livestock

R10 catalogued per-species gait. Extending to common livestock:

| Species | Stride freq | DSP band |
|---|---|---|
| Dairy cow walking | 0.6-1.2 Hz | low |
| Dairy cow lame | 0.4-0.8 Hz + asymmetry | low + irregular |
| Pig walking | 1.0-2.0 Hz | low-mid |
| Sheep walking | 1.5-2.5 Hz | mid |
| Chicken (layer) | 3.0-5.0 Hz | upper |
| Horse walking | 1.0-1.8 Hz | low-mid |
| Horse lame | 0.7-1.4 Hz + asymmetry | low-mid irregular |

**Per-species gait drift** (compared to within-species baseline) detects welfare issues earlier than visual inspection. Asymmetry > 15% indicates lameness; rate drop > 20% indicates illness.

## R14 V1 vital-signs primitives for livestock

R14 V1 breathing-rate detection works the same way physically. Per-species normal ranges:

| Species | Normal breathing rate (BPM) | Stress threshold |
|---|---|---|
| Cattle | 10-30 | >40 |
| Pig | 10-25 | >35 |
| Sheep | 12-25 | >30 |
| Horse | 8-16 | >20 |
| Chicken | 15-40 | >50 |

The rate-level primitive (R13 ruled out contour) is sufficient for welfare-anomaly detection. **Heat stress detection** is the highest-leverage application — overheated cattle drop milk production by 30-50% before visual signs.

## R12 PABS predator detection (high impact)

Predator-induced livestock losses in the US alone are ~$232M/year (USDA 2015). Current mitigation is fencing + guard dogs + electric. R12 PABS extends this with **passive RF monitoring**:

- ESP32 nodes at pasture perimeter
- R12 PABS detects "structure entered the protected zone" (a coyote, wolf, dog, etc.)
- R10 gait classifier disambiguates predator from cattle/sheep
- Alert via cellular / Tailscale to farmer phone

Per-pasture cost: ~$100 (8 anchors at perimeter). Cost-effective at ~10% of typical guard-dog programme.

## Honest scope

- **Synthetic data only** — all loop numbers are simulated indoor. Outdoor / pasture deployments need bench validation.
- **Per-species RCS measurements** are needed — body-mass scaling is approximate; actual radar cross-sections vary by species shape (cow is roughly cylindrical, pig is rounded).
- **Chicken-scale deployments** are economically marginal due to cost sensitivity.
- **High-density pig barns** may exceed R6.2.5's 4-occupant tested limit (typical pig stall is 0.5-2 m² per pig with 8-100 pigs per barn).
- **Weather-affected outdoor RF** is not in loop scope (rain attenuation, dew on antennas).
- **Animal welfare audits** require regulatory approval per jurisdiction — operational, not technical.
- **No animal-welfare ethics review** has been done; the loop only specifies the sensing infrastructure.

## Cog roadmap

| Cog | Timeline | Primitive composition |
|---|---|---|
| `cog-cattle-monitor` | 5y | R10 gait + R14 V1 + R6.2.5 + R12.1 fall |
| `cog-pig-welfare` | 5y | R6.2.5 + R14 V1 + multi-subject correlation |
| `cog-predator-alert` | 5y | R12 PABS + R10 species classifier |
| `cog-lameness-detector` | 10y | R10 gait asymmetry + temporal drift |
| `cog-birthing-alert` | 10y | R14 V1 breathing signature |
| `cog-free-range-tracker` | 15y | R6.2.2 sparse N-anchor + Tailscale mesh |

## What R19 enables

1. **Animal welfare at industrial scale** — first vertical that significantly addresses non-human subjects.
2. **Predator detection without electric fences** — passive, no animal-disturbing infrastructure.
3. **Early lameness detection** — R10 gait taxonomy directly applied to dairy cattle.
4. **Birthing alerts** — R14 V1 + species-specific breathing patterns.
5. **Sixth+seventh vertical confirming loop's vertical-agnostic generality** — same primitives, new domain.

## What R19 DOES NOT enable

- Replacement of veterinary care — R19 detects anomalies, vets diagnose + treat.
- Per-animal genetic / pedigree tracking — separate from sensing layer.
- Replacement of RFID ear tags entirely — RFID is cheap and well-established for individual ID; R19 supplements rather than replaces.

## Composes with prior threads

- R1, R3, R5, R6/R6.1, R6.2.5: physics + placement infrastructure
- R7 mincut: necessary at pasture-edge for adversarial RF (cell, GPS, drone RF)
- R10 gait taxonomy: directly extends to livestock species
- R12 PABS / R12.1: predator detection + cattle-fall detection
- R13 NEGATIVE: rules out BP / HRV-contour for livestock (use behaviour instead)
- R14 V1: rate-level breathing for welfare scoring
- R15 biometric: per-animal RF fingerprint for ID-without-tag
- R16/R17/R18 (parallel verticals): same architecture, new domain
- ADR-113: placement matrix — livestock cogs would use modified rows
- ADR-105-109: federation + privacy + provenance (farmer-consent regime)

## Seven exotic verticals now

1. R10 wildlife (animal conservation)
2. R11 maritime (vessel safety)
3. R14 empathic appliances (home)
4. R16 healthcare (clinical)
5. R17 industrial (safety)
6. R18 disaster (rescue, integrates MAT crate)
7. **R19 livestock (agriculture, welfare)**

Seven distinct domains. Same architecture. The pattern is now overwhelming evidence that the loop's output is genuinely vertical-agnostic infrastructure.

## R19's special angle

This is the **first non-human-centric vertical** in the loop. Animal welfare is its own ethical territory; the privacy framework (R14 + R3 + R15 + ADR-106) doesn't apply the same way (animals can't consent), but is replaced by **animal welfare regulations** (USDA, EU, California Prop 12). The architecture is the same; the regulatory regime differs.

## Connection back

Every loop output referenced. R19 + R18 are the two verticals that have **direct external partnerships** as critical-path (USDA / animal welfare orgs for R19; FEMA / urban-SAR for R18). The other verticals (R16/R17/R14) have natural commercial partners (hospitals, employers, homeowners).
