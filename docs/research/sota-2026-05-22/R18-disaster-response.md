# R18 — Disaster response: collapsed-building survivor detection (composes wifi-densepose-mat)

**Status:** exotic vertical sketch + integration with existing repo crate · **2026-05-22**

## Premise

After an earthquake, building collapse, or industrial explosion, survivors trapped under rubble have a **72-hour critical window** for rescue. Current detection methods (search dogs, thermal imaging, acoustic sensors, fibre-optic listening devices) each have limitations:

- Search dogs: scarce, trainable for ~20-30 minutes between rests
- Thermal: blocked by debris, weather-dependent
- Acoustic: requires silent rescue site (often impossible)
- Fibre-optic: slow deployment per survey area

**WiFi CSI / radar sensing** offers a unique combination: penetrates rubble (debris is less attenuating than steel), works in darkness/dust/smoke, no operator-active signal (passive listening). The repo already has a dedicated crate for this:

> `wifi-densepose-mat` — Mass Casualty Assessment Tool — disaster survivor detection
> (from CLAUDE.md crate table)

R18 integrates the existing MAT crate with the loop's findings to specify a complete disaster-response stack.

## The MAT crate's existing scope

From the workspace dependency graph (CLAUDE.md):
- `wifi-densepose-mat` depends on `core, signal, nn`
- Used by `wifi-densepose-wasm` (browser deployment) + `wifi-densepose-cli`

The crate is **shipped today** but predates this loop's research output. R18 catalogues what the loop adds:

| Capability | MAT crate today | + Loop findings |
|---|---|---|
| Detect "there is a survivor here" | yes (core function) | R12.1 pose-PABS makes detection precise + reduces false alarms by 9.36× |
| Estimate survivor count | yes | R6.2.5 multi-subject union; bounded to ~4 with current placement |
| Localise survivor | partial | R1 ToA CRLB sets the precision floor (~25 cm at 4-anchor convex hull); R6 Fresnel gives sensitivity envelope |
| Through-rubble propagation | yes (mat-specific) | R11 maritime through-seam analysis transfers (debris is RF-leaky, not RF-opaque) |
| Vital-signs from trapped survivor | partial | R14 V1 + R15 breathing rate primitive — works through 1-2 m of rubble |
| Distinguish survivor from rescue worker | not addressed | R3 + AETHER if a "rescue worker signature library" is loaded |
| Mass-casualty triage signal | partial | R15 biometric stability primitives — declining HRV / breathing → triage priority bump |
| Adversarial environment (other RF sources at scene) | not addressed | R7 mincut adversarial defence essential |
| Audit / chain of evidence for legal | not addressed | ADR-109 Dilithium-signed event log |

## Through-rubble propagation (R11 maritime parallel)

R11 maritime found that steel bulkheads at 2.4 GHz have a 3.25 µm skin depth → utterly opaque. **Earthquake debris is mostly NOT steel** — typical building collapse rubble is concrete + drywall + wood + insulation, mostly partially RF-transparent:

| Material | Approximate 2.4 GHz attenuation |
|---|---:|
| Steel (1 mm) | 2,674 dB (opaque) |
| Reinforced concrete (10 cm) | 20-30 dB |
| Drywall (1.5 cm) | 1-2 dB |
| Wood (5 cm) | 2-4 dB |
| Insulation (foam, 10 cm) | 5-8 dB |
| Brick (10 cm) | 8-12 dB |
| Glass / dust mixture | 3-6 dB |
| Rubble pile (mixed, 1-2 m) | **40-80 dB** (much less than steel) |

An ESP32-S3 with its 121 dB link budget has **~40-80 dB margin** through typical rubble of 1-2 m depth. **Survivors at this depth are detectable.** Deeper rubble (3-5 m) becomes marginal; pure-steel rubble (rare except basement collapses with rebar) is impossible.

This is dramatically better than the maritime through-bulkhead case where steel was the dominant material.

## Three deployment scenarios

### Scenario A: Building-collapse rapid-response (5y, current MAT scope)

| Requirement | Loop primitive | Configuration |
|---|---|---|
| Per-survey-zone deployment | R6.2.2 N-anchor | 4-6 anchors per ~20 m² survey area |
| Through-rubble detection | MAT crate baseline | (already shipped) |
| Survivor count + position | R1 + R6.2.5 + R12.1 | ~25 cm position precision |
| Vital signs confirmation | R14 V1 + R15 breathing | rate-level only per R13 NEGATIVE |
| Survivor-vs-rescuer disambiguation | R3 + rescue-worker signature library | per-deployment loaded library |
| Adversarial RF | R7 mincut | critical at deployment sites (cell, BLE, mesh radios) |
| Real-time triage updates | ADR-105 within-installation fed | local on-device, no cloud |

Cost per survey unit: ~$200 (multi-anchor ESP32 array + portable battery + ruggedised enclosure). FEMA / urban-search-and-rescue purchase model.

### Scenario B: Earthquake-region pre-staged sensors (10y)

Permanent installations at seismic-risk sites (hospitals, schools, transit hubs). After tremor activity, sensors **automatically activate** survivor-detection mode. The detection-mode cog ships in opt-in form (R14 framework).

### Scenario C: Cross-disaster federated learning (15y)

Each disaster generates new training data. ADR-107 cross-installation federation allows multiple disaster sites to **federate learning** about debris-propagation patterns without sharing raw rescue data. ADR-108 quantum-resistant key exchange protects rescue site sovereignty.

## What loop primitives add to the existing MAT crate

1. **R12.1 pose-PABS closed loop**: 9.36× false-alarm reduction is critical for time-pressured rescue operations.
2. **R6.2.5 multi-subject union**: critical for multi-survivor scenarios (e.g. school cafeteria collapse).
3. **R1 ToA CRLB**: gives FEMA the precision number for survey-unit placement.
4. **R7 mincut adversarial defence**: disaster sites have heavy RF interference; R7 prevents false negatives from compromised links.
5. **R14 V1 vitals + R15 rate-level breathing**: rules out HRV-contour (R13 NEGATIVE) but breathing rate IS reliable for confirming "the heat signature we found is alive".
6. **ADR-105-109 federation chain**: cross-disaster federated learning + audit trail integrity for legal evidence.
7. **ADR-113 placement matrix**: gives field operators a deterministic placement recipe rather than tribal knowledge.

## Honest scope

- **No bench-validated disaster-site data** — all loop numbers are synthetic. MAT crate has been tested in lab; real disaster validation is rare for ethical reasons (you can't simulate dead bodies; you have to wait for real events).
- **R7 mincut at disaster sites** is a hostile-RF requirement, not nice-to-have. Sites have firefighter radios, FEMA mesh, satellite phones — all interfering.
- **Cross-disaster federation** raises serious consent questions: rescued survivors and victims' families may not consent to their data being used for training future models. This is an ethical research question, not just technical.
- **Time-pressure changes everything**: in a real rescue, false-positive at 1× minute cost is acceptable but false-negative at minute cost is fatal. R12.1's 9.36× lift is critical but the threshold has to be tuned aggressively toward false-positive.
- **MAT crate API is shipped** but doesn't yet consume R6.1 multi-scatterer forward model. Integration work needed.

## Through-rubble vital-signs feasibility

The same R6.1 analysis that gave 4.7 dB multi-scatterer penalty in clear air applies, plus 40-80 dB rubble attenuation. SNR margin:

```
Link budget:               121 dB
Rubble loss (1-2 m):     -40 to -80 dB
Multi-scatterer penalty:  -4.7 dB
SNR margin needed:        -10 dB
Available for vitals:    +37 to -27 dB
```

**Breathing-rate detection at 1 m rubble depth is feasible (+37 dB margin).** At 2 m it's marginal (+7 dB). At 3 m it's infeasible. This matches what MAT crate's existing range estimates probably already say; R6.1 makes the budget explicit.

## Cog roadmap

| Cog | Timeline | Primitive |
|---|---|---|
| `cog-mat-survivor-detect` (existing) | NOW | wifi-densepose-mat |
| `cog-mat-pose-pabs` | 5y | + R12.1 closed loop |
| `cog-mat-multi-survivor` | 5y | + R6.2.5 multi-subject |
| `cog-mat-vitals-confirm` | 5y | + R14 V1 + R15 (rate-level) |
| `cog-mat-survivor-vs-rescuer` | 10y | + R3 + rescue-worker library |
| `cog-mat-cross-deploy-fed` | 15y | + ADR-105-108 (consent-bounded) |

## What R18 enables

1. **A clear path from MAT crate (today's scope) to fully-instrumented disaster-response system** (15y horizon).
2. **Direct integration of loop primitives** with existing repo code — most concrete vertical so far.
3. **Quantified rubble-depth budget**: 1 m feasible, 2 m marginal, 3 m infeasible.
4. **Six-cog roadmap** spanning 0-15y.

## What R18 DOES NOT enable

- Real disaster validation without partnerships with FEMA / urban-search-and-rescue teams
- Cross-disaster federation without resolving ethical consent questions
- Steel-rubble cases (basement collapse with rebar) — physics rules these out
- Underwater rescue (R11 saltwater finding rules this out at WiFi bands)

## R18 vs R10/R11/R14/R16/R17 (vertical comparison)

| | R18 disaster | R16 healthcare | R17 industrial |
|---|---|---|---|
| Repo asset | existing MAT crate | none yet | none yet |
| Through-medium | rubble (40-80 dB) | air | air |
| Mobility | trapped (static) | stationary | mobile |
| Coverage | survey-unit (~20 m²) | ward (30 m²) | zone (100-1000 m²) |
| Privacy | survivor consent post-hoc | HIPAA | OSHA |
| Failure cost | survivor dies | clinical miss | safety incident |
| R7 mincut | binding (hostile RF) | nice-to-have | binding |

**Disaster + industrial both require R7 mincut as binding.** Healthcare doesn't (controlled environment).

## Composes with prior threads

- R1 (CRLB): position precision in survey unit
- R6/R6.1: through-rubble forward model
- R6.2.5 + R6.2.2: multi-survivor union coverage
- R7 (mincut): **binding** at disaster sites
- R10 (foliage attenuation parallel): rubble attenuation analogous to foliage
- R11 (maritime through-bulkhead): same physics framework, different material parameters
- R12 / R12.1 (PABS): false-alarm reduction in rescue ops
- R13 NEGATIVE: rules out blood-pressure / HRV-contour
- R14 V1 + R15: vital-signs confirmation
- R3 + AETHER: survivor-vs-rescuer disambiguation
- ADR-105-109: federation + audit chain
- ADR-113: placement matrix gives field-operator recipe

## R18 is the third "vertical that demonstrates loop generality"

After R16 (healthcare) and R17 (industrial), R18 is the third vertical showing the loop's primitives compose without new research. **Three out of three target verticals (clinical, industrial, disaster) work with the same architecture.** This is strong evidence that the loop's output is genuinely vertical-agnostic.

## Connection back

Every loop thread referenced above. R18 is also the **first** vertical to integrate with an existing repo crate (`wifi-densepose-mat`), making the loop-to-production path most direct for this domain.
