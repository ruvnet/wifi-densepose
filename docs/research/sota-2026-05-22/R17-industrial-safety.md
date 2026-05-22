# R17 — Industrial safety: factory floor + warehouse + construction site monitoring

**Status:** exotic vertical sketch · **2026-05-22**

## Premise

Industrial environments account for ~2.8 million workplace injuries per year in the US alone (BLS 2023), with similar per-capita rates globally. Most go undetected for minutes because no one is watching — workers operate alone in large open spaces (warehouses, refineries), behind machinery, or on isolated construction sites. The leading injury types are:

- **Slips, trips, falls** (~24% of all injuries)
- **Overexertion** (~30%) — repetitive strain, lifting incidents
- **Contact with object/equipment** (~24%) — struck-by, caught-in
- **Lone-worker incapacitation** (low frequency, high severity)

CSI sensing offers a unique modality for this domain: large coverage areas, no PII concerns (workers can be opt-in by employment contract), no cameras (workers prefer this), and continuous operation despite dust / debris / low light.

This thread sketches how the loop's primitives compose into an industrial safety stack.

## Three deployment scenarios

### Scenario A: Warehouse / fulfilment centre (5y)

| Requirement | Loop primitive | Configuration |
|---|---|---|
| Worker count per zone | R6.2.5 multi-subject | N=4-6 per ~100 m² zone |
| Fall / collapse detection | R12.1 pose-PABS | per-zone threshold |
| Worker presence in hazardous area (forklift lane) | R12 PABS + R6.2.5 | "structure" detection in defined zones |
| Multi-zone coordination | R6.2.5 + ADR-105 federation | nightly training of "normal" patterns |
| Lone-worker silent-alarm | R14 V1 vitals (rate-level breathing only per R13) | passive — no wearable required |
| Adversarial RF (other devices) | R7 mincut | multi-link consistency |
| Audit trail | ADR-109 Dilithium-signed | incident-evidence integrity |

Cost per zone (100 m²): ~$80 (4-6× $15 BOM + mounting). Compares to 1 safety camera at ~$500-$2,000 + cabling + monitoring software.

### Scenario B: Construction site (10y)

Construction sites are RF-hostile (concrete, rebar, heavy machinery) and outdoor (variable conditions). The R6 family's recommendations still apply but with different parameters:

| Requirement | Loop primitive | Configuration |
|---|---|---|
| Worker location tracking | R6.2.2 N-anchor + R1 ToA | 4-cm precision at 4-anchor convex hull |
| Fall-from-height detection | R12.1 pose-PABS + R10 motion intensity | spike on vertical velocity + impact signature |
| Confined-space entry detection | R12 PABS + R6.2.5 | per-confined-space ESP32 anchors |
| Adverse-weather operation | R6.1 multi-scatterer + R10 attenuation | foliage-class attenuation but with rain |
| Multi-site coordination | ADR-107 cross-installation federation | per-project model |

The loop's R7 mincut adversarial defence is **essential** here — construction sites have legitimate RF noise (cellular, BLE-tagged tools, walkie-talkies) that R7 disambiguates from sensor compromise.

### Scenario C: Refinery / chemical plant (15y)

Highest-stakes industrial monitoring. Existing infrastructure is gas detectors + cameras + worker badges. CSI sensing **adds**:

| Capability | Loop primitive |
|---|---|
| Continuous "is the worker still upright?" | R12.1 pose-PABS |
| Multi-worker coordination in hazardous zones | R6.2.5 multi-subject |
| Vital-signs anomaly during chemical-exposure incident | R14 V1 + R15 breathing rate |
| Real-time post-incident triage | R12 PABS + R6.2.5 multi-subject locating |
| Audit + regulatory evidence | ADR-109 Dilithium |
| Tamper-evident telemetry | ADR-107 + ADR-108 quantum-resistant |

Particularly valuable when workers wear PPE that blocks visual / wearable sensors but doesn't substantially affect WiFi propagation.

## What's different from healthcare (R16)?

| Dimension | Healthcare (R16) | Industrial (R17) |
|---|---|---|
| Subjects | Stationary patients | Mobile workers |
| Subject signal strength | High (lying still) | Variable (walking, lifting, climbing) |
| Hostile RF | Moderate (medical devices) | High (machinery, cell, BLE tools) |
| Zone size | Small (~30 m² per ward) | Large (100-1000 m² per zone) |
| Regulatory | HIPAA / FDA | OSHA / equivalent |
| Privacy | Patient-consent + BAA | Worker consent via employment + opt-in |
| Cost sensitivity | High (hospital budgets are tight) | Moderate (industrial CapEx is justified by injury cost) |
| Failure mode | Missed clinical event | Missed safety event (potentially fatal) |

**Industrial safety needs different cog packaging**: lower-resolution-but-larger-coverage rather than per-patient precision. R6.2 placement matrix accommodates this via the `presence` row (N=3, body-centric) rather than the `vital-signs` row.

## The R7 mincut becomes critical

In a healthcare setting, the threat model is mostly "compromised supplier" — relatively low frequency, high impact. In industrial settings, the **ambient RF environment itself is adversarial**: cell jamming for safety reasons, intentional BLE tags, walkie-talkies, etc.

R7 Stoer-Wagner mincut adversarial detection is the right defence:
- **N ≥ 4 anchors per zone** (already required by ADR-113 for multi-feature cogs)
- **Multi-link consistency check** on per-zone CSI patterns
- **Per-anchor isolation** if mincut detects single-link compromise

This is a stronger requirement than R7 originally specified for home deployments. ADR-113 explicitly requires N ≥ 4 for industrial-safety cogs.

## R12.1 pose-PABS specialised for industrial

The pose tracker (ADR-079) was trained on indoor body-pose data. Industrial workers wear:
- Hard hats (slightly different head Doppler signature)
- High-vis vests (largely RF-transparent)
- Safety harnesses (different leg / torso scatterer geometry)
- Tool belts (extra scatterers below waist)
- Steel-toed boots (highly reflective at lower body)

The body model from R6.1 needs PPE-specific adjustments. Approximate adjustment is +5-15% per-part reflectivity for PPE-wearing workers. The exact numbers need bench measurement.

A future cog `cog-industrial-pose` would fine-tune the existing pose extractor (ADR-079) on PPE-wearing worker data. ~1-2 weeks of labelled-data work.

## R10 gait taxonomy + worker fatigue detection

R10 gave per-species gait frequencies. Within humans:
- Walking: 1.2-2.5 Hz
- Jogging: 2.0-3.0 Hz
- **Fatigued walking**: 0.8-1.5 Hz (slower, asymmetric stride)
- **Impaired walking** (substance influence or injury): asymmetry > 25%

A `cog-worker-fatigue` could detect early fatigue from gait drift over a shift. This is mid-term (10y) work but has direct OSHA-aligned value.

## Honest scope

- **Synthetic data only** — all loop numbers are simulated. Industrial environments differ enough from bedrooms that bench validation is required before clinical-grade claims.
- **PPE-specific body model** is unbuilt (R6.1 body model is bare-clothed).
- **Outdoor / weather effects** on CSI are not in the loop's scope; R10's foliage-attenuation model partly transfers.
- **Worker consent** is operational, not architectural; ADR-113 + R14 framework handles consent flow design but not the legal-specific employment-contract paperwork.
- **Insurance and liability** are major considerations for "missed safety event" failure modes; falls outside this thread.
- **Audit trail integration** with industrial safety information systems (e.g. SAP, Maximo, etc.) is per-customer integration work.

## What R17 enables

1. **A second exotic vertical** demonstrating the loop's output composes to industrial safety.
2. **Specialised cog roadmap**:
   - `cog-fall-detection` (R12.1) — reused from healthcare with industrial-PPE tuning
   - `cog-zone-occupancy` (R12 PABS + R6.2.5) — hazardous-area entry detection
   - `cog-lone-worker-vitals` (R14 V1) — silent alarm for incapacitation
   - `cog-worker-fatigue` (R10 + R15) — pre-incident gait analysis (10y)
   - `cog-multi-zone-orchestrator` (R6.2.5 + ADR-105) — federated normal-pattern learning
3. **R7 mincut critical-path identification**: industrial RF environment makes mincut adversarial defence binding rather than optional.
4. **Cross-vertical generality demonstrated**: the same primitives that make R16 (healthcare) work also make R17 (industrial) work, just with different ADR-113 matrix rows.

## What R17 DOES NOT enable

- Direct OSHA-certified deployment without bench validation + PPE-specific tuning
- Outdoor-only construction sites without weather-aware extensions
- Cross-modality fusion with existing safety camera + sensor systems (separate integration)
- Replacing wearable-based worker tracking (still needed for cellular dead-zones)

## Composes with prior threads

- R1 (CRLB): worker location precision for zone-entry detection
- R5 (saliency): primitive-specific saliency
- R6 / R6.1: physics foundation
- R6.2.5: multi-subject industrial-scale union
- R7 (mincut): becomes binding for industrial RF environment
- R10 (gait taxonomy): worker fatigue thread
- R12 / R12.1 (PABS): fall + intruder detection
- R13 NEGATIVE: BP / HRV-contour ruled out, same as healthcare
- R14 (empathic appliances → V1 vitals): rate-level vital signs
- R15 (RF biometric): per-worker ID for lone-worker monitoring
- R16 (healthcare): parallel composition pattern
- ADR-113 placement matrix: covered by `presence` and `vital-signs` rows
- ADR-105-109: privacy + federation + provenance + PQC chain

## R17 parallel to R16

| | R16 healthcare | R17 industrial |
|---|---|---|
| Subjects | patients in beds | workers on floor |
| Subject mobility | stationary | mobile |
| Coverage size | 30 m² ward | 100-1000 m² zone |
| ADR-113 row | vital-signs (chest, N=5) | presence (body, N=3-4) |
| Privacy regime | HIPAA / FDA | OSHA / employment |
| Cost vs status quo | $30/bed vs $3,000 monitor | $80/zone vs camera+cabling+software |
| R7 mincut role | nice-to-have | **binding requirement** |
| Failure cost | missed clinical event | missed safety event (potentially fatal) |

Same architecture, different parameter regime. The R6 family + ADR-113 absorbs the parametric variation.

## Closing observation

R16 + R17 together demonstrate that the loop's primitives form a **vertical-agnostic infrastructure layer**. Specific verticals are mostly cog packaging + ADR-113 row selection + per-domain calibration. The expensive parts (privacy chain, federation, placement physics) are reused.

This is the mark of well-factored research: outputs that generalise beyond their original problem.

## Connection back

Every prior loop thread + ADR is referenced above. R17 is the **second vertical** to demonstrate the loop's primitives are sufficient to specify a complete production deployment without new research.
