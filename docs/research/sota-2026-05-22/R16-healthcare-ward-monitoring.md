# R16 — Healthcare ward monitoring: a vertical that composes the loop's primitives

**Status:** exotic vertical sketch + concrete primitive composition · **2026-05-22**

## Premise

Hospitals run on a paradox: patients need continuous monitoring, yet cameras and microphones are unacceptable in patient rooms for privacy and dignity reasons. Wearable monitors solve part of this (continuous HR / SpO₂) but require subject compliance and battery management. CSI sensing — passive, no light, no microphone, through-wall-capable — is the right modality for ward-level continuous observation **if** the privacy and clinical-grade accuracy constraints can be met.

The RuView research loop has produced exactly the primitives needed:

| Healthcare requirement | Loop primitive |
|---|---|
| Continuous breathing rate per patient | R14 V1 + R15 breathing-rate primitive |
| Continuous heart-rate per patient | R14 V1 + R15 HRV-rate primitive (R13 ruled out HRV-contour) |
| Patient identity tracking per bed | R3 + ADR-024 AETHER re-ID |
| Fall / out-of-bed detection | R12 PABS + R12.1 closed loop |
| Bed-position deviation alert | R12 PABS pose-aware |
| Intruder / unexpected occupant | R12 PABS multi-subject extension |
| Multi-bed coverage in ward | R6.2.5 multi-subject union + R6.2.4 3D |
| HIPAA / medical-grade privacy | ADR-106 medical-grade DP profile (σ=1.5, ε=2) |
| Tamper-resistant clinical evidence | ADR-100 + ADR-109 signed cog distribution |
| Multi-installation hospital fleet | ADR-107 + ADR-108 cross-installation quantum-resistant federation |

**The healthcare-ward vertical is not a research problem — it is an integration problem.** All the components exist; the work is composition + clinical validation.

## Three deployment scenarios

### Scenario A: ICU bedside monitoring (5y)

| Requirement | Loop primitive | Configuration |
|---|---|---|
| Continuous vitals per patient | R14 V1 + R15 | `cog-vital-signs` |
| Patient identity (1 patient per bed) | R3 + AETHER (no cross-bed contamination) | per-installation embedding space |
| Out-of-bed detection | R12 PABS + R12.1 | pose-aware closed loop |
| Bed-position deviation (e.g. patient slumping) | R12.1 PABS-after-pose-update | continuous |
| Alert latency budget | <30 s | local on-device, no cloud round-trip |
| Privacy | HIPAA-aligned | ADR-106 medical-grade profile (ε=2) |
| Placement (per ADR-113) | 2D chest, N=4, low-mount opposite-bed | one Cognitum Seed per bed-side pair |

Cost per bed: ~$30 (2× ESP32-S3 BOM + mounting + per-installation calibration). Compares to ~$3,000 for a hospital-grade continuous monitor.

### Scenario B: General ward multi-patient coverage (10y)

| Requirement | Loop primitive | Configuration |
|---|---|---|
| Multi-patient simultaneous monitoring | R6.2.5 multi-subject union | N=5-6 anchors per ward room |
| Per-patient breathing / HR rate | R14 V1 + R15 | `cog-vital-signs` running on each Cognitum Seed |
| Inter-bed identity preservation | R3 + AETHER | per-ward embedding space |
| Nurse / visitor presence detection | R12 PABS multi-subject | separates expected (staff) from unexpected (intruder) |
| Patient fall (anywhere in room) | R12 PABS + R12.1 | spike on any unexpected pose change |
| Federation across ward beds (per-ward local) | ADR-105 within-installation | nightly federated training |
| Federation across hospital wards | ADR-107 + ADR-108 | cross-installation with Kyber + SA |
| Audit trail integrity | ADR-109 Dilithium-signed cog | tamper-resistant clinical evidence |

Cost per ward (8-bed): ~$120 (8× $15 BOM). Plus per-ward installation time of ~2 hours. Compares to staffing one extra nurse per ward for ~$200K/year continuous observation.

### Scenario C: At-home post-discharge monitoring (15y)

Same primitives, but in a patient's home. The empathic-appliance framework (R14) applies — V1 stress-responsive lighting becomes V1 vitals-aware lighting. V2 HVAC becomes V2 respiratory-anomaly-aware climate. Patient empowered to monitor own recovery without wearables or daily clinic visits.

Critical regulatory difference: at-home requires explicit patient opt-in + clinician oversight + telemedicine integration. The R14 privacy framework already specifies opt-in-by-default and on-device-data; the clinical-grade telemedicine layer is an additional integration.

## The clinical-vs-research-grade scope

| Capability | Loop produces | Hospital needs | Gap |
|---|---|---|---|
| Breathing rate | ±1 BPM (R15) | ±0.5 BPM | Bench validation needed |
| Heart rate | ±5 BPM rate (R15, R13 ruled out contour) | ±2 BPM | Sufficient at rate level |
| HRV contour | **NOT achievable** (R13 NEGATIVE, 5 dB short) | preferred | Replace with PPG wearable for ICU |
| Blood pressure | **NOT achievable** (R13 NEGATIVE) | clinical-grade | Replace with arm cuff |
| Pose / fall detection | 92.9% PCK@20 (ADR-079) | 99%+ | Improvement needed; OK for screening |
| Identity (per-bed in stable env) | ~100% AETHER (R3) | ~100% | Fine for ward |
| Multi-subject in same room | 100% N=5 (R6.2.5) | required | Fine for ward |
| Alert latency | <1 s on-device (R12.1) | <30 s | Comfortable margin |
| Privacy / DP | ε=2 medical-grade (ADR-106) | HIPAA + BAA | Need BAA infrastructure |
| Audit trail | ADR-109 signed | clinical evidence requirements | Sufficient with regulatory review |
| Bench validation | NONE (synthetic only) | required | Critical-path |

**Two gaps that block clinical deployment**:
1. **Bench validation** of breathing-rate accuracy on real patients (loop is synthetic-only).
2. **BAA infrastructure** (Business Associate Agreement) with hospital — operational, not technical.

Both are solvable in 6-12 months. Neither requires further research.

## Why the privacy chain is essential here

Healthcare data is the most-regulated personal data in most jurisdictions (HIPAA in the US, GDPR Article 9 in EU). The privacy chain from R14 + R15 + ADR-105-109 is what makes ward-deployment legally defensible:

- **ADR-106 medical-grade DP (ε=2)**: meets HIPAA-aligned anonymisation requirements
- **R15 on-device biometric primitives**: per-patient signatures never leave the bed
- **ADR-107 secure aggregation**: cross-hospital federation possible without raw data exchange
- **ADR-108/109 PQC**: ensures HIPAA-grade records remain integrity-protected through 2040+
- **R14 opt-in / override / data-stays-on-device**: matches HIPAA patient-consent requirements

Without this chain, the same sensing capability would create a surveillance liability rather than a clinical asset.

## What this DOES enable

1. **A complete clinical-deployment roadmap** without needing new research — just composition + bench validation + BAA.
2. **A cost-comparison story**: $30/bed vs $3,000/bed continuous monitor; $120/ward vs $200K/year staffing.
3. **A regulatory-aligned privacy story**: ADR-106 medical-grade DP profile maps directly to HIPAA expectations.
4. **A clear cog roadmap**: `cog-vital-signs` + `cog-fall-detection` (built on R12.1 PABS) + `cog-bed-occupancy` (built on R12 PABS) all reuse existing loop primitives.

## What this DOES NOT enable

- Replacement of clinical-grade arterial-line or 12-lead ECG. CSI sensing is **screening + continuous trend monitoring**, not diagnostic.
- Replacement of nursing observation for high-acuity patients. The complementary role is "free up nurse time for cases that need attention".
- Pediatric or geriatric special-case modeling without dedicated training data.
- ICU drug-interaction monitoring or any pharmaceutical-side decision support.

## Honest scope

- **Bench validation gap is real.** All loop numbers are synthetic. Real patient data validation is critical-path.
- **Multi-patient density** of typical wards (8 beds per ~30 m² room) may exceed R6.2.5's 4-occupant tested limit. R6.2.5.1 (8+ occupants) hasn't been benchmarked.
- **Hospital RF environment** is harsh — Bluetooth medical devices, WiFi networks, MRI shielding. R7 mincut adversarial defence handles some of this but not all.
- **Clinical workflow integration** (alert routing, EHR integration, nursing-station displays) is substantial engineering work outside the sensing layer.
- **Patient consent for sensing** is a separate workflow from BAA — patients-on-admission consent flow is required.
- **Regulatory approval** (FDA Class II in US, CE-MDR in EU) for any clinical-decision-affecting cog is 6-18 months and ~$500K-$2M per device class.

## R16 verticals catalogued (10-20 year horizon)

Within healthcare, the cogs that follow the same composition:

1. **`cog-vital-signs`** (5y) — breathing + HR rate, R15-grade. ICU bedside + general ward.
2. **`cog-fall-detection`** (5y) — R12.1 pose-PABS closed loop. Reduces nurse staffing demand.
3. **`cog-bed-occupancy`** (5y) — R12 PABS + R6.2.5 multi-subject. Census + room-utilisation analytics.
4. **`cog-respiratory-anomaly`** (10y) — temporal-pattern analysis on R15 breathing primitive. Early warning for sepsis / pulmonary deterioration.
5. **`cog-post-discharge`** (15y) — at-home recovery monitoring. Composes V1/V2/V3 with telemedicine.
6. **`cog-elderly-care`** (20y) — gait stability tracking via R10 + R15 limb-timing biometric. Pre-fall risk assessment.

## Composes with loop's full output

This vertical sketch confirms that the loop's 9-ADR + 13-thread + 9-tick R6 family is sufficient to specify a complete clinical-deployment system. No new research needed; only:

1. Bench validation on real patient data (6-12 months)
2. BAA + hospital partnership (operational)
3. Cog implementation per the placement matrix (ADR-113)
4. Federation rollout per ADR-105-109
5. FDA / CE regulatory pathway (per cog category)

## Connection back to every loop thread

- **R1 (ToA CRLB)**: bed-position precision feeds fall-detection threshold.
- **R5 (saliency)**: explains which subcarriers drive breathing detection (R14).
- **R6 / R6.1**: physics foundation.
- **R6.2.5**: multi-bed ward placement.
- **R7 (mincut)**: adversarial defence against medical-device RF noise.
- **R10 (gait taxonomy)**: per-patient gait fingerprint for `cog-elderly-care`.
- **R11 (maritime)**: parallel exotic-vertical (different bounded context, same architecture).
- **R12 / R12.1 (PABS)**: fall + intruder detection.
- **R13 (NEGATIVE BP)**: ruled out blood-pressure cog — clinical workflow uses arm cuff.
- **R14 (empathic appliances)**: V1/V2/V3 framework translates to at-home scenario.
- **R15 (biometric primitives)**: per-patient ID + vital primitives.
- **R3 (cross-room re-ID)**: per-ward patient identity preservation.
- **ADR-105/106/107/108/109/113**: privacy + federation + provenance + placement all binding.
