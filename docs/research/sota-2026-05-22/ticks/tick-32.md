# Tick 32 — 2026-05-22 10:23 UTC

**Thread:** R16 (healthcare ward monitoring — new exotic vertical)
**Verdict:** A vertical that **composes** loop primitives rather than introducing new research. All required components exist; the gap is bench validation + BAA + regulatory pathway. 5y / 10y / 15y deployment scenarios catalogued.

## What shipped

- `docs/research/sota-2026-05-22/R16-healthcare-ward-monitoring.md` — vertical sketch + primitive composition + cost analysis + honest scope.

## Why R16 fits the cron prompt's "exotic vertical / 10-20y horizon" criteria

Hospitals run on a paradox: continuous monitoring needed, cameras unacceptable. CSI sensing is the right modality if privacy + accuracy constraints met. R16 demonstrates the loop's 9-ADR + 13-thread output is sufficient to specify a complete clinical-deployment system — no new research needed, only composition.

## Three scenarios

| Scenario | Timeline | Cost vs status quo |
|---|---|---|
| ICU bedside | 5y | $30/bed vs $3,000 hospital-grade monitor |
| General ward (8-bed) | 10y | $120/ward vs $200K/year continuous-observation staffing |
| At-home post-discharge | 15y | empathic-appliance V1/V2/V3 + telemedicine |

## Healthcare requirement → loop primitive mapping

| Need | Loop primitive |
|---|---|
| Continuous breathing / HR rate | R14 V1 + R15 (rate-level only per R13 NEGATIVE) |
| Patient identity per bed | R3 + AETHER |
| Fall detection | R12.1 pose-PABS closed loop |
| Intruder / unexpected occupant | R12 PABS multi-subject |
| Multi-bed coverage | R6.2.5 + ADR-113 placement matrix |
| HIPAA / medical-grade privacy | ADR-106 medical-grade profile (ε=2) |
| Audit trail | ADR-109 Dilithium-signed cog |
| Multi-installation hospital fleet | ADR-107 + ADR-108 cross-install quantum-resistant |

## Two gaps blocking clinical deployment (both solvable, neither new research)

1. **Bench validation** on real patient data (6-12 months)
2. **BAA infrastructure** with hospital partner (operational, not technical)

## What R13 NEGATIVE rules out

- Blood pressure cog — keep arm cuff in workflow
- HRV contour — keep PPG wearable for ICU

## What R12.1 + R6.2.5 enables

- Fall detection: 9.36× lift (R12.1)
- 100% coverage for 4-occupant multi-bed room (R6.2.5)
- Per-bed identity preservation (R3 + AETHER)

## Six cog roadmap items

| Cog | Timeline | Primitive |
|---|---|---|
| cog-vital-signs | 5y | R14 V1 + R15 |
| cog-fall-detection | 5y | R12.1 |
| cog-bed-occupancy | 5y | R12 PABS + R6.2.5 |
| cog-respiratory-anomaly | 10y | temporal R15 breathing |
| cog-post-discharge | 15y | V1/V2/V3 + telemedicine |
| cog-elderly-care | 20y | R10 gait + R15 limb-timing |

## Honest scope

- Synthetic data only (bench validation pending)
- 8-bed wards may exceed R6.2.5's 4-occupant tested limit
- Hospital RF environment harsh (R7 mincut handles some)
- Clinical workflow integration is substantial engineering
- Regulatory approval (FDA/CE) is 6-18 months + $500K-$2M per device class

## Why this matters

R16 confirms the loop's output is **architecturally complete** for a clinical-deployment system. Same primitives that ship empathic appliances (R14) ship healthcare. Same privacy framework (ADR-106) maps to HIPAA. Same federation (ADR-105-109) handles multi-hospital fleets.

**Composition, not research, is the remaining work.**

## Composes with every loop thread

- R1 (CRLB) — bed-position precision for fall threshold
- R5 — subcarrier explanation for breathing detection
- R6/R6.1 — physics foundation
- R6.2.5 — multi-bed ward placement
- R7 — adversarial defence against medical-device RF
- R10 — gait fingerprint for elderly-care
- R11 — parallel exotic vertical (maritime cabin = ICU bedside parallel)
- R12/R12.1 — fall + intruder
- R13 NEGATIVE — rules out BP/HRV-contour
- R14 — V1/V2/V3 framework translates to at-home
- R15 — per-patient ID + vitals
- R3 — per-ward identity preservation
- All ADRs (105-109 + 113) binding

## Coordination

`ticks/tick-32.md`. No PROGRESS.md edit. Branch `research/sota-r16-healthcare-ward`.

## Loop now has 5 exotic vertical sketches

R10 (wildlife) / R11 (maritime) / R14 (empathic appliances) / **R16 (healthcare ward)** / + R3-R15 cross-thread = covering wildlife conservation, maritime safety, home automation, clinical care, and security/identity.

~1.5h to cron stop.
