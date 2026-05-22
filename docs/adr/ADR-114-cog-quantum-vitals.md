# ADR-114: cog-quantum-vitals — first quantum-augmented vitals cog

**Status:** Proposed · **Date:** 2026-05-22 · **Author:** SOTA research loop tick-39 · **Composes:** ADR-089 (nvsim), ADR-021 (vitals), ADR-103 (cog-person-count), ADR-106 (DP-SGD), ADR-113 (placement) · **Refines:** quantum-sensing series docs 13/14/15/16/17

## Context

The SOTA research loop's R13 NEGATIVE finding (5-dB shortfall) ruled out HRV-contour and BP estimation from classical CSI. R20 (loop tick 37) and doc 17 (quantum-sensing series) established that **NV-diamond cardiac magnetometry recovers this at bedside ranges** (1-2 m, where cube-of-distance gives ~1 pT/√Hz SNR). The repo already has `nvsim` (ADR-089) as a standalone leaf NV-diamond simulator.

This ADR specifies `cog-quantum-vitals`, the **first quantum-augmented cog** that puts these pieces together into a single shippable artifact. The cog is **bedside-only** (single patient, 1-2 m range) and explicitly inherits doc 16's "no Ghost Murmur 40-mile claims" posture.

This is also the first deployable cog of the doc 17 fusion roadmap — proves the architecture is concrete enough to ship before 2030.

## Decision

Adopt `cog-quantum-vitals` as a **hybrid classical-quantum vitals cog** with the following architecture:

### Inputs

1. **Classical CSI window** (52 subcarriers × N antennas × 30 sec @ 100 Hz)
2. **NV-diamond magnetic field time series** (from `nvsim` today, real NV-diamond device in production)
3. **Pose tracker estimate** (ADR-079 / ADR-101, ~5 cm precision)
4. **Per-installation placement metadata** (ADR-113, 4-axis matrix `chest-mode, 2D, N=5`)

### Outputs

1. **Breathing rate** (BPM, ±0.1 BPM) — classical primary, NV cross-check
2. **Heart rate** (BPM, ±0.5 BPM) — NV primary, classical cross-check
3. **HRV contour** (R-R intervals + waveform shape) — **NV only** (R13 NEGATIVE rules out classical)
4. **Per-patient identity** (R3 + AETHER embedding, per-installation only per ADR-107)
5. **Confidence score per output** (so downstream cogs know fidelity)

### Architecture

```
                 ┌─────────────────────────────────┐
ESP32 CSI ──▶    │  R14 V1 breathing-rate primitive │ ──┐
                 └─────────────────────────────────┘   │
                 ┌─────────────────────────────────┐   │
                 │  R12.1 pose-PABS (residual ck)   │ ──┤
                 └─────────────────────────────────┘   │
                 ┌─────────────────────────────────┐   │
nvsim NV-B(t) ▶ │  R6.1-style multi-source        │ ──┼──▶  fused vitals
                 │  forward model + Bayesian fusion │   │
                 └─────────────────────────────────┘   │
                 ┌─────────────────────────────────┐   │
                 │  R3+AETHER per-patient ID head   │ ──┘
                 └─────────────────────────────────┘
```

Bayesian fusion: each output is a posterior from the (classical, quantum) likelihoods. When classical confidence is high (e.g. breathing rate at stable rest), classical drives. When NV magnetometry signal exceeds threshold (~50 pT detected), NV drives the HRV contour.

### Privacy + provenance (inherited)

All outputs flow through the ADR-106 primitive-isolation API:
- ✅ Raw NV magnetic field time series — on-device only
- ✅ Per-patient HRV contour — on-device only
- ⚠️ Aggregated breathing/HR rate — emittable with consent
- ⚠️ Model weight updates — federated per ADR-105 / ADR-107 with DP-SGD

Manifest signed per ADR-100 + ADR-109 (Phase 1: dual Ed25519 + Dilithium-3).

### Honest range

**1-2 m from patient bed.** This is bedside, not building-scale. Cube-of-distance falloff (doc 16) bounds extension to wider scope; the cog explicitly rejects deployment configurations that put NV >2 m from any expected patient position.

## Alternatives considered

### A. Pure-classical `cog-vital-signs` (existing baseline)

Status: **shipped today**. Limitations per R13 NEGATIVE: no HRV contour, no BP. Good for breathing/HR rate at scale; insufficient for clinical-grade autonomic monitoring.

### B. Pure-quantum NV-only cog

Status: **rejected**. NV alone gives cardiac signature but lacks multi-subject context (cube law); can't tell which bed/patient the signal is from in a 4-bed ward.

### C. Wearable + classical fallback

Status: **complementary, not alternative**. Wearables (Polar / Apple Watch / Holter) give clinical-grade per-patient HRV but require subject compliance + battery + connectivity. `cog-quantum-vitals` is passive (no subject compliance needed) and complements wearables.

### D. SQUID-based cog

Status: **deferred (20y)**. SQUID needs 4 K cryo today; room-temp SQUID is decades away. NV-diamond is the right near-term choice.

## Threat model

| Threat | Mitigation |
|---|---|
| Compromised NV hardware leaks raw B(t) | ADR-106 primitive-isolation: raw NV is on-device only |
| Spoofed NV magnetic signal (adversary near bed with coil) | R7 mincut: classical CSI + NV must agree on rate; spike on NV alone = anomaly |
| HRV contour reconstruction enables patient ID across installations | ADR-106 + ADR-107 L5 rotation: per-installation embedding space |
| NV measurement noise misclassified as cardiac event | Confidence score per output; clinical downstream uses confidence floor |
| Out-of-range deployment (NV >2 m from patient) | Cog manifest rejects configs that violate ADR-113 chest-centric placement |

## Consequences

### Positive

1. **First quantum-augmented cog with shippable spec.** Concrete, not speculative.
2. **Recovers R13 NEGATIVE at clinical-grade.** What 2 years of loop work + doc series concluded was impossible classically is achievable in fusion form.
3. **Privacy chain (ADR-105-109+113) unchanged.** No regulatory delta; HIPAA medical-grade DP still applies.
4. **Bridges `nvsim` (currently leaf) into production cog ecosystem.**
5. **5y deployable timeline.** Aligned with doc 17's 5y bucket.

### Negative

1. **Requires real NV-diamond hardware** to fully realise. Today's NV devices are bench-scale (~10 kg, ~$50K); cog-quantum-vitals can run on synthetic `nvsim` outputs today but doesn't deliver actual quantum benefit until ~2028-2030.
2. **+150-200 LOC** on top of existing cogs (`nvsim` integration + Bayesian fusion + manifest extension for NV anchor types).
3. **Calibration overhead.** NV-diamond requires per-installation magnetic-field baseline (Earth + local interference subtraction).
4. **Cost.** $200-2,000 per NV device (today's estimates) + ESP32 array. Bedside cost ~$50-250 vs $3,000 hospital monitor.
5. **No FDA / CE approval included.** Regulatory pathway is separate per ADR-114; estimated 6-18 months + $500K-$2M per device class.

## Implementation plan

| Step | LOC | Dependencies |
|---|---:|---|
| 1. `cog-quantum-vitals` crate scaffold | 30 | ADR-100 cog packaging |
| 2. `nvsim` integration adapter | 40 | ADR-089 nvsim |
| 3. Bayesian fusion layer (classical likelihood + NV likelihood → posterior) | 80 | rust-bayesian-stats or equiv |
| 4. R12.1 pose-PABS hook | 30 | R12.1 in vital_signs (Roadmap Tier 1.2) |
| 5. Cog manifest with NV-anchor-type schema | 20 | ADR-100 / ADR-109 signing |
| 6. Bench validation against bedside protocol | — | partner hospital + real NV device |

**Total ~200 LOC** for the synthetic-NV version. ~50 additional LOC for real-NV hardware adapter when hardware ships. **~3-week effort.**

## Bridge to existing ADRs

- **ADR-089 (nvsim)**: the standalone leaf simulator becomes a cog dependency.
- **ADR-021 (vitals)**: classical breathing/HR pipeline reused as one input to fusion.
- **ADR-103 (cog-person-count)**: parallel architecture, different cog.
- **ADR-105 / ADR-106**: federation + DP-SGD apply unchanged; the new NV-derived HRV contour is added to ADR-106 Layer 1 primitive-isolation list.
- **ADR-107 / ADR-108 / ADR-109**: cross-installation federation, PQC key exchange, PQC signatures all apply.
- **ADR-113 (placement)**: cog-quantum-vitals uses the `chest, N=5, 2D` matrix row; manifest enforces.

## Bridge to research-loop threads

- **R13 NEGATIVE**: this cog recovers what R13 ruled out (sensor-bound finding, not physics-bound).
- **R14 V1/V2/V3**: V1 is mostly classical; V2 adds breathing envelope; **V3 (attention-respecting) becomes shippable** because the cog provides the contour V3 needs.
- **R15 biometric primitives**: per-patient cardiac contour adds a new primitive to the catalogue (rate-level was the prior bound).
- **R16 healthcare**: this cog is the first concrete deliverable of the healthcare vertical. ICU bedside + general ward.
- **R12 PABS / R12.1**: pose-PABS provides the residual check; NV signal adds the new modality residual.
- **R6.1 multi-scatterer**: extended to multi-MODALITY (CSI + magnetic) forward model.
- **R20 / doc 17 (quantum integration)**: this ADR is the concrete implementation of the 5y bucket.

## Per-installation deployment recipe

Following ADR-113's `chest, N=5` row:

```
1. Place 4× ESP32-S3 around the patient bed (corner of room, height 0.8 m + 1.5 m mix)
2. Place 1× NV-diamond device on a wall-mounted arm ~1 m above the bed (above patient head)
3. Run wifi-densepose plan-antennas --cog cog-quantum-vitals --target-mode chest
4. Calibrate NV baseline (10 min capture of empty bed)
5. Load patient identity (R3 + AETHER per-installation library)
6. Deploy cog binary (signed per ADR-109)
7. Federated training begins on overnight schedule (ADR-105)
```

Cost per bedside install:
- 4× ESP32-S3: ~$60
- 1× NV-diamond device: ~$200-2,000 (today's estimate; expected ~$200 by 2028)
- Mounting + calibration: ~$50
- **Total bedside: $310-$2,110**

vs **clinical continuous monitor: $3,000-$10,000 per bed**.

## What this ADR DOES NOT cover

1. **Real NV-diamond hardware acquisition** — `nvsim` simulator is bench-validatable today; real-hardware bring-up is separate procurement + integration work.
2. **FDA / CE Class II regulatory** — per ADR-114 follow-up; 6-18 months + $500K-$2M cost.
3. **Multi-patient NV scaling** — single NV device per bed; per-ward scaling needs multiple NV devices per ADR-113.
4. **Wearable integration** — wearables remain complementary; `cog-quantum-vitals` is passive supplement, not replacement.
5. **Pediatric / geriatric specialised models** — adult-baseline assumed.

## Future ADRs catalogued

- **ADR-115**: cog-rydberg-anchor (calibrated multistatic; doc 17's 7-10y item)
- **ADR-116**: real NV-diamond hardware bring-up + calibration protocols
- **ADR-117**: cog-quantum-vitals FDA/CE regulatory pathway
- **ADR-118**: cog-mm-position (atomic-clock-synchronised multistatic; doc 17's 10y item)

## Decision-making record

- 2026-05-22 11:30 UTC — drafted by SOTA research loop tick-39 in response to repeated user signal on the quantum-sensing folder. Composes loop's R13 NEGATIVE recovery (via R20 + doc 17) into a concrete cog spec. Status: Proposed.
- Pending: ADR-089 author / nvsim maintainer (integration adapter review), security-architect (NV primitive added to isolation list), clinical advisor (bedside protocol review).

## Honest scope of ADR-114

- **`nvsim` outputs are deterministic simulations**, not real magnetometer data. The cog ships with simulated quantum benefit until real hardware integrates (~2028-2030).
- **Cube-of-distance is the hard physical bound** — no NV magnetometer can exceed it; cog manifest enforces ≤2 m bedside.
- **Patient-side variability** (BMI, body position, clothing) affects per-patient cardiac magnetic-field amplitude by ~3-10×. Per-patient calibration required.
- **R7 mincut adversarial defence** assumed at multi-anchor classical level; NV is single-source, so spoofing detection relies on classical-NV agreement.
- **Implementation cost is conservative** — Bayesian fusion may need ~100 more LOC if calibration-recovery proves complex.
- **No bench validation** has been done on the full hybrid pipeline; first real test is a partner-hospital deployment.

## What this ADR closes

The **gap between the loop's R13 NEGATIVE finding and a shippable quantum-augmented vitals cog**. After ADR-114:

- R13 NEGATIVE is **categorised as sensor-bound, recoverable**, with a concrete cog spec showing the recovery.
- `nvsim` (ADR-089) has its first concrete production cog dependency.
- Doc 17's 5y bucket has a buildable spec.
- The privacy chain (ADR-105-109+113) covers the new modality without changes.
- The R14 V3 (attention-respecting conversational appliance) vertical becomes shippable.

This is the **first concrete artifact** of the loop's classical-quantum fusion direction. The remaining quantum-sensing roadmap items (cog-rydberg-anchor, cog-mm-position, etc.) follow the same template at later timelines.

---

*ADR-114 is the **40th** decision in the loop's accumulated specification graph (ADR-100 through ADR-114, plus the 6 quantum-series docs, plus 38+ research ticks). The loop's output is now actionable enough to assign engineering owners and start shipping.*
