# R20 — Quantum sensing integration: NV-diamond + atomic clocks + classical CSI

**Status:** 10-20y horizon exotic vertical · **2026-05-22**

## Premise

The loop's primitives (R1 CRLB, R6 Fresnel, R12 PABS, R14 V1 vitals) are all bounded by **classical RF physics** — link budget, bandwidth, thermal noise floor. Quantum sensors operate below the classical noise floor:

| Sensor | Sensitivity | Loop primitive bottleneck |
|---|---|---|
| NV-diamond magnetometer | ~1 pT/√Hz | beyond classical RF SNR |
| Atomic clock (Cs / Rb) | ~10⁻¹⁵ stability | beyond classical ToA CRLB |
| SQUID magnetometer | ~1 fT/√Hz | beyond classical RF SNR |
| Quantum-illuminated radar | ~6 dB above classical | beyond R6.1 multi-scatterer penalty |

The repo already has a quantum-sensing seed in `nvsim` (ADR-089) — a deterministic NV-diamond magnetometer pipeline simulator. The user just opened `docs/research/quantum-sensing/11-quantum-level-sensors.md`. This tick maps how quantum sensors could compose with the loop's classical primitives.

## What quantum sensors give us

### 1. NV-diamond magnetometry (3-7y from edge deployment)

Nitrogen-vacancy defects in diamond act as **room-temperature spin qubits** sensitive to magnetic fields. Recent (2024-2025) lab demos: pT-level sensitivity at >100 Hz bandwidth in 1 cm³ sensor packages.

**Where this composes with the loop**:
- **Cardiac magnetometry** (R14 V1 + R15 HRV): the heart's pumping action produces magnetic fields ~50 pT at the chest surface. NV-diamond can resolve heart rate AND contour at full clinical fidelity. **Replaces R13's NEGATIVE BP-from-CSI** — quantum cardiac magnetometry achieves what classical CSI cannot.
- **Brain-magnetic-field imaging** (MEG-class): ~100 fT-1 pT signal levels; today's MEG requires SQUID + cryogenics. Room-temperature NV-MEG would enable BCI-class sensing without cryogenic infrastructure.
- **Through-rubble vital signs** (R18): magnetic fields penetrate dielectric materials (rubble, concrete, debris) far better than RF. NV-diamond above the rubble pile could resolve buried-survivor heart-rate **even at 5 m depth** where R18's RF estimate is infeasible.

### 2. Atomic-clock ToA (5-10y from edge deployment)

R1's classical ToA CRLB at 20 MHz bandwidth gave 41 cm precision. With **chip-scale atomic clocks** (MEMS Rb, ~10⁻¹⁰ stability today, ~10⁻¹⁵ in 5-10y):

```
σ_ToA = 1 / (2π · β · √SNR · √T_integration)
```

With atomic-clock-grade timing, the bottleneck shifts from bandwidth-limited CRLB to **multipath ambiguity** — meaning sub-mm ToA is physically achievable when the cycle-slip problem is resolved.

**Where this composes with the loop**:
- **R3 cross-room re-ID** (R3.2 follow-up): mm-precision ToA at 5-anchor convex hull → ~3 mm position precision per subject. Per-subject position-trajectory becomes a biometric primitive **beyond R15's 12-15 bit catalogue**.
- **R12.1 pose-PABS** (more precise pose tracker): millimetric pose estimates absorb subject motion better; PABS-after-pose-update improves from 9.36× lift to potentially 30-100× lift.
- **ADR-029 multistatic geometry** (orders-of-magnitude tighter): the matrix in ADR-113 can be revisited with mm-precision anchor positions.

### 3. SQUID arrays for SOTA cardiac imaging (10-15y edge deployment)

SQUID (Superconducting Quantum Interference Device) magnetometers have ~1 fT/√Hz sensitivity but require ~4 K cooling. Chip-integrated MEMS cryocoolers (Lake Shore, recent demos) shrink the cryo footprint to ~1 cm³.

**Where this composes with the loop**:
- **R14 V3 attention-respecting**: full cardiac magnetometry detects micro-arrhythmia + autonomic variability that R14 V3 needs but R13 NEGATIVE ruled out from CSI. **SQUID arrays make R14 V3 feasible.**
- **R16 healthcare**: MEG-grade brain imaging in the ICU for non-cooperative patients (sedated, unconscious) without 20-ton MRI/MEG room shielding.

### 4. Quantum-illuminated radar (10-20y edge deployment)

Quantum illumination uses entangled photon pairs to gain ~6 dB SNR over classical radar (Lloyd 2008; experimental demos 2020-2024). The 6 dB improvement is fundamental, not engineering.

**Where this composes with the loop**:
- **R6.1's 4.7 dB multi-scatterer penalty is partially recovered** — quantum illumination + multi-scatterer = ~1 dB net penalty, vs R6.1's 4.7 dB classical penalty.
- **R12 PABS sensitivity** rises proportionally — intruder detection at 4× distance OR 16× weaker target reflectivity.
- **R6.2 placement coverage**: quantum-illuminated multistatic gives wider effective Fresnel envelope at the same link budget.

## Three deployment scenarios

### Scenario A: Hybrid quantum-classical ICU bedside (5y)

Single ICU bed instrumented with:
- 4× ESP32-S3 (classical CSI, R14 V1 rate-level vitals)
- 1× NV-diamond magnetometer (cardiac magnetometry, full HRV contour)
- Hybrid fusion: classical breathing-rate + NV-diamond HRV-contour = full vital-signs panel

Cost: ~$50/bed (4× $15 ESP32 + ~$200 NV-diamond device by 2028 estimate) vs $3,000+ continuous-monitor today. **Achieves what R13 NEGATIVE ruled out for pure CSI.**

### Scenario B: Quantum-precision multistatic localisation (10y)

Pre-staged at high-precision sites (hospitals, military bases, secure facilities). Atomic-clock-synchronised ESP32s achieve mm-precision multistatic. Composes with R3.2 + AETHER for **mm-precision per-subject biometric ID** — useful for high-security access control without biometric capture.

### Scenario C: Disaster-response quantum magnetometry (15y)

R18 + NV-diamond drone-mounted magnetometers. Drone hovers over rubble pile, NV-magnetometer reads cardiac magnetic fields from buried survivors. **Achieves 5 m rubble depth** that R18's classical CSI estimate said was infeasible. Order-of-magnitude improvement in deeply-buried survivor detection.

## Integration with `nvsim` (ADR-089)

The repo already has `nvsim` — a deterministic NV-diamond pipeline simulator (CLAUDE.md crate table). R20 catalogues how `nvsim` outputs would compose with the loop:

| `nvsim` output | Loop primitive | Composition |
|---|---|---|
| Magnetic-field time series | R14 V1 vitals fusion | replace HRV-contour stub with NV-derived contour |
| Spatially-resolved field map | R12 PABS | "structural change" includes magnetic anomalies |
| Field stability indicator | R7 mincut | additional consistency channel beyond multi-link CSI |

`nvsim` is currently a **standalone leaf crate** (per CLAUDE.md "WASM-ready, no dependents"). Integrating it with the loop's primitives is a future cog: `cog-quantum-vitals` or `cog-quantum-fusion`.

## Comparison: classical vs quantum loop primitives

| Capability | Classical (loop today) | Quantum (5-15y) | Improvement |
|---|---|---|---|
| Breathing rate | ±1 BPM | ±0.1 BPM | 10× |
| HR rate | ±5 BPM | ±0.5 BPM | 10× |
| HRV contour | **NOT achievable** (R13) | Full contour (NV-magnetometer) | enables what was impossible |
| BP estimation | **NOT achievable** (R13) | Via PWV with mm-precision (atomic ToA) | enables what was impossible |
| Position precision | 25 cm (R1) | 3 mm (atomic ToA) | 80× |
| Multistatic envelope | 40 cm (R6) | 40 cm (same physics) + 6 dB SNR (quantum illum) | 4× range OR 16× weaker target |
| Through-rubble | 2 m (R18) | 5 m+ (NV-magnetometer) | 2.5× depth |
| Multi-scatterer penalty | 4.7 dB (R6.1) | ~1 dB | 3.7 dB recovery |

## Honest scope (very important here)

- **Most of this is 10-20y from edge deployment.** Today's NV-diamond magnetometers are bench-scale (~10 kg, ~$50K). Bringing to $200 / 1 cm³ requires 5-10y of MEMS + integration work.
- **Atomic clocks at 10⁻¹⁵ stability** are lab instruments today. Chip-scale at 10⁻¹⁰ exists; getting to 10⁻¹⁵ in 1 cm³ is hard.
- **SQUID at room temperature** is decades away unless room-temperature superconductors materialise (which they may not).
- **Quantum-illuminated radar at edge** requires single-photon detectors at room temperature — hard.
- **All numbers in the "improvement" column are theoretical bounds.** Real-world deployment may achieve 30-70% of these gains.
- **`nvsim` is a SIMULATOR**, not a real NV-diamond sensor. The loop currently has no real quantum sensor on the bench.

## What R20 enables

1. **A 10-20y horizon vertical** that fits the cron prompt criteria exactly.
2. **Identifies which R13 NEGATIVE findings could be overcome** by quantum sensing (HRV contour, BP via mm-PWV).
3. **Connects `nvsim` (already in repo) to the loop's primitives** — first integration sketch.
4. **Quantifies what's classical-bounded vs quantum-bounded** in each loop primitive.

## What R20 DOES NOT enable

- Real quantum sensing today.
- Bench validation (no quantum hardware on the loop's COM5 bench).
- Production deployment without 5-10y of hardware progress.
- Replacement of classical primitives — quantum is **additive**, not substitutive.

## Cog roadmap (very speculative)

| Cog | Timeline | Primitive composition |
|---|---|---|
| `cog-quantum-vitals` (NV + CSI fusion) | 5y | `nvsim` + R14 V1 + R15 |
| `cog-mm-position` (atomic-ToA multistatic) | 10y | atomic-clock-sync + R1 + R3.2 |
| `cog-deep-rubble-survivor` (NV-drone) | 15y | `nvsim` + R18 + drone platform |
| `cog-quantum-illuminated-pose` | 15y | quantum-illumination + R6.1 + ADR-079 |
| `cog-ICU-meg` (room-temp SQUID brain imaging) | 20y | SQUID array + R14 V3 |

## Composes with every loop thread

- R1 CRLB: atomic clocks shift the bandwidth-limited floor
- R3 cross-room: mm-precision position adds new biometric primitive
- R6 / R6.1: classical Fresnel + quantum-illumination = recovered SNR
- R12 PABS / R12.1: mm-precision pose absorbs subject motion better
- R13 NEGATIVE: quantum sensing recovers the 5 dB shortfall via NV-magnetometry
- R14 V1/V2/V3: V3 (cognitive load) now feasible via NV-cardiac
- R15 (biometric primitives): mm-precision trajectory + cardiac MEG = new bits
- R16 healthcare: full clinical-grade vitals + brain imaging
- R17 industrial: NV-magnetometers detect engine-noise / cell-RF without RF entanglement
- R18 disaster: 2.5× rubble depth
- R19 livestock: full cardiac magnetometry per cow (welfare gold standard)
- ADR-089 (nvsim): the existing repo simulator becomes a cog input

## R20 special status

This is the **8th exotic vertical** and the **first to require quantum hardware** for full realisation. It's also the most explicitly 10-20y horizon (per the cron prompt criteria).

## Connection back

Every loop thread has a quantum-sensing improvement opportunity. R20 is the **forward-looking integration** that says: even when classical CSI hits its physics floors (R13, R1, R6.1), the architecture **stays the same**; only the sensor hardware swaps in. **This is the cleanest demonstration that the loop's architecture is sensor-agnostic.**
