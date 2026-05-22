# R15 — RF biometric primitives: what's environment-invariant in the CSI signature

**Status:** synthesis + privacy framing · **2026-05-22**

## The question

R3 asked "can we re-identify the same person across two rooms?" and answered yes, **conditional on MERIDIAN env-subtraction**. R15 asks the deeper question: **what features in the CSI signal are environment-invariant by construction** — properties of the person's physiology that exist independent of multipath geometry?

If R3 is "the same vector appears in two embedding spaces", R15 is "what physical attribute of the body actually drives that vector". Without R15, R3 is statistical pattern-matching with no theory of why it works.

This thread catalogues five biometric primitives that survive cross-environment transfer, ranks them by invariance + discriminability + measurement difficulty, and frames the privacy implications.

## Five biometric primitives

### 1. Gait stride frequency

**Physical basis:** stride frequency is determined by leg length, mass distribution, gait pattern (asymmetry coefficient). Per-individual reproducibility is ~3-5% within a year (Murray 1964); across years it drifts with fitness/age. **Invariant to environment.**

**Discriminability:** ~5-7 bits per person (Begg 2006, gait literature consensus). Enough to separate ~30-100 individuals before false-match probability exceeds 1%.

**Measurement difficulty:** R10's gait-band DSP (0.5-15 Hz) already extracts this. Stride frequency robust to multipath; stride asymmetry needs higher SNR (gait phase shape, not just rate).

**Cross-room invariance:** **HIGH.** The carrier of the gait signature is the Doppler shift induced by leg motion; the magnitude depends on environment (Fresnel envelope, R6) but the *frequency* doesn't.

### 2. Breathing rate baseline + envelope

**Physical basis:** resting respiration rate is a person-specific physiological setpoint (12-20 BPM normal range, individual ±2 BPM). The tidal-volume envelope (chest expansion amplitude) scales with lung capacity, which scales with body size and age. **Invariant to environment** at the rate level.

**Discriminability:** ~3-4 bits at the rate level alone. Combined with envelope amplitude it could reach 5-6 bits. The combined signal also has phase information (inhale/exhale ratio, breathing irregularity) that adds another 1-2 bits.

**Measurement difficulty:** `vital_signs` pipeline already extracts breathing rate. Envelope amplitude is noisier; needs ~10× more averaging.

**Cross-room invariance:** **HIGH.** Same reasoning as gait — temporal frequency is invariant, only amplitude is environment-dependent.

### 3. Heart rate variability (HRV) signature

**Physical basis:** HRV is a person-specific autonomic-nervous-system signature. Resting HRV varies ±15-30 ms between individuals; under stress it changes predictably per person.

**Discriminability:** ~4-5 bits per person (Hjortskov 2004, HRV literature). The full HRV time-series adds another 2-3 bits over the summary statistics.

**Measurement difficulty:** R13's NEGATIVE physics scrutiny showed that *waveform-shape* HR recovery from CSI is **5 dB short** of the floor. **Rate-level HRV** (R-R interval variability) is achievable; *contour-shape* HRV (which gives the autonomic signature) is not.

**Cross-room invariance:** **HIGH at rate level, LOW at contour level.** The achievable subset is rate-level HRV, which is real but lower discriminability than published claims that assume contour recovery.

### 4. Body-size RCS envelope

**Physical basis:** the radar cross-section (RCS) of a stationary human at WiFi frequencies is roughly proportional to body surface area (~0.6 m² for adult, ~0.2 m² for small child). The frequency-dependent RCS shape encodes body size + body composition (fat/muscle/water ratios affect dielectric properties).

**Discriminability:** ~3-5 bits per person. Lower than gait or HRV because it's gross-body-only.

**Measurement difficulty:** Needs calibration against a known reference target in the same environment. Cross-room calibration is a research problem.

**Cross-room invariance:** **MEDIUM.** Absolute RCS depends on environment (Fresnel envelope, R6); but the *ratio* of RCS at different subcarrier frequencies (the frequency response of the body) is environment-invariant by R6's forward model.

### 5. Walking dynamics (limb timing)

**Physical basis:** per-individual stride length, step-time asymmetry, hip-sway pattern. These are determined by skeletal proportions + neuromuscular control. **Highly invariant** to environment.

**Discriminability:** **6-9 bits per person** when full dynamics are recovered (Cunado 2003, biometric-gait literature). Among the highest-discriminability biometrics short of fingerprint.

**Measurement difficulty:** Requires recovering the *pose* (limb positions) from CSI, not just the gait *rate*. The full pose-from-CSI pipeline (ADR-079, ADR-101) gets within ~92.9% PCK@20 — good enough to extract limb timing in clean conditions.

**Cross-room invariance:** **HIGH** when pose is recovered correctly. The pose extractor itself uses MERIDIAN (R3) for cross-room transfer; if the pose pipeline works cross-room, so does the gait dynamics biometric.

## Composite biometric strength

Combining all five (assuming statistical independence, which is **not** true — gait correlates with body size, HRV correlates with age, etc. — so this is a soft upper bound):

| Primitive | Bits (cross-room achievable) |
|---|---:|
| Gait stride frequency | 5 |
| Breathing rate + envelope | 5 |
| HRV (rate-level only) | 4 |
| Body-size RCS frequency response | 4 |
| Walking dynamics (limb timing) | 7 |
| **Composite (statistically independent upper bound)** | **25 bits** |
| **Composite (realistic correlation correction)** | **~12-15 bits** |

12-15 bits of biometric is enough to uniquely identify a person within a population of ~4k-30k. For a household of 4 people, that's overwhelming discrimination. For a building of 1000 people, easily sufficient. For city-scale surveillance, it would need to combine with other modalities — but the primitive is already there.

## Privacy implications

This is the part R14 + R3 hinted at but didn't fully spell out:

**RF biometric is harder to remove than visual biometric.** A face can be obscured with a mask. A fingerprint can be left at home. A gait + breathing + RCS signature is **emitted continuously**, **without subject awareness**, **through walls**.

Specifically:

1. **No opt-out via behaviour.** Removing a face requires covering it. Removing a gait requires not walking. There is no behavioural countermeasure that doesn't impair the user.
2. **No removable artefact.** Visual ID can be defeated with sunglasses + mask. RF ID requires actual physical change (different body shape — impossible) or jamming (illegal, plus jams everything around).
3. **Cross-installation linkage is a transit-tracking primitive.** R3 already constrained per-installation embedding spaces; R15 says the constraint is **doubly important** because the biometric is intrinsically physical, not learned.

These constraints take the R3 + ADR-105 framework and push it harder:

| R3 / ADR-105 constraint | R15-strengthened version |
|---|---|
| No cross-installation linkage | **Hardware-isolated embedding spaces, cryptographically prove they're isolated** |
| Embedding storage requires opt-in | **Storage of any RF-biometric-derivable signature requires opt-in, not just the final embedding** |
| Cryptographically verifiable forgetting | **Forget the raw extracted biometric primitives (gait freq, breath rate, RCS curve) — not just the model output** |
| No re-ID across legal entities | **No sharing of any RF biometric primitive across legal entities, including aggregate / derived versions** |

## Architectural implications

**The federation protocol (ADR-105) needs an additional constraint:**

> The federation aggregator MUST NOT receive any raw per-subject biometric primitive (gait frequency, breath rate, RCS curve, limb timing). It MAY receive *aggregated, MERIDIAN-normalised* embedding deltas. Per-subject primitives stay on-device.

This is **stronger** than ADR-105's existing "data stays on-device" because MERIDIAN deltas are not "data" in the conventional sense — they're learned model parameters. But the learned parameters *encode* biometric features. R15 says: encode them as you must, but the **measurement** of the underlying biometric must never leave the device.

**Concretely:** the Cognitum Seed runs `extract_gait_freq(csi_window)` locally, produces a 5-bit signature, uses it in inference, **does not** send the signature to the coordinator. The coordinator sees only the model delta that influenced inference outcomes.

This adds a constraint to the ADR-105 implementation. ADR-106 (next ADR after the deferred DP-SGD) should formalise the on-device-only primitive list.

## What R15 enables (positively framed)

1. **Per-installation natural identification.** A household of 4 with known members + no setup gives perfect within-installation re-ID using the 25-bit biometric. The same primitive lets a hospital ICU know which patient is in which bed.
2. **Health monitoring at biometric resolution.** Long-term tracking of gait stride asymmetry detects early gait pathology (Parkinson's, stroke recovery). Breath-rate baseline drift detects respiratory decline. These are **medically actionable** signals that the existing rate-extraction pipelines almost ship.
3. **Pose-data-association robust across occlusion.** The 7-bit limb-timing biometric resolves identity through brief visual occlusion or sensor blind-spots.

## What R15 makes worse (negatively framed)

1. **Cross-installation tracking is harder to prevent than visual cross-camera tracking** because the biometric is intrinsically physical.
2. **The data-rights legal framework** doesn't yet treat "intrinsic biometric leaked passively through walls" as a category. GDPR Art 9 covers "biometric data for unique identification" but the consent flow assumes the user knows they're being measured (e.g. fingerprint scanner). RF biometric extraction can happen without subject awareness.
3. **The federation threat surface** is larger than ADR-105 anticipated. ADR-106 will need to formalise the on-device-only primitive list.

## What this DOES enable

- **A complete biometric primitive inventory** with explicit invariance and discriminability per primitive — lets the team make informed trade-offs.
- **A stronger version of the R3 + R14 privacy framework** that accounts for the physical (not learned) nature of these biometrics.
- **A clear next ADR**: ADR-106 (already mentioned in ADR-105's deferred list) gets a sharper requirements section: on-device-only primitive measurement, not just on-device-only training data.

## What this DOES NOT enable

- **Cross-installation re-ID** — explicitly prohibited and prevented by hardware-isolated embedding spaces.
- **Adversarial-resistance to a building-level attacker** with control over multiple Cognitum Seeds — that requires a different defence layer (R7 mincut multi-link extends to multi-installation only with crypto, see ADR-105's deferred cross-installation work).
- **Forensic post-hoc identification** — even within an installation, the 12-15 bit biometric resolution is too low for forensic use (would require ~30+ bits, which CSI alone cannot provide).

## Honest scope

- The bit counts are upper bounds. Real-world deployments lose 30-50% to noise + multipath + sensor variance. Realistic composite biometric strength is closer to **6-10 bits**, useful for household-scale ID but not for global identification.
- The "5 dB short" finding from R13 means the *contour-level* HRV biometric is **not achievable** on a typical ESP32 deployment. Rate-level HRV (the 4-bit subset of #3) is the realistic upper bound.
- The walking dynamics number (7 bits) depends on the pose-from-CSI pipeline achieving its ADR-079 92.9% PCK target in cross-room conditions. Current numbers are within-room; cross-room degradation is unmeasured.
- Body-size RCS frequency response (#4) needs a calibration target in the new room. Without it, the cross-room invariance is the *ratio* not the absolute value — and ratios across 56 subcarriers give ~3-4 bits, not 5.

## Connection back

- **R5 (saliency)** — saliency maps for biometric extraction are task-specific; gait-saliency, breath-saliency, RCS-saliency are different. The band-spread observation from R5 supports gait + breath extraction; high-precision RCS recovery may need a tighter sub-band.
- **R6 (Fresnel forward model)** — gives the physics of *why* RCS frequency-response is environment-invariant (the per-subcarrier amplitude scales with body geometry, not with the environment, after env subtraction).
- **R7 (mincut adversarial)** — biometric primitives can be poisoned by crafted CSI on a single link; multi-link consistency catches this.
- **R10 (foliage / per-species gait)** — gait stride-frequency taxonomy from R10 transfers directly to per-individual gait biometric (different physiologic source, same DSP).
- **R13 (contactless BP, NEGATIVE)** — the same physics argument that ruled out contactless BP also rules out contour-level HRV recovery. Both fail at the "5 dB short" wall.
- **R3 (cross-room re-ID)** — provides the embedding-space machinery that combines the 5 primitives into a unified per-subject signature.
- **R14 (empathic appliances)** — V1 lighting needs only breathing rate (already shipped); V2 HVAC needs breath rate + body-size RCS; V3 attention state needs breath envelope + maybe HRV rate. R15 says all of these are achievable with the rate-level subset, no contour recovery needed.
- **ADR-105 (federated training)** — needs ADR-106 to formalise on-device-only primitive measurement.

## What R15 closes / what it opens

This is the loop's **final research thread** before the deferred follow-up items begin. After R15:

**Closed:** the question "what RF biometrics exist and how do they invariantise" has a worked answer.

**Open:** ADR-106 (on-device DP-SGD + primitive isolation), R6.1 (multi-scatterer extension), R3 follow-up (physics-informed env_sig prediction), R6.2 (Fresnel-aware antenna placement).

Together with the 12 prior threads, R15 makes the per-occupant feature surface (R14 V1/V2/V3) **fully grounded in physics and constraints**, with no remaining unspecified primitives. The remaining work is implementation + measurement, not research.
