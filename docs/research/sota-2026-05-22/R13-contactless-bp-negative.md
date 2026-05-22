# R13 — Contactless blood pressure from CSI: NEGATIVE RESULT

**Status:** physics-floor scrutiny → **don't pursue as a primary product feature** · **2026-05-22**

## TL;DR

Published claims of "contactless BP from WiFi CSI" exist (Yang 2022, Liu 2021, others), with reported MAE of ±8-12 mmHg. **The physics says these claims are either (a) over-fit per-subject calibration that doesn't generalise, or (b) require hardware capabilities that production ESP32-S3 systems don't have at the typical deployment configuration.**

The honest verdict for the RuView roadmap: **do not ship BP as a primary feature.** It would be slower, less accurate, and harder to deploy than a $20 arm cuff. The breathing-rate and heart-rate features we already ship work because their motion amplitudes are 30-100× larger than the pulse waveform we'd need to recover for BP.

This thread spells out **exactly why**, with numbers, so anyone trying to add BP from CSI in the future has the scrutiny in hand.

## The two published approaches

### Approach A: Pulse Transit Time (PTT)

Measure the delay between pulse arrival at two body sites (e.g. carotid + femoral), convert to BP via the Bramwell-Hill / Moens-Korteweg equations. Calibration-free in principle if both sites are observable.

### Approach B: Pulse-contour ML

Train a model on (PPG waveform → cuff BP) pairs, recover a synthetic PPG-like waveform from CSI, infer BP. Requires per-subject calibration to defeat individual physiological variation.

Both are *physically possible*. Both have *practical floors* that make them inferior to a cuff.

## Floor 1 — PTT temporal resolution

PTT for a healthy adult is ~78.6 ms (55 cm carotid-femoral distance, 7 m/s PWV). The sensitivity is ~**0.5 ms per mmHg** (Geddes 1981, lit consensus). So:

| Target BP precision | Required PTT resolution |
|---:|---:|
| 1 mmHg | **0.5 ms** |
| 5 mmHg | 2.5 ms |
| 10 mmHg | 5.0 ms |
| 20 mmHg | 10.0 ms |

| Configuration | CSI rate | Temporal resolution | Achievable precision |
|---|---:|---:|---|
| ESP32-S3 maximum (Hernandez 2020) | ~1000 Hz | 1.0 ms | 1 mmHg — **possible at max** |
| ESP32-S3 typical deployment | ~100 Hz | 10.0 ms | 20 mmHg — **bad** |
| ESP32-S3 sensing-server actual | 30-50 Hz | 20-33 ms | **40-60 mmHg — useless** |

The "ESP32 typical" configuration cannot in principle achieve clinically meaningful BP precision via PTT. Reaching the 1 mmHg target requires running CSI at 1 kHz, which is **possible** on ESP32-S3 but **degrades** every other sensing feature (less averaging per window → noisier breathing / HR / pose). It's a destructive trade-off.

## Floor 2 — Spatial separation of two body sites

PTT requires resolving the carotid pulse signal and the femoral pulse signal **independently**. Their anatomic distance on an adult human is ~55 cm. The Fresnel envelope from R6 sets the spatial-resolution floor:

| Link length | First-Fresnel radius at midpoint |
|---|---:|
| 2 m | 25 cm |
| 5 m | 40 cm |
| 10 m | 56 cm |

For a single Tx-Rx pair to resolve carotid and femoral as **separate scatterers**, they must lie outside each other's Fresnel envelope. **A 5 m bedroom link's Fresnel envelope is wider than the carotid-femoral separation** — both sites contribute to the same window. The summed CSI cannot be uniquely decomposed into per-site signals.

Multistatic with multiple anchors could in principle invert the spatial mixing — but the inverse problem is severely ill-posed with the 4-6 anchors that are practically deployable. R12 already showed that this kind of structural-inverse-problem is the regime where naive approaches fail (negative result).

**Conclusion:** PTT from CSI requires either an unusually short link (< 1.5 m, with subject between two co-planar antennas) or a non-trivial multistatic array with a custom forward operator. Neither matches a typical RuView room deployment.

## Floor 3 — Contour recovery SNR

For Approach B (contour-based ML), we need to recover the **shape** of the pulse waveform, not just its rate. Per-motion CSI phase change at 2.4 GHz:

| Source | Amplitude | CSI phase change |
|---|---:|---:|
| Chest breathing (tidal volume) | 8 mm | **46°** |
| HR ballistocardiographic | 0.3 mm | 1.7° |
| Subject "still" micro-motion | 2 mm | 11.5° |

**Breathing motion is ~27× larger than the pulse motion** at the chest. A 4th-order Butterworth bandpass (HR band 0.8-3.0 Hz, rejecting respiration at 0.1-0.4 Hz) gives ~40 dB rejection of breathing, lifting the HR-band SNR to ~20 dB above the breathing residual.

But **subject motion** at 2 mm amplitude bleeds into the HR band — most "still" subjects exhibit micromovement at 1-3 Hz from postural correction, talking, swallowing. That micromotion is ~7× larger than the pulse signal and **shares its frequency band**. Realistic HR-band SNR with a still-but-not-motionless subject: **+20 dB**.

Literature consensus (Mukkamala 2015) for **pulse-contour shape recovery** is +25 dB minimum. We're 5 dB short. Rate is recoverable (we already ship this); shape isn't.

**Conclusion:** Contour-based BP from chest-aimed CSI is *infeasible* on a realistic subject. The published successes are either (a) measured on motionless lab subjects with a clean 25+ dB SNR (unrealistic for home deployment), or (b) overfit per-subject ML with no generalisation.

## Floor 4 — Comparison to the trivial baseline

| Device | Accuracy | Price | Latency | Calibration |
|---|---:|---:|---:|---:|
| Arm cuff (BIHS Grade A) | ±2 mmHg | $20 | 30 s | none |
| Wrist cuff (consumer) | ±5 mmHg | $30 | 60 s | none |
| Best published CSI BP (Yang 2022) | ±10 mmHg | n/a | 30 s | per-subject |
| RuView CSI (hypothetical) | ±10-15 mmHg | $9 (ESP32) | 30 s | per-subject |

CSI BP is **5-7× worse** than a $20 arm cuff, requires **per-subject calibration**, and saves the user *nothing* in time or convenience compared to a wrist cuff. The "contactless" benefit is real but doesn't outweigh the accuracy gap.

## What this means for ADR-029 / sensing-server

**Do not add BP as a feature.** Adding it would:

1. Force CSI rate up to 1 kHz, degrading every other sensing pipeline.
2. Require per-subject calibration UX, defeating the "no-setup" deployment story.
3. Introduce a feature that is provably worse than a $20 device the user can buy.
4. Erode credibility for the features that *do* work (breathing, HR, motion, occupancy) by association with a feature that doesn't.

The same argument applies to **other low-SNR continuous physiological signals**: blood glucose (no plausible CSI signature), SpO₂ (motion amplitude ~0), arterial stiffness (would need PTT, same floor as BP). Stick to the signals where the motion amplitude is large: breathing (8 mm), gross HR rate (0.3 mm + 1 Hz spectral isolation), posture/pose/occupancy.

## What this DOES tell us about R14

R14 (empathic appliances) assumed BP would *not* be available. This scrutiny confirms that assumption. The V1 / V2 / V3 vertical sketches in R14 are validated: they depend only on signals (breathing rate, HR rate, motion intensity) that *do* meet the physics floor.

## What this DOES NOT close

Some niche scenarios *might* be feasible:

1. **Single-subject pre-medical-event detection.** Trend-not-absolute monitoring — "this person's breathing has been irregular and HR variability has dropped". Doesn't need BP, just rate-and-variability features we already ship.
2. **Ballistocardiogram-based HR from a controlled bed-instrumented deployment.** Bed-frame ESP32 with subject lying still → 25+ dB SNR achievable. Out of scope for room-deployed sensing, in scope for a hypothetical `cog-bedside`.
3. **PWV with multiple Tx-Rx anchors AND a known anatomical model.** Requires per-installation calibration and ~6 anchors. Plausible but expensive — not a consumer feature.

These three niches *might* close some day. The general "BP from a $9 ESP32 in the corner" claim does not.

## Why this is a positive contribution

A research loop that only publishes successes biases toward overclaiming. The most honest thing this loop can do for the field is to **mark BP-from-CSI as off-roadmap with explicit numbers**, so future contributors don't waste cycles attempting it. This scrutiny + the R12 eigenshift scrutiny = the loop's two negative results, both worth more than another marginal positive.

## Honest scope (of the scrutiny itself)

- All four floor numbers are best-case. Real deployments worsen each by 2-5×.
- The 25 dB contour-shape requirement is from PPG literature. WiFi CSI may need *more* dB because its noise model is different from optical sensors. So the 20 dB shortfall is a *floor* on the shortfall, not a tight estimate.
- We didn't test the published BP claims directly (no labelled BP dataset in the repo). The scrutiny is purely physics-floor, not empirical replication.
- If 802.11be EHT320 channels become widely available, the bandwidth budget improves but the spatial floor (Fresnel envelope) is set by carrier wavelength, not bandwidth — so the spatial problem doesn't go away.

## Connection back

- **R1** (ToA CRLB) — bandwidth-bound floor on temporal resolution; PTT inherits this. The 0.5 ms target is below the 20 MHz HT20 single-shot CRLB (~14 ns at infinite SNR, but >5 ms in practice). Confirms PTT-from-WiFi-bandwidth is bound by averaging window length.
- **R6** (Fresnel forward model) — provides the spatial-resolution floor that defeats two-site PTT at typical room ranges. The cleanest "R6 explains why this doesn't work" example.
- **R5** (saliency) — band-spread occupancy showed why the *whole* chest motion is observable across the band; isolating a 0.3 mm pulse signal from an 8 mm breathing signal requires temporal-band filtering, not spatial saliency.
- **R12** (eigenshift, also negative) — the loop's other negative result. Same pattern: a plausible-sounding ML approach fails because the underlying signal doesn't dominate the noise/drift floor.
- **R14** (empathic appliances) — confirms R14's design choice of breathing rate + HR rate only, no BP.
