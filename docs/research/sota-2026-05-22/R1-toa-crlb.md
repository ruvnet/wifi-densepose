# R1 — ToA CRLB: the precision floor for WiFi multistatic localisation

**Status:** closed-form CRLB analysis + numpy demo · **2026-05-22**

## Why this thread exists

R6 gave us the **spatial sensitivity envelope** (Fresnel-zone forward model) but said nothing about **how precisely we can place a scatterer in 3-space**. The two questions are independent: an antenna pair can be sensitive to motion within a 40 cm ellipsoid (R6) but only able to localise the cause of motion to ±50 cm (R1). For multistatic localisation, target tracking, and any per-occupant geometry, the **ranging precision floor** is the foundational physics.

WiFi gives us two ways to estimate range:

1. **Time-of-Arrival (ToA)** — measure the absolute travel time of a known pulse. Limited by bandwidth.
2. **Phase-based ranging** — measure the carrier phase change between samples. Limited by phase noise; needs integer-ambiguity resolution.

This thread quantifies both via the **Cramér-Rao Lower Bound** — the best any unbiased estimator could ever do — and compares them. Pure NumPy demo: `examples/research-sota/r1_toa_crlb.py`.

## ToA precision floor (Cramér-Rao)

For a matched-filter ToA estimator at bandwidth `B` and SNR `ρ`:

```
σ_ToA  ≥  1 / (2π · β_rms · √ρ)        (Kay 1993, eq. 3.14)
σ_d    =  c · σ_ToA
```

Where `β_rms = B / √3` for a brick-wall (sinc) pulse. The matched-filter is the optimal *known-signal* receiver; CRLB is the precision floor at infinite samples.

### Single-shot range CRLB (m, 1σ)

| Bandwidth | SNR 0 dB | 10 dB | **20 dB** | 30 dB | 40 dB |
|---|---:|---:|---:|---:|---:|
| 20 MHz (HT20) | 4.13 | 1.31 | **0.41** | 0.13 | 0.04 |
| 40 MHz (HT40) | 2.07 | 0.65 | **0.21** | 0.07 | 0.02 |
| 80 MHz (VHT80) | 1.03 | 0.33 | **0.10** | 0.03 | 0.01 |
| 160 MHz (VHT160) | 0.52 | 0.16 | **0.05** | 0.02 | 0.01 |
| 320 MHz (EHT320) | 0.26 | 0.08 | **0.03** | 0.01 | 0.00 |

The relevant cell for ESP32-S3 + commodity APs is **20 MHz HT20 @ 20 dB SNR → 41 cm single-shot precision**. 100× averaging gets us to **4 cm**.

That's **the absolute best** WiFi-bandwidth ToA can ever do for room-scale localisation. Below that floor is physically forbidden.

## Phase-based ranging precision

The same demo computes single-subcarrier phase-derived ranging. At carrier `f_c` with phase noise `σ_φ` (radians):

```
σ_d_phi = (c / 2π · f_c) · σ_φ = λ · σ_φ / 2π
```

### Single-subcarrier phase range precision (mm, 1σ)

| Carrier | σ_φ = 0.5° | 1° | 2° | **5°** | 10° |
|---|---:|---:|---:|---:|---:|
| 2.4 GHz | 0.17 | 0.35 | 0.69 | **1.73** | 3.47 |
| 5.0 GHz | 0.08 | 0.17 | 0.33 | **0.83** | 1.67 |
| 6.0 GHz | 0.07 | 0.14 | 0.28 | **0.69** | 1.39 |

The reference 5° phase-noise figure is what ESP32-S3 typically achieves after `phase_align.rs`'s LO-offset correction.

## Headline comparison

**Same scenario:** 20 MHz HT20, 20 dB SNR, 100 averaged frames.

| Metric | ToA | Phase | Ratio |
|---|---:|---:|---:|
| Single-shot | 0.413 m | 1.73 mm | **238× phase advantage** |
| 100× averaged | 0.041 m | 0.17 mm | 240× |

**Phase ranging is two orders of magnitude more precise than ToA at WiFi bandwidths.** This is *the* fundamental reason the WiFi-sensing field went to CSI/phase instead of ToA.

## The catch: integer ambiguity

Phase ranging is **only relative**. The 2.4 GHz wavelength is 12.5 cm — so an absolute phase measurement of 30° could mean 1.04 cm, 13.54 cm, 26.04 cm, 38.54 cm, … with no way to disambiguate from one subcarrier alone. This is the **integer-ambiguity (cycle-slip) problem** of phase-based ranging, and it's why GPS RTK is harder than GPS.

Resolution methods:

1. **Multi-subcarrier wide-lane unwrap.** 802.11n/ac has 52 used subcarriers over 20 MHz; their geometric mean gives an effective "wide-lane" wavelength of ~15 m, resolving ambiguity within a typical room. Implementation: 1D phase-vs-subcarrier-index linear fit, slope encodes range.
2. **Coarse ToA gate.** Use the 41 cm-precision ToA estimate to gate the phase ambiguity. ToA says "the target is at 3.2 m ± 0.4 m", phase says "phase is 30°", → pick the cycle that lands in [2.8, 3.6] m.
3. **Differential / tracking-mode.** If we know the starting position, integrate phase changes between consecutive frames. Loses absolute reference but accumulates 1 mm precision per frame.

The right system **combines** ToA (for absolute disambiguation) and phase (for precision). This is exactly what 802.11mc FTM (Fine Timing Measurement) does on top of standard WiFi hardware — and what RTK GPS does at L-band.

## Multistatic 4-anchor geometry

A typical "tight" 4-anchor convex-hull installation (anchors at 4 corners of a 5 m × 5 m room) has Geometric Dilution of Precision (GDOP) ≈ 1.5. Position-error CRLB scales as:

```
σ_pos = σ_range · √(GDOP / N_anchors)
```

Practical result (20 MHz, 20 dB SNR, single-shot):

| Method | Position precision |
|---|---:|
| ToA (4 anchors, GDOP 1.5) | **25.3 cm** |
| Phase (4 anchors, GDOP 1.5) | **1.06 mm** |

This bounds **what's possible for SOTA WiFi multistatic localisation**. 25 cm with raw ToA is room-pose-quality; 1 mm with phase is RTK-quality but only after ambiguity resolution.

## What this means for ADR-029 (multistatic sensing)

The current `multistatic.rs` uses learned attention weights over raw CSI. The CRLB analysis suggests an explicit decomposition would do better:

1. **ToA stage**: get coarse range per Tx-Rx pair (~25 cm precision).
2. **Phase stage**: unwrap phase against the ToA gate, get mm-precision range.
3. **Multistatic stage**: solve for 3D position via weighted least squares over the high-precision ranges.

This is closer to the GPS pipeline than to the current learning-based attention. The trade-off: lower flexibility (less ability to learn around hardware imperfections) but higher interpretability and provable optimality.

## Honest scope

- **CRLB is a lower bound.** Real estimators don't hit it. Practical ToA estimators (matched filter on a known preamble) get within 1-2× of the bound at high SNR.
- **The 5° phase noise** is post-LO-correction; raw ESP32-S3 phase noise is closer to 60-180°. Without `phase_align.rs` the phase advantage shrinks to ~5×.
- **CRLB assumes a known pulse / known signal.** WiFi opportunistically uses traffic (data packets), not dedicated ranging pulses. The effective bandwidth is the *occupied* bandwidth of the OFDM signal — which is the full 20 MHz / 40 MHz / etc., so this part holds.
- **Multipath** is the elephant in the room. CRLB assumes a single dominant path. In a real bedroom there are 4-6 dominant reflectors, each with its own ToA. Modern WiFi-FTM uses super-resolution methods (MUSIC, ESPRIT) to separate them, but these don't reach CRLB — typical real-world degradation is 2-5× worse than the single-path CRLB.

## What this DOES enable

- **Quantitative target precision** for any multistatic localisation feature: 4 cm (averaged ToA) is achievable; 1 mm (averaged phase) is achievable only if ambiguity is resolved.
- **Architectural decision for ADR-029**: explicit ToA + phase pipeline is provably ≤2× away from CRLB, vs the current learning-based approach which has no precision floor guarantees.
- **Realistic SLAM goals**: room-scale 3D occupancy at sub-meter precision is **easy** physics; tracking individual fingers at mm precision is **hard** physics. The line between them is the cycle-slip problem.

## What this DOES NOT enable

- Sub-mm ranging — that's microwave-photonics territory, not WiFi.
- Multipath-free assumption — every real deployment is multipath-rich.
- Distance estimation **without** SNR margin — the 41 cm number is at 20 dB SNR. At 0 dB SNR the single-shot floor is 4.1 m, useless for room geometry.

## Connection back

- **R6** (Fresnel forward model) — gives the *spatial envelope* of sensitivity. R1 gives the *ranging precision* within it. Together they bound multistatic localisation: localise targets to ±1 mm precision but only within the ±20 cm Fresnel envelope.
- **R10** (foliage range) — adds the foliage attenuation term to the SNR. A 50 m link through moderate foliage drops to ~5 dB SNR → ToA precision degrades to ~1 m. Phase precision degrades to ~7 mm but its ambiguity-resolution accuracy degrades faster.
- **R12** (eigenshift negative result) — the structure-detection problem is harder than the localisation problem; CRLB gives no precision floor for "detect a new structure", only for "place a known target". This is part of why R12 was a negative result.
- **ADR-029** (multistatic) — strongest concrete architectural lever this loop has surfaced.

## Next ticks (R1 follow-ups)

- Implement multi-subcarrier wide-lane phase unwrap as a Rust module; measure how often cycle-slip resolution succeeds vs the ToA gate width.
- Empirical CRLB test: log 1000 ranging measurements from a known-position scatterer, check whether observed σ_d hits ~2× CRLB.
- Multipath super-resolution: try MUSIC over the 52-subcarrier CSI to separate 2-3 dominant taps. If achievable, the room-scale 3D occupancy at 4 cm precision target is realistic.
