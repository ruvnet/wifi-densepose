# R11 — Maritime sensing: through-bulkhead RF is impossible, through-seam works

**Status:** physics scrutiny + honest verdict + 10-20y vertical map · **2026-05-22**

## TL;DR

The romantic "through-bulkhead WiFi sensing for ships and submarines" framing is **physically wrong** at WiFi bands. Steel bulkheads have a skin depth of **3.25 µm at 2.4 GHz** — a single millimetre of mild steel produces 2,674 dB attenuation, more than the link budget of any portable device by a factor of 10²². No amount of clever DSP recovers a signal through closed metal.

What **does** work is **through-seam** sensing — exploiting the diffraction leakage through gaskets, vent slots, hatch seals, and porthole gaskets. This thread maps which maritime scenarios are physically feasible and which aren't.

## Physics

### Skin depth in steel

```
δ = 1 / √(π·f·μ·σ)
```

For mild steel (σ = 1·10⁷ S/m, μ_r = 1):

| Frequency | Skin depth | Per-mm attenuation |
|---|---:|---:|
| 2.4 GHz | **3.25 µm** | **2,674 dB/mm** |
| 5.0 GHz | 2.25 µm | 3,859 dB/mm |

A 1 mm steel sheet attenuates 2,674 dB at 2.4 GHz — utterly impassable.

### Saltwater attenuation

For seawater (σ = 4.8 S/m, ε_r = 81) via the lossy-dielectric model:

| Frequency | Attenuation |
|---|---:|
| 2.4 GHz | **852.8 dB/m** |
| 5.0 GHz | 867.7 dB/m |

Saltwater is similarly opaque. A head 30 cm underwater = 256 dB additional loss = invisible. Submarine RF comms work at VLF (10-30 kHz) for exactly this reason; WiFi-band underwater detection is hopeless.

### Slot diffraction (the loophole)

For a narrow slot of width `w << λ` in an otherwise opaque conductor, the diffraction loss approximates:

```
L_slot ≈ 20·log10(λ / 2w)   when w < λ/2
       ≈ 0                   when w ≥ λ/2
```

At 2.4 GHz λ = 12.5 cm, so any slot wider than 6.25 cm is effectively transparent. A typical cabin-door gasket gap is 2-5 mm — significant attenuation (~22-30 dB) but well within link budget.

## Composite scenarios

`examples/research-sota/r11_maritime_propagation.py` computes the composite (FSPL + bulk + slot + saltwater) for seven scenarios. ESP32-S3 link budget = 121 dB, 10 dB SNR margin reserved for DSP.

| Scenario | Path used | Total loss | SNR margin | Verdict |
|---|---|---:|---:|---:|
| Man-overboard, surface-floating @ 200 m | air | 86 dB | **+25 dB** | ✅ feasible |
| Man-overboard, head 30 cm underwater | air→water | 342 dB | -231 dB | ❌ impossible |
| Crew vitals through 10 mm closed steel door | bulk steel | 1,049 dB | -938 dB | ❌ impossible |
| Crew vitals through cabin door, 2 mm seam | seam | 80 dB | **+31 dB** | ✅ feasible |
| Crew vitals through cabin door, 5 mm seam | seam | 72 dB | **+39 dB** | ✅ feasible |
| Container intrusion (30 mm vent slot) | seam | 67 dB | **+45 dB** | ✅ feasible |
| Through submarine pressure hull (30 mm steel) | bulk steel | 1,040 dB | -929 dB | ❌ impossible |

## Verticals catalogued

### ✅ Feasible at WiFi bands

1. **Man-overboard surface detection.** ESP32 + omnidirectional antenna on a ship's mast, monitoring CSI on a beacon worn by crew. Pull-down of the beacon below the waterline → CSI signature flips from "surface scatterer with sea-state Doppler" to "no signal" within 1 second. False-positive rejection via gait-frequency-band check (R10) on the surface-state CSI.
2. **Through-seam vitals in confined spaces.** Submarine berth compartments, ship cabins, lifeboat interiors. Sensor in adjacent compartment monitors heart-rate / breathing via 2-5 mm gasket leakage. Use case: **lone-watch monitoring** without crew compromise (no camera, no microphone).
3. **Container intrusion / contents change.** Sea-cargo container with at least one vent slot >2 cm leaks RF. Sensor outside monitors CSI signature; sudden change indicates contents shifted or door opened. Use case: tamper detection on bonded customs cargo, long-haul container security.
4. **Hatch-seal integrity audit.** A known-position transmitter inside a compartment, receiver outside. Closed-and-sealed hatch → only seam leakage (specific dB attenuation per gasket condition). Drift in this attenuation over time = gasket degradation. **Predictive maintenance** for watertight integrity.
5. **Engine room thermal-anomaly detection (via condensation).** RF propagation in moist air is bandwidth-dependent. Sustained CSI-amplitude drift = condensation envelope shifting = thermal anomaly. Indirect, but adds a sensing modality to engine rooms without IR cameras.

### ❌ Not feasible at WiFi bands

1. Through-hull submarine comms (use VLF/ELF instead — different industry).
2. Underwater swimmer detection (use sonar / acoustic — different industry).
3. Through-watertight-bulkhead sensing into a sealed compartment with no leakage path.
4. Through-radome of any reasonable thickness (most radomes are thin enough to pass — but this isn't the use case).

### Re-framed verticals (with caveats)

1. **Pirate-skiff approach detection (10y).** Air-link sensing from a vessel's superstructure can detect small boats approaching at radar-blind low altitudes. Range: ~100 m at 2.4 GHz (R10's foliage-less air model). The maritime version of R10's wildlife sensing.
2. **Crew situational awareness in dark / smoke (15y).** Through-seam vitals + breathing patterns inside compartments tell fire-control whether occupants are conscious. Real value-add when smoke obstructs cameras.
3. **Whale-strike avoidance (20y).** Surface-floating mammals can be detected at the surface by CSI Doppler signature; the practical issue is **range** (whales are slow, ship is fast — need 200+ m detection). The R6 Fresnel envelope at 200 m link length is ~3.5 m wide; large enough to catch a whale-sized target, marginal for smaller mammals.

## How this composes with prior threads

- **R6** (Fresnel forward model): the per-subcarrier signature of through-seam leakage is a band-passed version of the open-air signature, distorted by the slot's frequency response. Detectable, but the saliency profile differs from R5's open-room measurement.
- **R10** (foliage): the through-air maritime scenarios (man-overboard, pirate-skiff) reuse R10's free-space link budget directly. ~100 m at 2.4 GHz in clear-air conditions.
- **R1** (CRLB): 4-anchor multistatic on a small ship's superstructure (4 corners of a 10 m wheelhouse) achieves ~30 cm ToA position precision; >10 m operational ranges put us in the room-pose-quality regime.
- **R7** (mincut adversarial): essential for maritime. Single-link spoofing is easy (jammer on the dock). Multi-link consistency over 4 superstructure sensors is the only way to harden against this.

## Honest scope

- All numbers are **best-case** — ignore vessel vibration, electromagnetic noise from engine ignition systems, salt-spray on antennas, multipath from steel surfaces (which dominates real maritime CSI).
- **Salt-spray** on PCB antennas degrades them by 3-10 dB after a few hours of operation. Marine-grade conformal coating extends this, but installation is harder than land deployments.
- **Vibration** from engines / wave-slap modulates CSI at ~5-30 Hz. This is **in-band** with the gait frequencies used for R10's species classifier — making maritime gait-classification much harder than land.
- **No GPS in steel compartments.** Multistatic positioning would need an alternative reference (inertial + RF anchors on the vessel itself). This is solvable but adds installation complexity.
- The 200 m air-link range assumes a clear horizon. Real vessels have superstructure occluding many bearings; effective coverage is more like a 90° forward arc.

## What this DOES enable

- A **physically honest** maritime sensing roadmap that doesn't promise through-bulkhead capability that doesn't exist.
- Clear product categories where ESP32 + RuView stack adds value: man-overboard surface detection, through-seam vitals, container tamper detection.
- A predictive-maintenance angle (hatch-seal degradation) that has no current sensor alternative.

## What this DOES NOT enable

- Through-hull submarine sensing — physics says no at any practical bandwidth.
- Underwater sensing at WiFi frequencies — physics says no.
- Single-sensor multistatic localisation on a ship — vibration noise needs multi-sensor consensus.

## Next ticks (R11 follow-ups)

- Through-seam frequency response measurement. Place ESP32 + known signal source on opposite sides of a cabin door with a controlled gasket gap; characterise the slot transfer function vs. the slot-diffraction model.
- Vibration-suppression filter: design a notch/comb filter that removes 5-30 Hz engine-modulation from CSI, validate on a real boat (no boat available in repo, but the filter design is reproducible).
- ADR sketch for `cog-maritime-watch`: man-overboard + through-seam vitals as a maritime-specific cog package. Same ADR-103 pattern as `cog-person-count`, different model + different feature set.

## Connection back

- **R5** (saliency) — through-seam slot acts as a frequency-selective filter; the saliency profile through a seam differs from open-air saliency. New experiment opportunity.
- **R6** (Fresnel) — Fresnel envelope still applies through seam, but the slot acts as an additional spatial filter, restricting the **effective transmit position**. The composite "Fresnel-zone-AND-slot-aligned" envelope is much narrower.
- **R10** (foliage) — air-side maritime scenarios reuse R10's link-budget primitives unmodified.
- **R12** (eigenshift) — the structure-detection problem is even harder on ships because the natural drift floor includes vessel motion and engine vibration. PABS over Fresnel+vibration basis is the maritime version.
- **R14** (empathic appliances) — through-seam vitals + the V1 stress-responsive lighting framework could plausibly become "crew wellness monitoring in confined ship cabins". Privacy framework from R14 transfers directly.
