# Honest Classical-Quantum Fusion: Composing the SOTA Loop with the Quantum-Sensing Series

## SOTA Research Document — Quantum Sensing Series (17/—)

| Field | Value |
|---|---|
| **Date** | 2026-05-22 |
| **Domain** | Classical CSI loop primitives × quantum-sensing series (11-16) × honest composition |
| **Status** | Research integration — bridges the 11-16 quantum-sensing series with the 2026-05-22 SOTA research loop |
| **Refines** | docs 11, 12, 13, 14, 15, 16; ADR-089 (nvsim); ADR-029 (multistatic); ADR-021 (vitals) |
| **Companion docs** | SOTA loop's `R1, R3, R5-R15, R16-R20` + ADR-105 through ADR-109 + ADR-113 |
| **Audience** | RuView contributors deciding whether/how to integrate quantum sensors with the existing classical stack |

---

## TL;DR

Doc 16 (Ghost Murmur) reality-checked overclaimed 40-mile NV magnetometry and sketched a sober RuView-grounded version. Doc 17 takes the next step: **maps the SOTA loop's classical findings (R1-R20) onto the quantum-sensing series and identifies the highest-leverage honest fusion points**.

Two claims:

1. **The classical loop already specifies what NOT to attempt quantum-side.** R13 NEGATIVE ruled out BP and HRV-contour from classical CSI for physical-floor reasons. Doc 16 ruled out 40-mile cardiac magnetometry for cube-of-distance reasons. **Combined, these two negatives bound what any honest quantum-classical fusion can claim.**
2. **The intersection of classical-bounded and quantum-bounded gives us a precise specification** for a "honest fusion" cog. The cog adds NV-diamond cardiac magnetometry to the existing classical stack at **1-2 m bedside ranges** (where the cube law gives ~1 pT/√Hz SNR), not 40 miles.

This document is the bridge between two reality-checks. It produces:

- A specification for `cog-quantum-vitals` (1-2 m bedside; classical + NV fusion)
- A mapping of which loop primitives benefit most from which quantum modality
- An explicit "what we are NOT building" list

---

## 1. The loop output (recap for quantum-sensing-series readers)

The 2026-05-22 SOTA loop produced 37+ ticks across 5 research strands:

| Strand | Output | Quantum-sensing intersection |
|---|---|---|
| Physics floor | R1 CRLB, R6 Fresnel, R6.1 multi-scatterer | **atomic clocks beat R1; quantum illumination beats R6.1** |
| Spatial intelligence | R5 saliency, R6.2 placement (9-tick family), R12 PABS | quantum-illumination boosts PABS sensitivity |
| Identity / biometrics | R3 cross-room re-ID, R15 RF biometric primitives | mm-precision position via atomic ToA = new biometric |
| Negative results | R12→POSITIVE, R13 contactless BP/HRV NEGATIVE, R3.1 architecture-error | **R13 NEGATIVE is recoverable via NV-magnetometry** |
| Exotic verticals | R10 wildlife, R11 maritime, R14 home, R16 healthcare, R17 industrial, R18 disaster (integrates `mat`), R19 livestock, R20 quantum integration | All compose with quantum modalities at parameter swaps |
| Privacy + federation chain | ADR-105/106/107/108/109/113 | Cog-distribution + DP for quantum-augmented cogs |

## 2. Mapping per quantum modality (from docs 11-16)

### 2.1 NV-diamond magnetometers (docs 11.2.1, 13, 14, 15, 16)

**Classical bottleneck this beats**: R13 NEGATIVE (CSI HRV-contour 5 dB short of recoverable).

**Honest range**: cube-of-distance falloff means NV is bedside (1-2 m), not building-scale. Doc 16 already established this.

**Fusion proposal**: `cog-quantum-vitals` bedside add-on. ESP32 array provides multi-subject context (R6.2.5), occupancy (R12 PABS), breathing rate (R14 V1); NV-diamond provides the per-patient HRV contour that ESP32 cannot.

| Capability | Classical alone | NV alone | Fusion |
|---|---|---|---|
| Multi-bed coverage | ✅ R6.2.5 | ✗ (cube law) | ✅ classical drives |
| Breathing rate | ✅ R14 | ✅ but redundant | classical is enough |
| HRV contour | ❌ R13 | ✅ at <2 m | **NV adds this** |
| Through-rubble | ✅ R18 (1-2 m) | ✅ better (5 m) | classical screens, NV confirms |
| Cost | ESP32 ~$15/anchor | ~$200-2K/device | hybrid amortises |

The fusion's value is **per-patient HRV at clinical fidelity**, not multi-subject. Doc 16's sober posture transfers directly.

### 2.2 SQUID magnetometers (doc 11.2.2)

**Classical bottleneck this beats**: same as NV (R13 NEGATIVE) plus 1000× higher sensitivity for **MEG-class** brain imaging.

**Honest range**: 4 K cryogenics today; room-temp SQUID is 15-20y out. **Not near-term for edge deployment.**

**Fusion proposal (long horizon)**: `cog-ICU-meg` for sedated ICU patients. The loop's R16 healthcare vertical specifies the placement matrix; SQUID array sits inside it for brain-activity monitoring without 20-ton MRI shielding.

This is the loop's most speculative quantum integration. Out of scope for any near-term roadmap line.

### 2.3 Rydberg atom sensors (doc 11.2.3, 11.4)

**Classical bottleneck this beats**: R1's ToA CRLB at 20 MHz bandwidth. Rydberg vapor cells provide self-calibrated broadband RF detection from DC to THz.

**Honest range**: lab-scale today (10 cm vapor cell); industrial deployment 5-10y.

**Fusion proposal**: `cog-rydberg-localiser` — Rydberg sensor as one anchor in the R6.2.2 multistatic array. The Rydberg anchor provides **absolute amplitude calibration** that the ESP32 array can't deliver (ESP32 RX sensitivity varies by ±3 dB per device). Calibrated multistatic enables Cramér-Rao-bound-tight ToA estimation per R1.

| Capability | Classical ESP32 only | Rydberg + ESP32 fusion |
|---|---|---|
| ToA precision | 25 cm (R1 + multistatic) | Approaches CRLB floor (~10 cm) |
| Self-calibration | ✗ | ✅ (Rydberg is SI-traceable) |
| Cost | $15/anchor | $200+ for Rydberg, $15 for rest |

This is the cleanest **near-term** quantum-classical fusion: one expensive precision anchor + many cheap classical ones.

### 2.4 SERF magnetometers (doc 11.2.4)

**Classical bottleneck this beats**: very-low-frequency (DC-1 kHz) biomagnetic detection where ESP32 has zero coverage.

**Honest range**: vapor cell heated to 150°C; requires magnetic shielding for shipped sensitivity. Lab + niche industrial.

**Fusion proposal**: out of scope for typical RuView deployment. Useful for highly specialised biomedical scenarios in shielded rooms.

## 3. The "honest fusion" pattern

Combining doc 16's sober posture with this loop's outputs:

```
                  CLASSICAL CSI                                  QUANTUM SENSOR
                  (R1-R20 primitives)                            (doc 11 catalogue)

  STRENGTHS       multi-subject, large coverage,                bedside fidelity,
                  cheap, federation-ready,                      contour-level signals,
                  privacy-preserving (ADR-106)                  beyond classical noise floor

  WEAKNESSES      R13 NEGATIVE (no BP/HRV-contour),             cube-of-distance falloff,
                  R6.1 4.7 dB penalty,                          cryogenics (SQUID),
                  ToA CRLB-bound at 20 MHz                      cost ($200-$10K/device today)

                  ↓                                              ↓
                                FUSION
                  ESP32 array provides MULTI-SUBJECT CONTEXT;
                  quantum sensor provides PER-PATIENT FIDELITY
                  Honest claim: ~$50/bed clinical-grade vitals
                  by 2030, vs $3,000 hospital monitor today.
```

This is the same pattern as doc 16's Ghost Murmur sober version: don't claim 40 miles, claim bedside; let the classical infrastructure carry the geometry while the quantum sensor carries the fidelity.

## 4. Cog roadmap (integrates docs 14-16 + loop R20)

| Cog | Series-anchor doc | Loop primitives composed | Timeline |
|---|---|---|---|
| `cog-quantum-vitals` (NV + CSI) | docs 13, 14, 15 (nvsim) | R14 V1 + R15 rate-level + NV HRV contour | 5y |
| `cog-rydberg-anchor` (calibrated multistatic) | doc 11.4 | R1 CRLB + R6.2.2 N-anchor + Rydberg | 7-10y |
| `cog-mm-position` (atomic clock) | doc 11 (not deep-dived) | R1 + R3.2 + atomic clock | 10y |
| `cog-deep-rubble-survivor` (NV drone) | docs 13, 16 | R18 + NV via drone | 15y |
| `cog-ICU-meg` (room-temp SQUID) | doc 11.2.2 | R14 V3 + SQUID array | 20y |

All five cogs **stay sober** — no Ghost Murmur 40-mile claims. All are bedside / single-room / short-range deployments.

## 5. What this does NOT enable (the doc 16 inheritance)

- **No 40-mile cardiac magnetometry.** Doc 16's reality check stands.
- **No through-multiple-walls quantum sensing at any range.** Magnetic fields fall as 1/r³; even quantum sensors can't fix that.
- **No replacement of medical devices** without FDA / CE Class II approval per device class.
- **No quantum-enhanced WiFi protocol changes** — Layer 1 stays classical; fusion is at the application/cog layer.

## 6. What this DOES enable

1. **A clear integration story** between the existing 6-doc quantum-sensing series and the SOTA loop's 37+ ticks.
2. **Five concrete fusion-cog roadmap items** spanning 5-20y, all with honest scope.
3. **A "what we are NOT building" list** that protects against future overclaim.
4. **A bridge** for journalists / researchers / contributors who want to understand what's plausible vs press-release.
5. **A composition of R13 NEGATIVE recovery** with doc 16's sober range scope: the loop says R13 ruled out classical CSI HRV-contour; doc 17 says NV-diamond recovers it, but only at bedside ranges (cube law).

## 7. Honest scope of this integration doc

- **Doc 17 is a synthesis**, not a research contribution itself. The substance lives in docs 11-16 + loop ticks.
- **Fusion benchmarks have not been measured**: no bench-validated joint NV+ESP32 setup exists in the repo.
- **Cube-of-distance is the gating physics** for any magnetometry application. Improvements come from sensitivity (NV: 1 pT/√Hz; SERF: 0.16 fT/√Hz) and AI noise stripping, **not from beating physics**.
- **The 5y/10y/15y/20y timelines** assume sustained MEMS + integration progress. Setbacks plausible.
- **Privacy framework (ADR-106 medical-grade ε=2)** applies to quantum-augmented vitals data the same way.
- **No replacement of mature wearable monitors** (Polar / Apple Watch / clinical telemetry). Fusion supplements; doesn't replace.

## 8. Integration with `nvsim` (ADR-089)

Per docs 14 + 15, `nvsim` is the repo's deterministic NV-diamond pipeline simulator (standalone leaf crate, WASM-ready). Doc 17 makes the integration concrete:

```
nvsim_output (magnetic field time series, magnetic field map, stability indicator)
            ↓
    ┌───────────────┬─────────────────┬───────────────────┐
    ↓               ↓                 ↓                   ↓
  R14 V1         R12 PABS           R7 mincut         R6.1 forward
  (fusion)       (structural)      (consistency)     (residual basis)
                                                       ↓
                                              cog-quantum-vitals
                                              (5y deployable)
```

This is the **specific code-path** that gets `nvsim` (currently a standalone leaf) into production via the loop's primitives. ~150 LOC of glue code in a new `cog-quantum-vitals` crate.

## 9. Cross-reference index (every loop output → quantum-series doc)

| Loop output | Quantum-series anchor doc |
|---|---|
| R13 NEGATIVE (5 dB shortfall) | doc 13 (NV neural magnetometry) recovers it for HRV |
| R14 V1 (breathing rate stress) | doc 12 (quantum biomedical) — classical is enough |
| R14 V3 (attention state contour) | doc 13 + doc 11.2.2 SQUID for MEG |
| R6.1 4.7 dB penalty | doc 11.3.3 quantum illumination (+6 dB) |
| R1 ToA CRLB (25 cm) | doc 11.4 Rydberg + atomic clock chain (~10 cm) |
| R12.1 pose-PABS | doc 11.4 Rydberg-calibrated anchor → tighter pose |
| R18 disaster (1-2 m rubble) | doc 13 NV cardiac → 5+ m depth |
| R20 vertical (quantum integration) | doc 17 (this) consolidates |

This index lets a reader navigate: "I'm interested in X loop finding; here's the quantum context that extends it."

## 10. Connection back

This document is the **explicit handshake** between the SOTA research loop (2026-05-22) and the quantum-sensing research series (2026-03-08 onwards). The two series produced complementary outputs — the loop on classical CSI primitives, the quantum series on quantum sensors. Doc 17 stitches them together with the same "sober scope, honest claims" posture that doc 16 established.

The closing observation matches doc 16's: **the architectural value of RuView is in honest, well-factored sensing infrastructure that survives reality-checks**. Adding quantum sensors doesn't change the architecture; it adds parameters. The same R3, R7, R12, R14, ADR-106, ADR-113 framework applies. **The loop's output is the contract; quantum sensors are an upgrade path.**

---

*Doc 17 closes the 11-16 series' loop with the 2026-05-22 SOTA research loop. Doc 18+ (future) might cover specific implementation milestones for `cog-quantum-vitals` or expand on quantum-illumination radar at edge.*
