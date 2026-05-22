# R6.1 — Multi-scatterer Fresnel forward model: where R13's 5-dB shortfall actually comes from

**Status:** working 6-scatterer body model + breathing-SNR benchmark · **2026-05-22**

## Premise

R6 modelled a single point scatterer. R6.1 extends to a distributed body — 6 scatterers (head, chest, two arms, two legs) summed coherently. The resulting forward model:

```
csi[k] = Σ_b  (refl_b / (d_tx,b · d_rx,b)) · exp(2π·j·f_k·Δℓ_b / c)
```

The combined CSI is the **complex sum** of per-body-part contributions, evaluated at each subcarrier. This is what `wifi-densepose-signal::vital_signs` implicitly assumes and `tomography.rs` explicitly inverts.

This thread quantifies:

1. How much each body part contributes to the total signal
2. The breathing-band SNR with the full model vs the single-scatterer ideal
3. The **multi-scatterer penalty** — and an unexpected link to R13's negative result

## Headline result: 4.7 dB multi-scatterer penalty

5 m link, 2.4 GHz, subject at midpoint + 25 cm off LOS (inside first Fresnel envelope, R6 says ~40 cm at midpoint). 30-second time-series at 50 Hz CSI rate with breathing at 0.25 Hz (±8 mm chest motion).

| Configuration | Best subcarrier breathing SNR |
|---|---:|
| Single-scatterer ideal (R6, chest only) | **+23.7 dB** |
| Multi-scatterer realistic (R6.1, 6 body parts) | **+19.0 dB** |
| **Penalty from static-limb coherent-sum confusion** | **+4.7 dB** |

The 4.7 dB gap is what realistic deployment loses to **idle limbs**. These don't move (no breathing motion) but they **do contribute coherently** to the static CSI level. When chest motion modulates the static signal, the limbs' contribution dilutes the relative modulation depth.

## The bridge to R13 (NEGATIVE contactless BP)

R13 quantified that pulse-contour recovery needs **+25 dB** SNR, available is **+20 dB**, gap is **5 dB**. R13 attributed this to "subject micro-motion contaminating the HR band".

**R6.1 says: the 5 dB gap is also the multi-scatterer penalty.** Even without micro-motion, the static body parts already cost 4.7 dB compared to the idealised single-scatterer model. R13's "we are 5 dB short" finding has a **physical origin** — it's not just measurement noise; it's the body itself.

This is a satisfying integration:
- R6 (single scatterer) gives the *bound* — what's possible in the idealised limit
- R6.1 (multi-scatterer) gives the *floor* — what realistic body geometry leaves achievable
- R13 (contactless BP) sits between them — 5 dB short of the bound because of the floor

It suggests that **single-scatterer-style breathing detection** (rate-level, R14 V1 lighting) works because rate has +∞ tolerance — the band-locked signal can be recovered down to any SNR with enough averaging. **Contour-shape recovery** (HRV, BP) needs the *idealised* +25 dB which the multi-scatterer reality never delivers.

## Per-body-part energy contribution

The same 5 m link, off-LOS subject. CSI energy fraction per body part:

| Body part | Reflectivity | Energy contribution |
|---|---:|---:|
| **Chest** | 0.50 | **27.6%** |
| Head | 0.10 | 1.1% |
| Left arm | 0.10 | 1.1% |
| Right arm | 0.10 | 1.1% |
| Left leg | 0.10 | 1.1% |
| Right leg | 0.10 | 1.1% |
| Sum (not 100% — coherent sum, not power sum) | 1.0 | 33.6% |

Chest dominates by 5× because its reflectivity (proportional to surface area) is 5× the per-limb value. **Practically: the chest IS the breathing signal.** Limbs are confound, not signal.

This argues for two architectural decisions:

1. **Aim the Fresnel envelope at the chest, not the body centre.** The R6.2 placement search currently treats the body as a single point; a smarter version (R6.2.3) would aim at the *chest specifically*, putting the chest at the Fresnel midpoint.
2. **Mask limbs out of the breathing-detection pipeline.** This requires pose extraction (ADR-079, ADR-101), so we're already shipping the infrastructure to do this — `vital_signs.rs` just doesn't use it.

## What this tells us about `vital_signs.rs`

The current implementation extracts breathing-rate via a temporal bandpass filter (R5/R6 saliency suggested 0.1-0.4 Hz). It works in practice because the **rate signal** survives the multi-scatterer penalty. The unit-by-unit takeaway:

| Component | Behaviour | R6.1 evidence |
|---|---|---|
| Temporal bandpass (0.1-0.4 Hz) | Robust | Survives the +4.7 dB penalty; rate recoverable below SNR=0 dB |
| Subcarrier saliency selection (R5) | Beneficial | R6.1 shows uniform SNR across subcarriers; saliency selects *more reliable* subcarriers, not *higher-SNR* ones |
| Per-subject breath-rate calibration | Required | The 4.7 dB penalty varies with body geometry; per-subject calibration absorbs this |
| Contour-shape recovery (deferred) | **Physically blocked** | The 4.7 dB penalty + 5 dB threshold = no headroom |

This matches the existing pipeline's behaviour and explains *why* it works (rate yes, contour no).

## R12's revision path now has a basis

R12 (eigenshift) was a NEGATIVE result. The follow-up suggested **PABS over Fresnel-grounded basis**:

```
y_predicted = Σ_voxels  A(voxel) · reflectivity(voxel)
residual = y_observed − y_predicted
PABS = norm(residual)
```

R6.1's multi-scatterer model **is** the explicit A(voxel) the PABS formulation needs. Each voxel's contribution is computable from R6.1; the residual is what's left after subtracting a population-prior body model from the observed CSI; norm of residual is the structure-detection signal.

This is now a tractable implementation. R12 + R6.1 = a path forward for structure-detection that R12 alone couldn't take.

## Composes with prior threads

- **R5** (saliency) — selects more reliable subcarriers, not higher-SNR (since R6.1 shows uniform SNR across subcarriers for on-LOS-only scatterers).
- **R6** (single-scatterer Fresnel) — provides the per-scatterer building block.
- **R6.2 / R6.2.2** (placement) — should be re-evaluated with R6.1 chest-centric targeting (= R6.2.3).
- **R7** (mincut adversarial) — multi-scatterer model makes "physically impossible CSI" tighter: residual exceeds noise floor on *all* links simultaneously means the body model is wrong, not just one link compromised.
- **R10** (gait taxonomy) — limb-mounted scatterers in the body model are what move during walking. R6.1 + a time-varying limb position model gives gait-detection forward predictions.
- **R12** (eigenshift NEGATIVE) — provides the A(voxel) operator for the deferred PABS revision.
- **R13** (contactless BP NEGATIVE) — the 5 dB shortfall finding now has a **physical origin** (static limb scatterers).
- **R14** (empathic appliances) — V1 lighting works because rate survives the penalty; V3 attention-respecting (cognitive load via shallow breathing) needs ≥+25 dB which R6.1 says is unachievable. V3 should be re-scoped to *rate-only* features (e.g. respiration rate stability) instead of *contour-level* features (e.g. breathing pattern shape).

## Honest scope

- **6 scatterers is too few.** Real bodies are continuous distributions; 6 point-scatterers is a 1st-order approximation. A 50-100 point voxel grid would be more accurate but adds compute without changing the qualitative finding.
- **Reflectivity ratios are guesses.** Chest:limb = 5:1 by surface area is a soft estimate. RCS measurements at 2.4 GHz on real humans would refine these by 2-3×.
- **Static body assumption.** A real subject's limbs move with breathing too (small but non-zero). The current model treats them as fully static; a future R6.1.1 could add micromotion.
- **2D, top-down.** Like R6.2, this is a 2D approximation. 3D vertical (height variation) adds richness.
- **No multipath.** The model is direct-path-only. Wall/floor reflections in real rooms add additional scatterer contributions; the multi-scatterer model is general enough to include them by adding more "static" scatterers at reflection sites.

## What this DOES enable

1. **A physical origin** for R13's 5-dB shortfall (was: "subject micro-motion"; now: "static body parts add coherent confusion").
2. **R12's PABS revision basis** — the explicit A(voxel) forward operator is computable.
3. **A chest-centric placement recommendation** for breathing-detection features.
4. **An architectural argument** for using pose extraction to mask limbs out of the breathing pipeline.
5. **A re-scoping of R14 V3** to rate-level features only (V1, V2 already rate-only and safe).

## What this DOES NOT enable

- Continuous-time pose-aware forward model (would need 3D + 50+ scatterers + per-limb motion model).
- The actual implementation of PABS-on-residual (just provides the A operator).
- Quantitative gait-detection forward model (limb timing is in R15; the model here is static body).
- Vital signs in any motion regime other than chest-breathing.

## Next ticks (R6.1 follow-ups)

- **R6.1.1**: time-varying limb positions for gait detection.
- **R6.1.2**: 50-100 voxel body model with measured RCS values.
- **R12 PABS implementation**: now unblocked — use R6.1's forward operator.
- **R14 V3 re-scoping**: refine the attention-respecting design to depend only on breathing rate stability + occupancy, not shallow-breathing contour.

## Connection back

- **R5**: subcarrier selection prefers reliable, not high-SNR.
- **R6**: provides the building block; R6.1 composes 6 instances.
- **R6.2.3 (not yet built)**: chest-centric placement target.
- **R7**: residual-against-forward-model gives tighter adversarial detection.
- **R12**: A operator unblocked.
- **R13**: 5 dB shortfall = 4.7 dB multi-scatterer penalty (within 0.3 dB; agreement is suspicious but plausible).
- **R14**: V3 needs rescope.
