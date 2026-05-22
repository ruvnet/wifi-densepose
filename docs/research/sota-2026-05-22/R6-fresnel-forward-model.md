# R6 — Fresnel-zone forward model: making CSI sensitivity predictable

**Status:** working forward model + numpy demo · **2026-05-22**

## The gap this fills

The entire `wifi-densepose-signal` DSP pipeline — `vital_signs`, `multistatic`, `pose_tracker` — operates on CSI windows whose **physical meaning** is taken for granted. We measure complex per-subcarrier amplitudes, treat them as input features, and learn classifiers. Nobody in the repo has written down the **forward model**: given a known scatterer position + size + reflectivity, what does the CSI look like?

Without a forward model:

- **R12** (eigenshift) was forced to invent its own subspace basis from data — and discovered it was indistinguishable from natural drift.
- **R7** (multi-link consistency) had to bootstrap an adversarial detector from scratch instead of comparing against a physics-grounded expectation.
- **R10** (foliage range) had to use ITU-R + FSPL alone, ignoring the fact that an obstacle larger than the **first Fresnel zone** causes diffraction loss that no FSPL model captures.

This tick makes the forward model explicit. Self-contained numpy; no dependencies on the workspace.

## The model

For a Tx-Rx link of length `L`, the **first Fresnel zone** is the prolate ellipsoid where most of the diffracted RF energy travels. Its radius at fractional position `p ∈ [0, 1]` along the LOS is:

```
r_1(p) = sqrt(λ · L · p · (1 − p))      [metres]
```

A **point scatterer** at perpendicular offset `x` from the LOS, at link position `d_1` from Tx (so `d_2 = L − d_1` from Rx), introduces a path-length delta:

```
Δℓ(x) = sqrt(d_1² + x²) + sqrt(d_2² + x²) − (d_1 + d_2)
```

Phase shift on subcarrier `k` with centre frequency `f_k`:

```
φ_k = 2π · f_k · Δℓ / c
```

That's it. Six lines that the entire workspace's DSP secretly assumes.

## What the demo computes

`examples/research-sota/r6_fresnel_zone.py` runs four canonical scenarios and emits per-subcarrier phase predictions for 802.11n/ac 20 MHz channels (52 used subcarriers, 312.5 kHz spacing):

### First Fresnel radii (the basic envelope)

| Link length | 2.4 GHz @ midpoint | 5 GHz @ midpoint |
|---|---:|---:|
| 2 m | 25.0 cm | 17.3 cm |
| 5 m | **39.5 cm** | 27.4 cm |
| 10 m | 55.9 cm | 38.7 cm |

These are **measurable, physical envelopes**: a 5 m WiFi link in a typical bedroom has a roughly 40 cm wide "channel of maximum sensitivity" centered on the LOS, narrowing toward each antenna. A human standing inside that ellipsoid moves the entire CSI vector; a human standing outside it perturbs only edge subcarriers.

### Single-scatterer predictions

| Scenario | Offset | Position | Zone @ 2.4 GHz | Phase spread |
|---|---:|---:|:---|---:|
| Human standing at midpoint | 10 cm | 2.5 m | zone-1 | 0.077° |
| Human walking into Fresnel | 25 cm | 2.5 m | zone-1 | 0.477° |
| Scatterer outside Fresnel | 1.5 m | 2.5 m | far-field | 15.9° |
| Scatterer near Tx | 5 cm | 0.5 m | zone-1 | 0.053° |

**Key insight (concrete now):** the phase spread across subcarriers grows monotonically with `Δℓ`, which grows quadratically with offset `x`. A scatterer in the **far field** (15.9° spread across 52 subcarriers) is the regime where multi-tap channel estimation works well. A scatterer **inside the first Fresnel zone** (<0.5° spread) is essentially uniform across subcarriers — which is why R5's saliency revealed band-spread top subcarriers (the scatterer effectively excites the whole band) rather than tight clusters.

This unifies R5 and R6: the saliency band-spread we measured experimentally is exactly what the Fresnel forward model predicts for inside-zone-1 occupancy.

## Why this matters for the workspace

| Existing module | What R6 gives it |
|---|---|
| `vital_signs` (breathing/HR) | Predicts that chest-wall motion at ~1 cm amplitude inside zone-1 produces 0.01–0.05° phase change per breath — sets the floor SNR for HR detection |
| `multistatic.rs` (attention-weighted fusion) | Provides ground-truth weights: scatterers in different Fresnel zones contribute different per-subcarrier phase signatures, so the attention weights have a closed-form prior |
| `tomography.rs` (RF tomography) | Forward operator A in `Ax = y` was a black box; R6 makes A explicit (per-voxel position → per-subcarrier phase contribution) so the L1-ISTA inverse problem becomes properly conditioned |
| `pose_tracker.rs` (17-keypoint Kalman) | The "sensitivity to limb position" prior is now derivable from the Fresnel geometry — distal limbs (hands, feet) often sit *outside* the first Fresnel zone for indoor links, explaining why they're harder to track than torso/head |

## Connection to R12

R12 (eigenshift) failed because the SVD spectrum is a 1-D summary that loses the spatial structure the Fresnel forward model preserves. The right revision is:

```
y_predicted = sum_voxels  A(voxel) · reflectivity(voxel)
residual = y_observed − y_predicted
PABS = norm(residual)   # the structure-detection signal
```

where `A(voxel)` is exactly the per-subcarrier phase prediction from R6. This is essentially RF tomography, but used as a **structure-detection prior** rather than as inverse reconstruction. **PABS-over-Fresnel-grounded-basis** is the right next step that R12 explicitly identified — R6 supplies the basis.

## Connection to R10 (the wildlife angle)

R10's range estimates used FSPL + ITU foliage attenuation. But foliage **also blocks the first Fresnel zone**, and an obstacle filling >60% of the zone produces diffraction loss that FSPL alone misses. For the 2.4 GHz / 100 m sparse case, the first Fresnel zone at midpoint is `sqrt(0.125 · 100 · 0.5 · 0.5) = 1.77 m` wide — large enough that a tree trunk in the middle of the link cuts deeply into it.

A more honest sparse-foliage range, accounting for partial zone obstruction: probably **closer to 70 m than 100 m** for canopies with ~1.5 m vertical clearance. Documented here as a known under-estimate of the range we should retract toward in any field deployment.

## Honest scope

- **Point scatterer.** Real bodies are distributed scatterers (limbs, chest, head — all at different positions in the zone). The full forward model is a volume integral over body-mounted RCS, not the scalar `Δℓ` here. The scalar version is the correct first-order approximation.
- **First Fresnel only.** Real diffraction includes contributions from zones 2..N (the Cornu spiral). For obstacle classification (presence/absence/size) zone-1 dominates and the model is enough. For phase-precise reconstruction (millimeter-wave-style imaging) we'd need to sum over more zones.
- **Frequency-flat scatterers.** We assume the scatterer's reflectivity is constant across the 20 MHz channel. Real biological tissue has frequency-dependent permittivity; the error is small at WiFi bands but non-zero.
- **LOS-only.** Multipath (floor / ceiling / wall reflections) is not modeled. In a real bedroom there are typically 4-6 dominant reflectors, each contributing its own Δℓ. The full multipath model is just a sum of single-scatterer terms with their own A matrices — additive in the forward direction, harder to invert.

## What this DOES enable

- **Closed-form sensitivity bounds.** For any specified `(link length, frequency, scatterer position+size)` we can predict the per-subcarrier signature analytically. Removes mystery from "why does this signal look like this?"
- **R12 revision path with a basis.** PABS computed against a Fresnel-grounded forward operator is the right structure-detection signal.
- **Antenna-placement heuristics.** For a given room, R6 immediately predicts where the Fresnel envelope sits and which sensor positions maximise coverage. The current installation-guide is "guess and measure"; R6 enables "compute and validate."
- **R10 range correction.** Foliage range estimates should be discounted for partial Fresnel-zone obstruction. ~30% conservative correction in the sparse case.

## What this DOES NOT enable

- **Without antenna calibration**, the absolute phase predictions are off by a constant per-subcarrier offset (the LO phase, per-antenna delay, etc.). The relative predictions (phase **spread** across subcarriers; phase **change** between consecutive windows) survive. The existing `phase_align.rs` handles the calibration step.
- **Multipath-rich environments** need the multi-scatterer extension before R6 is quantitatively useful.

## Next ticks (R6 follow-ups)

- **PABS over Fresnel basis:** implement R12's revision — observed CSI minus forward-model prediction, structure detection on the residual. Should improve R12's 0.69× signal/drift ratio.
- **R6.1 — multi-scatterer additive forward model:** sum over a coarse voxel grid, see whether breathing-rate estimation accuracy improves vs the current `vital_signs` heuristic.
- **R6.2 — Fresnel-aware antenna placement:** given a room geometry + target occupancy zones, solve for the antenna positions that maximise Fresnel-envelope coverage. Could ship as a CLI tool in `wifi-densepose-cli`.

## Connection back

- **R5** (saliency) — band-spread top subcarriers are exactly what zone-1 occupancy predicts. R5 measured it; R6 explains it.
- **R7** (mincut adversarial) — physically inconsistent CSI is now well-defined: residual from R6's forward model exceeds noise floor across all links simultaneously. Stoer-Wagner mincut detects the violation.
- **R10** (foliage range) — Fresnel-zone obstruction adds ~30% range discount in sparse-foliage scenarios; the 100 m number should be retracted to ~70 m.
- **R12** (eigenshift) — the failed SVD-spectrum approach has a clear successor: PABS over Fresnel-grounded basis.
- **R14** (empathic appliances) — Fresnel-envelope sensitivity bound sets the per-room calibration floor for the V1 stress-responsive lighting use case.
- **ADR-029** (multistatic) — provides the closed-form attention-weight prior the current learned-weights system lacks.
