# R12 — RF weather mapping: structural drift from passive WiFi (negative-ish result + revised plan)

**Status:** first experiment landed — **NEGATIVE-ish, with a clear next step** · **2026-05-22**

## The 10-year vision

Every WiFi access point in a building is, incidentally, a coherent radio source flooding the structure with energy. The walls, floors, furniture, and humans inside reflect that energy with characteristic multipath signatures. The persistent-room field model in `wifi-densepose-signal/src/ruvsense/field_model.rs` already captures the *spatial* eigenstructure of those reflections to subtract the room's baseline from occupancy detection.

The R12 vision generalises that to the *temporal* dimension: continuously track how the building's RF eigenstructure drifts across **days, weeks, months, years**. The hypothesis:

- **A new piece of furniture** changes the multipath profile in one specific way (additional reflector at a specific location).
- **Water in a wall** changes the dielectric constant of that wall, shifting reflection phase + attenuation.
- **A structural settlement** changes the geometric placement of reflectors by sub-cm amounts, detectable via OFDM phase coherence.
- **A missing ceiling tile** changes Fresnel-zone coupling between rooms.
- **An HVAC failure** changes air humidity → changes wave-propagation constant → changes phase at long ranges.

Pre-2026 SOTA mostly uses CSI for activity recognition. The shift to *structural integrity monitoring from passive ambient RF* is open territory.

## First experiment (this tick)

`examples/research-sota/r12_rf_weather_eigenshift.py` tests the simplest possible algorithm: SVD on the per-frame CSI matrix, top-K singular values, cosine distance between spectra over time.

Setup:
- Take 1,077 CSI windows from the existing paired data.
- Split first-half (10,760 frames) = "before", last-half (10,780 frames) = "after".
- Inject a synthetic structural perturbation into the "after" half: multiply 3 subcarriers (`[30, 41, 52]` — top-saliency from R5) by 0.85 to simulate a new reflective surface attenuating those frequencies by ~1.4 dB.
- Top-10 singular values per half. Cosine distance between spectra.

## Result

| | Cosine distance from BEFORE |
|---|---|
| AFTER (no perturbation, control) | 0.00035 |
| AFTER (with 3-subcarrier perturbation) | **0.00024** |
| Signal / natural-drift ratio | **0.69×** |

**Verdict: WEAK.** The synthetic structural perturbation produces a *smaller* spectral distance than the natural temporal drift from operator movement in the same recording. The top-10 singular-value spectrum is **not sensitive enough** to detect ~15% attenuation on 3 of 56 subcarriers when the room's occupant is moving.

## Why this fails — and how to fix it

The top-K singular-value spectrum captures the **dominant energy** in the channel state. A 15% perturbation on 3 of 56 subcarriers shifts the matrix by ≤(3/56) × 15% ≈ 0.8% of total energy. That's well below the natural temporal variance from a moving operator.

Three concrete revisions for next attempts:

1. **Use the FULL eigenvector basis, not just the spectrum.** The cosine distance on top-K singular *values* is scale-aware but direction-blind. Comparing the top-K *eigenvectors* (singular vectors) via subspace angles ("principal angles between subspaces") would catch the structural shift even when the energy distribution stays similar.

2. **Detect specific subcarriers via residual analysis.** Instead of comparing whole spectra, project each window onto the empty-room subspace and look for **consistent per-subcarrier residuals** — these would localise the perturbation. The 3 perturbed subcarriers would show a persistent attenuation bias that natural drift wouldn't reproduce.

3. **Multi-day baseline.** This experiment uses a single 30-min recording. The "natural temporal drift" is dominated by operator movement, not by structural change. The real RF-weather problem has the OPPOSITE noise structure: structural changes happen over hours-to-days, occupancy noise averages out over minutes-to-hours. Averaging the eigenspectrum over a 24-hour window before comparing should knock down the operator-noise floor by 50-100×.

## What still holds

The 10-year vision isn't refuted — the algorithm choice was wrong. Specifically:

- The **physics is real**: dielectric changes in walls cause measurable CSI shifts (well-documented in 2020-era CSI building-monitoring literature).
- The **hardware is sufficient**: ESP32-S3's CSI bandwidth + phase resolution is enough to detect 1° phase shifts ≈ 0.5 mm displacement at 5 GHz.
- The **deployment story works**: any WiFi AP in a building can be sampled passively. No physical installation cost.
- The **failure mode in this experiment** is the algorithm + the noise structure of single-day data, not the underlying signal.

## What this DOES prove

- The simple "SVD spectrum cosine distance" approach **does not work** in single-day data. Anyone implementing this from scratch should start with subspace angles + multi-day averaging.
- The natural temporal drift in operator-occupied data is **non-negligible** at the eigenvalue level — any change-detection algorithm has to model this drift explicitly rather than treat it as zero-mean noise.

## What's next on this thread

- Implement **principal angles between subspaces** (PABS) as the comparison metric instead of cosine on singular values. PABS catches subspace rotations that singular-value cosines miss.
- Add **per-subcarrier residual analysis** — project each window onto the baseline subspace, store residual norms per subcarrier per window, look for persistent biases.
- Need **multi-day data** at minimum. Even better: 7-day data with a deliberate structural change at day 4 (e.g. move a chair 1 m). Currently no such dataset exists in the repo.

## Connection back

- R5 (band-spread saliency): the perturbation chose top-saliency subcarriers, but it still wasn't detected — suggests R5's saliency is **task-specific** (count-task saliency ≠ structure-detection saliency). Useful counter-data point.
- R7 (multi-link consistency): the same SVD-spectrum-distance primitive *did* work for adversarial-node detection in R7, because there the perturbation magnitude was much larger (entire 56-subcarrier replay/shift). Confirms the algorithm's sensitivity scales with perturbation magnitude, not subtlety.
- R8 (RSSI-only): RSSI is the trace of the CSI covariance matrix. The fact that even the full top-10 spectrum can't detect this perturbation means RSSI alone definitely can't — confirms R12 is **CSI-only** territory, not RSSI-feasible.

## 10-year vertical applications (preserved despite negative result)

The vision is right; the algorithm needs work. Verticals to chase once PABS + multi-day data exist:

- **Building structural monitoring** for insurance companies — early water-damage detection from RF signature shift.
- **Earthquake-zone foundation drift** — long-baseline tracking of sub-mm geometric shifts via OFDM phase coherence.
- **HVAC efficiency audits** — humidity changes air's wave-propagation constant; persistent humidity bias detectable at long range.
- **Museum / archive climate stability** — same physics, lower allowable drift.
- **Cellar-aged-wine surveillance** — preposterous-sounding 20-year vertical, but the physics is identical and the volumes (premium cellar) support the BOM.
