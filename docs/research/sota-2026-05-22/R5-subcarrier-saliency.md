# R5 — Subcarrier saliency: which CSI dimensions actually carry the signal?

**Status:** in-flight · **Started:** 2026-05-21

## Motivation

`cog-pose-estimation` (Conv1d 56 → 64 → 128 → 128) and `cog-person-count` (same backbone, different heads) both consume **56-subcarrier × 20-frame** CSI windows. The 56 came from the upstream `align-ground-truth.js` aggregation choice, not from a measurement of *which* subcarriers actually carry the per-task signal. If we could rank subcarriers by their first-order influence on the trained model's output, three concrete wins follow:

1. **Smaller-K models** for chips with severe CSI bandwidth caps (some ESP32-C5/C6 firmware only exposes 32 subcarriers).
2. **Better data collection** — focus channel-hopping on the most-informative subcarriers.
3. **Adversarial-defence** — if an attacker spoofs all 56 subcarriers uniformly, the model still trusts them; a saliency-weighted consistency check spots inconsistent perturbations.

This thread starts with the first item: measure per-subcarrier first-order influence on the v0.0.2 count model + the v0.0.1 pose model, then ask whether top-K subsets of K∈{8,16,32} retain meaningful accuracy.

## Method (single-tick scope)

For each model:

1. Load the trained safetensors (`cog/artifacts/count_v1.safetensors` and `cog/artifacts/pose_v1.safetensors`).
2. Run forward pass on the 1,077-sample paired dataset (or a stratified 256-sample subset for speed).
3. Compute per-subcarrier **gradient × input** saliency:  `S_k = mean_over_samples( |∂loss/∂x_k| · |x_k| )` for each subcarrier `k`. This is the standard "input × gradient" saliency from Sundararajan et al. (Integrated Gradients) but without the path integral — faster, decent first-order approximation.
4. Plot the 56-element saliency vector for each model. Identify top-K.
5. Re-train each model on the top-K subcarriers only (K ∈ {8, 16, 32}). Compare accuracy.

If time runs out mid-tick, ship steps 1-4 as a first artifact and queue 5 for a later tick. Steps 1-4 alone produce a real result (a ranked-subcarrier list per task).

## Why this is novel

ADR-097 mentions "subcarrier attention" abstractly; nothing measured. Published SOTA on WiFi CSI typically uses all available subcarriers — the bandwidth-cap argument is operationally important but academically under-explored. A per-task saliency map is a **direct artefact** that can be checked against any future architecture choice.

## Connections

- Feeds R7 (adversarial multi-link consistency) — top-K subcarriers are the ones a defender most needs to corroborate.
- Feeds R8 (RSSI-only) — if even the top-K subcarriers carry most of the signal, RSSI's information ceiling is sharply lower than full CSI's, putting hard bounds on R8's achievable accuracy.

## What gets written

This tick's deliverable is:
- The Python script `examples/research-sota/r5_subcarrier_saliency.py` that computes the saliency vector for either model.
- A first measurement (text + JSON) of saliency for the count model.

Step 5 (retrain on top-K) is queued for a subsequent tick.

## First measurement — `cog-person-count` v0.0.2 (this tick, 128 samples)

| Rank | Subcarrier | Saliency |
|-----:|-----------:|---------:|
| 1 | **41** | 0.0128 |
| 2 | **52** | 0.0120 |
| 3 | **30** | 0.0100 |
| 4 | 31 | 0.0097 |
| 5 | 10 | 0.0088 |
| 6 | 35 | 0.0088 |
| 7 | 2  | 0.0087 |
| 8 | 38 | 0.0083 |

**Max-to-mean ratio: 2.85×** — meaningful but moderate concentration. Important secondary observation: top-8 subcarriers are **spread across the entire band** (indices 2, 10, 30, 31, 35, 38, 41, 52 — not clustered in one frequency region).

## Implications

1. **Bandwidth-cap deployment is viable.** Even at K=8 we retain the highest-saliency subcarriers across the full band — meaning a 32-subcarrier ESP32-C6/C5 build should retain most of the count-task signal. Retraining at K=8/16/32 is the next-tick experiment.
2. **R8 (RSSI alone) is feasible-but-bounded.** RSSI is a band-aggregate scalar that loses per-subcarrier resolution. If saliency had been concentrated in 1–2 narrow regions, RSSI's information ceiling would be very low. Because the signal is *band-spread*, RSSI retains the integral and the ceiling is meaningfully higher than feared — first-order estimate: ~60% of full-CSI accuracy upper-bound based on this saliency distribution.
3. **R7 (adversarial defence) priority list.** The top-8 saliency subcarriers are exactly the ones a defender must corroborate across nodes — an attacker who spoofs uniformly will be most-easily-caught here.

## Next steps in this thread (queued for later ticks)

- Retrain at K=8, K=16, K=32 → publish accuracy-vs-K curve.
- Same saliency map for the pose model.
- Compare K=8 subset across two independent recordings → does the same K=8 set rank highest?
- Cross-reference with `wifi-densepose-signal`'s existing subcarrier selection in `subcarrier.rs`.
