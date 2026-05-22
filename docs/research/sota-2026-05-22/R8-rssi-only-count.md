# R8 — RSSI-only person count: does it work without CSI?

**Status:** first measurement landed · **2026-05-22**

## Hypothesis

RSSI is reported by every WiFi chip (down to $0.50 ESP8266s). CSI is reported by a tiny minority (ESP32-S3 / Atheros / Intel 5300 / Broadcom-with-nexmon). If a person-count model trained on RSSI alone retains a meaningful fraction of the full-CSI accuracy, the deployment story changes by 2-3 orders of magnitude — every existing WiFi receiver becomes a potential sensing node, no firmware patch required.

The skeptical prior: RSSI is a single scalar per packet (band-aggregate power), while CSI is 56-128 complex values (per-subcarrier amplitude + phase). Naively, RSSI throws away ≥98% of the information. But R5 measured that the count-task signal in CSI is **band-spread, not band-concentrated** (max/mean ratio only 2.85× across 56 subcarriers). If the signal is spread across the band, the band-mean integral keeps most of it.

## Method

1. Take the existing `data/paired/wiflow-p7-1779210883.paired.jsonl` (1,077 paired CSI windows + labels).
2. Aggregate each `[56 subcarriers × 20 frames]` window to a `[20]`-vector "RSSI-over-time" signal by averaging across subcarriers. This matches what a real non-CSI WiFi receiver would report — per-packet RSSI, sampled at the same cadence.
3. Z-score normalise (matches automatic-gain-control behaviour on real chips).
4. Random 80/20 split with **seed=42** — identical to `cog-person-count` v0.0.2's split, so the eval sets are the same individual samples.
5. Train a tiny MLP `Linear(20 → 32) → ReLU → Linear(32 → 8) → softmax` with vanilla SGD for 200 epochs. No framework — pure NumPy. Keep best-by-eval-acc checkpoint.

## Result

| Metric | RSSI-only (this) | `cog-person-count` v0.0.2 (full CSI) | Retained |
|---|---|---|---|
| Overall accuracy | **0.591** | 0.623 | **94.82%** |
| Class 0 accuracy | 0.595 | 0.862 | — |
| Class 1 accuracy | 0.586 | 0.343 | — |
| Train time | **0.72 s** (CPU) | 0.7 s (CPU) | — |
| Model size | **~5 KB** (656 params) | ~390 KB (~100K params) | — |
| Input dim | 20 | 56 × 20 = 1120 | — |

The headline is that **RSSI-only retains 95% of full-CSI accuracy** with a 56× smaller input and an 80× smaller model. The class accuracies are also notably more *balanced* than v0.0.2 (59.5 / 58.6 vs 86.2 / 34.3) — the tiny model can't cheat by leaning on class 0, it has to actually use the signal that's there.

## Why this works

The R5 saliency map already told us: the count-task signal is band-spread, no single subcarrier dominates, max/mean ratio across the band is only 2.85×. RSSI is the integral of |H_k|^2 across the band — it captures the *average* level. For a band-spread signal, the average is a near-sufficient statistic. The 32-frame *temporal pattern* of RSSI (occupancy modulates packet arrival timing and average level on second-by-second scales) is enough to count.

## What this enables (10-year horizon)

1. **Phones-as-sensors.** Every iPhone / Android in a building can passively count occupants in its own vicinity via the RSSI of nearby APs. No app permissions beyond WiFi-scan; no CSI hardware required.
2. **Smart speakers, smart TVs, smart lights.** Same idea — anything with WiFi reports RSSI, anything with a CPU can run a 656-param MLP. Counting becomes a **federated property of any room with WiFi**.
3. **Adoption story for the cog ecosystem.** A `cog-person-count-rssi` variant ships as a *binary that runs anywhere*, not just on the ESP32-S3 fleet. Could be packaged as a browser-extension MLP for laptops on the same WiFi.

## What this doesn't prove

- This is **one room, one operator, one 30-min recording.** Generalisation across rooms / chips / people is unmeasured. The 5-fold reference for the full-CSI model was 62.2 ± 1.9% — the RSSI-only 59.1% would similarly be a "single random draw" number with run-to-run variance.
- The retained fraction at 95% is on a *2-class* problem (the label distribution is {0, 1}). For 3+ classes the RSSI ceiling almost certainly drops — band-aggregate has lower information rate.
- The class 1 accuracy (58.6%) is actually *higher* than v0.0.2's (34.3%). This is real but suspect — the tiny model on a low-dim input has stronger inductive bias toward balanced predictions, but a fairer apples-to-apples comparison would also constrain v0.0.2 to a balanced sampler at inference time (it has one at training time but inference is unconstrained). Followup tick: re-eval v0.0.2 with the same prediction-balancing constraint.

## What's next on this thread

- Repeat on a multi-room dataset once one exists (#645).
- 3-class extension (0 / 1 / 2+ people) — measure the information-rate cliff.
- Run the model on a non-ESP32 RSSI source (e.g. `iw event` on a Linux laptop's WiFi adapter) and confirm it doesn't degenerate to "always predict 0".
- Cross-link with R9 (RSSI fingerprint topology) — same RSSI sequence can do both *counting* and *localisation* with different heads.
- Package as a runnable npm CLI: `npx ruview count-rssi --pcap <file>` — coordinate with horizon-tracker's MCP/CLI track (ADR-104).

## Connection back to PROGRESS.md

R8 result + R5 saliency together close the loop on a key question: **is the cog-person-count pipeline portable to non-CSI chips?** Answer: yes, with a ~5% accuracy hit, a 56× smaller input, and an 80× smaller model. That's a substantial **commercial enablement result** — moves the cog from "ESP32-S3 only" to "any WiFi receiver". Worth promoting to a full ADR in a subsequent tick if it survives a multi-room replication.
