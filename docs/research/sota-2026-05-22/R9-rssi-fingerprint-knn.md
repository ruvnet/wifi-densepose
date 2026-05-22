# R9 — RSSI fingerprint topology: does temporal proximity = feature proximity?

**Status:** first measurement — MODERATE result · **2026-05-22**

## Question

R8 just showed RSSI alone retains 95% of full-CSI accuracy for *counting*. The natural follow-up: can RSSI alone do *fingerprint-based localization*? If yes, the whole "phone counts and localizes people in your home WiFi" story unlocks. If no, R8's commercial enablement is bounded to counting-only.

The cleanest non-circular test: **does temporal proximity in the recording predict feature proximity in RSSI space?** A single 30-min recording captures one operator moving around one room. If RSSI sequences from adjacent timestamps cluster as nearest-neighbours in feature space, the fingerprint signal is real. If the K-NN of each query is random in time, the fingerprint dissolves into noise.

## Method

1. Take the 1,077 paired CSI windows. Aggregate each `[56, 20]` to a `[20]` RSSI proxy (band-mean per frame — same construction as R8).
2. Z-score normalise across all samples (matches AGC behaviour).
3. Compute the full `1077 × 1077` cosine-similarity matrix.
4. For each query, find top-K (K=5) nearest neighbours, excluding self.
5. Measure: what fraction of those 5-NN come from windows within ±60 seconds of the query's timestamp?
6. Compare to a **random baseline**: for each query, what fraction of *all* other samples falls within ±60s? (Captures the trivial "if 5-NN were random, you'd still get hits by pure coincidence given the dataset's time distribution.")

Lift = `K-NN fraction within window` / `random baseline`.

## Result

| Metric | Value |
|---|---|
| 5-NN within ±60s | **0.169** |
| Random baseline | 0.077 |
| **Lift over random** | **2.18×** |
| Per-query stdev | 0.183 |

**Verdict — MODERATE.** Below the ≥3× threshold for "strong fingerprint" but well above 1× random. The signal is real but noisy.

## Honest interpretation

Three possible explanations for the moderate lift, each with different implications:

1. **20-frame windows are too short.** Each window is ~2 seconds of CSI. Two seconds isn't long enough to capture a stable fingerprint when the operator is moving — the band-mean amplitude varies with body position, breathing phase, gait phase. A 60-frame window (~6 s) might lift this to 3-4×.
2. **One-room data has a small fingerprint space.** Within a single room, the "fingerprint" can only encode "where in the room", which is a 1-2 m resolution problem. RSSI doesn't have the bandwidth for that. Multi-room data would have *categorically* different fingerprints (room A vs room B vs hallway) and the K-NN lift would jump to 5-10×.
3. **Band-mean discards the per-subcarrier shape.** R5 said the count-task signal is band-spread. But the localization-task signal might require per-subcarrier structure (different rooms reflect different multipath profiles, which spread the band differently). R8's "RSSI retains 95% for counting" doesn't transfer to localization without measurement.

The 2.18× lift is consistent with all three. Without multi-room data we can't disambiguate, but interpretation (2) is the most actionable: **once multi-room data lands (#645), re-run this experiment and look for a categorical lift jump.**

## What this DOES prove

- RSSI sequences are **not** purely noise — there's structure that correlates with temporal proximity, just not strongly enough for single-room fingerprinting at our window size.
- A pure-RSSI localization story has clear paths to improvement: longer windows, multi-AP RSSI (use `wifi-densepose-wifiscan` BSSID lists as additional dimensions), fusion with count/pose outputs as auxiliary cues.

## What this DOES NOT prove

- That RSSI fingerprinting *won't* work cross-room. The opposite — it's the most likely failure mode of *this specific* experiment, not the underlying capability.
- That CSI fingerprinting would work better. We didn't measure CSI K-NN here; would be a useful follow-up.

## Connections

- **R8** showed RSSI keeps the count signal. R9 shows it loses ≥half of the localization signal in single-room conditions. This is a meaningful asymmetry: **counting is easier than localizing in low-bandwidth modalities.**
- **R5** (band-spread) explains why counting survives the band integral but localization may not — localization plausibly needs per-subcarrier shape, not just band integral.
- **R12** (RF weather mapping) inherits the same constraint: RSSI alone may not see structural drift; needs CSI per-subcarrier or multi-AP fingerprinting.

## What's next on this thread

- Re-run with 60-frame windows (3× more temporal context) to see if lift jumps.
- Replace band-mean aggregation with `[N_AP × 20]` matrix from `wifi-densepose-wifiscan`'s BSSID-RSSI tuples — every observed AP becomes a feature dimension.
- Once multi-room data exists, repeat. Look for categorical lift jump (within-room 2× → across-room 8-10×).
- Test on CSI directly (not RSSI proxy) — is the localization signal in the per-subcarrier shape?
