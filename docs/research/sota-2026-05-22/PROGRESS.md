# SOTA Research Loop — 2026-05-22

Started: 2026-05-21 ~20:00 ET. **Auto-stops: 2026-05-22 08:00 ET.** Cron `d6e5c473` (`*/10 * * * *`).

## Mandate

Push WiFi-CSI sensing past 2026 published SOTA in three axes:

1. **Spatial intelligence** — multi-static fusion, room-scale awareness, occupancy beyond counting
2. **RF feature engineering** — phase, ToA, subcarrier dynamics, Fresnel zones
3. **RSSI alone** — what's achievable without CSI capture (massive deployment story — every WiFi chip emits RSSI)

Plus practical verticals (exotic & beyond) on a 10–20 year horizon.

Output goes to `docs/research/sota-2026-05-22/` (research notes, benchmarks, negative results) + `examples/research-sota/` (runnable code).

## Working principle

Each loop tick picks ONE **unfinished thread** from below and produces ONE concrete artifact:
- a research note (Markdown with sources + measured numbers if possible)
- an experiment / micro-benchmark
- a working example under `examples/research-sota/`
- a negative result ("X doesn't work because Y, here's the data")
- an ADR if the thread is mature enough to land

Stay 8 minutes / tick. Commit + PR + auto-merge per piece. Future-tick re-entry is via this PROGRESS.md.

## Research vectors

### Spatial Intelligence

- [ ] **R1. Multi-static Time-of-Arrival (ToA) from OFDM phase coherence.** Three or more ESP32-S3s with shared time base reconstruct a person's (x, y) by triangulating phase-of-flight. 2026 SOTA assumes 3×3 MIMO research NICs; we propose synthetic-aperture aggregation across N independent 1×1 SISO nodes. Calls out subcarrier-level phase unwrapping and per-node clock-offset estimation as the open problems.
- [ ] **R2. Persistent room field model — eigenstructure perturbation.** Already in `wifi-densepose-signal/src/ruvsense/field_model.rs` (SVD on empty-room CSI). Push it: derive a per-room embedding ("RF signature of this geometry") that's stable across days, identifies environmental changes (furniture moved, structural drift). Vertical: building-integrity monitoring.
- [ ] **R3. Cross-room re-identification via gait CSI signatures.** Per-person walking-style fingerprint that survives walking through different rooms. Different from `AETHER` (in-room re-ID) — this is *inter*-room continuity.
- [ ] **R4. Federated learning of room models.** Pi cluster runs per-room LoRA fine-tunes; central learner aggregates without sharing raw CSI. Privacy-preserving spatial intelligence.

### RF Feature Engineering

- [ ] **R5. Subcarrier attention over time → "RF saliency map".** Visualize which subcarriers carry the most information per task. ADR-097 hints at this; nothing in repo computes it. Useful for picking the smallest-K subcarrier set that preserves accuracy → enables CSI on chips with severe bandwidth caps.
- [ ] **R6. Fresnel-zone forward model for through-wall sensing.** Code in `wifi-densepose-signal/src/ruvsense/tomography.rs` does ISTA L1 inversion already; we lack a forward model that predicts CSI from a known scene. Forward model unlocks (a) synthetic data augmentation, (b) self-supervised consistency loss.
- [x] **R7. Stoer-Wagner adversarial-node detection.** DONE — 3/3 detection rate (replay/shift/noise). See `R7-multilink-consistency.md`. Cross-links: R5 top-8 saliency subcarriers are priority targets for partial-spectrum attackers; fills `cog-person-count::fusion::fuse_with_mincut_clip()` stub (ADR-103 v0.2.0). Next tick: Stackelberg-game adaptive attacker.

### RSSI Alone (no CSI)

- [x] **R8. RSSI-only person count.** DONE — 59.1% = 94.82% of full-CSI (62.3%). 656 params, 5 KB, 0.72 s CPU. See `R8-rssi-only-count.md`. Cross-links: R5 band-spread saliency explains the retained accuracy; R9 extends same stream to localisation; ADR-104 MCP server should grow `ruview_count_infer --rssi` mode for non-CSI chips. Next: 3-class ceiling, multi-room replication.
- [ ] **R9. RSSI fingerprint topology — graph neural network on WiFi-scan beacons.** Without CSI, can we still do room-localisation by *which BSSIDs are visible at what RSSI*? Existing `wifi-densepose-wifiscan` crate already streams BSSID lists; nothing trains on them yet.

### Exotic & Future (10–20 year)

- [ ] **R10. Through-foliage wildlife sensing.** Same physics as through-wall, but at much lower SNR. Gait recognition on a per-species basis. Practical: non-invasive population monitoring without cameras.
- [ ] **R11. Through-bulkhead maritime crew tracking.** Steel attenuates but doesn't eliminate WiFi multipath. Limited range, requires per-vessel calibration.
- [ ] **R12. RF "weather" mapping.** Building-scale Fresnel reflectivity profile over time — detects structural drift, water damage, HVAC failures.
- [ ] **R13. Contactless blood pressure from sub-mm chest displacement.** Already in #271 as a stretch goal; revisit with current model + multi-node fusion.
- [ ] **R14. Empathic appliances.** Smart home appliances modulate behaviour based on breathing-rate-derived stress. Long-horizon — needs both the sensing accuracy *and* an ethical framework.
- [ ] **R15. RF biometric across rooms.** Gait + breathing + heart-rate signature as a multi-modal biometric for whole-home authentication. Replaces fingerprint/face on the home-network layer.

## Done

### 2026-05-21 kickoff tick
- ✅ **R5 in-flight** — `examples/research-sota/r5_subcarrier_saliency.py` runs; first measurement on `cog-person-count` v0.0.2 ships: top-8 subcarriers spread across the band, max/mean ratio 2.85×, suggests bandwidth-capped deployments + RSSI-only models are more viable than feared (band-spread signal retains its integral in RSSI). See `R5-subcarrier-saliency.md` §"First measurement" + §"Implications".

### 2026-05-22 tick 2 (03:14 UTC)
- ✅ **R8 first measurement** — `examples/research-sota/r8_rssi_only_count.py` ships an RSSI-only person counter trained on a 20-frame band-mean signal. **Result: 59.1% accuracy = 94.82% of the full-CSI v0.0.2 baseline (62.3%).** Tiny model: 656 params (~5 KB), 56× smaller input, trains in 0.72 s on CPU. **Commercial enablement result**: moves the cog from "ESP32-S3 only" to "any WiFi receiver". Class accuracy balanced (59.5 / 58.6 vs v0.0.2's skewed 86.2 / 34.3). Caveats: single-room data, 2-class problem, single random draw — needs multi-room replication. See `R8-rssi-only-count.md` for full method + interpretation + 3 follow-up experiments queued. Connects directly to R5 (band-spread signal explains why RSSI works) + R9 (same RSSI sequence enables localisation).

### 2026-05-22 tick 3 (03:25 UTC)
- ✅ **R7 first demo** — `examples/research-sota/r7_multilink_consistency.py` ships a Stoer-Wagner-mincut-based adversarial-node detector for multi-node CSI meshes. **Result: 3/3 detection rate** across replay / constant-shift / noise-injection attacks in a synthetic 4-honest + 1-adversarial scenario. Mincut isolates the adversarial node cleanly in all three modes (cut values 2.56–3.57, partition_B = `{4}` consistently). Pure-NumPy demo, no framework deps. **Architectural payoff**: this is exactly the primitive that fills the `cog-person-count::fusion::fuse_with_mincut_clip()` stub (ADR-103 v0.2.0). Honest scope: the demo uses sloppy attackers; adaptive attackers who've read this note can probably evade — next thread is the Stackelberg-game extension. See `R7-multilink-consistency.md`.

## Negative results

(populated when we discover something doesn't work — these are explicit, not failures)

## Index by date

- 2026-05-21 — kickoff (this file)
- 2026-05-22 — tick 2: R8 RSSI-only count (59.1% / 94.82% retained)
- 2026-05-22 — tick 3: R7 multi-link consistency detection (3/3 attack modes detected by Stoer-Wagner mincut)
