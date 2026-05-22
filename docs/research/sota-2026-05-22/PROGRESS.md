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
- [ ] **R7. Quantum-inspired Stoer-Wagner sampling for adversarial robustness.** Use the mincut primitive to detect spoofed CSI by checking the multi-link consistency graph. Lands in `cognitum-rvcsi` if it works.

### RSSI Alone (no CSI)

- [ ] **R8. RSSI-only presence + vitals.** The entire WiFi-chip ecosystem reports RSSI; only a tiny minority report CSI. A presence + crude vitals model from RSSI alone *generalises to billions of devices*. Hard problem (very low information rate) but enormous downstream value. Start with literature survey + first model experiment.
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

## Negative results

(populated when we discover something doesn't work — these are explicit, not failures)

## Index by date

- 2026-05-21 — kickoff (this file)
