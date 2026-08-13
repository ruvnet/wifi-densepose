# ADR-281: New modality surfaces — BLE Channel Sounding, delay-Doppler-native tensors, P3162 import, and factorized pose

| Field | Value |
|-------|-------|
| **Status** | Accepted — **implemented** (`adapters.rs` BLE CS + ranging evidence, `tensor.rs::delay_doppler_map`, `frame.rs` P3162 import profile, `heads.rs` factorized pose; 8 new test suites) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273; extends ADR-274 |
| **Relates to** | ADR-279 (`FieldAxis` native axes), ADR-152 (geometry conditioning intake), ADR-021/263 (radar hardware) |

## 0. PROOF discipline

Grades per ADR-273 §0. Bluetooth SIG cm-level claims vs the ~20–50 cm practical review, OTFS ISAC field trials, IEEE P3162, PerceptAlign's >60 % cross-domain error reduction, and RePos's 10–21 % MPJPE gains are EXTERNAL-UNVERIFIED design inputs. Our numbers below are MEASURED-CODE / MEASURED-SYNTHETIC.

## 1. BLE Channel Sounding (§2) — likely the fastest path to consumer-scale spatial anchoring

`BleCsFrame` carries per-frequency-step round-trip tone phases plus optional RTT. Two rules:

- **Phase-based ranging and RTT are separate evidence sources.** `ble_cs_range` computes both — `d_phase = |dθ/df|·c/4π` from the unwrapped phase-vs-frequency slope, `d_rtt = rtt·c/2` — and *cross-validates* instead of averaging. Agreement raises confidence; divergence beyond 0.5 m yields `RangingAnomaly::Divergent` (multipath bias, relay attack, timing fault, or calibration problem) with confidence capped ≤ 0.2. Measured: exact recovery at 1.5/5/12 m (< 1 µm error on clean synthetic phases, 40 steps × 1 MHz); a relay-style RTT inflation to ~51 m against a 5 m phase estimate is flagged, not blended (`ble_cs_flags_relay_style_divergence_instead_of_averaging`).
- The tensor view (`BleCsAdapter`, `nrf54-cs` in the registry) **never detrends phase** — the ranging ramp *is* the measurement; the preserved ramp is asserted in test.

Single-source evidence (no RTT) is capped at confidence 0.5 — one mechanism alone is never high-trust ranging.

## 2. Delay-Doppler-native support (§3)

`FieldAxis` (ADR-279) makes delay/Doppler first-class native axes so OTFS-style captures are stored natively, and `RfTensor::delay_doppler_map` provides the standard transform for frequency-time tensors: IDFT over bins (→ delay) × DFT over snapshots (→ Doppler). Measured: a synthetic scatterer at (delay 7, Doppler 3) produces a unit peak with < 1e-9 leakage everywhere else. The transform is implemented **separably** (delay IDFT per snapshot, then Doppler DFT per delay row — `O(B²S + S²B)` vs the direct form's `O(B²S²)`), proven equivalent to the direct reference to < 1e-10 and **measured 8.3× faster** (520 µs vs 4.34 ms at 56×8 in the criterion bench). Rule: derived features may be small, but delay-Doppler maps are not collapsed into scalar motion energy before provenance and local storage.

## 3. IEEE P3162 synthetic-aperture import (§5)

`SyntheticApertureSoundingDataset` (frequency range, aperture poses, directional PDP, coordinate system, processing-manifest hash) is the validated import profile — the calibration bridge between measured environments, Sionna-class simulators, and learned RF scene models. Schema + validation only; parsers arrive with the first real dataset.

## 4. Factorized pose (RePos) + log-age gating

`FactorizedPoseHead` separates what generalizes from what conditions:

- **relative skeleton** branch reads the environment-invariant content representation (cannot learn room-position shortcuts);
- **root localization** branch reads the geometry-conditioned representation (sensor pose is signal there — the PerceptAlign lesson);
- `absolute = root + relative` (`PoseOutput::absolute_joints_m`), with **calibrated per-joint and root residual σ** so every pose output carries uncertainty (ADR-273 item 8).

**The leakage experiment, head-algebra level** (`factorized_pose_head_algebra_resists_hand_built_room_feature`): training rooms where room position *correlates* with body scale (the trap real deployments set), held-out room breaking the correlation — factorized MPJPE **0.0003 m** vs monolithic absolute-head **0.2534 m** (845× worse). The representations there are hand-built 2/3-dim vectors, so the content/geometry separation is true by construction: MEASURED-CODE for the head's algebra only; not a pose-accuracy claim and not evidence about the encoder (#1438 item 1).

**The same trap end-to-end** (`tests/pose_shortcut_e2e.rs`): body scale exists only as radar cross-section in the ADR-276 physics, the room only as global link-geometry metadata, and the representations are the real pretrained `RfEncoder::encode_content`/`encode` outputs under a certified strict room holdout. SYNTHETIC-measured: factorized MPJPE **0.069 m → 0.085 m** (train → held-out room) vs monolithic **0.049 m → 0.305 m** — the monolithic head fits the training rooms *better* (the shortcut is genuinely attractive) and degrades **3.6×** worse when the correlation breaks.

Budget: the structured pose head is the largest adapter at **740 params vs the 40,856-param backbone (1.8 %)** — documented ceiling for structured heads is **< 2 %** (scalar heads keep the 1 % gate), both asserted in `every_head_fits_the_one_percent_budget_at_deployment_config`.

Age gating now matches the age-aware-CSI recipe exactly: the freshness gate input is `log(1 + sample_age_ms)` (`encoder::age_feature`), giving millisecond and multi-second staleness comparable input scale; the finite-difference gradient check re-proves the backward pass through the changed input.

## 5. Consequences

- Bluetooth/UWB anchors slot in as *geometric* evidence while WiFi carries ambient activity — the complement strategy, in code.
- The Gaussian primitive gained the lifecycle fields the update-loop spec requires (`first_seen_ns`, `doppler_variance`, bounded `source_receipts` lineage merged on fusion) — static structure is distinguishable from transients by lifetime, and every primitive traces to source frames.
- Roadmap, explicitly not done: real nRF54 CS capture path, OTFS waveform generation, P3162 file parsing, pose heads on real MM-Fi-style data.
