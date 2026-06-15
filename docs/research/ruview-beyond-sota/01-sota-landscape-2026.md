# SOTA Landscape 2026 — The Bar a Beyond-SOTA RuView Must Clear

**Series**: ruview-beyond-sota (01)
**Date**: 2026-06-09
**Status**: Research survey / target definition
**Builds on (does not duplicate)**: `docs/research/sota-2026-05-22/00-summary.md` (physics floors, placement, privacy chain), `docs/research/BFLD/01-sota-survey.md` (beamforming-feedback leakage SOTA), `docs/research/neural-decoding/21-sota-neural-decoding-landscape.md` (sensor-fidelity framing), `docs/research/rf-topological-sensing/00-rf-topological-sensing-index.md` (mincut/topology resolution limits), ADR-150 (RF foundation encoder + measured MM-Fi campaign), ADR-147 (OccWorld benchmark proof).

## 0. Evidence legend

Every claim in this document carries one of three tags. **No RuView benchmark number in this document is invented**; all RuView numbers come from repo-internal measured artifacts.

| Tag | Meaning |
|-----|---------|
| **[V]** | Verified in this session via web search (June 2026); source linked in §8 |
| **[K]** | Training-knowledge claim (pre-2026 literature); plausible but **not re-verified** — treat as needing citation check before external publication |
| **[I]** | Internal RuView measurement or artifact (ADR, issue, witness bundle) — measured, not literature |

---

## 1. SOTA reference table per capability axis

### 1.1 Pose estimation (WiFi CSI)

| Method | Year | Metric | Dataset / protocol | Tag |
|--------|------|--------|--------------------|-----|
| DensePose From WiFi (Geng, Huang, De la Torre) | 2023 | Dense-pose UV regions from CSI, "comparable to image-based approaches" (same-layout); commonly cited AP≈43.5 / AP@50≈87.2 | 3×3 antenna, single-layout lab | exact AP numbers **[K]**; paper existence **[V]** (arXiv 2301.00250) |
| MetaFi++ (Zhou et al.) | 2023 | PCK@50 = **97.30%** same-domain real-world (MetaFi: 95.23%); drops to **81.7–86.5%** under stricter protocols | Own capture; protocol-sensitive | **[V]** |
| Person-in-WiFi 3D (CVPR 2024) | 2024 | End-to-end multi-person 3D; 20.4 M params, **54 FPS**; MPJPE ≈ 90–100 mm on own dataset | Own multi-person dataset | FPS/params **[V]**; MPJPE range **[K]** |
| GraphPose-Fi (arXiv 2511.19105) | 2025 | SOTA on MM-Fi random split: **MPJPE 160.6 mm**, best PCK at all thresholds | MM-Fi, random split (S1) | **[V]** |
| CSDS (Electronics 14(4):756) | 2025 | Wi-Pose: PCK@5 = **0.6407**, PCK@50 = **0.8824** | Wi-Pose | **[V]** |
| PerceptAlign (arXiv 2601.12252) | 2026 | Cross-layout 3D: MPJPE **222.4 mm** (Scene 4) / **317.1 mm** (Scene 5), >54% better than prior cross-layout SOTA; in easier settings MPJPE 181.5 mm, PCK@20/50 = 44.2/79.5 | Cross-layout protocol | **[V]** |
| WiFlow (arXiv 2602.08661) | 2026 | Lightweight continuous HPE, spatio-temporal decoupling | — | **[V]** (existence; numbers not extracted) |
| **RuView / AetherArena** | 2026 | **81.63% torso-PCK@20 in-domain (random split), beating MultiFormer's 72.25%** on metric/protocol-matched MM-Fi; **leakage-free cross-subject collapses to ~11.6% torso-PCK zero-shot**; official-split harness baseline ~63–65% PCK@20; **11 KB LoRA few-shot calibration → 72.5%** | MM-Fi (issue #876, ADR-150 §3) | **[I]** |

**The honest reading of the pose axis**: same-domain WiFi pose is "solved-looking" (PCK@50 in the 90s) and meaningless for deployment. The 2025–2026 literature has shifted to cross-layout/cross-subject protocols, where numbers collapse (PerceptAlign PCK@20 = 44.2 cross-layout **[V]**; RuView cross-subject zero-shot 11.6% **[I]**). ADR-150's measured finding — that the cross-subject gap is **subject-distribution shift, not an algorithmic gap**, and that **few-shot in-room calibration (5–200 frames) closes it** — is ahead of where the published literature is: no published WiFi-pose paper we found ships a per-room ~11 KB adapter calibration mechanism. **[I]**

### 1.2 Presence / person count

| Method | Year | Metric | Tag |
|--------|------|--------|-----|
| Large-scale commodity router deployment (>10 M routers) | 2025 | **92.6% motion-detection accuracy** across diverse homes | **[V]** (ISAC survey, arXiv 2510.14358) |
| LeakyBeam (NDSS 2025) | 2025 | Occupancy through walls at 20 m from **plaintext BFI alone**: TPR 82.7%, TNR 96.7% | **[V]** (also in BFLD survey §4.2) |
| Time-Selective RNN multi-room presence (arXiv 2304.13107) | 2023 | Device-free multi-room presence from CSI | **[V]** (existence) |
| Academic person counting (0–5 occupants, lab) | 2020–2024 | typically 90–97% exact-count accuracy, degrading sharply >5 people | **[K]** |
| **RuView** | 2026 | `cog-person-count` ships with calibrated uncertainty (`count_p95_low/high`); multistatic placement recipe with **100% coverage for 1–4 occupants at N=5 nodes (synthetic physics)** | **[I]** (sota-2026-05-22 R6.2.5, ADR-113) |

### 1.3 Vital signs (HR / BR)

| Method | Year | Metric | Tag |
|--------|------|--------|-----|
| PhaseBeat (ACM Health) | 2020 | HR median error **1.19 bpm**; BR median error **0.25 breaths/min** | **[V]** |
| MDPI Sensors 24(7):2111 non-contact HR | 2024 | HR accuracy 96.8%, **median error 0.8 bpm** | **[V]** |
| PulseFi (arXiv 2510.24744) | 2025 | Low-cost ML cardiopulmonary + **apnea** monitoring from CSI | **[V]** (existence; numbers not extracted) |
| mmWave FMCW vitals (60 GHz class) | 2023–2026 | HR MAE typically 1–3 bpm at 1–3 m, single subject; age-balanced reference dataset published (Sci Data 2026) | dataset **[V]**; MAE range **[K]** |
| Contactless blood pressure (WiFi-band) | — | **NEGATIVE** — below classical physics floor; recoverable only via quantum magnetometry path | **[I]** (R13/R20 arc, ADR-114) |
| **RuView** | 2026 | `wifi-densepose-vitals` (ADR-021) extracts HR/BR from ESP32 CSI; chest-centric placement gives **+27 pp coverage** for vitals cogs (synthetic) | **[I]** — **no accuracy-vs-ECG validation number exists in-repo yet; do not claim one** |

**Bar**: published single-subject, line-of-sight, 1–3 m WiFi HR is ~0.8–1.2 bpm median error **[V]**. Nobody credibly publishes multi-person, through-wall, walking-subject HR at that accuracy — that is open territory.

### 1.4 Localization (ToA / CRLB)

| Method | Year | Metric | Tag |
|--------|------|--------|-----|
| 802.11mc FTM | shipped | 1–2 m typical accuracy | **[V]** (FTM survey, arXiv 2509.03901) |
| 802.11az (+ 802.11bk) | released | **sub-1 m**, 160 MHz channels, secured ranging, HE-LTF repetitions | **[V]** |
| AI single-link decimeter localization | 2025 | **0.63 m average error** single-link, beating Widar2.0 / Dynamic-MUSIC | **[V]** |
| SpotFi / Chronos / Widar lineage | 2015–2021 | 0.4–1 m with multi-AP CSI AoA/ToF | **[K]** |
| **RuView** | 2026 | CRLB / Fisher-information machinery in `ruvector/src/viewpoint/geometry.rs`; tomography ISTA voxel grid; **theoretical** limits derived internally: 30–60 cm at 16 nodes/1 m spacing, 8.8 cm information-theoretic dense limit | **[I]** (rf-topological-sensing doc 09 — synthetic derivations, no bench numbers) |

### 1.5 Through-wall

| Method | Year | Metric | Tag |
|--------|------|--------|-----|
| RF-Pose / RF-Pose3D (MIT, FMCW 5.4–7.2 GHz) | 2018 | Through-wall skeletal pose, ~specialized radar not commodity WiFi | **[K]** |
| Commodity 2.4 GHz through-wall imaging (arXiv 1903.03895) | 2019 | Coarse imaging through walls with commodity WiFi | **[V]** (existence) |
| Radio tomographic imaging (RTI) lineage | 2010–2013 | Through-wall tracking via RSS networks, ~0.5–1 m tracking error | **[V]** (papers) / error figure **[K]** |
| LeakyBeam (NDSS 2025) | 2025 | Through-wall **occupancy** at 20 m, passive, commodity | **[V]** |
| **RuView** | 2026 | RF tomography module (`tomography.rs`, ISTA L1 voxel solver) + CIR (ADR-134) exist as code; **PABS structure detection: 1,161× static / 9.36× dynamic intruder lift (synthetic)** | **[I]** |

Notably, the 2025–2026 web literature shows through-wall *pose* (not just presence) on commodity WiFi remains essentially where it was in 2019 — no verified commodity-WiFi through-wall pose benchmark surfaced in our searches. The frontier moved to privacy attacks (BFI) instead.

### 1.6 Identity / re-ID (capability and threat simultaneously)

| Method | Year | Metric | Tag |
|--------|------|--------|-----|
| BFId (KIT, ACM CCS 2025) | 2025 | **~99.5% (near-100%) re-ID across 197 subjects** from beamforming feedback alone, ≥5 s of BFI | **[V]** (also BFLD survey §4.1) |
| Transformer CSI identification | 2025 | **99.82%** on stationary subjects | **[V]** |
| WhoFi (arXiv 2507.12869) | 2025 | Deep person re-ID via WiFi channel encoding, ~95% rank-1 class results | existence **[V]**; exact number **[K]** |
| Wi-Gait | 2023 | 92.9% over 10 subjects, robust to walking cofactors | **[V]** |
| **RuView** | 2026 | AETHER contrastive re-ID embeddings (ADR-024) in pose tracker; **BFLD**: first *defensive* identity-leak detector (identity_risk_score) — the literature attacks, RuView audits | **[I]** |

### 1.7 Adjacent modality: mmWave radar (the accuracy ceiling WiFi is chasing)

| Method | Year | Metric | Tag |
|--------|------|--------|-----|
| mmChainPose | 2025 | **27.0 mm MPJPE** / 0.8706 OKS on MARS (mmWave point cloud) | **[V]** |
| ProbRadarM3F (arXiv 2405.05164) | 2024–25 | SOTA AP across joints, probability-map fusion | **[V]** |
| Seeed MR60BHA2-class 60 GHz FMCW | shipped | Commodity $15 HR/BR/presence module — already in RuView's hardware table | **[I]** |

mmWave is ~6× better than the best WiFi MPJPE (27 mm vs 160 mm) **[V]**. The strategic implication: WiFi will not beat mmWave on raw geometry; it wins on ubiquity, cost, through-wall propagation, and standardized waveforms (§2). RuView already hedges with the ESP32-C6 + MR60BHA2 fusion node. **[I]**

---

## 2. IEEE 802.11bf — status and implications

**Status (verified)**: IEEE **802.11bf-2025 is ratified and published** (IEEE SA lists the amendment; ratification late 2024 / publication 2025) **[V]**. It amends MAC/PHY of HE (Wi-Fi 6) and EHT (Wi-Fi 7) plus DMG/EDMG (60 GHz) to support WLAN sensing in 1–7.125 GHz and >45 GHz bands **[V]**. The Wi-Fi Alliance has Wi-Fi Sensing as an active certification work area built on 802.11bf (presence/proximity, gestures, vital signs) **[V]**. Market reports claim >47 chipset vendors with 802.11bf-compatible programs as of early 2026 — single weak source, treat as directional **[V, low confidence]**.

**What it implies for RuView**:

1. **Sounding-on-demand becomes standard.** 802.11bf defines a sensing-measurement procedure (sensing initiator/responder, trigger-based sounding, threshold-based reporting). Today RuView relies on Espressif's vendor CSI API and Nexmon firmware patches; post-bf, commodity Wi-Fi 7 silicon will expose scheduled sensing measurements without firmware hacks. The rvCSI normalized `CsiFrame` schema is the right abstraction layer to absorb a future bf adapter (`rvcsi-adapter-*`). **[I]**
2. **The moat moves up the stack.** When every router can sense, raw CSI access stops being differentiating. Differentiators become: multistatic fusion, coherence gating / anti-hallucination, calibration mechanisms, witness-grade verification, and privacy auditing — exactly RuView's existing bets (ADR-029/135/150/028, BFLD). **[I]**
3. **Privacy pressure intensifies.** 802.11bf standardizes the capability that BFId/LeakyBeam exploit. BFLD's identity-leak detection and the ADR-105–109 privacy/PQC chain become regulatory assets, not nice-to-haves. **[V]+[I]**
4. **Threshold-based reporting** in bf (report only when channel changes exceed threshold) is architecturally the same idea as RuView's coherence gate — validation that the gate belongs at the protocol layer. **[K]** (bf reporting detail from training knowledge)

---

## 3. RF foundation model landscape ("GPT for RF")

Verified 2025–2026 attempts, all young, none dominant:

| Model | Approach | Downstream tasks | Tag |
|-------|----------|------------------|-----|
| **LWM (Large Wireless Model)** | Pretrained on large-scale CSI → general channel embeddings | LoS/NLoS, beats raw features in low-data regimes | **[V]** |
| **LatentWave** (arXiv 2606.06373) | JEPA pretraining on wireless spectrograms + CSI | RF classification, 5G NR positioning, beam prediction, LoS/NLoS | **[V]** |
| **WirelessJEPA** (arXiv 2601.20190) | Multi-antenna spatio-temporal latent prediction | Cross-task transfer | **[V]** |
| **IQFM** | Contrastive SSL on raw I/Q | Modulation classification, beam prediction, RF fingerprinting, few-shot | **[V]** |
| **Multimodal Wireless FMs** (arXiv 2511.15162), **WMFM** (arXiv 2512.23897), **SoM** (arXiv 2506.07647) | Vision + RF multimodal for 6G ISAC | Sensing-communication integration | **[V]** |
| **DeepSig OmniSIG** | Commercial AI-native RF sensing, 500 MHz/GPU spectrum | Signal ID (LTE/5G/Wi-Fi) | **[V]** |

**Critical observation**: every verified RF foundation model targets *communication-side* tasks (beam prediction, LoS/NLoS, modulation, positioning). **None of them is a human-sensing foundation model** — none pretrains for pose/vitals/identity invariances. ADR-150's measured negative result is the sharpest data point in this space: pose-contrastive pretraining across subjects **failed on MM-Fi because the invariance is not in the data** (loss never left the ln(B) floor) **[I]**. The literature has not yet published this failure mode; the field's "GPT for RF sensing" narrative is ahead of its evidence. The defensible foundation-model objective (per ADR-150 §3.5–3.6) is **reduce few-shot calibration cost**, not zero-shot invariance. **[I]**

---

## 4. "Beyond SOTA" for RuView — precise definition

Targets below are **bar definitions**, not claims. RuView numbers in the "current" column are measured [I]; targets must be proven via the AetherArena witness protocol (ADR-149) before being asserted anywhere.

| Capability | Published SOTA (2026) | RuView measured today | RuView beyond-SOTA target | Key obstacle |
|------------|----------------------|----------------------|---------------------------|--------------|
| Pose, in-domain (MM-Fi) | GraphPose-Fi 160.6 mm MPJPE; MultiFormer 72.25% torso-PCK@20 **[V]** | **81.63% torso-PCK@20** (already > published) **[I]** | Hold #1 under leakage-free audit + per-joint tables published with witness rows | Protocol fragmentation; reviewers distrust WiFi-pose numbers |
| Pose, cross-subject zero-shot | ~collapse everywhere; PerceptAlign PCK@20 44.2 cross-layout **[V]** | 11.6% torso zero-shot; 63–65% in-harness official split **[I]** | Stop chasing it (measured dead end); instead **few-shot frontier** below | Subject-distribution shift is in the data, not the model (ADR-150 §3.2) |
| Pose, deployment calibration | **No published per-room adapter mechanism found** | **11 KB LoRA, 100–200 frames → 72.5%; cross-env K=5 → 60.1%** **[I]** | ≤20 frames → ≥70% PCK@20, adapter ≤11 KB, 30 s on-site; publish as the first calibration-service benchmark | Needs diverse-room capture fleet to validate beyond MM-Fi |
| Presence/motion (commodity) | 92.6% across 10 M routers **[V]** | Synthetic placement recipe 100% coverage N=5 **[I]** | ≥99% presence with calibrated p95 bounds on $6–15 ESP32 mesh, bench-validated | All placement numbers are synthetic; Tier-2.3 bench validation outstanding |
| Person count | ~90–97% lab, ≤5 people **[K]** | cog ships uncertainty intervals **[I]** | Exact count 1–6 people ≥95% with honest intervals, multistatic, real bench | Multi-person CSI superposition; no public multi-occupancy benchmark |
| Vital signs HR | 0.8–1.2 bpm median, single subject, LoS, 1–3 m **[V]** | No in-repo ECG-validated number — **must not be claimed** | ≤1.5 bpm MAE vs ECG ground truth, *multi-person or through-wall*, witness-bundled | R13 physics floor: ~5 dB shortfall at distance; needs chest-centric placement + PABS |
| Vital signs BP | NEGATIVE at WiFi band (matches internal R13) | nvsim quantum path only **[I]** | First validated quantum-classical fused bedside vitals (ADR-114) | NV-diamond hardware maturity, 2028+ |
| Localization | 0.63 m single-link AI; sub-1 m 802.11az **[V]** | CRLB machinery, no bench number **[I]** | ≤30 cm multistatic on ESP32 mesh (internal theory says feasible at N=16) | ESP32 clock sync / phase offset (TDM protocol exists, unproven at this accuracy) |
| Through-wall | Occupancy yes (LeakyBeam); commodity pose: nothing credible **[V]** | tomography + CIR code, PABS 9.36× lift (synthetic) **[I]** | First witnessed commodity-WiFi through-wall *person localization* (not pose) ≤1 m | Wall attenuation eats the R6.1 4.7 dB multi-scatterer budget |
| Identity / re-ID | ~99.5% @ 197 subjects (attack) **[V]** | AETHER + **BFLD defensive auditing** (no published competitor) **[I]** | Ship the first identity-leak risk score with DP budget hook; keep re-ID opt-in only | Calibrating risk score at 802.11ax 4/2-bit quantization (BFLD open Q2) |
| Verification | **Nothing comparable published** — no WiFi-sensing paper ships deterministic re-verification | ADR-028 witness bundles, SHA-256 proof, 7/7 self-verify, 1,031+ tests **[I]** | Make witness-grade reproduction the *expected* standard: every public claim = one-command verification | Community adoption, not technology |
| Foundation encoder | Comms-task FMs only (LWM/JEPA family) **[V]** | Masked-CSI + coherence head planned; pose-contrastive refuted **[I]** | First *sensing* FM whose acceptance metric is calibration-sample reduction (frames-to-72% halved) | SSL must match production CSI pipeline (ADR-149 resampling risk) |

---

## 5. Where RuView already matches/exceeds published work

1. **In-domain MM-Fi pose** — 81.63% torso-PCK@20 vs MultiFormer 72.25%, metric- and protocol-matched (issue #876). **[I]**
2. **Deployment-calibration mechanism** — the 11 KB LoRA per-room adapter with measured frames-to-accuracy curves (§3.4–3.6 of ADR-150) has no published equivalent; the literature is still arguing about zero-shot generalization that ADR-150 measured to be a data property.
3. **Deterministic witness verification** — ADR-028's SHA-256 pipeline proof + self-verifying bundles exceeds the reproducibility practice of every WiFi-sensing paper surveyed (none ship deterministic re-verification).
4. **Multistatic cost point** — $6–15/node ESP32 mesh with TDM sync, channel hopping, placement recipes (ADR-113) vs literature setups using Intel 5300/AX210 laptops or USRPs; ~$30/bed vs $3,000 clinical monitor framing (R16).
5. **Defensive identity auditing (BFLD)** — the field publishes attacks (BFId, LeakyBeam, WhoFi); RuView is building the only detector/auditor, plus a PQC-hardened federation privacy chain (ADR-105–109) with no published counterpart.
6. **Anti-hallucination coherence gating** — confidence gated by RF integrity (ADR-135, ADR-150 §2.4); WiFi-pose papers uniformly lack a "the model knows when the channel is bad" signal.
7. **Negative-result discipline** — physics floors (R13 BP, R6.1 4.7 dB), refuted pose-contrastive pretraining — published SOTA papers do not report these, which inflates the apparent literature bar.

## 6. Where RuView lags

1. **Bench validation** — nearly all multistatic/placement/tomography numbers are synthetic-physics; the 92.6%-on-10M-routers deployment **[V]** is real-world evidence at a scale RuView cannot approach.
2. **Vital-sign ground truth** — no in-repo ECG/respiration-belt validated HR/BR error; published work has 0.8 bpm median **[V]**. This is the most urgent claim gap.
3. **Raw geometric accuracy** — mmWave (27 mm MPJPE **[V]**) and even best-WiFi MPJPE (160.6 mm **[V]**) have no RuView MPJPE counterpart published; AetherArena reports PCK only.
4. **802.11bf-native capture** — RuView is on vendor CSI APIs and Nexmon patches; no bf sensing-procedure adapter exists yet in rvCSI.
5. **Multi-person pose** — Person-in-WiFi-3D does end-to-end multi-person at 54 FPS **[V]**; RuView's pose path is effectively single-person (multi-person exists only in count/placement work).
6. **Dataset scale and diversity** — MM-Fi only; ADR-150 §3.3 shows the binding constraint is room/device/protocol diversity, which requires the capture fleet that doesn't exist yet.

## 7. Strategic synthesis

The 2026 bar is bimodal: **lab in-domain numbers are saturated** (PCK@50 > 95%, HR < 1 bpm) and **deployment numbers are collapsed** (cross-layout PCK@20 ≈ 44, zero-shot cross-subject ≈ 11%). 802.11bf-2025 commoditizes raw sensing; foundation models commoditize comms-side embeddings. "Beyond SOTA" for RuView is therefore *not* a leaderboard delta — it is owning the three layers the field hasn't built: **(a)** witnessed, deterministic, leakage-audited evaluation; **(b)** the few-shot calibration service (11 KB adapters) as the deployment answer the zero-shot literature lacks; **(c)** the privacy/integrity layer (BFLD + coherence gate) that 802.11bf-era regulation will demand. Each row in §4's target table is gated on the AetherArena witness protocol — a target becomes a claim only when it ships with a one-command reproduction.

---

## 8. Verified sources (accessed 2026-06-09 via web search)

Pose: [GraphPose-Fi](https://arxiv.org/html/2511.19105v1) · [PerceptAlign / cross-layout](https://arxiv.org/html/2601.12252) · [CSDS](https://www.mdpi.com/2079-9292/14/4/756) · [Person-in-WiFi 3D](https://aiotgroup.github.io/Person-in-WiFi-3D/) · [DensePose From WiFi](https://arxiv.org/abs/2301.00250) · [MetaFi++](https://www.researchgate.net/publication/369644995_MetaFi_WiFi-Enabled_Transformer-based_Human_Pose_Estimation_for_Metaverse_Avatar_Simulation) · [WiFlow](https://arxiv.org/html/2602.08661v2)
Vitals: [PhaseBeat](https://dl.acm.org/doi/abs/10.1145/3377165) · [Non-contact HR (Sensors 24:2111)](https://www.mdpi.com/1424-8220/24/7/2111) · [PulseFi](https://arxiv.org/pdf/2510.24744) · [mmWave vitals dataset (Sci Data)](https://www.nature.com/articles/s41597-026-07172-9)
Localization: [FTM survey 802.11mc/az/bk](https://arxiv.org/abs/2509.03901) · [Decimeter single-link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12846125/) · [SelfLoc 802.11az](https://www.mdpi.com/2079-9292/14/13/2675)
802.11bf: [IEEE SA 802.11bf-2025](https://standards.ieee.org/ieee/802.11bf/11574/) · [TGbf](https://www.ieee802.org/11/Reports/tgbf_update.htm) · [NIST overview](https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing) · [Wi-Fi Alliance work areas](https://www.wi-fi.org/current-work-areas) · [ISAC survey (10M-router 92.6%)](https://arxiv.org/pdf/2510.14358)
Identity: [BFId / KIT CCS 2025 coverage](https://www.gblock.app/articles/wifi-signal-person-identification-surveillance-study-may-2026) · [WhoFi](https://arxiv.org/html/2507.12869v1) · [Wi-Gait](https://www.sciencedirect.com/science/article/abs/pii/S1389128623001962) · [LeakyBeam NDSS 2025](https://www.ndss-symposium.org/ndss-paper/lend-me-your-beam-privacy-implications-of-plaintext-beamforming-feedback-in-wifi/)
Through-wall: [RTI through-wall](https://ieeexplore.ieee.org/document/6214374/) · [Commodity 2.4 GHz imaging](https://arxiv.org/pdf/1903.03895) · [Multi-room presence](https://arxiv.org/pdf/2304.13107)
Foundation models: [LatentWave](https://arxiv.org/html/2606.06373) · [WirelessJEPA](https://arxiv.org/pdf/2601.20190) · [Multimodal Wireless FMs](https://arxiv.org/pdf/2511.15162) · [WMFM](https://arxiv.org/html/2512.23897) · [SoM](https://arxiv.org/pdf/2506.07647) · [RF-native AI / LWM, IQFM, OmniSIG](https://aicompetence.org/rf-native-ai-models-for-the-invisible-spectrum/)
mmWave: [mmChainPose](https://www.sciencedirect.com/science/article/abs/pii/S0925231225026918) · [ProbRadarM3F](https://arxiv.org/html/2405.05164v3)

Internal [I] sources: ADR-150 (§1, §3.2–3.6), ADR-147, ADR-028, ADR-113/114, issue #876, `docs/research/sota-2026-05-22/00-summary.md`, `docs/research/BFLD/01-sota-survey.md`, `docs/research/rf-topological-sensing/`.
