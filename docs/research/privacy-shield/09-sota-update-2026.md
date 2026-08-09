# 09 — SOTA Update (2025–2026) and VEIL Improvement Backlog

Source: a fan-out deep-research run (5 angles → 20 primary sources → 93 claims →
top 25 adversarially verified with 3-vote panels → 24 confirmed, 1 refuted).
Each finding carries its **evidence class** (`MEASURED` with metric / `CLAIMED`
/ `SYNTHETIC` / `STANDARDS-MINUTE`) and a primary URL. This file records what
changed in the field and the concrete backlog it implies for VEIL (ADR-288/289).
Nothing here upgrades VEIL's own numbers to `MEASURED` — that still requires a
captured hardware log (CLAUDE.md).

---

## 1. The threat surface got worse (and cheaper)

| Finding | Evidence | Source |
|---|---|---|
| **BFId** — first *identity* inference from plaintext BFI: **99.5% over 197 people**, perspective/gait-independent; BFI carries ~740 features vs 212 for CSI, so it *beats* CSI for identity; one eavesdropper captures BFI from all clients | `MEASURED` (top-1, N=197, CCS 2025) | [dl.acm.org/10.1145/3719027.3765062](https://dl.acm.org/doi/10.1145/3719027.3765062) |
| **LeakyBeam** — through-wall occupancy at **20 m** (TPR 82.7% / TNR 96.7%) **and breathing/vital-sign** leakage from *stationary* occupants; single antenna, Wireshark, no keys | `MEASURED` (NDSS 2025) | [ndss 2025-5](https://www.ndss-symposium.org/wp-content/uploads/2025-5-paper.pdf) |
| **WiKI-Eve / SThief** — keystroke & PIN/password theft from BFI (88.9% per-keystroke; 65.8% top-10 app passwords; POS keypads) with no device compromise | `MEASURED` (CCS 2023 / IEEE) | [WiKI-Eve](https://dl.acm.org/doi/10.1145/3576915.3623088) · [SThief](https://ieeexplore.ieee.org/document/10621321/) |
| **BFIAttack** — **reconstructs full CSI from sniffed BFI**: closed-form ≥93% (single-antenna, 1 attempt); MLE with physics/standard constraints 73% (multi-antenna, ≤5 attempts). Collapses the BFI-vs-CSI distinction | `MEASURED` (arXiv Apr 2026) | [arxiv 2604.04179](https://arxiv.org/html/2604.04179v1) |
| **BeamSense** — BFI sensing is standards-compliant, needs no firmware mod, ~10% higher activity accuracy than CSI | `MEASURED` | [BFISense/BeamSense](https://www.researchgate.net/publication/402468114_BFISense_Using_Beamforming_Feedback_Information_for_Wi-Fi_Sensing) |

**Implication:** the attacker is a *passive, keyless, single commodity antenna at
~20 m, through walls*, that can (a) identify people, (b) read vitals and
keystrokes, and (c) **reconstruct CSI from the BFI itself.** VEIL's threat model
must treat all four as baseline.

---

## 2. Defenses — the field validates VEIL's family and adds stronger primitives

| Defense | Mechanism | Effect | Evidence | Source |
|---|---|---|---|---|
| **LeakyBeam defense** | AP-side **per-packet random unitary** `Q_obf` on the LTF via the 802.11 spatial-mapping mechanism (standard says "not restricted"); AP recovers `V = Q_obf · V_obf`; **clients unmodified** | attack **89.7% → ~51%** across 8 APs (~1.6M packets/49 h) | `MEASURED` | [ndss 2025-5](https://www.ndss-symposium.org/wp-content/uploads/2025-5-paper.pdf) |
| **PrivISAC (RIS)** | Paired per-row RIS vectors, one randomly active per slot; preserves comm-direction response, corrupts sensing direction; time-domain mask/demask for the authorized RX | **93% → ~30%**, and **29% vs. retrained 5-location adaptive attacker** | `MEASURED` (64-element FPGA RIS, Intel 5300, ~2,700 OTA samples) | [arxiv 2601.04488](https://arxiv.org/html/2601.04488) |
| **DP-Givens** | ε-DP stochastic quantizer on the Givens rotation/phase angles; closed-form angular sensitivity → principled ε budget; preserves 802.11 feedback structure | frontier: attacker error 19% → ~73%; beamforming gain 0.97 → 0.89 median (0.54 at full) | `SYNTHETIC` (Monte-Carlo) | [arxiv 2512.18529](https://arxiv.org/pdf/2512.18529) |
| **Adaptive-DP (CSI spectrogram)** | Importance-weighted (non-uniform) DP budget across the time-frequency plane | better privacy-utility than flat noise at equal ε∈[0.5,2]; cuts identity + membership inference | `CLAIMED` (unrefereed) | [arxiv 2512.20323](https://arxiv.org/abs/2512.20323) |
| **BeamDancer** | Randomized native-beamforming obfuscation | defeats supervised + unsupervised localization and micro-Doppler; **compliant, not jamming** | `MEASURED` (IEEE TWC 2024) — **do NOT cite its ">96% PDR" (refuted here)** | [ieee 10739908](https://ieeexplore.ieee.org/document/10739908/) |
| **TX-side CSI obfuscation (+ counter-attacks)** | Filter the whole frame incl. LTS; DNN de-obfuscation for authorized sensing | **security contested**: "Defeating CSI obfuscation" + SnoopFi FIA/CRA recover the signal | `CLAIMED` design + published rebuttal | [C&S 2025](https://www.sciencedirect.com/science/article/abs/pii/S0167404825002834) |

**Where VEIL sits:** VEIL's keyed Givens rotation is the *same family* as the
LeakyBeam per-packet unitary and the DP-Givens knob — and unlike additive/DP
dither, VEIL's transform is **secret and orthogonal**, which is exactly the
property that should resist the BFIAttack closed-form/MLE inversion (the attacker
has no key, so there is no closed-form to invert to). That is now the decisive
claim to *test*, not assume.

---

## 3. Compliance / legal line

- **BeamDancer (IEEE TWC 2024)** is the peer-reviewed precedent for VEIL's
  stance: **jamming and geofencing are non-compliant / non-scalable; exploiting
  the standard beamforming mechanism stays 802.11-compliant** (validated without
  disabling firmware). Cite it as the compliance precedent — but **not** its
  refuted throughput figure.
- **Governance gap (unfilled):** *no* claim on the 802.11bf-2025 standard's
  privacy provisions, the withdrawn secure-LTF-from-11az proposal, or
  GDPR/HIPAA/EMSEC/ICD-705 boundaries **survived 3-vote verification** in this
  run. Blog/secondary sources assert a withdrawn privacy proposal, but it needs
  primary WG-minute/draft sourcing before VEIL relies on it. Tracked as an open
  question.

---

## 4. VEIL improvement backlog (derived, prioritized)

Priority = (verified severity) × (fit to VEIL). `[code]` = crate change,
`[docs]` = documentation, `[hw]` = hardware path.

1. **`[code]` ✅ implemented — Reconstruction-aware attacker (decisive).** A
   BFIAttack-style adversary (`attacker::ReconstructionAttacker`,
   `AttackerKind::Reconstruction`) recovers the direction of the CSI consistent
   with the *captured* report and classifies it; the test
   `reconstruction_attacker_collapses` confirms the keyed *orthogonal secret*
   rotation leaves it at chance (no key → it only ever recovers the rotated
   direction) while it still wins on unprotected traffic. *(BFIAttack, MEASURED)*
2. **`[code]` ✅ implemented — Adaptive, multi-capture attacker.**
   `attacker::AdaptivePoolingAttacker` (`AttackerKind::AdaptivePooling`) pools all
   captures per identity and whitens by per-dimension std before matching (the
   PrivISAC adaptive/retraining adversary); `adaptive_pooling_attacker_collapses`
   confirms collapse still holds. *(PrivISAC, MEASURED)*
3. **`[code]` ✅ implemented — Per-packet random-unitary mode.**
   `protector::ObfMode::PerPacketUnitary` applies a fresh unitary per packet,
   AP-side and **client-transparent** (LeakyBeam family; 802.11 spatial mapping
   "not restricted" as the compliance basis);
   `per_packet_unitary_mode_collapses_and_is_compliant` verifies it. *(LeakyBeam
   defense, MEASURED)*
4. **`[code]` ✅ implemented — DP-Givens ε knob.** `ShieldConfig.dp_epsilon` adds
   an ε-scaled angular dither, renormalized to preserve emission energy (still
   not jamming); `throughput::dp_residual` makes ε a real privacy↔throughput knob
   (`dp_epsilon_lowers_throughput_as_it_tightens`), and the combined
   rotation+DP still collapses and stays compliant. Outputs `SYNTHETIC`.
   *(DP-Givens, SYNTHETIC)*

> Items 1–4 landed with the reference **witness unchanged**
> (`0x350d…f448`) — the new controls/attackers are opt-in fields; the shipped
> default config and its numbers are byte-identical.
5. **`[code/docs]` Privacy–throughput *frontier*, not binary claims.** Report
   attacker-error-vs-privacy and gain/PDR-vs-privacy curves (we already have the
   throughput-vs-bits and reid-vs-passes curves; add the joined frontier).
6. **`[docs]` Threat-model upgrade.** Elevate identity/gait re-ID, through-wall
   vitals, keystroke/PIN, and **BFI→CSI reconstruction** to primary threats in
   ADR-288 §threat and bundle 02; add the passive/keyless/20 m/through-wall
   adversary as the default. *(done in this update)*
7. **`[docs]` Security honesty.** State that VEIL's shield security is `CLAIMED`
   until it survives published de-obfuscation attacks (SnoopFi / "Defeating CSI
   obfuscation"); add learned de-obfuscation to the attacker roadmap.
8. **`[code/docs]` Evaluation battery.** Adopt BeamDancer's three-attacker matrix
   (supervised localizer + unsupervised clusterer + model-based Doppler) as a
   minimum test set, plus identity + membership-inference metrics.
9. **`[hw]` Hardware-validation path.** Mirror the RIS/8-AP OTA testbeds for P5.
   **Correction:** ESP32 is an *attacker/sensor* node only (its WiFi lower layer
   is a closed blob exposing CSI *read*, not TX-feedback shaping); the protector
   needs **openwifi (SDR/FPGA), Nexmon (C firmware patches), or vendor
   firmware** + key agreement for the keyed-reversible version. See roadmap §P4.
10. **`[docs]` Governance sourcing.** Fill the 802.11bf privacy-provision gap
    with primary WG minutes/draft; scope FCC Part 15, GDPR/HIPAA (inferred
    biometric/health), and EMSEC/ICD-705 deployability.

---

## 5. Open questions the evidence did not close

- Does VEIL's obfuscation degrade **CSI *reconstructed* from BFI** (BFIAttack),
  or only raise raw-BFI feature noise? *(the decisive effectiveness question)*
- What is VEIL's **own MEASURED** privacy–throughput frontier on silicon (the
  only measured PDR number in the field was refuted; the DP curves are
  simulation-only)?
- Does 802.11bf-2025 contain any privacy provision or a withdrawn one, and what
  are the concrete FCC/GDPR/HIPAA/ICD-705 deployment boundaries?

---

## Sources (primary, verified in this run)

- BFId — CCS 2025: https://dl.acm.org/doi/10.1145/3719027.3765062
- LeakyBeam (attack + per-packet-unitary defense) — NDSS 2025: https://www.ndss-symposium.org/wp-content/uploads/2025-5-paper.pdf
- BFIAttack (BFI→CSI reconstruction) — arXiv 2026: https://arxiv.org/html/2604.04179v1
- WiKI-Eve — CCS 2023: https://dl.acm.org/doi/10.1145/3576915.3623088
- SThief — IEEE: https://ieeexplore.ieee.org/document/10621321/
- BeamSense/BFISense: https://www.researchgate.net/publication/402468114_BFISense_Using_Beamforming_Feedback_Information_for_Wi-Fi_Sensing
- PrivISAC (RIS) — arXiv 2026: https://arxiv.org/html/2601.04488
- DP-Givens — arXiv 2512.18529: https://arxiv.org/pdf/2512.18529
- Adaptive-DP spectrogram — arXiv 2512.20323: https://arxiv.org/abs/2512.20323
- BeamDancer — IEEE TWC 2024: https://ieeexplore.ieee.org/document/10739908/
- TX-side CSI obfuscation — Computers & Security 2025: https://www.sciencedirect.com/science/article/abs/pii/S0167404825002834

*Refuted (do not cite): BeamDancer ">96% PDR in LoS" (verification 1–2). Two DP
mechanisms are SYNTHETIC/CLAIMED, not silicon. Governance/standard pillar
unverified in this run.*
