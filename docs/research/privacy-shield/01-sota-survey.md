# 01 — State of the Art

Scope: what a passive or active adversary can extract about *who* is in a space
and *what they are doing* from WiFi, the standard that broadens that surface, and
the countermeasures that try to prevent it. Claims are tagged **MEASURED** (from
a primary source, with metric), **CLAIMED** (asserted without an independent
measurement), or analytical inference (flagged).

---

## 1. The attack surface: beamforming feedback (BFI)

Since WiFi 5 (802.11ac), a client (beamformee) measures the downlink channel,
compresses the steering matrix **V** into **Givens-rotation angles φ/ψ**, and
transmits them **in cleartext** so the AP can steer beams. Anyone in monitor
mode can capture these frames for *every* client simultaneously — no network
access, and the target need carry no device. Quantization is coarse (802.11ac
angle steps of π/4…π/32 rad) yet retains rich motion and body information.

| Work | Venue / year | Result | Label |
|---|---|---|---|
| **BFId** — identity inference from BFI | ACM CCS 2025 (KIT/KASTEL) | Re-identifies individuals from BFI alone; novel 197-person dataset. Press reports **99.5%** in a controlled study (ACM full text was not openable to confirm class count/split) | MEASURED (paper); 99.5% is CLAIMED via press |
| **LeakyBeam** — occupancy through walls | NDSS 2025 | Occupancy detection **TPR 82.7% / TNR 96.7%** at **20 m, through walls**, from plaintext BFI. Proposes a BFI-obfuscation defense | MEASURED (attack); defense overhead CLAIMED |
| **BFIAttack** — CSI reconstruction from BFI | arXiv 2026 (USF) | Reconstructs CSI from BFI, then defeats CSI defenses. ASR: device auth 95.5% / user auth 92.6% / key-gen 94.2% (single-antenna), 1.5–6 m | MEASURED |
| **BeamSense** — activity recognition from BFI | Computer Networks vol. 258, 2025 (Northeastern) | Human activity recognition **up to 99.28%** on commodity 802.11ac, no firmware mod, ~10% better than CSI | MEASURED |
| **Wi-BFI** — capture tooling | arXiv 2309.04408, 2023 | Pip-installable extraction of 802.11 BFI from commercial devices | tooling |

**Takeaway for the defender.** BFI is the highest-leverage surface: unencrypted,
management-plane, device-free, capturable en masse with off-the-shelf tools. It
is also a *stepping stone* — BFIAttack shows BFI can reconstruct the CSI that all
older attacks assume.

---

## 2. The older adjacent surface: CSI identity/gait/activity

CSI requires special extraction (Intel 5300 / Atheros / ESP32) but is the
foundation the BFI attacks build on. Person-ID exploits **gait** as a biometric.
Representative MEASURED results (commodity WiFi, CSI amplitude):

| System | Accuracy | N (candidates) | Note |
|---|---|---|---|
| WiWho (IPSN 2016) | 92%→80% | 2→6 | 2–3 m straight walk |
| WiFi-ID (2016) | 93%→77% | 2→6 | wavelet features |
| WiPIN (2018) | 92–100% | ≤30 | operation-free |
| Deep-WiID (2019) | 92.5–99.7% | 6→15 | GRU |
| WiNet / LWID (2020) | 98.5% / 98.8% | 40 / 50 | CNN |

**Pattern the defender must exploit and not overstate:** accuracy is high in
small closed sets but *degrades as N grows and conditions become realistic*
(cross-day, cross-location, cross-walking-style). Chance is **1/N**; a 99% result
on N=5 is far weaker evidence than 99% on N=197. Open-world scale is largely
unproven (see *SoK: Security Evaluation of Wi-Fi CSI Biometrics*, 2025).

---

## 3. The standard: IEEE 802.11bf-2025

IEEE Std **802.11bf-2025** (Amendment 4: *Enhancements for WLAN Sensing*) was
published **26 September 2025**. It standardizes WLAN sensing in 1–7.125 GHz and
above 45 GHz, defining sensing capability signaling, measurement/sounding
setup, feedback types, and both passive (ambient-traffic) and active
(dedicated null-packet) sensing modes.

- **Attack-surface implication (analytical).** 802.11bf turns CSI/measurement
  acquisition from proprietary hacks into open, vendor-agnostic, machine-readable
  MAC signaling across heterogeneous devices — institutionalizing exactly the
  measurements the BFI attacks abuse. The standard frames sensing as a feature,
  not a threat.
- **The privacy gap (MEASURED from standards minutes).** A 2023 proposal for a
  BFI "secure transmission mechanism" (IEEE 802.11-23/0782) was **withdrawn**;
  "the group did not align on the characterization of [the] privacy problem."
  The standard shipped without privacy protections, and its own analysis admits
  passive eavesdroppers can extract location, respiration, heart rate, and
  identity.

---

## 4. Countermeasures (the defense literature)

All operate on the defender's *own* transmissions; none are jamming.

| Countermeasure | Venue / year | Mechanism | Effect | Label |
|---|---|---|---|---|
| **IRShield** | IEEE S&P 2022 | IRS/reconfigurable surface randomizes reflected paths | Attacker motion-detection **≤5%** | MEASURED |
| **PhyCloak** | USENIX NSDI 2016 | Full-duplex obfuscator injects Doppler/phase distortion into sensing only | **88.69%** gesture-spoof; throughput can rise (whitelist legit sensors) | MEASURED (spoof); throughput CLAIMED |
| **DP-Givens dithering** | IEEE DySPAN 2026 | Differentially-private stochastic quantization of BFI φ/ψ angles | Attacker speed-class error 19%→~73% (chance); **fine (3-bit) resolution ≈ non-private baseline throughput** | MEASURED |
| **MIMOCrypt / WiShield** | 2023 / IEEE JSAC 2024 | Secret precoding / MIMO CSI manipulation so only the intended RX decodes | Anti-tracking | CLAIMED/formal |
| **CSI Fuzzing / DP feature release** | IEEE 2024–25 | Randomized CSI features with DP budget | Formal DP guarantee | CLAIMED/formal |
| **ScatterShield** | ACM IMWUT 2025 | Backscatter tags inject controlled clutter | Defeats unauthorized sensing | MEASURED |
| **Adversarial packet perturbation** | ACM MobiCom 2024 | Small in-spec packet perturbations degrade attacker model | Symmetric defense | MEASURED |

**The fundamental tradeoff (MEASURED, DySPAN 2026).** Perturbing precoding/
feedback that an attacker exploits also degrades legitimate beamforming gain —
*but the cost collapses at fine feedback resolution*:

| Randomization | Attacker error | Beamforming gain retained |
|---|---|---|
| none | 19% | 100% |
| moderate (p=0.3) | >50% | median >90% |
| maximum (p≥0.9) | ~73% (≈chance) | median ~58% |

At **high (3-bit) feedback resolution, privacy was "nearly indistinguishable
from the non-private baseline"** in link performance. This is the empirical basis
for VEIL's design choice (compliant fine-resolution feedback shaping — see
[03-countermeasure-design.md](03-countermeasure-design.md)).

---

## 5. Where VEIL sits

The literature has two families: **external** obfuscation (IRShield/ScatterShield
— extra hardware, perturbs the channel) and **transmitter-side** feedback/precoder
shaping (DP-Givens, MIMOCrypt — no extra hardware, perturbs your own report).
VEIL is in the second family and adds the missing property the others do not all
combine: a transform that is simultaneously **energy-preserving** (provably
compliant), **key-reversible** (throughput-preserving for the legitimate link),
and **session-fresh** (defeats cross-session re-identification), unified around
the Givens-rotation primitive the report already uses.

---

## Sources

- BFId — ACM CCS 2025: https://dl.acm.org/doi/10.1145/3719027.3765062 · KIT record: https://publikationen.bibliothek.kit.edu/1000185756
- LeakyBeam — NDSS 2025: https://www.ndss-symposium.org/ndss-paper/lend-me-your-beam-privacy-implications-of-plaintext-beamforming-feedback-in-wifi/
- BFIAttack — arXiv 2604.04179: https://arxiv.org/html/2604.04179v1
- BeamSense — Computer Networks 2025: https://dl.acm.org/doi/10.1016/j.comnet.2024.111020 · arXiv 2303.09687: https://arxiv.org/pdf/2303.09687
- Wi-BFI — arXiv 2309.04408: https://arxiv.org/pdf/2309.04408
- SoK: Security Evaluation of Wi-Fi CSI Biometrics — arXiv 2511.11381: https://arxiv.org/pdf/2511.11381
- WiWho (IPSN 2016): https://dl.acm.org/doi/10.5555/2959355.2959359 · WiPIN — arXiv 1810.04106: https://arxiv.org/pdf/1810.04106
- Survey on Wi-Fi Sensing for Human Identity — MDPI Electronics 2023: https://www.mdpi.com/2079-9292/12/23/4858
- IEEE Std 802.11bf-2025: https://standards.ieee.org/ieee/802.11bf/11574/ · Overview — IEEE COMST 2024: https://ieeexplore.ieee.org/document/10547188/ · NIST: https://www.nist.gov/publications/ieee-80211bf-enabling-widespread-adoption-wi-fi-sensing
- 802.11bf privacy proposal withdrawal (802.11-23/0782), summarized: https://pascalpiron.substack.com/p/wifi-sensing-and-the-privacy-fix
- IRShield — IEEE S&P 2022 / arXiv 2112.01967: https://arxiv.org/abs/2112.01967 · https://ieeexplore.ieee.org/document/9833676/
- PhyCloak — USENIX NSDI 2016: https://www.usenix.org/conference/nsdi16/technical-sessions/presentation/qiao
- Protecting Human Activity Signatures in Compressed 802.11 CSI Feedback — DySPAN 2026 / arXiv 2512.18529: https://arxiv.org/abs/2512.18529
- MIMOCrypt — arXiv 2309.00250: https://arxiv.org/pdf/2309.00250 · WiShield — IEEE JSAC 2024: https://dl.acm.org/doi/abs/10.1109/JSAC.2024.3414597
- ScatterShield — ACM IMWUT 2025: https://dl.acm.org/doi/abs/10.1145/3770653
- Practical Adversarial Attack on WiFi Sensing — ACM MobiCom 2024: https://dx.doi.org/10.1145/3636534.3649367
- Privacy-Preserving Wi-Fi Data Generation via DP — INFOCOM 2025: https://www.eng.auburn.edu/~szm0001/papers/INFOCOM25.pdf
