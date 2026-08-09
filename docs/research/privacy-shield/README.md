# Privacy Shield Research Bundle — VEIL

**VEIL** (Verifiable Emission-shaping for Identity-Leakage prevention) is a
privacy *firewall* for WiFi sensing: it prevents unauthorized identity and
activity inference from a room's WiFi while preserving normal communications. It
is the **countermeasure** counterpart to [BFLD](../BFLD/) — where BFLD *detects*
when beamforming feedback becomes identifying, VEIL *acts* by shaping the node's
own compliant waveform (channel sounding, precoder phase, beam/feedback
schedules) so identity and activity inference fail, while a legitimate receiver
sees an essentially unchanged link.

**This must use compliant waveform controls, never jamming.** Every technique
here operates on the defender's *own* legitimately transmitted, standards-
conformant frames. Nothing adds energy to interfere with another station's
transmission (the statutory definition of jamming, 47 U.S.C. §333/§302a).

---

## Table of contents

| File | Purpose |
|------|---------|
| [01-sota-survey.md](01-sota-survey.md) | State of the art: identity/activity inference attacks (BFI + CSI), the IEEE 802.11bf-2025 standard, and privacy-preserving countermeasures |
| [02-threat-model.md](02-threat-model.md) | Adversary classes, what VEIL defends and what it explicitly does not, trust boundary |
| [03-countermeasure-design.md](03-countermeasure-design.md) | The compliant waveform controls, the separable-subspace principle, keyed Givens-rotation shield, and how it maps to the crate |
| [04-compliance-and-regulatory.md](04-compliance-and-regulatory.md) | The legal line between compliant waveform control and jamming, with statutory citations |
| [05-experiment-protocol.md](05-experiment-protocol.md) | The attacker-vs-protector experiment: metrics, acceptance bar, reproducer, and results |
| [06-market-and-buyers.md](06-market-and-buyers.md) | First buyers, procurement drivers, competitive landscape, and the standards-body gap |
| [07-implementation-and-roadmap.md](07-implementation-and-roadmap.md) | Crate layout, reuse map, hardware path, phased rollout, and open problems |
| [08-optimization.md](08-optimization.md) | Hyper-optimization: throughput-optimal feedback resolution, minimum robust mixing budget, Pareto frontier, and the adopted config |
| [09-sota-update-2026.md](09-sota-update-2026.md) | 2025–2026 SOTA update (verified, cited): stronger attacks (BFI→CSI reconstruction, through-wall vitals, keystroke), validated compliant defenses, and the derived VEIL improvement backlog |

Formal decision: [ADR-288](../../adr/ADR-288-veil-privacy-shield-compliant-waveform.md).
Reference implementation: [`v2/crates/wifi-densepose-privshield`](../../../v2/crates/wifi-densepose-privshield).

---

## Executive summary

1. **The threat is real and now standardized.** IEEE 802.11ac/ax beamforming
   feedback (BFI) — the compressed Givens-rotation angle matrices (φ/ψ) a client
   sends the AP — travels **unencrypted on the management plane**. Any device in
   monitor mode can capture it for every client at once, no network access, and
   the target need carry no device. **BFId** (KIT, ACM CCS 2025) re-identifies
   individuals from BFI alone; **LeakyBeam** (NDSS 2025) detects occupancy
   through walls at ~20 m from BFI; **BeamSense** recognizes activities at up to
   99.28% from BFI. IEEE Std **802.11bf-2025** (published 26 Sep 2025)
   standardizes the sensing measurement/feedback surface these attacks abuse.

2. **The standards body declined to fix it.** A 2023 proposal for a BFI
   "secure transmission mechanism" (IEEE 802.11-23/0782) was **withdrawn** —
   the working group did not align on characterizing sensing privacy as a
   distinct problem. 802.11bf shipped without privacy protections. This is the
   single strongest demand signal: the gap is structural and acknowledged.

3. **No targeted anti-sensing product ships (as of 2026).** Every countermeasure
   in the literature — IRShield, PhyCloak, MIMOCrypt, DP-Givens dithering,
   ScatterShield — is research-stage. The only shipping substitute is broadband
   RF shielding (SCIF/TEMPEST film/paint), which is blunt: it kills *all* RF and
   cannot coexist with wanted WiFi. The whitespace is a **selective, coexisting,
   software/PHY** shield.

4. **The VEIL mechanism.** Identity leaks through the *fine* cross-subcarrier
   phase structure of a beamforming report; throughput rides the *dominant*
   beam direction. These are (mostly) separable subspaces. VEIL composes extra
   **keyed Givens rotations** over the fine subspace only. The rotation is
   *orthogonal* (energy-preserving ⇒ not jamming), *keyed per session* (the
   legitimate receiver inverts it ⇒ throughput preserved), and *fresh each
   session* (a sniffer cannot average it back ⇒ re-ID collapses to chance).

5. **Measured on the reference model (SYNTHETIC), at the hyper-optimized
   operating point.** On the default synthetic scene (16 candidate identities),
   a passive re-identifier scores **100% with the shield off** and **4.7% with
   it on** (chance = 6.25%), while modeled link throughput stays at **97.6%** of
   baseline and the emission energy ratio is **1.000000** (compliant). The shield
   config is chosen by the `optimize` module — 96 Givens passes (2× the proven-
   minimum 48 for robust collapse across both attacker metrics and N∈{16,32}) at
   5-bit feedback resolution — not hand-picked (see
   [08-optimization.md](08-optimization.md)). Reproduce:
   `cargo test -p wifi-densepose-privshield`.

6. **Scope, honestly.** VEIL defends against a *third-party passive sniffer*. It
   does **not** hide identity from the associated AP (that party holds the key)
   — that is BFLD's detection/policy problem. VEIL is a reference model, not
   hardware: real-silicon validation (per CLAUDE.md) is future work with a
   captured-log witness.

---

## Evidence discipline

Per repository policy, every quantitative claim is tagged:

- **MEASURED** — from a cited primary source with its metric and conditions.
- **CLAIMED** — asserted by a source (vendor PR, press, standards minutes)
  without an independent measurement.
- **SYNTHETIC** — produced by VEIL's own deterministic model; reproduced by
  `cargo test`, describing the model and not real hardware.

WiFi sensing is never presented here as camera-grade, and no VEIL result implies
a defense guarantee on real silicon until a hardware witness exists.
