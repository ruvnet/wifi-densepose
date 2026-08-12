# 02 — Threat Model

VEIL protects a physical space (a room, a ward, a boardroom, a SCIF) from
*unauthorized* WiFi-based inference of **who is present** and **what they are
doing**, without denying the space its own working WiFi. This file states the
adversary classes, exactly what VEIL defends, and — just as importantly — what
it does **not**.

---

## 1. Assets

| Asset | Why it matters |
|---|---|
| **Identity linkage** | Re-identifying a specific person across time/sessions from their RF signature (BFId-class attack) |
| **Occupancy / presence** | Whether the space is occupied, and by how many (LeakyBeam-class, through-wall) |
| **Activity / motion** | Gait, gestures, keystrokes, respiration inferred from channel dynamics (BeamSense-class) |
| **Communication utility** | The legitimate WiFi link must keep working (≥95% throughput bar) |

---

## 2. Adversary classes

| Class | Position | Capability | In VEIL scope? |
|---|---|---|---|
| **A1 — external passive sniffer** | Outside the trust boundary (adjacent room, van, hallway), monitor mode | Captures plaintext BFI/CSI for every station; runs BFId/LeakyBeam/BeamSense offline | **Primary target — yes** |
| **A2 — external active sensor** | Nearby, transmits its own probing/sounding to solicit measurable responses | Elicits sensing responses; 802.11bf "active" mode | **Partial** — cadence randomization + non-response policy help; full defense needs MAC-layer policy |
| **A3 — associated but curious AP** | Inside the link; the party VEIL shares keys with | Sees the un-rotated report by construction | **Out of scope** — this is BFLD's detection/privacy-class problem (ADR-118/141) |
| **A4 — supply-chain / firmware** | Compromised radio firmware | Can bypass any transmit-side control | Out of scope (integrity problem, not a waveform problem) |
| **A5 — physical / RF-denial** | Wants to *block* WiFi | — | Explicitly rejected: VEIL never jams |

VEIL's design centers on **A1**, the attacker the literature demonstrates and
the one no shipping product addresses.

---

## 3. What VEIL guarantees (and the evidence class)

1. **Cross-session identity unlinkability against A1.** Because the fine-subspace
   signature is rotated by a fresh secret orthogonal transform each session, an
   A1 attacker cannot average captures back to a stable per-person template.
   *Evidence: SYNTHETIC — re-ID collapses from 100% to ~chance in the reference
   experiment (`cargo test`); real-silicon witness is future work.*
2. **Communication preservation.** The transform is key-reversible by the
   legitimate receiver, and acts only on the identity-bearing fine subspace, so
   link throughput stays ≥95%. *Evidence: SYNTHETIC model + MEASURED external
   corroboration (DySPAN 2026: fine-resolution feedback shaping is near-free).*
3. **Compliance.** The transform is orthogonal ⇒ energy-preserving ⇒ adds no
   interfering emission ⇒ not jamming. *Evidence: machine-checked energy ratio =
   1.000000 in the `compliance` module; statutory analysis in
   [04-compliance-and-regulatory.md](04-compliance-and-regulatory.md).*

---

## 4. What VEIL does NOT do (non-goals, stated to prevent over-claiming)

- **It does not hide identity from the associated AP (A3).** That party holds the
  session key. Protecting against a malicious AP requires detection and policy
  (BFLD), not waveform shaping.
- **It is not RF denial or jamming.** It never degrades another station's link.
- **It does not, by itself, defeat within-session motion detection.** A single
  session's rotation is fixed, so coarse presence/motion may still be inferable
  within one capture window; sounding-cadence randomization mitigates but does
  not eliminate this. Identity *re-ID* (the brief's metric) is the guaranteed
  target; motion obfuscation is partial and tracked as future work.
- **It is not a camera-grade or medical-grade claim in any direction.**
- **It is not validated on hardware yet.** All quantitative defense results are
  SYNTHETIC until a captured boot/runtime log exists (CLAUDE.md hardware rule).

---

## 5. Trust boundary

```
        ┌────────────────────── protected space ──────────────────────┐
        │                                                              │
        │   [person]      [person]        legitimate STA ⇄ AP (VEIL)   │
        │      │             │                 │  shares session key   │
        │      └──── RF ──────┘                 │  rotates fine subspace│
        │            reflections                ▼  of its own BFI       │
        │                               compliant, key-reversible,      │
        │                               energy-preserving emission      │
        └───────────────────────────────────────┬──────────────────────┘
                                                 │  plaintext BFI on air
                                                 ▼
                    A1 external passive sniffer (monitor mode)
                    sees a freshly-rotated signature each session
                    → cannot build a stable per-person template
                    → re-identification → chance
```

The key never crosses the boundary to A1. The AP inside the boundary is trusted
for key-sharing (A3 out of scope). No emission crosses the boundary with intent
or effect of interfering with another station (A5 rejected).
