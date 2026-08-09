# 03 — Countermeasure Design

How VEIL prevents unauthorized sensing with compliant waveform controls, and how
the design maps to [`v2/crates/wifi-densepose-privshield`](../../../v2/crates/wifi-densepose-privshield).

---

## 1. The separable-subspace principle

A compressed beamforming report is not homogeneous. Two blocks carry different
information:

- **Dominant beam direction (comm block).** The coarse steering the AP uses to
  aim data at the client. It varies with position and traffic and carries **no**
  stable identity. **Throughput rides here.**
- **Fine cross-subcarrier phase structure (fine block).** The high-order
  multipath detail. It is *stable per person* across sessions and is what
  re-identification exploits (BFId). **Identity leaks here.** Communication
  barely uses it.

The whole design rests on this: **identity leakage and data throughput live in
(mostly) separable subspaces.** A transform confined to the fine block can wreck
re-identification while sparing the beam the link depends on. This is consistent
with the DySPAN-2026 MEASURED result that shaping fine-resolution feedback is
nearly free in throughput.

---

## 2. The four compliant waveform controls

VEIL alters "channel sounding, phase, or beam schedules" — exactly the levers the
brief names — all within the 802.11 waveform envelope:

| Control | What it varies | Purpose |
|---|---|---|
| **Keyed precoder rotation** (primary) | A fresh secret orthogonal transform of the *fine* subspace each session, composed from extra Givens rotations | Destroys cross-session identity linkage; energy-preserving; key-reversible |
| **Feedback quantization / dither** | Sub-step noise on reported φ/ψ angles | Adds report-level uncertainty; tunes the privacy–throughput point via `feedback_bits` |
| **Sounding-cadence randomization** | Jitter on NDP sounding intervals | Under-samples motion for an eavesdropper; charged as the throughput overhead |
| **MU-group / stream-mapping shuffle** | Which STAs are grouped, stream-to-antenna mapping | Rotates the spatial signature over time |

All four modify the node's **own** standards-conformant frames. None adds energy
on top of another station (see [04](04-compliance-and-regulatory.md)).

---

## 3. Why the keyed Givens rotation is the right primitive

The compressed beamforming report is *already* a product of Givens rotations
(the φ/ψ angles). VEIL composes **additional keyed Givens rotations** over the
fine block. This choice gives three properties at once:

1. **Orthogonal ⇒ energy-preserving.** A Givens rotation preserves the vector's
   L2 norm exactly. Composing many still preserves it. So the emission carries
   the same power it always would — **no added energy, no interference, not
   jamming.** The `compliance` module checks this: energy ratio = 1.000000.
2. **Keyed & reversible ⇒ throughput-preserving.** The legitimate AP/STA shares
   the per-session key, derives the identical rotation schedule, and applies the
   inverse (negated angles, reversed order) to recover the true precoder. It pays
   only the tiny residual from quantizing the extra angles at `feedback_bits`
   resolution — negligible at 7+ bits — plus the sounding overhead.
3. **Fresh per session ⇒ unlinkable.** A different rotation each session means an
   A1 sniffer sees `R_e · signature` for a new random `R_e` every time. Averaging
   over sessions (the natural enrollment attack) drives
   `mean_e(R_e · signature) → 0` for *every* identity, so all templates collapse
   toward the origin and become indistinguishable — re-identification → chance.
   This is the marginalized-mutual-information argument: over unknown rotations,
   the signature carries no stable discriminative information.

This is the shared-secret precoding idea (cf. MIMOCrypt) specialized to the
identity-bearing subspace and unified around the report's native primitive.

---

## 4. Detect-then-act

Per the brief ("detect sensing activity and alter…"), VEIL need not perturb
continuously. The `SensingDetector` exposes the decision rule: when the observed
rate of sensing/NDP solicitations crosses a threshold, the control plane
(ADR-280) engages the shield. Continuous operation is also valid; gating just
saves the (already small) overhead when no sensing is present.

---

## 5. Module map

| Concept above | Crate module | Key items |
|---|---|---|
| Deterministic, WASM-safe randomness + keys | `prng` | `Rng` (SplitMix64), `fnv1a_64`, `derive_key` |
| Givens algebra, energy conservation | `linalg` | `apply_givens`, `norm`, `dist_sq` |
| SYNTHETIC two-subspace BFI model | `identity` | `SceneConfig`, `Channel`, `BfiSample` (`comm()`/`fine()`) |
| The four controls (shield) | `protector` | `ShieldConfig`, `Protector::protect`/`recover`, `SensingDetector` |
| Passive re-ID adversary | `attacker` | `NearestCentroidAttacker` |
| Privacy–throughput tradeoff | `throughput` | `LinkModel::throughput_ratio`, `beamforming_residual` |
| "Not jamming" audit | `compliance` | `ComplianceReport::audit`/`is_compliant` |
| Attacker-vs-protector head-to-head | `experiment` | `ExperimentConfig`, `run`, `ExperimentReport` |
| Byte-stable deterministic witness | `proof` | `Proof::EXPECTED_WITNESS`, `Proof::witness` |

---

## 6. The privacy–throughput knob

The design exposes one honest tuning knob, matching the literature:

- **`feedback_bits` high (7–9):** the legitimate receiver's residual is ~1e-5 →
  throughput ≈ baseline; privacy is full (rotation is fresh regardless of bits).
  This is VEIL's operating point.
- **`feedback_bits` low (≤3) or extra additive dither:** more robustness to a
  key-recovery-adjacent attacker, at measurable throughput cost.
- **`sounding_overhead`:** the dominant (small) throughput cost, from cadence
  randomization; trades motion-obfuscation strength against airtime.

The `throughput` module computes the ratio from these, so the tradeoff is
inspectable rather than asserted (`cargo test throughput`).

---

## 7. Honest limitations of the model

- The two-subspace split is an abstraction; on real hardware comm and identity
  information are only *approximately* separable, so the real throughput cost of
  fully hiding identity may be higher than the model's ~2%. The DySPAN-2026
  MEASURED curve is the external sanity check that it is *small* at fine
  resolution, not zero.
- The nearest-centroid attacker is deliberately simple. The collapse argument is
  classifier-independent (it is about the signal, not the model), but a hardware
  study must confirm a strong learned attacker also collapses.
- Within-session motion is not addressed by the rotation alone (see threat
  model §4).
