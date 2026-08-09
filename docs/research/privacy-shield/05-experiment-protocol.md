# 05 — Experiment Protocol: Attacker vs. Protector

This is the "start today" deliverable from the brief: **make one RuView node the
attacker and one the protector, and measure whether protection drives identity
recognition toward chance while keeping throughput above 95%.** It is realized as
a deterministic, reproducible experiment in
[`v2/crates/wifi-densepose-privshield`](../../../v2/crates/wifi-densepose-privshield).

Because it runs on **SYNTHETIC** data (no radio is touched), its numbers describe
the model, not real hardware — reproduced by `cargo test`, and to be
re-established on silicon with a captured log before any deployment claim.

---

## 1. Setup

- **Protector node.** Emits beamforming feedback shaped by the VEIL controls
  (keyed per-session fine-subspace rotation + configured feedback resolution and
  sounding overhead). Models a legitimate AP/STA protecting a room.
- **Attacker node.** A passive sniffer that enrolls a template per candidate from
  captured reports, then classifies fresh captures (nearest-centroid) — the
  BFId-class re-identification threat.
- **Scene.** `SceneConfig` default: 64-dim report, 8 comm dims, **16 candidate
  identities** (chance = 1/16 = 6.25%), per-identity stable fine-block signature
  + per-session environmental nuisance.

Two runs of the attacker are compared: **shield off** (the attacker sees raw
reports) and **shield on** (every captured report is VEIL-protected). The same
attacker faces both.

---

## 2. Metrics and acceptance bar

| Metric | Definition | Bar |
|---|---|---|
| **Re-ID accuracy, shield off** | Top-1 identity accuracy on unprotected traffic | Must be well above chance (threat is real) — bar ≥ 0.5 |
| **Re-ID accuracy, shield on** | Top-1 identity accuracy on protected traffic | Must fall into the chance band `1/N · 2 + 0.03` |
| **Throughput ratio** | Protected link capacity ÷ baseline capacity | **≥ 0.95** |
| **Compliance** | Emission energy ratio ≈ 1 and non-interfering | `is_compliant == true` |

Overall `passed()` requires all four.

---

## 3. Results (SYNTHETIC, hyper-optimized default configuration)

Reproduce with `cargo test -p wifi-densepose-privshield` (all 35 tests + doctest
pass). The default shield config is the `optimize` module's output — 96 Givens
passes at 5-bit feedback resolution (see
[08-optimization.md](08-optimization.md)). Salient values from the reference run:

| Metric | Value |
|---|---|
| Candidate identities | 16 |
| Chance level | 6.25% |
| Chance band (acceptance) | ≤ 15.5% |
| **Re-ID accuracy, shield OFF** | **100.0%** |
| **Re-ID accuracy, shield ON** | **4.7%** |
| **Throughput ratio** | **97.60%** |
| Emission energy ratio | 1.000000 |
| Overall verdict | **PASS** |

Reading the result: the attacker is a *perfect* re-identifier without protection
(the synthetic signatures are cleanly separable), and VEIL drives it *to the
chance floor* (4.7% sits just below the ideal 6.25%, i.e. no better than
guessing) — while the modeled link keeps 97.6% of its throughput and the
emission conserves energy exactly (compliant, not jamming). The same collapse
holds under a Cosine-metric attacker and at N=32, confirming it is a property of
the signal, not the classifier.

---

## 4. Determinism and the witness

The experiment is byte-reproducible: no OS entropy, no wall-clock, no threads.
`proof::Proof` folds the salient outputs (quantized to avoid last-bit f32
round-off) into an FNV-1a witness pinned as `EXPECTED_WITNESS`. Any drift in the
PRNG stream, rotation schedule, throughput formula, or scene geometry changes the
witness and fails `witness_matches_pinned`. This is the same
deterministic-proof discipline as `nvsim` and the Python `verify.py`.

---

## 5. Sensitivity and what to vary next

`ExperimentConfig` exposes the levers for a fuller study:

- **`scene.identities`** — larger N lowers the chance floor; confirm collapse
  holds as candidates grow.
- **`scene.env_sigma` / `beam_amplitude`** — nuisance and comm energy; stress the
  separability assumption.
- **`shield.feedback_bits`** — trace the privacy–throughput curve (the
  `throughput` tests already show coarse resolution costs more).
- **`shield.givens_passes`** — mixing strength; fewer passes should degrade the
  collapse gracefully.
- **Stronger attacker** — swap in a learned classifier to confirm the collapse is
  signal-level, not classifier-level (the argument says it must be, but a
  hardware study should verify).

---

## 6. Path to a real two-node measurement

The synthetic experiment is the design proof. The hardware path (per CLAUDE.md,
requires a captured log to claim MEASURED):

1. Two ESP32-S3/C6 or Nexmon-capable nodes: one runs Wi-BFI capture (attacker),
   one runs a VEIL-shaped feedback profile (protector).
2. Enroll and test the same BFId-style classifier on captured BFI, shield off vs.
   on; log throughput via iperf across the legitimate link.
3. Success = the same shape as §3 on real captures, with the boot/runtime log as
   the witness. Until then, all defense numbers remain SYNTHETIC.
