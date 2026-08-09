# 08 — Hyper-Optimization

The reference crate first shipped a **hand-picked** shield config (112 Givens
passes, 7-bit feedback). This file records how the `optimize` module replaces
that guess with a *derived*, robustness-verified optimum, and what it found. All
numbers are **SYNTHETIC / L0**, reproduced by
`cargo test -p wifi-densepose-privshield`.

---

## 1. What is being optimized, and against what

Two knobs, two objectives, one hard constraint:

| Knob | Costs | Does it trade against privacy? |
|---|---|---|
| `feedback_bits` (angle resolution) | Throughput: **residual** falls with bits, **feedback airtime** rises with bits | No — the keyed rotation is applied regardless of resolution |
| `givens_passes` (rotation mixing) | Compute only | Yes — more mixing ⇒ lower re-ID |

**Constraint:** re-ID must collapse into the chance band `1/N · 2 + 0.03` — and
it must do so *robustly*: for **both** attacker metrics (Euclidean and Cosine)
and **both** identity counts (N = 16 and N = 32, the harder, lower-chance case).

The key structural fact: **rotation mixing is throughput-free.** The per-session
rotation is derived from the shared link secret on both ends (like MIMOCrypt) —
it is never transmitted — so extra Givens passes cost compute, not airtime. That
means privacy margin is essentially free; the only throughput tradeoff lives in
`feedback_bits`.

---

## 2. Throughput is a 1-D problem with an interior optimum

Because the residual falls with bits while feedback airtime rises, throughput
has a genuine interior optimum in `feedback_bits` (`LinkModel`, default SNR 20 dB,
`feedback_overhead_per_bit = 0.0008`):

| bits | throughput ratio |
|---|---|
| 1 | 0.9681 |
| 2 | 0.9757 |
| **3** | **0.9769** ← unconstrained optimum |
| 4 | 0.9766 |
| **5** | **0.9760** ← shipped (spec-allowed) |
| 7 | 0.9744 (the old hand-picked value) |
| 9 | 0.9728 |
| 12 | 0.9704 |

The unconstrained optimum is **3 bits** — which coincides with the DySPAN-2026
MEASURED finding that ~3-bit feedback is the privacy–utility sweet spot, because
the receiver compensates the keyed rotation and extra bits mostly buy airtime.
802.11 compressed beamforming quantizes ψ/φ to roughly 5–9 bits, so the shipped
shield uses the throughput-best **spec-allowed** value, **5 bits** (0.9760),
rather than the out-of-spec 3-bit optimum. Either way it beats the old 7-bit
choice.

---

## 3. Mixing: the minimum robust budget, and a free margin

Worst-case shield-on re-ID vs. `givens_passes` (bits = 5; worst over Euclidean
and Cosine):

| passes | re-ID @ N=16 | re-ID @ N=32 | robust collapse? |
|---|---|---|---|
| 16 | 0.75 | 0.62 | no |
| 24 | 0.50 | 0.35 | no |
| 32 | 0.20 | 0.14 | no (N=32 band is 0.0925) |
| **48** | 0.12 | 0.057 | **yes** ← proven minimum |
| 64 | 0.078 | 0.044 | yes |
| **96** | **0.047** | **0.018** | **yes** ← shipped (2× margin) |
| 112 | 0.078 | 0.042 | yes (the old default — no better than 96) |

The proven minimum for robust collapse is **48 passes** — the hand-picked 112 was
**2.3× over-provisioned**. Since mixing is throughput-free, the shield ships
**96 passes** (`PRIVACY_MARGIN_FACTOR = 2` × 48, rounded up to a candidate): it
drives re-ID *below chance* at N=16 (0.047 < 0.0625) at zero throughput cost, and
is still cheaper compute than the original 112.

---

## 4. The adopted config, and why it beats the original

| | Old (hand-picked) | Hyper-optimized (shipped) |
|---|---|---|
| Givens passes | 112 | **96** (from proven-min 48 × 2) |
| Feedback bits | 7 | **5** (spec-optimal) |
| Shield-on re-ID (N=16) | 0.078 | **0.047** |
| Throughput ratio | 0.9744 | **0.9760** |
| Robust across metrics & N | not checked | **verified** |

The optimum is **strictly better on privacy and throughput at once**, and is now
*verified* rather than assumed. `ShieldConfig::default()` is exactly the
optimizer's output; the test `optimize::shipped_default_equals_optimizer_output`
fails if they ever drift apart.

---

## 5. The Pareto frontier (and an honest note)

`optimize::pareto_frontier` enumerates non-dominated (worst-case re-ID,
throughput) points over a pass × bits grid. In this model the frontier
**collapses toward the max-mixing, 5-bit point**, because mixing is
throughput-free — so beyond the throughput knob (bits) there is no privacy–
throughput tradeoff to trace. That degeneracy is itself the finding: *the only
thing privacy costs here is feedback resolution, and even that is cheap.* On real
hardware, where comm/identity subspaces are only approximately separable and
where more aggressive mixing may touch the data-carrying beam, this frontier is
expected to open up — a hardware study (roadmap P5) will re-measure it.

---

## 6. Robustness caveats (unchanged from the threat model)

- The collapse is verified against two classifiers and two N; a learned
  attacker on real captures must still be checked (P2/P5).
- `feedback_bits` affects only throughput in this model, not re-ID; on hardware,
  coarse quantization also adds obfuscation, which would *help* privacy — the
  model conservatively ignores that.
- All optimization results are SYNTHETIC until a hardware witness exists.
