# 07 — Implementation and Roadmap

---

## 1. What ships in this bundle

- **Reference crate** `v2/crates/wifi-densepose-privshield` (VEIL): a
  deterministic, dependency-free, WASM-ready pure-compute leaf implementing the
  full attacker-vs-protector experiment, the four compliant controls, the
  throughput model, the compliance audit, the `optimize` hyper-optimizer, and a
  byte-stable proof. 35 tests + doctest pass; builds for
  `wasm32-unknown-unknown`; clippy-clean.
- **This research bundle** (`docs/research/privacy-shield/`).
- **[ADR-288](../../adr/ADR-288-veil-privacy-shield-compliant-waveform.md)** — the
  formal decision record.

The crate is intentionally a **leaf with no internal RuView dependencies**
(mirrors `wifi-densepose-aether`), so it can be reasoned about, fuzzed, and
ported independently, and so it can never accidentally acquire a path to a radio.

---

## 2. Reuse map (how VEIL composes with existing RuView)

| Existing subsystem | Relationship |
|---|---|
| **BFLD** (ADR-118/120/121, `wifi-densepose-bfld`) | Detection layer. Its `identity_risk_score` is the natural trigger for VEIL's `SensingDetector` — detect leakage, then shield |
| **Privacy control plane** (ADR-141) | VEIL protection steps emit `ComplianceReport`s that fit the runtime-attestation model (which mode, which actions, which fields) |
| **Active sensing / governed actuation** (ADR-280) | VEIL is a defensive `SensingAction`: a governed, privacy-ceiling-bounded emission-shaping action the control plane can schedule |
| **Givens/beamforming primitives** | VEIL reuses the report's native Givens-rotation structure rather than inventing a new transform |
| **Deterministic proof discipline** (`nvsim`, `archive/v1/verify.py`) | VEIL's `proof` module follows the same pinned-witness pattern |

---

## 3. Phased rollout

| Phase | Deliverable | Evidence class |
|---|---|---|
| **P1 — reference model (this PR)** | Crate + experiment + docs + ADR | SYNTHETIC (cargo test) |
| **P2 — sensitivity study** | Sweep N, noise, resolution, mixing; add a learned attacker to confirm signal-level collapse | SYNTHETIC |
| **P3 — BFLD integration** | Wire `identity_risk` → `SensingDetector` → shield engage; emit attestation | SYNTHETIC + integration tests |
| **P4 — firmware feedback shaping** | Implement keyed fine-subspace rotation + cadence randomization in ESP32/Nexmon feedback path | build + hardware |
| **P5 — two-node hardware measurement** | Wi-BFI attacker vs. VEIL protector on real silicon; iperf throughput; captured log | **MEASURED** (with witness) |
| **P6 — deployment profiles** | Per-segment profiles (SCIF, boardroom, ward) with regulatory review | operational |

No defense claim graduates from SYNTHETIC to MEASURED without a captured
boot/runtime log (CLAUDE.md hardware rule).

---

## 4. Open problems (tracked honestly)

1. **Real-hardware separability.** Comm and identity information are only
   *approximately* separable on real radios; the true throughput cost of full
   identity hiding may exceed the model's ~2%. P2/P5 must bound it.
2. **Within-session motion leakage.** A fixed per-session rotation does not
   obfuscate coarse motion within one capture window. Needs stronger cadence
   randomization or amplitude shaping; currently a stated non-goal for the re-ID
   metric.
3. **Active adversary (A2).** An attacker that transmits its own soundings is
   only partially addressed by cadence control; a MAC-layer non-response policy
   is needed.
4. **Key management.** The per-session rotation key must be derived from the
   negotiated link secret; VEIL's PRNG is explicitly *not* cryptographic and must
   not be used for real key material.
5. **Regulatory review per jurisdiction.** The energy-conservation argument is
   portable, but power/mask/timing limits and any transmit-nulling profile need
   local review before field use.

---

## 5. Validation commands

```bash
# Reference experiment + all unit/proof/doc tests
cargo test -p wifi-densepose-privshield --no-default-features

# WASM portability (leaf builds with no radio path)
cargo build -p wifi-densepose-privshield --target wasm32-unknown-unknown

# Lints
cargo clippy -p wifi-densepose-privshield --all-targets
```
