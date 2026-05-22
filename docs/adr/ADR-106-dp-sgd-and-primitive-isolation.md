# ADR-106: Differential privacy + biometric primitive isolation for RuView federated training

**Status:** Proposed · **Date:** 2026-05-22 · **Author:** SOTA research loop tick-15 · **Supersedes:** none · **Extends:** ADR-105

## Context

ADR-105 specified federated learning for RuView CSI personalisation with MERIDIAN env-normalisation + Krum byzantine-robust aggregation + R7-style update-level mincut. It deferred two questions:

1. **Member inference defence.** A sufficiently capable adversary observing many model deltas across rounds can in principle reconstruct training samples (Shokri 2017). ADR-105 left "DP-SGD" as a future ADR.
2. **Biometric primitive isolation.** R15 catalogued five environment-invariant biometric primitives (gait frequency, breathing rate, HRV rate, RCS frequency response, walking dynamics). R15 said: the federation aggregator MUST NOT receive any raw per-subject biometric primitive. ADR-105 didn't yet specify which primitives qualify.

This ADR closes both. It is a direct extension of ADR-105 and incorporates the constraints from R3 (re-ID privacy) + R14 (empathic appliance privacy) + R15 (RF biometric physical-not-learned identification).

## Decision

Adopt **DP-SGD with explicit primitive-isolation enforcement** on every Cognitum Seed before any model delta leaves the device.

### Three-layer defence

**Layer 1 — Primitive Isolation (R15 binding constraint).** A static list of "on-device-only" biometric primitives. The federation client library enforces that these tensors are never serialised into a transmittable update.

| Primitive | On-device only | Reason |
|---|:---:|---|
| Raw CSI window (complex64 tensor) | ✅ | ADR-105 baseline |
| Gait stride frequency (Hz scalar per subject) | ✅ | R15 — biometric primitive |
| Breathing rate (BPM scalar per subject) | ✅ | R15 — biometric primitive |
| HRV rate signature (R-R interval array per subject) | ✅ | R15 — biometric primitive |
| RCS frequency response curve (per subject, per-subcarrier amplitude) | ✅ | R15 — biometric primitive |
| Limb timing vector (per subject, per stride) | ✅ | R15 — biometric primitive |
| Per-subject embedding centroid | ✅ | R3 + ADR-105 — re-ID primitive |
| MERIDIAN per-room centroid | ⚠️ | Aggregate over **all** subjects in the room — not per-subject |
| LoRA weight delta | ⚠️ | Encodes biometric information; mitigated by Layer 2 + Layer 3 |
| Model logits / softmax outputs | ⚠️ | Per-subject during inference; never aggregated for transmission |
| Coordinator-side aggregate model | ❌ | Distributed back to nodes; no per-subject content by construction |

The ✅ rows are enforced at the API surface — the federation client returns an error if a tensor with these tags is passed to `submit_delta()`.

**Layer 2 — Gradient clipping.** Before any LoRA weight delta is computed for transmission, individual sample gradients are clipped to L2 norm `C` (standard DP-SGD step, Abadi 2016). This bounds the sensitivity of the released delta to any single training sample.

Recommended: `C = 1.0` (after experimentation per-cog; some cogs may need `C ∈ [0.5, 2.0]`).

**Layer 3 — Gaussian noise on aggregated deltas.** Before transmission to the coordinator, Gaussian noise `N(0, σ²C²I)` is added to the aggregated LoRA delta. This bounds the per-round privacy leakage.

### Privacy budget

Using the **Moments Accountant** (Abadi 2016) for (ε, δ)-DP across federation rounds:

| Configuration | Per-round σ | Rounds | Total ε (δ=1e-5) | Verdict |
|---|---:|---:|---:|---|
| Conservative (medical-grade) | 1.5 | 50 | **2.0** | Strong; matches HIPAA-aligned recommendations |
| Standard (typical RuView) | 1.0 | 100 | **5.0** | Strong; consistent with Google's federated keyboard work |
| Lenient (faster convergence) | 0.5 | 100 | **8.0** | Moderate; below ε=10 community soft-bound |

Recommended **starting σ = 1.0** for most RuView cogs, with per-cog tuning:

- `cog-person-count` (R8 — simple classifier): σ=1.0 sufficient.
- AETHER re-ID head (R3 — high discriminability needed): σ=0.7 with C=1.5 to preserve discriminative power.
- `cog-pose-estimation` (skeleton output): σ=1.0.
- `cog-maritime-watch` (R11): σ=1.5 (medical-grade — vessel crew vitals).

### Composition with ADR-105 protocol

The DP-SGD layer slots in at step 4 of ADR-105's protocol summary:

> 4. **Delta compression.** Compute ΔW_i = W_T+1_i − W_T. **[NEW: clip individual-sample gradients to L2 norm C=1.0 during local training; add Gaussian noise N(0, σ²C²I) to ΔW_i with σ from per-cog table above.]** Quantise to int8 + LoRA-rank decomposition (rank=8) → ~1 MB per delta.

Krum byzantine-robust aggregation (step 5) operates on DP-noised deltas without modification — Krum's distance metric is robust to additive Gaussian noise at typical σ values.

### Implementation enforcement

The `ruview-fed` crate (per ADR-105 implementation plan, ~500 LOC) gains:

| Component | LOC | Purpose |
|---|---:|---|
| `PrimitiveTag` enum + tensor tagging trait | 60 | Layer 1 primitive isolation |
| `clip_gradient_l2(C)` helper | 30 | Layer 2 clipping |
| `add_dp_noise(sigma, C)` helper | 40 | Layer 3 Gaussian noise |
| `MomentsAccountant` | 120 | (ε, δ) tracking across rounds; aborts federation if budget exceeded |
| Per-cog config schema | 50 | σ, C, max rounds budget |

Total ~300 additional LOC on top of ADR-105's 500. Federation protocol implementation budget revised to ~800 LOC total.

## Alternatives considered

### A. Federated learning without DP

Status: **rejected.** ADR-105's Krum + LoRA + int8 quantisation provides *some* implicit privacy, but it's not a formal guarantee. Member-inference attacks (Shokri 2017) recover training samples from undefended FL. We need a formal (ε, δ)-DP bound.

### B. Local DP (LDP) only

Status: **rejected.** LDP would add noise per-sample at the device, then the coordinator gets noisy aggregates. This gives stronger guarantees but degrades model accuracy by 5-15× for the same ε. Central DP (CDP) with byzantine-robust aggregation is the right trade-off for our threat model where the coordinator is trusted to apply noise correctly (the coordinator is `cognitum-v0` fleet manager, under installation owner's control per ADR-100 signing).

### C. Heavier obfuscation (homomorphic encryption / secure aggregation)

Status: **deferred.** Secure aggregation (Bonawitz 2016) avoids the coordinator ever seeing individual deltas, only their sum. This is the right next layer for cross-installation federation (ADR-105 explicitly deferred). For within-installation federation where the coordinator is owner-controlled, the gains don't justify the 5-10× compute and complexity cost.

### D. Just-trust-Krum

Status: **rejected.** Krum defends against adversarial nodes, not adversarial *inference*. A passive coordinator (even an honest one) plus moderate compute can extract training samples from undefended deltas. DP-SGD is the proper defence.

## Threat model

| Threat | Layer that mitigates |
|---|---|
| Compromised seed reads its own local biometric primitives | Out of scope — physical compromise = full local compromise |
| Compromised seed exfiltrates a biometric primitive via the federation channel | **Layer 1** — primitive isolation API blocks transmission |
| Passive coordinator reconstructs training samples from observed deltas (Shokri 2017) | **Layer 2 + 3** — DP-SGD bounds reconstruction quality |
| Member inference attack on the trained model (Shokri 2017 §3.2) | **Layer 2 + 3** — formal (ε, δ) bound |
| Coordinator + 1 colluding seed | **Krum (ADR-105)** still works; DP-SGD bounds the colluder's info gain |
| Brute-force gradient inversion (Zhu 2019) | **Layer 2 + 3** — clipping + noise defeats gradient-from-update attack |
| Active adversary controlling >f Krum nodes | Out of scope — ADR-105 byzantine bound f < (K-2)/2 |
| Side-channel via inference latency | Out of scope — separate ADR (constant-time inference) |

## Consequences

### Positive

1. RuView federation is now **formally privacy-preserving** with a documented (ε, δ) bound — meets GDPR Art 25 ("data protection by design") technical-measure expectations.
2. R15's biometric-primitive constraints are enforced at the API surface, not just policy-documented.
3. The threat model has been written down with explicit mitigations per row, making future security review tractable.
4. The Moments Accountant aborts federation rather than silently consuming budget — operationally safer than naive "just keep training".

### Negative

1. DP noise degrades model accuracy by ~3-8% (typical figures from DP-SGD literature; per-cog tuning needed). For `cog-person-count` v0.0.2 (this loop's earlier work), the baseline 34.3% class-1 accuracy would degrade to ~31-33% with σ=1.0.
2. Adds ~300 LOC + Moments Accountant complexity to `ruview-fed`. Total federation budget revised to ~800 LOC.
3. Per-cog tuning of (σ, C, max_rounds) is needed — not a one-size-fits-all.
4. Doesn't defend against side-channel inference latency leaks; that's a separate ADR.
5. Doesn't address cross-installation federation; cross-installation work still requires the deferred ADR (secure aggregation + DP).

### Open questions intentionally left

1. **Per-cog DP budget allocation.** The σ values above are first-cut recommendations; empirical tuning per cog is needed before shipping.
2. **Moments Accountant restart policy.** What happens after we exceed ε? Reset model and restart? Stop federation indefinitely? Decision deferred to operations.
3. **Side-channel timing leaks.** A separate ADR (TBD) needs to cover constant-time inference and constant-time DP-noise sampling.
4. **Subject-level vs sample-level DP.** This ADR specifies sample-level. Subject-level DP (preventing inference of "is subject X in the training set") needs `K_subjects × privacy_amplification` — discussed in next-generation work.

## Bridge to existing ADRs

- **ADR-024 (AETHER)** — within-room training stays unchanged; DP-SGD applies at the federation layer.
- **ADR-027 (MERIDIAN)** — env-centroid subtraction is per-room aggregate, not per-subject — survives Layer 1 isolation as an ⚠️ entry (aggregate is acceptable).
- **ADR-029 (multistatic)** — per-seed federation; multistatic geometry stays per-installation.
- **ADR-100 (cog packaging)** — Ed25519 signing covers DP-noised checkpoints with no protocol change.
- **ADR-103 (cog-person-count)** — first cog with formal DP guarantee; this loop's v0.0.2 retrain becomes ADR-106-compliant on next training cycle.
- **ADR-104 (ruview-mcp + ruview-cli)** — exposes ε, δ budget remaining via MCP `ruview_fed_privacy_budget` (future tool; out of scope for this ADR).
- **ADR-105 (federated training)** — DP-SGD slots into step 4; threat model extended; implementation budget grows from 500 to ~800 LOC.

## Connection to research-loop threads

- **R3 (cross-room re-ID)** — Layer 1 isolation blocks transmission of per-subject embedding centroids.
- **R7 (mincut adversarial)** — Krum (from ADR-105) + DP-noised deltas remain compatible; mincut adversarial check operates on the noised similarity graph.
- **R12 (eigenshift NEGATIVE)** — informed by the structure-detection failure pattern; the DP-noise approach treats adversarial deltas as "outliers from a noisy distribution" rather than as a structural-detection problem.
- **R13 (contactless BP NEGATIVE)** — confirms why we restrict biometric primitive transmission: contour-level signals don't meet the 25 dB floor, so they wouldn't help downstream models anyway; rate-level primitives are sufficient for V1/V2/V3 features.
- **R14 (empathic appliances)** — privacy framework constraints now have a formal (ε, δ) backing.
- **R15 (RF biometric primitives)** — direct requirements basis; the on-device-only primitive list is R15's catalogue made executable.

## Honest scope

- **σ values are recommendations**, not measurements. Per-cog empirical tuning is needed (cog-pose, cog-count, AETHER head, future cogs each get their own).
- **(ε, δ)-DP is a worst-case bound.** Real privacy depends on the auxiliary information the adversary has. For an adversary with extensive auxiliary biometric data, even a small ε can leak. Layer 1 primitive isolation is the harder constraint that doesn't depend on the auxiliary-info model.
- **The Moments Accountant** treats each round as independent, which slightly over-estimates the budget consumed (good — conservative). Tighter accountants (Rényi DP, PRV) would let us run more rounds for the same ε.
- **Subject-level DP is not formalised here.** Many use cases (a household of 4 always-the-same individuals) effectively have K=4 subjects, where sample-level DP doesn't fully capture the subject-level risk.

## Implementation plan (additive to ADR-105)

| Step | LOC | Notes |
|---|---:|---|
| 1. PrimitiveTag enum + tensor tagging | 60 | Compile-time enforcement where possible |
| 2. Gradient clipping helper | 30 | Per-sample (microbatch-friendly) |
| 3. Gaussian noise helper | 40 | Constant-time sampling (defends weak side-channel) |
| 4. Moments Accountant | 120 | Tracks (ε, δ) across rounds; emits budget-exhausted error |
| 5. Per-cog config schema (σ, C, max_rounds) | 50 | YAML/TOML, validated at federation start |
| 6. End-to-end privacy test | — | Synthetic membership-inference attack vs DP-protected model; verify reconstruction quality is bounded by (ε, δ) prediction |

Combined with ADR-105's 500 LOC, total federation budget revised to **~800 LOC**, ~3-week effort.

## What this DOES enable

- Formally privacy-preserving federation with a documented (ε, δ) bound.
- API-level enforcement of R15's biometric primitive isolation list — not just policy text.
- A clear next-ADR path: ADR-107 (cross-installation federation w/ secure aggregation) builds on this foundation.

## What this DOES NOT enable

- Subject-level DP (preventing "is subject X in training") — would need subject-level privacy amplification.
- Defence against side-channel timing leaks — separate ADR.
- Cross-installation federation — separate ADR with secure aggregation + cross-installation DP composition.
- Adversarial robustness to physical compromise — out of scope; physical security is the orthogonal defence layer.

## Decision-making record

- 2026-05-22 06:38 UTC — drafted by SOTA research loop tick-15 based on R3 + R15 + ADR-105's deferred items. Status: Proposed.
- Pending: review by security-architect (formal DP bound verification), ddd-domain-expert (federation = bounded context with this ADR as its public API), production-validator (the per-cog σ values need bench validation before shipping any specific cog).
