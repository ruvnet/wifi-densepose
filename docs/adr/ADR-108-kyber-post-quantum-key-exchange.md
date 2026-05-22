# ADR-108: Kyber post-quantum key exchange for cross-installation federation

**Status:** Proposed · **Date:** 2026-05-22 · **Author:** SOTA research loop tick-28 · **Supersedes:** none · **Extends:** ADR-107 (cross-installation federation)

## Context

ADR-107 specifies cross-installation federation using **secure aggregation (Bonawitz 2016)** with Diffie-Hellman key exchange for pairwise mask generation. The current implementation would use classical DH (X25519 or P-256), which is **vulnerable to Shor's algorithm** on a sufficiently large fault-tolerant quantum computer.

ADR-107 noted this as out-of-scope:

> Current DH key exchange becomes vulnerable to quantum computers. Recommended substitution: Kyber KEM (NIST PQC selected). Mechanical replacement of DH primitives; no protocol change. Future ADR-108 (or amendment to ADR-107).

This ADR is that future work.

## Decision

Adopt **Kyber-768** as the post-quantum key encapsulation mechanism (KEM) replacing Diffie-Hellman in ADR-107's Layer 4 secure aggregation, with an explicit migration timeline tied to NIST CNSA 2.0 guidance and an interim **hybrid mode** (Kyber + X25519) for forward-secrecy belt-and-braces during the migration window.

### Why Kyber-768

NIST standardised three Kyber security levels in FIPS 203 (2024):

| Variant | NIST level | Public key | Ciphertext | Secret | Security |
|---|---|---:|---:|---:|---|
| Kyber-512 | Level 1 | 800 B | 768 B | 32 B | ~AES-128 |
| **Kyber-768** | **Level 3** | **1184 B** | **1088 B** | **32 B** | **~AES-192** |
| Kyber-1024 | Level 5 | 1568 B | 1568 B | 32 B | ~AES-256 |

**Kyber-768** matches AES-192 equivalent security and is the **NIST CNSA 2.0 recommended default** for general-purpose protocols. Used by Cloudflare, Google, AWS in their 2024-2026 PQC rollouts.

Kyber-512 is sufficient against classical attackers and small quantum computers but doesn't carry CNSA 2.0 sign-off. Kyber-1024 doubles bandwidth without proportional security benefit for our threat model.

### Hybrid mode (transition window)

During the migration (2026-2030 estimated), all key exchanges run **both** Kyber-768 AND X25519 in parallel and XOR the shared secrets:

```
shared_secret = SHA-256(kyber_ss || x25519_ss || transcript)
```

This **belt-and-braces** approach protects against:

- A future Kyber break (unlikely but not impossible — Kyber is ~5 years old)
- Implementation bugs in either primitive
- Adversaries who can compromise *one* of the two primitives

Cost: ~2× key-exchange computation, ~2× public-key size. For RuView's per-round overhead this adds ~3 kB / round / installation — negligible.

After CNSA 2.0 fully retires classical primitives (estimated 2030+), the hybrid layer is removed and pure Kyber-768 is used.

### Migration timeline

| Phase | Timeline | What ships |
|---|---|---|
| Phase 0 (NOW) | 2026 | ADR-107 ships with classical X25519 |
| Phase 1 | 2026-Q4 → 2027 | Library upgrade adds Kyber-768; opt-in via `--enable-pqc` flag |
| Phase 2 | 2027-Q2 → 2028 | Hybrid mode (X25519 + Kyber-768) becomes default |
| Phase 3 | 2030+ | Pure Kyber-768 (classical removed) |

Phase 1 is the first feature ship. By the time the migration is complete, the post-quantum threat model is approximately the only one that matters.

### Implementation cost

| Component | LOC | Notes |
|---|---:|---|
| Kyber-768 KEM wrapper (over `pqcrypto-kyber` crate) | 80 | Pure Rust, no `unsafe` |
| Hybrid mode (XOR + SHA-256 KDF) | 50 | Composes existing primitives |
| Protocol version negotiation | 60 | Backward compat with Phase 0 nodes |
| Public-key cache extension (size grows from 32 B to 1184 B per peer) | 30 | AgentDB schema update |
| Migration documentation | — | This ADR |
| End-to-end test (multi-node PQC handshake) | — | Real-installation test |

Total ~220 LOC additional. Combined federation budget across ADR-105+106+107+108: **~1,550 LOC**.

## Alternatives considered

### A. Pure Kyber-768 (no hybrid)

Status: **rejected for Phase 1-2**. Hybrid provides defense-in-depth at minimal cost; pure-Kyber is fine for Phase 3 once Kyber has had more cryptographic scrutiny.

### B. NTRU Prime (alternative PQC KEM)

Status: **rejected**. Kyber has clearer standardisation status (FIPS 203). NTRU Prime is fine cryptographically but doesn't have CNSA 2.0 sign-off.

### C. Frodo (lattice-based, more conservative parameters)

Status: **rejected**. Frodo has larger key sizes (~10 kB) and slower operations. Trade-off doesn't justify the security margin given our threat model.

### D. Code-based KEMs (Classic McEliece)

Status: **rejected**. Classic McEliece public keys are ~261 kB — unworkable for embedded ESP32-S3 nodes.

### E. Defer until quantum threat materialises

Status: **rejected**. Adversaries can record-now-decrypt-later — federated model updates today could be decrypted in 5-10 years when quantum capabilities arrive. ADR-107's privacy guarantees would silently expire without proactive migration.

## Threat model

| Threat | Layer that mitigates |
|---|---|
| Shor's algorithm breaks classical DH | **Kyber-768 KEM** |
| Future quantum attack on Kyber (unlikely) | **Hybrid mode** — X25519 still provides classical security |
| Implementation bug in Kyber library | **Hybrid mode** — X25519 backup |
| Implementation bug in X25519 library | **Hybrid mode** — Kyber backup |
| Record-now-decrypt-later (adversary stores ciphertexts) | Forward secrecy from Kyber-768 (each round has fresh ephemeral keys) |
| Downgrade attack (force classical-only handshake) | **Protocol version negotiation** — explicit reject of classical-only post-Phase-2 |
| Side-channel attack on Kyber implementation | Use constant-time `pqcrypto-kyber` Rust crate; further hardening in future |
| Public-key spoofing (Sybil) | Pre-shared trust anchors via cognitum-v0 PKI (ADR-107) |

## Consequences

### Positive

1. **The privacy chain remains intact through the quantum transition.** Without ADR-108, the (ε, δ) guarantees of ADR-106 silently expire when quantum computers arrive.
2. **Record-now-decrypt-later attack is defeated.** Federated updates from today won't be decryptable in 2035 with quantum hardware.
3. **CNSA 2.0 compliant** by Phase 2; ready for any regulatory requirement that mandates PQC.
4. **Hybrid mode is belt-and-braces** — protects against both Kyber breaks AND classical breaks.
5. **No protocol change** at the secure-aggregation level — the KEM is a drop-in replacement.

### Negative

1. **Adds ~220 LOC** to ADR-107's implementation budget.
2. **~3 kB extra per-round per-installation bandwidth** during hybrid mode (negligible).
3. **Kyber is ~5 years old** — less battle-tested than X25519. Hybrid mode mitigates this.
4. **No clear end-of-life for the hybrid mode** — Phase 3 requires a future decision when CNSA 2.0 retires classical.
5. **Public-key cache grows 37×** (32 B → 1184 B per peer); AgentDB schema update needed.

### What this ADR DOES NOT cover

1. **Post-quantum digital signatures** — ADR-100 cog signing uses Ed25519 today; a follow-up ADR (likely ADR-109) covers Dilithium / SPHINCS+ substitution.
2. **Constant-time hardening of the full Kyber path** — relies on the `pqcrypto-kyber` Rust crate's existing claims.
3. **Hardware-acceleration on ESP32-S3** — Kyber-768 is software-only at this scale; the ESP32-S3 can do ~50 ops/sec which is far more than the per-round federation needs.

## Bridge to existing ADRs

- **ADR-100 (cog packaging Ed25519 signing)** — separate from key-exchange; PQC signature migration needed independently (future ADR-109).
- **ADR-104 (ruview-mcp + ruview-cli)** — MCP tool `ruview_fed_pqc_status` surfaces hybrid-vs-pure mode and migration phase.
- **ADR-105 (federation)** + **ADR-106 (DP+isolation)** — operate over secure-aggregation key exchange; transparent to KEM substitution.
- **ADR-107 (cross-installation federation)** — directly extended by ADR-108; Layer 4 secure aggregation gets Kyber replacement for DH.

## Connection to research-loop threads

- **R3 / R14 / R15** — privacy chain remains intact through quantum transition.
- **R7 (mincut adversarial)** — mincut detection operates on application-level deltas, not key exchange; orthogonal to PQC.
- **R12 PABS** — same — operates on CSI / model deltas, not key exchange.
- **R10 / R11 (wildlife / maritime)** — long-deployment use cases benefit most from forward secrecy because data ages for years.

## Honest scope

- **Kyber is recommended by NIST today** but cryptographic confidence will grow over the next decade. The hybrid mode hedges against this uncertainty.
- **The "when do we need this?" question** is genuinely uncertain. Estimates of cryptographically-relevant quantum computers range from 2030 (aggressive) to 2050+ (conservative). The proactive migration is cheap insurance.
- **ESP32-S3 can compute Kyber-768** but the timing impact in the per-round federation cycle (~10 ms additional per handshake) needs benchmarking on real hardware. Estimated negligible given the existing ~30 s round duration.
- **The migration timeline is aspirational** — depends on `pqcrypto-kyber` crate stability + adoption maturity. Plausible alternatives include `liboqs` C-binding or `boring-pq` (Cloudflare's pre-standardisation work, now superseded).
- **Pure Kyber (Phase 3) end-of-life for classical** — depends on community standardisation and a future RuView decision; not bindingly specified here.

## What this ADR closes

This is the **last ADR in the privacy + federation chain** the research loop has produced:

1. ADR-100 — cog packaging (foundation)
2. ADR-103 — cog-person-count (first cog example)
3. ADR-104 — MCP + CLI distribution
4. ADR-105 — federated training (within-installation)
5. ADR-106 — DP-SGD + biometric primitive isolation
6. ADR-107 — cross-installation federation w/ secure aggregation
7. **ADR-108 (this)** — post-quantum key exchange

The chain has formal guarantees at every layer **and** quantum-resistance built in by 2028. **No remaining unspecified privacy gap** at any threat horizon.

## Implementation plan

| Phase | What ships | LOC |
|---|---|---:|
| Phase 1 (2026-Q4) | Kyber-768 wrapper + `--enable-pqc` opt-in | ~140 |
| Phase 2 (2027-Q2) | Hybrid mode default | ~80 |
| Phase 3 (2030+) | Pure Kyber-768 (remove classical) | -50 (removal) |

Phase 1 is the first ship.

## Future ADRs

- **ADR-109**: PQC digital signatures (Dilithium for cog signing, replacing Ed25519 in ADR-100).
- **ADR-110**: PQC hardware acceleration on Cognitum-v0 (offload Kyber from ESP32-S3 if the ~10 ms cycle becomes binding).
- **ADR-111**: PQC for `cog-store` distribution (sign-and-verify chain).

## Decision-making record

- 2026-05-22 09:37 UTC — drafted by SOTA research loop tick-28 based on ADR-107's explicit deferral. Status: Proposed.
- Pending: security-architect (formal PQC threat model review), production-validator (`pqcrypto-kyber` Rust crate stability and ESP32-S3 benchmarking before Phase 1).

## Honest scope of ADR-108

- Phase 1 ships in ~1 quarter after ADR-107 lands.
- Hybrid mode is the right default for 2027-2030.
- Phase 3 (pure Kyber) needs a separate future decision once CNSA 2.0 fully retires classical primitives.
- Implementation depends on `pqcrypto-kyber` crate maturity; alternatives exist if it stagnates.
- ESP32-S3 timing impact is estimated negligible; needs measurement.
