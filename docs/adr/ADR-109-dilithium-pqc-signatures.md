# ADR-109: Dilithium post-quantum digital signatures for cog distribution

**Status:** Proposed · **Date:** 2026-05-22 · **Author:** SOTA research loop tick-30 · **Extends:** ADR-100 (cog packaging Ed25519 signing) · **Sister-of:** ADR-108 (Kyber post-quantum key exchange)

## Context

ADR-100 specified Ed25519 signatures for cog packaging (binaries on GCS at `gs://cognitum-apps/cogs/{arm,x86_64}/`, signed with `COGNITUM_OWNER_SIGNING_KEY`). ADR-108 closed the **key exchange** side of post-quantum migration with Kyber-768. This ADR closes the **digital signature** side with Dilithium-3.

The two pieces are independent — DH/Kyber protects confidentiality (federation updates), Ed25519/Dilithium protects integrity (signed cog binaries, ADR-100 distribution). Both need PQC migration on similar timelines to keep the privacy + provenance chain quantum-resistant.

ADR-108 cited:

> ADR-109: PQC signatures (Dilithium for cog signing, replacing Ed25519 in ADR-100).

This is that work.

## Decision

Adopt **Dilithium-3** as the post-quantum signature scheme replacing Ed25519 in ADR-100's cog signing pipeline. Use the same migration pattern as ADR-108: **hybrid mode (Ed25519 + Dilithium-3)** during the transition window (2026-2030); pure Dilithium-3 afterwards.

### Why Dilithium-3

NIST standardised three Dilithium security levels in FIPS 204 (2024):

| Variant | NIST level | Public key | Signature | Security |
|---|---|---:|---:|---|
| Dilithium-2 | Level 2 | 1,312 B | 2,420 B | ~AES-128 |
| **Dilithium-3** | **Level 3** | **1,952 B** | **3,293 B** | **~AES-192** |
| Dilithium-5 | Level 5 | 2,592 B | 4,595 B | ~AES-256 |

**Dilithium-3** at NIST Level 3 matches AES-192 equivalent security, mirroring our Kyber-768 choice from ADR-108. This is the NIST CNSA 2.0 recommended default for general signing.

### Hybrid mode (transition window)

Sign **both** with Ed25519 AND Dilithium-3 during the migration. Manifest format:

```json
{
  "cog_name": "cog-person-count",
  "version": "0.0.2",
  "sha256": "...",
  "signatures": {
    "ed25519": "...",  // ADR-100 classical
    "dilithium3": "..." // ADR-109 PQC
  },
  "sig_policy": "BOTH_REQUIRED_PHASE_2"
}
```

Verification policy by phase:

| Phase | Verification |
|---|---|
| Phase 0 (NOW 2026) | Ed25519 only (ADR-100 baseline) |
| Phase 1 (2026-Q4 → 2027) | Ed25519 required + Dilithium-3 emitted (best-effort verify) |
| Phase 2 (2027-Q2 → 2028) | **BOTH required** — defence in depth |
| Phase 3 (2030+) | Dilithium-3 required, Ed25519 deprecated/removed |

### Migration timeline (matches ADR-108)

| Phase | Timeline | What ships |
|---|---|---|
| Phase 0 | 2026 | ADR-100 ships with Ed25519 only |
| Phase 1 | 2026-Q4 → 2027 | Cog signer produces both signatures; verifier accepts either |
| Phase 2 | 2027-Q2 → 2028 | Both signatures required; downgrade to single signature rejected |
| Phase 3 | 2030+ | Pure Dilithium-3, Ed25519 removed |

### Implementation cost

| Component | LOC | Notes |
|---|---:|---|
| Dilithium-3 signer (over `pqcrypto-dilithium` Rust crate) | 90 | Pure Rust, no `unsafe` |
| Manifest schema extension (multi-sig field + policy) | 60 | Backward-compatible JSON additive |
| Verifier with phase-aware policy enforcement | 80 | Tied to manifest `sig_policy` |
| GCS bucket policy update (allow new key types) | — | Operational, not code |
| `cogd` daemon: re-sign existing cogs in dual-sig | 40 | One-time backfill script |
| End-to-end test (install signed cog on Pi cluster) | — | Real-installation test |

Total ~270 LOC additional. Combined federation + signing budget across ADR-100 + ADR-105 + ADR-106 + ADR-107 + ADR-108 + ADR-109: **~1,820 LOC**.

## Alternatives considered

### A. SPHINCS+ (hash-based signatures)

Status: **deferred to ADR-110 if needed**. SPHINCS+ is conservatively-secure (worst-case based on hash function security only) but has much larger signatures (~17-50 kB) and slower signing. For cog distribution where keys rarely change, Dilithium-3's 3.3 kB signatures are the better trade-off. SPHINCS+ might be a fallback if Dilithium suffers a cryptanalytic break.

### B. Falcon (lattice signatures with smaller footprint)

Status: **considered**. Falcon-512 has smaller signatures (666 B) than Dilithium-3 (3,293 B) but slower signing and more complex implementation (floating-point Gaussian sampling). Dilithium-3 is the safer choice given the Rust crate maturity (`pqcrypto-dilithium` vs `pqcrypto-falcon`).

### C. Pure Dilithium-3 (no hybrid)

Status: **rejected for Phase 1-2**. Same belt-and-braces reasoning as ADR-108: Dilithium is ~5 years old; hybrid hedges against breaks.

### D. Defer until quantum threat materialises

Status: **rejected**. Same record-now-decrypt-later argument as ADR-108, applied to signatures: an adversary who can break Ed25519 in 2035 can backdate signatures on cog binaries to install malicious code retroactively. Provenance chain breaks.

## Threat model

| Threat | Mitigation |
|---|---|
| Shor's algorithm breaks Ed25519 | Dilithium-3 signature |
| Future quantum break on Dilithium-3 (unlikely) | Hybrid mode — Ed25519 still classical-secure |
| Implementation bug in Dilithium library | Hybrid mode — Ed25519 backup |
| Implementation bug in Ed25519 library | Hybrid mode — Dilithium backup |
| Backdated signature attack (quantum-era forgery on old binaries) | **Hybrid mode is essential** — Ed25519 forgery is hard even for quantum (no key compromise), so quantum + Ed25519 = still requires breaking Dilithium |
| Compromised owner key (operational) | Out of scope — key management ADR (future) |
| Downgrade attack (force single-sig acceptance post-Phase-2) | **Manifest `sig_policy` field** enforces required signatures |

## Consequences

### Positive

1. **Provenance chain stays intact through quantum transition.** Without ADR-109, the integrity of installed cog binaries silently expires when quantum computers arrive.
2. **Backdating attack defeated.** An adversary in 2035 cannot forge a Dilithium-3 signature on a 2026 cog binary even with quantum hardware.
3. **CNSA 2.0 compliant** by Phase 2.
4. **Hybrid mode is belt-and-braces** — protects against breaks in either primitive.
5. **No protocol change** — multi-signature manifest is a standard JSON additive pattern.

### Negative

1. **Adds ~270 LOC** to ADR-100's signing implementation.
2. **Manifest size grows**: Ed25519 (64 B sig) + Dilithium-3 (3,293 B sig) = ~3.4 kB total. Per-cog manifest overhead is now ~4 kB. Across 50 cogs in the catalogue, ~200 kB extra. Negligible.
3. **Signer needs both keys**: classical + PQC keypairs. Adds key-management complexity.
4. **Dilithium-3 verifier latency**: ~0.5-1 ms vs Ed25519's ~30 µs. On ESP32-S3 with no hardware acceleration, ~5-10 ms per verification. For occasional cog-install events, fine.
5. **Pure Dilithium retirement of Ed25519 needs future decision** (Phase 3, post-2030).

### What this ADR DOES NOT cover

1. **PQC for HTTPS / TLS** to the cog distribution servers — Cloudflare / GCS run their own PQC migration on their schedule.
2. **Owner key rotation policy** — separate future ADR.
3. **Hardware acceleration for Dilithium verification on ESP32-S3** — if 5-10 ms latency becomes binding, offload to cognitum-v0 fleet manager.
4. **Cross-signing with external CA** — if RuView ever needs a third-party CA chain, that's a future ADR.

## Bridge to existing ADRs

- **ADR-100 (cog packaging Ed25519 signing)** — directly extended; Ed25519 stays in hybrid mode.
- **ADR-104 (ruview-mcp + ruview-cli)** — `ruview_cog_install` MCP tool gains signature-policy parameter.
- **ADR-105 / ADR-106 / ADR-107 / ADR-108** — federation operates on signed cog binaries; ADR-109 ensures the signing layer is quantum-resistant in lockstep with ADR-108's key exchange.

## Connection to research-loop threads

- **R14 / R15** — privacy + biometric framework requires provenance integrity; ADR-109 ensures cog updates are tamper-proof against quantum adversaries.
- **R12 PABS / R12.1 (security feature)** — intruder-detection cog must itself be signed; the cog can't trust its own model weights if the signing chain is broken.
- **R10 / R11 (long-deployment wildlife / maritime)** — most affected by backdating attacks because installed cogs sit on edge nodes for years.
- **R7 (mincut adversarial)** — adversarial detection assumes the model itself is trustworthy. ADR-109 protects that assumption.

## Honest scope

- **Dilithium is ~5 years old** but has had substantial NIST scrutiny. Hybrid mitigates uncertainty.
- **5-10 ms verification on ESP32-S3** is estimated, not measured. Needs benchmarking on the COM5 device.
- **Migration depends on `pqcrypto-dilithium` Rust crate maturity** — alternatives include `liboqs` C-binding.
- **Owner key management** (storing the Dilithium signing key in gcloud secrets) is the highest-risk operational change. Compromise of the signing key is unrecoverable; no quantum-resistance argument can fix that.
- **Phase 3 retirement** of Ed25519 needs a future decision once CNSA 2.0 fully retires classical signatures.

## What this ADR closes

The **provenance side** of the post-quantum migration. Combined with ADR-108 (key exchange), RuView's full cryptographic chain is quantum-resistant by Phase 2 (2027-2028).

ADR chain after this tick:

| # | ADR | What it closes |
|---|---|---|
| 1 | ADR-100 | cog packaging |
| 2 | ADR-103 | cog-person-count |
| 3 | ADR-104 | MCP + CLI |
| 4 | ADR-105 | within-installation federation |
| 5 | ADR-106 | DP-SGD + primitive isolation |
| 6 | ADR-107 | cross-installation + SA |
| 7 | ADR-108 | PQC key exchange (Kyber) |
| 8 | **ADR-109 (this)** | **PQC signatures (Dilithium)** |

**The cryptographic chain is now complete** for both confidentiality (ADR-108) and integrity (ADR-109) at the quantum-resistant tier.

## Future ADRs (catalogued)

- **ADR-110**: PQC hardware acceleration on Cognitum-v0 (if ESP32-S3 Dilithium verification latency becomes binding).
- **ADR-111**: Owner key rotation policy (operational, key compromise recovery).
- **ADR-112**: Cross-signing with external CA (if third-party trust needed).
- **ADR-113**: Multistatic placement strategy (formalises the R6 family findings into an architectural specification — would amend ADR-029).

## Implementation plan

| Phase | What ships | LOC |
|---|---|---:|
| Phase 1 (2026-Q4) | Dilithium-3 signer + dual-sig manifest, verifier accepts either | ~170 |
| Phase 2 (2027-Q2) | Both signatures required; downgrade rejected | ~70 |
| Phase 3 (2030+) | Pure Dilithium-3, Ed25519 removed | -30 (removal) |

Phase 1 ships ~1 quarter after ADR-108 lands.

## Decision-making record

- 2026-05-22 09:56 UTC — drafted by SOTA research loop tick-30, sister-ADR to ADR-108. Status: Proposed.
- Pending: security-architect (Dilithium implementation review), production-validator (`pqcrypto-dilithium` Rust crate stability + ESP32-S3 verification benchmark).

## Closing observation

ADR-109 closes the **last predictable cryptographic gap** in the RuView privacy + provenance chain. The remaining unspecified items (owner key management, cross-signing, hardware acceleration) are operational or contingent on specific future requirements; the architectural foundation is now complete.

Combined federation + signing implementation budget: **~1,820 LOC**, ~7-week effort across the full chain (ADR-105 → ADR-109). This is the engineering cost of shipping privacy-preserving + quantum-resistant federated RuView.
