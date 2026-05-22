# Tick 30 — 2026-05-22 10:01 UTC

**Thread:** ADR-109 (Dilithium PQC signatures for cog distribution)
**Verdict:** Sister-ADR to ADR-108. Closes the **provenance side** of post-quantum migration. Combined chain (ADR-100 + ADR-105–109) now fully quantum-resistant for both confidentiality and integrity by Phase 2 (2027-2028).

## What shipped

- `docs/adr/ADR-109-dilithium-pqc-signatures.md` — full ADR draft.

## Headline

Replaces Ed25519 in ADR-100 cog signing with **Dilithium-3** (NIST FIPS 204, ~AES-192 equivalent, CNSA 2.0 default).

Migration timeline (matches ADR-108):

| Phase | Timeline | Cog signing |
|---|---|---|
| Phase 0 | NOW (2026) | Ed25519 only (ADR-100 baseline) |
| Phase 1 | 2026-Q4 → 2027 | Dual-sig (Ed25519 + Dilithium-3), accepts either |
| Phase 2 | 2027-Q2 → 2028 | **BOTH required** — defence in depth |
| Phase 3 | 2030+ | Pure Dilithium-3 |

## Why now (backdating argument)

An adversary who can break Ed25519 in 2035 (with quantum computers) can **backdate** signatures on cog binaries to install malicious code retroactively. The provenance chain breaks even for binaries deployed today. Hybrid mode prevents this: forging a 2026 cog signature still requires breaking BOTH Ed25519 AND Dilithium-3.

## Bandwidth + LOC

Manifest size: 64 B (Ed25519) + 3,293 B (Dilithium-3) = ~4 kB per cog. Catalogue overhead ~200 kB across 50 cogs. Negligible.

LOC: +270 on top of ADR-100. Combined chain budget: **~1,820 LOC**.

## ADR chain after this tick (8 ADRs)

| # | ADR | Closes |
|---|---|---|
| 1 | ADR-100 | cog packaging |
| 2 | ADR-103 | cog-person-count |
| 3 | ADR-104 | MCP + CLI |
| 4 | ADR-105 | within-install federation |
| 5 | ADR-106 | DP-SGD + primitive isolation |
| 6 | ADR-107 | cross-install + SA |
| 7 | ADR-108 | PQC key exchange (Kyber) |
| 8 | **ADR-109** | **PQC signatures (Dilithium)** |

**Cryptographic chain complete** for both confidentiality (ADR-108) and integrity (ADR-109) at quantum-resistant tier.

## Future ADRs catalogued

- **ADR-110**: PQC hardware acceleration on Cognitum-v0
- **ADR-111**: Owner key rotation policy
- **ADR-112**: Cross-signing with external CA
- **ADR-113**: Multistatic placement strategy (formalises R6 family findings, would amend ADR-029)

## Composes with prior threads

- R14 / R15 privacy + biometric framework requires provenance integrity
- R12 PABS / R12.1 security feature: intruder-detection cog must itself be signed
- R10 / R11 long-deployment cogs most affected by backdating attacks
- R7 mincut adversarial assumes the model itself is trustworthy

## Honest scope

- Dilithium ~5 years old; hybrid mitigates uncertainty
- ESP32-S3 verification latency ~5-10 ms estimated; needs benchmarking
- `pqcrypto-dilithium` Rust crate dependency
- Owner key management is highest-risk operational change (compromise unrecoverable)
- Phase 3 Ed25519 retirement needs future decision

## Coordination

`ticks/tick-30.md`. No PROGRESS.md edit. Branch `research/sota-adr109-dilithium-signatures`.

## Loop's cryptographic + privacy story complete

5 ADRs (105-109) define the full federated learning + privacy + quantum-resistance chain:
- ADR-105: within-installation federation
- ADR-106: differential privacy + biometric isolation
- ADR-107: cross-installation + secure aggregation
- ADR-108: PQC key exchange (Kyber-768)
- **ADR-109**: PQC signatures (Dilithium-3)

Combined ~1,820 LOC, ~7-week engineering. This is what shipping privacy-preserving + quantum-resistant federated RuView costs.

~1.9h to cron stop.
