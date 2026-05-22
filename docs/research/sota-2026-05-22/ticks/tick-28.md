# Tick 28 — 2026-05-22 09:40 UTC

**Thread:** ADR-108 (Kyber post-quantum key exchange)
**Verdict:** Final ADR in the privacy + federation chain. Closes the quantum-resistance gap deferred from ADR-107. Hybrid mode (Kyber-768 + X25519) for 2027-2030 migration; pure Kyber-768 for Phase 3.

## What shipped

- `docs/adr/ADR-108-kyber-post-quantum-key-exchange.md` — full ADR draft.

## Headline

| Phase | Timeline | Cryptography |
|---|---|---|
| Phase 0 | NOW (2026) | Classical X25519 (ADR-107 default) |
| Phase 1 | 2026-Q4 → 2027 | Kyber-768 opt-in via `--enable-pqc` |
| Phase 2 | 2027-Q2 → 2028 | Hybrid (X25519 + Kyber-768) becomes default |
| Phase 3 | 2030+ | Pure Kyber-768 (classical retired) |

**Why Kyber-768**: NIST FIPS 203 (2024); ~AES-192 equivalent; CNSA 2.0 default; used by Cloudflare/Google/AWS in 2024-2026 rollouts.

**Why hybrid for Phase 2**: belt-and-braces against future Kyber breaks (Kyber is ~5 years old) OR classical breaks OR implementation bugs in either primitive.

## Why now (the record-now-decrypt-later argument)

Adversaries can record federated updates today and decrypt them in 2035 when quantum capabilities arrive. Without ADR-108, the (ε, δ) guarantees of ADR-106 **silently expire** when quantum computers arrive.

## Bandwidth + LOC budgets

Bandwidth: ~3 kB/round/installation extra during hybrid mode (negligible).

LOC: +220 on top of ADR-107.

**Total federation budget across ADR-105+106+107+108**: ~1,550 LOC.

## ADR chain closes

Final ADR in the privacy + federation chain:

| # | ADR | What it closes |
|---|---|---|
| 1 | ADR-100 | cog packaging (foundation) |
| 2 | ADR-103 | first cog example (cog-person-count) |
| 3 | ADR-104 | MCP + CLI distribution |
| 4 | ADR-105 | within-installation federation |
| 5 | ADR-106 | DP-SGD + biometric primitive isolation |
| 6 | ADR-107 | cross-installation + secure aggregation |
| 7 | **ADR-108** | **post-quantum key exchange** |

**No remaining unspecified privacy gap** at any threat horizon (classical OR quantum).

## Composes with prior threads

- R3 / R14 / R15 / R7 / R12 PABS — privacy chain intact through quantum transition
- R10 / R11 (long-deployment wildlife / maritime) — benefit most from forward secrecy because data ages for years

## Honest scope

- Kyber is ~5 years old (less battle-tested than X25519); hybrid mode mitigates
- "When do we need this?" is uncertain (2030 aggressive / 2050+ conservative); proactive migration is cheap insurance
- ESP32-S3 timing impact (~10 ms per handshake) estimated negligible vs 30 s round duration; needs benchmarking
- Migration timeline depends on `pqcrypto-kyber` Rust crate maturity
- Phase 3 retirement of classical needs future decision

## Future ADRs catalogued

- **ADR-109**: PQC signatures (Dilithium for cog signing, replaces Ed25519 in ADR-100)
- **ADR-110**: PQC hardware acceleration on Cognitum-v0 if timing becomes binding
- **ADR-111**: PQC for `cog-store` distribution chain

## Coordination

`ticks/tick-28.md`. No PROGRESS.md edit. Branch `research/sota-adr108-kyber`.

## Remaining loop work

- R12.1: pose-PABS closed loop (needs Rust, out of scope for synthetic ticks)
- Loop retrospective / 00-summary.md (~2.3h until cron stop — premature)

~2.3h to cron stop. **28 ticks landed.** 4 ADRs in the privacy chain (105/106/107/108). Loop covers everything except R12.1 implementation.
