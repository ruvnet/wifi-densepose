# ADR-105: Federated learning for RuView CSI personalization

**Status:** Proposed · **Date:** 2026-05-22 · **Author:** SOTA research loop tick-13 · **Supersedes:** none

## Context

RuView's per-occupant features (R14 empathic appliances, R3 cross-room re-ID, R8 per-person counting) require **personalised models** that learn the household's specific subjects, motion patterns, and environmental quirks. Personalisation requires training data, but the privacy framework from R14 + R3 explicitly forbids sending raw CSI off-device:

1. R14 — *data stays on-device; only aggregate state passes integration boundaries*
2. R3 — *no cross-installation linkage of embeddings*

These constraints rule out centralised training on user CSI. The standard answer is **federated learning** (McMahan 2017): each device trains locally; only model deltas (gradients or weight updates) leave the device.

CSI has three properties that change the standard FedAvg recipe:

1. **Non-IID data.** Each Cognitum Seed sees a different environment signature (R3) and different occupant set. Naive FedAvg drifts toward the most-represented environment.
2. **High-bandwidth raw data.** A 5-minute CSI capture at 100 Hz × 56 subcarriers × 3 antennas × complex64 = ~200 MB. Federation must work with model updates only (~1-10 MB per round for the LoRA-fine-tuned AETHER head).
3. **Adversarial node risk.** A compromised seed can poison the global model via crafted updates. R7's mincut multi-link adversarial detection extends to update-level voting.

This ADR specifies the federation protocol.

## Decision

Adopt **MERIDIAN-FedAvg with byzantine-robust aggregation** as the RuView federated training protocol.

### Protocol summary

1. **Round initiation.** Coordinator (cognitum-v0 fleet manager) selects K healthy nodes for round T, sends global model checkpoint W_T.
2. **Local training.** Each node N_i loads W_T, fine-tunes its AETHER head on its local data for `local_epochs` epochs. Local data is **never** transmitted off-device.
3. **MERIDIAN normalisation.** Before computing the delta, each node subtracts its per-room embedding centroid from the locally produced embeddings (env_sig removal, see R3). This makes deltas environment-agnostic.
4. **Delta compression.** Compute ΔW_i = W_T+1_i − W_T. Quantise to int8 + LoRA-rank decomposition (rank=8) → ~1 MB per delta.
5. **Byzantine-robust aggregation.** Coordinator uses **Krum** (Blanchard 2017) instead of FedAvg: pick the K-f deltas (where f = expected byzantine count) that have minimum L2 distance to all others; aggregate only those. Cuts off outliers that suggest poisoning.
6. **Multi-link consistency check (R7 extension).** Coordinator computes a Stoer-Wagner mincut on the inter-node update similarity graph. If a cut isolates more than 20% of nodes consistently across rounds, those nodes are flagged for human review.
7. **Global update.** W_T+1 = W_T + lr_global · Krum_aggregate(ΔW_i).
8. **Convergence check.** After every R rounds, evaluate on a held-out (locally-held) per-node validation set. Federation stops when held-out accuracy plateaus.

### Update frequency

| Cog | Suggested federation frequency | Reason |
|---|---|---|
| `cog-person-count` (R8/R5 work) | Weekly | Counting model is well-trained; only need updates when household composition shifts |
| AETHER re-ID head (R3) | Daily | Re-ID drifts with seasonal multipath changes |
| `cog-pose-estimation` | Monthly | Base pose is stable; finetune only for new room geometries |
| `cog-maritime-watch` (R11) | Per-vessel-deployment | Vessel motion regimes vary; ship-specific fine-tune |

### Bandwidth analysis

Per round (typical RuView 4-seed installation):

| Phase | Bytes per node | Total |
|---|---:|---:|
| Coordinator → node: global checkpoint | 8 MB | 4 × 8 = 32 MB (multicast: 8 MB) |
| Local training (no transmission) | 0 | 0 |
| Node → coordinator: int8+LoRA delta | 1 MB | 4 × 1 = **4 MB** |
| Aggregation + push: new global checkpoint | 8 MB | 8 MB |
| **Total per round** | ~ 5 MB / node | **~12-44 MB** |

At weekly cadence × 4-week month, that's ~50-180 MB / month / installation. **Well under** typical home broadband caps (300 GB/month standard cap = 0.06% of bandwidth budget).

### Required SDK / infrastructure

- **AgentDB hierarchical store** (already in repo) — per-node embedding centroid storage.
- **ruvllm-microlora** (already in repo) — LoRA-rank decomposition of deltas.
- **cognitum-fleet** service on cognitum-v0 (port 9002, see CLAUDE.local.md) — coordinator role.
- **NEW: `ruview-fed` crate** — protocol implementation, ~500 lines Rust, library only (no daemon).

## Alternatives considered

### A. Centralised training on user CSI

Status: **rejected**. Violates R14 (data stays on-device) and R3 (no cross-installation linkage).

### B. FedAvg without byzantine-robust aggregation

Status: **rejected**. A single compromised seed can shift the global model arbitrarily. R7 mincut adversarial work showed this is a real attack surface; Krum (or any byzantine-robust replacement) is required.

### C. Federation across installations (not just within)

Status: **deferred to a future ADR**. Cross-installation federation requires:
- Cryptographic embedding-space alignment (so that "person A in install X" and "person A in install Y" have unifiable signatures)
- Stronger consent framework (cross-installation = legal-entity boundary per R3)
- Differential privacy guarantees on deltas

A worked design needs ~6 person-months of legal + crypto work. Not in scope for this ADR.

### D. Pure on-device per-installation training (no federation)

Status: **alternative path for small deployments**. A single-seed installation has no peers to federate with. Use on-device-only fine-tune of pre-trained base model. The federation protocol gracefully degrades to "no federation = local training only".

## Threat model

| Threat | Mitigation (within this ADR) |
|---|---|
| Compromised seed poisons global model | Krum aggregation + mincut consistency check (R7) |
| Coordinator (cognitum-v0) compromised | Multi-coordinator fallback; signed model checkpoints (Ed25519, ADR-100 pattern) |
| Eavesdropper recovers training data from deltas | LoRA rank-8 + int8 quantisation is information-theoretically lossy; differential privacy noise (σ=0.01) on deltas if higher assurance needed |
| Adversarial training signal injection (via crafted CSI) | R7 multi-link consistency (across antennas in same seed) catches this; federated mincut adds inter-seed consistency layer |
| Member inference attack on the trained model | LoRA + DP-SGD on local training, see future ADR-106 for the formal DP budget |

## Consequences

### Positive

1. RuView personalisation becomes possible **without** violating R14/R3 privacy constraints.
2. Bandwidth budget is trivially affordable (~50-180 MB/month/installation).
3. R7 mincut extends naturally to update-level federation defence.
4. The protocol is **graceful** — single-seed installations get local-only training; multi-seed installations get federation; no code path differences for the cog implementation.
5. **Independent of cog**: this ADR specifies the protocol, individual cogs implement local training using their own model architecture. `cog-pose`, `cog-count`, AETHER head, future cogs all use the same federation surface.

### Negative

1. Adds ~500 lines of new Rust code (the `ruview-fed` crate).
2. Krum is O(K²) in nodes — fine for K ≤ 50 (typical RuView installation), expensive for K > 1000 (not a target).
3. Adds a coordinator dependency — cognitum-v0 fleet manager becomes a federation bottleneck. The multi-coordinator-fallback mitigation adds complexity.
4. Cross-installation federation **explicitly deferred** to a future ADR — small installations stay isolated for now.
5. Doesn't address member inference attacks; ADR-106 needed for that.

### Bridge to existing ADRs

- **ADR-024 (AETHER):** within-room embedding training stays unchanged; federation just shares the head weights.
- **ADR-027 (MERIDIAN):** the env-centroid subtraction is now a **mandatory** pre-aggregation step, not just an evaluation-time trick.
- **ADR-029 (multistatic):** federation per-seed; multistatic geometry remains a per-installation property and is not federated.
- **ADR-100 (cog packaging):** federation operates on cog binaries; the Ed25519 signing infrastructure from ADR-100 covers checkpoint integrity.
- **ADR-103 (cog-person-count):** the v0.0.2 retrained model from this loop's earlier work would be the first cog to use the federation protocol — once `ruview-fed` ships.
- **ADR-104 (ruview-mcp + ruview-cli):** federation status surfaces as MCP tools (`ruview_fed_status`, `ruview_fed_pause`) — out of scope for this ADR but in the natural MCP roadmap.

## Implementation plan

| Step | Owner | LOC | Notes |
|---|---|---:|---|
| 1. `ruview-fed` crate scaffold | TBD | 100 | Workspace member, no external deps initially |
| 2. Krum aggregator | TBD | 80 | Pure Rust, no GPU |
| 3. LoRA+int8 delta codec | TBD | 120 | Reuse ruvllm-microlora |
| 4. MERIDIAN centroid hook | TBD | 50 | Extend AgentDB hierarchical store |
| 5. Inter-seed mincut consistency | TBD | 100 | Reuse ruvector-mincut |
| 6. CLI surface (`wifi-densepose-cli fed status / fed pause`) | TBD | 80 | Add to existing CLI |
| 7. End-to-end test on 4-seed cognitum-cluster (the Pi+Hailo fleet from CLAUDE.local.md) | TBD | — | Real-hardware test |

Total ~500 lines + tests. A reasonable 2-week effort once `ruview-fed` is unblocked.

## What this DOES NOT cover

1. **Cross-installation federation** — deferred to a future ADR (legal + DP work).
2. **Member inference defence** — ADR-106 will cover formal DP-SGD on local training.
3. **Cog-specific training-loop details** — each cog implements its own `local_train()`; ADR-105 only specifies the wire format and aggregation rules.
4. **Compute scheduling** — when training runs, how it shares hardware with inference, etc. Cognitum fleet manager territory.

## Negative results we built on

This ADR's threat model and update-level mincut design are direct outputs of the loop's two negative results:

- **R12 (eigenshift)** — naive structure-detection failed; informed the byzantine-robust aggregation choice (don't trust outlier updates).
- **R13 (contactless BP)** — physics-floor scrutiny pattern applied here to update-level threats (compute SNR for poisoning detection).

## Connection back to research-loop threads

- **R3 (cross-room re-ID):** MERIDIAN normalisation requirement is direct.
- **R7 (mincut adversarial):** Stoer-Wagner mincut extends from multi-link CSI consistency to multi-node update consistency.
- **R8 / R5:** first cog to use the federation protocol once `ruview-fed` ships.
- **R11 (maritime):** per-vessel-deployment fine-tune cadence accommodated.
- **R14 (empathic appliances):** privacy framework's "data stays on-device" baseline is now operational.

## Decision-making record

- 2026-05-22 06:13 UTC — drafted by SOTA research loop tick-13 based on R3 + R7 + R14 + R6 synthesis. Status: Proposed.
- Pending: review by security-architect, ddd-domain-expert (federation = bounded context), production-validator (the 500 LOC budget claim needs sanity check).

## Honest scope of this ADR

- The bandwidth numbers assume LoRA rank-8 + int8 quantisation. Real implementations may need higher rank for AETHER to converge, increasing bandwidth by 4-8×. Still well within home broadband.
- Krum is byzantine-robust against `f < (K-2)/2` byzantine nodes. For K=4, that means 1 byzantine; for K=10, 4. RuView installations rarely have K>10 seeds, so the practical bound is ~4 byzantine.
- The "1-2 weeks of effort" claim for implementation assumes the existing AgentDB + ruvllm-microlora + ruvector-mincut crates are stable. If any of those need rework, the federation work blocks behind that.
