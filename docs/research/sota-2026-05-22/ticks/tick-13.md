# Tick 13 — 2026-05-22 06:13 UTC

**Thread:** R4 (federated learning)
**Verdict:** ADR-105 drafted. Federated CSI training is the unique design that satisfies R14 (data-stays-on-device) + R3 (no cross-installation linkage) + R7 (multi-node adversarial defence) simultaneously.

## What shipped

- `docs/adr/ADR-105-federated-csi-training.md` — full ADR draft covering protocol, threat model, bandwidth analysis, alternatives, implementation plan.

This tick chose the "one ADR" unit option from the cron prompt rather than another numpy demo — federation is fundamentally a protocol-design problem, not a numerical-experiment problem. Architectural decisions are the right unit when the question is "what's the right shape of the thing" not "what number does it give".

## Headline protocol

**MERIDIAN-FedAvg with Byzantine-robust (Krum) aggregation + R7 mincut update-level consistency.**

Per-round bandwidth (4-seed installation):
- Coordinator → nodes (multicast): 8 MB checkpoint
- Each node → coordinator: 1 MB delta (LoRA-rank-8 + int8 quantisation)
- Total per round: ~12 MB
- Weekly × monthly = ~50-180 MB/month/installation (0.06% of typical broadband cap)

## Why ADR-105 not another numpy demo

R3 (last tick) said: "re-ID is the primitive that makes empathic appliances ship". R4 says: "federation is the protocol that makes re-ID training privacy-compliant." Together they trace the full pipeline from physics (R6) → embeddings (R3) → personalised features (R14) → trained how (R4) → defended how (R7).

The protocol is the deliverable. ADR-105 specifies it; ruview-fed crate implementation (~500 LOC) is the next-quarter work.

## Composes with every prior thread

- **R3** — MERIDIAN env centroid subtraction is **mandatory** pre-aggregation step.
- **R7** — Stoer-Wagner mincut extended from multi-link CSI to multi-node update consistency.
- **R12 / R13** — two negative results informed the byzantine-robust + SNR-threshold-on-updates choices.
- **R14** — privacy framework's "data stays on-device" baseline is now operational.
- **ADR-024 (AETHER), ADR-027 (MERIDIAN), ADR-029 (multistatic), ADR-100 (cog packaging), ADR-103 (cog-person-count), ADR-104 (MCP+CLI)** — all referenced in the ADR's "bridge to existing ADRs" section.

## Honest scope landed

- Cross-installation federation explicitly **deferred** to a future ADR (legal + DP work needed)
- Member inference defence → ADR-106 with formal DP-SGD
- The 500 LOC + 2-week-effort estimates assume AgentDB / microlora / mincut crates are stable
- Krum byzantine bound: f < (K-2)/2 — practical f ≤ 4 for typical RuView installs

## Coordination

`ticks/tick-13.md`. No PROGRESS.md edit. Branch `research/sota-r4-federated-adr105`.

## Remaining threads

R15 (RF biometric across rooms) — now largely subsumed by R3 + ADR-105 cross-installation deferral. Could write a short "scoping note" for R15 in next tick to close the loop, or pick up the deferred items: physics-informed env_sig prediction (next R3 follow-up), or ADR-106 (DP-SGD on local training).

~5.7h to cron stop. 13 threads landed (2 negative results, 1 ADR, 10 research notes with demos).
