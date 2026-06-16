---
name: ruview-rvagent
description: Explore and prototype rvAgent + RVF integration for RuView agentic flows. Use when working on cross-cog coordination, operator-facing agents reading BFLD / pose / vitals events live, or persisting agent state alongside sensing data in the same RVF container.
allowed-tools: Read, Grep, Glob, Bash
---

# RuView rvAgent + RVF integration

Surface area for wiring `vendor/ruvector/crates/rvAgent/` into RuView so the existing sensing pipeline becomes the substrate an agentic flow can read, reason about, and respond to.

## Quickstart — published MCP server (`@ruvnet/rvagent` v0.1.0)

Installing this plugin registers `@ruvnet/rvagent` as an MCP server. On activation, Claude Code spawns `npx -y @ruvnet/rvagent` and exposes its tools directly:

| Tool | Purpose |
|------|---------|
| `bfld_last_scan` | Most recent BFLD event from the sensing server |
| `bfld_subscribe` | Stream BFLD events for a window |
| `presence_now` | Current room-level presence state |
| `vitals_get_breathing` | Latest breathing-rate sample |
| `vitals_get_heart_rate` | Latest heart-rate sample |
| `vitals_get_all` | Composite vitals snapshot |
| `vitals_fetch` | Historical vitals window |

Override the sensing-server URL via the `RVAGENT_SENSING_URL` env var (default `http://localhost:3000`). Source lives at `tools/ruview-mcp/`; ADR-124 captures the design.

Smoke-check the wiring: `npm view @ruvnet/rvagent version` should return `0.1.0` (or newer).

## When to use this skill

- "I want an agent that reacts to BFLD presence in the kitchen and pages the carer."
- "I need cog-pose-estimation and cog-bfld to negotiate before publishing a synthesized event."
- "Can the witness chain attest both the sensing event AND the agent decision in one RVF blob?"
- "How do we keep rvAgent's tool outputs class-3 compliant when the source BFLD event is Restricted?"

## Key surfaces

| Surface | File | Notes |
|---------|------|-------|
| rvAgent core | `vendor/ruvector/crates/rvAgent/rvagent-core/src/agi_container.rs` (627 LOC) | RVF-compatible state container |
| rvAgent middleware | `vendor/ruvector/crates/rvAgent/rvagent-middleware/` | Witness, sanitizer, SONA, HNSW |
| Agent personas | `vendor/ruvector/crates/rvAgent/.ruv/agents/rvagent-{queen,coder,tester,security}.md` | Reference patterns |
| RVF container | `v2/crates/wifi-densepose-sensing-server/src/rvf_container.rs` | Add `SEG_AGENT_STATE`, `SEG_DECISION` |
| BFLD event | `v2/crates/wifi-densepose-bfld/src/event.rs` | `BfldEvent::to_json()` → `ToolOutput` |
| BFLD pipeline handle | `v2/crates/wifi-densepose-bfld/src/pipeline_handle.rs` | `BfldPipelineHandle::send` |

## Research dossier

Full integration analysis lives at `docs/research/rvagent-rvf-integration/README.md`.

Three shippable touchpoints, each independent:

1. **RVF wire**: two new segment types (`SEG_AGENT_STATE = 0x08`, `SEG_DECISION = 0x09`) let rvAgent sessions interleave with RuView sensing sessions in the same blob.
2. **Tool surface**: `BfldEvent → ToolOutput` shim turns BFLD events into agent context with no new IPC.
3. **Cog subagents**: `cog-pose-estimation` / `cog-person-count` / `cog-ha-matter` / `cog-bfld` register as rvAgent subagents under a queen-agent router.

## Open questions

- Workspace inclusion of `vendor/ruvector/crates/rvAgent/` (path dep vs published crate)
- Sync ↔ async adapter (BFLD `Publish` is sync, rvAgent backends are tokio)
- Privacy-class composition (does rvAgent's sanitizer consume `PrivacyClass`?)
- Soul Signature ↔ `SoulMatchOracle` bridge
- Whether `BfldPipelineHandle::send` lands as a public MCP tool via `rvagent-mcp`

## Next decision

ADR-124 (proposed) — "rvAgent + RVF integration for RuView agentic flows" — would capture segment assignments, cog-subagent contract, and the privacy-class composition rule. Land before scaffolding `v2/crates/wifi-densepose-agent`.
