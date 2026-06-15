# ruview-rvagent — explore rvAgent + RVF agentic flows for RuView

You are helping the operator explore or prototype the integration of `vendor/ruvector/crates/rvAgent/` (a production Rust AI-agent framework) with RuView's existing sensing pipeline (`v2/crates/wifi-densepose-*`) and the RVF cognitive container format (`v2/crates/wifi-densepose-sensing-server/src/rvf_container.rs`).

## Live MCP server: `@ruvnet/rvagent` v0.1.0

The TypeScript MCP server (`tools/ruview-mcp/`, published as `@ruvnet/rvagent`) is live on npm and exposes `bfld_last_scan`, `bfld_subscribe`, `presence_now`, `vitals_get_breathing`, `vitals_get_heart_rate`, `vitals_get_all`, `vitals_fetch`. Add to a Codex MCP config:

```json
{
  "mcpServers": {
    "rvagent": {
      "command": "npx",
      "args": ["-y", "@ruvnet/rvagent"],
      "env": { "RVAGENT_SENSING_URL": "http://localhost:3000" }
    }
  }
}
```

This is the operator-facing tool surface; the Rust crate below remains the substrate for deeper RVF-aware agentic flows.

## Trigger phrasing

- "wire rvAgent into RuView"
- "I want a queen agent that fans out to cog-pose-estimation and cog-bfld"
- "persist agent decisions in the same witness bundle as sensing events"
- "how do I keep agent outputs class-3 compliant?"

## What to read first

1. `docs/research/rvagent-rvf-integration/README.md` — full integration thesis, open questions, next steps.
2. `vendor/ruvector/crates/rvAgent/README.md` — what rvAgent ships (8 crates, 14 middlewares).
3. `vendor/ruvector/crates/rvAgent/.ruv/agents/rvagent-queen.md` — queen-agent persona that coordinates cog subagents.
4. `v2/crates/wifi-densepose-bfld/src/{event.rs,pipeline_handle.rs}` — the BFLD event surface and the operator-facing handle that an agent would call.
5. `v2/crates/wifi-densepose-sensing-server/src/rvf_container.rs` — segment types; `SEG_AGENT_STATE = 0x08` and `SEG_DECISION = 0x09` are the proposed additions.

## Three shippable touchpoints (each independent)

1. **RVF wire** — add `SEG_AGENT_STATE` + `SEG_DECISION` segments so rvAgent and RuView sessions can interleave in one blob (witness-bundle covers both halves).
2. **Tool shim** — `BfldEvent::to_json()` already exists; wrap as `rvagent_tools::ToolOutput`.
3. **Cog subagents** — register `cog-pose-estimation`, `cog-person-count`, `cog-ha-matter`, (proposed) `cog-bfld` under the queen via the `Subagent` trait.

## Open questions to surface

- Is `vendor/ruvector/crates/rvAgent/` on the v2 workspace path?
- Sync ↔ async adapter location (BFLD `Publish` is sync; rvAgent backends are tokio).
- Privacy-class composition — does `rvagent-middleware::sanitizer` consume `BfldEvent::privacy_class`?
- Soul Signature ↔ `SoulMatchOracle` bridge (ADR-121 §2.6).
- Should `BfldPipelineHandle::send` land as a public MCP tool via `rvagent-mcp`?

## Suggested next action

Draft ADR-124 — "rvAgent + RVF integration for RuView agentic flows" — capturing segment assignments, cog-subagent contract, and privacy-class composition. Land **before** scaffolding `v2/crates/wifi-densepose-agent`.
