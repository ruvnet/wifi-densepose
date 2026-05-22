# Horizon: 12-hour Autonomous SOTA Run — 2026-05-22

**Horizon ID:** `sota-2026-05-22`
**Started:** 2026-05-21 ~20:00 ET
**Auto-stop:** 2026-05-22 08:00 ET
**Cron:** `d6e5c473` (`*/10 * * * *`) — single-tick research contributions running in parallel

---

## Three concurrent objectives

| Objective | Description | Primary branch |
|-----------|-------------|---------------|
| **A** | Keep the cron research loop productive — curate PROGRESS.md between ticks | (main, via PR) |
| **B** | Build `ruview` MCP server + CLI (`tools/ruview-mcp/`, `tools/ruview-cli/`) | `feat/ruview-mcp-cli` |
| **C** | Write ADR-104: ruview MCP/CLI distribution decision record | (same branch as B) |

---

## Milestones

### M1 — Scaffold `tools/ruview-mcp/` + `tools/ruview-cli/`
**Target:** +1h (by ~21:00 ET)
**Status:** `COMPLETE` — merged as PR #705 (squash commit `5a6c585aa`)
**Branch:** `feat/ruview-mcp-cli-pr` (deleted after merge)

Deliverables:
- `tools/ruview-mcp/package.json` — `@ruv/ruview-mcp`, TypeScript, `@modelcontextprotocol/sdk`
- `tools/ruview-mcp/src/index.ts` — minimal MCP server with 5 tool stubs
- `tools/ruview-mcp/src/tools/` — one file per tool
- `tools/ruview-cli/package.json` — `@ruv/ruview-cli` + `ruview` bin
- `tools/ruview-cli/src/index.ts` — 4-verb CLI stub via yargs/commander
- `tsconfig.json` for both packages
- Shared `tools/ruview-shared/` for HTTP client + types

Completion criteria: `npm run build` succeeds in both packages, MCP server can be registered with `claude mcp add`.

---

### M2 — Wire `ruview_pose_infer` + `ruview_count_infer`
**Target:** +3h (by ~23:00 ET)
**Status:** `in_progress`

Wire inference via subprocess to cog binaries (`cog-pose-estimation`, `cog-person-count`). MCP tools and CLI subcommands both delegate to the cog binary's `health` + a synthetic-frame run.

Completion criteria: `ruview_pose_infer` returns finite keypoint array; `ruview_count_infer` returns `{count, confidence}`.

---

### M3 — Wire `ruview_csi_latest` + `ruview_registry_list`
**Target:** +5h (by ~01:00 ET)
**Status:** `pending`

Connect to sensing-server `/api/v1/sensing/latest` (ADR-102 endpoint) and `/api/v1/edge/registry`. CLI: `npx ruview csi tail` streams live frames.

Completion criteria: both tools return structured JSON from a running sensing-server (or graceful 503 WARN if server not reachable).

---

### M4 — Wire `ruview_train_count`
**Target:** +7h (by ~03:00 ET)
**Status:** `pending`

Fire the Candle training pipeline as a background subprocess; return a job ID; expose `ruview_job_status` to poll. Training output streamed to `~/.ruview/jobs/<id>.log`.

Completion criteria: `ruview_train_count` returns `{job_id, status: "queued"}` within 200 ms.

---

### M5 — ADR-104: ruview MCP/CLI distribution
**Target:** +8h (by ~04:00 ET)
**Status:** `pending`

Full ADR covering: problem, design (5 MCP tools + 5 CLI subcommands + library mapping), security (6-row threat table), packaging (npm `@ruv/ruview-mcp` + `@ruv/ruview-cli`), distribution, failure modes, acceptance gates.

Completion criteria: ADR file at `docs/adr/ADR-104-ruview-mcp-cli-distribution.md`, merged to main.

---

### M6 — Integration tests
**Target:** +10h (by ~06:00 ET)
**Status:** `pending`

Jest/Vitest tests: spawn MCP server, call each tool stub, assert structured output shape. CI-green on Node 20.

Completion criteria: `npm test` passes in `tools/ruview-mcp/`.

---

### M7 — Final summary + handoff
**Target:** +11h (by ~07:00 ET)
**Status:** `pending`

Write final section to this HORIZON.md: what shipped, what deferred, exact `npm publish` commands.

---

## Cron coordination (Objective A)

The `d6e5c473` cron picks threads from `PROGRESS.md` independently. Rules for safe co-operation:
- Horizon-tracker writes to HORIZON.md, not PROGRESS.md, except for cross-link notes.
- When a cron tick lands a new artifact, horizon-tracker distills its finding into PROGRESS.md's "Done" section + adds cross-links (e.g. R5 → R8 RSSI feasibility).
- If a thread shows 2+ consecutive ticks without a new artifact, horizon-tracker adds `blocked: <reason>` to that thread's section.

Current cross-links identified at session start:
- **R5 → R8**: band-spread top-8 saliency distribution raises RSSI-only ceiling to ~60% of full-CSI upper-bound.
- **R5 → R7**: top-8 subcarriers are exactly the ones a defender must corroborate across nodes.
- **R5 → R1**: saliency map should be re-run on multi-static captures (different geometry = different salient subcarriers?).

---

## Drift indicators (checked each milestone)

| Indicator | Threshold | Current |
|-----------|-----------|---------|
| Timeline | M1 >2h behind → defer scope | On track |
| Scope | MCP server grows beyond 5 tools | On track |
| Approach | MCP SDK incompatible with available node | TBD at M1 |
| Dependency | ruvector npm packages not findable | TBD at M1 |
| Priority | Cron consuming PROGRESS.md locks | None yet |

---

## Session log

### Session 1 — 2026-05-21 (horizon init + M1)

**Started:** Initial read of PROGRESS.md, ADR-100/101/102/103, R5 saliency note.
**Accomplished:**
- HORIZON.md initialized.
- `tools/ruview-mcp/` and `tools/ruview-cli/` scaffolded with TypeScript, MCP SDK, Yargs.
- 6 MCP tools defined (stubs): csi_latest, pose_infer, count_infer, registry_list, train_count, job_status.
- 6 CLI subcommands defined: csi tail, pose infer, count infer, cogs list, train count, job status.
- `docs/adr/ADR-104-ruview-mcp-cli-distribution.md` written (full depth, 6-row threat table).
- 6/6 smoke tests pass.
- PR #705 created and merged.
- PROGRESS.md updated: R7 and R8 cross-links added (cron produced these results in parallel).
**Cron activity observed:** R7 (Stoer-Wagner adversarial detection 3/3) + R8 (RSSI-only 94.82% retained) landed while M1 was in progress.
**Next:** M2 — wire real inference via sensing-server + cog subprocess.
