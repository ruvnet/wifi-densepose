# @ruvnet/rvagent — SENSE-BRIDGE MCP Server

**SENSE-BRIDGE** is a dual-transport [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that bridges the RuView WiFi-DensePose sensing stack to AI agents (Claude Code, Cursor, ruflo swarms, and any MCP-compatible client).

Install once; AI agents can then call `ruview_presence_now`, `ruview_vitals_get_heart_rate`, `ruview_bfld_last_scan`, and more — without writing HTTP or WebSocket client code.

## Quickstart

```bash
# 1. Add to Claude Code (stdio transport — the default)
claude mcp add rvagent -- npx -y @ruvnet/rvagent

# 2. Or run directly
RUVIEW_SENSING_SERVER_URL=http://cognitum-v0:3000 npx @ruvnet/rvagent

# 3. Streamable HTTP (remote agents, ruflo swarms) — explicit opt-in
RUVIEW_SENSING_SERVER_URL=http://cognitum-v0:3000 \
RVAGENT_HTTP_TOKEN=your-secret \
RVAGENT_HTTP_PORT=3001 npx @ruvnet/rvagent
# POST JSON-RPC to http://127.0.0.1:3001/mcp (initialize first; then send the
# returned mcp-session-id header on every request)
```

Requirements: **Node.js >= 20**. The `wifi-densepose-sensing-server` Rust binary must be reachable at `RUVIEW_SENSING_SERVER_URL` (default `http://localhost:3000`).

## Tools

Canonical tool names are underscore-form (ADR-264 — host tool-name validators
commonly enforce `^[a-zA-Z0-9_-]{1,64}$`). The pre-0.1.1 dotted names
(`ruview.presence.now`, …) are still accepted at call time as deprecated
aliases; `tools/list` advertises the underscore form only.

| Tool | Description | ADR |
|------|-------------|-----|
| `ruview_csi_latest` | Latest 56×20 CSI window from the sensing-server | ADR-101/102 |
| `ruview_pose_infer` | Single-shot 17-keypoint pose inference via cog binary | ADR-101 |
| `ruview_count_infer` | Single-shot person-count inference via cog binary | ADR-103 |
| `ruview_registry_list` | Cognitum edge module registry (category/search filters) | ADR-102 |
| `ruview_train_count` | Kick off a count-cog training run (background job) | ADR-103 |
| `ruview_job_status` | Poll a training job (persists across server restarts) | ADR-103 |
| `ruview_presence_now` | Current occupancy: `present`, `n_persons`, `confidence` | ADR-124 §4.1 |
| `ruview_vitals_get_breathing` | Breathing rate bpm (null if unavailable) | ADR-124 §4.1 |
| `ruview_vitals_get_heart_rate` | Heart rate bpm (null if unavailable) | ADR-124 §4.1 |
| `ruview_vitals_get_all` | Full `EdgeVitalsMessage` surface | ADR-124 §4.1 |
| `ruview_bfld_last_scan` | Latest BFLD scan: `identity_risk_score`, `privacy_class`, `n_frames` | ADR-118/124 |
| `ruview_bfld_subscribe` | Subscribe to `ruview/<node_id>/bfld/*` events for `duration_s` seconds | ADR-122/124 |
| *(roadmap, ADR-124 §4.1/4.1a)* | `pose.latest`, `primitives.*`, `node.*`, `vector.*`, and the `policy.*` governance layer are catalogued in `src/schemas/` but **not yet implemented** | ADR-124 |

**Transport security (ADR-124 §6, hardened per ADR-264)**:
- **stdio** (default): process-level isolation — no auth needed for local Claude Code / Cursor.
- **Streamable HTTP** (`/mcp`, opt-in via `RVAGENT_HTTP_PORT`): one transport + one MCP server per session (routed by `mcp-session-id`), Origin validation (localhost on any port allowed; anything else → 403), optional bearer token (`RVAGENT_HTTP_TOKEN` → 401 on mismatch), 1 MiB request-body cap (413), binds `127.0.0.1` by default per MCP spec.

**Schema validation**: each tool declares one Zod schema; the CallTool gate parses exactly once and the advertised JSON Schema is generated from the same Zod source. Invalid arguments return `McpError(InvalidParams)` rather than a wrapped string.

## ADR cross-reference

| ADR | Decision |
|-----|----------|
| [ADR-124](../../docs/adr/ADR-124-rvagent-mcp-ruvector-npm-integration.md) | SENSE-BRIDGE: dual-transport MCP server + ruvector npm + ruflo integration |
| [ADR-264](../../docs/adr/ADR-264-rvagent-mcp-and-cli-npm-deep-review.md) | npm deep review — exports fix, map-free tarball, naming, session-per-transport |
| [ADR-118](../../docs/adr/ADR-118-bfld-beamforming-feedback-layer-for-detection.md) | BFLD pipeline — source of `bfld_last_scan` wire format |
| [ADR-122](../../docs/adr/ADR-122-bfld-ruview-ha-matter-exposure.md) | MQTT topic routing `ruview/<node_id>/bfld/*` |
| [ADR-115](../../docs/adr/ADR-115-home-assistant-integration.md) | `EdgeVitalsMessage` WebSocket surface (`ws.py:74-88` parity) |
| [ADR-055](../../docs/adr/ADR-055-integrated-sensing-server.md) | Sensing-server REST API (`/api/v1/*`) |

## Development

```bash
cd tools/ruview-mcp
npm install
npm run build   # tsc
npm test        # jest — 99 tests across 7 suites
```

Source: `tools/ruview-mcp/src/`. Tests: `tools/ruview-mcp/tests/`.
Tracking issue: [#787](https://github.com/ruvnet/RuView/issues/787).
