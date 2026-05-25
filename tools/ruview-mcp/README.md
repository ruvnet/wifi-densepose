# @ruvnet/rvagent — SENSE-BRIDGE MCP Server

**SENSE-BRIDGE** is a dual-transport [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that bridges the RuView WiFi-DensePose sensing stack to AI agents (Claude Code, Cursor, ruflo swarms, and any MCP-compatible client).

Install once; AI agents can then call `ruview.presence.now`, `ruview.vitals.get_heart_rate`, `ruview.bfld.last_scan`, and more — without writing HTTP or WebSocket client code.

## Quickstart

```bash
# 1. Add to Claude Code
claude mcp add rvagent -- npx @ruvnet/rvagent stdio

# 2. Or run directly
RUVIEW_SENSING_SERVER_URL=http://cognitum-v0:3000 npx @ruvnet/rvagent stdio

# 3. Streamable HTTP (remote agents, ruflo swarms)
RUVIEW_SENSING_SERVER_URL=http://cognitum-v0:3000 \
RVAGENT_HTTP_TOKEN=your-secret \
npx @ruvnet/rvagent http --port 3001
# POST JSON-RPC to http://127.0.0.1:3001/mcp
```

Requirements: **Node.js >= 20**. The `wifi-densepose-sensing-server` Rust binary must be reachable at `RUVIEW_SENSING_SERVER_URL` (default `http://localhost:3000`).

## Feature matrix

| Tool | Description | ADR |
|------|-------------|-----|
| `ruview.presence.now` | Current occupancy: `present`, `n_persons`, `confidence` | ADR-124 §4.1 |
| `ruview.vitals.get_breathing` | Breathing rate bpm (null if unavailable) | ADR-124 §4.1 |
| `ruview.vitals.get_heart_rate` | Heart rate bpm (null if unavailable) | ADR-124 §4.1 |
| `ruview.vitals.get_all` | Full `EdgeVitalsMessage` surface | ADR-124 §4.1 |
| `ruview.bfld.last_scan` | Latest BFLD scan: `identity_risk_score`, `privacy_class`, `n_frames` | ADR-118/124 |
| `ruview.bfld.subscribe` | Subscribe to `ruview/<node_id>/bfld/*` events for `duration_s` seconds | ADR-122/124 |
| *(next iters)* | `pose.latest`, `primitives.*`, `node.*`, `vector.*`, `policy.*` | ADR-124 §4.1/4.1a |

**Transport security (ADR-124 §6)**:
- **stdio**: process-level isolation — no auth needed for local Claude Code / Cursor.
- **Streamable HTTP** (`POST /mcp`): Origin header validation (cross-origin → 403), optional bearer token (`RVAGENT_HTTP_TOKEN` → 401 on mismatch), binds `127.0.0.1` by default per MCP spec.

**Schema validation**: every tool call runs `zod.safeParse` before dispatch; invalid arguments return `McpError(InvalidParams)` rather than a wrapped string.

**Policy layer** (ADR-124 §4.1a): `ruview.policy.*` tools gate every sensing call — `vitals.*` is default-deny until a policy grant is registered via `npx @ruvnet/rvagent policy grant`. Presence and node-list are allow by default.

## ADR cross-reference

| ADR | Decision |
|-----|----------|
| [ADR-124](../../docs/adr/ADR-124-rvagent-mcp-ruvector-npm-integration.md) | SENSE-BRIDGE: dual-transport MCP server + ruvector npm + ruflo integration |
| [ADR-118](../../docs/adr/ADR-118-bfld-beamforming-feedback-layer-for-detection.md) | BFLD pipeline — source of `bfld.last_scan` wire format |
| [ADR-122](../../docs/adr/ADR-122-bfld-ruview-ha-matter-exposure.md) | MQTT topic routing `ruview/<node_id>/bfld/*` |
| [ADR-115](../../docs/adr/ADR-115-home-assistant-integration.md) | `EdgeVitalsMessage` WebSocket surface (`ws.py:74-88` parity) |
| [ADR-055](../../docs/adr/ADR-055-integrated-sensing-server.md) | Sensing-server REST API (`/api/v1/*`) |

## Development

```bash
cd tools/ruview-mcp
npm install
npm run build   # tsc
npm test        # jest — 93 tests across 7 suites
```

Source: `tools/ruview-mcp/src/`. Tests: `tools/ruview-mcp/tests/`.
Tracking issue: [#787](https://github.com/ruvnet/RuView/issues/787).
