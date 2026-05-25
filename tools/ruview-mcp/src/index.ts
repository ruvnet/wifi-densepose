#!/usr/bin/env node
/**
 * @ruv/ruview-mcp — RuView MCP Server
 *
 * Exposes RuView's WiFi-DensePose sensing capabilities as Model Context Protocol
 * (MCP) tools that Claude Code, Cursor, Codex, and other MCP-compatible agents
 * can call directly.
 *
 * Tools exposed:
 *   ruview_csi_latest    — pull the latest CSI window from the sensing-server
 *   ruview_pose_infer    — single-shot 17-keypoint pose estimation
 *   ruview_count_infer   — single-shot person count with confidence interval
 *   ruview_registry_list — list cogs from the Cognitum edge registry (ADR-102)
 *   ruview_train_count   — kick off a count-cog training run (returns job ID)
 *   ruview_job_status    — poll a background training job
 *
 * Usage:
 *   node dist/index.js                   # stdio transport (default)
 *   RUVIEW_SENSING_SERVER_URL=http://cognitum-v0:3000 node dist/index.js
 *
 * To register with Claude Code:
 *   claude mcp add ruview -- node /path/to/tools/ruview-mcp/dist/index.js
 *
 * See ADR-104 for the full design rationale and security model.
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  McpError,
  ErrorCode,
} from "@modelcontextprotocol/sdk/types.js";

import { loadConfig } from "./config.js";
import { csiLatestSchema, csiLatest } from "./tools/csi-latest.js";
import { poseInferSchema, poseInfer } from "./tools/pose-infer.js";
import { countInferSchema, countInfer } from "./tools/count-infer.js";
import { registryListSchema, registryList } from "./tools/registry-list.js";
import {
  trainCountSchema,
  trainCount,
  jobStatusSchema,
  jobStatus,
} from "./tools/train-count.js";
import { TOOL_INPUT_SCHEMAS } from "./schemas/index.js";
import { bfldLastScan } from "./tools/bfld-last-scan.js";
import { bfldSubscribe } from "./tools/bfld-subscribe.js";
import { presenceNow } from "./tools/presence-now.js";
import { vitalsGetBreathing } from "./tools/vitals-get-breathing.js";
import { vitalsGetHeartRate } from "./tools/vitals-get-heart-rate.js";
import { vitalsGetAll } from "./tools/vitals-get-all.js";

const PACKAGE_VERSION = "0.1.0";
const SERVER_NAME = "rvagent";

// ── Tool registry ──────────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "ruview_csi_latest",
    description:
      "Pull the latest CSI window from a running wifi-densepose-sensing-server. " +
      "Returns 56-subcarrier × 20-frame amplitude/phase arrays suitable for " +
      "downstream inference or research analysis.",
    inputSchema: {
      type: "object" as const,
      properties: {
        sensing_server_url: {
          type: "string",
          description:
            "Base URL of the sensing-server (default: RUVIEW_SENSING_SERVER_URL or http://localhost:3000).",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      const input = csiLatestSchema.parse(args);
      return csiLatest(input, config);
    },
  },
  {
    name: "ruview_pose_infer",
    description:
      "Run a single-shot 17-keypoint COCO pose estimation inference using the " +
      "cog-pose-estimation Cog binary (ADR-101). Accepts a CSI window JSON file " +
      "or uses the live sensing-server if no window is provided. " +
      "Returns [{keypoints: [[x,y]×17], confidence}] per detected person.",
    inputSchema: {
      type: "object" as const,
      properties: {
        window_path: {
          type: "string",
          description: "Path to a CSI window JSON file. Omit to use the live sensing-server.",
        },
        cog_binary: {
          type: "string",
          description: "Path to cog-pose-estimation binary.",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      const input = poseInferSchema.parse(args);
      return poseInfer(input, config);
    },
  },
  {
    name: "ruview_count_infer",
    description:
      "Run a single-shot person-count inference using the cog-person-count Cog " +
      "binary (ADR-103). Returns {count, confidence, count_p95_low, count_p95_high} " +
      "with a Stoer-Wagner multi-node fusion upper bound when multiple nodes are active.",
    inputSchema: {
      type: "object" as const,
      properties: {
        window_path: {
          type: "string",
          description: "Path to a CSI window JSON file. Omit to use the live sensing-server.",
        },
        cog_binary: {
          type: "string",
          description: "Path to cog-person-count binary.",
        },
        max_persons: {
          type: "integer",
          minimum: 1,
          maximum: 7,
          description: "Upper bound on person count (1–7). Default: 7.",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      const input = countInferSchema.parse(args);
      return countInfer(input, config);
    },
  },
  {
    name: "ruview_registry_list",
    description:
      "List cogs from the Cognitum edge module registry (ADR-102). " +
      "Fetches /api/v1/edge/registry from the sensing-server, which proxies the " +
      "canonical GCS catalog (105 cogs, 11 categories). Supports category filter and search.",
    inputSchema: {
      type: "object" as const,
      properties: {
        category: {
          type: "string",
          description:
            "Filter by category: health, security, building, retail, industrial, " +
            "research, ai, swarm, signal, network, developer.",
        },
        search: {
          type: "string",
          description: "Search substring matched against cog id and name (case-insensitive).",
        },
        refresh: {
          type: "boolean",
          description: "Bypass the 1-hour registry cache.",
        },
        sensing_server_url: {
          type: "string",
          description: "Override the sensing-server URL.",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      const input = registryListSchema.parse(args);
      return registryList(input, config);
    },
  },
  {
    name: "ruview_train_count",
    description:
      "Kick off a cog-person-count training run using the Candle GPU trainer " +
      "(ADR-103). The paired JSONL file provides CSI windows + camera-derived " +
      "person-count labels. Returns a job_id to poll with ruview_job_status.",
    inputSchema: {
      type: "object" as const,
      required: ["paired_jsonl"],
      properties: {
        paired_jsonl: {
          type: "string",
          description:
            "Path to the paired JSONL training file (produced by scripts/align-ground-truth.js).",
        },
        epochs: {
          type: "integer",
          minimum: 1,
          maximum: 10000,
          description: "Training epochs (default: 400).",
        },
        learning_rate: {
          type: "number",
          description: "Initial learning rate (default: 0.001).",
        },
        output_dir: {
          type: "string",
          description:
            "Directory for model artifacts (default: v2/crates/cog-person-count/cog/artifacts/).",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      const input = trainCountSchema.parse(args);
      return trainCount(input, config);
    },
  },
  {
    name: "ruview_job_status",
    description:
      "Poll the status of a background training job started by ruview_train_count. " +
      "Returns {status, epochs_done, epochs_total, recent_log} for the given job_id.",
    inputSchema: {
      type: "object" as const,
      required: ["job_id"],
      properties: {
        job_id: {
          type: "string",
          description: "UUID returned by ruview_train_count.",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      const input = jobStatusSchema.parse(args);
      return jobStatus(input, config);
    },
  },
  // ── ADR-124 BFLD tools (Phase 4 Refinement) ──────────────────────────────
  {
    name: "ruview.bfld.last_scan",
    description:
      "Return the most recent BFLD scan result for a node (ADR-118/ADR-121). " +
      "Fields: node_id, identity_risk_score [0,1], privacy_class, n_frames, timestamp_ms. " +
      "Proxied from sensing-server GET /api/v1/bfld/<node_id>/last_scan which aggregates " +
      "the MQTT state topics ruview/<node_id>/bfld/* (ADR-122 §2.2).",
    inputSchema: {
      type: "object" as const,
      properties: {
        node_id: {
          type: "string",
          description: "Target node id. Omit to use the single active node.",
        },
        sensing_server_url: {
          type: "string",
          description: "Override sensing-server URL for this call only.",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      return bfldLastScan(args as Parameters<typeof bfldLastScan>[0], config);
    },
  },
  {
    name: "ruview.bfld.subscribe",
    description:
      "Subscribe to BFLD events on ruview/<node_id>/bfld/* for duration_s seconds (ADR-122). " +
      "Returns {ok, subscription_id, expires_at, topic}. When the sensing-server is unreachable, " +
      "returns a synthetic envelope with ok:false,warn:true so the caller can distinguish " +
      "a network error from an invalid request.",
    inputSchema: {
      type: "object" as const,
      required: ["duration_s"],
      properties: {
        node_id: {
          type: "string",
          description: "Target node id. Omit to use the single active node.",
        },
        duration_s: {
          type: "number",
          minimum: 0,
          maximum: 3600,
          description: "Subscription duration in seconds (max 3600).",
        },
        sensing_server_url: {
          type: "string",
          description: "Override sensing-server URL for this call only.",
        },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) => {
      return bfldSubscribe(args as Parameters<typeof bfldSubscribe>[0], config);
    },
  },
  // ── ADR-124 Presence + Vitals tools (Phase 4 Refinement iter 5) ──────────
  {
    name: "ruview.presence.now",
    description:
      "Return current occupancy for a node: present, n_persons, confidence, timestamp_ms. " +
      "Wraps EdgeVitalsMessage.presence + n_persons (ADR-124 §4.1, ws.py:74-88).",
    inputSchema: {
      type: "object" as const,
      properties: {
        node_id: { type: "string", description: "Target node id." },
        sensing_server_url: { type: "string", description: "Override sensing-server URL." },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) =>
      presenceNow(args as Parameters<typeof presenceNow>[0], config),
  },
  {
    name: "ruview.vitals.get_breathing",
    description:
      "Return breathing rate for a node: breathing_rate_bpm (null if unavailable), " +
      "confidence, timestamp_ms. Wraps EdgeVitalsMessage.breathing_rate_bpm (ws.py:82).",
    inputSchema: {
      type: "object" as const,
      properties: {
        node_id: { type: "string", description: "Target node id." },
        window_s: { type: "number", description: "Averaging window in seconds (max 300)." },
        sensing_server_url: { type: "string", description: "Override sensing-server URL." },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) =>
      vitalsGetBreathing(args as Parameters<typeof vitalsGetBreathing>[0], config),
  },
  {
    name: "ruview.vitals.get_heart_rate",
    description:
      "Return heart rate for a node: heartrate_bpm (null if unavailable), " +
      "confidence, timestamp_ms. Wraps EdgeVitalsMessage.heartrate_bpm (ws.py:83).",
    inputSchema: {
      type: "object" as const,
      properties: {
        node_id: { type: "string", description: "Target node id." },
        window_s: { type: "number", description: "Averaging window in seconds (max 300)." },
        sensing_server_url: { type: "string", description: "Override sensing-server URL." },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) =>
      vitalsGetHeartRate(args as Parameters<typeof vitalsGetHeartRate>[0], config),
  },
  {
    name: "ruview.vitals.get_all",
    description:
      "Return the full EdgeVitalsMessage for a node (all fields except raw): " +
      "presence, n_persons, confidence, breathing_rate_bpm, heartrate_bpm, motion, zone_id. " +
      "Full surface of ws.py:74-88.",
    inputSchema: {
      type: "object" as const,
      properties: {
        node_id: { type: "string", description: "Target node id." },
        sensing_server_url: { type: "string", description: "Override sensing-server URL." },
      },
    },
    handler: async (args: unknown, config: ReturnType<typeof loadConfig>) =>
      vitalsGetAll(args as Parameters<typeof vitalsGetAll>[0], config),
  },
] as const;

// ── Server bootstrap ────────────────────────────────────────────────────────

async function main(): Promise<void> {
  const config = loadConfig();

  const server = new Server(
    {
      name: SERVER_NAME,
      version: PACKAGE_VERSION,
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List tools handler.
  server.setRequestHandler(ListToolsRequestSchema, () => ({
    tools: TOOLS.map((t) => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema,
    })),
  }));

  // Call tool handler — uniform Zod validation gate (ADR-124 §3 Architecture).
  // If TOOL_INPUT_SCHEMAS has a schema for the tool name, run safeParse first.
  // Parse failures throw McpError(InvalidParams) so the client sees a typed
  // JSON-RPC error rather than a wrapped string error.
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    const tool = TOOLS.find((t) => t.name === name);

    if (!tool) {
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              ok: false,
              error: `Unknown tool "${name}". Available tools: ${TOOLS.map((t) => t.name).join(", ")}`,
            }),
          },
        ],
        isError: true,
      };
    }

    // Schema validation gate — applies to all tools registered in TOOL_INPUT_SCHEMAS.
    const schemaEntry = Object.prototype.hasOwnProperty.call(TOOL_INPUT_SCHEMAS, name)
      ? TOOL_INPUT_SCHEMAS[name as keyof typeof TOOL_INPUT_SCHEMAS]
      : undefined;
    if (schemaEntry !== undefined) {
      const parsed = schemaEntry.safeParse(args ?? {});
      if (!parsed.success) {
        throw new McpError(
          ErrorCode.InvalidParams,
          `Invalid arguments for tool "${name}": ${parsed.error.message}`
        );
      }
    }

    try {
      const result = await tool.handler(args ?? {}, config);
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (e: unknown) {
      if (e instanceof McpError) throw e; // propagate typed errors unchanged
      const message = e instanceof Error ? e.message : String(e);
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              ok: false,
              error: message,
            }),
          },
        ],
        isError: true,
      };
    }
  });

  // Wire up stdio transport.
  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Log to stderr so it doesn't interfere with the MCP stdio protocol.
  process.stderr.write(
    `[@ruvnet/rvagent] Server v${PACKAGE_VERSION} started. ` +
      `Sensing server: ${config.sensingServerUrl}\n`
  );
}

main().catch((e) => {
  process.stderr.write(`[ruview-mcp] Fatal: ${String(e)}\n`);
  process.exit(1);
});
