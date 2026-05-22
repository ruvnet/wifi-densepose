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

const PACKAGE_VERSION = "0.0.1";
const SERVER_NAME = "ruview";

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

  // Call tool handler.
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
    `[ruview-mcp] Server v${PACKAGE_VERSION} started. ` +
      `Sensing server: ${config.sensingServerUrl}\n`
  );
}

main().catch((e) => {
  process.stderr.write(`[ruview-mcp] Fatal: ${String(e)}\n`);
  process.exit(1);
});
