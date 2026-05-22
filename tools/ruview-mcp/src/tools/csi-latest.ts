/**
 * MCP tool: ruview_csi_latest
 *
 * Pull the most recent CSI window from the local sensing-server.
 * Wraps GET /api/v1/sensing/latest (ADR-102 endpoint, schema version 2).
 *
 * Returns the full CsiWindow JSON so the calling agent can inspect raw
 * subcarrier data, feed it to ruview_pose_infer, or store it for analysis.
 */

import { z } from "zod";
import type { RuviewConfig, SensingLatestResponse } from "../types.js";
import { sensingGet } from "../http.js";
import { validateSensingLatestResponse } from "../validate.js";

export const csiLatestSchema = z.object({
  /** Override the sensing-server URL for this call only. */
  sensing_server_url: z
    .string()
    .url()
    .optional()
    .describe(
      "Base URL of the sensing-server (default: RUVIEW_SENSING_SERVER_URL or http://localhost:3000)"
    ),
});

export type CsiLatestInput = z.infer<typeof csiLatestSchema>;

export async function csiLatest(
  input: CsiLatestInput,
  config: RuviewConfig
): Promise<object> {
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;

  const result = await sensingGet<SensingLatestResponse>(
    baseUrl,
    "/api/v1/sensing/latest",
    config.apiToken
  );

  if (!result.ok) {
    return {
      ok: false,
      warn: true,
      error: result.error,
      hint:
        "Ensure the wifi-densepose-sensing-server is running. " +
        "Start it with `cargo run -p wifi-densepose-sensing-server` or " +
        "set RUVIEW_SENSING_SERVER_URL to the correct address.",
    };
  }

  const validation = validateSensingLatestResponse(result.data);
  if (!validation.valid) {
    return {
      ok: false,
      warn: true,
      error: `Sensing-server response failed schema validation: ${validation.errors.join("; ")}`,
      raw_response: result.data,
      hint:
        "The sensing-server may have upgraded its schema. " +
        "Check schema_version in the raw_response and update " +
        "ruview-mcp/src/types.ts if needed.",
    };
  }

  return {
    ok: true,
    ts: result.data.window.ts,
    schema_version: result.data.schema_version,
    captured_at: result.data.captured_at,
    n_paths: result.data.window.n_paths,
    node_mac: result.data.window.node_mac,
    subcarriers: result.data.window.amplitudes.length,
    frames: result.data.window.amplitudes[0]?.length ?? 0,
    window: result.data.window,
  };
}
