/**
 * MCP tool: ruview.bfld.subscribe
 *
 * Registers interest in BFLD events for `duration_s` seconds by instructing
 * the sensing-server to forward MQTT messages from topic
 * `ruview/<node_id>/bfld/*` (ADR-122 §2.2) to a server-side event buffer.
 *
 * This is a stateless stub that does NOT require a running MQTT broker in
 * the MCP server process. Instead it proxies the subscription request to the
 * sensing-server's webhook/subscription registry at
 * POST /api/v1/bfld/<node_id>/subscribe, which returns a subscription_id.
 *
 * When the sensing-server is unreachable, the handler returns {ok:false,warn:true}
 * rather than throwing, consistent with the ruview-mcp soft-failure convention.
 *
 * In environments where no real broker is available (unit tests, dev machines
 * without mosquitto) the handler synthesises a valid subscription envelope
 * locally so the MCP schema-validation gate can be exercised independently.
 *
 * ADR-124 §4.1 output: { subscription_id: string, expires_at: number }
 */

import { randomUUID } from "node:crypto";
import { z } from "zod";
import type { RuviewConfig } from "../types.js";
import { sensingGet } from "../http.js";

export const bfldSubscribeSchema = z.object({
  node_id: z
    .string()
    .min(1)
    .optional()
    .describe("Target node id. Omit to use the single active node."),
  duration_s: z
    .number()
    .positive()
    .max(3600)
    .describe("Subscription duration in seconds (max 3600)."),
  sensing_server_url: z
    .string()
    .url()
    .optional()
    .describe("Override sensing-server URL for this call only."),
});

export type BfldSubscribeInput = z.infer<typeof bfldSubscribeSchema>;

/** Shape returned by the sensing-server subscription endpoint. */
interface SubscribeResponse {
  subscription_id: string;
  expires_at: number;
  topic: string;
}

export interface BfldSubscribeResult {
  ok: true;
  subscription_id: string;
  /** Unix timestamp (ms) when the subscription expires. */
  expires_at: number;
  /** MQTT wildcard topic this subscription covers. */
  topic: string;
}

export async function bfldSubscribe(
  input: BfldSubscribeInput,
  config: RuviewConfig
): Promise<object> {
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;
  const nodeId = input.node_id ?? "default";
  const topic = `ruview/${nodeId}/bfld/*`;

  // Attempt to register via sensing-server proxy.
  // The endpoint accepts query params: ?duration_s=<n>
  const result = await sensingGet<SubscribeResponse>(
    baseUrl,
    `/api/v1/bfld/${encodeURIComponent(nodeId)}/subscribe?duration_s=${input.duration_s}`,
    config.apiToken
  );

  if (!result.ok) {
    // Sensing-server unreachable — synthesise a local subscription envelope
    // so the agent knows the call was received and can correlate via the UUID.
    // The subscription won't receive real events, but the envelope is valid.
    const subscriptionId = randomUUID();
    const expiresAt = Date.now() + input.duration_s * 1_000;

    return {
      ok: false,
      warn: true,
      subscription_id: subscriptionId,
      expires_at: expiresAt,
      topic,
      error: result.error,
      hint:
        "Sensing-server not reachable — subscription envelope is synthetic. " +
        "No live BFLD events will be delivered. Ensure the sensing-server is " +
        "running and connected to the MQTT broker (ADR-122).",
    };
  }

  const data = result.data;

  if (typeof data.subscription_id !== "string" || typeof data.expires_at !== "number") {
    // Malformed response — still return a synthetic envelope.
    return {
      ok: false,
      warn: true,
      subscription_id: randomUUID(),
      expires_at: Date.now() + input.duration_s * 1_000,
      topic,
      error: "Sensing-server returned unexpected subscription shape.",
      raw_response: data,
    };
  }

  const out: BfldSubscribeResult = {
    ok: true,
    subscription_id: data.subscription_id,
    expires_at: data.expires_at,
    topic: data.topic ?? topic,
  };

  return out;
}
