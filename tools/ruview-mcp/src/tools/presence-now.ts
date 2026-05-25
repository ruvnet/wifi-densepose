/**
 * MCP tool: ruview.presence.now (ADR-124 §4.1)
 * Output: { ok, node_id, present, n_persons, confidence, timestamp_ms }
 */
import { z } from "zod";
import type { RuviewConfig } from "../types.js";
import { fetchVitals, resolveNodeId } from "./vitals-fetch.js";

export const presenceNowSchema = z.object({
  node_id: z.string().min(1).optional().describe("Target node id."),
  sensing_server_url: z.string().url().optional(),
});
export type PresenceNowInput = z.infer<typeof presenceNowSchema>;

export async function presenceNow(input: PresenceNowInput, config: RuviewConfig): Promise<object> {
  const nodeId = resolveNodeId(input.node_id);
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;
  const r = await fetchVitals(nodeId, baseUrl, config.apiToken);
  if (!r.ok) return r;
  return {
    ok: true,
    node_id: r.data.node_id,
    present: r.data.presence,
    n_persons: r.data.n_persons,
    confidence: r.data.confidence,
    timestamp_ms: r.data.timestamp_ms,
  };
}
