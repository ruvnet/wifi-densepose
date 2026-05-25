/**
 * MCP tool: ruview.vitals.get_heart_rate (ADR-124 §4.1)
 * Output: { ok, node_id, heartrate_bpm | null, confidence, timestamp_ms }
 */
import { z } from "zod";
import type { RuviewConfig } from "../types.js";
import { fetchVitals, resolveNodeId } from "./vitals-fetch.js";

export const vitalsGetHeartRateSchema = z.object({
  node_id: z.string().min(1).optional().describe("Target node id."),
  window_s: z.number().positive().max(300).optional().describe("Averaging window (s, max 300)."),
  sensing_server_url: z.string().url().optional(),
});
export type VitalsGetHeartRateInput = z.infer<typeof vitalsGetHeartRateSchema>;

export async function vitalsGetHeartRate(
  input: VitalsGetHeartRateInput,
  config: RuviewConfig
): Promise<object> {
  const nodeId = resolveNodeId(input.node_id);
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;
  const r = await fetchVitals(nodeId, baseUrl, config.apiToken);
  if (!r.ok) return r;
  return {
    ok: true,
    node_id: r.data.node_id,
    heartrate_bpm: r.data.heartrate_bpm,
    confidence: r.data.confidence,
    timestamp_ms: r.data.timestamp_ms,
  };
}
