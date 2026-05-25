/**
 * MCP tool: ruview.vitals.get_all (ADR-124 §4.1)
 * Output: EdgeVitalsResult — full EdgeVitalsMessage minus `raw`.
 */
import { z } from "zod";
import type { RuviewConfig } from "../types.js";
import { fetchVitals, resolveNodeId } from "./vitals-fetch.js";

export const vitalsGetAllSchema = z.object({
  node_id: z.string().min(1).optional().describe("Target node id."),
  sensing_server_url: z.string().url().optional(),
});
export type VitalsGetAllInput = z.infer<typeof vitalsGetAllSchema>;

export async function vitalsGetAll(
  input: VitalsGetAllInput,
  config: RuviewConfig
): Promise<object> {
  const nodeId = resolveNodeId(input.node_id);
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;
  const r = await fetchVitals(nodeId, baseUrl, config.apiToken);
  if (!r.ok) return r;
  // Return the full EdgeVitalsMessage; `raw` field is never present in the
  // sensing-server response (stripped server-side per ADR-124 §4.1 spec).
  return { ok: true, ...r.data };
}
