/**
 * MCP tool: ruview.bfld.last_scan
 *
 * Returns the most recent BFLD scan result for a node, sourced from the
 * sensing-server's REST proxy of the BFLD MQTT state topics defined in
 * ADR-122 §2.2. The sensing-server aggregates the per-entity state topics
 * (presence, person_count, confidence, identity_risk) into a single JSON
 * object at GET /api/v1/bfld/<node_id>/last_scan.
 *
 * Wire format (ADR-118 BfldEvent, class-permissive fields only):
 *   node_id              string   — originating node
 *   identity_risk_score  number   — [0,1], None at privacy_class Restricted
 *   privacy_class        number   — 0=raw,1=derived,2=anonymous,3=restricted
 *   n_frames             number   — person_count proxy (frames accumulated)
 *   timestamp_ms         number   — capture timestamp in ms since epoch
 *
 * Returns {ok:false, warn:true} when the sensing-server is not reachable
 * so the caller can treat unavailability as a soft warning rather than
 * a hard error (mirrors the pattern in csi-latest.ts).
 */

import { z } from "zod";
import type { RuviewConfig } from "../types.js";
import { sensingGet } from "../http.js";

export const bfldLastScanSchema = z.object({
  node_id: z
    .string()
    .min(1)
    .optional()
    .describe("Target node id. Omit to use the single active node."),
  sensing_server_url: z
    .string()
    .url()
    .optional()
    .describe("Override sensing-server URL for this call only."),
});

export type BfldLastScanInput = z.infer<typeof bfldLastScanSchema>;

/** Shape returned by the sensing-server BFLD last-scan proxy endpoint. */
interface BfldScanResponse {
  node_id: string;
  identity_risk_score: number | null;
  privacy_class: number;
  person_count: number;
  confidence: number;
  presence: boolean;
  timestamp_ns: number;
}

/** ADR-124 §4.1 output contract for ruview.bfld.last_scan. */
export interface BfldLastScanResult {
  ok: true;
  node_id: string;
  identity_risk_score: number | null;
  privacy_class: number;
  /** person_count used as n_frames proxy (ADR-118 BfldEvent.person_count). */
  n_frames: number;
  /** Converted from BfldEvent.timestamp_ns (nanoseconds → milliseconds). */
  timestamp_ms: number;
}

export async function bfldLastScan(
  input: BfldLastScanInput,
  config: RuviewConfig
): Promise<object> {
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;
  const nodeId = input.node_id ?? "default";

  const result = await sensingGet<BfldScanResponse>(
    baseUrl,
    `/api/v1/bfld/${encodeURIComponent(nodeId)}/last_scan`,
    config.apiToken
  );

  if (!result.ok) {
    return {
      ok: false,
      warn: true,
      error: result.error,
      hint:
        "Ensure the sensing-server is running and the BFLD pipeline is active " +
        "(ADR-118). The node must have published at least one BfldEvent since " +
        "the last server restart.",
    };
  }

  const data = result.data;

  // Validate the minimum required fields are present.
  if (typeof data.node_id !== "string" || typeof data.timestamp_ns !== "number") {
    return {
      ok: false,
      warn: true,
      error: "Sensing-server returned an unexpected BFLD response shape.",
      raw_response: data,
    };
  }

  const out: BfldLastScanResult = {
    ok: true,
    node_id: data.node_id,
    identity_risk_score: data.identity_risk_score ?? null,
    privacy_class: data.privacy_class,
    n_frames: data.person_count,
    timestamp_ms: Math.round(data.timestamp_ns / 1_000_000),
  };

  return out;
}
