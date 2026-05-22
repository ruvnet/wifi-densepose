/**
 * MCP tool: ruview_registry_list
 *
 * List installed/available cogs from the Cognitum edge module registry.
 *
 * Fetches `/api/v1/edge/registry` from the sensing-server, which proxies the
 * canonical GCS catalog with a 1-hour TTL cache (ADR-102).  The result is the
 * full 105-cog catalog as of the last upstream sync.
 *
 * Use the optional `category` filter to narrow results.  Available categories
 * (from the v2.1.0 registry): health, security, building, retail, industrial,
 * research, ai, swarm, signal, network, developer.
 */

import { z } from "zod";
import type { RuviewConfig, RegistryListResult, CogEntry } from "../types.js";
import { sensingGet } from "../http.js";

export const registryListSchema = z.object({
  /** Filter cogs by category. */
  category: z
    .string()
    .optional()
    .describe(
      "Filter by category (health, security, building, retail, industrial, " +
        "research, ai, swarm, signal, network, developer). Omit for all."
    ),
  /** Filter cogs whose id or name contains this substring (case-insensitive). */
  search: z
    .string()
    .optional()
    .describe("Search substring matched against cog id and name (case-insensitive)."),
  /** Force-bypass the sensing-server's 1-hour cache. */
  refresh: z
    .boolean()
    .optional()
    .default(false)
    .describe("Bypass the 1-hour registry cache. Use sparingly."),
  /** Override the sensing-server URL for this call only. */
  sensing_server_url: z
    .string()
    .url()
    .optional()
    .describe("Override the sensing-server URL."),
});

export type RegistryListInput = z.infer<typeof registryListSchema>;

// The upstream registry JSON shape (ADR-102).
interface UpstreamRegistryPayload {
  registry: {
    cogs?: CogEntry[];
    apps?: CogEntry[];
    [key: string]: unknown;
  };
  fetched_at: number;
  ttl_seconds: number;
  stale: boolean;
  upstream_url: string;
  upstream_sha256: string;
}

export async function registryList(
  input: RegistryListInput,
  config: RuviewConfig
): Promise<object> {
  const baseUrl = input.sensing_server_url ?? config.sensingServerUrl;
  const qs = input.refresh ? "?refresh=1" : "";

  const result = await sensingGet<UpstreamRegistryPayload>(
    baseUrl,
    `/api/v1/edge/registry${qs}`,
    config.apiToken
  );

  if (!result.ok) {
    return {
      ok: false,
      warn: true,
      error: result.error,
      hint:
        "Ensure the sensing-server is running and the edge registry endpoint is enabled. " +
        "See ADR-102 for configuration (--no-edge-registry disables it).",
    };
  }

  const payload = result.data;
  // Registry entries may be under `cogs` or `apps` depending on the catalog version.
  let cogs: CogEntry[] = (payload.registry.cogs ?? payload.registry.apps ?? []) as CogEntry[];

  // Apply filters.
  if (input.category) {
    const cat = input.category.toLowerCase();
    cogs = cogs.filter((c) => c.category?.toLowerCase() === cat);
  }
  if (input.search) {
    const q = input.search.toLowerCase();
    cogs = cogs.filter(
      (c) =>
        c.id?.toLowerCase().includes(q) || c.name?.toLowerCase().includes(q)
    );
  }

  const out: RegistryListResult = {
    fetched_at: payload.fetched_at,
    ttl_seconds: payload.ttl_seconds,
    stale: payload.stale,
    upstream_url: payload.upstream_url,
    upstream_sha256: payload.upstream_sha256,
    cogs,
  };

  return {
    ok: true,
    total_cogs: cogs.length,
    ...out,
  };
}
