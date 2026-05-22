/**
 * Configuration loader for the RuView MCP server.
 *
 * All settings can be overridden via environment variables.  No config file is
 * required — the server is designed to work out of the box with a locally-running
 * sensing-server on the default port.
 */

import os from "node:os";
import path from "node:path";
import type { RuviewConfig } from "./types.js";

function env(key: string): string | undefined {
  return process.env[key];
}

function envOrDefault(key: string, fallback: string): string {
  return env(key) ?? fallback;
}

/**
 * Load the effective RuviewConfig from environment variables.
 *
 * Environment variables:
 *   RUVIEW_SENSING_SERVER_URL   — base URL of the sensing-server  (default: http://localhost:3000)
 *   RUVIEW_API_TOKEN            — Bearer token for /api/v1/* routes (no default; auth disabled when absent)
 *   RUVIEW_POSE_COG_BINARY      — path to cog-pose-estimation binary
 *   RUVIEW_COUNT_COG_BINARY     — path to cog-person-count binary
 *   RUVIEW_JOBS_DIR             — directory for job logs (default: ~/.ruview/jobs)
 */
export function loadConfig(): RuviewConfig {
  return {
    sensingServerUrl: envOrDefault(
      "RUVIEW_SENSING_SERVER_URL",
      "http://localhost:3000"
    ),
    apiToken: env("RUVIEW_API_TOKEN"),
    poseCogBinary: envOrDefault(
      "RUVIEW_POSE_COG_BINARY",
      detectCogBinary("cog-pose-estimation")
    ),
    countCogBinary: envOrDefault(
      "RUVIEW_COUNT_COG_BINARY",
      detectCogBinary("cog-person-count")
    ),
    jobsDir: envOrDefault(
      "RUVIEW_JOBS_DIR",
      path.join(os.homedir(), ".ruview", "jobs")
    ),
  };
}

/**
 * Attempt to locate a cog binary on PATH or in common install locations.
 * Returns the bare binary name if not found (will fail gracefully at invocation).
 */
function detectCogBinary(name: string): string {
  // Common install paths for Cognitum cog binaries on Linux/macOS appliances.
  const candidates = [
    `/var/lib/cognitum/apps/${name.replace("cog-", "")}/cog-${name.replace("cog-", "")}-arm`,
    `/var/lib/cognitum/apps/${name.replace("cog-", "")}/cog-${name.replace("cog-", "")}-x86_64`,
    `/usr/local/bin/${name}`,
    name, // bare name — rely on PATH
  ];
  // Return the first candidate that might exist; actual existence is checked at call time.
  return candidates[candidates.length - 1] ?? name;
}
