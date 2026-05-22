/**
 * Configuration loader for the RuView CLI.
 * Mirrors tools/ruview-mcp/src/config.ts — sourced from environment variables.
 */

import os from "node:os";
import path from "node:path";

export interface RuviewCliConfig {
  sensingServerUrl: string;
  apiToken: string | undefined;
  poseCogBinary: string;
  countCogBinary: string;
  jobsDir: string;
}

function envOrDefault(key: string, fallback: string): string {
  return process.env[key] ?? fallback;
}

export function loadConfig(): RuviewCliConfig {
  return {
    sensingServerUrl: envOrDefault(
      "RUVIEW_SENSING_SERVER_URL",
      "http://localhost:3000"
    ),
    apiToken: process.env["RUVIEW_API_TOKEN"],
    poseCogBinary: envOrDefault("RUVIEW_POSE_COG_BINARY", "cog-pose-estimation"),
    countCogBinary: envOrDefault("RUVIEW_COUNT_COG_BINARY", "cog-person-count"),
    jobsDir: envOrDefault(
      "RUVIEW_JOBS_DIR",
      path.join(os.homedir(), ".ruview", "jobs")
    ),
  };
}
