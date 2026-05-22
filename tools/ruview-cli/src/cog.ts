/**
 * Subprocess wrapper for Cognitum Cog binaries (CLI variant).
 * Mirrors tools/ruview-mcp/src/cog.ts.
 */

import { spawn } from "node:child_process";

export type Result<T> = { ok: true; data: T } | { ok: false; error: string };

const COG_TIMEOUT_MS = 15_000;

export async function runCog(binary: string, args: string[]): Promise<Result<string>> {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";

    const child = spawn(binary, args, {
      timeout: COG_TIMEOUT_MS,
      stdio: ["ignore", "pipe", "pipe"],
    });

    child.stdout?.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    child.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });

    child.on("error", (e) => {
      resolve(err(
        `Failed to launch "${binary}" (${args.join(" ")}): ${e.message}. ` +
        `Set RUVIEW_POSE_COG_BINARY / RUVIEW_COUNT_COG_BINARY or install the cog.`
      ));
    });

    child.on("close", (code) => {
      if (code !== 0) {
        resolve(err(`Cog "${binary} ${args.join(" ")}" exited with code ${code}. stderr: ${stderr.trim() || "(empty)"}`));
      } else {
        resolve({ ok: true, data: stdout });
      }
    });
  });
}

function err(error: string): { ok: false; error: string } {
  return { ok: false, error };
}
