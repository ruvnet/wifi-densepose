/**
 * Subprocess wrapper for Cognitum Cog binaries.
 *
 * The cog binaries implement the ADR-100 runtime contract:
 *   cog-<id> version
 *   cog-<id> manifest
 *   cog-<id> health
 *   cog-<id> run --config <path>
 *
 * This module shells out to those binaries.  If the binary is absent or returns
 * a non-zero exit code, the call fails-open with a WARN-level structured error
 * (same pattern cog-pose-estimation uses for missing model weights).
 */

import { spawn } from "node:child_process";
import type { Result } from "./http.js";
import { ok, err } from "./http.js";

const COG_TIMEOUT_MS = 15_000;

/**
 * Run a cog binary with the given subcommand arguments.
 * Returns stdout as a string on success, or an error message.
 */
export async function runCog(
  binary: string,
  args: string[]
): Promise<Result<string>> {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";

    const child = spawn(binary, args, {
      timeout: COG_TIMEOUT_MS,
      stdio: ["ignore", "pipe", "pipe"],
    });

    child.stdout?.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on("error", (e) => {
      resolve(
        err(
          `Failed to launch cog binary "${binary}" (${args.join(" ")}): ${e.message}. ` +
            `Set RUVIEW_POSE_COG_BINARY / RUVIEW_COUNT_COG_BINARY to the installed path, ` +
            `or install the cog on the Cognitum appliance first.`
        )
      );
    });

    child.on("close", (code) => {
      if (code !== 0) {
        resolve(
          err(
            `Cog "${binary} ${args.join(" ")}" exited with code ${code}. ` +
              `stderr: ${stderr.trim() || "(empty)"}`
          )
        );
      } else {
        resolve(ok(stdout));
      }
    });
  });
}

/**
 * Call `cog-<id> health` and return the exit code + output.
 */
export async function cogHealth(binary: string): Promise<Result<string>> {
  return runCog(binary, ["health"]);
}

/**
 * Call `cog-<id> version` and return the version string.
 */
export async function cogVersion(binary: string): Promise<Result<string>> {
  return runCog(binary, ["version"]);
}

/**
 * Run a cog inference with a synthetic CSI window piped via a temp config.
 *
 * The ADR-100 contract doesn't define a single-shot "infer" subcommand — the
 * cog's `run` subcommand is long-running.  Instead, we:
 * 1. Verify health returns 0.
 * 2. Emit a WARN explaining that single-shot inference requires a live
 *    sensing-server connection, then return a stub result.
 *
 * Full single-shot inference (M2 milestone) will use the sensing-server's
 * `/api/v1/sensing/latest` to build a real CSI window and feed it through the
 * cog via a short-lived `run` session.
 */
export async function cogInferStub(
  binary: string,
  taskLabel: string
): Promise<Result<{ backend: string; latency_ms: number; stub: true }>> {
  const health = await cogHealth(binary);
  if (!health.ok) {
    return err(
      `[WARN] ${taskLabel} cog health check failed — ${health.error}. ` +
        `Returning stub result. Install the cog or set the correct binary path.`
    );
  }
  return ok({
    backend: "stub",
    latency_ms: 0,
    stub: true,
  });
}
