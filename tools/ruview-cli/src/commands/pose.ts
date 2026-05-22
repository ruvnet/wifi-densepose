/**
 * ruview pose — Pose estimation commands.
 *
 * pose infer  — run single-shot 17-keypoint inference.
 */

import type { Argv } from "yargs";
import { runCog } from "../cog.js";
import { loadConfig } from "../config.js";

export function poseCommand(cli: Argv): void {
  cli.command(
    "pose <action>",
    "Pose estimation commands",
    (y) =>
      y
        .positional("action", {
          choices: ["infer"] as const,
          description: "Action to perform",
        })
        .option("window", {
          type: "string",
          description: "Path to a CSI window JSON file (omit to use live sensing-server)",
        })
        .option("binary", {
          type: "string",
          description: "Path to cog-pose-estimation binary (default: RUVIEW_POSE_COG_BINARY)",
        }),
    async (args) => {
      const config = loadConfig();
      const binary = (args["binary"] as string | undefined) ?? config.poseCogBinary;

      if (args.action === "infer") {
        // M1: verify health, emit stub.
        const health = await runCog(binary, ["health"]);
        if (!health.ok) {
          process.stderr.write(
            `[WARN] Cog health check failed: ${health.error}\n` +
              `Set RUVIEW_POSE_COG_BINARY or install cog-pose-estimation (ADR-101).\n`
          );
          process.stdout.write(
            JSON.stringify({
              ok: false,
              warn: true,
              error: health.error,
              stub: true,
              result: { n_persons: 0, persons: [], backend: "stub", latency_ms: 0 },
            }) + "\n"
          );
          process.exit(0); // Fail-open; non-zero would break pipelines.
        }

        process.stdout.write(
          JSON.stringify({
            ok: true,
            stub: true,
            note: "M1 stub — real inference wired in M2. Cog health passed.",
            result: {
              ts: Date.now() / 1000,
              n_persons: 0,
              persons: [],
              backend: "stub",
              latency_ms: 0,
            },
          }) + "\n"
        );
      }
    }
  );
}
