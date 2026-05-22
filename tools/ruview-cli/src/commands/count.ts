/**
 * ruview count — Person count commands.
 *
 * count infer  — run single-shot person-count inference.
 */

import type { Argv } from "yargs";
import { runCog } from "../cog.js";
import { loadConfig } from "../config.js";

export function countCommand(cli: Argv): void {
  cli.command(
    "count <action>",
    "Person count commands",
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
          description: "Path to cog-person-count binary (default: RUVIEW_COUNT_COG_BINARY)",
        })
        .option("max-persons", {
          type: "number",
          default: 7,
          description: "Upper bound on person count (1–7, default: 7)",
        }),
    async (args) => {
      const config = loadConfig();
      const binary = (args["binary"] as string | undefined) ?? config.countCogBinary;

      if (args.action === "infer") {
        const health = await runCog(binary, ["health"]);
        if (!health.ok) {
          process.stderr.write(
            `[WARN] Cog health check failed: ${health.error}\n` +
              `Set RUVIEW_COUNT_COG_BINARY or install cog-person-count (ADR-103).\n`
          );
          process.stdout.write(
            JSON.stringify({
              ok: false,
              warn: true,
              error: health.error,
              stub: true,
              result: {
                count: 0,
                confidence: 0,
                count_p95_low: 0,
                count_p95_high: 0,
                backend: "stub",
                latency_ms: 0,
              },
            }) + "\n"
          );
          process.exit(0);
        }

        process.stdout.write(
          JSON.stringify({
            ok: true,
            stub: true,
            note: "M1 stub — real inference wired in M2. Cog health passed.",
            result: {
              ts: Date.now() / 1000,
              count: 0,
              confidence: 0,
              count_p95_low: 0,
              count_p95_high: 0,
              backend: "stub",
              latency_ms: 0,
            },
          }) + "\n"
        );
      }
    }
  );
}
