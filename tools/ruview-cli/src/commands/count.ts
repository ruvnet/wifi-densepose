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
        const t0 = Date.now();
        const health = await runCog(binary, ["health"]);
        const latencyMs = Date.now() - t0;

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
              result: { count: 0, confidence: 0, count_p95_low: 0, count_p95_high: 0, backend: "unavailable", latency_ms: 0 },
            }) + "\n"
          );
          process.exit(0);
        }

        let backend = "unknown";
        let count = 0;
        let confidence = 0;
        let p95Low = 0;
        let p95High = 0;

        for (const line of health.data.split("\n")) {
          try {
            const ev = JSON.parse(line.trim()) as Record<string, unknown>;
            if (ev["event"] === "health.ok") {
              const fields = ev["fields"] as Record<string, unknown>;
              backend = String(fields["backend"] ?? "unknown");
              count = Number(fields["synthetic_count"] ?? 0);
              confidence = Number(fields["synthetic_confidence"] ?? 0);
              const p95 = fields["synthetic_p95_range"] as number[];
              p95Low = p95?.[0] ?? 0;
              p95High = p95?.[1] ?? 0;
              break;
            }
          } catch { /* skip */ }
        }

        process.stdout.write(
          JSON.stringify({
            ok: true,
            synthetic_window: true,
            note: "M2: real inference on synthetic CSI window via cog health check.",
            result: {
              ts: Date.now() / 1000,
              count,
              confidence,
              count_p95_low: p95Low,
              count_p95_high: p95High,
              backend,
              latency_ms: latencyMs,
            },
          }) + "\n"
        );
      }
    }
  );
}
