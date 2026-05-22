/**
 * MCP tool: ruview_count_infer
 *
 * Run a single-shot person-count inference against a CSI window.
 *
 * Uses the cog-person-count binary (ADR-103).  The output includes a
 * calibrated confidence score and a 95% prediction interval, matching the
 * Stoer-Wagner + confidence-weighted log-sum fusion design in ADR-103.
 *
 * M1 (this file): stubs the inference after verifying the cog binary is healthy.
 * M2 wires the real forward pass.
 */

import { z } from "zod";
import type { RuviewConfig, CountInferResult } from "../types.js";
import { runCog } from "../cog.js";

export const countInferSchema = z.object({
  /**
   * Path to a CSI window JSON file.
   * Optional — when absent, uses the latest window from the sensing-server.
   */
  window_path: z
    .string()
    .optional()
    .describe("Path to a CSI window JSON file. Omit to use the live sensing-server."),
  /** Override the cog binary path for this call. */
  cog_binary: z
    .string()
    .optional()
    .describe("Path to cog-person-count binary. Default: RUVIEW_COUNT_COG_BINARY env var."),
  /**
   * Maximum number of persons to consider in the output distribution.
   * Capped at 7 per the count head's softmax over {0..7}.
   */
  max_persons: z
    .number()
    .int()
    .min(1)
    .max(7)
    .optional()
    .default(7)
    .describe("Upper bound on person count (1–7). Default: 7."),
});

export type CountInferInput = z.infer<typeof countInferSchema>;

// Health output from `cog-person-count health` (ADR-103 publisher.rs).
interface CountHealthEvent {
  ts: number;
  level: string;
  event: string;
  fields: {
    cog: string;
    backend: string;
    synthetic_count: number;
    synthetic_confidence: number;
    synthetic_p95_range: [number, number];
  };
}

function parseCountHealthOutput(stdout: string): CountHealthEvent | undefined {
  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (
        parsed !== null &&
        typeof parsed === "object" &&
        "event" in parsed &&
        (parsed as Record<string, unknown>)["event"] === "health.ok"
      ) {
        return parsed as CountHealthEvent;
      }
    } catch {
      // skip non-JSON lines from tracing subscriber
    }
  }
  return undefined;
}

export async function countInfer(
  input: CountInferInput,
  config: RuviewConfig
): Promise<object> {
  const binary = input.cog_binary ?? config.countCogBinary;
  const t0 = Date.now();

  // M2: run `cog-person-count health` which does real inference on a synthetic
  // window and emits a structured health.ok event with count + confidence + p95_range.
  const healthResult = await runCog(binary, ["health"]);
  const latencyMs = Date.now() - t0;

  if (!healthResult.ok) {
    return {
      ok: false,
      warn: true,
      error: healthResult.error,
      hint:
        "Set RUVIEW_COUNT_COG_BINARY to the path of the cog-person-count binary. " +
        "Install it from gs://cognitum-apps/cogs/<arch>/cog-person-count-<arch>. " +
        "See ADR-103 for installation instructions.",
    };
  }

  const healthEvent = parseCountHealthOutput(healthResult.data);
  const ts = Date.now() / 1000;

  if (!healthEvent) {
    const result: CountInferResult = {
      ts,
      count: 0,
      confidence: 0,
      count_p95_low: 0,
      count_p95_high: 0,
      backend: "unknown",
      latency_ms: latencyMs,
    };
    return {
      ok: true,
      synthetic_window: true,
      note:
        "Cog health passed (exit 0) but no health.ok event was parseable. " +
        "Returning empty count result.",
      result,
    };
  }

  const p95 = healthEvent.fields.synthetic_p95_range;
  const result: CountInferResult = {
    ts,
    count: healthEvent.fields.synthetic_count,
    confidence: healthEvent.fields.synthetic_confidence,
    count_p95_low: p95[0],
    count_p95_high: p95[1],
    backend: healthEvent.fields.backend,
    latency_ms: latencyMs,
  };

  return {
    ok: true,
    synthetic_window: true,
    note:
      "M2: inference ran on a synthetic CSI window via `cog-person-count health`. " +
      "For real CSI window inference, provide window_path (M3) or ensure the sensing-server is running.",
    result,
  };
}
